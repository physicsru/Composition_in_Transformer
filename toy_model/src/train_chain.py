"""
Training + autonomous evaluation for the strict w2/d2 chain task (docs/shallow_training_deep_composition_plan_2026-09-16.md).

  python train_chain.py --data_dir ../data/chain_L --save_dir ../runs/chain_B_s1 --protocol B --seed 1 [--resume]

Schedule (plan §4.2): phase 1 = `--phase1_updates` updates of 256 w2 rows; phase 2 = `--phase2_updates` updates of
128 w2 + 128 d2 rows. Loss per update = sum over rows of sum_t w_t CE_t / 256 with the protocol weights of
data/chain_format.py (final 1, bridge 1, END 1, copies share 1; w2 sub-questions halved). Streams are frozen per
(seed, stream name) so every protocol sees the same question ids at the same update.

Every `--eval_every` updates: teacher-forced CE per token role (train subsets, validation d2), FREE greedy rollouts
(no gold history) on the exhaustive atomic set, validation d2 (model selection), validation w2 (second question
conditioned on the model's own first answer) and a small test subset; metrics.jsonl + model checkpoint
ckpt_UUUUUU.pt; last.pt holds the full state (optimizer, streams, RNG) for exact resume; best_by_val.pt tracks
validation d2 final-answer accuracy (ties: lower validation loss). At the end (or with --eval_only) the full test
matrix is rolled out for the final and best checkpoints -> final_eval_<tag>.json, predictions_<tag>.jsonl.
"""
import argparse
import collections
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data.chain_format import (ROLE_NAMES, ChainVocab, completion_length, parse_completion, prompt_ids, render_chain,
                               render_d2, render_w2, score_chain)
from data.local_format import extend_vocab
from model import GPT2LikeEncoder


# ----------------------------------------------------------------------------------------------------------------- data
def load_json(d, name):
    return json.load(open(os.path.join(d, f"{name}.json")))


class Rendered:
    """Padded tensors for rendered rows: inputs = ids[:-1], targets = ids[1:], weights/roles aligned with targets."""

    def __init__(self, rows, pad, L, device):
        n = len(rows)
        inp = torch.full((n, L), pad, dtype=torch.long); tgt = torch.full((n, L), -100, dtype=torch.long)
        w = torch.zeros((n, L)); roles = torch.full((n, L), -1, dtype=torch.long); pm = torch.ones((n, L), dtype=torch.bool)
        for i, (ids, ww, rr) in enumerate(rows):
            k = len(ids) - 1
            inp[i, :k] = torch.tensor(ids[:-1]); tgt[i, :k] = torch.tensor(ids[1:])
            w[i, :k] = torch.tensor(ww[1:]); roles[i, :k] = torch.tensor(rr[1:]); pm[i, :k] = False
        self.input_ids, self.target_ids, self.weights, self.roles, self.pad_mask = (t.to(device) for t in (inp, tgt, w, roles, pm))
        self.n = n
        self.n_tokens = int((~pm).sum()); self.n_supervised = int((w > 0).sum())


class Stream:
    def __init__(self, n, seed, name):
        self.n = n; self.rng = np.random.default_rng([seed, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)])
        self.order = None; self.cursor = 0; self.cycles = 0

    def take(self, k):
        out = []
        while k > 0:
            if self.order is None or self.cursor >= self.n:
                self.order = self.rng.permutation(self.n); self.cursor = 0; self.cycles += 1
            chunk = self.order[self.cursor:self.cursor + k]; self.cursor += len(chunk); k -= len(chunk); out.append(chunk)
        return np.concatenate(out)

    def state(self):
        return dict(order=None if self.order is None else self.order.tolist(), cursor=self.cursor, cycles=self.cycles,
                    rng=self.rng.bit_generator.state)

    def load(self, st):
        self.order = None if st["order"] is None else np.array(st["order"]); self.cursor = st["cursor"]; self.cycles = st["cycles"]
        self.rng.bit_generator.state = st["rng"]


# --------------------------------------------------------------------------------------------------------------- rollout
ROLLOUT_BATCH = 1000


@torch.no_grad()
def rollout(model, prompts, max_new, device, batch=None, n_loops=1):
    """Greedy free generation for prompts of EQUAL length. Returns (N, max_new) generated ids.
    n_loops: recurrent-depth budget (the same for every generated token; docs/experiments_loop_next.md §6)."""
    batch = batch or ROLLOUT_BATCH
    outs = []
    for s in range(0, len(prompts), batch):
        x = torch.tensor(prompts[s:s + batch], dtype=torch.long, device=device)
        for _ in range(max_new):
            logits = model(x, n_loops=n_loops)[:, -1, :]
            x = torch.cat([x, logits.argmax(-1, keepdim=True)], 1)
        outs.append(x[:, -max_new:].cpu())
    return torch.cat(outs).numpy()


def eval_chains(model, v, protocol, rel, chains, device, keep_predictions=False, loops=None):
    """chains: list of dict(x, rels, states, ...). Groups by depth. Returns per-chain scores (and parsed outputs).
    loops: None (plain model) or a callable depth -> recurrent-depth budget T."""
    results = []
    by_d = collections.defaultdict(list)
    for c in chains:
        by_d[len(c["rels"])].append(c)
    for d, cs in by_d.items():
        prompts = [prompt_ids(v, c["x"], c["rels"]) for c in cs]
        gen = rollout(model, prompts, completion_length(protocol, d) + 2, device, n_loops=loops(d) if loops else 1)
        for c, g in zip(cs, gen):
            parsed = parse_completion(v, protocol, g.tolist(), d)
            sc = score_chain(v, protocol, rel, c["x"], c["rels"], c["states"], parsed)
            rec = dict(sc)
            if keep_predictions:
                rec.update(id=c.get("id"), cat=c.get("cat"), d=d, pred_states=parsed["states"], pred_rels=parsed["rels"],
                           ended=parsed["ended"], valid_format=parsed["valid_format"])
            results.append((c, rec))
    return results


def summarize(results):
    n = len(results)
    if n == 0:
        return {}
    fe = collections.Counter(r["first_error_type"] for _, r in results if r["first_error_type"])
    stop = collections.Counter(r["stop"] for _, r in results)
    own_c = sum(r["own_update_correct"] for _, r in results); own_t = sum(r["own_update_total"] for _, r in results)
    return dict(n=n, final_acc=sum(r["final_correct"] for _, r in results) / n, traj_acc=sum(r["traj_correct"] for _, r in results) / n,
                own_update_acc=(own_c / own_t) if own_t else None, stop={k: v / n for k, v in stop.items()},
                first_error={k: v / n for k, v in fe.items()})


@torch.no_grad()
def eval_w2_val(model, v, protocol, rel, val_w2, device, n_loops=1):
    """Second question answered with the model's OWN first output as history (plan §5)."""
    correct = 0; both = 0
    for q in val_w2:
        (x, r), (z, s) = q["q1"], q["q2"]
        p1 = prompt_ids(v, x, [r]); g1 = rollout(model, [p1], completion_length(protocol, 1) + 2, device, n_loops=n_loops)[0].tolist()
        end1 = g1.index(v.END) + 1 if v.END in g1 else len(g1)
        hist = p1 + g1[:end1] + prompt_ids(v, z, [s])
        g2 = rollout(model, [hist], completion_length(protocol, 1) + 2, device, n_loops=n_loops)[0].tolist()
        s1 = score_chain(v, protocol, rel, x, [r], [rel[r][x]], parse_completion(v, protocol, g1, 1))
        s2 = score_chain(v, protocol, rel, z, [s], [rel[s][z]], parse_completion(v, protocol, g2, 1))
        correct += int(s2["final_correct"]); both += int(s1["final_correct"] and s2["final_correct"])
    return dict(n=len(val_w2), second_acc=correct / len(val_w2), both_acc=both / len(val_w2))


@torch.no_grad()
def teacher_forced(model, R: Rendered, device, batch=1024, n_loops=1):
    """Mean CE per role and the weighted loss per row."""
    sums = collections.defaultdict(float); cnts = collections.defaultdict(int); wloss = 0.0
    for s in range(0, R.n, batch):
        sl = slice(s, s + batch)
        logits = model(R.input_ids[sl], pad_mask=R.pad_mask[sl], n_loops=n_loops)
        ce = F.cross_entropy(logits.transpose(1, 2), R.target_ids[sl], reduction="none", ignore_index=-100)
        wloss += float((ce * R.weights[sl]).sum())
        for k, name in enumerate(ROLE_NAMES):
            m = R.roles[sl] == k
            if k == 0:
                continue
            sums[name] += float(ce[m].sum()); cnts[name] += int(m.sum())
    out = {f"ce/{k}": sums[k] / cnts[k] for k in sums if cnts[k]}
    out["loss_per_row"] = wloss / R.n
    return out


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--save_dir", required=True)
    ap.add_argument("--protocol", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--no_rope", action="store_true", help="NoPE control: no positional encoding at all (plan §6.1)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--n_layer", type=int, default=2); ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--rope_base", type=float, default=100.0)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight_decay", type=float, default=0.1); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rows_per_update", type=int, default=256)
    ap.add_argument("--d3_per_update", type=int, default=0, help="CONTROL: rows from train_d3.json per phase-2 update (taken out of the d2 share)")
    ap.add_argument("--phase1_updates", type=int, default=50000); ap.add_argument("--phase2_updates", type=int, default=200000)
    ap.add_argument("--eval_every", type=int, default=5000); ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--small_test_per_cell", type=int, default=500); ap.add_argument("--small_test_depths", default="2,3,4,8")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval_only", default=None, help="checkpoint to evaluate on the full test matrix (no training)")
    ap.add_argument("--no_final_eval", action="store_true")
    ap.add_argument("--rollout_batch", type=int, default=1000, help="chains per rollout batch (protocol D at d=32 needs ~250 on a shared GPU)")
    ap.add_argument("--final_max_per_cell", type=int, default=0, help="limit chains per (cat, d) cell in the final eval (0 = all; smoke tests)")
    ap.add_argument("--final_depths", default="", help="comma-separated depths for the final eval (default: all in the dataset)")
    ap.add_argument("--final_extra_ckpts", default="", help="'auto' = also evaluate the two checkpoints before the last; or a comma-separated list")
    ap.add_argument("--eval_tag", default=None, help="tag for --eval_only outputs (default 'evalonly')")
    # recurrent-depth arm R (docs/experiments_loop_next.md §6): the same block stack is applied T times
    ap.add_argument("--loop_T", default="", help="training loop budgets sampled per update, e.g. 2,4,8 (empty = plain model)")
    ap.add_argument("--loop_cap", type=int, default=128, help="test rule T(d) = smallest power of two >= max(8, d), capped here")
    ap.add_argument("--loop_fixed", type=int, default=8, help="fixed-budget control reported next to T(d) in the final eval")
    ap.add_argument("--loop_matrix_at", default="50000,150000,250000", help="updates at which the full depth x T matrix is computed on the monitoring set")
    ap.add_argument("--loop_matrix_T", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--zero_init_out", action="store_true", help="zero-init the attention / MLP output projections (looped model)")
    ap.add_argument("--vocab_ext", action="store_true", help="use the frozen extended vocabulary shared with the local-executor arms")
    ap.add_argument("--select_by", choices=["d2", "w2"], default="d2", help="checkpoint selection: validation d2 (original) or w2-only free execution (new arms)")
    args = ap.parse_args()
    global ROLLOUT_BATCH
    ROLLOUT_BATCH = args.rollout_batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = load_json(args.data_dir, "meta"); rel = meta["rel_map"]
    base_vocab = load_json(args.data_dir, "vocab")
    v = ChainVocab(extend_vocab(base_vocab) if args.vocab_ext else base_vocab)
    loop_T = [int(x) for x in args.loop_T.split(",")] if args.loop_T else []
    looped = bool(loop_T)

    def T_of_d(d):
        """pre-registered budget rule (plan §6): smallest power of two >= max(8, d), capped at --loop_cap; 1 for the plain model."""
        if not looped:
            return 1
        t = 8
        while t < d:
            t *= 2
        return min(args.loop_cap, t)
    train_w2 = load_json(args.data_dir, "train_w2"); train_d2 = load_json(args.data_dir, "train_d2")
    val_d2 = load_json(args.data_dir, "val_d2"); val_w2 = load_json(args.data_dir, "val_w2"); tests = load_json(args.data_dir, "tests")
    P = args.protocol
    rw2 = [render_w2(v, P, rel, q) for q in train_w2]; rd2 = [render_d2(v, P, rel, q) for q in train_d2]
    train_d3 = load_json(args.data_dir, "train_d3") if (args.d3_per_update and os.path.exists(os.path.join(args.data_dir, "train_d3.json"))) else []
    if args.d3_per_update and not train_d3:
        raise SystemExit("--d3_per_update > 0 but the dataset has no train_d3.json (generate with --train_d3 1)")
    rd3 = [render_chain(v, P, rel, q) for q in train_d3]
    L = max([max(len(r[0]) for r in rw2), max(len(r[0]) for r in rd2)] + ([max(len(r[0]) for r in rd3)] if rd3 else [])) - 1
    W2 = Rendered(rw2, v.pad, L, device); D2 = Rendered(rd2, v.pad, L, device)
    D3 = Rendered(rd3, v.pad, L, device) if rd3 else None
    VAL = Rendered([render_d2(v, P, rel, q) for q in val_d2], v.pad, L, device)
    VALW2 = Rendered([render_w2(v, P, rel, q) for q in val_w2], v.pad, L, device)
    SUB_W2 = Rendered(rw2[:1000], v.pad, L, device); SUB_D2 = Rendered(rd2[:1000], v.pad, L, device)
    atomic = [dict(x=x, rels=[r], states=[rel[r][x]]) for r in range(v.R) for x in range(v.E)]
    small_depths = {int(d) for d in args.small_test_depths.split(",")}
    small_tests = []
    per_cell = collections.Counter()
    for t in tests:
        key = (t["cat"], t["d"])
        if t["d"] in small_depths and per_cell[key] < args.small_test_per_cell:
            small_tests.append(t); per_cell[key] += 1
    val_chains = [dict(x=q["x"], rels=[q["r"], q["s"]], states=[rel[q["r"]][q["x"]], rel[q["s"]][rel[q["r"]][q["x"]]]]) for q in val_d2]

    model = GPT2LikeEncoder(len(v.vocab), d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, dropout=0.0,
                            max_len=args.max_len, rope_base=args.rope_base, use_rope=not args.no_rope, zero_init_out=args.zero_init_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    T1 = T_of_d(1)

    def cells_of(res):
        cells = collections.defaultdict(list)
        for c, r in res:
            cells[f"{c['cat']}|d{c['d']}"].append((c, r))
        return {k: summarize(vv) for k, vv in sorted(cells.items())}

    def full_eval(tag, ckpt_path=None):
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
        model.eval()
        pool = tests
        final_depths = {int(x) for x in args.final_depths.split(",")} if args.final_depths else None
        if args.final_max_per_cell or final_depths:
            cnt = collections.Counter(); pool = []
            for t in tests:
                if final_depths and t["d"] not in final_depths:
                    continue
                if not args.final_max_per_cell or cnt[(t["cat"], t["d"])] < args.final_max_per_cell:
                    pool.append(t); cnt[(t["cat"], t["d"])] += 1
        t_eval = time.time()
        res = eval_chains(model, v, P, rel, pool, device, keep_predictions=True, loops=T_of_d)
        cells = collections.defaultdict(list)
        for c, r in res:
            cells[(c["cat"], c["d"])].append((c, r))
            for flag in ("repeated_relation", "revisits_entity"):
                if c.get(flag):
                    cells[(f"{c['cat']}|{flag}", c["d"])].append((c, r))
        summary = {f"{k[0]}|d{k[1]}": summarize(vv) for k, vv in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))}
        summary["atomic"] = summarize(eval_chains(model, v, P, rel, atomic, device, loops=T_of_d))
        if looped:
            summary["loop_rule"] = dict(rule="min(cap, smallest power of two >= max(8, d))", cap=args.loop_cap, T_by_depth={d: T_of_d(d) for d in sorted({t["d"] for t in pool})})
            summary[f"fixed_T{args.loop_fixed}"] = cells_of(eval_chains(model, v, P, rel, pool, device, loops=lambda d: args.loop_fixed))
        # plan §6.1: EXTERNAL step-wise calling -- the program feeds the model one-step prompts Q s_t r_(t+1) ANS and
        # chains the predicted entities itself (identifies whether the atomic update is reliable; not autonomous)
        stepwise = collections.defaultdict(lambda: [0, 0])
        for d in sorted({t["d"] for t in pool}):
            cs = [t for t in pool if t["d"] == d]
            cur = [c["x"] for c in cs]; ok = [True] * len(cs)
            for step in range(d):
                prompts = [prompt_ids(v, s_, [c["rels"][step]]) for s_, c in zip(cur, cs)]
                gen = rollout(model, prompts, completion_length(P, 1) + 1, device, n_loops=T1)
                nxt = []
                for i, g in enumerate(gen):
                    pr = parse_completion(v, P, g.tolist(), 1)
                    st_ = pr["states"][-1] if pr["states"] else -1
                    ok[i] = ok[i] and (st_ == cs[i]["states"][step]); nxt.append(st_ if st_ >= 0 else 0)
                cur = nxt
            for c, o in zip(cs, ok):
                stepwise[f"{c['cat']}|d{d}"][0] += int(o); stepwise[f"{c['cat']}|d{d}"][1] += 1
        summary["stepwise_external"] = {k: dict(n=n, final_acc=c / n) for k, (c, n) in sorted(stepwise.items())}
        summary["val_d2"] = summarize(eval_chains(model, v, P, rel, val_chains, device, loops=T_of_d))
        summary["val_w2"] = eval_w2_val(model, v, P, rel, val_w2, device, n_loops=T1)
        summary["checkpoint"] = ckpt_path; summary["eval_seconds"] = time.time() - t_eval
        json.dump(summary, open(os.path.join(args.save_dir, f"final_eval_{tag}.json"), "w"), indent=1)
        with open(os.path.join(args.save_dir, f"predictions_{tag}.jsonl"), "w") as f:
            for c, r in res:
                f.write(json.dumps(dict(r, x=c["x"], rels=c["rels"], gold_states=c["states"], new_adjacencies=c.get("new_adjacencies"))) + "\n")
        print(f"[final eval {tag}] " + " ".join(f"{k}:{s['final_acc']:.3f}/{s['traj_acc']:.3f}" for k, s in summary.items() if isinstance(s, dict) and "final_acc" in s and "|" in k and "repeated" not in k and "revisits" not in k))
        return summary

    if args.eval_only:
        full_eval(args.eval_tag or "evalonly", args.eval_only); return

    def loop_matrix(u):
        """depth x T matrix on the monitoring set (plan §6: fixed checkpoints, fixed 500 chains per cell, every T)."""
        out = {}
        for T in [int(x) for x in args.loop_matrix_T.split(",")]:
            out[f"T{T}"] = cells_of(eval_chains(model, v, P, rel, small_tests, device, loops=lambda d, T=T: T))
        json.dump(dict(update=u, matrix=out), open(os.path.join(args.save_dir, f"loop_matrix_{u:06d}.json"), "w"), indent=1)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    s_w2 = Stream(W2.n, args.seed, "w2"); s_d2 = Stream(D2.n, args.seed, "d2"); s_d3 = Stream(D3.n, args.seed, "d3") if D3 else None
    loop_rng = np.random.default_rng([args.seed, int(hashlib.sha256(b"loopT").hexdigest()[:8], 16)])
    matrix_at = {int(x) for x in args.loop_matrix_at.split(",")} if looped else set()
    total = args.phase1_updates + args.phase2_updates
    start = 1; best = (-1.0, float("inf"))
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl"); log_path = os.path.join(args.save_dir, "train_log.jsonl")
    last_path = os.path.join(args.save_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); s_w2.load(st["s_w2"]); s_d2.load(st["s_d2"])
        if s_d3 and st.get("s_d3"): s_d3.load(st["s_d3"])
        if st.get("loop_rng"): loop_rng.bit_generator.state = st["loop_rng"]
        torch.set_rng_state(st["torch_rng"]); start = st["update"] + 1; best = tuple(st["best"])
        print(f"resumed from update {st['update']}")
    else:
        open(metrics_path, "w").close(); open(log_path, "w").close()
        manifest = dict(protocol=P, seed=args.seed, n_params=n_params, rows_per_update=args.rows_per_update,
                        phase1_updates=args.phase1_updates, phase2_updates=args.phase2_updates, L=L,
                        w2=dict(rows=W2.n, tokens=W2.n_tokens, supervised_positions=W2.n_supervised),
                        d2=dict(rows=D2.n, tokens=D2.n_tokens, supervised_positions=D2.n_supervised),
                        d3=(dict(rows=D3.n, tokens=D3.n_tokens, supervised_positions=D3.n_supervised, per_update=args.d3_per_update) if D3 else None),
                        data_hashes=meta.get("hashes"), model=dict(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, max_len=args.max_len, rope_base=args.rope_base, use_rope=not args.no_rope),
                        optimizer=dict(lr=args.lr, weight_decay=args.weight_decay, warmup=args.warmup), vocab_size=len(v.vocab),
                        vocab_ext=args.vocab_ext, select_by=args.select_by,
                        loop=(dict(train_T=loop_T, cap=args.loop_cap, fixed_control=args.loop_fixed, zero_init_out=args.zero_init_out,
                                   rule="min(cap, smallest power of two >= max(8, d))", matrix_at=sorted(matrix_at), matrix_T=args.loop_matrix_T) if looped else None))
        json.dump(manifest, open(os.path.join(args.save_dir, "manifest.json"), "w"), indent=1)
    print(f"protocol {P} | params {n_params:,} | L {L} | w2 rows {W2.n} d2 rows {D2.n} | device {device}")

    run_ce = collections.defaultdict(float); run_cnt = collections.defaultdict(int); run_loss = 0.0; run_n = 0; tok_seen = 0
    t0 = time.time()
    for u in range(start, total + 1):
        model.train()
        phase = 1 if u <= args.phase1_updates else 2
        if phase == 1:
            idx_w2 = s_w2.take(args.rows_per_update); idx_d2 = None
        else:
            n_d3 = args.d3_per_update if D3 else 0
            idx_w2 = s_w2.take(args.rows_per_update // 2); idx_d2 = s_d2.take(args.rows_per_update // 2 - n_d3)
            idx_d3 = s_d3.take(n_d3) if n_d3 else None
        parts = [(W2, torch.from_numpy(idx_w2).to(device), "w2")]
        if idx_d2 is not None:
            parts.append((D2, torch.from_numpy(idx_d2).to(device), "d2"))
            if phase == 2 and D3 is not None and args.d3_per_update:
                parts.append((D3, torch.from_numpy(idx_d3).to(device), "d3"))
        inp = torch.cat([R.input_ids[i] for R, i, _ in parts]); tgt = torch.cat([R.target_ids[i] for R, i, _ in parts])
        w = torch.cat([R.weights[i] for R, i, _ in parts]); roles = torch.cat([R.roles[i] for R, i, _ in parts]); pm = torch.cat([R.pad_mask[i] for R, i, _ in parts])
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, u / max(1, args.warmup))
        T_u = int(loop_rng.choice(loop_T)) if looped else 1
        logits = model(inp, pad_mask=pm, n_loops=T_u)
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none", ignore_index=-100)
        loss = (ce * w).sum() / args.rows_per_update
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            run_loss += float(loss); run_n += 1; tok_seen += int((~pm).sum())
            off = 0
            for R, i, name in parts:
                k = len(i); sl = slice(off, off + k); off += k
                for rk, rname in enumerate(ROLE_NAMES):
                    if rk == 0:
                        continue
                    m = roles[sl] == rk
                    if m.any():
                        run_ce[f"{name}/{rname}"] += float(ce[sl][m].sum()); run_cnt[f"{name}/{rname}"] += int(m.sum())
        if u % 100 == 0 or u == 1:
            with open(log_path, "a") as f:
                f.write(json.dumps(dict(update=u, phase=phase, loss=run_loss / max(1, run_n), lr=opt.param_groups[0]["lr"], tokens_seen=tok_seen,
                                        elapsed=time.time() - t0, T=T_u, **{f"ce/{k}": run_ce[k] / run_cnt[k] for k in run_ce if run_cnt[k]})) + "\n")
            run_ce.clear(); run_cnt.clear(); run_loss = 0.0; run_n = 0
        if u % args.eval_every == 0 or u == total or u == args.phase1_updates:
            model.eval()
            t_eval = time.time()
            rec = dict(update=u, phase=phase, elapsed=time.time() - t0, tokens_seen=tok_seen,
                       tf_val=teacher_forced(model, VAL, device, n_loops=T1), tf_train_d2=teacher_forced(model, SUB_D2, device, n_loops=T1),
                       tf_train_w2=teacher_forced(model, SUB_W2, device, n_loops=T1), tf_val_w2=teacher_forced(model, VALW2, device, n_loops=T1))
            rec["atomic"] = summarize(eval_chains(model, v, P, rel, atomic, device, loops=T_of_d))
            rec["val_d2"] = summarize(eval_chains(model, v, P, rel, val_chains, device, loops=T_of_d))
            rec["val_w2"] = eval_w2_val(model, v, P, rel, val_w2, device, n_loops=T1)
            res = eval_chains(model, v, P, rel, small_tests, device, loops=T_of_d)
            cells = collections.defaultdict(list)
            for c, r in res:
                cells[f"{c['cat']}|d{c['d']}"].append((c, r))
            rec["test_small"] = {k: summarize(vv) for k, vv in sorted(cells.items())}
            if u in matrix_at:
                loop_matrix(u)
            rec["eval_seconds"] = time.time() - t_eval
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"u {u} ph{phase} | val_d2 {rec['val_d2']['final_acc']:.3f}/{rec['val_d2']['traj_acc']:.3f} | atomic {rec['atomic']['final_acc']:.3f} | "
                  + " ".join(f"{k}:{s['final_acc']:.2f}" for k, s in rec["test_small"].items()) + f" | eval {rec['eval_seconds']:.0f}s | {time.time() - t0:.0f}s")
            if args.select_by == "w2":
                score = (rec["val_w2"]["both_acc"], -rec["tf_val_w2"]["loss_per_row"]); best_name = "best_by_valw2.pt"
            else:
                score = (rec["val_d2"]["final_acc"], -rec["tf_val"]["loss_per_row"]); best_name = "best_by_val.pt"
            if score > (best[0], -best[1]):
                best = (score[0], -score[1]); torch.save(dict(model=model.state_dict(), update=u), os.path.join(args.save_dir, best_name))
            if u % args.save_every == 0 or u == total:
                torch.save(dict(model=model.state_dict(), update=u), os.path.join(args.save_dir, f"ckpt_{u:06d}.pt"))
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), update=u, s_w2=s_w2.state(), s_d2=s_d2.state(),
                                s_d3=(s_d3.state() if s_d3 else None), loop_rng=loop_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                best=list(best), tokens_seen=tok_seen), last_path)
    if not args.no_final_eval:
        full_eval("final")
        extra = [] if args.final_extra_ckpts == "" else args.final_extra_ckpts.split(",")
        if args.final_extra_ckpts == "auto":
            extra = [f"ckpt_{total - k * args.save_every:06d}.pt" for k in (1, 2) if total - k * args.save_every > 0]
        for ck in extra:
            if os.path.exists(os.path.join(args.save_dir, ck)):
                full_eval(ck.replace(".pt", ""), os.path.join(args.save_dir, ck))
        best_name = "best_by_valw2.pt" if args.select_by == "w2" else "best_by_val.pt"
        if os.path.exists(os.path.join(args.save_dir, best_name)):
            full_eval("best", os.path.join(args.save_dir, best_name))
    print("done")


if __name__ == "__main__":
    main()
