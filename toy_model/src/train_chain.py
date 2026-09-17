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
def rollout(model, prompts, max_new, device, batch=None):
    batch = batch or ROLLOUT_BATCH
    """Greedy free generation for prompts of EQUAL length. Returns (N, max_new) generated ids."""
    outs = []
    for s in range(0, len(prompts), batch):
        x = torch.tensor(prompts[s:s + batch], dtype=torch.long, device=device)
        for _ in range(max_new):
            logits = model(x)[:, -1, :]
            x = torch.cat([x, logits.argmax(-1, keepdim=True)], 1)
        outs.append(x[:, -max_new:].cpu())
    return torch.cat(outs).numpy()


def eval_chains(model, v, protocol, rel, chains, device, keep_predictions=False):
    """chains: list of dict(x, rels, states, ...). Groups by depth. Returns per-chain scores (and parsed outputs)."""
    results = []
    by_d = collections.defaultdict(list)
    for c in chains:
        by_d[len(c["rels"])].append(c)
    for d, cs in by_d.items():
        prompts = [prompt_ids(v, c["x"], c["rels"]) for c in cs]
        gen = rollout(model, prompts, completion_length(protocol, d) + 2, device)
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
def eval_w2_val(model, v, protocol, rel, val_w2, device):
    """Second question answered with the model's OWN first output as history (plan §5)."""
    correct = 0; both = 0
    for q in val_w2:
        (x, r), (z, s) = q["q1"], q["q2"]
        p1 = prompt_ids(v, x, [r]); g1 = rollout(model, [p1], completion_length(protocol, 1) + 2, device)[0].tolist()
        end1 = g1.index(v.END) + 1 if v.END in g1 else len(g1)
        hist = p1 + g1[:end1] + prompt_ids(v, z, [s])
        g2 = rollout(model, [hist], completion_length(protocol, 1) + 2, device)[0].tolist()
        s1 = score_chain(v, protocol, rel, x, [r], [rel[r][x]], parse_completion(v, protocol, g1, 1))
        s2 = score_chain(v, protocol, rel, z, [s], [rel[s][z]], parse_completion(v, protocol, g2, 1))
        correct += int(s2["final_correct"]); both += int(s1["final_correct"] and s2["final_correct"])
    return dict(n=len(val_w2), second_acc=correct / len(val_w2), both_acc=both / len(val_w2))


@torch.no_grad()
def teacher_forced(model, R: Rendered, device, batch=1024):
    """Mean CE per role and the weighted loss per row."""
    sums = collections.defaultdict(float); cnts = collections.defaultdict(int); wloss = 0.0
    for s in range(0, R.n, batch):
        sl = slice(s, s + batch)
        logits = model(R.input_ids[sl], pad_mask=R.pad_mask[sl])
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
    args = ap.parse_args()
    global ROLLOUT_BATCH
    ROLLOUT_BATCH = args.rollout_batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = load_json(args.data_dir, "meta"); rel = meta["rel_map"]
    v = ChainVocab(load_json(args.data_dir, "vocab"))
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
                            max_len=args.max_len, rope_base=args.rope_base, use_rope=not args.no_rope).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    def full_eval(tag, ckpt_path=None):
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        model.eval()
        pool = tests
        if args.final_max_per_cell:
            cnt = collections.Counter(); pool = []
            for t in tests:
                if cnt[(t["cat"], t["d"])] < args.final_max_per_cell:
                    pool.append(t); cnt[(t["cat"], t["d"])] += 1
        res = eval_chains(model, v, P, rel, pool, device, keep_predictions=True)
        cells = collections.defaultdict(list)
        for c, r in res:
            cells[(c["cat"], c["d"])].append((c, r))
            for flag in ("repeated_relation", "revisits_entity"):
                if c.get(flag):
                    cells[(f"{c['cat']}|{flag}", c["d"])].append((c, r))
        summary = {f"{k[0]}|d{k[1]}": summarize(vv) for k, vv in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))}
        summary["atomic"] = summarize(eval_chains(model, v, P, rel, atomic, device))
        # plan §6.1: EXTERNAL step-wise calling -- the program feeds the model one-step prompts Q s_t r_(t+1) ANS and
        # chains the predicted entities itself (identifies whether the atomic update is reliable; not autonomous)
        stepwise = collections.defaultdict(lambda: [0, 0])
        for d in sorted({t["d"] for t in pool}):
            cs = [t for t in pool if t["d"] == d]
            cur = [c["x"] for c in cs]; ok = [True] * len(cs)
            for step in range(d):
                prompts = [prompt_ids(v, s_, [c["rels"][step]]) for s_, c in zip(cur, cs)]
                gen = rollout(model, prompts, completion_length(P, 1) + 1, device)
                nxt = []
                for i, g in enumerate(gen):
                    pr = parse_completion(v, P, g.tolist(), 1)
                    st_ = pr["states"][-1] if pr["states"] else -1
                    ok[i] = ok[i] and (st_ == cs[i]["states"][step]); nxt.append(st_ if st_ >= 0 else 0)
                cur = nxt
            for c, o in zip(cs, ok):
                stepwise[f"{c['cat']}|d{d}"][0] += int(o); stepwise[f"{c['cat']}|d{d}"][1] += 1
        summary["stepwise_external"] = {k: dict(n=n, final_acc=c / n) for k, (c, n) in sorted(stepwise.items())}
        summary["val_d2"] = summarize(eval_chains(model, v, P, rel, val_chains, device))
        summary["val_w2"] = eval_w2_val(model, v, P, rel, val_w2, device)
        summary["checkpoint"] = ckpt_path
        json.dump(summary, open(os.path.join(args.save_dir, f"final_eval_{tag}.json"), "w"), indent=1)
        with open(os.path.join(args.save_dir, f"predictions_{tag}.jsonl"), "w") as f:
            for c, r in res:
                f.write(json.dumps(dict(r, x=c["x"], rels=c["rels"], gold_states=c["states"], new_adjacencies=c.get("new_adjacencies"))) + "\n")
        print(f"[final eval {tag}] " + " ".join(f"{k}:{s['final_acc']:.3f}/{s['traj_acc']:.3f}" for k, s in summary.items() if isinstance(s, dict) and "final_acc" in s and "|" in k and "repeated" not in k and "revisits" not in k))
        return summary

    if args.eval_only:
        full_eval("evalonly", args.eval_only); return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    s_w2 = Stream(W2.n, args.seed, "w2"); s_d2 = Stream(D2.n, args.seed, "d2"); s_d3 = Stream(D3.n, args.seed, "d3") if D3 else None
    total = args.phase1_updates + args.phase2_updates
    start = 1; best = (-1.0, float("inf"))
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl"); log_path = os.path.join(args.save_dir, "train_log.jsonl")
    last_path = os.path.join(args.save_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); s_w2.load(st["s_w2"]); s_d2.load(st["s_d2"])
        if s_d3 and st.get("s_d3"): s_d3.load(st["s_d3"])
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
                        optimizer=dict(lr=args.lr, weight_decay=args.weight_decay, warmup=args.warmup), vocab_size=len(v.vocab))
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
        logits = model(inp, pad_mask=pm)
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
                                        elapsed=time.time() - t0, **{f"ce/{k}": run_ce[k] / run_cnt[k] for k in run_ce if run_cnt[k]})) + "\n")
            run_ce.clear(); run_cnt.clear(); run_loss = 0.0; run_n = 0
        if u % args.eval_every == 0 or u == total or u == args.phase1_updates:
            model.eval()
            rec = dict(update=u, phase=phase, elapsed=time.time() - t0, tokens_seen=tok_seen,
                       tf_val=teacher_forced(model, VAL, device), tf_train_d2=teacher_forced(model, SUB_D2, device), tf_train_w2=teacher_forced(model, SUB_W2, device))
            rec["atomic"] = summarize(eval_chains(model, v, P, rel, atomic, device))
            rec["val_d2"] = summarize(eval_chains(model, v, P, rel, val_chains, device))
            rec["val_w2"] = eval_w2_val(model, v, P, rel, val_w2, device)
            res = eval_chains(model, v, P, rel, small_tests, device)
            cells = collections.defaultdict(list)
            for c, r in res:
                cells[f"{c['cat']}|d{c['d']}"].append((c, r))
            rec["test_small"] = {k: summarize(vv) for k, vv in sorted(cells.items())}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"u {u} ph{phase} | val_d2 {rec['val_d2']['final_acc']:.3f}/{rec['val_d2']['traj_acc']:.3f} | atomic {rec['atomic']['final_acc']:.3f} | "
                  + " ".join(f"{k}:{s['final_acc']:.2f}" for k, s in rec["test_small"].items()) + f" | {time.time() - t0:.0f}s")
            score = (rec["val_d2"]["final_acc"], -rec["tf_val"]["loss_per_row"])
            if score > (best[0], -best[1]):
                best = (score[0], -score[1]); torch.save(dict(model=model.state_dict(), update=u), os.path.join(args.save_dir, "best_by_val.pt"))
            if u % args.save_every == 0 or u == total:
                torch.save(dict(model=model.state_dict(), update=u), os.path.join(args.save_dir, f"ckpt_{u:06d}.pt"))
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), update=u, s_w2=s_w2.state(), s_d2=s_d2.state(),
                                s_d3=(s_d3.state() if s_d3 else None), torch_rng=torch.get_rng_state(), best=list(best), tokens_seen=tok_seen), last_path)
    if not args.no_final_eval:
        full_eval("final")
        if os.path.exists(os.path.join(args.save_dir, "best_by_val.pt")):
            full_eval("best", os.path.join(args.save_dir, "best_by_val.pt"))
    print("done")


if __name__ == "__main__":
    main()
