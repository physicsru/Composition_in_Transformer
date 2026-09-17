"""
Local-executor arms L / L-w2 / L-history of docs/experiments_loop_next.md (§4, §5, §7, §8, §9).

  python train_local.py --data_dir ../data/chain_loop --save_dir ../runs/loop_L_s1 --seed 1            # L
  python train_local.py ... --w2_only                                                                   # L-w2
  python train_local.py ... --history --max_len 4096                                                    # L-history

Source questions are ONLY the frozen w2 / d2 questions of the chain dataset; they are compiled into local calls
(data/local_format.py). Schedule as train_chain.py: 50k updates of 256 w2 presentations, then 200k updates of
128 w2 + 128 d2 (L-w2: 256 w2). Every presentation supplies 2 supervised STEP calls + 1 supervised HALT call
(w2: one of its two HALTs chosen by a frozen RNG, the other visible without loss);
loss = sum over calls of the weighted token CE / 256 = mean over presentations of
0.5 * mean(L_STEP1, L_STEP2) + 0.5 * L_HALT_selected.  AdamW lr 3e-4, wd 0.1, warmup 100, dropout 0.

Arm L: every call is its own sequence (context reset).  Arm L-history: the calls of one task (a d2 question, or one
sub-question of a w2 row) are one sequence, RoPE positions restart at 0 in every call, causal mask over the record.

Every --eval_every updates: the exhaustive local table (10,000 (x, r) + 500 (x, EOS) single-call greedy outputs),
the w2-only selection metric (free execution of the 20 validation w2 rows = 40 one-step tasks; ties: their
teacher-forced loss) -> best_by_valw2.pt, the fixed monitoring set (first --monitor_per_cell chains of every
(category, depth) cell) executed with REAL network calls by the runner (own predicted entity fed back, no forced
decoding, budget d+2 calls), and for L / L-w2 the table replay of the same chains with the four diagnostics of §9.2
plus the real-call vs table-replay agreement.  Checkpoints ckpt_UUUUUU.pt every --save_every, last.pt = full
state (optimizer, streams, HALT RNG, torch RNG, cumulative exposure counts).  At the end the full test matrix is
run for the final checkpoint, the two preceding ones (240k, 245k) and best_by_valw2 -> final_eval_<tag>.json
(+ predictions_<tag>.jsonl with the raw tokens of every call for final / best).
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

from data.local_format import (HALT_LEN, MAX_OUT, PROMPT_LEN, STEP_LEN, LocalVocab, compile_d2, compile_w2, extend_vocab,
                               halt_prompt, parse_call_output, score_run, step_prompt)
from model import GPT2LikeEncoder
from train_chain import Stream, load_json


# ------------------------------------------------------------------------------------------------------------ units
class Units:
    """Pre-rendered training units of one source type. history=False: one unit per call; True: one unit per task.
    A source question owns `per_q` consecutive units. Tensors are aligned with the TARGETS (inputs = ids[:-1])."""

    def __init__(self, tasks_per_q, history, pad, device):
        rows = []
        self.per_q = None
        for tasks in tasks_per_q:
            units = []
            for ti, calls in enumerate(tasks):
                if history:
                    ids, w, pos, kind = [], [], [], []
                    for c in calls:
                        ids += c["ids"]; w += c["w"]; pos += list(range(len(c["ids"])))
                        kind += [(1 if c["kind"] == "step" else 2) if wt > 0 else 0 for wt in c["w"]]
                    units.append((ids, w, pos, ti, kind))
                else:
                    for c in calls:
                        units.append((c["ids"], c["w"], list(range(len(c["ids"]))), ti,
                                      [(1 if c["kind"] == "step" else 2) if wt > 0 else 0 for wt in c["w"]]))
            if self.per_q is None:
                self.per_q = len(units)
            assert len(units) == self.per_q
            rows += units
        n = len(rows); L = max(len(r[0]) for r in rows) - 1
        inp = torch.full((n, L), pad, dtype=torch.long); tgt = torch.full((n, L), -100, dtype=torch.long)
        w = torch.zeros((n, L)); pos = torch.zeros((n, L), dtype=torch.long); pm = torch.ones((n, L), dtype=torch.bool)
        kind = torch.zeros((n, L), dtype=torch.int8); task = torch.zeros(n, dtype=torch.long)
        for i, (ids, ww, pp, ti, kk) in enumerate(rows):
            k = len(ids) - 1
            inp[i, :k] = torch.tensor(ids[:-1]); tgt[i, :k] = torch.tensor(ids[1:]); w[i, :k] = torch.tensor(ww[1:])
            pos[i, :k] = torch.tensor(pp[:-1]); pm[i, :k] = False; kind[i, :k] = torch.tensor(kk[1:], dtype=torch.int8); task[i] = ti
        self.inp, self.tgt, self.w, self.pos, self.pm, self.kind, self.task = (t.to(device) for t in (inp, tgt, w, pos, pm, kind, task))
        self.n_q = len(tasks_per_q); self.n = n; self.L = L
        self.n_tokens = int((~pm).sum()); self.n_supervised = int((w > 0).sum())

    def batch(self, qi: torch.Tensor, sel: torch.Tensor):
        """qi: question indices (long, device); sel: selected task index per question for the HALT loss (0 for d2)."""
        rows = (qi[:, None] * self.per_q + torch.arange(self.per_q, device=qi.device)[None]).reshape(-1)
        w = self.w[rows].clone()
        kill = (self.task[rows] != sel.repeat_interleave(self.per_q))[:, None] & (self.kind[rows] == 2)
        w[kill] = 0.0
        return self.inp[rows], self.tgt[rows], w, self.pos[rows], self.pm[rows], self.kind[rows]


# ----------------------------------------------------------------------------------------------------------- runner
@torch.no_grad()
def generate_calls(model, prompts, poss, max_out, device):
    x = torch.tensor(prompts, dtype=torch.long, device=device)
    p = torch.tensor(poss, dtype=torch.long, device=device) if poss is not None else None
    gen, margins = [], []
    for _ in range(max_out):
        logits = model(x, pos_ids=p)[:, -1]
        top2 = logits.topk(2, -1).values
        margins.append(top2[:, 0] - top2[:, 1])
        nxt = logits.argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], 1)
        if p is not None:
            p = torch.cat([p, p[:, -1:] + 1], 1)
        gen.append(nxt)
    return torch.cat(gen, 1).cpu().tolist(), torch.stack(margins, 1).cpu().tolist()


@torch.no_grad()
def run_tasks(model, v, rel, tasks, history, device, rollout_batch=1000, max_out=MAX_OUT, budget_extra=2):
    """Execute tasks (dict x, rels, states) with the runner of plan §4.3: context reset per call (or the task's record
    kept in context with per-call positions when history=True), the model's own entity fed back, no forced decoding,
    call budget d + budget_extra. Returns [(task, score, steps)] in the input order."""
    out = [None] * len(tasks)
    by_d = collections.defaultdict(list)
    for i, t in enumerate(tasks):
        by_d[len(t["rels"])].append(i)
    for d, idxs in sorted(by_d.items()):
        L_max = (d * (PROMPT_LEN + STEP_LEN) + PROMPT_LEN + max_out) if history else (PROMPT_LEN + max_out)
        batch = max(16, min(rollout_batch, int(2.0e5 / L_max)))
        for s in range(0, len(idxs), batch):
            ids_b = idxs[s:s + batch]; ts = [tasks[i] for i in ids_b]; n = len(ts)
            cur = [t["x"] for t in ts]; ptr = [0] * n; alive = list(range(n)); steps = [[] for _ in range(n)]
            ctx = [[] for _ in range(n)]; cpos = [[] for _ in range(n)]
            for call in range(d + budget_extra):
                if not alive:
                    break
                prompts, poss, heads = [], [], []
                for i in alive:
                    head = ts[i]["rels"][ptr[i]] if ptr[i] < d else None
                    pr = step_prompt(v, cur[i], head) if head is not None else halt_prompt(v, cur[i])
                    heads.append(head)
                    prompts.append((ctx[i] if history else []) + pr)
                    poss.append((cpos[i] if history else []) + list(range(PROMPT_LEN)))
                gen, _ = generate_calls(model, prompts, poss if history else None, max_out, device)
                new_alive = []
                for j, i in enumerate(alive):
                    pc = parse_call_output(v, gen[j]); head = heads[j]
                    expected = int(rel[head][cur[i]]) if head is not None else cur[i]
                    steps[i].append(dict(call=call, pointer=ptr[i], state_in=cur[i], head=head, action=pc["action"],
                                         entity=pc["entity"], legal=pc["legal"], expected=expected, raw=pc["raw"]))
                    if pc["legal"] and pc["action"] == "STEP" and head is not None:
                        if history:
                            ctx[i] = prompts[j] + gen[j][:STEP_LEN]
                            cpos[i] = poss[j] + list(range(PROMPT_LEN, PROMPT_LEN + STEP_LEN))
                        cur[i] = pc["entity"]; ptr[i] += 1; new_alive.append(i)
                alive = new_alive
            for i, t in enumerate(ts):
                out[ids_b[i]] = (t, score_run(d, t["states"], steps[i]), steps[i])
    return out


@torch.no_grad()
def local_table(model, v, rel, E, R, device, batch=1000, max_out=MAX_OUT):
    """Greedy single-call outputs for all 10,000 (x, r) and 500 (x, EOS) canonical inputs (plan §9.1)."""
    keys = [(x, r) for r in range(R) for x in range(E)] + [(x, None) for x in range(E)]
    prompts = [step_prompt(v, x, r) if r is not None else halt_prompt(v, x) for x, r in keys]
    table = {}
    for s in range(0, len(prompts), batch):
        gen, marg = generate_calls(model, prompts[s:s + batch], None, max_out, device)
        for (x, r), g, m in zip(keys[s:s + batch], gen, marg):
            pc = parse_call_output(v, g)
            table[(x, r)] = dict(action=pc["action"], entity=pc["entity"], legal=pc["legal"], raw=g,
                                 margin=float(min(m[:pc["n_used"]])) if pc["legal"] else float(min(m)))
    st = dict(step=dict(n=E * R, correct=0, wrong_entity=0, halt_instead=0, illegal=0),
              halt=dict(n=E, correct=0, wrong_entity=0, step_instead=0, illegal=0), min_margin_correct=None)
    mm = []
    for (x, r), e in table.items():
        b = st["step"] if r is not None else st["halt"]
        if not e["legal"]:
            b["illegal"] += 1
        elif r is not None and e["action"] == "HALT":
            b["halt_instead"] += 1
        elif r is None and e["action"] == "STEP":
            b["step_instead"] += 1
        elif e["entity"] == (int(rel[r][x]) if r is not None else x):
            b["correct"] += 1; mm.append(e["margin"])
        else:
            b["wrong_entity"] += 1
    st["step"]["acc"] = st["step"]["correct"] / st["step"]["n"]; st["halt"]["acc"] = st["halt"]["correct"] / st["halt"]["n"]
    st["all_correct"] = bool(st["step"]["correct"] == st["step"]["n"] and st["halt"]["correct"] == st["halt"]["n"])
    st["min_margin_correct"] = float(min(mm)) if mm else None
    return table, st


def replay(table, rel, task, update="model", control="model", budget_extra=2):
    """Table replay of one chain (plan §9.2). update / control in {model, correct}. Missing entities are reported as
    their own error types (missing_update / missing_halt_entity) and never filled in with the truth."""
    d = len(task["rels"]); cur = task["x"]; ptr = 0; steps = []
    for call in range(d + budget_extra):
        head = task["rels"][ptr] if ptr < d else None
        e = table[(cur, head)]
        expected = int(rel[head][cur]) if head is not None else cur
        if control == "model":
            action = e["action"] if e["legal"] else None
        else:
            action = "STEP" if head is not None else "HALT"
        rec = dict(call=call, pointer=ptr, state_in=cur, head=head, action=action, entity=None, legal=action is not None, expected=expected)
        if action is None:
            steps.append(rec); break
        if update == "correct":
            rec["entity"] = expected
        elif e["legal"] and e["action"] == action:
            rec["entity"] = e["entity"]
        elif not e["legal"]:
            rec.update(legal=False, error="format"); steps.append(rec); break
        else:
            rec.update(legal=False, error="missing_update" if action == "STEP" else "missing_halt_entity"); steps.append(rec); break
        steps.append(rec)
        if action == "STEP" and head is not None:
            cur = rec["entity"]; ptr += 1
        else:
            break
    return score_run(d, task["states"], steps), steps


def summarize(items):
    """items: iterable of score dicts."""
    items = list(items); n = len(items)
    if n == 0:
        return {}
    fe = collections.Counter(s["first_error_type"] for s in items if s["first_error_type"])
    own_c = sum(s["own_update_correct"] for s in items); own_t = sum(s["own_update_total"] for s in items)
    return dict(n=n, final_acc=sum(s["final_correct"] for s in items) / n, traj_acc=sum(s["traj_correct"] for s in items) / n,
                own_update_acc=(own_c / own_t) if own_t else None, own_update_total=own_t, halted=sum(s["halted"] for s in items) / n,
                mean_calls=sum(s["n_calls"] for s in items) / n, first_error={k: v / n for k, v in sorted(fe.items())})


def cell_summaries(results, with_flags=True):
    cells = collections.defaultdict(list)
    for t, sc, _ in results:
        cells[(t["cat"], t["d"])].append(sc)
        if with_flags:
            for flag in ("repeated_relation", "revisits_entity"):
                if t.get(flag):
                    cells[(f"{t['cat']}|{flag}", t["d"])].append(sc)
    return {f"{k[0]}|d{k[1]}": summarize(v) for k, v in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))}


def steps_agree(a, b):
    return [(s["action"], s["entity"], s["legal"]) for s in a] == [(s["action"], s["entity"], s["legal"]) for s in b]


@torch.no_grad()
def teacher_forced(model, U: Units, device, batch=1024):
    sums = collections.defaultdict(float); cnts = collections.defaultdict(int); wl = 0.0
    for s in range(0, U.n, batch):
        sl = slice(s, s + batch)
        logits = model(U.inp[sl], pad_mask=U.pm[sl], pos_ids=U.pos[sl])
        ce = F.cross_entropy(logits.transpose(1, 2), U.tgt[sl], reduction="none", ignore_index=-100)
        wl += float((ce * U.w[sl]).sum())
        for k, name in ((1, "step"), (2, "halt")):
            m = U.kind[sl] == k
            sums[name] += float(ce[m].sum()); cnts[name] += int(m.sum())
    out = {f"ce/{k}": sums[k] / cnts[k] for k in sums if cnts[k]}
    out["loss_per_question"] = wl / U.n_q
    return out


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--save_dir", required=True)
    ap.add_argument("--history", action="store_true", help="arm L-history: keep the calls of one task in context")
    ap.add_argument("--w2_only", action="store_true", help="arm L-w2: phase 2 uses 256 w2 presentations instead of 128 w2 + 128 d2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--n_layer", type=int, default=2); ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64); ap.add_argument("--rope_base", type=float, default=100.0)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight_decay", type=float, default=0.1); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rows_per_update", type=int, default=256)
    ap.add_argument("--phase1_updates", type=int, default=50000); ap.add_argument("--phase2_updates", type=int, default=200000)
    ap.add_argument("--eval_every", type=int, default=5000); ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--monitor_per_cell", type=int, default=500)
    ap.add_argument("--rollout_batch", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval_only", default=None, help="checkpoint to evaluate on the full test matrix (no training)")
    ap.add_argument("--eval_tag", default=None)
    ap.add_argument("--no_final_eval", action="store_true")
    ap.add_argument("--final_max_per_cell", type=int, default=0, help="limit chains per cell in the full evaluation (0 = all; smoke tests)")
    ap.add_argument("--final_extra_ckpts", default="auto", help="comma-separated extra checkpoints for the full evaluation; auto = the two before the last")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    arm = "L-history" if args.history else ("L-w2" if args.w2_only else "L")

    meta = load_json(args.data_dir, "meta"); rel = meta["rel_map"]; E = meta["num_entities"]; R = meta["num_relations"]
    vocab = extend_vocab(load_json(args.data_dir, "vocab")); v = LocalVocab(vocab)
    vocab_hash = hashlib.sha256(json.dumps(vocab).encode()).hexdigest()[:16]
    train_w2 = load_json(args.data_dir, "train_w2"); train_d2 = load_json(args.data_dir, "train_d2")
    val_d2 = load_json(args.data_dir, "val_d2"); val_w2 = load_json(args.data_dir, "val_w2"); tests = load_json(args.data_dir, "tests")
    H = args.history
    W2 = Units([compile_w2(v, rel, q) for q in train_w2], H, v.pad, device)
    D2 = Units([compile_d2(v, rel, q) for q in train_d2], H, v.pad, device)
    VW2 = Units([compile_w2(v, rel, q) for q in val_w2], H, v.pad, device)
    # exposure bookkeeping per source question: STEP facts (r * E + x) and HALT entities
    w2_facts = np.array([[q["q1"][1] * E + q["q1"][0], q["q2"][1] * E + q["q2"][0]] for q in train_w2])
    w2_halts = np.array([[rel[q["q1"][1]][q["q1"][0]], rel[q["q2"][1]][q["q2"][0]]] for q in train_w2])
    d2_facts = np.array([[q["r"] * E + q["x"], q["s"] * E + rel[q["r"]][q["x"]]] for q in train_d2])
    d2_halts = np.array([[rel[q["s"]][rel[q["r"]][q["x"]]]] for q in train_d2])
    exp = dict(step=np.zeros(E * R, dtype=np.int64), halt_sup=np.zeros(E, dtype=np.int64), halt_vis=np.zeros(E, dtype=np.int64),
               presentations=dict(w2=0, d2=0))
    # tasks for the runner
    val_w2_tasks = [dict(x=x, rels=[r], states=[rel[r][x]]) for q in val_w2 for (x, r) in (q["q1"], q["q2"])]
    val_d2_tasks = [dict(x=q["x"], rels=[q["r"], q["s"]], states=[rel[q["r"]][q["x"]], rel[q["s"]][rel[q["r"]][q["x"]]]]) for q in val_d2]
    per_cell = collections.Counter(); monitor = []
    for t in tests:
        if per_cell[(t["cat"], t["d"])] < args.monitor_per_cell:
            monitor.append(t); per_cell[(t["cat"], t["d"])] += 1

    model = GPT2LikeEncoder(len(vocab), d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, dropout=0.0,
                            max_len=args.max_len, rope_base=args.rope_base).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    def evaluate(pool, tag=None, keep_steps=False):
        """Real-call execution of pool (+ table replay / diagnostics / agreement for the reset arms)."""
        model.eval()
        res = run_tasks(model, v, rel, pool, H, device, args.rollout_batch)
        out = dict(cells=cell_summaries(res))
        table, tstats = local_table(model, v, rel, E, R, device)
        out["table"] = tstats
        if not H:
            diag = {}
            for name, (upd, ctl) in dict(model_model=("model", "model"), model_correct=("model", "correct"),
                                          correct_model=("correct", "model"), correct_correct=("correct", "correct")).items():
                rr = [(t, replay(table, rel, t, upd, ctl)[0], None) for t in pool]
                diag[name] = cell_summaries(rr, with_flags=False)
            out["replay"] = diag
            agree = collections.defaultdict(lambda: [0, 0])
            for t, sc, steps in res:
                _, rsteps = replay(table, rel, t)
                k = f"{t['cat']}|d{t['d']}"; agree[k][0] += int(steps_agree(steps, rsteps)); agree[k][1] += 1
            out["agreement_real_vs_table"] = {k: dict(n=n, agree=c / n) for k, (c, n) in sorted(agree.items())}
        if tag is not None:
            with open(os.path.join(args.save_dir, f"table_{tag}.json"), "w") as f:
                json.dump([dict(x=x, head=r, **{k: e[k] for k in ("action", "entity", "legal", "raw", "margin")}) for (x, r), e in table.items()], f)
            if keep_steps:
                with open(os.path.join(args.save_dir, f"predictions_{tag}.jsonl"), "w") as f:
                    for t, sc, steps in res:
                        f.write(json.dumps(dict(id=t.get("id"), cat=t.get("cat"), d=len(t["rels"]), x=t["x"], rels=t["rels"], gold=t["states"],
                                                new_adjacencies=t.get("new_adjacencies"), score=sc,
                                                calls=[[s["pointer"], s["state_in"], -1 if s["head"] is None else s["head"], s["raw"]] for s in steps])) + "\n")
        return out

    def selection_metric():
        r = run_tasks(model, v, rel, val_w2_tasks, H, device, args.rollout_batch)
        tf = teacher_forced(model, VW2, device)
        return dict(n=len(r), traj_acc=sum(sc["traj_correct"] for _, sc, _ in r) / len(r), final_acc=sum(sc["final_correct"] for _, sc, _ in r) / len(r), **tf)

    def full_eval(tag, ckpt_path=None):
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
        pool = tests
        if args.final_max_per_cell:
            cnt = collections.Counter(); pool = []
            for t in tests:
                if cnt[(t["cat"], t["d"])] < args.final_max_per_cell:
                    pool.append(t); cnt[(t["cat"], t["d"])] += 1
        t0 = time.time()
        out = evaluate(pool, tag=tag, keep_steps=tag in ("final", "best", "evalonly"))
        out["val_w2"] = selection_metric()
        vd = run_tasks(model, v, rel, val_d2_tasks, H, device, args.rollout_batch); out["val_d2"] = summarize(sc for _, sc, _ in vd)
        out["checkpoint"] = ckpt_path; out["arm"] = arm; out["eval_seconds"] = time.time() - t0
        json.dump(out, open(os.path.join(args.save_dir, f"final_eval_{tag}.json"), "w"), indent=1)
        print(f"[final eval {tag}] table step {out['table']['step']['acc']:.4f} halt {out['table']['halt']['acc']:.4f} | "
              + " ".join(f"{k}:{s['final_acc']:.3f}/{s['traj_acc']:.3f}" for k, s in out["cells"].items() if "|repeated" not in k and "|revisits" not in k)
              + f" | {out['eval_seconds']:.0f}s")
        return out

    if args.eval_only:
        full_eval(args.eval_tag or "evalonly", args.eval_only); return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    s_w2 = Stream(W2.n_q, args.seed, "w2"); s_d2 = Stream(D2.n_q, args.seed, "d2")
    halt_rng = np.random.default_rng([args.seed, int(hashlib.sha256(b"halt").hexdigest()[:8], 16)])
    total = args.phase1_updates + args.phase2_updates
    start = 1; best = (-1.0, float("inf"))
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl"); log_path = os.path.join(args.save_dir, "train_log.jsonl")
    last_path = os.path.join(args.save_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); s_w2.load(st["s_w2"]); s_d2.load(st["s_d2"])
        halt_rng.bit_generator.state = st["halt_rng"]; torch.set_rng_state(st["torch_rng"]); start = st["update"] + 1; best = tuple(st["best"])
        exp = dict(step=np.array(st["exp"]["step"]), halt_sup=np.array(st["exp"]["halt_sup"]), halt_vis=np.array(st["exp"]["halt_vis"]), presentations=st["exp"]["presentations"])
        print(f"resumed from update {st['update']}")
    else:
        open(metrics_path, "w").close(); open(log_path, "w").close()
        calls_w2 = 4; calls_d2 = 3
        manifest = dict(arm=arm, history=H, w2_only=args.w2_only, seed=args.seed, n_params=n_params, vocab_size=len(vocab), vocab_hash=vocab_hash,
                        data_hashes=meta.get("hashes"), rows_per_update=args.rows_per_update, phase1_updates=args.phase1_updates, phase2_updates=args.phase2_updates,
                        source_questions=dict(w2=W2.n_q, d2=D2.n_q, unique=W2.n_q + (0 if args.w2_only else D2.n_q), unique_atomic_facts=E * R),
                        units=dict(w2=dict(units=W2.n, per_question=W2.per_q, L=W2.L, tokens=W2.n_tokens, supervised_positions_max=W2.n_supervised),
                                   d2=dict(units=D2.n, per_question=D2.per_q, L=D2.L, tokens=D2.n_tokens, supervised_positions_max=D2.n_supervised)),
                        calls_per_update=dict(phase1=dict(calls=256 * calls_w2, step_supervised=512, halt_supervised=256, halt_visible=512),
                                              phase2=(dict(calls=256 * calls_w2, step_supervised=512, halt_supervised=256, halt_visible=512) if args.w2_only
                                                      else dict(calls=128 * calls_w2 + 128 * calls_d2, step_supervised=512, halt_supervised=256, halt_visible=384))),
                        tokens_per_call=dict(step=PROMPT_LEN + STEP_LEN, halt=PROMPT_LEN + HALT_LEN),
                        loss="sum_calls w_t CE_t / rows_per_update; w = 1/16 per STEP token, 1/6 per selected-HALT token",
                        model=dict(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, max_len=args.max_len, rope_base=args.rope_base, per_call_positions=H),
                        optimizer=dict(lr=args.lr, weight_decay=args.weight_decay, warmup=args.warmup), runner=dict(max_out=MAX_OUT, budget="d+2", forced_decoding=False))
        json.dump(manifest, open(os.path.join(args.save_dir, "manifest.json"), "w"), indent=1)
    print(f"arm {arm} | params {n_params:,} | units w2 {W2.n}x{W2.L} d2 {D2.n}x{D2.L} | device {device}")

    run_ce = collections.defaultdict(float); run_cnt = collections.defaultdict(int); run_loss = 0.0; run_n = 0; tok_seen = 0; calls_seen = 0
    t0 = time.time()
    for u in range(start, total + 1):
        model.train()
        phase = 1 if u <= args.phase1_updates else 2
        n_w2 = args.rows_per_update if (phase == 1 or args.w2_only) else args.rows_per_update // 2
        n_d2 = args.rows_per_update - n_w2
        qi_w2 = s_w2.take(n_w2); sel_w2 = halt_rng.integers(2, size=n_w2)
        parts = [W2.batch(torch.from_numpy(qi_w2).to(device), torch.from_numpy(sel_w2).to(device))]
        exp["step"] += np.bincount(w2_facts[qi_w2].reshape(-1), minlength=E * R); exp["halt_vis"] += np.bincount(w2_halts[qi_w2].reshape(-1), minlength=E)
        exp["halt_sup"] += np.bincount(w2_halts[qi_w2, sel_w2], minlength=E); exp["presentations"]["w2"] += int(n_w2)
        if n_d2:
            qi_d2 = s_d2.take(n_d2)
            parts.append(D2.batch(torch.from_numpy(qi_d2).to(device), torch.zeros(n_d2, dtype=torch.long, device=device)))
            exp["step"] += np.bincount(d2_facts[qi_d2].reshape(-1), minlength=E * R); exp["halt_vis"] += np.bincount(d2_halts[qi_d2, 0], minlength=E)
            exp["halt_sup"] += np.bincount(d2_halts[qi_d2, 0], minlength=E); exp["presentations"]["d2"] += int(n_d2)
        Lb = max(p[0].shape[1] for p in parts)
        def padto(t, val):
            return t if t.shape[1] == Lb else torch.cat([t, torch.full((t.shape[0], Lb - t.shape[1]), val, dtype=t.dtype, device=t.device)], 1)
        inp = torch.cat([padto(p[0], v.pad) for p in parts]); tgt = torch.cat([padto(p[1], -100) for p in parts]); w = torch.cat([padto(p[2], 0.0) for p in parts])
        pos = torch.cat([padto(p[3], 0) for p in parts]); pm = torch.cat([padto(p[4], True) for p in parts]); kind = torch.cat([padto(p[5], 0) for p in parts])
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, u / max(1, args.warmup))
        logits = model(inp, pad_mask=pm, pos_ids=pos)
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none", ignore_index=-100)
        loss = (ce * w).sum() / args.rows_per_update
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            run_loss += float(loss); run_n += 1; tok_seen += int((~pm).sum()); calls_seen += sum(p[0].shape[0] for p in parts)
            for k, name in ((1, "step"), (2, "halt")):
                m = (kind == k) & (w > 0)
                if m.any():
                    run_ce[name] += float(ce[m].sum()); run_cnt[name] += int(m.sum())
        if u % 100 == 0 or u == 1:
            with open(log_path, "a") as f:
                f.write(json.dumps(dict(update=u, phase=phase, loss=run_loss / max(1, run_n), lr=opt.param_groups[0]["lr"], tokens_seen=tok_seen,
                                        units_seen=calls_seen, elapsed=time.time() - t0, **{f"ce/{k}": run_ce[k] / run_cnt[k] for k in run_ce if run_cnt[k]})) + "\n")
            run_ce.clear(); run_cnt.clear(); run_loss = 0.0; run_n = 0
        if u % args.eval_every == 0 or u == total or u == args.phase1_updates:
            model.eval(); te = time.time()
            rec = dict(update=u, phase=phase, elapsed=time.time() - t0, tokens_seen=tok_seen, presentations=dict(exp["presentations"]),
                       exposure=dict(step_min=int(exp["step"].min()), step_max=int(exp["step"].max()), halt_sup_min=int(exp["halt_sup"].min()),
                                     halt_sup_max=int(exp["halt_sup"].max()), halt_vis_min=int(exp["halt_vis"].min()), halt_vis_max=int(exp["halt_vis"].max())),
                       tf_train_w2=teacher_forced(model, Units([compile_w2(v, rel, q) for q in train_w2[:1000]], H, v.pad, device), device),
                       tf_train_d2=teacher_forced(model, Units([compile_d2(v, rel, q) for q in train_d2[:1000]], H, v.pad, device), device))
            rec["val_w2"] = selection_metric()
            vd = run_tasks(model, v, rel, val_d2_tasks, H, device, args.rollout_batch); rec["val_d2"] = summarize(sc for _, sc, _ in vd)
            ev = evaluate(monitor)
            rec["monitor"] = {k: s for k, s in ev["cells"].items() if "|repeated" not in k and "|revisits" not in k}
            rec["table"] = ev["table"]
            if "replay" in ev:
                rec["replay_model_model"] = {k: dict(final_acc=s["final_acc"], traj_acc=s["traj_acc"]) for k, s in ev["replay"]["model_model"].items()}
                rec["replay_model_correct"] = {k: s["traj_acc"] for k, s in ev["replay"]["model_correct"].items()}
                rec["replay_correct_model"] = {k: s["traj_acc"] for k, s in ev["replay"]["correct_model"].items()}
                rec["agreement_real_vs_table"] = ev["agreement_real_vs_table"]
            rec["eval_seconds"] = time.time() - te
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"u {u} ph{phase} | table {rec['table']['step']['acc']:.4f}/{rec['table']['halt']['acc']:.4f} | valw2 {rec['val_w2']['traj_acc']:.3f} | "
                  + " ".join(f"{k}:{s['traj_acc']:.2f}" for k, s in rec["monitor"].items() if k.startswith("one_new")) + f" | eval {rec['eval_seconds']:.0f}s | {time.time() - t0:.0f}s")
            score = (rec["val_w2"]["traj_acc"], -rec["val_w2"]["loss_per_question"])
            if score > (best[0], -best[1]):
                best = (score[0], -score[1]); torch.save(dict(model=model.state_dict(), update=u), os.path.join(args.save_dir, "best_by_valw2.pt"))
            if u % args.save_every == 0 or u == total:
                torch.save(dict(model=model.state_dict(), update=u), os.path.join(args.save_dir, f"ckpt_{u:06d}.pt"))
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), update=u, s_w2=s_w2.state(), s_d2=s_d2.state(), halt_rng=halt_rng.bit_generator.state,
                                torch_rng=torch.get_rng_state(), best=list(best), tokens_seen=tok_seen,
                                exp=dict(step=exp["step"].tolist(), halt_sup=exp["halt_sup"].tolist(), halt_vis=exp["halt_vis"].tolist(), presentations=exp["presentations"])), last_path)
    json.dump(dict(presentations=exp["presentations"], step_per_fact=exp["step"].tolist(), halt_supervised_per_entity=exp["halt_sup"].tolist(),
                   halt_visible_per_entity=exp["halt_vis"].tolist()), open(os.path.join(args.save_dir, "exposure.json"), "w"))
    if not args.no_final_eval:
        full_eval("final")
        extra = [] if args.final_extra_ckpts == "" else args.final_extra_ckpts.split(",")
        if args.final_extra_ckpts == "auto":
            extra = [f"ckpt_{total - k * args.save_every:06d}.pt" for k in (1, 2) if total - k * args.save_every > 0]
        for ck in extra:
            p = os.path.join(args.save_dir, ck)
            if os.path.exists(p):
                full_eval(ck.replace(".pt", ""), p)
        if os.path.exists(os.path.join(args.save_dir, "best_by_valw2.pt")):
            full_eval("best", os.path.join(args.save_dir, "best_by_valw2.pt"))
    print("done")


if __name__ == "__main__":
    main()
