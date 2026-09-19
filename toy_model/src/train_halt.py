"""
Sixth batch: externally set loop count (D) vs learned halting (H) on the paper-aligned full-sequence loop Transformer
(docs/experiments_learned_halting_2026-09-19.md, docs/experiments_paper_aligned_loop_2026-09-19.md; log docs/experiments_halting_log.md).

  python train_halt.py --data_dir ../data/chain_loop --atomic ../data/atomic_joint_2026-09-19/train_atomic.json \
      --save_dir ../runs/halt_H_s1 --arm H --seed 1 [--resume] [--cuda_graph 1]

Arms   D  the program sets R = d (number of relations in the input): atomic queries are read after 1 loop, d2 after 2, a depth-d
          test chain after d loops.                                                     loss = CE at loop d
       H  the model decides CONTINUE / STOP every loop with a stop head on the read-out state; trained with the expected final-answer
          loss under its own halting distribution, unrolled --b_train loops, remaining mass collected at the last unrolled loop
          (logged as truncation mass, never credited as an active STOP), plus --lam * E[T] (main arm: lam = 0).
          No halting labels, no intermediate entities, no loop index.   Inference: STOP at the first loop with p_t >= --stop_thr
          (pre-registered 0.5), global cap --b_eval loops for every depth; hitting the cap = timeout = failure.
       P  paper-style baseline (extra, not part of the D / H comparison): R ~ clip(Poisson(4), 2, 8) per batch, CE at loop R.
Data   one update = 128 single-chain queries = 32 atomic + 16 w2 sources rendered as 32 independent atomic queries + 64 d2; the
       mean of the 128 final-answer CEs. Inputs are compact [x, r] / [x, r1, r2] (no wrappers, no END); the label is never an input.
       Training depth <= 2; deeper chains exist only in evaluation.
Optim  AdamW lr 1e-4, wd 0.01, 2000 warm-up updates then constant, no label smoothing, global grad-norm clip 1.0, fp32.
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

from model.loop_gpt import LoopGPT
from train_chain import Stream, load_json

N_AT, N_W2SRC, N_D2 = 32, 16, 64
N_ROWS = N_AT + 2 * N_W2SRC + N_D2
GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def named_rng(seed, name):
    return np.random.default_rng([seed, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)])


# --------------------------------------------------------------------------------------------------------------- losses
def loss_fn(model, arm, tok, last, tgt, R=None, b_train=8, lam=0.0):
    """-> (loss, stats dict of 0-d / 1-d tensors). Row layout: [32 atomic | 32 w2-derived atomic | 64 d2]."""
    h = model.embed(tok); n1 = N_AT + 2 * N_W2SRC; st = {}
    if arm == "D":
        h = model.step(h); z1 = model.read(h, last)
        h2 = model.step(h[n1:]); z2 = model.read(h2, last[n1:])
        ce = F.cross_entropy(torch.cat([model.logits(z1[:n1]), model.logits(z2)]).float(), tgt, reduction="none")
        loss = ce.mean()
    elif arm == "P":
        for _ in range(R):
            h = model.step(h)
        ce = F.cross_entropy(model.logits(model.read(h, last)).float(), tgt, reduction="none"); loss = ce.mean()
    else:
        ces, ps = [], []
        for _ in range(b_train):
            h = model.step(h); z = model.read(h, last)
            ces.append(F.cross_entropy(model.logits(z).float(), tgt, reduction="none")); ps.append(model.stop_logit(z).float())
        ce_t = torch.stack(ces); sl = torch.stack(ps); p = torch.sigmoid(sl)           # (B_train, N)
        # survival prod_{j<=t}(1 - p_j) in log space: log(1 - sigmoid(s)) = logsigmoid(-s)  (cumprod's backward syncs with the host,
        # which a captured CUDA graph does not allow; the log form is also the numerically safer one)
        surv = torch.exp(torch.cumsum(F.logsigmoid(-sl), 0)); prev = torch.cat([torch.ones_like(surv[:1]), surv[:-1]])
        q = torch.cat([(p * prev)[:-1], prev[-1:]])                                    # last loop collects ALL remaining mass
        steps = torch.arange(1, b_train + 1, device=tok.device, dtype=q.dtype)[:, None]
        ET = (q * steps).sum(0); ce = (q * ce_t).sum(0); loss = (ce + lam * ET).mean()
        st.update(ce_by_loop_atomic=ce_t[:, :n1].mean(1), ce_by_loop_d2=ce_t[:, n1:].mean(1), q_atomic=q[:, :n1].mean(1), q_d2=q[:, n1:].mean(1),
                  tail_atomic=surv[-1, :n1].mean(), tail_d2=surv[-1, n1:].mean(), ET_atomic=ET[:n1].mean(), ET_d2=ET[n1:].mean())
    st.update(ce_atomic=ce[:N_AT].mean(), ce_w2=ce[N_AT:n1].mean(), ce_d2=ce[n1:].mean())
    return loss, st


# ----------------------------------------------------------------------------------------------------------- evaluation
class EvalSet:
    def __init__(self, items, e0, r0, device):
        """items: dict(x, rels, gold[, cat, d, id]); grouped by depth (equal length -> no padding inside a group)."""
        self.items = items; self.n = len(items); self.groups = []
        by_d = collections.defaultdict(list)
        for i, it in enumerate(items):
            by_d[len(it["rels"])].append(i)
        for d, idx in sorted(by_d.items()):
            tok = torch.tensor([[e0 + items[i]["x"]] + [r0 + r for r in items[i]["rels"]] for i in idx], device=device)
            self.groups.append((d, np.array(idx), tok, torch.tensor([e0 + items[i]["gold"] for i in idx], device=device)))

    def batches(self, tokens_per_batch=60000):
        for d, idx, tok, gold in self.groups:
            bs = max(32, tokens_per_batch // (d + 1))
            for s in range(0, len(idx), bs):
                yield d, idx[s:s + bs], tok[s:s + bs], gold[s:s + bs]


@torch.no_grad()
def eval_fixed(model, ES, budgets):
    """budgets: list of ints (the same R for every depth) and / or the string 'd' (R = depth). -> {budget: correct (N,) bool}"""
    out = {b: np.zeros(ES.n, dtype=bool) for b in budgets}
    for d, idx, tok, gold in ES.batches():
        want = sorted({(d if b == "d" else b) for b in budgets}); last = torch.full((len(idx),), d, device=tok.device)
        h = model.embed(tok); snap = {}
        for t in range(1, want[-1] + 1):
            h = model.step(h)
            if t in want:
                snap[t] = (model.logits(model.read(h, last)).argmax(-1) == gold).cpu().numpy()
        for b in budgets:
            out[b][idx] = snap[d if b == "d" else b]
    return out


@torch.no_grad()
def eval_halt(model, ES, b_eval, thr=0.5, sample_seed=None, rule="head"):
    """Per-sample stopping with an active set that shrinks. rule 'head': stop head (p >= thr, or Bernoulli(p) if sample_seed);
    rule 'kl': the paper's inference rule (KL(a_t || a_{t-1}) < 0.01 and entropy(a_t) < 3.0, at least 2 loops).
    -> correct (N,), T (N,) loop at which the sample stopped (b_eval if it never did), stopped (N,) bool."""
    correct = np.zeros(ES.n, dtype=bool); T = np.full(ES.n, b_eval, dtype=np.int32); stopped = np.zeros(ES.n, dtype=bool)
    gen = torch.Generator(device=ES.groups[0][2].device).manual_seed(sample_seed) if sample_seed is not None else None
    for d, idx, tok, gold in ES.batches():
        h = model.embed(tok); act = torch.arange(len(idx), device=tok.device); last = torch.full((len(idx),), d, device=tok.device); prev_lp = None
        for t in range(1, b_eval + 1):
            h = model.step(h); z = model.read(h, last[:len(act)]); lg = model.logits(z)
            if rule == "head":
                p = torch.sigmoid(model.stop_logit(z)); stop = (torch.rand(p.shape, device=p.device, generator=gen) < p) if gen is not None else (p >= thr)
            else:
                lp = F.log_softmax(lg.float(), -1); ent = -(lp.exp() * lp).sum(-1)
                stop = torch.zeros(len(act), dtype=torch.bool, device=tok.device) if prev_lp is None else (((lp.exp() * (lp - prev_lp)).sum(-1) < 0.01) & (ent < 3.0))
            if t == b_eval:
                ok = (lg.argmax(-1) == gold[act]).cpu().numpy(); correct[idx[act.cpu().numpy()]] = ok        # recorded, but counted as timeout
                break
            if bool(stop.any()):
                a = act[stop].cpu().numpy(); correct[idx[a]] = (lg[stop].argmax(-1) == gold[act[stop]]).cpu().numpy(); T[idx[a]] = t; stopped[idx[a]] = True
                keep = ~stop; act = act[keep]; h = h[keep]
                if rule == "kl":
                    lp = lp[keep]
                if len(act) == 0:
                    break
            if rule == "kl":
                prev_lp = lp
    return correct, T, stopped


def by_cell(items, **arrays):
    agg = collections.defaultdict(list)
    for i, it in enumerate(items):
        agg[f"{it['cat']}|d{it['d']}"].append(i)
    out = {}
    for k, ix in sorted(agg.items()):
        ix = np.array(ix); rec = dict(n=len(ix))
        for name, arr in arrays.items():
            rec[name] = float(np.mean(arr[ix]))
        if "T" in arrays:
            rec["T_p10_p50_p90"] = [float(x) for x in np.percentile(arrays["T"][ix], [10, 50, 90])]
        out[k] = rec
    return out


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--atomic", required=True); ap.add_argument("--save_dir", required=True)
    ap.add_argument("--arm", choices=["D", "H", "P"], required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=768); ap.add_argument("--n_head", type=int, default=12); ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--pos", choices=["nope", "rope"], default="nope", help="nope = paper-aligned main setting; rope = relative positions (RoPE base 100), a labelled variant")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--weight_decay", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--updates", type=int, default=1000000)
    ap.add_argument("--b_train", type=int, default=8, help="H: loops unrolled in training (not tied to the depth of the query)")
    ap.add_argument("--b_eval", type=int, default=256, help="H: global loop cap at inference, the same for every depth; reaching it is a timeout")
    ap.add_argument("--lam", type=float, default=0.0, help="H: compute cost weight on E[T]"); ap.add_argument("--stop_thr", type=float, default=0.5)
    ap.add_argument("--stop_bias", type=float, default=0.0, help="initial bias of the stop head")
    ap.add_argument("--eval_every", type=int, default=10000); ap.add_argument("--monitor_per_cell", type=int, default=100)
    ap.add_argument("--cuda_graph", type=int, default=0); ap.add_argument("--resume", action="store_true"); ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--train_precision", choices=["fp32", "tf32", "bf16"], default="fp32",
                    help="matmul precision of the TRAINING step only (tf32 = TensorFloat-32 matmuls, bf16 = autocast); every evaluation runs in strict fp32")
    ap.add_argument("--stop_after", type=int, default=0); ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    arm = args.arm; device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = load_json(args.data_dir, "meta"); rel = meta["rel_map"]; vocab = load_json(args.data_dir, "vocab"); t2i = {t: i for i, t in enumerate(vocab)}
    e0, r0 = t2i["<e_0>"], t2i["<r_0>"]; PAD = t2i["<pad>"]; E = meta["num_entities"]
    atomic = json.load(open(args.atomic)); train_w2 = load_json(args.data_dir, "train_w2"); train_d2 = load_json(args.data_dir, "train_d2")
    assert all(a["y"] == rel[a["r"]][a["x"]] for a in atomic) and len(atomic) == E * meta["num_relations"]
    val_d2 = load_json(args.data_dir, "val_d2"); tests = load_json(args.data_dir, "tests")
    # source tables on the device: [x, r, s or PAD, last index, gold]   (w2 sources become two independent atomic queries)
    T_at = torch.tensor([[e0 + a["x"], r0 + a["r"], PAD, 1, e0 + a["y"]] for a in atomic], device=device)
    T_w2 = torch.tensor([[[e0 + x, r0 + r, PAD, 1, e0 + rel[r][x]] for (x, r) in (q["q1"], q["q2"])] for q in train_w2], device=device)
    T_d2 = torch.tensor([[e0 + q["x"], r0 + q["r"], r0 + q["s"], 2, e0 + rel[q["s"]][rel[q["r"]][q["x"]]]] for q in train_d2], device=device)

    model = LoopGPT(len(vocab), args.d_model, args.n_head, args.n_layer, pos=args.pos); model.seeded_init(args.seed)
    with torch.no_grad():
        model.stop.bias.fill_(args.stop_bias)
    model.to(device); assert model.wte.weight.data_ptr() == model.wte.weight.data_ptr()
    hsh = lambda names: hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for n, p in sorted(model.named_parameters()) if n in names)).hexdigest()[:16]
    n_params = sum(p.numel() for p in model.parameters()); cap = args.smoke or None

    for t in tests:
        t["gold"] = t["states"][-1]
    per_cell = collections.defaultdict(list)
    for t in tests:
        per_cell[(t["cat"], t["d"])].append(t)
    first = lambda n: [t for k in per_cell for t in per_cell[k][:min(n, cap or n)]]
    mon_items = first(args.monitor_per_cell); MON = EvalSet(mon_items, e0, r0, device)
    at_items = [dict(x=a["x"], rels=[a["r"]], gold=a["y"], cat="atomic", d=1) for a in atomic][:cap]; AT = EvalSet(at_items, e0, r0, device)
    mk2 = lambda rows, cat: [dict(x=q["x"], rels=[q["r"], q["s"]], gold=rel[q["s"]][rel[q["r"]][q["x"]]], cat=cat, d=2) for q in rows][:cap]
    VD2 = EvalSet(mk2(val_d2, "val_d2"), e0, r0, device); FIT = EvalSet(mk2(train_d2[::10], "train_d2"), e0, r0, device)
    MAIN_FIXED = 8                                                   # arm P: pre-registered common budget (its largest training R)

    def main_protocol(ES):
        """the arm's own pre-registered inference protocol -> dict(correct[, T, stopped])"""
        if arm == "D":
            return dict(correct=eval_fixed(model, ES, ["d"])["d"])
        if arm == "P":
            return dict(correct=eval_fixed(model, ES, [MAIN_FIXED])[MAIN_FIXED])
        c, T, s = eval_halt(model, ES, args.b_eval, args.stop_thr)
        return dict(correct=c & s, T=T.astype(np.float64), stopped=s, timeout=~s, wrong_stop=s & ~c, correct_at_stop_or_cap=c)

    def shallow():
        out = {}
        for name, ES in (("atomic", AT), ("val_d2", VD2), ("train_d2_fit", FIT)):
            r = main_protocol(ES); out[name] = {k: float(np.mean(v)) for k, v in r.items()}
        return out

    def final_eval():
        path = os.path.join(args.save_dir, "final_eval.json")
        if os.path.exists(path):
            print("final_eval.json exists, skipping"); return
        model.eval(); t0 = time.time(); res = dict(arm=arm, seed=args.seed, lam=args.lam, b_eval=args.b_eval, stop_thr=args.stop_thr)
        items = first(10 ** 9) if cap else tests; FULL = EvalSet(items, e0, r0, device); raw = dict(ids=np.array([t["id"] for t in items]))
        r = main_protocol(FULL); res["main"] = by_cell(items, **r); raw.update({f"main_{k}": v for k, v in r.items()})
        res["shallow"] = shallow()
        # diagnostics on the SAME weights: R = d for every arm, the common fixed-budget grid on the first 500 per cell, the paper's KL / entropy stop rule
        rd = eval_fixed(model, FULL, ["d"])["d"]; res["forced_R_eq_d"] = by_cell(items, correct=rd); raw["forced_R_eq_d"] = rd
        g_items = first(500); G = EvalSet(g_items, e0, r0, device); rg = eval_fixed(model, G, GRID)
        res["fixed_grid_first500"] = {f"R{b}": by_cell(g_items, correct=rg[b]) for b in GRID}
        c, T, s = eval_halt(model, G, args.b_eval, rule="kl"); res["kl_entropy_rule_first500"] = by_cell(g_items, correct=c & s, T=T.astype(np.float64), stopped=s)
        if arm == "H":
            c, T, s = eval_halt(model, G, args.b_eval, sample_seed=20260919); res["sampled_stop_first500"] = by_cell(g_items, correct=c & s, T=T.astype(np.float64), stopped=s)
            for name, ES in (("atomic", AT), ("val_d2", VD2)):
                rr = eval_fixed(model, ES, ["d"] + GRID[:6]); res.setdefault("shallow_fixed", {})[name] = {str(b): float(v.mean()) for b, v in rr.items()}
        res["eval_seconds"] = time.time() - t0
        np.savez_compressed(os.path.join(args.save_dir, "final_answers.npz"), **raw); json.dump(res, open(path, "w"), indent=1)
        m = res["main"]; print("[final main] " + " ".join(f"{k}:{m[k]['correct']:.3f}" for k in m if not k.startswith("all_seen")), flush=True)

    capturable = device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=(torch.tensor(args.lr, device=device) if capturable else args.lr), betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=args.weight_decay, capturable=capturable)
    s_at = Stream(len(atomic), args.seed, "atomic"); s_w2 = Stream(len(train_w2), args.seed, "w2"); s_d2 = Stream(len(train_d2), args.seed, "d2")
    r_rng = named_rng(args.seed, "halt_R")
    ctr = dict(atomic=0, w2_sources=0, w2_queries=0, d2=0, row_loops=0, R_hist={})
    start = 1; best = None
    P = lambda f: os.path.join(args.save_dir, f); last_path = P("last.pt")
    resume_state = None
    if (args.resume or args.eval_only) and os.path.exists(last_path):
        resume_state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"]); s_at.load(resume_state["s_at"]); s_w2.load(resume_state["s_w2"]); s_d2.load(resume_state["s_d2"])
        r_rng.bit_generator.state = resume_state["r_rng"]; ctr = resume_state["ctr"]; start = resume_state["update"] + 1; best = resume_state["best"]
        print(f"resumed from update {resume_state['update']}")
    elif not args.eval_only:
        for f in ("metrics.jsonl", "train_log.jsonl"):
            open(P(f), "w").close()
        src = hashlib.sha256(b"".join(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), "rb").read() for f in ("train_halt.py", "model/loop_gpt.py"))).hexdigest()[:16]
        json.dump(dict(arm=arm, seed=args.seed, n_params=n_params, stop_head_params=model.stop.weight.numel() + 1, backbone_init_hash=hsh(set(model.backbone_names())),
                       model=dict(d_model=args.d_model, n_head=args.n_head, n_layer=args.n_layer, position=args.pos, tied_lm_head=True, residual_out_proj_init=0.0, dropout=0.0, ln_eps=1e-5, act="gelu_tanh", readout="last valid input position"),
                       batch=dict(queries=N_ROWS, atomic=N_AT, w2_sources=N_W2SRC, w2_rendering="two independent atomic queries per source", d2=N_D2, loss="mean of 128 final-answer CEs"),
                       optimizer=dict(name="AdamW", lr=args.lr, weight_decay=args.weight_decay, warmup=args.warmup, schedule="linear warm-up then constant", clip=args.clip, label_smoothing=0.0, precision="fp32"),
                       updates=args.updates, halting=dict(b_train=args.b_train, b_eval=args.b_eval, lam=args.lam, stop_thr=args.stop_thr, stop_bias=args.stop_bias, tail="all remaining mass at the last unrolled loop (truncation mass)") if arm == "H" else None,
                       P_budget="R ~ clip(Poisson(4), 2, 8) per batch; main test budget R = 8" if arm == "P" else None,
                       selection="final checkpoint is primary; best.pt = highest val_d2 accuracy under the arm's own protocol (earliest on ties); deep tests never used",
                       data_hashes=meta.get("hashes"), atomic_sha256=hashlib.sha256(open(args.atomic, "rb").read()).hexdigest(), source_hash=src, cuda_graph=bool(args.cuda_graph)),
                  open(P("manifest.json"), "w"), indent=1)
    print(f"arm {arm} seed {args.seed} | params {n_params:,} | device {device} | cuda_graph {args.cuda_graph}", flush=True)
    if args.eval_only:
        final_eval(); return

    params = [p for p in model.parameters()]
    tok = torch.zeros((N_ROWS, 3), dtype=torch.long, device=device); last = torch.zeros(N_ROWS, dtype=torch.long, device=device); tgt = torch.zeros(N_ROWS, dtype=torch.long, device=device)

    def fwd_bwd(R):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.train_precision == "bf16" and device.type == "cuda")):
            loss, st = loss_fn(model, arm, tok, last, tgt, R=R, b_train=args.b_train, lam=args.lam)
        loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, args.clip, foreach=True); opt.step()
        return loss, st, gn

    def set_tf32(on):
        torch.backends.cuda.matmul.allow_tf32 = bool(on); torch.backends.cudnn.allow_tf32 = bool(on)

    def train_step(R):
        opt.zero_grad(set_to_none=True)
        return fwd_bwd(R)

    graphs = {}
    R_values = list(range(2, 9)) if arm == "P" else [None]
    set_tf32(args.train_precision == "tf32")
    if args.cuda_graph and device.type == "cuda":
        snap = {k: v.clone() for k, v in model.state_dict().items()}
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for R in R_values:
                for _ in range(3):
                    train_step(R)
        torch.cuda.current_stream().wait_stream(side)
        for R in R_values:
            g = torch.cuda.CUDAGraph(); opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(g):
                loss, st, gn = fwd_bwd(R)
            graphs[R] = (g, loss, st, gn)
        model.load_state_dict(snap)                                  # in place: undo the warm-up / capture updates exactly
        for p_, s_ in opt.state.items():
            for v_ in s_.values():
                v_.zero_()
    if resume_state is not None:                                     # optimizer state restored IN PLACE (graph-safe); eager path identical
        if not opt.state:
            train_step(R_values[0]); model.load_state_dict(resume_state["model"])
        saved = resume_state["opt"]["state"]
        for i, p_ in enumerate(params):
            if i in saved and p_ in opt.state:
                for k_, v_ in saved[i].items():
                    opt.state[p_][k_].copy_(v_)
    del resume_state

    t0 = time.time(); log = open(P("train_log.jsonl"), "a")
    for u in range(start, args.updates + 1):
        model.train()
        ia = torch.from_numpy(s_at.take(N_AT)).to(device); iw = torch.from_numpy(s_w2.take(N_W2SRC)).to(device); idd = torch.from_numpy(s_d2.take(N_D2)).to(device)
        rows = torch.cat([T_at[ia], T_w2[iw].reshape(-1, 5), T_d2[idd]]); tok.copy_(rows[:, :3]); last.copy_(rows[:, 3]); tgt.copy_(rows[:, 4])
        R = int(np.clip(r_rng.poisson(4), 2, 8))                     # the stream advances in every arm; only P uses it
        Ru = R if arm == "P" else None
        lr_now = args.lr * min(1.0, u / max(1, args.warmup))
        if capturable:
            opt.param_groups[0]["lr"].fill_(lr_now)
        else:
            opt.param_groups[0]["lr"] = lr_now
        set_tf32(args.train_precision == "tf32")
        if graphs:
            g, loss, st, gn = graphs[Ru]; g.replay()
        else:
            loss, st, gn = train_step(Ru)
        set_tf32(False)                                            # everything outside the training step (all evaluations) is strict fp32
        ctr["atomic"] += N_AT; ctr["w2_sources"] += N_W2SRC; ctr["w2_queries"] += 2 * N_W2SRC; ctr["d2"] += N_D2
        ctr["row_loops"] += {"D": N_ROWS + N_D2, "H": N_ROWS * args.b_train, "P": N_ROWS * R}[arm]
        if arm == "P":
            ctr["R_hist"][str(R)] = ctr["R_hist"].get(str(R), 0) + 1
        if u % 100 == 0 or u == 1:
            rec = dict(update=u, lr=lr_now, elapsed=time.time() - t0, loss=float(loss), grad_norm=float(gn), **{k: (v.tolist() if v.dim() else float(v)) for k, v in st.items()})
            log.write(json.dumps(rec) + "\n"); log.flush()
        ev = (u % args.eval_every == 0) or u == args.updates or (u in (1000, 2000, 5000)) or (args.smoke and u % 20 == 0)
        if ev:
            model.eval(); te = time.time(); rec = dict(update=u, elapsed=time.time() - t0, shallow=shallow())
            r = main_protocol(MON); rec["monitor"] = by_cell(mon_items, **r)
            if arm == "H":
                rec["monitor_forced_R_eq_d"] = {k: v["correct"] for k, v in by_cell(mon_items, correct=eval_fixed(model, MON, ["d"])["d"]).items()}
            rec["counters"] = dict(ctr); rec["eval_seconds"] = time.time() - te; rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
            with open(P("metrics.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            sh = rec["shallow"]; m = rec["monitor"]
            print(f"u {u} | atomic {sh['atomic']['correct']:.3f} val_d2 {sh['val_d2']['correct']:.3f} fit {sh['train_d2_fit']['correct']:.3f} | " + " ".join(f"{k.split('|')[1]}:{m[k]['correct']:.2f}" for k in m if k.startswith("one_new"))
                  + (f" | T(d2/d8/d128) {m['one_new|d2']['T']:.1f}/{m['one_new|d8']['T']:.1f}/{m['one_new|d128']['T']:.1f} timeout(d8) {m['one_new|d8']['timeout']:.2f}" if arm == "H" and "one_new|d128" in m else "")
                  + f" | eval {rec['eval_seconds']:.0f}s | {time.time() - t0:.0f}s", flush=True)
            score = sh["val_d2"]["correct"]
            if best is None or score > best[0]:
                best = [score, u]; torch.save(dict(model=model.state_dict(), update=u), P("best.pt"))
        if u % args.eval_every == 0 or u == args.updates or u == args.stop_after:
            if u % 100000 == 0 or u == args.updates:
                torch.save(dict(model=model.state_dict(), update=u), P(f"ckpt_{u:07d}.pt"))
            osd = opt.state_dict(); osd = dict(state={i: {k: v.clone() for k, v in s.items()} for i, s in osd["state"].items()})
            torch.save(dict(model=model.state_dict(), opt=osd, update=u, s_at=s_at.state(), s_w2=s_w2.state(), s_d2=s_d2.state(), r_rng=r_rng.bit_generator.state, ctr=ctr, best=best), last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)
        if u == args.stop_after:
            print(f"stopped at update {u} ({(time.time() - t0) / max(1, u - start + 1) * 1000:.1f} ms / update incl. evals)", flush=True); return
    final_eval(); print("done", flush=True)


if __name__ == "__main__":
    main()
