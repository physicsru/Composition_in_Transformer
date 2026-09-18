"""
Fifth batch: autonomous latent execution from w2 / d2 only (docs/experiments_latent_batch2_20runs.md).

  python train_latent.py --data_dir ../data/chain_loop --save_dir ../runs/latent2_A_s1 --cond A --seed 1 [--resume]

Conditions   A latent_ntp (answer NTP only) | B + entity aux | C + read aux | D + both | E single pass (T = 1, A's parameters)
             F untied depth (8 separate cores, loop t uses core t).   Aux losses: phase 2, d2 rows only, weight 0.5 each,
             averaged over the T loops of the row; in phase 1 the aux graph is NOT built (A-D identical until 50k).
Schedule     250k updates: 50k x 256 w2, then 200k x (128 w2 + 128 d2); AdamW 3e-4 / wd 0.1 / warmup 100, global grad-norm clip 1.0,
             T ~ U{2,4,8} per update (E: 1), full backprop through all loops. Independent RNG streams: sources (w2, d2),
             layouts, T; evaluation never touches them.
Loss         L_ans = mean CE of the answer entity tokens + CE(END_ANSWER) per source row, mean over the 256 rows.
Monitoring   shallow validation + the 64-per-cell small monitor (depth 2/3/4/8/32, one_new/random) every 200 updates in 1k-5k and
             50k-55k, otherwise every 5k; the full depth x T grid on 64 per cell every 25k; raw monitor answers are kept.
Checkpoints  last.pt (full resume state) every 5k; model-only ckpt_UUUUUU.pt every 25k and at 240k / 245k / 250k; best.pt =
             secondary selection AFTER phase 1 by 0.5 val_w2 strict row acc + 0.5 val_d2 strict acc (ties: shallow loss, earlier).
Final        (idempotent, also --eval_only) full 120k at the pre-registered budgets (T=8; shared: also T=256; E: T=1), the budget
             matrix on the first 500 per cell with right/wrong transitions, tail checkpoints and best on 500 per cell, compact /
             multi-layout controls, suffix-change pairs, atomic / validation / train-fit, read-head and entity-probe diagnostics.
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

from data.latent_format import (ADDR_EOP, MAX_ANS, N_SLOTS, LatentVocab, TrainTables, addr_to_key, frozen_slots, render_chain)
from model.latent_executor import LatentExecutor
from train_chain import Stream, load_json

COND = dict(A=dict(name="latent_ntp", mode="shared", state=0, read=0), B=dict(name="latent_state", mode="shared", state=1, read=0),
            C=dict(name="latent_read", mode="shared", state=0, read=1), D=dict(name="latent_both", mode="shared", state=1, read=1),
            E=dict(name="single_pass", mode="single", state=0, read=0), F=dict(name="untied_depth", mode="untied", state=0, read=0))
GRID = dict(shared=[1, 2, 4, 8, 16, 32, 64, 128, 256], single=[1], untied=[1, 2, 4, 8])
MAIN_T = dict(shared=[8, 256], single=[1], untied=[8]); MON_T = dict(shared=[8, 64], single=[1], untied=[8])
DEPTHS = (2, 3, 4, 8, 16, 32, 64, 128); CATS = ("all_seen", "one_new", "random")


def named_rng(seed, name):
    return np.random.default_rng([seed, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)])


# ------------------------------------------------------------------------------------------------------------ evaluation
class EvalSet:
    """Single-region chains with frozen layouts, grouped by depth as device tensors."""

    def __init__(self, v, items, slots, device):
        self.items = items; self.n = len(items); self.groups = []
        by_d = collections.defaultdict(list)
        for i, it in enumerate(items):
            by_d[len(it["rels"])].append(i)
        for d, idx in sorted(by_d.items()):
            rows = [render_chain(v, items[i]["x"], items[i]["rels"], slots[i]) for i in idx]
            self.groups.append((np.array(idx), torch.tensor([r[0] for r in rows], device=device), torch.tensor([r[1] for r in rows], device=device),
                                torch.tensor([v.e0 + items[i]["gold"] for i in idx], device=device)))

    @torch.no_grad()
    def run(self, model, v, T_list, batch=2000):
        """-> {T: (status (N,) int8: 0 correct / 1 wrong entity / 2 format, raw (N, MAX_ANS) int16)}; all budgets from ONE pass."""
        out = {T: (np.full(self.n, 2, dtype=np.int8), np.full((self.n, MAX_ANS), -1, dtype=np.int16)) for T in T_list}
        for idx, tok, pos, gold in self.groups:
            for s in range(0, len(idx), batch):
                sl = slice(s, s + batch); pad = torch.zeros_like(tok[sl], dtype=torch.bool)
                M, rope_m = model.encode(tok[sl], pos[sl], pad)
                _, _, _, snaps = model.think(M, rope_m, pad, max(T_list), decode_at=list(T_list))
                for T in T_list:
                    gen = model.generate(snaps[T], v.ANSWER, v.END_ANSWER, MAX_ANS)
                    fmt = (gen[:, 0] >= v.e0) & (gen[:, 0] < v.e0 + v.E) & (gen[:, 1] == v.END_ANSWER)
                    st = torch.where(fmt & (gen[:, 0] == gold[sl]), 0, torch.where(fmt, 1, 2))
                    out[T][0][idx[sl]] = st.cpu().numpy().astype(np.int8); out[T][1][idx[sl]] = gen.cpu().numpy().astype(np.int16)
        return out


def cell_stats(items, status, keyf):
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for it, s in zip(items, status):
        agg[keyf(it)][int(s)] += 1
    return {k: dict(n=sum(c), acc=c[0] / sum(c), wrong_entity=c[1] / sum(c), format=c[2] / sum(c)) for k, c in sorted(agg.items())}


def cell_key(it): return f"{it['cat']}|d{it['d']}"


@torch.no_grad()
def eval_w2(model, v, rel, val_w2, T, device):
    toks, poss, gold = [], [], []
    for i, q in enumerate(val_w2):
        (x, r), (z, s) = q["q1"], q["q2"]; sl = [int(frozen_slots(6, i, 1)[0]), int(frozen_slots(6, i, 1, 1)[0])]
        toks.append([v.TASK, v.e0 + x, v.r0 + r, v.EOP, v.TASK, v.e0 + z, v.r0 + s, v.EOP]); poss.append([0, 1, 2 + sl[0], 130, 131, 132, 133 + sl[1], 261])
        gold.append([v.e0 + rel[r][x], v.e0 + rel[s][z]])
    tok = torch.tensor(toks, device=device); pos = torch.tensor(poss, device=device); pad = torch.zeros_like(tok, dtype=torch.bool); g = torch.tensor(gold, device=device)
    M, rm = model.encode(tok, pos, pad); z = model.think(M, rm, pad, T)[0]
    gen = model.generate(z, v.ANSWER, v.END_ANSWER, MAX_ANS)
    ise = lambda c: (gen[:, c] >= v.e0) & (gen[:, c] < v.e0 + v.E)
    fmt = ise(0) & ise(1) & (gen[:, 2] == v.END_ANSWER); q1 = fmt & (gen[:, 0] == g[:, 0]); q2 = fmt & (gen[:, 1] == g[:, 1])
    ans_in = torch.cat([torch.full((len(g), 1), v.ANSWER, device=device), g], 1)
    tgt = torch.cat([g, torch.full((len(g), 1), v.END_ANSWER, device=device)], 1)
    ce = F.cross_entropy(model.decode(z, ans_in).transpose(1, 2), tgt, reduction="none")
    loss = float((ce * torch.tensor([0.5, 0.5, 1.0], device=device)).sum(1).mean())
    return dict(n=len(g), row_acc=float((q1 & q2).float().mean()), q1_acc=float(q1.float().mean()), q2_acc=float(q2.float().mean()), format=float((~fmt).float().mean()), L_ans=loss)


@torch.no_grad()
def d2_diag(model, v, rel, rows, T, device, kind_variant):
    """teacher-forced L_ans + read-head / entity-probe diagnostics on d2 questions (frozen layouts). NOT execution ground truth."""
    toks, poss, y, x1, addr = [], [], [], [], []
    for i, q in enumerate(rows):
        sl = frozen_slots(2, i, 2, kind_variant); b = rel[q["r"]][q["x"]]
        t, p = render_chain(v, q["x"], [q["r"], q["s"]], sl); toks.append(t); poss.append(p); y.append(rel[q["s"]][b]); x1.append(b)
        addr.append([2 + int(sl[0]), 2 + int(sl[1]), ADDR_EOP])
    tok = torch.tensor(toks, device=device); pos = torch.tensor(poss, device=device); pad = torch.zeros_like(tok, dtype=torch.bool)
    y = torch.tensor(y, device=device); x1 = torch.tensor(x1, device=device); key = addr_to_key(pos, pad, torch.tensor(addr, device=device))
    ans_in = torch.stack([torch.full_like(y, v.ANSWER), v.e0 + y], 1); tgt = torch.stack([v.e0 + y, torch.full_like(y, v.END_ANSWER)], 1)
    out = model(tok, pos, pad, T, ans_in=ans_in, collect=True)
    loss = float(F.cross_entropy(out["logits"].transpose(1, 2), tgt, reduction="none").sum(1).mean())
    per_loop = []
    for t in range(T):
        k = key[:, min(t, 2)]; p = out["reads"][t]
        ent = model.ent_head(out["states"][t]).argmax(-1); tg = x1 if t == 0 else y
        per_loop.append(dict(loop=t + 1, read_hit=float((p.argmax(-1) == k).float().mean()), read_prob=float(p.gather(1, k[:, None]).mean()), probe_entity_acc=float((ent == tg).float().mean())))
    return dict(n=len(rows), T=T, L_ans=loss, per_loop=per_loop)


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--save_dir", required=True)
    ap.add_argument("--cond", choices=list(COND), required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight_decay", type=float, default=0.1); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--aux_weight", type=float, default=0.5)
    ap.add_argument("--rows_per_update", type=int, default=256)
    ap.add_argument("--phase1_updates", type=int, default=50000); ap.add_argument("--phase2_updates", type=int, default=200000)
    ap.add_argument("--train_T", default="2,4,8")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--eval_only", action="store_true"); ap.add_argument("--no_final_eval", action="store_true")
    ap.add_argument("--stop_after", type=int, default=0, help="stop (after saving last.pt) at this update; throughput tests / walltime slicing")
    ap.add_argument("--smoke", type=int, default=0, help="smoke test: evaluation cells limited to this many chains, eval every 20 updates")
    args = ap.parse_args()
    cfg = COND[args.cond]; mode = cfg["mode"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = load_json(args.data_dir, "meta"); rel = meta["rel_map"]
    v = LatentVocab(load_json(args.data_dir, "vocab"))
    train_w2 = load_json(args.data_dir, "train_w2"); train_d2 = load_json(args.data_dir, "train_d2")
    val_d2 = load_json(args.data_dir, "val_d2"); val_w2 = load_json(args.data_dir, "val_w2")
    tests = load_json(args.data_dir, "tests"); ev_manifest = load_json(args.data_dir, "latent_eval_manifest")
    for t in tests:
        t["gold"] = t["states"][-1]
    tables = TrainTables(v, rel, train_w2, train_d2, device)          # the ONLY source of training batches: w2 / d2 rows
    model = LatentExecutor(len(v.vocab), v.E, d=args.d_model, n_head=args.n_head, mode=mode); model.seeded_init(args.seed); model.to(device)
    counts = model.param_groups_count()
    hsh = lambda names: hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for n, p in sorted(model.named_parameters()) if names(n))).hexdigest()[:16]
    train_T = [int(x) for x in args.train_T.split(",")]; total = args.phase1_updates + args.phase2_updates
    T_shallow = 1 if mode == "single" else 8; cap = args.smoke or None

    # frozen evaluation sets (first 500 of every cell = monitoring / diagnostics; nothing here feeds training or selection)
    per_cell = collections.defaultdict(list)
    for t in tests:
        per_cell[cell_key(t)].append(t)
    first = lambda n: [t for k in per_cell for t in per_cell[k][:min(n, cap or n)]]
    tslots = lambda its: [frozen_slots(1, t["id"], t["d"]) for t in its]
    mon_items = [t for t in first(64) if t["d"] in (2, 3, 4, 8, 32) and t["cat"] != "all_seen"]; grid_items = first(64)
    MON = EvalSet(v, mon_items, tslots(mon_items), device); GRID64 = EvalSet(v, grid_items, tslots(grid_items), device)
    vd_items = [dict(x=q["x"], rels=[q["r"], q["s"]], gold=rel[q["s"]][rel[q["r"]][q["x"]]], cat="val", d=2) for q in val_d2][:cap]
    VD2 = EvalSet(v, vd_items, [frozen_slots(2, i, 2) for i in range(len(vd_items))], device)

    def shallow(T):
        st = VD2.run(model, v, [T])[T][0]
        return dict(T=T, val_d2=dict(n=len(st), acc=float((st == 0).mean()), format=float((st == 2).mean())), val_w2=eval_w2(model, v, rel, val_w2, T, device),
                    val_d2_L_ans=d2_diag(model, v, rel, val_d2[:min(200, cap or 200)], T, device, 0)["L_ans"])

    def final_eval():
        path = os.path.join(args.save_dir, "final_eval.json")
        if os.path.exists(path):
            print("final_eval.json exists, skipping"); return
        model.eval(); t0 = time.time(); res = dict(cond=args.cond, seed=args.seed, mode=mode, main_T=MAIN_T[mode], grid_T=GRID[mode])
        sd = lambda f: torch.load(os.path.join(args.save_dir, f), map_location=device, weights_only=False)["model"]
        final_sd = {k: p.clone() for k, p in model.state_dict().items()}
        items = [t for k in per_cell for t in per_cell[k][:cap]] if cap else tests
        FULL = EvalSet(v, items, tslots(items), device)
        is_mon = np.array([per_cell[cell_key(t)].index(t) < 500 for t in items]) if cap else np.array([(t["id"] - per_cell[cell_key(t)][0]["id"]) < 500 for t in items])
        r = FULL.run(model, v, MAIN_T[mode]); raw = {}
        res["main"] = {}
        for T, (st, gen) in r.items():
            sub = lambda m: cell_stats([it for it, k in zip(items, m) if k], st[m], cell_key)
            res["main"][f"T{T}"] = dict(all=cell_stats(items, st, cell_key), monitor500=sub(is_mon), confirm4500=sub(~is_mon),
                                        repeated_relation=cell_stats([it for it in items if it["repeated_relation"]], st[[it["repeated_relation"] for it in items]], cell_key),
                                        no_entity_revisit=cell_stats([it for it in items if not it["revisits_entity"]], st[[not it["revisits_entity"] for it in items]], cell_key),
                                        one_new_by_position=cell_stats([it for it in items if it["cat"] == "one_new" and it["d"] > 2],
                                                                       st[[it["cat"] == "one_new" and it["d"] > 2 for it in items]],
                                                                       lambda it: f"d{it['d']}|q{min(3, int(4 * it['new_adjacencies'][0] / (it['d'] - 1)))}"))
            raw[f"main_T{T}_status"] = st; raw[f"main_T{T}_gen"] = gen
        raw["main_ids"] = np.array([t["id"] for t in items])
        # budget matrix on the first 500 per cell (never used to pick T or a checkpoint) + right/wrong transitions along T
        m_items = first(500); MAT = EvalSet(v, m_items, tslots(m_items), device); rm = MAT.run(model, v, GRID[mode])
        res["matrix"] = {f"T{T}": cell_stats(m_items, rm[T][0], cell_key) for T in GRID[mode]}
        raw["matrix_ids"] = np.array([t["id"] for t in m_items]); raw["matrix_status"] = np.stack([rm[T][0] for T in GRID[mode]])
        chain_T = [T for T in GRID[mode] if T >= 8]
        res["transitions"] = {}
        cell_ix = collections.defaultdict(list)
        for i, it in enumerate(m_items):
            cell_ix[cell_key(it)].append(i)
        for a, b in zip(chain_T, chain_T[1:]):
            res["transitions"][f"T{a}->T{b}"] = {k: dict(wrong_to_right=float(((rm[a][0][ix] != 0) & (rm[b][0][ix] == 0)).mean()), right_to_wrong=float(((rm[a][0][ix] == 0) & (rm[b][0][ix] != 0)).mean()))
                                                 for k, ix in ((k, np.array(ix)) for k, ix in cell_ix.items())}
        # layout controls: compact layout on the same 500, and 128 chains per cell under 4 layouts (answer consistency)
        CMP = EvalSet(v, m_items, [np.arange(t["d"]) for t in m_items], device); rc = CMP.run(model, v, MAIN_T[mode])
        res["compact_layout"] = {f"T{T}": cell_stats(m_items, rc[T][0], cell_key) for T in MAIN_T[mode]}
        l_items = first(128); lay = [tslots(l_items), [np.arange(t["d"]) for t in l_items]] + [[frozen_slots(5, t["id"], t["d"], k) for t in l_items] for k in (1, 2)]
        gens = [EvalSet(v, l_items, s, device).run(model, v, MAIN_T[mode]) for s in lay]
        res["layout_consistency"] = {}
        for T in MAIN_T[mode]:
            same = np.all([np.all(g[T][1] == gens[0][T][1], 1) for g in gens], 0); allc = np.all([g[T][0] == 0 for g in gens], 0)
            agg = collections.defaultdict(list)
            for it, s_, c_ in zip(l_items, same, allc):
                agg[cell_key(it)].append((s_, c_))
            res["layout_consistency"][f"T{T}"] = {k: dict(n=len(x), same_answer_all4=float(np.mean([a for a, _ in x])), correct_all4=float(np.mean([b for _, b in x]))) for k, x in sorted(agg.items())}
        # suffix-change pairs: does the answer depend on the late part of the program?
        prs = ev_manifest["suffix_pairs"] if not cap else ev_manifest["suffix_pairs"][:cap]
        pa = [dict(x=p["x"], rels=p["rels_a"], gold=p["gold_a"]) for p in prs]; pb = [dict(x=p["x"], rels=p["rels_b"], gold=p["gold_b"]) for p in prs]
        sl = [frozen_slots(4, p["pair"], p["d"]) for p in prs]; ra = EvalSet(v, pa, sl, device).run(model, v, MAIN_T[mode]); rb = EvalSet(v, pb, sl, device).run(model, v, MAIN_T[mode])
        res["suffix_pairs"] = {}
        for T in MAIN_T[mode]:
            agg = collections.defaultdict(list)
            for i, p in enumerate(prs):
                agg[f"d{p['d']}"].append((ra[T][0][i] == 0 and rb[T][0][i] == 0, not np.array_equal(ra[T][1][i], rb[T][1][i])))
            res["suffix_pairs"][f"T{T}"] = {k: dict(n=len(x), both_correct=float(np.mean([a for a, _ in x])), answer_changes=float(np.mean([b for _, b in x]))) for k, x in agg.items()}
        # shallow: atomic (single-task packets, evaluation only), validation, train-pair fit, diagnostics
        at = [dict(x=x, rels=[r], gold=rel[r][x], cat="atomic", d=1) for r in range(v.R) for x in range(v.E)][:cap]
        ast = EvalSet(v, at, [frozen_slots(3, a["rels"][0] * v.E + a["x"], 1) for a in at], device).run(model, v, [T_shallow])[T_shallow][0]
        fit = [dict(x=q["x"], rels=[q["r"], q["s"]], gold=rel[q["s"]][rel[q["r"]][q["x"]]], cat="train_d2", d=2) for q in train_d2[::10]][:cap]
        fst = EvalSet(v, fit, [frozen_slots(2, i, 2, 1) for i in range(len(fit))], device).run(model, v, [T_shallow])[T_shallow][0]
        res["shallow"] = dict(shallow(T_shallow), atomic=dict(n=len(ast), acc=float((ast == 0).mean()), format=float((ast == 2).mean())), train_d2_fit=dict(n=len(fst), acc=float((fst == 0).mean())))
        res["diagnostics_val_d2"] = dict(note="read-head hit rate and entity probe are labelled diagnostics, not an observed execution trace; the probe head is trained only in B / D, the read head only in C / D",
                                         state_aux_trained=bool(cfg["state"]), read_aux_trained=bool(cfg["read"]), **d2_diag(model, v, rel, val_d2[:cap], T_shallow, device, 0))
        # tail checkpoints and secondary best: first 500 per cell at the main budgets only
        res["tail"] = {}
        for name in [f"ckpt_{total - 10000:06d}.pt", f"ckpt_{total - 5000:06d}.pt", "best.pt"]:
            if os.path.exists(os.path.join(args.save_dir, name)):
                ck = torch.load(os.path.join(args.save_dir, name), map_location=device, weights_only=False); model.load_state_dict(ck["model"])
                rt = MAT.run(model, v, MAIN_T[mode]); res["tail"][name] = dict(update=ck["update"], **{f"T{T}": cell_stats(m_items, rt[T][0], cell_key) for T in MAIN_T[mode]}, shallow=shallow(T_shallow))
        model.load_state_dict(final_sd)
        res["eval_seconds"] = time.time() - t0; res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
        np.savez_compressed(os.path.join(args.save_dir, "final_answers.npz"), **raw)
        json.dump(res, open(path, "w"), indent=1)
        for T in MAIN_T[mode]:
            a = res["main"][f"T{T}"]["all"]
            print(f"[final T{T}] " + " ".join(f"{k}:{a[k]['acc']:.3f}" for k in a if not k.startswith("all_seen")))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    s_w2 = Stream(len(train_w2), args.seed, "w2"); s_d2 = Stream(len(train_d2), args.seed, "d2")
    lay_rng = named_rng(args.seed, "latent_layout"); t_rng = named_rng(args.seed, "latent_T")
    ctr = dict(w2=0, d2=0, answer_tokens=0, state_labels=0, read_labels=0, clipped=0, flops=0.0, T_hist={str(t): 0 for t in train_T + [1]}, core_updates=[0] * len(model.cores))
    start = 1; best = None
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl"); log_path = os.path.join(args.save_dir, "train_log.jsonl"); raw_path = os.path.join(args.save_dir, "monitor_raw.jsonl")
    last_path = os.path.join(args.save_dir, "last.pt")
    if (args.resume or args.eval_only) and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); s_w2.load(st["s_w2"]); s_d2.load(st["s_d2"])
        lay_rng.bit_generator.state = st["lay_rng"]; t_rng.bit_generator.state = st["t_rng"]; torch.set_rng_state(st["torch_rng"].cpu())
        ctr = st["ctr"]; start = st["update"] + 1; best = st["best"]; print(f"resumed from update {st['update']}")
    elif not args.eval_only:
        for p in (metrics_path, log_path, raw_path):
            open(p, "w").close()
        src = hashlib.sha256(b"".join(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), "rb").read() for f in ("train_latent.py", "model/latent_executor.py", "data/latent_format.py"))).hexdigest()[:16]
        json.dump(dict(cond=args.cond, cond_name=cfg["name"], mode=mode, seed=args.seed, state_aux=cfg["state"], read_aux=cfg["read"], aux_weight=args.aux_weight, train_T=train_T if mode != "single" else [1],
                       params=counts, init_hash=hsh(lambda n: True), init_hash_common_with_A=hsh(lambda n: not n.startswith("cores.") or n.startswith("cores.0.")),
                       vocab_size=len(v.vocab), vocab_hash=v.hash, vocab_tail=v.vocab[-5:], data_hashes=meta.get("hashes"), test_layout_hash=ev_manifest["test_layout_hash"], source_hash=src,
                       schedule=dict(rows_per_update=args.rows_per_update, phase1=args.phase1_updates, phase2=args.phase2_updates), optimizer=dict(name="AdamW", lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=args.weight_decay, warmup=args.warmup, clip=args.clip),
                       main_T=MAIN_T[mode], grid_T=GRID[mode], monitor_T=MON_T[mode], precision="fp32", kv_cache=False,
                       flops_note="analytic estimate: linear layers + attention matmuls, training = 3 x forward; aux heads included when active"),
                  open(os.path.join(args.save_dir, "manifest.json"), "w"), indent=1)
    print(f"cond {args.cond} ({cfg['name']}, {mode}) seed {args.seed} | params {counts} | device {device}")
    if args.eval_only:
        final_eval(); return

    d = args.d_model
    def fwd_flops(n_mem_tok, B, T, n_ans_tok, n_aux):
        Lm = n_mem_tok / B; nl = model.n_latent
        enc = 2 * (12 * d * d * n_mem_tok + 2 * B * Lm * Lm * d)
        core = 2 * T * ((4 + 2 + 8) * d * d * B * nl + 2 * d * d * n_mem_tok + 2 * B * nl * nl * d + 2 * B * nl * Lm * d)
        dec = (4 + 2 + 8) * d * d * n_ans_tok + 2 * d * d * B * nl + d * len(v.vocab) * n_ans_tok
        return 2.0 * (enc + core + dec) + 2.0 * d * v.E * n_aux

    def is_eval(u):
        if args.smoke:
            return u % 20 == 0
        return (1000 <= u <= 5000 and u % 200 == 0) or (args.phase1_updates <= u <= args.phase1_updates + 5000 and u % 200 == 0) or u % 5000 == 0 or u == total

    acc = collections.defaultdict(float); n_acc = 0; t0 = time.time(); ema_gn = None
    for u in range(start, total + 1):
        model.train(); phase = 1 if u <= args.phase1_updates else 2
        n_w2 = args.rows_per_update if phase == 1 else args.rows_per_update // 2; n_d2 = args.rows_per_update - n_w2
        qi_w2 = s_w2.take(n_w2); qi_d2 = s_d2.take(n_d2) if n_d2 else np.zeros(0, dtype=np.int64)
        T_draw = int(t_rng.choice(train_T)); T = 1 if mode == "single" else T_draw          # the T stream advances identically in every condition
        b = tables.batch(qi_w2, qi_d2, lay_rng)
        use_state = bool(cfg["state"]) and phase == 2; use_read = bool(cfg["read"]) and phase == 2   # phase 1: the aux graph is not built at all
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, u / max(1, args.warmup))
        out = model(b["tok"], b["pos"], b["pad"], T, ans_in=b["ans_in"], collect=use_state or use_read)
        ce = F.cross_entropy(out["logits"].transpose(1, 2), b["ans_tgt"], reduction="none", ignore_index=-100)
        L_ans = (ce * b["ans_w"]).sum() / args.rows_per_update; loss = L_ans; n_aux = 0
        if use_state:
            st = torch.stack(out["states"])[:, n_w2:]; tg = b["aux"]["x2"][None].expand(T, -1).clone(); tg[0] = b["aux"]["x1"]
            ce_s = F.cross_entropy(model.ent_head(st).flatten(0, 1), tg.flatten(), reduction="none").view(T, -1)
            L_state = ce_s.mean(0).sum() / args.rows_per_update; loss = loss + args.aux_weight * L_state; n_aux = T * n_d2
            acc["L_state"] += float(L_state); acc["ce_state_exec"] += float(ce_s[:2].mean()); acc["ce_state_post"] += float(ce_s[2:].mean()) if T > 2 else 0.0
        if use_read:
            rd = torch.stack(out["reads"])[:, n_w2:]; key = addr_to_key(b["pos"][n_w2:], b["pad"][n_w2:], b["aux"]["addr"])
            kt = key[:, [min(t, 2) for t in range(T)]].t()                                     # loop 1 -> r1 slot, loop 2 -> r2 slot, loops >= 3 -> EOP
            nl = -torch.log(rd.gather(2, kt[..., None]).squeeze(-1) + 1e-12)
            L_read = nl.mean(0).sum() / args.rows_per_update; loss = loss + args.aux_weight * L_read
            acc["L_read"] += float(L_read); acc["nll_read_exec"] += float(nl[:2].mean()); acc["nll_read_post"] += float(nl[2:].mean()) if T > 2 else 0.0
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)); opt.step()
        ctr["w2"] += n_w2; ctr["d2"] += n_d2; ctr["answer_tokens"] += 3 * n_w2 + 2 * n_d2; ctr["clipped"] += int(gn > args.clip); ctr["T_hist"][str(T)] += 1
        ctr["state_labels"] += T * n_d2 * int(use_state); ctr["read_labels"] += T * n_d2 * int(use_read)
        for c in range(T if mode == "untied" else 1):
            ctr["core_updates"][c] += 1
        ctr["flops"] += 3.0 * fwd_flops(int((~b["pad"]).sum()), n_w2 + n_d2, T, 3 * n_w2 + 2 * n_d2, n_aux)
        acc["loss"] += float(loss); acc["L_ans"] += float(L_ans); acc["grad_norm"] += gn; n_acc += 1
        if u % 100 == 0 or u == 1:
            with open(log_path, "a") as f:
                f.write(json.dumps(dict(update=u, phase=phase, T=T, lr=opt.param_groups[0]["lr"], elapsed=time.time() - t0, clipped_frac=ctr["clipped"] / u, **{k: val / n_acc for k, val in acc.items()})) + "\n")
            acc.clear(); n_acc = 0
        if is_eval(u):
            model.eval(); te = time.time(); rec = dict(update=u, phase=phase, elapsed=time.time() - t0, shallow=shallow(T_shallow))
            rmon = MON.run(model, v, MON_T[mode]); rec["monitor"] = {f"T{T_}": cell_stats(mon_items, rmon[T_][0], cell_key) for T_ in MON_T[mode]}
            with open(raw_path, "a") as f:
                f.write(json.dumps(dict(update=u, ids=[t["id"] for t in mon_items] if u == start or not os.path.getsize(raw_path) else None, answers={f"T{T_}": rmon[T_][1].tolist() for T_ in MON_T[mode]})) + "\n")
            if u % 25000 == 0 or u == total or (args.smoke and u % 40 == 0):
                rg = GRID64.run(model, v, GRID[mode]); rec["grid64"] = {f"T{T_}": cell_stats(grid_items, rg[T_][0], cell_key) for T_ in GRID[mode]}
            rec["counters"] = dict(ctr); rec["eval_seconds"] = time.time() - te
            rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            sh = rec["shallow"]; m8 = rec["monitor"][f"T{MON_T[mode][0]}"]
            print(f"u {u} ph{phase} | val_w2 {sh['val_w2']['row_acc']:.3f} val_d2 {sh['val_d2']['acc']:.3f} | " + " ".join(f"{k}:{m8[k]['acc']:.2f}" for k in m8) + f" | clip {ctr['clipped'] / u:.2f} | eval {rec['eval_seconds']:.0f}s | {time.time() - t0:.0f}s", flush=True)
            if phase == 2 and u > args.phase1_updates:                      # secondary best: only after phase 1, shallow validation only
                score = (0.5 * sh["val_w2"]["row_acc"] + 0.5 * sh["val_d2"]["acc"], -(sh["val_w2"]["L_ans"] + sh["val_d2_L_ans"]) / 2)
                if best is None or score > tuple(best[:2]):
                    best = [score[0], score[1], u]; torch.save(dict(model=model.state_dict(), update=u), os.path.join(args.save_dir, "best.pt"))
        if u % 5000 == 0 or u == total or u == args.stop_after:
            if u % 25000 == 0 or u >= total - 10000:
                torch.save(dict(model=model.state_dict(), update=u), os.path.join(args.save_dir, f"ckpt_{u:06d}.pt"))
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), update=u, s_w2=s_w2.state(), s_d2=s_d2.state(), lay_rng=lay_rng.bit_generator.state,
                            t_rng=t_rng.bit_generator.state, torch_rng=torch.get_rng_state(), ctr=ctr, best=best), last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)
        if u == args.stop_after:
            print(f"stopped at update {u} ({(time.time() - t0) / max(1, u - start + 1) * 1000:.1f} ms / update)"); return
    if not args.no_final_eval:
        final_eval()
    print("done")


if __name__ == "__main__":
    main()
