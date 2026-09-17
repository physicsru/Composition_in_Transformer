"""
N0 (docs/findings_and_next_experiments.md §4): bridge patching with a FIXED, stratified target/source manifest.

Targets are drawn from the common design pool of a role-control world (meta.role_split), so every arm, training seed
and checkpoint is evaluated on the same items:
  HT   per H: 15 inputs from S1 and 15 from U1  x  the 4 reserved partners        (<e_h><r_H><r_T>)   960
  TH   per H: 15 bridges from S2 and 15 from U2 x  the 4 reserved partners        (<e_h><r_T><r_H>)   960
  HH   per ordered (H1, H2) x four coverage cells (h in S1_H1?, b in S2_H2?), <= 20 each              <= 5,120
  TT   d2_train_unseen_{instance,pair} items, balanced over the second relation, 300
Sources per target and condition (two per condition, different first relations, drawn once with --seed):
  same_bridge          (h', T') with T'(h') = b, T' != r1
  wrong_bridge_S2      bridge b' from the design set S2 of the target's second relation (for r2 in H; actual training
                       role-2 coverage of r2 when r2 is a train relation), b' != b
  wrong_bridge_U2      bridge b' from U2 (or the actually-uncovered pool for train r2; N/A if empty)
  uniform_bridge       b' uniform over entities != b
The block-0 residual at position --pos of the source <e_h'><r_T'><r2> replaces the target's. Per target the two
sources are averaged first; then per (test, coverage cell, H or H-pair). Reports p(correct), margin, accuracy,
new-answer following (pred == r2(b')), original-answer keeping, and flips; bootstrap CIs resample H (H pairs for HH)
and targets within H. Also checks that the position-1 block-0 state of <e_x><r_T><r2> equals that of <e_x><r_T><e_ans>.

    python scripts/patch_bridge_balanced.py runs/skills_RC_k4_s1 --ckpts epoch525.pt [--pos 1] [--seed 20260916]
Writes <run>/patch_bridge_balanced_v1/{manifest.json,predictions.jsonl,summary.json}.
"""
import argparse, collections, glob, hashlib, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from role_tags import parse_tasks, training_sets, load_model  # noqa: E402

CONDS = ["no_patch", "same_bridge", "wrong_bridge_S2", "wrong_bridge_U2", "uniform_bridge"]


class RowPatcher:
    """Replace block output at one position with a per-row vector (B, d)."""

    def __init__(self, model, layer):
        self.pos, self.inject, self.store = None, None, None
        self.h = model.blocks[layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.store is not None:
            self.store.append(out[:, self.pos, :].detach())
        if self.inject is not None:
            out = out.clone(); out[:, self.pos, :] = self.inject; return out


def build_manifest(meta, role2_actual, test_items, seed, per_cell=15, hh_per_cell=20, n_tt=300):
    rel = meta["rel_map"]; E = meta["num_entities"]
    T = list(meta["train_relations"]); H = list(meta["heldout_relations"])
    inv = {j: {rel[j][h]: h for h in range(E)} for j in range(len(rel))}
    split = {int(k): v for k, v in meta["role_split"].items()}
    rng = np.random.default_rng(seed)
    targets = []

    def add(test, h, r1, r2, cell, group):
        b = rel[r1][h]
        targets.append(dict(id=len(targets), test=test, h=int(h), r1=int(r1), r2=int(r2), b=int(b), t=int(rel[r2][b]), cell=cell, group=group))

    for j in H:
        sp = split[j]
        for cell, pool in (("S1", sp["S1"]), ("U1", sp["U1"])):
            for h in rng.choice(pool, per_cell, replace=False):
                for t in sp["reserved"]:
                    add("HT", h, j, t, cell, f"r{j}")
        for cell, pool in (("S2", sp["S2"]), ("U2", sp["U2"])):
            for b in rng.choice(pool, per_cell, replace=False):
                for t in sp["reserved"]:
                    add("TH", inv[t][b], t, j, cell, f"r{j}")
    for j1 in H:
        s1 = set(split[j1]["S1"])
        for j2 in H:
            s2 = set(split[j2]["S2"])
            cells = collections.defaultdict(list)
            for h in range(E):
                b = rel[j1][h]
                cells[("S1" if h in s1 else "U1") + "/" + ("S2" if b in s2 else "U2")].append(h)
            for cell, hs in sorted(cells.items()):
                for h in rng.choice(hs, min(hh_per_cell, len(hs)), replace=False):
                    add("HH", h, j1, j2, cell, f"r{j1}->r{j2}")
    tt = [it for it in test_items if it["type"] in ("d2_train_unseen_instance", "d2_train_unseen_pair")]
    by_r2 = collections.defaultdict(list)
    for it in tt:
        (h, chain, t), = parse_tasks(it["target_text"]); by_r2[chain[1]].append((h, chain[0], chain[1]))
    k = 0
    while k < n_tt:
        for r2 in sorted(by_r2):
            if by_r2[r2] and k < n_tt:
                h, r1, _ = by_r2[r2].pop(int(rng.integers(len(by_r2[r2]))))
                add("TT", h, r1, r2, "-", f"r{r2}"); k += 1
    # sources
    for tg in targets:
        r2 = tg["r2"]; b = tg["b"]
        if r2 in split:
            S2, U2 = split[r2]["S2"], split[r2]["U2"]
        else:
            S2 = [x for x in range(E) if (x, r2) in role2_actual]; U2 = [x for x in range(E) if (x, r2) not in role2_actual]
        cands = [t for t in T if t != tg["r1"]]
        src = {}
        t1, t2 = rng.choice(cands, 2, replace=False)
        src["same_bridge"] = [(int(inv[t][b]), int(t), int(b)) for t in (t1, t2)]
        for name, pool in (("wrong_bridge_S2", S2), ("wrong_bridge_U2", U2), ("uniform_bridge", list(range(E)))):
            pool = [x for x in pool if x != b]
            if len(pool) == 0:
                src[name] = []; continue
            bs = rng.choice(pool, 2, replace=len(pool) < 2)
            ts = rng.choice(cands, 2, replace=False)
            src[name] = [(int(inv[t][bb]), int(t), int(bb)) for bb, t in zip(bs, ts)]
        tg["sources"] = src
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpts", default=None, help="comma-separated; default: last saved checkpoint")
    ap.add_argument("--pos", type=int, default=1)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    run = args.run.rstrip("/")
    cell = os.path.basename(run).replace("skills_", "").rsplit("_s", 1)[0]
    data = args.data or os.path.join(os.path.dirname(run), "..", "data", f"skills_{cell}")
    out_dir = args.out or os.path.join(run, "patch_bridge_balanced_v1")
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(data, "meta.json"))); rel = meta["rel_map"]; E = meta["num_entities"]
    H = set(meta["heldout_relations"])
    train_rows = json.load(open(os.path.join(data, "train.json")))
    _, _, role1, role2, _, _ = training_sets(train_rows, rel)
    test_items = json.load(open(os.path.join(data, "test.json")))
    vocab = json.load(open(os.path.join(data, "vocab.json"))); tok2id = {t: i for i, t in enumerate(vocab)}
    e_id = lambda e: tok2id[f"<e_{e}>"]; r_id = lambda r: tok2id[f"<r_{r}>"]
    targets = build_manifest(meta, role2, test_items, args.seed)
    for tg in targets:   # actual coverage in THIS arm's training rows
        tg["role1_seen"] = (tg["h"], tg["r1"]) in role1; tg["role2_seen"] = (tg["b"], tg["r2"]) in role2
        for name, srcs in tg["sources"].items():
            tg[f"src_role2_seen/{name}"] = [((bb, tg["r2"]) in role2) for _, _, bb in srcs]
    manifest = dict(seed=args.seed, pos=args.pos, layer=args.layer, n_targets=len(targets),
                    per_test=dict(collections.Counter(t["test"] for t in targets)),
                    data_hashes={k: hashlib.sha256(open(os.path.join(data, f"{k}.json"), "rb").read()).hexdigest()[:16] for k in ("train", "test", "meta")},
                    targets=targets)
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"))
    if args.ckpts:
        ckpts = [os.path.join(run, c) for c in args.ckpts.split(",")]
    else:
        allck = sorted(glob.glob(os.path.join(run, "epoch*.pt")), key=lambda f: int(re.search(r"epoch(\d+)", f).group(1)))
        ckpts = allck[-1:]
    summary = {}
    fout = open(os.path.join(out_dir, "predictions.jsonl"), "w")
    for ck_path in ckpts:
        model, _ = load_model(ck_path); ck = os.path.basename(ck_path)
        patcher = RowPatcher(model, args.layer)
        # --- consistency check: pos-1 block-0 state of <x><T><r2> vs <x><T><e_ans>
        rng = np.random.default_rng(0)
        xs = rng.integers(E, size=200); ts = rng.choice(list(meta["train_relations"]), size=200)
        r2s = rng.integers(len(rel), size=200)
        a = torch.tensor([[e_id(x), r_id(t), r_id(r2)] for x, t, r2 in zip(xs, ts, r2s)])
        bb = torch.tensor([[e_id(x), r_id(t), e_id(rel[t][x])] for x, t in zip(xs, ts)])
        with torch.no_grad():
            patcher.store = []; patcher.pos = 1; model(a); s1 = patcher.store[0]
            patcher.store = []; model(bb); s2 = patcher.store[0]; patcher.store = None
        consistency = float((s1 - s2).abs().max())
        # --- build all (target, cond, source) rows
        jobs = []   # (target idx, cond, src tuple or None)
        for tg in targets:
            jobs.append((tg["id"], "no_patch", None))
            for cond in CONDS[1:]:
                for src in tg["sources"].get(cond, []):
                    jobs.append((tg["id"], cond, src))
        preds = []
        B = 1024
        with torch.no_grad():
            for s in range(0, len(jobs), B):
                chunk = jobs[s:s + B]
                tgt = torch.tensor([[e_id(targets[i]["h"]), r_id(targets[i]["r1"]), r_id(targets[i]["r2"])] for i, _, _ in chunk])
                patched = [n for n, (_, c, _) in enumerate(chunk) if c != "no_patch"]
                inject = None
                if patched:
                    src_ids = torch.tensor([[e_id(chunk[n][2][0]), r_id(chunk[n][2][1]), r_id(targets[chunk[n][0]]["r2"])] for n in patched])
                    patcher.pos = args.pos; patcher.inject = None
                    patcher.store = []; model(src_ids); vec = patcher.store[0]; patcher.store = None
                    # rows without a patch keep their own state: capture it and overwrite only the patched rows
                    patcher.store = []; model(tgt); own = patcher.store[0].clone(); patcher.store = None
                    own[patched] = vec
                    patcher.inject = own
                    logits = model(tgt)[:, -1, :]
                    patcher.inject = None
                else:
                    logits = model(tgt)[:, -1, :]
                logp = torch.log_softmax(logits, -1); top2 = torch.topk(logits, 2).values; pred = logits.argmax(-1)
                for n, (i, cond, src) in enumerate(chunk):
                    tg = targets[i]; t_id = e_id(tg["t"])
                    rec = dict(ckpt=ck, id=i, cond=cond, pred=vocab[int(pred[n])], correct=int(pred[n] == t_id),
                               p_correct=float(logp[n, t_id].exp()),
                               margin=float(logits[n, t_id] - (top2[n, 1] if pred[n] == t_id else top2[n, 0])))
                    if src is not None:
                        alt = e_id(rel[tg["r2"]][src[2]])
                        rec.update(src_h=src[0], src_r1=src[1], src_b=src[2], follows=int(pred[n] == alt), p_alt=float(logp[n, alt].exp()),
                                   src_role2_seen=(src[2], tg["r2"]) in role2)
                    preds.append(rec); fout.write(json.dumps(rec) + "\n")
        # --- aggregate: per target x cond (mean over sources), then per (test, cell, group)
        per_tc = collections.defaultdict(list)
        for r in preds:
            per_tc[(r["id"], r["cond"])].append(r)
        tgt_val = {}
        for (i, cond), rs in per_tc.items():
            tgt_val[(i, cond)] = dict(correct=np.mean([r["correct"] for r in rs]), p=np.mean([r["p_correct"] for r in rs]),
                                      margin=np.mean([r["margin"] for r in rs]),
                                      follows=np.mean([r.get("follows", np.nan) for r in rs]) if cond != "no_patch" else np.nan)
        groups = collections.defaultdict(list)
        for tg in targets:
            groups[(tg["test"], tg["cell"])].append(tg)
        table = {}
        print(f"\n== {run} @ {ck}  block-{args.layer} output at pos {args.pos}; prefix consistency max|diff| = {consistency:.2e}")
        print(f"  {'test':4} {'cell':7} {'n_tgt':>5} {'cond':16} {'acc':>6} {'p_corr':>6} {'margin':>7} {'follow':>6} {'wrong->ok':>9} {'ok->wrong':>9}  {'same-none diff [90% CI by H]':>30}")
        for (test, cellname), tgs in sorted(groups.items()):
            row = {}
            for cond in CONDS:
                vals = [tgt_val[(tg["id"], cond)] for tg in tgs if (tg["id"], cond) in tgt_val]
                if not vals:
                    continue
                base = [tgt_val[(tg["id"], "no_patch")]["correct"] for tg in tgs if (tg["id"], cond) in tgt_val]
                cur = [v["correct"] for v in vals]
                wrong_ok = np.mean([(b < 0.5) and (c > 0.5) for b, c in zip(base, cur)]); ok_wrong = np.mean([(b > 0.5) and (c < 0.5) for b, c in zip(base, cur)])
                row[cond] = dict(n=len(vals), acc=float(np.mean(cur)), p=float(np.mean([v["p"] for v in vals])), margin=float(np.mean([v["margin"] for v in vals])),
                                 follows=float(np.nanmean([v["follows"] for v in vals])) if cond != "no_patch" else None,
                                 wrong_to_ok=float(wrong_ok), ok_to_wrong=float(ok_wrong))
                ci = ""
                if cond == "same_bridge":
                    # paired diff vs no_patch, bootstrap by group (H or H pair) then targets within group
                    byg = collections.defaultdict(list)
                    for tg in tgs:
                        if (tg["id"], cond) in tgt_val:
                            byg[tg["group"]].append(tgt_val[(tg["id"], cond)]["correct"] - tgt_val[(tg["id"], "no_patch")]["correct"])
                    gs = sorted(byg); brng = np.random.default_rng(1)
                    boots = []
                    for _ in range(args.n_boot):
                        pick = brng.choice(gs, len(gs), replace=True)
                        boots.append(np.mean([np.mean(brng.choice(byg[g], len(byg[g]), replace=True)) for g in pick]))
                    lo, hi = np.percentile(boots, [5, 95]); point = np.mean([np.mean(byg[g]) for g in gs])
                    row[cond].update(diff_vs_none=float(point), diff_ci90=[float(lo), float(hi)], n_groups=len(gs))
                    ci = f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}] ({len(gs)} groups)"
                r = row[cond]
                fol = f"{r['follows']:6.3f}" if r["follows"] is not None else "     -"
                print(f"  {test:4} {cellname:7} {r['n']:5d} {cond:16} {r['acc']:6.3f} {r['p']:6.3f} {r['margin']:7.2f} {fol} {r['wrong_to_ok']:9.3f} {r['ok_to_wrong']:9.3f}  {ci}")
            table[f"{test}|{cellname}"] = row
        # per-H (or per H pair) same_bridge and wrong-bridge follow rates, for the record
        per_group = collections.defaultdict(lambda: collections.defaultdict(list))
        for tg in targets:
            for cond in CONDS:
                if (tg["id"], cond) in tgt_val:
                    per_group[(tg["test"], tg["cell"], tg["group"])][cond].append(tgt_val[(tg["id"], cond)])
        pg = {f"{k[0]}|{k[1]}|{k[2]}": {c: dict(n=len(v), acc=float(np.mean([x["correct"] for x in v])),
                                             follows=(float(np.nanmean([x["follows"] for x in v])) if c != "no_patch" else None)) for c, v in d.items()}
              for k, d in per_group.items()}
        summary[ck] = dict(consistency_max_abs_diff=consistency, table=table, per_group=pg)
        patcher.h.remove()
    fout.close()
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"\noutputs: {out_dir}/manifest.json, predictions.jsonl, summary.json")


if __name__ == "__main__":
    main()
