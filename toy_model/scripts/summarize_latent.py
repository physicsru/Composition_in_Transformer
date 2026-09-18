"""Read-only summary of the fifth batch (docs/experiments_latent_batch2_20runs.md §7-§8).

    python scripts/summarize_latent.py                      # all runs/latent2_*_s*
    python scripts/summarize_latent.py --subset confirm4500 # pre-registered confirmation subset (default: all 5000 per cell)

Keeps apart: primary results (250k, fixed budgets T=8 / T=256; E: T=1; F has no T>8 -> N/A, never 0), the pre-registered
criteria of §8.1, factor effects on the paired seeds 1 / 7 / 123 (§8.2), illegal-format rates, and training progress of
unfinished runs (monitoring numbers, clearly labelled).
"""
import argparse, glob, json, os, re, collections

DEPTHS = (2, 3, 4, 8, 16, 32, 64, 128)


def load(run):
    out = dict(run=run); m = re.search(r"latent2_([A-F])_s(\d+)", run); out["cond"], out["seed"] = m.group(1), int(m.group(2))
    p = os.path.join(run, "final_eval.json"); out["final"] = json.load(open(p)) if os.path.exists(p) else None
    mp = os.path.join(run, "metrics.jsonl"); out["last"] = None
    if os.path.exists(mp) and os.path.getsize(mp):
        out["last"] = json.loads(open(mp).read().strip().split("\n")[-1])
    return out


def acc(fin, T, subset, cat, d):
    blk = fin["main"].get(f"T{T}")
    if blk is None:
        return None
    c = blk[subset].get(f"{cat}|d{d}"); return None if c is None else c["acc"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="*"); ap.add_argument("--subset", default="all", choices=["all", "confirm4500", "monitor500"])
    args = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
    runs = [load(r) for r in (args.runs or sorted(glob.glob(os.path.join(root, "latent2_[A-F]_s*"))) ) if re.search(r"latent2_[A-F]_s\d+$", r.rstrip("/"))]
    fmt = lambda x: " N/A " if x is None else f"{x:5.3f}"
    print(f"== PRIMARY (250k final, subset = {args.subset}); min over one_new / random per depth; all_seen shown separately for d2 (training-pair fit)")
    for T in (8, 256, 1):
        rows = [r for r in runs if r["final"] and f"T{T}" in r["final"]["main"]]
        if not rows:
            continue
        print(f"-- budget T={T}" + ("  (E only)" if T == 1 else "  (A-D only; F / E have no T=256 by design)" if T == 256 else ""))
        print("   run          fit_d2 | " + " ".join(f"  d{d:<3d}" for d in DEPTHS) + " | format_err(max)")
        for r in rows:
            f = r["final"]; vals = [min(x for x in (acc(f, T, args.subset, "one_new", d), acc(f, T, args.subset, "random", d)) if x is not None) if acc(f, T, args.subset, "one_new", d) is not None else None for d in DEPTHS]
            ferr = max(c["format"] for c in f["main"][f"T{T}"][args.subset].values())
            print(f"   {r['cond']}_s{r['seed']:<5d}      {fmt(acc(f, T, args.subset, 'all_seen', 2))} | " + " ".join(fmt(x) for x in vals) + f" | {ferr:.3f}")
    # §8.1 criteria per condition at each available budget
    print("== CRITERIA (§8.1): seeds meeting >= 0.95 on BOTH one_new and random (final checkpoint); tail checkpoints are listed in final_eval.json['tail']")
    for cond in "ABCDEF":
        rs = [r for r in runs if r["cond"] == cond and r["final"]]
        if not rs:
            continue
        for T in rs[0]["final"]["main_T"]:
            ok = lambda r, ds: all((acc(r["final"], T, args.subset, c, d) or 0) >= 0.95 for c in ("one_new", "random") for d in ds)
            print(f"   {cond} T={T:<3d}: d2 gate {sum(ok(r, (2,)) for r in rs)}/{len(rs)} | d3-d8 {sum(ok(r, (3, 4, 8)) for r in rs)}/{len(rs)} | d16-d128 {sum(ok(r, (16, 32, 64, 128)) for r in rs)}/{len(rs)}")
    # §8.2 factor effects on paired seeds
    fin = {(r["cond"], r["seed"]): r["final"] for r in runs if r["final"]}
    print("== FACTOR EFFECTS (§8.2), paired seeds 1 / 7 / 123, T=8, accuracy = mean of one_new and random; per seed, then mean [min, max]")
    def a(c, s, d, T=8):
        f = fin.get((c, s)); xs = [acc(f, T, args.subset, cat, d) for cat in ("one_new", "random")] if f else [None]
        return None if any(x is None for x in xs) else sum(xs) / 2
    effects = dict(state=lambda s, d: ((a("B", s, d) - a("A", s, d)) + (a("D", s, d) - a("C", s, d))) / 2, read=lambda s, d: ((a("C", s, d) - a("A", s, d)) + (a("D", s, d) - a("B", s, d))) / 2,
                   interaction=lambda s, d: a("D", s, d) - a("B", s, d) - a("C", s, d) + a("A", s, d), loops_A_minus_E=lambda s, d: a("A", s, d) - a("E", s, d, 1), sharing_A_minus_F=lambda s, d: a("A", s, d) - a("F", s, d))
    for name, fn in effects.items():
        line = []
        for d in DEPTHS:
            try:
                xs = [fn(s, d) for s in (1, 7, 123)]; line.append(f"d{d}: {sum(xs) / 3:+.3f} [{min(xs):+.2f},{max(xs):+.2f}]")
            except TypeError:
                line.append(f"d{d}: n/a")
        print(f"   {name:18s} " + "  ".join(line))
    # progress of unfinished runs (MONITORING numbers: 64 chains per cell, not results)
    un = [r for r in runs if not r["final"] and r["last"]]
    if un:
        print("== IN PROGRESS (monitoring set, 64 per cell -- not results)")
        for r in un:
            l = r["last"]; T0 = sorted(l["monitor"], key=lambda k: int(k[1:]))[0]; mon = l["monitor"][T0]
            print(f"   {r['cond']}_s{r['seed']:<5d} u{l['update']:6d} val_w2 {l['shallow']['val_w2']['row_acc']:.2f} val_d2 {l['shallow']['val_d2']['acc']:.2f} | {T0} " + " ".join(f"{k}:{mon[k]['acc']:.2f}" for k in mon))


if __name__ == "__main__":
    main()
