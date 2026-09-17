"""
Pre-registered readout for the role-control arms (docs/propose_experiment.md §4.5 / §5.3; docs/experiments_P0_P3.md).

Part A (metrics.jsonl): for every run, the mean ACC per test type over the epochs of the LAST FIVE saved checkpoints
(E, E-25, ..., E-100), plus the trailing --window mean; seeds are shown separately.
Part B (paired difference): for --main type, per-item predictions at those five checkpoints (scripts/role_tags.py,
run on demand) are averaged per item, then the difference cmp - ref is bootstrapped by held-out relation H
(resample the 8 H with replacement, and within each H resample bridge blocks - one block = all reserved-partner
items of one bridge). Reports the point estimate and 90% / 95% percentile intervals per seed, and whether the
90% interval lies entirely inside +-5pp (equivalence) or entirely above +5pp (substantive effect).

    python scripts/readout_role_control.py --arms RC_k1,RC_k2,RC_k4,RC_k8 --seeds 1,7,123 --ref RC_k1 --cmp RC_k8
    python scripts/readout_role_control.py --arms RC_none,RC_first4,RC_k4,RC_both4 --main hh_cov_both --ref RC_k4 --cmp RC_both4
"""
import argparse, collections, glob, json, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_skills import SHORT  # noqa: E402
PY = sys.executable
DEFAULT_TYPES = ("rc2_newpair_role_unseen,rc2_newpair_role_seen,rc2_trainpartner_unseen,rc1_newpair_role_unseen,"
                 "rc1_newpair_role_seen,hh_cov_both,hh_cov_other,d2_train_unseen_pair,atomic_heldout,w4_heldout")


EXPLICIT_EPOCHS = None   # set from --ckpt_epochs
STRIDE = 25              # set from --stride (stream-mode runs save every eval: use --stride 1)


def ckpt_epochs(run, n=5, stride=None):
    eps = sorted(int(re.search(r"epoch(\d+)", f).group(1)) for f in glob.glob(os.path.join(run, "epoch*.pt")))
    if not eps:
        return []
    if EXPLICIT_EPOCHS:
        return sorted(e for e in EXPLICIT_EPOCHS if e in eps)
    stride = STRIDE if stride is None else stride
    last = eps[-1]
    want = [last - stride * i for i in range(n)]
    return sorted(e for e in want if e in eps)


def load_metrics(run):
    rows = {}
    with open(os.path.join(run, "metrics.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["epoch"]] = r
    return rows


def acc(row, t, metric="ACC"):
    return row["val"].get(t, {}).get(metric)


def part_a(runs_dir, arms, seeds, types, window, metric):
    which = f"epochs {EXPLICIT_EPOCHS}" if EXPLICIT_EPOCHS else f"the last five saved checkpoints (stride {STRIDE})"
    print(f"\n== Part A: mean {metric} over {which}; [w{window}] = trailing {window}-epoch mean")
    print(f"  {'arm':10} {'seed':>4} {'E':>5}  " + " ".join(f"{SHORT.get(t, t)[:14]:>14}" for t in types))
    for arm in arms:
        for seed in seeds:
            run = os.path.join(runs_dir, f"skills_{arm}_s{seed}")
            if not os.path.exists(os.path.join(run, "metrics.jsonl")):
                print(f"  {arm:10} {seed:>4}  (no metrics.jsonl)")
                continue
            rows = load_metrics(run)
            eps = ckpt_epochs(run)
            last = max(rows)
            cells = []
            for t in types:
                v5 = [acc(rows[e], t, metric) for e in eps if e in rows and acc(rows[e], t, metric) is not None]
                vw = [acc(rows[e], t, metric) for e in range(max(1, last - window + 1), last + 1) if e in rows and acc(rows[e], t, metric) is not None]
                a = f"{np.mean(v5):.3f}" if v5 else "  -  "
                b = f"[{np.mean(vw):.3f}]" if vw else "[  -  ]"
                cells.append(f"{a + b:>14}")
            print(f"  {arm:10} {seed:>4} {last:>5}  " + " ".join(cells) + ("" if len(eps) == 5 else f"   (only {len(eps)} ckpts: {eps})"))


def per_item(run, ckpts, main):
    """Average correct per item over checkpoints, keyed by (h, r1, r2). Runs role_tags.py if needed."""
    pred_path = os.path.join(run, "role_tags", "predictions.jsonl")
    need = True
    if os.path.exists(pred_path):
        have = set()
        with open(pred_path) as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == main:
                    have.add(r["ckpt"])
        need = not set(ckpts) <= have
    if need:
        cmd = [PY, os.path.join(HERE, "role_tags.py"), run, "--ckpts", ",".join(ckpts), "--types", main]
        print("  running:", " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")})
    agg = collections.defaultdict(list)
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == main and r["ckpt"] in ckpts:
                agg[(r["h"], r["r1"], r["r2"], r["b"])].append(r["correct"])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def bootstrap_diff(ref_items, cmp_items, main, n_boot=2000, seed=0):
    """Paired cmp - ref; resample H (the held-out relation of the item) and bridge blocks within H."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(ref_items) & set(cmp_items))
    hrel = (lambda k: k[2]) if main.startswith(("rc2", "hh")) else (lambda k: k[1])   # rc2/hh: H is r2; rc1: H is r1
    block = (lambda k: k[3]) if main.startswith(("rc2", "hh")) else (lambda k: k[0])  # rc2/hh: bridge; rc1: input h
    by_h = collections.defaultdict(lambda: collections.defaultdict(list))
    for k in keys:
        by_h[hrel(k)][block(k)].append(cmp_items[k] - ref_items[k])
    hs = sorted(by_h)
    blocks = {h: [np.mean(v) for v in by_h[h].values()] for h in hs}
    point = float(np.mean([d for h in hs for d in blocks[h]]))
    boots = []
    for _ in range(n_boot):
        hh = rng.choice(hs, size=len(hs), replace=True)
        vals = []
        for h in hh:
            b = np.asarray(blocks[h])
            vals.append(b[rng.integers(len(b), size=len(b))].mean())
        boots.append(np.mean(vals))
    boots = np.asarray(boots)
    return point, np.percentile(boots, [5, 95]), np.percentile(boots, [2.5, 97.5]), len(keys), {h: float(np.mean(blocks[h])) for h in hs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=os.path.join(HERE, "..", "runs"))
    ap.add_argument("--arms", default="RC_k1,RC_k2,RC_k4,RC_k8")
    ap.add_argument("--seeds", default="1,7,123")
    ap.add_argument("--types", default=DEFAULT_TYPES)
    ap.add_argument("--metric", default="ACC")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--main", default="rc2_newpair_role_unseen")
    ap.add_argument("--ref", default=None, help="reference arm for the paired difference (e.g. RC_k1)")
    ap.add_argument("--cmp", default=None, help="comparison arm (e.g. RC_k8)")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--ckpt_epochs", default=None, help="explicit checkpoint epochs for the endpoint mean, e.g. 471,484,497,510,524 (N1)")
    ap.add_argument("--stride", type=int, default=25, help="checkpoint stride for the default 'last five' rule (stream-mode runs: 1)")
    args = ap.parse_args()
    global EXPLICIT_EPOCHS, STRIDE
    EXPLICIT_EPOCHS = [int(x) for x in args.ckpt_epochs.split(",")] if args.ckpt_epochs else None
    STRIDE = args.stride
    arms = args.arms.split(","); seeds = [int(s) for s in args.seeds.split(",")]; types = args.types.split(",")
    part_a(args.runs_dir, arms, seeds, types, args.window, args.metric)
    if not (args.ref and args.cmp):
        return
    print(f"\n== Part B: paired {args.cmp} - {args.ref} on {args.main}, per item averaged over the last five checkpoints; bootstrap by H (and bridge blocks within H)")
    for seed in seeds:
        rr = os.path.join(args.runs_dir, f"skills_{args.ref}_s{seed}"); rc = os.path.join(args.runs_dir, f"skills_{args.cmp}_s{seed}")
        e_ref, e_cmp = ckpt_epochs(rr), ckpt_epochs(rc)
        if len(e_ref) < 5 or len(e_cmp) < 5:
            print(f"  seed {seed}: not finished (ref ckpts {e_ref}, cmp ckpts {e_cmp})")
            continue
        ref_items = per_item(rr, [f"epoch{e:03d}.pt" for e in e_ref], args.main)
        cmp_items = per_item(rc, [f"epoch{e:03d}.pt" for e in e_cmp], args.main)
        point, ci90, ci95, n, per_h = bootstrap_diff(ref_items, cmp_items, args.main)
        m_ref = np.mean(list(ref_items.values())); m_cmp = np.mean(list(cmp_items.values()))
        if ci90[0] > args.margin:
            verdict = f"90% CI above +{args.margin:.2f}: substantive effect"
        elif ci90[1] < -args.margin:
            verdict = f"90% CI below -{args.margin:.2f}: substantive negative effect"
        elif ci90[0] > -args.margin and ci90[1] < args.margin:
            verdict = f"90% CI inside +-{args.margin:.2f}: equivalence (small effect within this budget)"
        else:
            verdict = "inconclusive"
        print(f"  seed {seed}: {args.ref} {m_ref:.3f}  {args.cmp} {m_cmp:.3f}  diff {point:+.3f}  90% [{ci90[0]:+.3f}, {ci90[1]:+.3f}]  95% [{ci95[0]:+.3f}, {ci95[1]:+.3f}]  n_items {n}  -> {verdict}")
        print("           per-H diff: " + ", ".join(f"r{h}: {d:+.3f}" for h, d in per_h.items()))


if __name__ == "__main__":
    main()
