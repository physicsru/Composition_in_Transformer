"""
Summarize skills runs from their metrics.jsonl (written by train.py every epoch).

    python scripts/summarize_skills.py runs/skills_w2d2_s1 runs/skills_v1_s1 ...
    python scripts/summarize_skills.py runs/skills_* --metric SEQ_ACC
    python scripts/summarize_skills.py runs/skills_w2d2_s1 --curve d2_heldout,d2_train_unseen_instance --every 100

Prints one row per run with the final-epoch value per test type, then the best value over
training (with the epoch it was reached). Types absent from a run are shown as "-".
"""

import argparse
import json
import os

DEFAULT_TYPES = [
    "atomic_train", "atomic_heldout",
    "d2_train_seen", "d2_train_unseen_instance", "d2_train_unseen_pair", "d2_train_uncomposed",
    "d2_mixed_first", "d2_mixed_second", "d2_heldout",
    "d3_train", "d3_heldout", "d3_mixed",
    "w2_heldout", "w3_heldout", "w4_heldout", "w2_mixed", "w4_mixed",
]
SHORT = {
    "atomic_train": "at_tr", "atomic_heldout": "at_ho",
    "d2_train_seen": "d2_seen", "d2_train_unseen_instance": "d2_inst", "d2_train_unseen_pair": "d2_pair",
    "d2_train_uncomposed": "d2_uncmp", "d2_mixed_first": "d2_mixF", "d2_mixed_second": "d2_mixS",
    "d2_heldout": "d2_ho", "d3_train": "d3_tr", "d3_heldout": "d3_ho", "d3_mixed": "d3_mix",
    "w2_heldout": "w2_ho", "w3_heldout": "w3_ho", "w4_heldout": "w4_ho", "w2_mixed": "w2_mix", "w4_mixed": "w4_mix",
    # role-control runs (--task role_control); appended to the default list when a run reports them
    "rc2_newpair_role_seen": "rc2_seen", "rc2_newpair_role_unseen": "rc2_unsn", "rc2_trainpartner_unseen": "rc2_tpU",
    "rc1_newpair_role_seen": "rc1_seen", "rc1_newpair_role_unseen": "rc1_unsn", "hh_cov_both": "hh_both", "hh_cov_other": "hh_othr",
}
RC_TYPES = ["rc2_newpair_role_seen", "rc2_newpair_role_unseen", "rc2_trainpartner_unseen",
            "rc1_newpair_role_seen", "rc1_newpair_role_unseen", "hh_cov_both", "hh_cov_other"]


def load(run):
    path = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metric", default="ACC", help="ACC (per position), SEQ_ACC (whole row), PROB, CE")
    ap.add_argument("--types", default=None, help="comma-separated test types (default: a fixed list)")
    ap.add_argument("--curve", default=None, help="comma-separated types: print their trajectory instead")
    ap.add_argument("--every", type=int, default=100, help="epoch stride for --curve")
    ap.add_argument("--window", type=int, default=1,
                    help="report the mean over the last N epochs instead of the final epoch (smooths lr spikes)")
    args = ap.parse_args()

    types = args.types.split(",") if args.types else DEFAULT_TYPES + RC_TYPES   # rc columns only show up when a run reports them
    runs = [(r, load(r)) for r in args.runs]

    if args.curve:
        ctypes = args.curve.split(",")
        for run, rows in runs:
            print(f"\n{run} ({args.metric})")
            print("  epoch  train_ce  " + "  ".join(f"{SHORT.get(t, t):>9}" for t in ctypes))
            for row in rows:
                e = row["epoch"]
                if e % args.every == 0 or row is rows[-1]:
                    vals = [row["val"].get(t, {}).get(args.metric) for t in ctypes]
                    print(f"  {e:5d}  {row['train_ce']:8.4f}  " + "  ".join(f"{fmt(v):>9}" for v in vals))
        return

    present = [t for t in types if any(t in row["val"] for _, rows in runs for row in rows[-1:])]
    width = max(len(os.path.basename(r.rstrip('/'))) for r, _ in runs) + 1
    header = f"{'run':<{width}} {'ep':>5} " + " ".join(f"{SHORT.get(t, t):>8}" for t in present)

    print(f"final epoch ({args.metric})" if args.window <= 1 else f"mean of last {args.window} epochs ({args.metric})")
    print(header)
    for run, rows in runs:
        name = os.path.basename(run.rstrip("/"))
        if not rows:
            print(f"{name:<{width}}  (no metrics.jsonl)")
            continue
        tail = rows[-max(1, args.window):]
        vals = []
        for t in present:
            series = [r["val"][t][args.metric] for r in tail if t in r["val"]]
            vals.append(sum(series) / len(series) if series else None)
        print(f"{name:<{width}} {rows[-1]['epoch']:>5} " + " ".join(f"{fmt(v):>8}" for v in vals))

    print(f"\nbest over training ({args.metric}, epoch in brackets; min for CE)")
    print(header)
    for run, rows in runs:
        name = os.path.basename(run.rstrip("/"))
        if not rows:
            continue
        cells = []
        for t in present:
            series = [(row["val"][t][args.metric], row["epoch"]) for row in rows if t in row["val"]]
            if not series:
                cells.append(f"{'-':>8}")
                continue
            best = min(series) if args.metric == "CE" else max(series)
            cells.append(f"{best[0]:.3f}[{best[1]}]".rjust(8))
        print(f"{name:<{width}} {rows[-1]['epoch']:>5} " + " ".join(cells))


if __name__ == "__main__":
    main()
