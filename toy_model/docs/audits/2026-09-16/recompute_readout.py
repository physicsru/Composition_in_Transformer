"""Read-only summaries of downloaded metrics; no model inference or checkpoint selection by score."""
import argparse
import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--toy-model-root", type=Path, default=BASE / "toy_model")
parser.add_argument("--output", type=Path, default=BASE / "recomputed_readout.json")
args = parser.parse_args()
ARMS = ["RC_none", "RC_k1", "RC_k2", "RC_k4", "RC_k8", "RC_first4", "RC_both4", "X_none", "X_w2", "X_d2", "X_w2d2"]
run_paths = [args.toy_model_root / "runs" / f"skills_{arm}_s{seed}" for arm in ARMS for seed in (1, 7, 123)]
TARGET_STEPS = [180000, 185000, 190000, 195000, 200000]
out = {"target_steps": TARGET_STEPS, "selection": "nearest logged global_step; earlier epoch breaks a tie", "runs": []}
for run in sorted(run_paths):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len({r["epoch"] for r in rows}) == len(rows)
    cfg = json.loads((run / "config.json").read_text())
    last = rows[-1]
    if "_RC_" in run.name:
        nearest = [min(rows, key=lambda r: (abs(r["global_step"] - step), r["epoch"])) for step in TARGET_STEPS]
        saved_last = (last["epoch"] // 25) * 25
        saved_epochs = [saved_last - 25 * i for i in reversed(range(5))]
        windows = {"reported_saved5": [r for r in rows if r["epoch"] in saved_epochs],
                   "nearest_original5": nearest,
                   "all_180k_200k": [r for r in rows if 180000 <= r["global_step"] <= 200000],
                   "last30": rows[-30:], "last": rows[-1:]}
    else:
        windows = {"last30": rows[-30:], "last": rows[-1:]}
    rec = {"run": run.name, "epoch": last["epoch"], "updates": last["global_step"], "windows": {}}
    for name, selected in windows.items():
        val = {}
        for typ in last["val"]:
            val[typ] = {m: mean(r["val"][typ][m] for r in selected) for m in ("ACC", "SEQ_ACC", "CE", "PROB")}
            val[typ]["N"] = last["val"][typ]["N"]
        rec["windows"][name] = {"epochs": [r["epoch"] for r in selected], "steps": [r["global_step"] for r in selected], "train_ce": mean(r["train_ce"] for r in selected), "val": val}
    out["runs"].append(rec)
args.output.write_text(json.dumps(out, indent=2) + "\n")
types = ["rc2_newpair_role_unseen", "rc2_newpair_role_seen", "rc1_newpair_role_unseen", "rc1_newpair_role_seen", "hh_cov_both", "hh_cov_other", "atomic_heldout", "d2_train_unseen_pair"]
print("run | updates | window | rc2U rc2S rc1U rc1S hhSS hhOther atom TT")
for r in out["runs"]:
    if "_RC_" not in r["run"]:
        continue
    for w in ("reported_saved5", "nearest_original5", "all_180k_200k"):
        v = r["windows"][w]["val"]
        print(r["run"], r["updates"], w, " ".join(f"{100*v[t]['ACC']:.2f}" for t in types))
print("\nX series last30: d2instance, d2pair, w4heldout ACC, w4heldout SEQ_ACC, updates")
for r in out["runs"]:
    if "_X_" in r["run"]:
        v = r["windows"]["last30"]["val"]
        print(r["run"], *(f"{100*v[t][m]:.2f}" for t,m in [("d2_train_unseen_instance","ACC"),("d2_train_unseen_pair","ACC"),("w4_heldout","ACC"),("w4_heldout","SEQ_ACC")]), r["updates"])
