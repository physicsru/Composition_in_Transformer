"""Freeze the evaluation protocol of the fifth batch (docs/experiments_latent_batch2_20runs.md §7.3, §8.3) BEFORE training:
the rule + hash of every test layout, the monitoring (first 500 per cell) / confirmation (last 4500) id split with a
disjointness check, and the 500 suffix-change pairs (same start and prefix, one relation changed in the second half,
different gold answers; depth 3 / 8 / 32 / 128 x 125).  Writes <data_dir>/latent_eval_manifest.json.

    python generate_latent_eval.py --data_dir ../data/chain_loop
"""
import argparse, collections, hashlib, json, os

import numpy as np

from data.latent_format import LAYOUT_SEED, frozen_slots


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--small", action="store_true", help="smoke datasets: no 500 / 4500 split check, suffix pairs from whatever depths exist")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data_dir, "meta.json"))); rel = meta["rel_map"]; R = meta["num_relations"]
    val = {tuple(p) for p in meta["val_pairs"]}
    tests = json.load(open(os.path.join(args.data_dir, "tests.json")))
    h = hashlib.sha256(); cells = collections.defaultdict(list)
    for t in tests:
        h.update(frozen_slots(1, t["id"], t["d"]).astype(np.int16).tobytes()); cells[f"{t['cat']}|d{t['d']}"].append(t["id"])
    split = {}
    for k, ids in cells.items():
        mon, conf = ids[:500], ids[500:]
        assert not (set(mon) & set(conf)) and (args.small or (len(mon) == 500 and len(conf) == 4500)), k
        split[k] = dict(monitor_first=mon[0], monitor_last=mon[-1], confirm_first=conf[0] if conf else None, confirm_last=conf[-1] if conf else None, n_monitor=len(mon), n_confirm=len(conf),
                        monitor_hash=hashlib.sha256(json.dumps(mon).encode()).hexdigest()[:16])
    rng = np.random.default_rng([LAYOUT_SEED, 40])
    pairs = []
    for d in ((3, 8, 32, 128) if not args.small else (3, 8)):
        base = [t for t in tests if t["cat"] == "random" and t["d"] == d][:500]; n = 0; want = 125 if not args.small else 10
        for t in base:
            if n == want:
                break
            for _try in range(50):
                p = int(rng.integers(d // 2, d)); r_new = int(rng.integers(R))
                rels = list(t["rels"])
                if r_new == rels[p]:
                    continue
                rels[p] = r_new
                adj = list(zip(rels, rels[1:]))
                if any(a in val for a in adj):
                    continue
                cur = t["x"]; st = []
                for r in rels:
                    cur = rel[r][cur]; st.append(cur)
                if st[-1] == t["states"][-1]:
                    continue
                pairs.append(dict(pair=len(pairs), d=d, base_id=t["id"], x=t["x"], changed_pos=p, rels_a=t["rels"], gold_a=t["states"][-1], rels_b=rels, gold_b=st[-1]))
                n += 1; break
        assert n == want, (d, n)
    out = dict(layout_rule="slots = sort(default_rng([LAYOUT_SEED, kind, index, variant]).choice(128, d, replace=False)); kind 1 tests (index = test id), "
                           "2 validation d2, 3 atomic (index = r * E + x), 4 suffix pairs (both chains share the layout), 5 extra layouts (variant 1..), 6 validation w2",
               layout_seed=LAYOUT_SEED, test_layout_hash=h.hexdigest()[:16], n_tests=len(tests), tests_hash=meta["hashes"]["tests"],
               split=split, split_rule="per (category, depth) cell in tests.json order: first 500 = monitoring / diagnostics, last 4500 = confirmation subset",
               suffix_pairs=pairs)
    json.dump(out, open(os.path.join(args.data_dir, "latent_eval_manifest.json"), "w"))
    print("test layouts hash", out["test_layout_hash"], "| cells", len(split), "| suffix pairs", collections.Counter(p["d"] for p in pairs))


if __name__ == "__main__":
    main()
