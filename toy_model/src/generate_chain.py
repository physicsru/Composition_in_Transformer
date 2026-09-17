"""Generate the strict w2/d2 chain dataset (data/builder_chain.py) and audit it.

    python generate_chain.py --output_dir ../data/chain_L [--num_entities 500 --num_relations 20 --world_seed 42 --k 4]
"""
import argparse
import json

from data.builder_chain import build_chain_dataset, save_chain_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_entities", type=int, default=500)
    ap.add_argument("--num_relations", type=int, default=20)
    ap.add_argument("--world_seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--graph_seed", type=int, default=1)
    ap.add_argument("--w2_seed", type=int, default=2)
    ap.add_argument("--val_seed", type=int, default=3)
    ap.add_argument("--test_seed", type=int, default=4)
    ap.add_argument("--test_depths", default="2,3,4,8,16,32")
    ap.add_argument("--n_test_per_cell", type=int, default=5000)
    ap.add_argument("--train_d3", type=int, default=0, choices=[0, 1], help="CONTROL: also generate depth-3 training rows (leaves the strict w2/d2 constraint)")
    args = ap.parse_args()
    ds = build_chain_dataset(args.num_entities, args.num_relations, args.world_seed, args.k, args.graph_seed, args.w2_seed,
                             args.val_seed, args.test_seed, test_depths=[int(x) for x in args.test_depths.split(",")],
                             n_test_per_cell=args.n_test_per_cell, train_d3=bool(args.train_d3))
    audit = save_chain_dataset(args.output_dir, ds)
    print(json.dumps(ds["meta"]["counts"]))
    print(json.dumps({k: v for k, v in audit.items() if k != "tests"}, indent=1))
    print("tests:", {k: v for k, v in audit["tests"].items()})
    print("AUDIT", "PASSED" if audit["passed"] else "FAILED")
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
