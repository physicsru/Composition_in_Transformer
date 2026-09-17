"""
Script to generate dataset for training.

Two tasks:
  functor (default): the paper's synthetic knowledge graph with atomic / compositional /
                     analogical facts (data.builder.build_dataset_with_functor).
  skills:            relations as atomic skills; held-out relations are never composed in
                     training (data.builder_skills.build_dataset_skills). Select with
                     `data.task: skills` in the config or `--task skills`.
"""

import argparse
import os

import yaml

from data.builder import build_dataset_with_functor, save_dataset
from data.builder_skills import build_dataset_skills, save_dataset_skills
from data.builder_role_control import build_dataset_role_control


def _pick(cli_value, cfg, key, default):
    """CLI value if given, else config value, else default."""
    return cli_value if cli_value is not None else cfg.get(key, default)


def skills_kwargs(args, data_config):
    if args.comp_depths is None:
        comp_depths = data_config.get("comp_depths", [2])
    else:
        comp_depths = [int(x) for x in str(args.comp_depths).split(",") if x.strip()]
    kwargs = dict(
        num_entities=_pick(args.num_entities, data_config, "num_entities", 500),
        num_relations=_pick(args.num_relations, data_config, "num_relations", 20),
        num_heldout_relations=_pick(args.num_heldout_relations, data_config, "num_heldout_relations", 8),
        seed=_pick(args.seed, data_config, "seed", 42),
        out_degree=_pick(args.out_degree, data_config, "out_degree", None),
        group_size=_pick(args.group_size, data_config, "group_size", 2),
        group_size_max=_pick(args.group_size_max, data_config, "group_size_max", None),
        grouped_frac=_pick(args.grouped_frac, data_config, "grouped_frac", 1.0),
        heldout_grouped_frac=_pick(args.heldout_grouped_frac, data_config, "heldout_grouped_frac", None),
        heldout_slot=_pick(args.heldout_slot, data_config, "heldout_slot", "any"),
        partner=_pick(args.partner, data_config, "partner", "any"),
        group_passes=_pick(args.group_passes, data_config, "group_passes", 1),
        single_passes=_pick(args.single_passes, data_config, "single_passes", 1),
        always_singles=bool(_pick(args.always_singles, data_config, "always_singles", False)),
        comp_depths=comp_depths,
        n_comp=_pick(args.n_comp, data_config, "n_comp", 20000),
        comp_relations=_pick(args.comp_relations, data_config, "comp_relations", None),
        comp_pair_holdout_frac=_pick(args.comp_pair_holdout_frac, data_config, "comp_pair_holdout_frac", 0.2),
        n_mixed=_pick(args.n_mixed, data_config, "n_mixed", 0),
        heldout_comp_frac=_pick(args.heldout_comp_frac, data_config, "heldout_comp_frac", 0.0),
        heldout_leak_pairs=_pick(args.heldout_leak_pairs, data_config, "heldout_leak_pairs", 0),
        heldout_leak_per_rel=_pick(args.heldout_leak_per_rel, data_config, "heldout_leak_per_rel", 0),
        heldout_leak_within=_pick(args.heldout_leak_within, data_config, "heldout_leak_within", 0.9),
        heldout_leak_slot=_pick(args.heldout_leak_slot, data_config, "heldout_leak_slot", "second"),
        n_test_per_type=_pick(args.n_test_per_type, data_config, "n_test_per_type", 500),
        test_depths=data_config.get("test_depths", [2, 3]),
        test_widths=data_config.get("test_widths", [2, 3, 4]),
    )
    return kwargs


def generate_skills(args, data_config, role_control=False):
    kwargs = skills_kwargs(args, data_config)
    print("Generating skills dataset..." if not role_control else "Generating role-control dataset...")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    if role_control:
        rc = dict(
            role_slots=_pick(args.role_slots, data_config, "role_slots", "second"),
            role_k=_pick(args.role_k, data_config, "role_k", 4),
            role_exposure=_pick(args.role_exposure, data_config, "role_exposure", 8),
            role_unseen=_pick(args.role_unseen, data_config, "role_unseen", 100),
            role_reserved=_pick(args.role_reserved, data_config, "role_reserved", 4),
            role_seed=_pick(args.role_seed, data_config, "role_seed", 7),
            partner_seed=_pick(args.partner_seed, data_config, "partner_seed", None),
        )
        for k, v in rc.items():
            print(f"  {k}: {v}")
        ds = build_dataset_role_control(kwargs, **rc)
    else:
        ds = build_dataset_skills(**kwargs)

    output_dir = args.output_dir or data_config.get("output_dir") or (
        f"data/skills.{kwargs['num_entities']}.{kwargs['num_relations']}.{kwargs['num_heldout_relations']}"
    )
    save_dataset_skills(output_dir, ds)

    meta = ds["meta"]
    print("\nDataset summary:")
    print(f"  train relations: {meta['train_relations']}")
    print(f"  held-out relations: {meta['heldout_relations']}")
    print(f"  composed relations: {meta['comp_relations']}")
    print(f"  train pairs / held-out pairs: {len(meta['train_pairs'])} / {len(meta['heldout_pairs'])}")
    print(f"  train rows: {meta['train_counts']}")
    if meta.get("task") == "role_control":
        print(f"  role-control audit: {meta['audit']}")
    print("  test rows per type:")
    for t, n in meta["test_counts"].items():
        print(f"    {t:>26}: {n}")


def generate_functor(args, data_config):
    # Override with command line arguments
    num_entities = args.num_entities or data_config.get("num_entities", 10)
    num_relations = args.num_relations or data_config.get("num_relations", 10000)
    sub_size = args.sub_size or data_config.get("sub_size", num_entities // 2)
    seed = args.seed or data_config.get("seed", 42)
    
    atomic_ood_ratio = data_config.get("atomic_ood_ratio", 0.0)
    compositional_ood_ratio = data_config.get("compositional_ood_ratio", 0.1)
    analogical_ood_ratio = data_config.get("analogical_ood_ratio", 0.4)
    include_f_inverse = data_config.get("include_f_inverse", False)
    duplicate_relation = data_config.get("duplicate_relation", False)
    
    # Generate dataset
    print("Generating dataset...")
    print(f"  num_entities: {num_entities}")
    print(f"  num_relations: {num_relations}")
    print(f"  sub_size: {sub_size}")
    print(f"  seed: {seed}")
    
    res = build_dataset_with_functor(
        num_entities, num_relations,
        sub_size=sub_size,
        atomic_ood_ratio=atomic_ood_ratio,
        compositional_ood_ratio=compositional_ood_ratio,
        analogical_ood_ratio=analogical_ood_ratio,
        seed=seed,
        include_f_inverse=include_f_inverse,
        duplicate_relation=duplicate_relation
    )
    
    (entities, relations,
     id_atomic_facts, ood_atomic_facts,
     id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts,
     id_analogical_facts, ood_analogical_facts) = res
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = data_config.get(
            "output_dir",
            f"data/composition_functor.{num_entities}.{num_relations}.{sub_size}"
        )
    
    # Save dataset
    save_dataset(
        output_dir,
        entities, relations,
        id_atomic_facts, ood_atomic_facts,
        id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts,
        id_analogical_facts, ood_analogical_facts,
    )
    
    # Print summary
    print("\nDataset summary:")
    print(f"  #entities: {len(entities)}")
    print(f"  #relations: {len(relations)}")
    print(f"  ID atomics: {len(id_atomic_facts)}")
    print(f"  OOD atomics: {len(ood_atomic_facts)}")
    print(f"  ID compositional: {len(id_compositional_facts)}")
    print(f"  near OOD compositional: {len(near_ood_compositional_facts)}")
    print(f"  far OOD compositional: {len(far_ood_compositional_facts)}")
    print(f"  ID analogical: {len(id_analogical_facts)}")
    print(f"  OOD analogical: {len(ood_analogical_facts)}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset for Emergent Analogy")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--task", type=str, default=None, choices=["functor", "skills", "role_control"],
                        help="Override data.task from the config (default: functor)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--num_entities", type=int, default=None)
    parser.add_argument("--num_relations", type=int, default=None)
    parser.add_argument("--sub_size", type=int, default=None, help="functor task only")
    parser.add_argument("--seed", type=int, default=None)
    # skills task knobs (see data/builder_skills.py)
    parser.add_argument("--num_heldout_relations", type=int, default=None)
    parser.add_argument("--out_degree", type=int, default=None,
                        help="sparse relations: each entity gets this many random relations (default: dense permutations)")
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--group_size_max", type=int, default=None,
                        help="variable width: draw each group's size from [group_size, group_size_max]")
    parser.add_argument("--grouped_frac", type=float, default=None)
    parser.add_argument("--heldout_grouped_frac", type=float, default=None)
    parser.add_argument("--heldout_slot", type=str, default=None, choices=["any", "first", "last"])
    parser.add_argument("--partner", type=str, default=None, choices=["any", "train", "heldout"])
    parser.add_argument("--group_passes", type=int, default=None)
    parser.add_argument("--always_singles", type=int, default=None, choices=[0, 1],
                        help="1 = one single-task row per atomic fact in addition to any groups")
    parser.add_argument("--single_passes", type=int, default=None,
                        help="copies of each single-task row (frequency control for ungrouped facts)")
    parser.add_argument("--comp_depths", type=str, default=None, help="comma-separated, e.g. 2,3")
    parser.add_argument("--n_comp", type=int, default=None)
    parser.add_argument("--comp_relations", type=int, default=None)
    parser.add_argument("--comp_pair_holdout_frac", type=float, default=None)
    parser.add_argument("--heldout_comp_frac", type=float, default=None,
                        help="leak: fraction of held-out-involving depth-2 instances put into training")
    parser.add_argument("--heldout_leak_pairs", type=int, default=None,
                        help="concentrated leak: number of held-out-involving relation pairs to leak")
    parser.add_argument("--heldout_leak_per_rel", type=int, default=None,
                        help="balanced concentrated leak: this many qualifying pairs per held-out relation")
    parser.add_argument("--heldout_leak_within", type=float, default=None,
                        help="fraction of each chosen pair's instances put into training (default 0.9)")
    parser.add_argument("--heldout_leak_slot", type=str, default=None, choices=["any", "first", "second"],
                        help="which pairs qualify: held-out relation in the second hop (default), first, or any")
    parser.add_argument("--n_mixed", type=int, default=None,
                        help="rows pairing one held-out atomic task with one training composition")
    parser.add_argument("--n_test_per_type", type=int, default=None)
    # role_control task knobs (see data/builder_role_control.py; P1/P2 of docs/propose_experiment.md)
    parser.add_argument("--role_slots", type=str, default=None, choices=["none", "first", "second", "both"])
    parser.add_argument("--role_k", type=int, default=None, help="training partners per active slot")
    parser.add_argument("--role_exposure", type=int, default=None, help="supervised answers per covered role fact")
    parser.add_argument("--role_unseen", type=int, default=None, help="role-unseen bridges/inputs per held-out relation")
    parser.add_argument("--role_reserved", type=int, default=None, help="reserved (never-paired) test partners per held-out relation")
    parser.add_argument("--role_seed", type=int, default=None)
    parser.add_argument("--partner_seed", type=int, default=None,
                        help="redraw only the per-H partner order / reserved partners from this seed (U/S sets stay those of role_seed)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config.get("data", {})
    task = args.task or data_config.get("task", "functor")
    if task == "skills":
        generate_skills(args, data_config)
    elif task == "role_control":
        generate_skills(args, data_config, role_control=True)
    else:
        generate_functor(args, data_config)


if __name__ == "__main__":
    main()
