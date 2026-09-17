"""
Role-controlled leak data for experiments P1 / P2 of docs/propose_experiment.md.

Starts from the ordinary skills dataset (the "F" base: E-co atomics + train-relation compositions,
built by ``build_dataset_skills``) and adds *role rows* that expose each held-out relation H in one
computational role with a controlled number of partner relations and a fixed coverage of facts:

  second slot:  <e_h><r_T><r_H><e_t>  with h = T^{-1}(b), t = H(b)   for b in S2_H (the covered bridges)
  first slot:   <e_h><r_H><r_T><e_t>  with t = T(H(h))               for h in S1_H (the covered inputs)

For every H, ``role_unseen`` bridges (U2_H) / inputs (U1_H) are NEVER used in that role by any training
row; ``role_reserved`` train relations (R_H) are never paired with H in training; the remaining train
relations form a fixed nested partner order P_H, and arm k uses P_H[:k]. Every covered fact receives
``role_exposure`` supervised answers in total (k partners x exposure/k repeats), so arms with different k
have the same number of answers per fact and the same total. H->H compositions are never generated.

Test types added (all exact instances unseen in training):
  rc2_newpair_role2_seen    reserved partners x S2_H   (cross-partner recombination of covered facts)
  rc2_newpair_role2_unseen  reserved partners x U2_H   (P1 main endpoint: transfer to uncovered role facts)
  rc2_trainpartner_unseen   training partners x U2_H   (auxiliary)
  rc1_newpair_role1_seen / rc1_newpair_role1_unseen / rc1_trainpartner_unseen   (first slot, symmetric)
  hh_cov_both               H1->H2 with h in S1_H1 and bridge in S2_H2 (P2 main mechanism readout)
  hh_cov_other              the remaining H1->H2 instances
The base test types atomic_*, d2_train_*, w*_* are kept; the base d2_mixed_* / d2_heldout are dropped
(superseded by the rc / hh types). meta.json records U/S sets, partner order, reserved partners, counts.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .builder_skills import _form, build_dataset_skills


def _row(rel, h: int, chain: Sequence[int]) -> Dict[str, str]:
    toks = [f"<e_{h}>"]
    x = h
    for j in chain:
        toks.append(f"<r_{j}>")
        x = rel[j][x]
    toks.append(f"<e_{x}>")
    return _form(toks)


def build_dataset_role_control(
    base_kwargs: Dict,
    role_slots: str = "second",          # none | first | second | both
    role_k: int = 4,                     # partners per active slot (must divide role_exposure)
    role_exposure: int = 8,              # supervised answers per covered fact (per active slot)
    role_unseen: int = 100,              # bridges / inputs per H kept out of the role
    role_reserved: int = 4,              # train relations per H never paired with H
    role_seed: int = 7,                  # U/S split and partner order (shared across arms)
    partner_seed: Optional[int] = None,  # if given, redraw ONLY the partner order / reserved partners from this seed
    hh_test_max: int = 32000,            # cap on exhaustive H->H test rows
) -> Dict:
    assert role_slots in ("none", "first", "second", "both"), role_slots
    assert role_exposure % max(1, role_k) == 0, "role_k must divide role_exposure"
    base = build_dataset_skills(**base_kwargs)
    meta = base["meta"]
    rel = meta["rel_map"]
    E = meta["num_entities"]
    T = list(meta["train_relations"])
    H = list(meta["heldout_relations"])
    inv = {j: {rel[j][h]: h for h in range(E)} for j in range(len(rel))}  # dense: permutations
    rng = np.random.default_rng(role_seed)
    prng = np.random.default_rng(partner_seed) if partner_seed is not None else None

    split: Dict[int, Dict] = {}
    for j in H:
        # draw order is fixed (U2/S2, U1/S1, partners) so U/S sets are identical for every partner_seed
        perm = rng.permutation(E).tolist()
        u2, s2 = sorted(perm[:role_unseen]), sorted(perm[role_unseen:])
        perm = rng.permutation(E).tolist()
        u1, s1 = sorted(perm[:role_unseen]), sorted(perm[role_unseen:])
        order = rng.permutation(T).tolist()
        if prng is not None:
            order = prng.permutation(T).tolist()
        split[j] = dict(U2=u2, S2=s2, U1=u1, S1=s1, reserved=sorted(order[:role_reserved]),
                        partners=order[role_reserved:])
    active = {"none": [], "first": ["first"], "second": ["second"], "both": ["first", "second"]}[role_slots]
    reps = role_exposure // max(1, role_k)

    role_rows: List[Dict[str, str]] = []
    n_role = {"first": 0, "second": 0}
    for j in H:
        sp = split[j]
        partners = sp["partners"][:role_k]
        if "second" in active:
            for b in sp["S2"]:
                for t in partners:
                    h = inv[t][b]
                    role_rows += [_row(rel, h, (t, j))] * reps
                    n_role["second"] += reps
        if "first" in active:
            for h in sp["S1"]:
                for t in partners:
                    role_rows += [_row(rel, h, (j, t))] * reps
                    n_role["first"] += reps

    # ---- tests
    tests: List[Dict] = []
    counts: Dict[str, int] = {}

    def add(name, rows):
        for r in rows:
            tests.append({**r, "type": name})
        counts[name] = len(rows)

    for j in H:
        sp = split[j]
        for slot in ("first", "second"):
            partners_train = sp["partners"][:role_k] if slot in active else []
            if slot == "second":
                seen = [_row(rel, inv[t][b], (t, j)) for t in sp["reserved"] for b in sp["S2"]]
                unseen = [_row(rel, inv[t][b], (t, j)) for t in sp["reserved"] for b in sp["U2"]]
                tp_unseen = [_row(rel, inv[t][b], (t, j)) for t in partners_train for b in sp["U2"]]
                pre = "rc2"
            else:
                seen = [_row(rel, h, (j, t)) for t in sp["reserved"] for h in sp["S1"]]
                unseen = [_row(rel, h, (j, t)) for t in sp["reserved"] for h in sp["U1"]]
                tp_unseen = [_row(rel, h, (j, t)) for t in partners_train for h in sp["U1"]]
                pre = "rc1"
            for r in seen:
                tests.append({**r, "type": f"{pre}_newpair_role_seen"})
            for r in unseen:
                tests.append({**r, "type": f"{pre}_newpair_role_unseen"})
            for r in tp_unseen:
                tests.append({**r, "type": f"{pre}_trainpartner_unseen"})
    for name in ("rc2_newpair_role_seen", "rc2_newpair_role_unseen", "rc2_trainpartner_unseen",
                 "rc1_newpair_role_seen", "rc1_newpair_role_unseen", "rc1_trainpartner_unseen"):
        counts[name] = sum(1 for r in tests if r["type"] == name)

    # exhaustive H->H, split by the predefined coverage sets
    hh_both, hh_other = [], []
    for j1 in H:
        s1 = set(split[j1]["S1"])
        for j2 in H:
            s2 = set(split[j2]["S2"])
            for h in range(E):
                b = rel[j1][h]
                (hh_both if (h in s1 and b in s2) else hh_other).append(_row(rel, h, (j1, j2)))
    if len(hh_both) + len(hh_other) > hh_test_max:
        keep = hh_test_max / (len(hh_both) + len(hh_other))
        hh_both = [r for r in hh_both if rng.uniform() < keep]
        hh_other = [r for r in hh_other if rng.uniform() < keep]
    add("hh_cov_both", hh_both)
    add("hh_cov_other", hh_other)

    keep_base = [r for r in base["test"] if not r["type"].startswith(("d2_mixed", "d2_heldout", "d3_heldout", "d3_mixed"))]
    for r in keep_base:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    train_rows = base["train"] + role_rows
    train_set = {r["target_text"] for r in train_rows}
    tests = keep_base + tests
    overlap = [r for r in tests if r["target_text"] in train_set
               and (r["type"].startswith(("rc", "hh", "d2_train_unseen")))]
    assert not overlap, f"{len(overlap)} composition test rows overlap training, e.g. {overlap[0]}"
    n_overlap_other = sum(1 for r in tests if r["target_text"] in train_set and not r["type"].startswith(("atomic", "d2_train_seen", "rc", "hh", "d2_train_unseen")))

    # ---- audits over the ACTUAL training rows
    import re
    tok = re.compile(r"<e_(\d+)>|<r_(\d+)>")
    role2_used, role1_used, pairs, hh = set(), set(), set(), 0
    for r in train_rows:
        t = [(int(a) if a else None, int(b) if b else None) for a, b in tok.findall(r["target_text"])]
        i = 0
        while i < len(t):
            h = t[i][0]; i += 1; chain = []
            while i < len(t) and t[i][1] is not None:
                chain.append(t[i][1]); i += 1
            i += 1
            if len(chain) >= 2:
                x = h
                for k_, jj in enumerate(chain):
                    (role1_used if k_ == 0 else role2_used).add((x, jj))
                    x = rel[jj][x]
                for a, c in zip(chain, chain[1:]):
                    pairs.add((a, c))
                    if a in set(H) and c in set(H):
                        hh += 1
    audit = {"hh_training_rows": hh}
    for j in H:
        sp = split[j]
        audit[f"r{j}"] = dict(
            role2_covered=sum(1 for b in range(E) if (b, j) in role2_used),
            role2_unseen_violations=sum(1 for b in sp["U2"] if (b, j) in role2_used),
            role1_covered=sum(1 for h in range(E) if (h, j) in role1_used),
            role1_unseen_violations=sum(1 for h in sp["U1"] if (h, j) in role1_used),
            reserved_violations=sum(1 for t in sp["reserved"] if (t, j) in pairs or (j, t) in pairs),
            partners_used=sorted({a for (a, c) in pairs if c == j} | {c for (a, c) in pairs if a == j}),
        )
    assert hh == 0
    for j in H:
        a = audit[f"r{j}"]
        assert a["role2_unseen_violations"] == 0 and a["role1_unseen_violations"] == 0 and a["reserved_violations"] == 0, a

    meta = dict(meta)
    meta.update(dict(
        task="role_control", role_slots=role_slots, role_k=role_k, role_exposure=role_exposure,
        role_unseen=role_unseen, role_reserved=role_reserved, role_seed=role_seed, partner_seed=partner_seed,
        role_split={str(j): split[j] for j in H}, role_rows=n_role, audit=audit,
        train_counts={**meta["train_counts"], "role_first": n_role["first"], "role_second": n_role["second"]},
        test_counts=counts, n_base_test_rows_in_train=n_overlap_other,
    ))
    return {"vocab": base["vocab"], "train": train_rows, "test": tests, "meta": meta,
            "counts": {"train": meta["train_counts"], "test": counts}}
