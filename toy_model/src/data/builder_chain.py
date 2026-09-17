"""
Strict shallow-training dataset "L" for docs/shallow_training_deep_composition_plan_2026-09-16.md (§3):
only width-2 (w2) and depth-2 (d2) training questions, no standalone atomic rows, no depth >= 3 anywhere in training.

World: the same E entities / R relations (random permutations) as the skills world with the same world seed
(SkillsWorld's draw order is reproduced, so rel_map is identical to data/skills_* for seed 42).

  w2  every atomic fact (x, r) appears exactly 8 times: 4 x in slot 1 and 4 x in slot 2, each time with a different
      partner fact, no fact paired with itself, no duplicate rows  -> 8F/2 = 4ER rows (40,000 at E=500, R=20).
      Built from 4 fixed-point-free permutations sigma_g of the F facts with pairwise-distinct pairs.
  d2  a directed relation graph with out-degree = in-degree = k (default 4), no self loops, strongly connected;
      every training edge (r, s) is used with ALL E starts  -> kRE rows (40,000).
  validation  20 relation pairs forming a 1-regular digraph disjoint from the training edges, 50 starts each
      (1,000 d2 questions), plus 20 new w2 pairings.  The remaining R^2 - kR - R ordered pairs (self pairs
      included) are the final "new adjacency" pool and are never used for training or selection.
  tests  per depth d in test_depths and category:
      all_seen   every adjacent relation pair is a training edge (d = 2 is a training-fit check)
      one_new    exactly one adjacency comes from the test pool (position balanced), the rest are training edges
      random     relations uniform at every step, chains containing a validation adjacency are resampled
      chains may repeat relations / entities and may cycle.

Questions are frozen as structured records (rendering into token strings is the protocol's job, see
data/chain_format.py). save_chain_dataset writes train_w2.json, train_d2.json, val_d2.json, val_w2.json,
test_chains.json, vocab.json, meta.json and audit.json (audit recomputed from the saved records).
"""
import collections
import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .builder_skills import SkillsWorld


def _derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def _strongly_connected(R: int, edges: set) -> bool:
    out = collections.defaultdict(list); inn = collections.defaultdict(list)
    for a, b in edges:
        out[a].append(b); inn[b].append(a)

    def reach(adj):
        seen = {0}; stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        return len(seen) == R
    return reach(out) and reach(inn)


def build_relation_graph(R: int, k: int, rng: np.random.Generator, max_tries: int = 10000) -> List[Tuple[int, int]]:
    """k-regular digraph (out = in = k), no self loops, unique edges, strongly connected: union of k derangements."""
    for _ in range(max_tries):
        edges = set()
        ok = True
        for _g in range(k):
            for _t in range(200):
                p = _derangement(rng, R)
                new = {(r, int(p[r])) for r in range(R)}
                if not (new & edges):
                    edges |= new; break
            else:
                ok = False; break
        if ok and _strongly_connected(R, edges):
            return sorted(edges)
    raise RuntimeError("could not build the relation graph")


def build_chain_dataset(num_entities: int = 500, num_relations: int = 20, world_seed: int = 42, k: int = 4,
                        graph_seed: int = 1, w2_seed: int = 2, val_seed: int = 3, test_seed: int = 4,
                        n_val_starts: int = 50, n_val_w2: int = 20, test_depths: Sequence[int] = (2, 3, 4, 8, 16, 32),
                        n_test_per_cell: int = 5000, w2_groups: int = 4, train_d3: bool = False) -> Dict:
    E, R = num_entities, num_relations
    world = SkillsWorld(E, R, 8, np.random.default_rng(world_seed))   # same draws as the skills world -> same rel_map
    rel = world.rel_map.astype(int)
    F = E * R
    fact = lambda f: (f % E, f // E)             # fact index -> (x, r)
    findex = lambda x, r: r * E + x

    # ---- d2 graph, validation pairs, test pool
    rng_g = np.random.default_rng([world_seed, graph_seed])
    edges = build_relation_graph(R, k, rng_g)
    edge_set = set(edges)
    rng_v = np.random.default_rng([world_seed, val_seed])
    val_pairs = None
    for _ in range(10000):
        p = _derangement(rng_v, R)
        cand = sorted((r, int(p[r])) for r in range(R))
        if not (set(cand) & edge_set):
            val_pairs = cand; break
    if val_pairs is None:
        # dense training graphs (large k): sample the derangement in the complement graph by randomised greedy matching
        for _ in range(100000):
            used = set(); pairs = []
            for r in rng_v.permutation(R):
                allowed = [s for s in range(R) if s != r and s not in used and (int(r), s) not in edge_set]
                if not allowed:
                    break
                s_ = int(rng_v.choice(allowed)); used.add(s_); pairs.append((int(r), s_))
            if len(pairs) == R:
                val_pairs = sorted(pairs); break
        else:
            raise RuntimeError("no validation derangement disjoint from the training graph")
    val_set = set(val_pairs)
    test_pool = sorted((a, b) for a in range(R) for b in range(R) if (a, b) not in edge_set and (a, b) not in val_set)
    assert len(test_pool) == R * R - len(edges) - R

    # ---- w2 rows
    rng_w = np.random.default_rng([world_seed, w2_seed])
    pairs_used = set(); w2_rows = []
    partners = [set() for _ in range(F)]          # unordered partner sets: every fact must get 8 distinct partners
    for g in range(w2_groups):
        sigma = _derangement(rng_w, F)
        # repair loop: a fact may not be paired (in either slot) with a fact it already met, nor with itself
        # violations: self pair, a partner already met in an earlier group, or a 2-cycle (f -> g -> f gives f the
        # same partner in both slots of this group)
        bad = lambda f: sigma[f] == f or int(sigma[f]) in partners[f] or sigma[int(sigma[f])] == f
        for _it in range(200000):
            viol = [f for f in range(F) if bad(f)]
            if not viol:
                break
            f = viol[int(rng_w.integers(len(viol)))]; f2 = int(rng_w.integers(F))
            sigma[f], sigma[f2] = sigma[f2], sigma[f]
        else:
            raise RuntimeError("could not repair the w2 group")
        for f in range(F):
            f2 = int(sigma[f]); partners[f].add(f2); partners[f2].add(f); pairs_used.add((f, f2))
            x1, r1 = fact(f); x2, r2 = fact(f2)
            w2_rows.append(dict(kind="w2", group=g, q1=[x1, r1], q2=[x2, r2]))
    # ---- d2 rows
    d2_rows = [dict(kind="d2", x=x, r=r, s=s) for (r, s) in edges for x in range(E)]
    # ---- optional d3 rows (CONTROL that leaves the strict w2/d2 constraint): every 2-edge walk r->s->t x all starts
    d3_rows = []
    if train_d3:
        succ = collections.defaultdict(list)
        for a, b in edges:
            succ[a].append(b)
        for (r, s_) in edges:
            for t in succ[s_]:
                for x in range(E):
                    d3_rows.append(dict(kind="d3", x=x, rels=[r, s_, t]))

    # ---- validation
    val_d2 = []
    for (r, s) in val_pairs:
        for x in rng_v.choice(E, n_val_starts, replace=False):
            val_d2.append(dict(kind="d2", x=int(x), r=r, s=s))
    val_w2 = []
    while len(val_w2) < n_val_w2:
        f1, f2 = int(rng_v.integers(F)), int(rng_v.integers(F))
        if f1 != f2 and f2 not in partners[f1]:
            partners[f1].add(f2); partners[f2].add(f1); pairs_used.add((f1, f2)); x1, r1 = fact(f1); x2, r2 = fact(f2)
            val_w2.append(dict(kind="w2", q1=[x1, r1], q2=[x2, r2]))

    # ---- tests
    rng_t = np.random.default_rng([world_seed, test_seed])
    out_nb = collections.defaultdict(list); in_nb = collections.defaultdict(list)
    for a, b in edges:
        out_nb[a].append(b); in_nb[b].append(a)

    def walk_forward(start_rel, steps):
        chain = [start_rel]
        for _ in range(steps):
            chain.append(int(rng_t.choice(out_nb[chain[-1]])))
        return chain

    def walk_backward(end_rel, steps):
        chain = [end_rel]
        for _ in range(steps):
            chain.append(int(rng_t.choice(in_nb[chain[-1]])))
        return chain[::-1]

    def states(x, rels):
        st = [x]
        for r in rels:
            st.append(int(rel[r][st[-1]]))
        return st

    tests = []
    for d in test_depths:
        for cat in ("all_seen", "one_new", "random"):
            seen_keys = set()
            n_target = n_test_per_cell
            tries = 0; n_cell = 0
            while n_cell < n_target and tries < 50 * n_target:
                tries += 1
                if cat == "all_seen":
                    rels = walk_forward(int(rng_t.integers(R)), d - 1); new_pos = []
                elif cat == "one_new":
                    pos = int(rng_t.integers(d - 1))                      # adjacency index 0..d-2
                    a, b = test_pool[int(rng_t.integers(len(test_pool)))]
                    left = walk_backward(a, pos)                          # ends with a
                    right = walk_forward(b, d - 2 - pos)                  # starts with b
                    rels = left + right; new_pos = [pos]
                else:
                    rels = [int(v) for v in rng_t.integers(R, size=d)]
                    adj = list(zip(rels, rels[1:]))
                    if any(p in val_set for p in adj):
                        continue
                    new_pos = [i for i, p in enumerate(adj) if p not in edge_set]
                x = int(rng_t.integers(E))
                key = (x, tuple(rels))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                st = states(x, rels)
                tests.append(dict(id=len(tests), cat=cat, d=d, x=x, rels=rels, states=st[1:], new_adjacencies=new_pos,
                                  repeated_relation=len(set(rels)) < len(rels), revisits_entity=len(set(st)) < len(st)))
                n_cell += 1
    vocab = ["<pad>"] + [f"<e_{i}>" for i in range(E)] + [f"<r_{j}>" for j in range(R)] + ["Q", "ANS", "END"]
    meta = dict(task="chain", num_entities=E, num_relations=R, world_seed=world_seed, k=k, graph_seed=graph_seed,
                w2_seed=w2_seed, val_seed=val_seed, test_seed=test_seed, rel_map=rel.tolist(), train_edges=edges,
                val_pairs=val_pairs, test_pool=test_pool, test_depths=list(test_depths), n_test_per_cell=n_test_per_cell,
                counts=dict(w2=len(w2_rows), d2=len(d2_rows), d3=len(d3_rows), val_d2=len(val_d2), val_w2=len(val_w2), tests=len(tests),
                            unique_atomic_facts=F, train_relation_pairs=len(edges), test_pool_pairs=len(test_pool)),
                train_d3=train_d3, note_d3="all_seen|d3 test chains are 2-edge walks on the training graph and therefore TRAINED when train_d3" if train_d3 else "")
    return dict(meta=meta, vocab=vocab, train_w2=w2_rows, train_d2=d2_rows, train_d3=d3_rows, val_d2=val_d2, val_w2=val_w2, tests=tests)


def audit_chain_dataset(ds: Dict) -> Dict:
    """Recompute the design guarantees from the saved records."""
    m = ds["meta"]; E, R = m["num_entities"], m["num_relations"]; rel = m["rel_map"]
    edges = {tuple(e) for e in m["train_edges"]}; val = {tuple(p) for p in m["val_pairs"]}; pool = {tuple(p) for p in m["test_pool"]}
    slot1 = collections.Counter(); slot2 = collections.Counter(); partners = collections.defaultdict(set); rows = set(); self_pairs = 0
    for r in ds["train_w2"]:
        f1 = tuple(r["q1"]); f2 = tuple(r["q2"])
        slot1[f1] += 1; slot2[f2] += 1; partners[f1].add(f2); partners[f2].add(f1)
        rows.add((f1, f2)); self_pairs += int(f1 == f2)
    facts = [(x, r) for r in range(R) for x in range(E)]
    first_role = collections.Counter(); second_role = collections.Counter(); d2_pairs = collections.Counter()
    for q in ds["train_d2"]:
        first_role[(q["x"], q["r"])] += 1; second_role[(rel[q["r"]][q["x"]], q["s"])] += 1; d2_pairs[(q["r"], q["s"])] += 1
    out_deg = collections.Counter(a for a, b in edges); in_deg = collections.Counter(b for a, b in edges)
    test_checks = collections.defaultdict(lambda: dict(n=0, bad=0))
    for t in ds["tests"]:
        adj = list(zip(t["rels"], t["rels"][1:])); key = f"{t['cat']}|d{t['d']}"
        test_checks[key]["n"] += 1
        if t["cat"] == "all_seen":
            ok = all(p in edges for p in adj)
        elif t["cat"] == "one_new":
            ok = sum(p in pool for p in adj) == 1 and all((p in edges) or (p in pool) for p in adj) and not any(p in val for p in adj)
        else:
            ok = not any(p in val for p in adj)
        test_checks[key]["bad"] += int(not ok)
    audit = dict(
        w2=dict(rows=len(ds["train_w2"]), unique_rows=len(rows), self_pairs=self_pairs,
                slot1_per_fact=[min(slot1[f] for f in facts), max(slot1[f] for f in facts)],
                slot2_per_fact=[min(slot2[f] for f in facts), max(slot2[f] for f in facts)],
                distinct_partners_per_fact=[min(len(partners[f]) for f in facts), max(len(partners[f]) for f in facts)]),
        d2=dict(rows=len(ds["train_d2"]), edges=len(edges), self_loops=sum(a == b for a, b in edges),
                out_degree=[min(out_deg.values()), max(out_deg.values())], in_degree=[min(in_deg.values()), max(in_deg.values())],
                rows_per_edge=[min(d2_pairs.values()), max(d2_pairs.values())],
                first_role_per_fact=[min(first_role[f] for f in facts), max(first_role[f] for f in facts)],
                second_role_per_fact=[min(second_role[f] for f in facts), max(second_role[f] for f in facts)],
                strongly_connected=_strongly_connected(R, edges)),
        splits=dict(val_pairs=len(val), val_disjoint_from_train=not (val & edges), val_one_regular=(len({a for a, b in val}) == R and len({b for a, b in val}) == R),
                    test_pool=len(pool), pool_disjoint=not (pool & (edges | val)), pool_plus_train_plus_val=len(pool) + len(edges) + len(val)),
        tests={k: v for k, v in sorted(test_checks.items())},
    )
    audit["passed"] = bool(
        audit["w2"]["unique_rows"] == audit["w2"]["rows"] and self_pairs == 0 and audit["w2"]["slot1_per_fact"] == [4, 4]
        and audit["w2"]["slot2_per_fact"] == [4, 4] and audit["w2"]["distinct_partners_per_fact"][0] == 8
        and audit["d2"]["self_loops"] == 0 and audit["d2"]["out_degree"] == [m["k"], m["k"]] and audit["d2"]["in_degree"] == [m["k"], m["k"]]
        and audit["d2"]["rows_per_edge"] == [E, E] and audit["d2"]["first_role_per_fact"] == [m["k"], m["k"]]
        and audit["d2"]["second_role_per_fact"] == [m["k"], m["k"]] and audit["d2"]["strongly_connected"]
        and audit["splits"]["val_disjoint_from_train"] and audit["splits"]["val_one_regular"] and audit["splits"]["pool_disjoint"]
        and audit["splits"]["pool_plus_train_plus_val"] == R * R and all(v["bad"] == 0 for v in test_checks.values()))
    return audit


def save_chain_dataset(output_dir: str, ds: Dict) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    names = ["train_w2", "train_d2", "train_d3", "val_d2", "val_w2", "tests", "vocab"]
    for n in names:
        with open(os.path.join(output_dir, f"{n}.json"), "w") as f:
            json.dump(ds[n], f)
    hashes = {n: hashlib.sha256(open(os.path.join(output_dir, f"{n}.json"), "rb").read()).hexdigest()[:16] for n in names}
    ds["meta"]["hashes"] = hashes
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(ds["meta"], f)
    audit = audit_chain_dataset(ds)
    with open(os.path.join(output_dir, "audit.json"), "w") as f:
        json.dump(audit, f, indent=1)
    return audit
