"""
Dataset builder for the *skills* task: compositional generalization over relations.

Vocabulary: entities ``<e_i>`` and relations ``<r_j>``. Every relation is a random
permutation of the entity set, so ``r(h)`` is defined for every ``h`` and
``r2(r1(h))`` for every pair, and no relation is the composition of two others
(a random set of permutations is not closed under composition).

Relations are split into TRAIN and HELD-OUT. Held-out relations appear in training
only as atomic facts (single rows, or inside width-k "co-occurrence" rows with other
independent atomic tasks); they never appear inside a composition. Compositions in
training use train relations only. This mirrors the W2D2 setting of the
compositional-generalization project: atomics + width-2 co-occurrence over all
skills + depth-2 compositions over train skills.

Row formats (tokens concatenated, no separators):

    atomic    <e_h><r_j><e_t>                                   t = r_j(h)
    width-k   <e_h1><r_j1><e_t1><e_h2><r_j2><e_t2> ...          k independent atomic tasks
    depth-k   <e_h><r_j1>...<r_jk><e_t>                         t = r_jk(...r_j1(h))

The loss is taken on every entity token that directly follows a relation token
(``data.dataset.answer_positions``), so a width-k row has k supervised positions
and a depth-k row has one.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Task = Tuple[int, int]  # (head entity, relation)


def _form(tokens: List[str]) -> Dict[str, str]:
    return {"input_text": "".join(tokens[:-1]), "target_text": "".join(tokens)}


class SkillsWorld:
    """The sampled world: relations as permutations, plus row rendering."""

    def __init__(self, num_entities: int, num_relations: int, num_heldout_relations: int,
                 rng: np.random.Generator, out_degree: Optional[int] = None):
        assert 0 < num_heldout_relations < num_relations, \
            "need at least one train relation and one held-out relation"
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.out_degree = out_degree
        self.entities = [f"<e_{i}>" for i in range(num_entities)]
        self.relations = [f"<r_{j}>" for j in range(num_relations)]
        perm = rng.permutation(num_relations).tolist()
        self.heldout_rel: List[int] = sorted(perm[:num_heldout_relations])
        self.train_rel: List[int] = sorted(perm[num_heldout_relations:])
        self.heldout_set = set(self.heldout_rel)
        if out_degree is None:
            # dense: rel_map[j, h] = r_j(h); each row is a permutation of the entities
            self.rel_map = np.stack([rng.permutation(num_entities) for _ in range(num_relations)])
        else:
            # sparse (Wang et al. 2024): every entity has out_degree distinct relations, each to a
            # uniformly random tail; a relation is a partial many-to-one function (-1 = undefined)
            assert 0 < out_degree <= num_relations
            self.rel_map = np.full((num_relations, num_entities), -1, dtype=np.int64)
            for h in range(num_entities):
                for j in rng.choice(num_relations, size=out_degree, replace=False):
                    self.rel_map[j, h] = int(rng.integers(num_entities))
        # rels_of[h] = relations defined on h (all of them in the dense case)
        self.rels_of: List[List[int]] = [
            [j for j in range(num_relations) if self.rel_map[j, h] >= 0] for h in range(num_entities)
        ]

    def defined(self, j: int, h: int) -> bool:
        return self.rel_map[j, h] >= 0

    def heads_of(self, j: int) -> List[int]:
        return [h for h in range(self.num_entities) if self.rel_map[j, h] >= 0]

    def is_heldout(self, j: int) -> bool:
        return j in self.heldout_set

    def apply(self, chain: Sequence[int], h: int) -> int:
        """Follow the chain from h; returns -1 if some hop is undefined (sparse relations only)."""
        for j in chain:
            if h < 0 or self.rel_map[j, h] < 0:
                return -1
            h = int(self.rel_map[j, h])
        return h

    def atomic_row(self, tasks: Sequence[Task]) -> Dict[str, str]:
        toks: List[str] = []
        for h, j in tasks:
            toks += [self.entities[h], self.relations[j], self.entities[int(self.rel_map[j, h])]]
        return _form(toks)

    def task_row(self, tasks: Sequence[Tuple[int, Sequence[int]]]) -> Dict[str, str]:
        """Several independent tasks in one row; each task is (h, chain) and outputs only its answer."""
        toks: List[str] = []
        for h, chain in tasks:
            toks += [self.entities[h]] + [self.relations[j] for j in chain] + [self.entities[self.apply(chain, h)]]
        return _form(toks)

    def chain_row(self, h: int, chain: Sequence[int]) -> Dict[str, str]:
        toks = [self.entities[h]] + [self.relations[j] for j in chain] + [self.entities[self.apply(chain, h)]]
        return _form(toks)


def _pair_within(lst: List[Task]) -> Tuple[List[List[Task]], List[Task]]:
    n_full = len(lst) - len(lst) % 2
    return [lst[i:i + 2] for i in range(0, n_full, 2)], lst[n_full:]


def _make_groups(items: List[Task], k: int, world: SkillsWorld, rng: np.random.Generator,
                 heldout_slot: str, partner: str, k_max: Optional[int] = None
                 ) -> Tuple[List[List[Task]], List[Task]]:
    """
    Partition atomic tasks into groups of size k (or of sizes drawn uniformly from
    [k, k_max] per group when k_max is given). Returns (groups, leftovers).

    ``heldout_slot`` and ``partner`` constrain groups that contain a held-out task and are
    only supported for a fixed k == 2:
      heldout_slot: any | first | last   -- where the held-out task sits in a mixed pair
      partner:      any | train | heldout -- what a held-out task is paired with
    """
    items = list(items)
    if heldout_slot == "any" and partner == "any":
        rng.shuffle(items)
        if k_max is None or k_max <= k:
            n_full = len(items) - len(items) % k
            return [items[i:i + k] for i in range(0, n_full, k)], items[n_full:]
        groups, i = [], 0
        while i < len(items):
            size = int(rng.integers(k, k_max + 1))
            if i + size > len(items):
                size = len(items) - i  # last group takes what is left (still >= 1 task)
            groups.append(items[i:i + size])
            i += size
        return groups, []
    assert k_max is None or k_max <= k, "heldout_slot / partner constraints need a fixed group size"

    assert k == 2, "heldout_slot / partner constraints are only implemented for group_size 2"
    H = [f for f in items if world.is_heldout(f[1])]
    T = [f for f in items if not world.is_heldout(f[1])]
    rng.shuffle(H)
    rng.shuffle(T)

    def orient(hf: Task, tf: Task) -> List[Task]:
        if heldout_slot == "first":
            return [hf, tf]
        if heldout_slot == "last":
            return [tf, hf]
        return [hf, tf] if rng.uniform() < 0.5 else [tf, hf]

    groups: List[List[Task]] = []
    leftovers: List[Task] = []
    if partner == "train":
        n = min(len(H), len(T))
        groups += [orient(hf, tf) for hf, tf in zip(H[:n], T[:n])]
        leftovers += H[n:]
        g, l = _pair_within(T[n:])
        groups += g
        leftovers += l
    elif partner == "heldout":
        for lst in (H, T):
            g, l = _pair_within(lst)
            groups += g
            leftovers += l
    else:  # partner == "any", slot constrained
        allf = H + T
        rng.shuffle(allf)
        g, leftovers = _pair_within(allf)
        for pair in g:
            a_ho, b_ho = world.is_heldout(pair[0][1]), world.is_heldout(pair[1][1])
            if a_ho != b_ho:
                hf, tf = (pair[0], pair[1]) if a_ho else (pair[1], pair[0])
                pair[:] = orient(hf, tf)
        groups += g
    return groups, leftovers


def _collect(n: int, sampler, key_fn, exclude=frozenset(), max_tries: Optional[int] = None) -> list:
    """Draw up to n distinct items from sampler() (None = rejected), skipping keys in exclude."""
    rows, seen = [], set()
    tries, max_tries = 0, max_tries or max(50 * n, 1000)
    while len(rows) < n and tries < max_tries:
        tries += 1
        item = sampler()
        if item is None:
            continue
        key = key_fn(item)
        if key in seen or key in exclude:
            continue
        seen.add(key)
        rows.append(item)
    return rows


def build_dataset_skills(
    num_entities: int,
    num_relations: int,
    num_heldout_relations: int,
    seed: int = 42,
    out_degree: Optional[int] = None,
    # width (co-occurrence) rows
    group_size: int = 2,
    group_size_max: Optional[int] = None,
    grouped_frac: float = 1.0,
    heldout_grouped_frac: Optional[float] = None,
    heldout_slot: str = "any",
    partner: str = "any",
    group_passes: int = 1,
    single_passes: int = 1,
    always_singles: bool = False,
    # composition rows (train relations only)
    comp_depths: Sequence[int] = (2,),
    n_comp: int = 20000,
    comp_relations: Optional[int] = None,
    comp_pair_holdout_frac: float = 0.2,
    n_mixed: int = 0,
    heldout_comp_frac: float = 0.0,
    heldout_leak_pairs: int = 0,
    heldout_leak_per_rel: int = 0,
    heldout_leak_within: float = 0.9,
    heldout_leak_slot: str = "second",
    # test sets
    n_test_per_type: int = 500,
    test_depths: Sequence[int] = (2, 3),
    test_widths: Sequence[int] = (2, 3, 4),
) -> Dict:
    """
    Build the skills dataset.

    Args:
        num_entities, num_relations: sizes of E and R. By default every relation is a random
            permutation of E (dense, total). With out_degree=k each entity instead has k random
            distinct relations with random tails (Wang et al. 2024): relations are sparse partial
            functions, atomic facts = num_entities * k, and every atomic fact has ~k continuations.
        num_heldout_relations: how many relations are never composed in training.
        group_size, group_size_max: width of the co-occurrence rows; with group_size_max the
            width of each row is drawn uniformly from [group_size, group_size_max] (E-co style:
            group_size 1, group_size_max 4 gives every task 0-3 random partners).
        grouped_frac: fraction of atomic facts presented inside width-k rows (the rest are
            single atomic rows). 0.0 = no co-occurrence practice (the "v1" cell).
        heldout_grouped_frac: same, applied to held-out relations only (the dose knob);
            None = same as grouped_frac.
        heldout_slot / partner: position and partner constraints for held-out tasks (k == 2).
        group_passes: how many independent groupings of the grouped facts to emit
            (1 = every grouped fact appears in exactly one width row).
        always_singles: emit one single-task row for EVERY atomic fact in addition to whatever
            groups it appears in (matched-volume ablations: the same atomic base in every arm).
        single_passes: how many copies of every single-task row to emit, so that facts kept out
            of groups (e.g. held-out relations with heldout_grouped_frac 0) can match the
            per-fact frequency of grouped facts (frequency control).
        comp_depths: depths of composition rows, e.g. [2] or [2, 3]; n_comp is split evenly.
        n_comp: number of composition rows (0 = the "C1" cell, no compositions).
        comp_relations: use only this many train relations inside compositions (diversity
            knob); None = all train relations.
        comp_pair_holdout_frac: fraction of composable relation pairs kept out of training
            entirely (-> test type d2_train_unseen_pair).
        heldout_comp_frac: "leak" knob. Fraction of the composable depth-2 instances that involve at
            least one held-out relation which are put into training as ordinary composition rows
            (0 = strict purity). The d2_mixed_* / d2_heldout tests exclude leaked instances.
        heldout_leak_per_rel: balanced variant of heldout_leak_pairs: exactly this many qualifying
            pairs per held-out relation (dose curve in "distinct partner relations per skill").
        heldout_leak_pairs / heldout_leak_within / heldout_leak_slot: concentrated leak. Pick this
            many relation pairs that involve a held-out relation (slot "second" = (train, held-out),
            "first" = (held-out, train), "any"), and put `heldout_leak_within` of each chosen pair's
            instances into training. The remaining instances of those pairs form the test type
            d2_leakpair_unseen (pair-level entry into the circuit); the other pairs of the same
            held-out relations stay in d2_mixed_* / d2_heldout (relation-level transfer).
        n_mixed: extra "mixed-context" rows, each = one held-out atomic task + one training
            composition (train relations only) in random order. Held-out relations are still never
            composed; they are merely practised inside the composition context.
        n_test_per_type: rows per test type (fewer if the type's support is smaller).
        test_depths / test_widths: which chain depths and widths to build tests for.

    Returns a dict with keys vocab, train, test (rows carry "type"), meta, counts.
    """
    assert heldout_slot in ("any", "first", "last"), heldout_slot
    assert partner in ("any", "train", "heldout"), partner
    assert group_size >= 2 or (group_size_max is not None and group_size_max >= 2), \
        "group_size must be >= 2 unless group_size_max makes the width variable"
    assert all(d >= 2 for d in comp_depths), "composition depth must be >= 2"
    rng = np.random.default_rng(seed)
    world = SkillsWorld(num_entities, num_relations, num_heldout_relations, rng, out_degree=out_degree)
    E, R = num_entities, num_relations
    train_rel, heldout_rel = world.train_rel, world.heldout_rel

    # ------------------------------------------------------------------ atomic + width rows
    facts: List[Task] = [(h, j) for j in range(R) for h in range(E) if world.defined(j, h)]
    hgf = grouped_frac if heldout_grouped_frac is None else heldout_grouped_frac
    grouped, singles = [], []
    for h, j in facts:
        p = hgf if world.is_heldout(j) else grouped_frac
        (grouped if rng.uniform() < p else singles).append((h, j))

    width_groups: List[List[Task]] = []
    for p in range(group_passes):
        groups, leftovers = _make_groups(grouped, group_size, world, rng, heldout_slot, partner,
                                         k_max=group_size_max)
        width_groups += groups
        if p == 0:
            singles += leftovers  # every fact appears in training at least once
    if always_singles:
        singles = list(facts)
    single_rows = [world.atomic_row([t]) for t in singles] * max(1, single_passes) + \
        [world.atomic_row(g) for g in width_groups if len(g) == 1]
    width_rows = [world.atomic_row(g) for g in width_groups if len(g) >= 2]

    # ------------------------------------------------------------------ structure helpers
    train_set, heldout_set = set(train_rel), set(heldout_rel)
    heads_by_rel: Dict[int, List[int]] = {j: world.heads_of(j) for j in range(R)}

    def pick(lst):
        return lst[rng.integers(len(lst))]

    # ------------------------------------------------------------------ composition rows
    if comp_relations is None or comp_relations >= len(train_rel):
        comp_rel = list(train_rel)
    else:
        comp_rel = sorted(rng.choice(train_rel, size=comp_relations, replace=False).tolist())
    comp_set = set(comp_rel)
    uncomposed_rel = [j for j in train_rel if j not in comp_set]

    all_pairs = [(a, b) for a in comp_rel for b in comp_rel]
    rng.shuffle(all_pairs)
    n_hold = int(round(len(all_pairs) * comp_pair_holdout_frac))
    heldout_pairs = all_pairs[:n_hold]
    train_pairs = all_pairs[n_hold:]
    heldout_pair_set = set(heldout_pairs)

    kind_set = {"train": train_set, "heldout": heldout_set, "comp": comp_set, "any": set(range(R))}
    kind_rels = {"train": train_rel, "heldout": heldout_rel, "comp": comp_rel, "any": list(range(R))}

    def rand_fact(kind: str):
        """A defined atomic fact (h, j) whose relation is of the given kind."""
        for _ in range(50):
            j = pick(kind_rels[kind])
            if heads_by_rel[j]:
                return (pick(heads_by_rel[j]), j)
        return None

    def extend(x: int, kind: str):
        """A relation of the given kind that is defined on entity x, or None."""
        cands = [j for j in world.rels_of[x] if j in kind_set[kind]]
        return pick(cands) if cands else None

    def sample_chain(kinds: Sequence[str]):
        """(h, chain) with one relation kind per hop, every hop defined."""
        f = rand_fact(kinds[0])
        if f is None:
            return None
        h, j = f
        chain = [j]
        x = int(world.rel_map[j, h])
        for kind in kinds[1:]:
            j = extend(x, kind)
            if j is None:
                return None
            chain.append(j)
            x = int(world.rel_map[j, x])
        return (h, tuple(chain))

    # every composable depth-2 instance over train pairs, enumerated exactly
    all_d2: List[Tuple[int, Tuple[int, ...]]] = []
    for a in comp_rel:
        for h in heads_by_rel[a]:
            mid = int(world.rel_map[a, h])
            for b in world.rels_of[mid]:
                if b in comp_set and (a, b) not in heldout_pair_set:
                    all_d2.append((h, (a, b)))
    rng.shuffle(all_d2)

    comp_items: List[Tuple[int, Tuple[int, ...]]] = []
    n_d2_train = 0
    if n_comp > 0 and train_pairs:
        per_depth = [n_comp // len(comp_depths)] * len(comp_depths)
        per_depth[0] += n_comp - sum(per_depth)
        for depth, n_d in zip(comp_depths, per_depth):
            if depth == 2:
                n_d2_train = min(n_d, len(all_d2))
                comp_items += all_d2[:n_d2_train]
            else:
                def samp(depth=depth):
                    item = sample_chain(["comp"] * depth)
                    if item is None:
                        return None
                    chain = item[1]
                    if any((chain[i], chain[i + 1]) in heldout_pair_set for i in range(len(chain) - 1)):
                        return None
                    return item
                comp_items += _collect(n_d, samp, key_fn=lambda x: x)
    # optional leak: a small fraction of the instances that involve a held-out relation
    leaked: List[Tuple[int, Tuple[int, ...]]] = []
    leak_pair_unseen: List[Tuple[int, Tuple[int, ...]]] = []
    leak_pairs_chosen: List[Tuple[int, int]] = []
    if heldout_comp_frac > 0 or heldout_leak_pairs > 0 or heldout_leak_per_rel > 0:
        all_d2_ho = []
        for a in range(R):
            for h in heads_by_rel[a]:
                mid = int(world.rel_map[a, h])
                for b in world.rels_of[mid]:
                    if world.is_heldout(a) or world.is_heldout(b):
                        all_d2_ho.append((h, (a, b)))
        rng.shuffle(all_d2_ho)
        if heldout_comp_frac > 0:
            leaked += all_d2_ho[:int(round(heldout_comp_frac * len(all_d2_ho)))]
        if heldout_leak_pairs > 0 or heldout_leak_per_rel > 0:
            assert heldout_leak_slot in ("any", "first", "second"), heldout_leak_slot
            by_pair: Dict[Tuple[int, int], list] = {}
            for item in all_d2_ho:
                a, b = item[1]
                ok = (heldout_leak_slot == "any"
                      or (heldout_leak_slot == "second" and not world.is_heldout(a) and world.is_heldout(b))
                      or (heldout_leak_slot == "first" and world.is_heldout(a) and not world.is_heldout(b)))
                if ok:
                    by_pair.setdefault((a, b), []).append(item)
            pairs = sorted(by_pair)
            rng.shuffle(pairs)
            if heldout_leak_per_rel > 0:
                # balanced: the same number of pairs for every held-out relation (keyed by its slot)
                leak_pairs_chosen = []
                for j in heldout_rel:
                    mine = [pr for pr in pairs if (pr[1] == j if heldout_leak_slot == "second" else
                                                   pr[0] == j if heldout_leak_slot == "first" else j in pr)]
                    leak_pairs_chosen += mine[:heldout_leak_per_rel]
            else:
                leak_pairs_chosen = pairs[:heldout_leak_pairs]
            for pr in leak_pairs_chosen:
                lst = by_pair[pr]
                rng.shuffle(lst)
                k = int(round(heldout_leak_within * len(lst)))
                leaked += lst[:k]
                leak_pair_unseen += lst[k:]
        leaked = list(dict.fromkeys(leaked))  # dedupe, keep order
        comp_items += leaked
    train_instances = set(comp_items)
    comp_rows = [world.chain_row(h, chain) for h, chain in comp_items]

    mixed_rows: List[Dict[str, str]] = []
    if n_mixed > 0 and comp_items and heldout_rel:
        for _ in range(n_mixed):
            fact = rand_fact("heldout")
            if fact is None:
                break
            tasks = [(fact[0], (fact[1],)), pick(comp_items)]
            if rng.uniform() < 0.5:
                tasks.reverse()
            mixed_rows.append(world.task_row(tasks))

    train_rows = single_rows + width_rows + comp_rows + mixed_rows

    # ------------------------------------------------------------------ test sets
    test_rows: List[Dict] = []
    counts: Dict[str, int] = {}

    def add_chain_type(name: str, sampler, exclude=frozenset()):
        def defined_sampler():
            item = sampler()
            if item is None or world.apply(item[1], item[0]) < 0:
                return None
            return item
        items = _collect(n_test_per_type, defined_sampler, key_fn=lambda x: x, exclude=exclude)
        for h, chain in items:
            test_rows.append({**world.chain_row(h, chain), "type": name})
        counts[name] = len(items)

    def add_width_type(name: str, sampler):
        def defined_sampler():
            tasks = sampler()
            if tasks is None or any(t is None or not world.defined(t[1], t[0]) for t in tasks):
                return None
            return tasks
        items = _collect(n_test_per_type, defined_sampler, key_fn=lambda x: tuple(x))
        for tasks in items:
            test_rows.append({**world.atomic_row(tasks), "type": name})
        counts[name] = len(items)

    def as_chain(fact):
        return None if fact is None else (fact[0], (fact[1],))

    # atomic memorization checks (all atomic facts are in training)
    add_chain_type("atomic_train", lambda: as_chain(rand_fact("train")))
    add_chain_type("atomic_heldout", lambda: as_chain(rand_fact("heldout")))

    if 2 in test_depths:
        seen_d2 = [x for x in comp_items if len(x[1]) == 2]
        if seen_d2:
            add_chain_type("d2_train_seen", lambda: pick(seen_d2))
        unseen_d2 = all_d2[n_d2_train:]
        if unseen_d2:
            add_chain_type("d2_train_unseen_instance", lambda: pick(unseen_d2))
        if heldout_pairs:
            def samp_unseen_pair():
                a, b = pick(heldout_pairs)
                if not heads_by_rel[a]:
                    return None
                return (pick(heads_by_rel[a]), (a, b))
            add_chain_type("d2_train_unseen_pair", samp_unseen_pair)
        if uncomposed_rel:
            def samp_uncomposed():
                item = sample_chain(["train", "train"])
                if item is None or all(j in comp_set for j in item[1]):
                    return None
                return item
            add_chain_type("d2_train_uncomposed", samp_uncomposed)
        if leak_pair_unseen:
            add_chain_type("d2_leakpair_unseen", lambda: pick(leak_pair_unseen), exclude=train_instances)
        add_chain_type("d2_mixed_first", lambda: sample_chain(["heldout", "train"]), exclude=train_instances)
        add_chain_type("d2_mixed_second", lambda: sample_chain(["train", "heldout"]), exclude=train_instances)
        add_chain_type("d2_heldout", lambda: sample_chain(["heldout", "heldout"]), exclude=train_instances)

    for d in sorted(set(test_depths)):
        if d < 3:
            continue
        add_chain_type(f"d{d}_train", lambda d=d: sample_chain(["train"] * d), exclude=train_instances)
        add_chain_type(f"d{d}_heldout", lambda d=d: sample_chain(["heldout"] * d))

        def samp_mixed(d=d):
            item = sample_chain(["any"] * d)
            if item is None or len({world.is_heldout(j) for j in item[1]}) < 2:
                return None
            return item
        add_chain_type(f"d{d}_mixed", samp_mixed)

    for k in sorted(set(test_widths)):
        if k < 2:
            continue
        add_width_type(f"w{k}_train", lambda k=k: [rand_fact("train") for _ in range(k)])
        add_width_type(f"w{k}_heldout", lambda k=k: [rand_fact("heldout") for _ in range(k)])

        def samp_wmixed(k=k):
            tasks = [rand_fact("any") for _ in range(k)]
            if any(t is None for t in tasks) or len({world.is_heldout(j) for _, j in tasks}) < 2:
                return None
            return tasks
        add_width_type(f"w{k}_mixed", samp_wmixed)

    meta = {
        "task": "skills",
        "num_entities": E,
        "num_relations": R,
        "out_degree": out_degree,
        "train_relations": train_rel,
        "heldout_relations": heldout_rel,
        "comp_relations": comp_rel,
        "uncomposed_train_relations": uncomposed_rel,
        "rel_map": world.rel_map.tolist(),
        "train_pairs": [list(p) for p in train_pairs],
        "leak_pairs_chosen": [list(p) for p in leak_pairs_chosen],
        "heldout_pairs": [list(p) for p in heldout_pairs],
        "config": {
            "num_entities": E, "num_relations": R, "num_heldout_relations": num_heldout_relations,
            "seed": seed, "out_degree": out_degree, "group_size": group_size, "group_size_max": group_size_max,
            "grouped_frac": grouped_frac,
            "heldout_grouped_frac": heldout_grouped_frac, "heldout_slot": heldout_slot,
            "partner": partner, "group_passes": group_passes, "single_passes": single_passes,
            "always_singles": always_singles,
            "comp_depths": list(comp_depths),
            "n_comp": n_comp, "comp_relations": comp_relations,
            "comp_pair_holdout_frac": comp_pair_holdout_frac, "n_mixed": n_mixed,
            "heldout_comp_frac": heldout_comp_frac, "heldout_leak_pairs": heldout_leak_pairs,
            "heldout_leak_per_rel": heldout_leak_per_rel,
            "heldout_leak_within": heldout_leak_within, "heldout_leak_slot": heldout_leak_slot,
            "n_test_per_type": n_test_per_type,
            "test_depths": list(test_depths), "test_widths": list(test_widths),
        },
        "train_counts": {"single": len(single_rows), "width": len(width_rows), "comp": len(comp_rows),
                         "mixed": len(mixed_rows), "comp_leaked_heldout": len(leaked)},
        "test_counts": counts,
    }
    return {
        "vocab": world.entities + world.relations,
        "train": train_rows,
        "test": test_rows,
        "meta": meta,
        "counts": {"train": meta["train_counts"], "test": counts},
    }


def save_dataset_skills(output_dir: str, ds: Dict) -> None:
    """Write train.json / test.json / vocab.json / meta.json and one file per test type."""
    os.makedirs(os.path.join(output_dir, "by_type"), exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(ds["train"], f)
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(ds["test"], f)
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(ds["vocab"], f)
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(ds["meta"], f)
    by_type: Dict[str, List[Dict]] = {}
    for row in ds["test"]:
        by_type.setdefault(row["type"], []).append(row)
    for t, rows in by_type.items():
        with open(os.path.join(output_dir, "by_type", f"{t}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f)
    print(f"Dataset saved to {output_dir}")
    print(f"  - vocab size: {len(ds['vocab'])}")
    print(f"  - train rows: {len(ds['train'])} {ds['counts']['train']}")
    print(f"  - test rows: {len(ds['test'])}")
