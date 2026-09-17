"""
Fixed logical data streams for update-based training (N2 of docs/findings_and_next_experiments.md §6).

Every optimizer update draws a FIXED number of single-answer rows from each declared stream, e.g.
``tt:128,atomic:64,first:32,second:32``:

  tt       depth-2 compositions of two TRAIN relations, taken from train.json (unique rows)
  atomic   single-task rows <e_h><r_j><e_t> for ALL (h, j) of the world, synthesised from meta.rel_map
  first    role rows with a held-out relation in the first hop  <e_h><r_H><r_T><e_t>   (unique rows of train.json)
  second   role rows with a held-out relation in the second hop <e_h><r_T><r_H><e_t>   (unique rows of train.json)

A stream name may be repeated (``first:32,first:32``): each occurrence is an independent stream instance with its
own ordering (channel A / B). Orderings are pre-frozen from ``seed``:
  tt / atomic   a fresh permutation of the unique rows per cycle
  first/second  rows are grouped by role FACT ((H, input) for first, (H, bridge) for second) with P partner rows
                each; a cycle = P blocks; block r visits every fact once (fresh fact permutation per cycle) and
                fact i uses partner index (i + r) mod P. With 32 rows/update and 3,200 facts x 4 partners, every
                100 updates cover each fact once and every 400 updates each fact x partner once.
The loss is the SUM of answer cross-entropies over the update divided by a fixed ``denominator`` (train.py), so
arms with fewer active streams do not renormalise.
"""
import collections
import hashlib
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

TOK = re.compile(r"<e_(\d+)>|<r_(\d+)>")


def parse_spec(spec: str) -> List[Tuple[str, int]]:
    out = []
    for part in spec.split(","):
        name, n = part.strip().split(":")
        out.append((name.strip(), int(n)))
    return out


def categorise(text: str, heldout: set) -> Tuple[str, Tuple]:
    """(category, key) of one training row; key identifies the role fact for role rows."""
    toks = [(int(a) if a else None, int(b) if b else None) for a, b in TOK.findall(text)]
    ents = [t[0] for t in toks if t[0] is not None]
    rels = [t[1] for t in toks if t[1] is not None]
    if len(ents) > 2 or len(rels) == 0:
        return "width", ()
    if len(rels) == 1:
        return "atomic", (ents[0], rels[0])
    if len(rels) != 2:
        return "deep", ()
    r1, r2 = rels
    if r1 in heldout and r2 in heldout:
        return "hh", ()
    if r1 in heldout:
        return "first", (r1, ents[0])           # fact = (H, input h); partner = r2
    if r2 in heldout:
        return "second", (r2, None)            # bridge filled in by the caller (needs rel_map)
    return "tt", ()


class FixedStreams:
    def __init__(self, train_rows: Sequence[Dict], meta: Dict, spec: str, seed: int, tok2id: Dict[str, int],
                 device: torch.device, pad_id: int = 0):
        self.spec = parse_spec(spec)
        rel = meta["rel_map"]; E = meta["num_entities"]; H = set(meta["heldout_relations"])
        self.tok2id = tok2id; self.device = device
        # ---- unique rows per category
        rows: Dict[str, Dict[str, Tuple]] = collections.defaultdict(dict)   # cat -> text -> fact key
        for r in train_rows:
            cat, key = categorise(r["target_text"], H)
            if cat == "second":
                toks = [(int(a) if a else None, int(b) if b else None) for a, b in TOK.findall(r["target_text"])]
                h = toks[0][0]; r1 = toks[1][1]; r2 = toks[2][1]
                key = (r2, rel[r1][h])          # fact = (H, bridge); partner = r1
            if cat in ("tt", "first", "second"):
                rows[cat][r["target_text"]] = key
        for j in range(len(rel)):
            for h in range(E):
                rows["atomic"][f"<e_{h}><r_{j}><e_{rel[j][h]}>"] = (h, j)
        self.texts = {cat: sorted(d) for cat, d in rows.items()}
        self.keys = {cat: [rows[cat][t] for t in self.texts[cat]] for cat in rows}
        self.counts = {cat: len(v) for cat, v in self.texts.items()}
        # ---- encode every unique row (input = target[:-1], loss on the entity after the last relation)
        L = 3
        self.enc: Dict[str, Dict[str, torch.Tensor]] = {}
        for cat, texts in self.texts.items():
            n = len(texts)
            inp = torch.full((n, L), pad_id, dtype=torch.long); tgt = torch.full((n, L), -100, dtype=torch.long)
            lm = torch.zeros((n, L), dtype=torch.bool); pm = torch.ones((n, L), dtype=torch.bool)
            for i, t in enumerate(texts):
                ids = [tok2id[x] for x in re.findall(r"<e_\d+>|<r_\d+>", t)]
                k = len(ids) - 1
                inp[i, :k] = torch.tensor(ids[:-1]); tgt[i, :k] = torch.tensor(ids[1:])
                lm[i, k - 1] = True; pm[i, :k] = False
            self.enc[cat] = dict(input_ids=inp.to(device), target_ids=tgt.to(device), loss_mask=lm.to(device), pad_mask=pm.to(device))
        # ---- stream instances
        self.streams = []
        for k, (cat, n) in enumerate(self.spec):
            if cat not in self.texts:
                raise ValueError(f"stream '{cat}' has no rows in this dataset (have {sorted(self.texts)})")
            rng = np.random.default_rng([seed, k, 977])
            st = dict(cat=cat, n=n, rng=rng, cursor=0, order=None, used=np.zeros(self.counts[cat], dtype=np.int64), cycles=0)
            if cat in ("first", "second"):
                by_fact = collections.defaultdict(list)
                for i, key in enumerate(self.keys[cat]):
                    by_fact[key].append(i)
                st["facts"] = sorted(by_fact)
                st["partners"] = [sorted(by_fact[f]) for f in st["facts"]]   # row indices, one per partner
                P = {len(p) for p in st["partners"]}
                assert len(P) == 1, f"unequal partner counts per fact in stream {cat}: {P}"
                st["P"] = P.pop()
            self.streams.append(st)

    def _new_cycle(self, st):
        cat = st["cat"]
        if cat in ("first", "second"):
            perm = st["rng"].permutation(len(st["facts"]))
            blocks = []
            for r in range(st["P"]):
                blocks.append(np.array([st["partners"][f][(i + r) % st["P"]] for i, f in enumerate(perm)], dtype=np.int64))
            st["order"] = np.concatenate(blocks)
        else:
            st["order"] = st["rng"].permutation(self.counts[cat]).astype(np.int64)
        st["cursor"] = 0; st["cycles"] += 1

    def next_batch(self) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, int]]]:
        parts = []; slices = []
        for st in self.streams:
            take = []
            need = st["n"]
            while need > 0:
                if st["order"] is None or st["cursor"] >= len(st["order"]):
                    self._new_cycle(st)
                chunk = st["order"][st["cursor"]:st["cursor"] + need]
                st["cursor"] += len(chunk); need -= len(chunk); take.append(chunk)
            idx = np.concatenate(take)
            np.add.at(st["used"], idx, 1)
            idx_t = torch.from_numpy(idx).to(self.device)
            parts.append({k: v[idx_t] for k, v in self.enc[st["cat"]].items()})
            slices.append((st["cat"], len(idx)))
        batch = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        return batch, slices

    def exposure_report(self) -> Dict:
        rep = {}
        for k, st in enumerate(self.streams):
            used = st["used"]
            r = dict(cat=st["cat"], rows_per_update=st["n"], unique_rows=int(len(used)), total_draws=int(used.sum()),
                     cycles_started=st["cycles"], per_row_min=int(used.min()), per_row_max=int(used.max()))
            if st["cat"] in ("first", "second"):
                per_fact = np.array([used[p].sum() for p in st["partners"]])
                r.update(facts=len(st["facts"]), partners_per_fact=st["P"], per_fact_min=int(per_fact.min()),
                         per_fact_max=int(per_fact.max()), per_fact_mean=float(per_fact.mean()))
            rep[f"stream{k}_{st['cat']}"] = r
        return rep

    def manifest(self) -> Dict:
        return dict(spec=self.spec, unique_rows=self.counts,
                    row_hashes={cat: hashlib.sha256("\n".join(t).encode()).hexdigest()[:16] for cat, t in self.texts.items()})
