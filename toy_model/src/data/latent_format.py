"""
Input / output protocol of the fifth batch (docs/experiments_latent_batch2_20runs.md §3): complete programs in a
fixed logical address space, final-answer-only outputs.

Logical input positions (262): two TASK regions of 131 positions each
    region g:  TASK = 131 g,  start entity = 131 g + 1,  program slots = 131 g + 2 ... 131 g + 129,  EOP = 131 g + 130
  w2   uses both regions, one relation each          TASK e_x r_a EOP | TASK e_z r_b EOP   ->  e_a(x) e_b(z) END_ANSWER
  d2 / test chains use region 0 only (region 1 is entirely masked)     TASK e_x r_1 ... r_d EOP  ->  e_y END_ANSWER
The d relations keep their left-to-right order and sit in d slots drawn without replacement from the 128 slots and
sorted ("ordered random slots"); unused slots are EMPTY and never enter a key softmax. We use the PACKED form: only
valid tokens are materialised, each carries its logical position id (dense == packed is checked in
scripts/test_latent_executor.py). Latent slots live at positions 262..265, the answer prefix starts at 266.

Vocabulary: the 524 tokens of data/chain_loop/vocab.json unchanged + TASK, EOP, EMPTY, ANSWER, END_ANSWER = 529.
"""
import hashlib
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

EXTRA = ["TASK", "EOP", "EMPTY", "ANSWER", "END_ANSWER"]
N_SLOTS, REGION, N_MEM_POS = 128, 131, 262
LATENT_POS0, N_LATENT, ANS_POS0, MAX_ANS = 262, 4, 266, 4
LAYOUT_SEED = 20260918
ADDR_TASK, ADDR_ENT, ADDR_SLOT0, ADDR_EOP = 0, 1, 2, 130


class LatentVocab:
    def __init__(self, base_vocab: List[str]):
        assert not (set(EXTRA) & set(base_vocab))
        self.vocab = list(base_vocab) + EXTRA
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.pad = self.tok2id["<pad>"]; self.e0 = self.tok2id["<e_0>"]; self.r0 = self.tok2id["<r_0>"]
        self.E = sum(t.startswith("<e_") for t in base_vocab); self.R = sum(t.startswith("<r_") for t in base_vocab)
        for t in EXTRA:
            setattr(self, t, self.tok2id[t])
        self.hash = hashlib.sha256(json.dumps(self.vocab).encode()).hexdigest()[:16]

    def is_e(self, t: int) -> bool: return self.e0 <= t < self.e0 + self.E


def sorted_slots(rng: np.random.Generator, d: int) -> np.ndarray:
    return np.sort(rng.choice(N_SLOTS, size=d, replace=False))


def frozen_slots(kind: int, idx: int, d: int, variant: int = 0) -> np.ndarray:
    """Frozen evaluation layout. kind: 1 tests, 2 validation d2, 3 atomic, 4 suffix pairs, 5 extra layouts, 6 val w2."""
    return sorted_slots(np.random.default_rng([LAYOUT_SEED, kind, idx, variant]), d)


def render_chain(v: LatentVocab, x: int, rels: Sequence[int], slots: Sequence[int]) -> Tuple[List[int], List[int]]:
    """packed (tokens, logical positions) of a single-region packet."""
    tok = [v.TASK, v.e0 + x] + [v.r0 + r for r in rels] + [v.EOP]
    pos = [ADDR_TASK, ADDR_ENT] + [ADDR_SLOT0 + int(s) for s in slots] + [ADDR_EOP]
    return tok, pos


def render_chain_dense(v: LatentVocab, x: int, rels: Sequence[int], slots: Sequence[int]):
    """dense 262-position rendering with EMPTY + mask (test only): returns tok, pos, pad (True = masked)."""
    tok = [v.EMPTY] * N_MEM_POS; pad = [True] * N_MEM_POS
    for t, p in zip(*render_chain(v, x, rels, slots)):
        tok[p] = t; pad[p] = False
    return tok, list(range(N_MEM_POS)), pad


def batch_chains(v: LatentVocab, items: Sequence[Dict], slots: Sequence[Sequence[int]], device) -> Tuple[torch.Tensor, ...]:
    """items of EQUAL depth -> tok, pos, pad tensors (no padding needed)."""
    rows = [render_chain(v, it["x"], it["rels"], s) for it, s in zip(items, slots)]
    tok = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    pos = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    return tok, pos, torch.zeros_like(tok, dtype=torch.bool)


class TrainTables:
    """Source questions as device tensors; batches are assembled with tensor ops (no Python loop per row)."""

    def __init__(self, v: LatentVocab, rel, train_w2, train_d2, device):
        rel_t = np.asarray(rel)
        w = np.array([[q["q1"][0], q["q1"][1], q["q2"][0], q["q2"][1]] for q in train_w2])
        self.w2 = torch.tensor(np.stack([w[:, 0], w[:, 1], rel_t[w[:, 1], w[:, 0]], w[:, 2], w[:, 3], rel_t[w[:, 3], w[:, 2]]], 1), device=device)
        d = np.array([[q["x"], q["r"], q["s"]] for q in train_d2]); b = rel_t[d[:, 1], d[:, 0]]; y = rel_t[d[:, 2], b]
        self.d2 = torch.tensor(np.stack([d[:, 0], d[:, 1], d[:, 2], b, y], 1), device=device)
        self.v = v; self.device = device

    def batch(self, qi_w2: np.ndarray, qi_d2: np.ndarray, lay: np.random.Generator):
        """-> dict(tok, pos, pad (B, 8); ans_in, ans_tgt, ans_w (B, 3); n_w2; aux for the d2 rows). The layout stream draws
        w2 slots first, then the d2 pair, every update, so it is identical for all conditions with the same seed."""
        v = self.v; dev = self.device; n1, n2 = len(qi_w2), len(qi_d2)
        s_w2 = torch.from_numpy(lay.integers(N_SLOTS, size=(n1, 2))).to(dev)
        a = lay.integers(N_SLOTS, size=n2); b = lay.integers(N_SLOTS - 1, size=n2); b = b + (b >= a)
        s_d2 = torch.from_numpy(np.sort(np.stack([a, b], 1), 1)).to(dev)
        w = self.w2[torch.from_numpy(qi_w2).to(dev)]; d = self.d2[torch.from_numpy(qi_d2).to(dev)] if n2 else self.d2[:0]
        full = lambda n, val: torch.full((n,), val, dtype=torch.long, device=dev)
        tok_w = torch.stack([full(n1, v.TASK), v.e0 + w[:, 0], v.r0 + w[:, 1], full(n1, v.EOP), full(n1, v.TASK), v.e0 + w[:, 3], v.r0 + w[:, 4], full(n1, v.EOP)], 1)
        pos_w = torch.stack([full(n1, 0), full(n1, 1), 2 + s_w2[:, 0], full(n1, 130), full(n1, 131), full(n1, 132), 133 + s_w2[:, 1], full(n1, 261)], 1)
        tok_d = torch.stack([full(n2, v.TASK), v.e0 + d[:, 0], v.r0 + d[:, 1], v.r0 + d[:, 2], full(n2, v.EOP)] + [full(n2, v.pad)] * 3, 1)
        pos_d = torch.stack([full(n2, 0), full(n2, 1), 2 + s_d2[:, 0], 2 + s_d2[:, 1], full(n2, 130)] + [full(n2, 0)] * 3, 1)
        pad = torch.zeros((n1 + n2, 8), dtype=torch.bool, device=dev); pad[n1:, 5:] = True
        ans_in = torch.cat([torch.stack([full(n1, v.ANSWER), v.e0 + w[:, 2], v.e0 + w[:, 5]], 1), torch.stack([full(n2, v.ANSWER), v.e0 + d[:, 4], full(n2, v.pad)], 1)])
        ans_tgt = torch.cat([torch.stack([v.e0 + w[:, 2], v.e0 + w[:, 5], full(n1, v.END_ANSWER)], 1), torch.stack([v.e0 + d[:, 4], full(n2, v.END_ANSWER), full(n2, -100)], 1)])
        ans_w = torch.cat([torch.tensor([0.5, 0.5, 1.0], device=dev).expand(n1, 3), torch.tensor([1.0, 1.0, 0.0], device=dev).expand(n2, 3)])
        aux = dict(x1=d[:, 3], x2=d[:, 4], addr=torch.stack([2 + s_d2[:, 0], 2 + s_d2[:, 1], full(n2, ADDR_EOP)], 1))   # LOGICAL addresses
        return dict(tok=torch.cat([tok_w, tok_d]), pos=torch.cat([pos_w, pos_d]), pad=pad, ans_in=ans_in, ans_tgt=ans_tgt, ans_w=ans_w, n_w2=n1, aux=aux)


def addr_to_key(pos: torch.Tensor, pad: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
    """logical address (B,) or (B, K) -> index of the packed key carrying that position id (asserts it exists and is unmasked)."""
    if addr.dim() == 1:
        addr = addr[:, None]
    hit = (pos[:, None, :] == addr[:, :, None]) & ~pad[:, None, :]
    assert bool(hit.any(-1).all()), "aux address not present in the packet"
    return hit.float().argmax(-1)


def parse_answer(v: LatentVocab, gen: Sequence[int], gold: Sequence[int]) -> str:
    """strict: exactly the gold entity tokens followed by END_ANSWER. Returns 'correct' | 'wrong_entity' | 'format'."""
    seq = list(gen)
    if v.END_ANSWER not in seq:
        return "format"
    body = seq[:seq.index(v.END_ANSWER)]
    if len(body) != len(gold) or not all(v.is_e(t) for t in body):
        return "format"
    return "correct" if body == list(gold) else "wrong_entity"
