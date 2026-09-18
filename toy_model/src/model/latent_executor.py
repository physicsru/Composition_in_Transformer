"""
Fixed problem memory + autonomously updated latent workspace (docs/experiments_latent_batch2_20runs.md §4).

    packet --Encoder (2 layers, bidirectional)--> M            (computed ONCE per solve, never overwritten)
    4 learned latent slots Z0;   Z_t = Core(Z_{t-1}, M)   for t = 1..T     (no loop index, no pointer, no gold state)
    Z_T --AnswerDecoder (1 layer, causal, cross-attends to Z_T only)--> answer tokens

mode 'shared'  the same 2-block core is applied T times                      (conditions A-D)
mode 'single'  the core is applied once                                       (E; same parameters as A)
mode 'untied'  8 cores with separate parameters, loop t uses core t, T <= 8   (F)

RoPE (base 100) is applied to Q / K after their projections, V is not rotated. Memory tokens carry their logical
position ids 0..261, latent slots 262..265, answer prefix 266.. . The supervised read head (conditions C / D) is the
memory cross-attention of the SECOND core block, latent slot 0, head 0: its actual soft attention probabilities are
returned for the loss, and its weighted values enter the residual stream as usual (no oracle read module, no mask).
The entity auxiliary head (LayerNorm + linear -> E classes on slot 0 after every loop) is instantiated in every
condition so parameters and init are identical; it only receives gradients in B / D.
"""
import hashlib
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    d = x.size(-1); return torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)


class Rope(nn.Module):
    def __init__(self, head_dim: int, base: float = 100.0, max_pos: int = 288):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        f = torch.einsum("l,d->ld", torch.arange(max_pos).float(), inv); emb = torch.cat([f, f], -1)
        self.register_buffer("cos", emb.cos(), persistent=False); self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, pos: torch.Tensor):
        """pos (B, L) or (L,) -> (cos, sin) broadcastable to (B, H, L, hd)"""
        if pos.dim() == 1:
            return self.cos[pos][None, None], self.sin[pos][None, None]
        return self.cos[pos][:, None], self.sin[pos][:, None]


class Attn(nn.Module):
    def __init__(self, d: int, n_head: int, manual: bool = False):
        super().__init__()
        self.h = n_head; self.hd = d // n_head; self.manual = manual
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)

    def forward(self, x, y, rope_q, rope_k, key_pad=None, causal=False):
        """x (B, Lq, C) queries, y (B, Lk, C) keys/values. Returns (out, probs or None); probs (B, H, Lq, Lk) only if manual."""
        B, Lq, C = x.shape; Lk = y.shape[1]
        sp = lambda t, L: t.view(B, L, self.h, self.hd).transpose(1, 2)
        q, k, v = sp(self.q(x), Lq), sp(self.k(y), Lk), sp(self.v(y), Lk)
        q = q * rope_q[0] + rotate_half(q) * rope_q[1]; k = k * rope_k[0] + rotate_half(k) * rope_k[1]
        mask = None
        if key_pad is not None:
            mask = ~key_pad[:, None, None, :]
        if causal:
            cm = torch.ones(Lq, Lk, dtype=torch.bool, device=x.device).tril()[None, None]
            mask = cm if mask is None else (mask & cm)
        probs = None
        if self.manual:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            if mask is not None:
                att = att.masked_fill(~mask, float("-inf"))
            probs = att.softmax(-1); out = probs @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, Lq, C)), probs


def ffn(d):
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


class EncLayer(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.ln1 = nn.LayerNorm(d); self.att = Attn(d, h); self.ln2 = nn.LayerNorm(d); self.ff = ffn(d)

    def forward(self, x, rope, pad):
        n = self.ln1(x); x = x + self.att(n, n, rope, rope, key_pad=pad)[0]
        return x + self.ff(self.ln2(x))


class CoreBlock(nn.Module):
    """latent self-attention -> cross-attention to the full memory -> FFN (pre-LN residual branches)."""

    def __init__(self, d, h, manual_cross):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.self_att = Attn(d, h); self.ln2 = nn.LayerNorm(d); self.cross = Attn(d, h, manual=manual_cross)
        self.ln3 = nn.LayerNorm(d); self.ff = ffn(d)

    def forward(self, z, M, rope_z, rope_m, pad):
        n = self.ln1(z); z = z + self.self_att(n, n, rope_z, rope_z)[0]
        c, probs = self.cross(self.ln2(z), M, rope_z, rope_m, key_pad=pad); z = z + c
        return z + self.ff(self.ln3(z)), probs


class Core(nn.Module):
    def __init__(self, d, h, n_blocks=2):
        super().__init__()
        # the LAST block's cross-attention is the (potentially) supervised read: always the manual path, in every condition
        self.blocks = nn.ModuleList([CoreBlock(d, h, manual_cross=(i == n_blocks - 1)) for i in range(n_blocks)])

    def forward(self, z, M, rope_z, rope_m, pad):
        probs = None
        for blk in self.blocks:
            z, p = blk(z, M, rope_z, rope_m, pad)
            probs = p if p is not None else probs
        return z, probs


class LatentExecutor(nn.Module):
    def __init__(self, vocab_size: int, n_entities: int, d: int = 256, n_head: int = 2, n_latent: int = 4, enc_layers: int = 2,
                 core_blocks: int = 2, mode: str = "shared", n_untied: int = 8, rope_base: float = 100.0,
                 latent_pos0: int = 262, ans_pos0: int = 266):
        super().__init__()
        assert mode in ("shared", "single", "untied")
        self.mode = mode; self.d = d; self.n_latent = n_latent; self.ans_pos0 = ans_pos0
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.rope = Rope(d // n_head, rope_base)
        self.enc = nn.ModuleList([EncLayer(d, n_head) for _ in range(enc_layers)]); self.enc_ln = nn.LayerNorm(d)
        self.latents = nn.Parameter(torch.zeros(n_latent, d))
        self.cores = nn.ModuleList([Core(d, n_head, core_blocks) for _ in range(n_untied if mode == "untied" else 1)])
        self.ent_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_entities))          # auxiliary entity read-out of slot 0
        self.z_ln = nn.LayerNorm(d)
        self.dec_ln1 = nn.LayerNorm(d); self.dec_self = Attn(d, n_head); self.dec_ln2 = nn.LayerNorm(d); self.dec_cross = Attn(d, n_head)
        self.dec_ln3 = nn.LayerNorm(d); self.dec_ff = ffn(d); self.dec_lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab_size, bias=False)
        self.register_buffer("latent_pos", torch.arange(latent_pos0, latent_pos0 + n_latent), persistent=False)

    # ------------------------------------------------------------------------------------------------------------ init
    def seeded_init(self, seed: int):
        """Every tensor is drawn from its own generator keyed by (seed, parameter name): identically named tensors are
        identical across conditions (A-E exactly; F's embedding / encoder / cores.0 / decoder / heads equal A's)."""
        def gen(name):
            return torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:12], 16))
        with torch.no_grad():
            for name, m in self.named_modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    m.weight.copy_(torch.empty(m.weight.shape).normal_(0.0, 0.02, generator=gen(name + ".weight")))
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.weight.fill_(1.0); m.bias.zero_()
            self.latents.copy_(torch.empty(self.latents.shape).normal_(0.0, 0.02, generator=gen("latents")))
            for core in self.cores:                       # residual branches of the core start as the identity map
                for blk in core.blocks:
                    blk.self_att.o.weight.zero_(); blk.cross.o.weight.zero_(); blk.ff[2].weight.zero_()

    # --------------------------------------------------------------------------------------------------------- forward
    def encode(self, tok, pos, pad):
        x = self.tok_emb(tok); rope = self.rope(pos)
        for layer in self.enc:
            x = layer(x, rope, pad)
        return self.enc_ln(x), rope

    def max_T(self) -> Optional[int]:
        return {"shared": None, "single": 1, "untied": len(self.cores)}[self.mode]

    def think(self, M, rope_m, pad, T: int, collect: bool = False, decode_at: Optional[List[int]] = None):
        """Z_t = Core(Z_{t-1}, M), t = 1..T. collect -> per-loop slot-0 states and read-head probabilities (slot 0, head 0 of
        the last block's cross-attention). decode_at -> also return the latent state after those loop counts (evaluation
        of several budgets in one pass; identical to separate runs because the loop is deterministic)."""
        mt = self.max_T(); assert mt is None or T <= mt, f"mode {self.mode} supports T <= {mt}"
        B = M.shape[0]; z = self.latents[None].expand(B, -1, -1); rope_z = self.rope(self.latent_pos)
        states, reads, snaps = [], [], {}
        for t in range(T):
            core = self.cores[t] if self.mode == "untied" else self.cores[0]
            z, probs = core(z, M, rope_z, rope_m, pad)
            if collect:
                states.append(z[:, 0]); reads.append(probs[:, 0, 0, :])
            if decode_at and (t + 1) in decode_at:
                snaps[t + 1] = z
        return z, states, reads, snaps

    def decode(self, z, ans_in):
        """z: final latent state (B, n_latent, C) -- the decoder never sees M. ans_in: (B, La) prefix starting with ANSWER."""
        La = ans_in.shape[1]; pos = torch.arange(self.ans_pos0, self.ans_pos0 + La, device=ans_in.device)
        rope_a = self.rope(pos); rope_z = self.rope(self.latent_pos); zk = self.z_ln(z)
        y = self.tok_emb(ans_in); n = self.dec_ln1(y)
        y = y + self.dec_self(n, n, rope_a, rope_a, causal=True)[0]
        y = y + self.dec_cross(self.dec_ln2(y), zk, rope_a, rope_z)[0]
        y = y + self.dec_ff(self.dec_ln3(y))
        return self.head(self.dec_lnf(y))

    def forward(self, tok, pos, pad, T: int, ans_in=None, collect: bool = False):
        """The ONLY inputs are the complete packet (tok, pos, pad), the budget T and (teacher-forced) answer prefix for the
        decoder. Gold entities / read addresses are never arguments of this function."""
        M, rope_m = self.encode(tok, pos, pad)
        z, states, reads, _ = self.think(M, rope_m, pad, T, collect=collect)
        logits = self.decode(z, ans_in) if ans_in is not None else None
        return dict(logits=logits, z=z, states=states, reads=reads, M=M)

    @torch.no_grad()
    def generate(self, z, answer_id: int, end_id: int, max_new: int = 4):
        """greedy over the full vocabulary, no format forcing; returns (B, max_new) (tokens after END_ANSWER are padding = -1)."""
        B = z.shape[0]; seq = torch.full((B, 1), answer_id, dtype=torch.long, device=z.device)
        done = torch.zeros(B, dtype=torch.bool, device=z.device); out = []
        for _ in range(max_new):
            nxt = self.decode(z, seq)[:, -1].argmax(-1)
            out.append(torch.where(done, torch.full_like(nxt, -1), nxt)); done = done | (nxt == end_id)
            seq = torch.cat([seq, nxt[:, None]], 1)
        return torch.stack(out, 1)

    def param_groups_count(self):
        cnt = lambda ms: sum(p.numel() for m in ms for p in m.parameters())
        return dict(embedding=self.tok_emb.weight.numel(), encoder=cnt([self.enc, self.enc_ln]), latents=self.latents.numel(),
                    core_total=cnt([self.cores]), core_single=cnt([self.cores[0]]), n_cores=len(self.cores),
                    decoder=cnt([self.z_ln, self.dec_ln1, self.dec_self, self.dec_ln2, self.dec_cross, self.dec_ln3, self.dec_ff, self.dec_lnf]),
                    output_head=self.head.weight.numel(), aux_entity_head=cnt([self.ent_head]), total=sum(p.numel() for p in self.parameters()))
