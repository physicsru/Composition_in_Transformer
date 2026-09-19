"""
Full-sequence weight-shared loop Transformer aligned with "Loop, Think, & Generalize" v2
(docs/experiments_paper_aligned_loop_2026-09-19.md §1) + a per-sample stop head for the learned-halting arm
(docs/experiments_learned_halting_2026-09-19.md §1).

    H_0 = Embed([x, r_1, ..., r_d])                       one embedding pass, no positional encoding (NoPE), no input re-injection
    H_t = F_theta(H_{t-1})                                 the SAME 4-block causal GPT-2-style stack, applied once per loop
    z_t = LN_f(H_t[last valid input position])
    a_t = softmax(W_tied z_t)                              LM head tied to the input embedding (same storage)
    p_t = sigmoid(w_stop . z_t + b_stop)                   stop head (instantiated in every arm, only trained in arm H)

GPT-2 block: x + attn(LN(x)); x + mlp(LN(x)), GELU(tanh approximation = gelu_new), LayerNorm eps 1e-5, every dropout 0.
Init: N(0, 0.02) for embeddings / linear weights, zero biases, and ZERO weights for the two residual output projections of every
block (attention c_proj and MLP c_proj), so F_theta starts as the identity. Every tensor is drawn from its own generator keyed by
(seed, parameter name): the backbone init is identical across arms and independent of the stop head.
Right padding + causal attention: a valid position never attends to a later (pad) position, so no padding mask is needed
(checked in scripts/test_loop_gpt.py).
"""
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rot(x):
    d = x.size(-1); return torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head; self.rope = None                     # (cos, sin) set by LoopGPT when pos == "rope"
        self.ln_1 = nn.LayerNorm(d, eps=1e-5); self.c_attn = nn.Linear(d, 3 * d); self.attn_proj = nn.Linear(d, d)
        self.ln_2 = nn.LayerNorm(d, eps=1e-5); self.c_fc = nn.Linear(d, 4 * d); self.mlp_proj = nn.Linear(4 * d, d)

    def forward(self, x):
        B, L, C = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).split(C, dim=-1)
        sp = lambda t: t.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        if self.rope is not None:                                  # relative positions: rotate Q / K after their projections, V untouched
            cos, sin = self.rope[0][:L].to(q.dtype), self.rope[1][:L].to(q.dtype)
            q = q * cos + _rot(q) * sin; k = k * cos + _rot(k) * sin
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_proj(a.transpose(1, 2).reshape(B, L, C))
        return x + self.mlp_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate="tanh"))


class LoopGPT(nn.Module):
    def __init__(self, vocab_size: int, d: int = 768, n_head: int = 12, n_layer: int = 4, pos: str = "nope", rope_base: float = 100.0, max_pos: int = 512):
        super().__init__()
        assert pos in ("nope", "rope")
        self.d = d; self.pos = pos
        self.wte = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.stop = nn.Linear(d, 1)                       # 769 parameters; unused (grad None) outside arm H
        if pos == "rope":                                  # parameter-free: the backbone init is identical to the NoPE model's
            hd = d // n_head; inv = 1.0 / (rope_base ** (torch.arange(0, hd, 2).float() / hd))
            f = torch.einsum("l,d->ld", torch.arange(max_pos).float(), inv); emb = torch.cat([f, f], -1)
            self.register_buffer("rope_cos", emb.cos(), persistent=False); self.register_buffer("rope_sin", emb.sin(), persistent=False)

    def seeded_init(self, seed: int):
        def gen(name):
            return torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:12], 16))
        with torch.no_grad():
            for name, m in self.named_modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    m.weight.copy_(torch.empty(m.weight.shape).normal_(0.0, 0.02, generator=gen(name + ".weight")))
                    if isinstance(m, nn.Linear):
                        m.bias.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.weight.fill_(1.0); m.bias.zero_()
            for blk in self.blocks:                        # residual output projections: scale 0
                blk.attn_proj.weight.zero_(); blk.mlp_proj.weight.zero_()

    def embed(self, tok):
        return self.wte(tok)

    def step(self, h):
        """one loop = one pass through the shared stack"""
        if self.pos == "rope":
            for blk in self.blocks:
                blk.rope = (self.rope_cos, self.rope_sin)
        for blk in self.blocks:
            h = blk(h)
        return h

    def read(self, h, last):
        """z_t at the last valid input position. last: (B,) index of the last real token."""
        # gather (backward = scatter_add) instead of advanced indexing, whose backward is a slow sort-based index_put
        return self.ln_f(torch.gather(h, 1, last[:, None, None].expand(-1, 1, h.shape[-1])).squeeze(1))

    def logits(self, z):
        return z @ self.wte.weight.t()                     # tied head: the SAME parameter storage as the input embedding

    def stop_logit(self, z):
        return self.stop(z).squeeze(-1)

    def backbone_names(self):
        return [n for n, _ in self.named_parameters() if not n.startswith("stop.")]
