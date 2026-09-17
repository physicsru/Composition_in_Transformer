"""
Attention-level diagnostics for two-hop items <e_h><r1><r2> -> t (b = r1(h)), on the 2-layer model.

(1) Attention read pattern: for every layer and head, the attention weight from the answer position (2) to positions
    0 (h), 1 (r1 / bridge state) and 2 (self), per item group. Answers "does the second hop read the bridge position?"
(2) Head-level K/V patching in the LAST block: replace, for chosen heads, the key and/or value of position 1 by the
    ones a SAME-BRIDGE source (<e_h'><r_T'><r2>, T'(h') = b) would have produced. K-only tells whether the routing
    (where to attend) is what the source changes; V-only whether the content read there is; KV both. Compared with
    the full block-0 residual patch of scripts/patch_bridge_balanced.py.

Targets: for role-control runs the balanced manifest of patch_bridge_balanced.py (TH S2/U2, HT, HH cells; TT);
for plain skills runs (X_*, F_*) the d2_train_unseen_instance / _pair items.

    python scripts/attn_probe.py runs/skills_RC_k4_s1 --ckpts epoch525.pt
    python scripts/attn_probe.py runs/skills_X_d2_s1 --ckpts epoch1400.pt
Writes <run>/attn_probe/summary.json.
"""
import argparse, collections, glob, json, math, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from model.gpt2 import apply_rope  # noqa: E402
from role_tags import parse_tasks, training_sets, load_model  # noqa: E402
from patch_bridge_balanced import build_manifest  # noqa: E402


class AttnTap:
    """Re-implements CausalSelfAttentionRoPE.forward so that attention weights can be captured and per-head K/V at one
    position can be overridden. Installed by monkeypatching attn.forward."""

    def __init__(self, attn):
        self.attn = attn; self.orig = attn.forward
        self.capture = None            # list to collect attention weights (B, H, L, L)
        self.kv_store = None           # list to collect (k, v) tensors (B, H, L, hd) (post-RoPE k)
        self.override = None           # dict(pos=int, heads=[..], k=(B,H,hd) or None, v=(B,H,hd) or None)
        attn.forward = self.forward

    def forward(self, x, pad_mask=None):
        a = self.attn
        B, L, C = x.shape
        qkv = a.qkv(x); q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, L, a.n_head, a.head_dim).transpose(1, 2); k = k.view(B, L, a.n_head, a.head_dim).transpose(1, 2)
        v = v.view(B, L, a.n_head, a.head_dim).transpose(1, 2)
        cos, sin = a.rope(seq_len=L, device=x.device, dtype=x.dtype)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        if self.override is not None:
            o = self.override; k = k.clone(); v = v.clone()
            for h in o["heads"]:
                if o.get("k") is not None:
                    k[:, h, o["pos"], :] = o["k"][:, h, :]
                if o.get("v") is not None:
                    v[:, h, o["pos"], :] = o["v"][:, h, :]
        if self.kv_store is not None:
            self.kv_store.append((k.detach().clone(), v.detach().clone()))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(a.head_dim)
        att = att.masked_fill(a.causal_mask[:, :, :L, :L], float("-inf"))
        if pad_mask is not None:
            att = att.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.capture is not None:
            self.capture.append(att.detach().clone())
        y = (att @ v).transpose(1, 2).contiguous().view(B, L, C)
        return a.out(y)

    def close(self):
        self.attn.forward = self.orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpts", default=None)
    ap.add_argument("--max_items", type=int, default=300, help="per group")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()
    run = args.run.rstrip("/")
    cell = os.path.basename(run).replace("skills_", "").rsplit("_s", 1)[0]
    data = args.data or os.path.join(os.path.dirname(run), "..", "data", f"skills_{cell}")
    out_dir = os.path.join(run, "attn_probe"); os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(data, "meta.json"))); rel = meta["rel_map"]; E = meta["num_entities"]
    T = list(meta["train_relations"]); inv = {j: {rel[j][h]: h for h in range(E)} for j in range(len(rel))}
    _, _, role1, role2, _, _ = training_sets(json.load(open(os.path.join(data, "train.json"))), rel)
    test_items = json.load(open(os.path.join(data, "test.json")))
    vocab = json.load(open(os.path.join(data, "vocab.json"))); tok2id = {t: i for i, t in enumerate(vocab)}
    e_id = lambda e: tok2id[f"<e_{e}>"]; r_id = lambda r: tok2id[f"<r_{r}>"]
    rng = np.random.default_rng(args.seed)

    # ---- target groups
    groups = collections.OrderedDict()
    if "role_split" in meta:
        tg = build_manifest(meta, role2, test_items, args.seed)
        for t in tg:
            key = f"{t['test']}|{t['cell']}"
            groups.setdefault(key, []).append(t)
        groups = collections.OrderedDict((k, v[:args.max_items]) for k, v in groups.items() if k.split("|")[0] in ("TH", "HT", "HH", "TT"))
    else:
        for typ in ("d2_train_unseen_instance", "d2_train_unseen_pair"):
            items = []
            for it in test_items:
                if it["type"] == typ:
                    (h, chain, t), = parse_tasks(it["target_text"]); b = rel[chain[0]][h]
                    items.append(dict(h=h, r1=chain[0], r2=chain[1], b=b, t=t, test=typ, cell="-"))
            groups[f"{typ}|-"] = items[:args.max_items]
    # same-bridge sources for every target (T' != r1)
    for tgs in groups.values():
        for t in tgs:
            cands = [x for x in T if x != t["r1"]]; tp = int(rng.choice(cands)); t["src"] = (inv[tp][t["b"]], tp)

    if args.ckpts:
        ckpts = [os.path.join(run, c) for c in args.ckpts.split(",")]
    else:
        allck = sorted(glob.glob(os.path.join(run, "epoch*.pt")), key=lambda f: int(re.search(r"epoch(\d+)", f).group(1)))
        ckpts = allck[-1:]
    summary = {}
    for ck_path in ckpts:
        model, _ = load_model(ck_path); ck = os.path.basename(ck_path); nL = len(model.blocks); H = model.blocks[0].attn.n_head
        taps = [AttnTap(blk.attn) for blk in model.blocks]
        res = {}
        print(f"\n== {run} @ {ck}   attention from the answer position (2) to positions [h, r1/bridge, self], per layer/head; then same-bridge K/V patching in block {nL - 1}")
        for gname, tgs in groups.items():
            if not tgs:
                continue
            tgt = torch.tensor([[e_id(t["h"]), r_id(t["r1"]), r_id(t["r2"])] for t in tgs])
            src = torch.tensor([[e_id(t["src"][0]), r_id(t["src"][1]), r_id(t["r2"])] for t in tgs])
            t_ids = torch.tensor([e_id(t["t"]) for t in tgs])
            with torch.no_grad():
                for tp in taps: tp.capture = []
                base_logits = model(tgt)[:, -1, :]
                atts = [tp.capture[0] for tp in taps]
                for tp in taps: tp.capture = None
                base_acc = float((base_logits.argmax(-1) == t_ids).float().mean())
                read = {f"L{l}H{h}": [float(atts[l][:, h, 2, p].mean()) for p in range(3)] for l in range(nL) for h in range(H)}
                # source K/V of the last block at position 1
                last = taps[-1]; last.kv_store = []
                model(src); k_src, v_src = last.kv_store[0]; last.kv_store = None
                patch = {}
                for heads in [[h] for h in range(H)] + [list(range(H))]:
                    for mode in ("K", "V", "KV"):
                        last.override = dict(pos=1, heads=heads, k=k_src[:, :, 1, :] if "K" in mode else None, v=v_src[:, :, 1, :] if "V" in mode else None)
                        lg = model(tgt)[:, -1, :]; last.override = None
                        patch[f"heads{''.join(map(str, heads))}|{mode}"] = float((lg.argmax(-1) == t_ids).float().mean())
            res[gname] = dict(n=len(tgs), acc=base_acc, read=read, same_bridge_patch=patch)
            rd = " ".join(f"{k}:[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}]" for k, v in read.items())
            pt = " ".join(f"{k}:{v:.2f}" for k, v in patch.items())
            print(f"  {gname:14} n={len(tgs):4d} acc {base_acc:.3f} | read {rd}")
            print(f"  {'':14}        same-bridge KV patch -> acc: {pt}")
        for tp in taps: tp.close()
        summary[ck] = res
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"\noutputs: {out_dir}/summary.json")


if __name__ == "__main__":
    main()
