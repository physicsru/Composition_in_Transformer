"""Implementation checks of docs/experiments_latent_batch2_20runs.md §10 (CPU, small world data/chain_smoke).

    python scripts/test_latent_executor.py            # all checks; exits non-zero on the first failure
"""
import hashlib, inspect, json, os, subprocess, sys, tempfile, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); SRC = os.path.join(ROOT, "src"); sys.path.insert(0, SRC)
import numpy as np, torch, torch.nn.functional as F  # noqa: E402
from data.latent_format import (ADDR_EOP, N_MEM_POS, LatentVocab, TrainTables, addr_to_key, frozen_slots, render_chain, render_chain_dense)  # noqa: E402
from model.latent_executor import LatentExecutor  # noqa: E402

torch.set_num_threads(2)
DATA = os.path.join(ROOT, "data", "chain_smoke"); J = lambda n: json.load(open(os.path.join(DATA, f"{n}.json")))
meta = J("meta"); rel = meta["rel_map"]; v = LatentVocab(J("vocab")); dev = torch.device("cpu")
tables = TrainTables(v, rel, J("train_w2"), J("train_d2"), dev)
ok = lambda name: print(f"PASS  {name}")


def mk(mode="shared", d=64, seed=1, randomize_out=False):
    m = LatentExecutor(len(v.vocab), v.E, d=d, n_head=2, mode=mode); m.seeded_init(seed)
    if randomize_out:                       # the zero-init residual branches make Z constant at init; open them for the flow checks
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for core in m.cores:
                for blk in core.blocks:
                    for w in (blk.self_att.o.weight, blk.cross.o.weight, blk.ff[2].weight):
                        w.copy_(torch.empty(w.shape).normal_(0, 0.05, generator=g))
    return m


# 1 ---- data boundary: only w2 / d2 rows reach the batch builder, at most 2 relations per region, hashes as frozen
big = json.load(open(os.path.join(ROOT, "data", "chain_loop", "meta.json"))); ck1 = json.load(open(os.path.join(ROOT, "data", "chain_chainC_k1", "meta.json")))
assert all(big["hashes"][k] == ck1["hashes"][k] for k in ("train_w2", "train_d2", "val_d2", "val_w2", "vocab")) and big["counts"]["w2"] == 40000 and big["counts"]["d2"] == 10000 and big["counts"]["d3"] == 0
b = tables.batch(np.arange(64), np.arange(64), np.random.default_rng(0))
isrel = (b["tok"] >= v.r0) & (b["tok"] < v.r0 + v.R) & ~b["pad"]
assert int(isrel[:64, :4].sum(1).max()) == 1 and int(isrel[:64, 4:].sum(1).max()) == 1 and int(isrel[64:].sum(1).max()) == 2
assert set(inspect.signature(TrainTables.__init__).parameters) == {"self", "v", "rel", "train_w2", "train_d2", "device"}
assert len(v.vocab) == len(J("vocab")) + 5 and v.vocab[-5:] == ["TASK", "EOP", "EMPTY", "ANSWER", "END_ANSWER"] and J("vocab") == v.vocab[:-5]
ok("1 data: frozen hashes, 40k w2 + 10k d2 + 0 d3, <= 2 relations per program in training batches, vocabulary = base + 5")

# 2 ---- forward interface: packet, mask, budget (and the answer prefix for the decoder); gold labels are not inputs
assert list(inspect.signature(LatentExecutor.forward).parameters) == ["self", "tok", "pos", "pad", "T", "ans_in", "collect"]
assert list(inspect.signature(LatentExecutor.think).parameters) == ["self", "M", "rope_m", "pad", "T", "collect", "decode_at"]
m = mk(randomize_out=True); o1 = m(b["tok"], b["pos"], b["pad"], 4, ans_in=b["ans_in"], collect=True)
b["aux"]["x1"] = b["aux"]["x1"].roll(1); b["aux"]["addr"] = b["aux"]["addr"].roll(1, 0)
o2 = m(b["tok"], b["pos"], b["pad"], 4, ans_in=b["ans_in"], collect=True)
assert torch.equal(o1["logits"], o2["logits"]) and torch.equal(o1["reads"][1], o2["reads"][1])
ok("2 interface: forward takes (tok, pos, pad, T, answer prefix); changing gold aux labels leaves every forward output unchanged")

# 3 ---- memory is fixed within a solve, Z keeps changing, gradients reach the first loop
oT2 = m(b["tok"], b["pos"], b["pad"], 2, ans_in=b["ans_in"], collect=True); oT8 = m(b["tok"], b["pos"], b["pad"], 8, ans_in=b["ans_in"], collect=True)
assert torch.equal(oT2["M"], oT8["M"]) and torch.allclose(oT2["states"][1], oT8["states"][1])
assert all(not torch.allclose(oT8["states"][t], oT8["states"][t + 1]) for t in range(7))
cap = []
def hook(mod, inp, out):
    out[0].retain_grad(); cap.append(out[0])
hd = m.cores[0].register_forward_hook(hook); o = m(b["tok"], b["pos"], b["pad"], 8, ans_in=b["ans_in"]); hd.remove(); m.zero_grad()
F.cross_entropy(o["logits"].transpose(1, 2), b["ans_tgt"], ignore_index=-100).backward()
assert len(cap) == 8 and float(cap[0].grad.abs().sum()) > 0 and float(m.latents.grad.abs().sum()) > 0 and float(m.enc[0].att.q.weight.grad.abs().sum()) > 0
ok("3 recurrence: M identical for T=2 / T=8, Z changes every loop, the answer loss back-propagates to loop 1, Z0 and the encoder (no detach / reset)")

# 4 ---- the whole program is visible: d = 128 is not truncated and a suffix change alters memory and output
x = 3; rels = [int(r) for r in np.random.default_rng(1).integers(v.R, size=128)]; rels2 = rels[:-1] + [(rels[-1] + 1) % v.R]
t1, p1 = render_chain(v, x, rels, np.arange(128)); t2, p2 = render_chain(v, x, rels2, np.arange(128)); assert len(t1) == 131 and max(p1) == ADDR_EOP
pk = lambda t, p: (torch.tensor([t]), torch.tensor([p]), torch.zeros(1, len(t), dtype=torch.bool))
a1 = m(*pk(t1, p1), 8, ans_in=torch.tensor([[v.ANSWER]])); a2 = m(*pk(t2, p2), 8, ans_in=torch.tensor([[v.ANSWER]]))
assert not torch.allclose(a1["M"], a2["M"]) and not torch.allclose(a1["logits"], a2["logits"])
ok("4 full program: d=128 packet has 131 tokens up to the EOP address; changing the LAST relation changes M and the answer logits")

# 5 ---- dense (262 positions, EMPTY masked) == packed; padding columns are inert; read loss / gradients agree
q = dict(x=5, rels=[2, 7]); sl = frozen_slots(2, 0, 2); tp, pp = render_chain(v, q["x"], q["rels"], sl); td, pd_, md = render_chain_dense(v, q["x"], q["rels"], sl)
def run(tok, pos, pad, m_):
    tok, pos, pad = torch.tensor([tok]), torch.tensor([pos]), torch.tensor([pad])
    o = m_(tok, pos, pad, 4, ans_in=torch.tensor([[v.ANSWER, v.e0 + 1]]), collect=True)
    key = addr_to_key(pos, pad, torch.tensor([[2 + int(sl[0]), 2 + int(sl[1]), ADDR_EOP]]))
    lr = -torch.log(torch.stack(o["reads"])[:, 0].gather(1, key[0, [0, 1, 2, 2]][:, None]) + 1e-12).mean()
    m_.zero_grad(); (o["logits"].logsumexp(-1).sum() + lr).backward()
    return o["logits"].detach(), float(lr), torch.cat([p.grad.flatten() for p in m_.parameters() if p.grad is not None])
lp, rp, gp = run(tp, pp, [False] * len(tp), m); ld, rd, gd = run(td, pd_, md, m)
lq, rq, gq = run(tp + [v.pad] * 5, pp + [0] * 5, [False] * len(tp) + [True] * 5, m)
assert len(td) == N_MEM_POS and torch.allclose(lp, ld, atol=1e-5) and abs(rp - rd) < 1e-5 and torch.allclose(gp, gd, atol=1e-4) and torch.allclose(lp, lq, atol=1e-5) and torch.allclose(gp, gq, atol=1e-4)
ok(f"5 dense == packed: logits, read loss ({rp:.4f} vs {rd:.4f}) and gradients agree; extra PAD columns change nothing")

# 7 ---- parameter accounting: E == A exactly; F has 8 separate cores, its common tensors equal A's
cA, cE, cF = (mk(mode, d=256) for mode in ("shared", "single", "untied"))
nA, nE, nF = (c.param_groups_count() for c in (cA, cE, cF))
assert nA == nE and nF["n_cores"] == 8 and nF["core_total"] == 8 * nA["core_single"] and nF["total"] == nA["total"] + 7 * nA["core_single"]
ptrs = {p.data_ptr() for core in cF.cores for p in core.parameters()}; assert len(ptrs) == sum(1 for core in cF.cores for _ in core.parameters())
sa = dict(cA.named_parameters()); assert all(torch.equal(p, sa[n]) for n, p in cF.named_parameters() if n in sa) and all(torch.equal(p, q_) for (_, p), (_, q_) in zip(cA.named_parameters(), cE.named_parameters()))
assert not torch.equal(cF.cores[0].blocks[0].ff[0].weight, cF.cores[1].blocks[0].ff[0].weight)
ok(f"7 parameters (d=256): A = E = {nA['total']:,} {nA}; F = {nF['total']:,} (8 independent cores, shared-name tensors identical to A)")

# 8 ---- the decoder sees only the final Z; the teacher-forced answer never enters the latent loop
assert list(inspect.signature(LatentExecutor.decode).parameters) == ["self", "z", "ans_in"]
bb = tables.batch(np.arange(8), np.arange(8), np.random.default_rng(0)); ans2 = bb["ans_in"].clone(); ans2[:, 1] = v.e0
z1 = m(bb["tok"], bb["pos"], bb["pad"], 4, ans_in=bb["ans_in"])["z"]; z2 = m(bb["tok"], bb["pos"], bb["pad"], 4, ans_in=ans2)["z"]; assert torch.equal(z1, z2)
g = m.generate(z1, v.ANSWER, v.END_ANSWER, 4); assert g.shape == (16, 4)
ok("8 decoder: decode(z, prefix) has no memory argument; Z_T is identical for different teacher-forced answers; generation is free greedy")

# 6, 9 ---- trainer-level: A-D identical through phase 1 (aux graph not built, aux head untouched), exact resume, F core participation
PY = sys.executable; tmp = tempfile.mkdtemp(prefix="latent_test_")
def train(cond, name, extra):
    out = os.path.join(tmp, name)
    r = subprocess.run([PY, os.path.join(SRC, "train_latent.py"), "--data_dir", DATA, "--save_dir", out, "--cond", cond, "--seed", "1", "--d_model", "64", "--smoke", "4", "--no_final_eval"] + extra,
                       cwd=SRC, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
    assert r.returncode == 0, r.stderr[-2000:]
    return out
def mhash(d):
    st = torch.load(os.path.join(d, "last.pt"), map_location="cpu", weights_only=False)
    return hashlib.sha256(b"".join(st["model"][k].numpy().tobytes() for k in sorted(st["model"]))).hexdigest()[:16], st
p1 = ["--phase1_updates", "6", "--phase2_updates", "0"]
hs = {c: mhash(train(c, f"p1_{c}", p1)) for c in "ABCD"}
assert len({h for h, _ in hs.values()}) == 1, {c: h for c, (h, _) in hs.items()}
names = [n for n, _ in mk().named_parameters()]; ent_idx = [i for i, n in enumerate(names) if n.startswith("ent_head")]
assert all(i not in hs["D"][1]["opt"]["state"] for i in ent_idx), "aux head must stay untouched (grad None) in phase 1"
init = mk(); assert all(torch.equal(hs["D"][1]["model"][n], p) for n, p in init.named_parameters() if n.startswith("ent_head")), "aux head changed in phase 1"
p2 = ["--phase1_updates", "3", "--phase2_updates", "5"]
h2 = {c: mhash(train(c, f"p2_{c}", p2))[0] for c in "ABCD"}; assert len(set(h2.values())) == 4, h2
ok(f"6 A-D: identical parameters after phase 1 ({hs['A'][0]}), aux head untouched and absent from the AdamW state; all four differ after phase 2")
straight = mhash(train("D", "straight", p2))[0]
train("D", "resumed", p2 + ["--stop_after", "4"]); resumed = mhash(train("D", "resumed", p2 + ["--resume"]))[0]
assert straight == resumed, (straight, resumed)
ok(f"9 resume: 8 updates straight == 4 + resume 4 ({straight}); model, AdamW, source / layout / T streams and counters restored")
stF = mhash(train("F", "F", ["--phase1_updates", "12", "--phase2_updates", "0"]))[1]; th = stF["ctr"]["T_hist"]; cu = stF["ctr"]["core_updates"]
assert cu[0] == cu[1] == 12 and cu[2] == cu[3] == th["4"] + th["8"] and cu[4] == cu[7] == th["8"], (cu, th)
stE = mhash(train("E", "E", ["--phase1_updates", "6", "--phase2_updates", "0"]))[1]; assert stE["ctr"]["T_hist"]["1"] == 6
ok(f"7b untied prefix participation {cu} matches the T draws {th}; E always runs T = 1")

# 10 ---- size / cost on a fixed small batch (CPU timing is indicative only; GPU throughput is measured by the smoke job)
bb = tables.batch(np.arange(128), np.arange(128), np.random.default_rng(0))
for name, c, T in (("A T=8", cA, 8), ("E T=1", cE, 1), ("F T=8", cF, 8)):
    t0 = time.time(); o = c(bb["tok"], bb["pos"], bb["pad"], T, ans_in=bb["ans_in"]); F.cross_entropy(o["logits"].transpose(1, 2), bb["ans_tgt"], ignore_index=-100).backward()
    print(f"      {name}: forward + backward of 256 rows {1000 * (time.time() - t0):.0f} ms on CPU")
ok("10 cost probe done")
print("ALL LATENT EXECUTOR CHECKS PASSED")
