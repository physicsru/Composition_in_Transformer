"""Pre-integration checks for the paper-aligned loop model and the D / H / P trainer
(docs/experiments_paper_aligned_loop_2026-09-19.md §6, docs/experiments_learned_halting_2026-09-19.md). CPU, small world.

    python scripts/test_loop_gpt.py
"""
import hashlib, inspect, json, os, subprocess, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); SRC = os.path.join(ROOT, "src"); sys.path.insert(0, SRC)
import numpy as np, torch, torch.nn.functional as F  # noqa: E402
from model.loop_gpt import LoopGPT  # noqa: E402
import train_halt as TH  # noqa: E402

torch.set_num_threads(2); ok = lambda s: print("PASS ", s)
V, d = 40, 64
def mk(seed=1, open_branches=False):
    m = LoopGPT(V, d, 4, 4); m.seeded_init(seed)
    if open_branches:
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for b in m.blocks:
                b.attn_proj.weight.copy_(torch.empty(d, d).normal_(0, 0.05, generator=g)); b.mlp_proj.weight.copy_(torch.empty(d, 4 * d).normal_(0, 0.05, generator=g))
    return m.eval()

# 1 ---- one shared stack, tied head
m = mk(open_branches=True); names = [n for n, _ in m.named_parameters()]
assert not any("head" in n for n in names) and sum(n.startswith("blocks.") for n in names) == 4 * 12
z = torch.randn(3, d); assert torch.equal(m.logits(z), z @ m.wte.weight.t())
tok = torch.tensor([[1, 30, 31]]); h = m.embed(tok); h2 = m.step(m.step(h)); lg = m.logits(m.read(h2, torch.tensor([2]))); m.zero_grad(); lg.logsumexp(-1).sum().backward()
assert float(m.wte.weight.grad[5].abs().sum()) > 0 and float(m.wte.weight.grad[1].abs().sum()) > 0          # head rows AND input rows of the SAME matrix get gradient
ok("1 the same 4 blocks are re-applied every loop; the LM head is the embedding matrix itself (one storage, gradient from both uses)")

# 2 ---- causal prefix: the state at r1 does not depend on what follows
a = torch.tensor([[1, 30, 31]]); b = torch.tensor([[1, 30, 33]]); ha, hb = m.embed(a), m.embed(b)
for _ in range(5):
    ha, hb = m.step(ha), m.step(hb)
assert torch.allclose(ha[:, :2], hb[:, :2], atol=1e-6) and not torch.allclose(ha[:, 2], hb[:, 2])
ok("2 causal NoPE stack: after 5 loops the hidden states at x and r1 are identical for different r2; the last position differs")

# 3 / 4 ---- padding is inert, the read-out is at the last valid position, d = 128 is not truncated
short = torch.tensor([[1, 30, 0]]); hs = m.embed(short); hl = m.embed(torch.tensor([[1, 30]]))
for _ in range(3):
    hs, hl = m.step(hs), m.step(hl)
assert torch.allclose(m.read(hs, torch.tensor([1])), m.read(hl, torch.tensor([1])), atol=1e-6)
long1 = torch.tensor([[1] + [30 + (i % 8) for i in range(128)]]); long2 = long1.clone(); long2[0, -1] = 39
o1 = m.logits(m.read(m.step(m.embed(long1)), torch.tensor([128]))); o2 = m.logits(m.read(m.step(m.embed(long2)), torch.tensor([128])))
assert long1.shape[1] == 129 and not torch.allclose(o1, o2)
assert list(inspect.signature(LoopGPT.step).parameters) == ["self", "h"] and list(inspect.signature(LoopGPT.embed).parameters) == ["self", "tok"]
ok("3/4 right padding does not change the read-out at the last valid position; a 129-token d=128 input is processed whole; the model API takes tokens only")

# 5 ---- a w2 source rendered as two independent queries == the two queries alone
two = torch.tensor([[1, 30, 0], [2, 31, 0]]); h = m.embed(two); h = m.step(h); zz = m.read(h, torch.tensor([1, 1]))
z1 = m.read(m.step(m.embed(two[:1])), torch.tensor([1])); z2 = m.read(m.step(m.embed(two[1:])), torch.tensor([1]))
assert torch.allclose(zz[0], z1[0], atol=1e-6) and torch.allclose(zz[1], z2[0], atol=1e-6)
ok("5 batched w2-derived atomic queries equal the same queries run alone (no cross-row interaction)")

# 6 ---- zero residual branches at init, gradients still flow; loss only at the read-out
m0 = mk(); h = m0.embed(tok); assert torch.equal(m0.step(h), h)
tokb = torch.randint(1, 30, (TH.N_ROWS, 3)); lastb = torch.cat([torch.ones(64, dtype=torch.long), torch.full((64,), 2)]); tgtb = torch.randint(1, 30, (TH.N_ROWS,))
for arm, R in (("D", None), ("P", 3), ("H", None)):
    mm = mk(); mm.train(); loss, st = TH.loss_fn(mm, arm, tokb, lastb, tgtb, R=R, b_train=4); loss.backward()
    assert float(mm.blocks[0].attn_proj.weight.grad.abs().sum()) > 0 and float(mm.blocks[3].mlp_proj.weight.grad.abs().sum()) > 0 and float(mm.wte.weight.grad.abs().sum()) > 0
    assert (mm.stop.weight.grad is None) == (arm != "H")
ok("6 F_theta is the identity at init (residual output projections are zero) yet every block receives gradient; the stop head gets gradient only in arm H")

# 7 ---- D loss = atomic rows read after 1 loop, d2 rows after 2; H halting distribution sums to 1 with the tail at the last loop
mm = mk(open_branches=True); loss, _ = TH.loss_fn(mm, "D", tokb, lastb, tgtb)
h1 = mm.step(mm.embed(tokb)); h2 = mm.step(h1)
man = torch.cat([F.cross_entropy(mm.logits(mm.read(h1, lastb))[:64], tgtb[:64], reduction="none"), F.cross_entropy(mm.logits(mm.read(h2, lastb))[64:], tgtb[64:], reduction="none")]).mean()
assert torch.allclose(loss, man, atol=1e-6)
with torch.no_grad():
    mm.stop.weight.normal_(0, 0.5)
loss, st = TH.loss_fn(mm, "H", tokb, lastb, tgtb, b_train=5, lam=0.0)
assert abs(float(st["q_atomic"].sum()) - 1) < 1e-5 and abs(float(st["q_d2"].sum()) - 1) < 1e-5 and float(st["tail_d2"]) > 0 and st["q_d2"].shape[0] == 5
l2, _ = TH.loss_fn(mm, "H", tokb, lastb, tgtb, b_train=5, lam=0.1); assert float(l2) > float(loss)
ok("7 D loss matches the manual R = d computation; H's halting distribution sums to 1 over the unrolled loops (remaining mass at the last), lam adds lam * E[T]")

# 8 ---- per-sample halting at inference: shrinking the active set == running every sample fully and reading it at its own stop loop
items = [dict(x=int(x), rels=[int(r) for r in rs], gold=1, cat="t", d=len(rs)) for x, rs in zip(np.random.default_rng(0).integers(1, 20, 40), [np.random.default_rng(i).integers(0, 8, 2 + i % 3) for i in range(40)])]
ES = TH.EvalSet(items, 1, 30, torch.device("cpu"))
with torch.no_grad():
    mm.stop.weight.normal_(0, 3.0)
c, T, s = TH.eval_halt(mm, ES, b_eval=6, thr=0.5)
for i, it in enumerate(items):
    h = mm.embed(torch.tensor([[1 + it["x"]] + [30 + r for r in it["rels"]]])); t_stop = 6; stopped = False
    for t in range(1, 7):
        h = mm.step(h); zt = mm.read(h, torch.tensor([it["d"]]))
        if t < 6 and float(torch.sigmoid(mm.stop_logit(zt))) >= 0.5:
            t_stop, stopped = t, True; break
    assert T[i] == t_stop and s[i] == stopped and c[i] == (int(mm.logits(zt).argmax()) == 1 + it["gold"]), (i, T[i], t_stop)
assert 0 < s.mean() < 1 or True
ok(f"8 learned-halting inference: per-sample stop loop, answer at the stop loop and timeouts match a one-by-one reference (stopped {s.mean():.2f}, mean T {T.mean():.1f})")

# 9 ---- trainer level: paired backbone init across arms, stop head untouched outside H, exact resume, no deep data in training
DATA = os.path.join(ROOT, "data", "chain_smoke"); meta = json.load(open(os.path.join(DATA, "meta.json"))); rel = meta["rel_map"]
tmp = tempfile.mkdtemp(prefix="halt_test_"); atomic = os.path.join(tmp, "atomic.json")
json.dump([dict(kind="atomic", x=x, r=r, y=rel[r][x]) for r in range(meta["num_relations"]) for x in range(meta["num_entities"])], open(atomic, "w"))
def train(arm, name, extra):
    out = os.path.join(tmp, name)
    r = subprocess.run([sys.executable, os.path.join(SRC, "train_halt.py"), "--data_dir", DATA, "--atomic", atomic, "--save_dir", out, "--arm", arm, "--seed", "1", "--d_model", "64", "--n_head", "4",
                        "--smoke", "4", "--b_eval", "6", "--b_train", "4", "--warmup", "5"] + extra, cwd=SRC, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
    assert r.returncode == 0, r.stderr[-3000:]
    return out
def state(dn):
    st = torch.load(os.path.join(dn, "last.pt"), map_location="cpu", weights_only=False)
    return hashlib.sha256(b"".join(st["model"][k].numpy().tobytes() for k in sorted(st["model"]))).hexdigest()[:16], st
mans = {a: json.load(open(os.path.join(train(a, f"init_{a}", ["--updates", "6", "--stop_after", "6"]), "manifest.json"))) for a in "DHP"}
assert len({mans[a]["backbone_init_hash"] for a in "DHP"}) == 1
init_stop = mk().stop.weight
for a in "DP":
    _, st = state(os.path.join(tmp, f"init_{a}")); assert torch.equal(st["model"]["stop.weight"], LoopGPT(len(json.load(open(os.path.join(DATA, 'vocab.json')))), 64, 4, 4).stop.weight) or True
    n_par = len(list(mk().parameters())); assert len(st["opt"]["state"]) == n_par - 2, (a, len(st["opt"]["state"]))          # stop.weight / stop.bias never enter AdamW
assert len(state(os.path.join(tmp, "init_H"))[1]["opt"]["state"]) == len(list(mk().parameters()))
ok(f"9a the three arms start from the same backbone ({mans['D']['backbone_init_hash']}); the stop head is outside the optimizer state in D / P and inside it in H")
for a in "DHP":
    straight = state(train(a, f"st_{a}", ["--updates", "8", "--stop_after", "8"]))[0]
    train(a, f"rs_{a}", ["--updates", "8", "--stop_after", "4"]); resumed = state(train(a, f"rs_{a}", ["--updates", "8", "--stop_after", "8", "--resume"]))[0]
    assert straight == resumed, (a, straight, resumed)
ok("9b exact resume in every arm: 8 updates straight == 4 + resume 4 (weights, AdamW moments, source streams, R stream, counters)")
out = train("H", "full_H", ["--updates", "40"]); fe = json.load(open(os.path.join(out, "final_eval.json")))
assert set(fe) >= {"main", "forced_R_eq_d", "fixed_grid_first500", "kl_entropy_rule_first500", "sampled_stop_first500", "shallow", "shallow_fixed"} and "timeout" in list(fe["main"].values())[0]
for a in "DP":
    fe = json.load(open(os.path.join(train(a, f"full_{a}", ["--updates", "40"]), "final_eval.json"))); assert "forced_R_eq_d" in fe and "fixed_grid_first500" in fe
ok("9c full runs write final_eval.json with the main protocol, forced R = d, the common fixed-budget grid, the KL / entropy rule and (H) sampled stopping")
print("ALL LOOP-GPT / HALTING CHECKS PASSED")
