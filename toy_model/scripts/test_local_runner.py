"""Plan §12 P0.3: verify the local-executor runner and scorer with a CORRECT finite table and with injected faults
(wrong entity, early HALT, missing HALT / STEP at EOS, illegal token). No model, no GPU: a scripted 'model' emits
the table's completion; the real batched runner (train_local.run_tasks) is exercised through a stub whose logits
select the scripted tokens, so parsing, feedback of the model's own state, the call budget and scoring are tested
exactly as in evaluation.

    python scripts/test_local_runner.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, torch  # noqa: E402
from data.local_format import (EXTRA_TOKENS, MAX_OUT, PROMPT_LEN, LocalVocab, extend_vocab, halt_completion, parse_call_output,
                               score_run, step_completion)  # noqa: E402
from train_local import replay, run_tasks  # noqa: E402

E, R = 12, 3
rng = np.random.default_rng(0)
rel = [rng.permutation(E).tolist() for _ in range(R)]
vocab = extend_vocab(["<pad>"] + [f"<e_{i}>" for i in range(E)] + [f"<r_{j}>" for j in range(R)] + ["Q", "ANS", "END"])
v = LocalVocab(vocab)


class ScriptedModel(torch.nn.Module):
    """Emits, for the LAST call in the context (tokens after the last OBS), the scripted completion for (state, head)."""

    def __init__(self, table):
        super().__init__(); self.table = table

    def forward(self, x, pad_mask=None, pos_ids=None, n_loops=1):
        B, L = x.shape; logits = torch.full((B, L, len(vocab)), -10.0)
        for b in range(B):
            seq = x[b].tolist(); start = max(i for i, t in enumerate(seq) if t == v.OBS)
            state = v.ent(seq[start + 2]); head = None if seq[start + 4] == v.EOS else v.relid(seq[start + 4])
            comp = self.table[(state, head)]; k = len(seq) - (start + PROMPT_LEN)   # tokens already generated in this call
            nxt = comp[k] if k < len(comp) else v.pad
            logits[b, -1, nxt] = 10.0
        return logits


def correct_table():
    t = {}
    for r in range(R):
        for x in range(E):
            t[(x, r)] = step_completion(v, rel[r][x])
    for x in range(E):
        t[(x, None)] = halt_completion(v, x)
    return t


def as_lookup(table):
    """the scripted completions in the format of train_local.local_table"""
    out = {}
    for k, comp in table.items():
        pc = parse_call_output(v, comp + [v.pad] * (MAX_OUT - len(comp)))
        out[k] = dict(action=pc["action"], entity=pc["entity"], legal=pc["legal"], raw=comp, margin=1.0)
    return out


def chains(n=30, d=5):
    out = []
    for _ in range(n):
        x = int(rng.integers(E)); rels = [int(r) for r in rng.integers(R, size=d)]; st = []; cur = x
        for r in rels:
            cur = rel[r][cur]; st.append(cur)
        out.append(dict(x=x, rels=rels, states=st, cat="t", d=d))
    return out


def run(table, tasks, history):
    return run_tasks(ScriptedModel(table), v, rel, tasks, history, torch.device("cpu"), rollout_batch=7)


tasks = chains()
ok = True
for history in (False, True):
    res = run(correct_table(), tasks, history)
    assert all(sc["traj_correct"] and sc["final_correct"] and sc["n_calls"] == 6 for _, sc, _ in res), "correct table must be correct"
    print(f"history={history}: correct table -> 30/30 trajectories correct, 6 calls each")
# 1. wrong entity at one (x, r): every chain that visits it fails with state_update (or propagated later), others pass
t = correct_table(); x0, r0 = tasks[0]["x"], tasks[0]["rels"][0]; t[(x0, r0)] = step_completion(v, (rel[r0][x0] + 1) % E)
res = run(t, tasks, False)
sc0 = res[0][1]; assert not sc0["final_correct"] and sc0["first_error_type"] == "state_update" and sc0["first_error_pos"] == 0 and sc0["halted"], sc0
assert sc0["own_update_correct"] == 4 and sc0["own_update_total"] == 5, sc0     # the remaining 4 updates from the WRONG state are right
n_fail = sum(not sc["traj_correct"] for _, sc, _ in res); print(f"wrong entity injected at one input: chain 0 -> state_update at call 0, halted with wrong final; {n_fail} chains fail")
# 2. early HALT at a relation head
t = correct_table(); t[(x0, r0)] = halt_completion(v, x0)
sc = run(t, tasks[:1], False)[0][1]; assert sc["first_error_type"] == "early_halt" and sc["n_calls"] == 1 and not sc["halted"], sc
print("early HALT injected: early_halt at call 0, run stops, not counted as halted")
# 3. STEP at EOS (missing HALT): the chain's end entity y
y = tasks[0]["states"][-1]; t = correct_table(); t[(y, None)] = step_completion(v, y)
sc = run(t, tasks[:1], False)[0][1]; assert sc["first_error_type"] == "overrun" and sc["n_steps_done"] == 5 and sc["n_calls"] == 6, sc
print("STEP at EOS injected: overrun at call 5 after 5 correct steps")
# 4. illegal token
t = correct_table(); t[(x0, r0)] = [v.STEP, v.e(rel[r0][x0]), v.HALT, v.END_CALL]
sc = run(t, tasks[:1], False)[0][1]; assert sc["first_error_type"] == "format" and sc["n_calls"] == 1, sc
t[(x0, r0)] = [v.Q, v.END_CALL]; sc = run(t, tasks[:1], False)[0][1]; assert sc["first_error_type"] == "format", sc
print("illegal outputs injected: format error, run stops")
# 5. HALT with a different entity
t = correct_table(); t[(y, None)] = halt_completion(v, (y + 1) % E)
sc = run(t, tasks[:1], False)[0][1]; assert sc["first_error_type"] == "halt_entity" and not sc["final_correct"], sc
print("HALT returning another entity: halt_entity error")
# 6. table replay diagnostics agree with the runner and isolate update vs control errors
t = correct_table(); t[(x0, r0)] = step_completion(v, (rel[r0][x0] + 1) % E); t[(y, None)] = step_completion(v, y)
lk = as_lookup(t); real = run(t, tasks[:1], False)[0]
mm, msteps = replay(lk, rel, tasks[0], "model", "model")
assert [(s["action"], s["entity"], s["legal"]) for s in real[2]] == [(s["action"], s["entity"], s["legal"]) for s in msteps], "replay must agree with the runner"
mc = replay(lk, rel, tasks[0], "model", "correct")[0]; cm = replay(lk, rel, tasks[0], "correct", "model")[0]; cc = replay(lk, rel, tasks[0], "correct", "correct")[0]
assert mc["first_error_type"] == "state_update" and not mc["final_correct"], mc          # update error remains under correct control
assert cm["first_error_type"] == "overrun" and cm["own_update_correct"] == 5, cm         # control error remains under correct update (true states)
assert cc["traj_correct"], cc
# missing entity is reported, never filled in: HALT at a relation head under correct control
t = correct_table(); t[(x0, r0)] = halt_completion(v, x0); mc = replay(as_lookup(t), rel, tasks[0], "model", "correct")[0]
assert mc["first_error_type"] == "missing_update", mc
print("table replay: agrees with the runner; model/correct -> state_update, correct/model -> overrun, correct/correct -> ok, missing update reported as missing_update")
# 7. history arm: the same faults through the concatenated-record runner
t = correct_table(); t[(x0, r0)] = step_completion(v, (rel[r0][x0] + 1) % E)
sc = run(t, tasks[:1], True)[0][1]; assert sc["first_error_type"] == "state_update" and sc["halted"], sc
print("history runner: wrong entity -> state_update, run continues to HALT with the model's own states")
print("ALL RUNNER TESTS PASSED")
