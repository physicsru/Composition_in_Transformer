"""
Local executor protocol "L" (docs/experiments_loop_next.md §4): every network call sees ONE local observation
(current entity, head of the instruction stream) and answers with an action:

  OBS STATE <e_x> HEAD <r_j> ANS   ->  STEP <e_y> ADVANCE END_CALL        y = r_j(x)      (a STEP call, 6 + 4 tokens)
  OBS STATE <e_x> HEAD EOS   ANS   ->  HALT <e_x> END_CALL                                (a HALT call, 6 + 3 tokens)

The runner keeps the relation list, reads the current HEAD, executes ADVANCE (moves the pointer), stores the entity
the model returned and resets the context between calls (arm L) or keeps the calls of one task in context (arm
L-history, RoPE positions restart at 0 in every call, causal mask over the whole record).

Training compiles the shallow source questions into call sequences (§7.2):
  d2 (x, r, s)      -> STEP(x, r), STEP(r(x), s), HALT(s(r(x)))                     one task, 3 calls
  w2 (x1,r1),(x2,r2) -> [STEP(x1, r1), HALT(r1(x1))]  and  [STEP(x2, r2), HALT(r2(x2))]   two independent tasks
Every source-question presentation supplies 2 supervised STEP calls + 1 supervised HALT call: a d2 has one HALT; of
the two w2 HALTs a frozen RNG picks one for the loss, the other stays visible (in the record) without loss.
Loss per source question  L_row = 0.5 * mean(L_STEP1, L_STEP2) + 0.5 * L_HALT_selected  with L_call = mean CE over
the call's output tokens, i.e. per-token weights 1/16 (STEP) and 1/6 (selected HALT); prompt tokens weight 0.

Vocabulary: the frozen chain vocabulary (data/vocab.json) + EXTRA_TOKENS appended in this order; all new arms
(including the looped full-sequence model R) share it.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from .chain_format import ChainVocab

EXTRA_TOKENS = ["OBS", "STATE", "HEAD", "EOS", "STEP", "ADVANCE", "HALT", "END_CALL"]
PROMPT_LEN, STEP_LEN, HALT_LEN, MAX_OUT = 6, 4, 3, 6
W_STEP, W_HALT = 0.5 * 0.5 / STEP_LEN, 0.5 / HALT_LEN


def extend_vocab(vocab: List[str]) -> List[str]:
    assert not (set(EXTRA_TOKENS) & set(vocab)), "vocabulary already extended"
    return list(vocab) + EXTRA_TOKENS


class LocalVocab(ChainVocab):
    def __init__(self, vocab: List[str]):
        super().__init__(vocab)
        for t in EXTRA_TOKENS:
            setattr(self, t, self.tok2id[t])


def step_prompt(v: LocalVocab, x: int, r: int) -> List[int]:
    return [v.OBS, v.STATE, v.e(x), v.HEAD, v.r(r), v.ANS]


def halt_prompt(v: LocalVocab, x: int) -> List[int]:
    return [v.OBS, v.STATE, v.e(x), v.HEAD, v.EOS, v.ANS]


def step_completion(v: LocalVocab, y: int) -> List[int]:
    return [v.STEP, v.e(y), v.ADVANCE, v.END_CALL]


def halt_completion(v: LocalVocab, x: int) -> List[int]:
    return [v.HALT, v.e(x), v.END_CALL]


def make_call(v: LocalVocab, rel, x: int, head: Optional[int]) -> Dict:
    """head = relation id or None (EOS). Returns dict(ids, w, kind, x, head, y) with w = per-token weight (1.0 units,
    scaled by the caller), prompt tokens 0."""
    if head is None:
        ids = halt_prompt(v, x) + halt_completion(v, x); w = [0.0] * PROMPT_LEN + [W_HALT] * HALT_LEN; y = x; kind = "halt"
    else:
        y = int(rel[head][x]); ids = step_prompt(v, x, head) + step_completion(v, y)
        w = [0.0] * PROMPT_LEN + [W_STEP] * STEP_LEN; kind = "step"
    return dict(ids=ids, w=w, kind=kind, x=x, head=head, y=y)


def compile_d2(v: LocalVocab, rel, q: Dict) -> List[List[Dict]]:
    """-> list of tasks (one), each a list of calls."""
    x, r, s = q["x"], q["r"], q["s"]; b = int(rel[r][x]); y = int(rel[s][b])
    return [[make_call(v, rel, x, r), make_call(v, rel, b, s), make_call(v, rel, y, None)]]


def compile_w2(v: LocalVocab, rel, q: Dict) -> List[List[Dict]]:
    """-> two independent tasks of 2 calls each (the HALT loss selection is applied by the trainer)."""
    out = []
    for (x, r) in (q["q1"], q["q2"]):
        t = int(rel[r][x])
        out.append([make_call(v, rel, x, r), make_call(v, rel, t, None)])
    return out


def parse_call_output(v: LocalVocab, out_ids: Sequence[int]) -> Dict:
    """Classify a raw generated call output (up to MAX_OUT tokens). Legal outputs are exactly
    STEP <e> ADVANCE END_CALL  or  HALT <e> END_CALL; anything else is a format error."""
    seq = list(out_ids)
    res = dict(action=None, entity=None, legal=False, raw=seq)
    if len(seq) >= STEP_LEN and seq[0] == v.STEP and v.is_e(seq[1]) and seq[2] == v.ADVANCE and seq[3] == v.END_CALL:
        res.update(action="STEP", entity=v.ent(seq[1]), legal=True, n_used=STEP_LEN)
    elif len(seq) >= HALT_LEN and seq[0] == v.HALT and v.is_e(seq[1]) and seq[2] == v.END_CALL:
        res.update(action="HALT", entity=v.ent(seq[1]), legal=True, n_used=HALT_LEN)
    else:
        res["n_used"] = len(seq)
    return res


def score_run(d: int, gold_states: Sequence[int], steps: List[Dict]) -> Dict:
    """steps: the runner's per-call records dict(pointer, state_in, head, action, entity, legal) in call order.
    A run is correct iff it made exactly d legal STEPs with the right entities and then a legal HALT returning the
    final state. Error types (first error): format, early_halt (HALT while relations remain), overrun (STEP at EOS),
    state_update (wrong entity at a STEP), halt_entity (HALT returned a different entity), no_halt (call budget hit)."""
    res = dict(final_correct=False, traj_correct=False, n_calls=len(steps), first_error_pos=None, first_error_type=None,
               n_steps_done=0, own_update_correct=0, own_update_total=0, halted=False)
    cur = None
    for i, st in enumerate(steps):
        if not st["legal"]:
            res.update(first_error_pos=i, first_error_type=st.get("error", "format")); return res
        if st["head"] is None:            # EOS
            if st["action"] == "STEP":
                res.update(first_error_pos=i, first_error_type="overrun"); return res
            res["halted"] = True
            ok = st["entity"] == st["state_in"]
            if not ok:
                res.update(first_error_pos=i, first_error_type="halt_entity"); return res
            res["final_correct"] = (st["entity"] == gold_states[-1]) and res["n_steps_done"] == d
            res["traj_correct"] = res["final_correct"] and res["own_update_correct"] == d
            if not res["final_correct"] and res["first_error_pos"] is None:
                res.update(first_error_pos=i, first_error_type="propagated")
            return res
        if st["action"] == "HALT":
            res.update(first_error_pos=i, first_error_type="early_halt"); return res
        # a STEP with a relation head
        res["own_update_total"] += 1
        res["own_update_correct"] += int(st["entity"] == st["expected"])
        if st["entity"] != gold_states[res["n_steps_done"]] and res["first_error_pos"] is None:
            res["first_error_pos"] = i
            res["first_error_type"] = "state_update" if st["entity"] != st["expected"] else "propagated"
        res["n_steps_done"] += 1
    if res["first_error_pos"] is None:
        res.update(first_error_pos=len(steps), first_error_type="no_halt")
    return res
