"""
Output protocols for the chain task (plan §4). Every question is rendered as  prompt + completion:

  prompt        Q x r_1 ... r_d ANS
  A  final      y END                                (y = r_d(...r_1(x)))
  B  entity CoT b_1 ... b_(d-1) y END                (one intermediate entity per hop, then the answer)
  C  rel+entity x r_1 b_1 r_2 b_2 ... r_d y END      (copy the state and the current relation before each update)
  D  queue CoT  x r_1 ... r_d  b_1 r_2 ... r_d  b_2 r_3 ... r_d ... b_(d-1) r_d  y END
                (after every state the model re-copies the relations that REMAIN; the relation to apply next is always
                 the token right after the current state and END follows a state with an empty remaining list, so both
                 "which relation is next" and "when to stop" are local rules that do not depend on d)

A w2 row is two independent one-step questions in one sequence:  Q x r ANS resp END Q z s ANS resp END, with
resp = t (A, B) or x r t (C); causal attention across the whole row, loss only on the two completions.

Per-token loss weights (plan §4.2): final 1, every bridge 1, END 1, the copy tokens share a total weight of 1
(mean CE), prompt 0. For w2 each sub-question's weights are halved so a row has the same total as one d2 question.
Token roles are kept so CE can be reported per role.
"""
from typing import Dict, List, Sequence, Tuple

ROLE_PROMPT, ROLE_COPY, ROLE_BRIDGE, ROLE_FINAL, ROLE_END = 0, 1, 2, 3, 4
ROLE_NAMES = ["prompt", "copy", "bridge", "final", "end"]


class ChainVocab:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.pad = self.tok2id["<pad>"]; self.Q = self.tok2id["Q"]; self.ANS = self.tok2id["ANS"]; self.END = self.tok2id["END"]
        self.E = sum(1 for t in vocab if t.startswith("<e_")); self.R = sum(1 for t in vocab if t.startswith("<r_"))
        self.e0 = self.tok2id["<e_0>"]; self.r0 = self.tok2id["<r_0>"]

    def e(self, i: int) -> int: return self.e0 + i
    def r(self, j: int) -> int: return self.r0 + j
    def is_e(self, tid: int) -> bool: return self.e0 <= tid < self.e0 + self.E
    def is_r(self, tid: int) -> bool: return self.r0 <= tid < self.r0 + self.R
    def ent(self, tid: int) -> int: return tid - self.e0
    def relid(self, tid: int) -> int: return tid - self.r0


def prompt_ids(v: ChainVocab, x: int, rels: Sequence[int]) -> List[int]:
    return [v.Q, v.e(x)] + [v.r(j) for j in rels] + [v.ANS]


def completion(v: ChainVocab, protocol: str, x: int, rels: Sequence[int], states: Sequence[int]
               ) -> Tuple[List[int], List[float], List[int]]:
    """states = [r_1(x), r_2(r_1(x)), ...] (length d). Returns (ids, weights, roles)."""
    d = len(rels); ids: List[int] = []; w: List[float] = []; roles: List[int] = []
    if protocol == "A":
        ids += [v.e(states[-1])]; w += [1.0]; roles += [ROLE_FINAL]
    elif protocol == "B":
        for b in states[:-1]:
            ids += [v.e(b)]; w += [1.0]; roles += [ROLE_BRIDGE]
        ids += [v.e(states[-1])]; w += [1.0]; roles += [ROLE_FINAL]
    elif protocol == "C":
        copies = 1 + d                        # x and every relation are copies of the prompt
        cw = 1.0 / copies
        ids += [v.e(x)]; w += [cw]; roles += [ROLE_COPY]
        for t, j in enumerate(rels):
            ids += [v.r(j)]; w += [cw]; roles += [ROLE_COPY]
            if t < d - 1:
                ids += [v.e(states[t])]; w += [1.0]; roles += [ROLE_BRIDGE]
            else:
                ids += [v.e(states[t])]; w += [1.0]; roles += [ROLE_FINAL]
    elif protocol == "D":
        copies = 1 + d * (d + 1) // 2         # x plus every remaining-list token
        cw = 1.0 / copies
        ids += [v.e(x)]; w += [cw]; roles += [ROLE_COPY]
        for t in range(d):
            for j in rels[t:]:
                ids += [v.r(j)]; w += [cw]; roles += [ROLE_COPY]
            if t < d - 1:
                ids += [v.e(states[t])]; w += [1.0]; roles += [ROLE_BRIDGE]
            else:
                ids += [v.e(states[t])]; w += [1.0]; roles += [ROLE_FINAL]
    else:
        raise ValueError(protocol)
    ids += [v.END]; w += [1.0]; roles += [ROLE_END]
    return ids, w, roles


def render_d2(v: ChainVocab, protocol: str, rel, q: Dict) -> Tuple[List[int], List[float], List[int]]:
    x, r, s = q["x"], q["r"], q["s"]; b = rel[r][x]; y = rel[s][b]
    p = prompt_ids(v, x, [r, s]); c, w, roles = completion(v, protocol, x, [r, s], [b, y])
    return p + c, [0.0] * len(p) + w, [ROLE_PROMPT] * len(p) + roles


def render_chain(v: ChainVocab, protocol: str, rel, q: Dict) -> Tuple[List[int], List[float], List[int]]:
    """Any depth: q = dict(x, rels)."""
    x = q["x"]; rels = list(q["rels"]); st = []; cur = x
    for j in rels:
        cur = rel[j][cur]; st.append(cur)
    p = prompt_ids(v, x, rels); c, w, roles = completion(v, protocol, x, rels, st)
    return p + c, [0.0] * len(p) + w, [ROLE_PROMPT] * len(p) + roles


def render_w2(v: ChainVocab, protocol: str, rel, q: Dict) -> Tuple[List[int], List[float], List[int]]:
    ids: List[int] = []; w: List[float] = []; roles: List[int] = []
    for (x, r) in (q["q1"], q["q2"]):
        t = rel[r][x]
        p = prompt_ids(v, x, [r]); c, cw, cr = completion(v, protocol, x, [r], [t])
        ids += p + c; w += [0.0] * len(p) + [0.5 * z for z in cw]; roles += [ROLE_PROMPT] * len(p) + cr
    return ids, w, roles


def completion_length(protocol: str, d: int) -> int:
    return {"A": 2, "B": d + 1, "C": 2 * d + 2, "D": (d + 1) + d * (d + 1) // 2 + 1}[protocol]


def parse_completion(v: ChainVocab, protocol: str, ids: Sequence[int], d: int) -> Dict:
    """Parse a generated completion (tokens after ANS). Returns states / rels actually produced, stop status, validity."""
    out = dict(states=[], rels=[], ended=False, n_tokens=len(ids), valid_format=True, copy_x=None)
    seq = list(ids)
    if v.END in seq:
        out["ended"] = True; seq = seq[:seq.index(v.END)]
    if protocol in ("A", "B"):
        if not all(v.is_e(t) for t in seq):
            out["valid_format"] = False
        out["states"] = [v.ent(t) for t in seq if v.is_e(t)]
    elif protocol == "D":
        # segments: entity followed by the remaining relations; rels[t] = first relation after state t (applied next)
        if not seq or not v.is_e(seq[0]):
            out["valid_format"] = False
        else:
            out["copy_x"] = v.ent(seq[0])
        segs = []; cur = None
        for t in seq:
            if v.is_e(t):
                cur = [v.ent(t), []]; segs.append(cur)
            elif v.is_r(t) and cur is not None:
                cur[1].append(v.relid(t))
            else:
                out["valid_format"] = False
        out["remaining"] = [sg[1] for sg in segs]                 # remaining list after each entity (x first)
        out["states"] = [sg[0] for sg in segs[1:]]
        out["rels"] = [sg[1][0] for sg in segs[:-1] if sg[1]]      # relation announced right after each state
        if any(not sg[1] for sg in segs[:-1]):                    # a non-final state with an empty list = premature stop
            out["valid_format"] = False
    else:
        if not seq or not v.is_e(seq[0]):
            out["valid_format"] = False
        else:
            out["copy_x"] = v.ent(seq[0])
        rest = seq[1:]
        for i in range(0, len(rest), 2):
            pair = rest[i:i + 2]
            if len(pair) < 2 or not v.is_r(pair[0]) or not v.is_e(pair[1]):
                out["valid_format"] = False; break
            out["rels"].append(v.relid(pair[0])); out["states"].append(v.ent(pair[1]))
    return out


def score_chain(v: ChainVocab, protocol: str, rel, x: int, rels: Sequence[int], gold_states: Sequence[int], parsed: Dict) -> Dict:
    """Final-answer / full-trajectory correctness, first error position + type, own-state update correctness."""
    d = len(rels); st = parsed["states"]
    res = dict(final_correct=False, traj_correct=False, first_error_pos=None, first_error_type=None,
               own_update_correct=0, own_update_total=0, stop="ok")
    if not parsed["ended"]:
        res["stop"] = "no_end"
    elif protocol == "A":
        res["stop"] = "ok" if len(st) == 1 else ("early_end" if len(st) < 1 else "overlong")
    else:
        res["stop"] = "ok" if len(st) == d else ("early_end" if len(st) < d else "overlong")
    if not parsed["valid_format"]:
        res["first_error_type"] = "format"; res["first_error_pos"] = 0
    if protocol == "A":
        res["final_correct"] = len(st) >= 1 and st[0] == gold_states[-1] and parsed["ended"] and len(st) == 1
        res["traj_correct"] = res["final_correct"]
        return res
    # B / C: step-by-step
    prev = x
    for t in range(min(d, len(st))):
        rel_t = rels[t]
        if protocol in ("C", "D"):
            if t >= len(parsed["rels"]) or parsed["rels"][t] != rels[t]:
                if res["first_error_pos"] is None:
                    res["first_error_pos"] = t; res["first_error_type"] = "relation_copy"
                rel_t = parsed["rels"][t] if t < len(parsed["rels"]) else rels[t]
        expected_own = rel[rel_t][prev]
        res["own_update_total"] += 1
        res["own_update_correct"] += int(st[t] == expected_own)
        if st[t] != gold_states[t] and res["first_error_pos"] is None:
            res["first_error_pos"] = t
            res["first_error_type"] = "state_update" if st[t] != expected_own else "propagated"
        prev = st[t]
    if len(st) < d and res["first_error_pos"] is None:
        res["first_error_pos"] = len(st); res["first_error_type"] = "early_end" if parsed["ended"] else "no_end"
    if len(st) > d and res["first_error_pos"] is None:
        res["first_error_pos"] = d; res["first_error_type"] = "overlong"
    res["final_correct"] = len(st) >= d and st[d - 1] == gold_states[-1] and res["stop"] == "ok"
    res["traj_correct"] = (res["stop"] == "ok" and parsed["valid_format"] and st[:d] == list(gold_states)
                           and (protocol not in ("C", "D") or (parsed["rels"][:d] == list(rels) and parsed["copy_x"] == x))
                           and (protocol != "D" or parsed.get("remaining") == [list(rels[t:]) for t in range(d + 1)]))
    return res
