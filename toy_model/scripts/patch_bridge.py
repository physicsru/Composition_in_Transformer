"""
P3 (docs/propose_experiment.md §6): causal bridge patching on two-hop items  <e_h><r1><r2> -> <e_t>,  b = r1(h).

For every TARGET item we build SOURCE items that share r2 and the layout, and copy the source's residual stream
(output of block `--layer`) at position `--pos` into the target's forward pass:

  no_patch        target as is
  same_bridge     source (h', r1') with r1'(h') = b, r1' != r1, r1' a train relation   -> should keep the answer t
  wrong_bridge    source (h', r1') with bridge b' != b (r1' train relation); the source bridge is drawn from the
                  role-SEEN and role-UNSEEN pools of r2 separately (wrong_bridge_seen / wrong_bridge_unseen)
                  -> "follows" = prediction becomes r2(b'), "keeps" = prediction stays t
  random_src      uniform-random bridge: source (x, T) with x uniform, T a train relation, taken from a width-1 row
                  <e_x><r_T><e_..>. Because the model is causal its block-0 state at position 1 is identical to that
                  of <e_x><r_T><r2>, so this is NOT a different kind of context; its bridge T(x) is recorded and
                  'follows' (prediction == r2(T(x))) is reported like for the other wrong-bridge conditions.

Positions: 1 = the r1 token (only h, r1 visible: a pure first-hop / bridge state), 2 = the r2 token (the answer
position). Layers: 0 = residual after block 1 (the "layer-1 residual"). Patching pos 2 of the last block's
output trivially determines the answer, so it is skipped.

Reported per target group (test type x role2_fact_seen of the TARGET) and per condition: accuracy, mean p(t),
mean margin, and for wrong-bridge conditions the fraction that follows the source bridge and the mean p(r2(b')).
Role tags are computed from the actual training rows exactly as in scripts/role_tags.py.

    python scripts/patch_bridge.py runs/skills_F_perrel1_s1 --ckpts epoch2000.pt \
        --types d2_mixed_second,d2_leakpair_unseen,d2_train_unseen_pair --layers 0 --positions 1,2 --n_src 4
"""
import argparse, collections, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from role_tags import parse_tasks, training_sets, load_model  # noqa: E402


class Patcher:
    """Capture / replace the output of model.blocks[layer] at one position."""

    def __init__(self, model, layer):
        self.block = model.blocks[layer]
        self.mode, self.pos, self.store, self.inject = None, None, None, None
        self.h = self.block.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.mode == "capture":
            self.store = out[:, self.pos, :].detach().clone()
        elif self.mode == "inject":
            out = out.clone()
            out[:, self.pos, :] = self.inject
            return out

    def close(self):
        self.h.remove()


@torch.no_grad()
def run(model, ids, patcher=None, mode=None, pos=None, inject=None):
    if patcher is not None:
        patcher.mode, patcher.pos, patcher.inject = mode, pos, inject
    logits = model(ids)
    if patcher is not None:
        patcher.mode = None
    return logits[:, -1, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpts", default=None, help="comma-separated; default: last checkpoint")
    ap.add_argument("--types", default="d2_mixed_second,d2_leakpair_unseen,d2_train_unseen_pair")
    ap.add_argument("--layers", default="0")
    ap.add_argument("--positions", default="1,2")
    ap.add_argument("--n_src", type=int, default=4, help="sources per target and condition")
    ap.add_argument("--max_items", type=int, default=400, help="per type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = args.run.rstrip("/")
    cell = os.path.basename(run_dir).replace("skills_", "").rsplit("_s", 1)[0]
    data = args.data or os.path.join(os.path.dirname(run_dir), "..", "data", f"skills_{cell}")
    out_dir = args.out or os.path.join(run_dir, "patch_bridge")
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(data, "meta.json")))
    rel = meta["rel_map"]; E = meta["num_entities"]
    T = list(meta["train_relations"]); ho = set(meta["heldout_relations"])
    inv = {j: {rel[j][h]: h for h in range(E)} for j in range(len(rel))}
    exact, pairs, role1, role2, atomic, _ = training_sets(json.load(open(os.path.join(data, "train.json"))), rel)
    test = json.load(open(os.path.join(data, "test.json")))
    vocab = json.load(open(os.path.join(data, "vocab.json")))
    tok2id = {t: i for i, t in enumerate(vocab)}
    e_id = lambda e: tok2id[f"<e_{e}>"]; r_id = lambda r: tok2id[f"<r_{r}>"]
    rng = np.random.default_rng(args.seed)
    types = args.types.split(",")

    # targets
    targets = []
    per_type = collections.Counter()
    for it in test:
        if it["type"] not in types:
            continue
        (h, chain, t), = parse_tasks(it["target_text"])
        if len(chain) != 2 or per_type[it["type"]] >= args.max_items:
            continue
        per_type[it["type"]] += 1
        b = rel[chain[0]][h]
        targets.append(dict(type=it["type"], h=h, r1=chain[0], r2=chain[1], b=b, t=t,
                            role2_seen=(b, chain[1]) in role2, role1_seen=(h, chain[0]) in role1,
                            pair_seen=chain in pairs))
    # bridge pools per r2: role-seen / role-unseen bridges
    pools = {}
    for r2 in {it["r2"] for it in targets}:
        seen = [b for b in range(E) if (b, r2) in role2]
        pools[r2] = dict(seen=seen, unseen=[b for b in range(E) if (b, r2) not in role2])

    def src_for_bridge(bb, r2, avoid_r1):
        """a train-relation first hop landing on bridge bb: (h', r1') with r1'(h') = bb."""
        cands = [r for r in T if r != avoid_r1]
        r1p = int(rng.choice(cands))
        return inv[r1p][bb], r1p

    layers = [int(x) for x in args.layers.split(",")]
    positions = [int(x) for x in args.positions.split(",")]
    if args.ckpts:
        ckpts = [os.path.join(run_dir, c) for c in args.ckpts.split(",")]
    else:
        allck = sorted([f for f in os.listdir(run_dir) if re.match(r"epoch\d+\.pt", f)], key=lambda f: int(re.search(r"\d+", f).group()))
        ckpts = [os.path.join(run_dir, allck[-1])]

    summary = {}
    fout = open(os.path.join(out_dir, "predictions.jsonl"), "w")
    for ck_path in ckpts:
        model, _ = load_model(ck_path)
        ck_name = os.path.basename(ck_path)
        n_layers = len(model.blocks)
        for layer in layers:
            patcher = Patcher(model, layer)
            for pos in positions:
                if layer == n_layers - 1 and pos == 2:
                    continue
                stats = collections.defaultdict(lambda: collections.defaultdict(list))
                for it in targets:
                    tgt_ids = torch.tensor([[e_id(it["h"]), r_id(it["r1"]), r_id(it["r2"])]])
                    grp = (it["type"], it["role2_seen"])
                    conds = {"no_patch": [(it["h"], it["r1"], it["b"])]}
                    conds["same_bridge"] = [src_for_bridge(it["b"], it["r2"], it["r1"]) + (it["b"],) for _ in range(args.n_src)]
                    for name in ("seen", "unseen"):
                        pool = [x for x in pools[it["r2"]][name] if x != it["b"]]
                        if pool:
                            picks = rng.choice(pool, size=min(args.n_src, len(pool)), replace=False)
                            conds[f"wrong_bridge_{name}"] = [src_for_bridge(int(bb), it["r2"], -1) + (int(bb),) for bb in picks]
                    # random unrelated width-1 context <e_x><r_y><e_..> : take position min(pos, 1)
                    # a width-1 row <e_x><r_T><e_..> has the SAME block-0 state at position 1 as <e_x><r_T><r2> (causal
                    # model), so this is a uniform-random-bridge source: record its bridge T(x) so 'follows' is computed
                    conds["random_src"] = [(hs, r1s, rel[r1s][hs]) for hs, r1s in
                                           ((int(rng.integers(E)), int(rng.choice(T))) for _ in range(args.n_src))]
                    for cond, srcs in conds.items():
                        for hs, r1s, bs in srcs:
                            if cond == "no_patch":
                                lg = run(model, tgt_ids)[0]
                            else:
                                if cond == "random_src":
                                    src_ids = torch.tensor([[e_id(hs), r_id(r1s), e_id(rel[r1s][hs])]])
                                    spos = min(pos, 1)
                                else:
                                    src_ids = torch.tensor([[e_id(hs), r_id(r1s), r_id(it["r2"])]])
                                    spos = pos
                                run(model, src_ids, patcher, "capture", spos)
                                vec = patcher.store
                                lg = run(model, tgt_ids, patcher, "inject", pos, vec)[0]
                            logp = torch.log_softmax(lg, -1)
                            pred = int(lg.argmax())
                            t_id = e_id(it["t"])
                            top2 = torch.topk(lg, 2).values
                            margin = float(lg[t_id] - (top2[1] if pred == t_id else top2[0]))
                            rec = dict(ckpt=ck_name, layer=layer, pos=pos, cond=cond, **{k: it[k] for k in ("type", "h", "r1", "r2", "b", "t", "role2_seen", "role1_seen", "pair_seen")},
                                       src_h=hs, src_r1=r1s, src_b=bs, pred=pred, correct=int(pred == t_id),
                                       p_t=float(logp[t_id].exp()), margin=margin)
                            if bs is not None and bs != it["b"]:
                                alt = e_id(rel[it["r2"]][bs])
                                rec["follows_src"] = int(pred == alt); rec["p_src_answer"] = float(logp[alt].exp())
                                rec["src_role2_seen"] = (bs, it["r2"]) in role2
                            fout.write(json.dumps(rec) + "\n")
                            s = stats[grp][cond]
                            s.append((rec["correct"], rec["p_t"], rec["margin"], rec.get("follows_src", np.nan), rec.get("p_src_answer", np.nan)))
                key = f"{ck_name}|L{layer}|pos{pos}"
                summary[key] = {}
                print(f"\n== {run_dir} @ {ck_name}  patch block-{layer} output at position {pos}   (n_src={args.n_src})")
                print(f"  {'type':22} {'role2':5} {'cond':20} {'n':>5} {'acc':>6} {'p_t':>6} {'margin':>7} {'follows':>8} {'p_src':>6}")
                for grp in sorted(stats):
                    for cond in ("no_patch", "same_bridge", "wrong_bridge_seen", "wrong_bridge_unseen", "random_src"):
                        if cond not in stats[grp]:
                            continue
                        a = np.array(stats[grp][cond], dtype=float)
                        row = dict(n=len(a), acc=a[:, 0].mean(), p_t=a[:, 1].mean(), margin=a[:, 2].mean(),
                                   follows=float(np.nanmean(a[:, 3])) if not np.all(np.isnan(a[:, 3])) else None,
                                   p_src=float(np.nanmean(a[:, 4])) if not np.all(np.isnan(a[:, 4])) else None)
                        summary[key][f"{grp[0]}|role2_seen={grp[1]}|{cond}"] = row
                        fol = f"{row['follows']:8.3f}" if row["follows"] is not None else f"{'-':>8}"
                        psrc = f"{row['p_src']:6.3f}" if row["p_src"] is not None else f"{'-':>6}"
                        print(f"  {grp[0]:22} {str(grp[1]):5} {cond:20} {row['n']:5d} {row['acc']:6.3f} {row['p_t']:6.3f} {row['margin']:7.2f} {fol} {psrc}")
            patcher.close()
    fout.close()
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"\noutputs: {out_dir}/predictions.jsonl, summary.json")


if __name__ == "__main__":
    main()
