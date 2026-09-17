"""
Error-type breakdown for depth-2 test items  <e_h><r1><r2> -> <e_t>,  bridge b = r1(h).

Every prediction is classified into the first matching category:
  correct          pred == t = r2(r1(h))
  bridge           pred == r1(h)              (stopped after the first hop)
  r2_only          pred == r2(h)              (applied r2 to the head, skipped r1)
  swap             pred == r1(r2(h))          (hops in the wrong order)
  r1_twice         pred == r1(r1(h))
  r2_twice         pred == r2(r2(h))
  atomic_of_h      pred == r_j(h) for some other relation j     (a one-hop answer for h)
  atomic_of_b      pred == r_j(b) for some other relation j     (second hop with the wrong relation)
  train_pair_ans   pred is the answer of a training composition with the same (r1, r2)   (pair memorised)
  train_head_ans   pred is the answer of a training composition with the same head h
  other
Also reports the mean probability the model assigns to the correct answer, the bridge, and r2(h),
the top-1 confidence, and (for two-hop targets) how often the bridge is in the top-5.

    python scripts/error_types.py runs/skills_X_d2_s1 [--ckpts epoch1400.pt] [--types d2_train_unseen_instance,d2_train_unseen_pair]
"""
import argparse, collections, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from role_tags import parse_tasks, load_model  # noqa: E402

CATS = ["correct", "bridge", "r2_only", "swap", "r1_twice", "r2_twice", "atomic_of_h", "atomic_of_b",
        "train_pair_ans", "train_head_ans", "other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpts", default=None, help="comma-separated; default: last saved checkpoint")
    ap.add_argument("--types", default="d2_train_unseen_instance,d2_train_unseen_pair")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = args.run.rstrip("/")
    cell = os.path.basename(run).replace("skills_", "").rsplit("_s", 1)[0]
    data = args.data or os.path.join(os.path.dirname(run), "..", "data", f"skills_{cell}")
    out_dir = args.out or os.path.join(run, "error_types")
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(data, "meta.json")))
    rel = meta["rel_map"]; E = meta["num_entities"]; R = len(rel)
    vocab = json.load(open(os.path.join(data, "vocab.json")))
    tok2id = {t: i for i, t in enumerate(vocab)}
    e_id = lambda e: tok2id[f"<e_{e}>"]; r_id = lambda r: tok2id[f"<r_{r}>"]
    ent_ids = torch.tensor([e_id(e) for e in range(E)])

    # training composition answers by pair and by head
    pair_ans = collections.defaultdict(set); head_ans = collections.defaultdict(set)
    for r in json.load(open(os.path.join(data, "train.json"))):
        for h, chain, t in parse_tasks(r["target_text"]):
            if len(chain) == 2:
                pair_ans[chain].add(t); head_ans[h].add(t)

    items = []
    for it in json.load(open(os.path.join(data, "test.json"))):
        if it["type"] not in args.types.split(","):
            continue
        (h, chain, t), = parse_tasks(it["target_text"])
        if len(chain) != 2:
            continue
        items.append((it["type"], h, chain[0], chain[1], rel[chain[0]][h], t))

    if args.ckpts:
        ckpts = [os.path.join(run, c) for c in args.ckpts.split(",")]
    else:
        allck = sorted(glob.glob(os.path.join(run, "epoch*.pt")), key=lambda f: int(re.search(r"epoch(\d+)", f).group(1)))
        ckpts = allck[-1:]
    summary = {}
    fout = open(os.path.join(out_dir, "predictions.jsonl"), "w")
    for ck_path in ckpts:
        model, _ = load_model(ck_path); ck = os.path.basename(ck_path)
        ids = torch.tensor([[e_id(h), r_id(r1), r_id(r2)] for _, h, r1, r2, _, _ in items])
        with torch.no_grad():
            logits = torch.cat([model(ids[i:i + 512])[:, -1, :] for i in range(0, len(ids), 512)])
        probs = torch.softmax(logits, -1)
        counts = collections.defaultdict(collections.Counter)
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for n, (typ, h, r1, r2, b, t) in enumerate(items):
            p = probs[n]; pred = int(logits[n].argmax())
            pe = {e: e_id(e) for e in (t, b)}
            cand = {
                "correct": t, "bridge": b, "r2_only": rel[r2][h], "swap": rel[r1][rel[r2][h]],
                "r1_twice": rel[r1][b], "r2_twice": rel[r2][rel[r2][h]],
            }
            cat = "other"
            for c in ("correct", "bridge", "r2_only", "swap", "r1_twice", "r2_twice"):
                if pred == e_id(cand[c]):
                    cat = c; break
            if cat == "other":
                pred_e = int(vocab[pred].strip("<e_>")) if vocab[pred].startswith("<e_") else None
                if pred_e is not None:
                    if any(rel[j][h] == pred_e for j in range(R)):
                        cat = "atomic_of_h"
                    elif any(rel[j][b] == pred_e for j in range(R)):
                        cat = "atomic_of_b"
                    elif pred_e in pair_ans[(r1, r2)]:
                        cat = "train_pair_ans"
                    elif pred_e in head_ans[h]:
                        cat = "train_head_ans"
            counts[typ][cat] += 1
            top5 = set(torch.topk(logits[n], 5).indices.tolist())
            a = agg[typ]
            a["p_correct"].append(float(p[pe[t]])); a["p_bridge"].append(float(p[pe[b]]))
            a["p_r2_only"].append(float(p[e_id(rel[r2][h])])); a["p_top1"].append(float(p.max()))
            a["bridge_in_top5"].append(float(pe[b] in top5)); a["correct_in_top5"].append(float(pe[t] in top5))
            a["p_train_pair_ans"].append(float(p[[e_id(x) for x in pair_ans[(r1, r2)]]].sum()) if pair_ans[(r1, r2)] else 0.0)
            fout.write(json.dumps(dict(ckpt=ck, type=typ, h=h, r1=r1, r2=r2, b=b, t=t, pred=vocab[pred], cat=cat,
                                       p_correct=float(p[pe[t]]), p_bridge=float(p[pe[b]]), p_top1=float(p.max()))) + "\n")
        summary[ck] = {typ: dict(counts=dict(counts[typ]), **{k: float(np.mean(v)) for k, v in agg[typ].items()}) for typ in counts}
        print(f"\n== {run} @ {ck}")
        print(f"  {'type':26} {'n':>4} " + " ".join(f"{c:>13}" for c in CATS))
        for typ in sorted(counts):
            n = sum(counts[typ].values())
            print(f"  {typ:26} {n:>4} " + " ".join(f"{counts[typ][c] / n:13.3f}" for c in CATS))
        print(f"  {'type':26} {'p_correct':>9} {'p_bridge':>9} {'p_r2only':>9} {'p_pairans':>9} {'p_top1':>7} {'br@5':>6} {'ok@5':>6}")
        for typ in sorted(agg):
            a = summary[ck][typ]
            print(f"  {typ:26} {a['p_correct']:9.3f} {a['p_bridge']:9.3f} {a['p_r2_only']:9.3f} {a['p_train_pair_ans']:9.3f} {a['p_top1']:7.3f} {a['bridge_in_top5']:6.3f} {a['correct_in_top5']:6.3f}")
    fout.close()
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
