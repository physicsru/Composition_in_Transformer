"""
Logit lens for depth-2 items  <e_h><r1><r2> -> t  (b = r1(h)): decode the residual stream after every block at
every position through ln_f + head, and report how often the top-1 token is the bridge b, the answer t, the head h,
or r2(h). Shows where (layer, position) the intermediate entity becomes linearly readable.

    python scripts/logit_lens.py runs/skills_X_d2_s1 [--ckpts epoch1400.pt] [--types d2_train_unseen_instance,...]
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from role_tags import parse_tasks, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpts", default=None)
    ap.add_argument("--types", default="d2_train_unseen_instance,d2_train_seen")
    ap.add_argument("--max_items", type=int, default=300)
    args = ap.parse_args()
    run = args.run.rstrip("/")
    cell = os.path.basename(run).replace("skills_", "").rsplit("_s", 1)[0]
    data = args.data or os.path.join(os.path.dirname(run), "..", "data", f"skills_{cell}")
    meta = json.load(open(os.path.join(data, "meta.json"))); rel = meta["rel_map"]
    vocab = json.load(open(os.path.join(data, "vocab.json"))); tok2id = {t: i for i, t in enumerate(vocab)}
    e_id = lambda e: tok2id[f"<e_{e}>"]; r_id = lambda r: tok2id[f"<r_{r}>"]
    per_type = {}
    for it in json.load(open(os.path.join(data, "test.json"))):
        if it["type"] in args.types.split(",") and len(per_type.get(it["type"], [])) < args.max_items:
            (h, chain, t), = parse_tasks(it["target_text"])
            if len(chain) == 2:
                per_type.setdefault(it["type"], []).append((h, chain[0], chain[1], rel[chain[0]][h], t))
    if args.ckpts:
        ckpts = [os.path.join(run, c) for c in args.ckpts.split(",")]
    else:
        allck = sorted(glob.glob(os.path.join(run, "epoch*.pt")), key=lambda f: int(re.search(r"epoch(\d+)", f).group(1)))
        ckpts = allck[-1:]
    for ck_path in ckpts:
        model, _ = load_model(ck_path)
        store = {}
        hooks = [blk.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, o.detach())) for L, blk in enumerate(model.blocks)]
        print(f"\n== {run} @ {os.path.basename(ck_path)}   top-1 of ln_f+head applied to the residual after block L at position P")
        print(f"  {'type':26} {'L':>2} {'P':>2} {'=bridge':>8} {'=answer':>8} {'=head':>6} {'=r2(h)':>7} {'p(bridge)':>9} {'p(answer)':>9}")
        for typ, items in per_type.items():
            ids = torch.tensor([[e_id(h), r_id(r1), r_id(r2)] for h, r1, r2, b, t in items])
            with torch.no_grad():
                model(ids)
                for L in sorted(store):
                    lg = model.head(model.ln_f(store[L]))
                    pr = torch.softmax(lg, -1)
                    for P in (1, 2):
                        top = lg[:, P].argmax(-1)
                        eb = torch.tensor([e_id(b) for *_, b, t in items]); et = torch.tensor([e_id(t) for *_, t in items])
                        eh = torch.tensor([e_id(h) for h, *_ in items]); e2 = torch.tensor([e_id(rel[r2][h]) for h, r1, r2, *_ in items])
                        n = len(items)
                        print(f"  {typ:26} {L:>2} {P:>2} {(top == eb).float().mean():8.3f} {(top == et).float().mean():8.3f} {(top == eh).float().mean():6.3f} {(top == e2).float().mean():7.3f} {pr[torch.arange(n), P, eb].mean():9.3f} {pr[torch.arange(n), P, et].mean():9.3f}")
        for hk in hooks:
            hk.remove()


if __name__ == "__main__":
    main()
