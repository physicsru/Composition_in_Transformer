"""
Evaluate a skills checkpoint on the held-out composition tests, splitting each type by whether the
relation pair was leaked into training (`leak_pairs_chosen` in meta.json) or never composed.

    python scripts/split_heldout.py runs/skills_F_perrel4_s1 [--data data/skills_F_perrel4] [--ckpt epoch0600.pt]

Prints accuracy per (test type, pair status) and, for the never-composed pairs of d2_mixed_second /
d2_mixed_first, the accuracy per held-out relation with the number of leaked pairs that relation has.
Run from toy_model/ with a torch-capable python.
"""
import argparse, collections, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch  # noqa: E402
from data.dataset import CompDataset, collate_pad  # noqa: E402
from model import GPT2LikeEncoder  # noqa: E402

REL = re.compile(r"<r_(\d+)>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=None, help="data dir (default: derived from the run name)")
    ap.add_argument("--ckpt", default=None, help="checkpoint file name (default: latest epochNNNN.pt)")
    ap.add_argument("--types", default="d2_mixed_second,d2_mixed_first,d2_heldout,d2_leakpair_unseen")
    args = ap.parse_args()
    run = args.run.rstrip("/")
    cell = os.path.basename(run).replace("skills_", "").rsplit("_s", 1)[0]
    data = args.data or os.path.join(os.path.dirname(run), "..", "data", f"skills_{cell}")
    meta = json.load(open(os.path.join(data, "meta.json")))
    chosen = {tuple(p) for p in meta.get("leak_pairs_chosen", [])}
    ho = set(meta["heldout_relations"])
    ck_path = os.path.join(run, args.ckpt) if args.ckpt else max(
        glob.glob(os.path.join(run, "epoch*.pt")), key=lambda f: int(re.search(r"epoch(\d+)", f).group(1)))
    ck = torch.load(ck_path, map_location="cpu")
    model = GPT2LikeEncoder(len(ck["vocab"]), **ck["config"])
    model.load_state_dict(ck["model"])
    model.eval()
    ds = CompDataset(os.path.join(data, "test.json"), os.path.join(data, "vocab.json"), 64, expect_type=True)
    types = set(args.types.split(","))
    by_status = collections.defaultdict(lambda: [0, 0])
    by_rel = collections.defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for i in range(len(ds)):
            it = ds.items[i]
            if it["type"] not in types:
                continue
            pair = tuple(int(x) for x in REL.findall(it["target_text"]))
            status = "leaked_pair" if pair in chosen else "never_composed"
            b = collate_pad([ds[i]])
            logits = model(b["input_ids"], pad_mask=b["pad_mask"])
            pos = b["loss_mask"][0].nonzero()[-1].item()
            ok = int(logits[0, pos].argmax().item() == b["target_ids"][0, pos].item())
            by_status[(it["type"], status)][0] += ok
            by_status[(it["type"], status)][1] += 1
            if status == "never_composed" and it["type"] in ("d2_mixed_second", "d2_mixed_first"):
                j = pair[1] if it["type"] == "d2_mixed_second" else pair[0]
                by_rel[(it["type"], j)][0] += ok
                by_rel[(it["type"], j)][1] += 1
    print(f"{run} @ {os.path.basename(ck_path)}   (chosen pairs: {len(chosen)})")
    for k in sorted(by_status):
        n = by_status[k][1]
        print(f"  {k[0]:>20} {k[1]:>15}: {by_status[k][0] / n:.3f} (n={n})")
    if chosen:
        n_pairs = {("d2_mixed_second", j): sum(1 for a, b in chosen if b == j) for j in ho}
        n_pairs.update({("d2_mixed_first", j): sum(1 for a, b in chosen if a == j) for j in ho})
        pooled = collections.defaultdict(lambda: [0, 0])
        for k, (c, n) in sorted(by_rel.items()):
            pooled[(k[0], n_pairs[k])][0] += c
            pooled[(k[0], n_pairs[k])][1] += n
        for t in ("d2_mixed_second", "d2_mixed_first"):
            row = {kp[1]: f"{v[0] / v[1]:.3f}(n={v[1]})" for kp, v in sorted(pooled.items()) if kp[0] == t}
            if row:
                print(f"  never-composed {t} by #leaked pairs of the held-out relation: {row}")


if __name__ == "__main__":
    main()
