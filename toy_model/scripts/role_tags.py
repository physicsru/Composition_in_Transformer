"""
P0 diagnostic (docs/propose_experiment.md §3): tag every two-hop test item by what the TRAINING ROWS
actually contained, evaluate a checkpoint grid, and report the four-cell table.

Tags per test item (h, r1, r2 -> t, bridge b = r1(h)), computed from the actual training rows
(every local task inside multi-task / mixed rows is parsed):
  exact_instance_seen  (h, r1, r2) appeared as a training composition
  pair_seen            (r1, r2) appeared in any training composition
  role1_fact_seen      (h, r1) was used as the FIRST hop of some training composition
  role2_fact_seen      (b, r2) was used as the SECOND hop of some training composition
  atomic_seen          both atomic facts (h, r1) and (b, r2) appear in atomic supervision
Per item and checkpoint: prediction, correct, CE, p_correct, margin (logit gap to the runner-up), plus
atomic_ok1 / atomic_ok2 (the same atomic facts queried as single-task rows). Also an ignore-head
baseline: accuracy when h is replaced by a random entity (drops to chance if the model uses h).

    python scripts/role_tags.py runs/skills_F_perrel1_s1 [--data data/skills_F_perrel1]
        [--ckpts epoch1800.pt,epoch2000.pt] [--types d2_mixed_second,d2_heldout,...] [--out runs/.../role_tags]

Writes <out>/predictions.jsonl, role_coverage.json, summary.json and prints the four-cell tables.
"""
import argparse, collections, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from data.dataset import CompDataset, collate_pad  # noqa: E402
from model import GPT2LikeEncoder  # noqa: E402

TOK = re.compile(r"<e_(\d+)>|<r_(\d+)>")


def parse_tasks(text):
    """Split a row into local tasks [(h, chain, t), ...] regardless of width / depth / mixing."""
    toks = [(int(a) if a else None, int(b) if b else None) for a, b in TOK.findall(text)]
    out, i = [], 0
    while i < len(toks):
        h = toks[i][0]
        i += 1
        chain = []
        while i < len(toks) and toks[i][1] is not None:
            chain.append(toks[i][1])
            i += 1
        out.append((h, tuple(chain), toks[i][0]))
        i += 1
    return out


def training_sets(train_rows, rel):
    exact, pairs, role1, role2, atomic = set(), set(), set(), set(), set()
    exposures = collections.Counter()
    for r in train_rows:
        for h, chain, t in parse_tasks(r["target_text"]):
            if len(chain) == 1:
                atomic.add((h, chain[0]))
            elif len(chain) >= 2:
                exact.add((h, chain))
                x = h
                for i, j in enumerate(chain):
                    if i == 0:
                        role1.add((x, j))
                    else:
                        role2.add((x, j))
                    x = rel[j][x]
                for a, b in zip(chain, chain[1:]):
                    pairs.add((a, b))
                    exposures[(a, b)] += 1
    return exact, pairs, role1, role2, atomic, exposures


def load_model(ck_path):
    ck = torch.load(ck_path, map_location="cpu")
    m = GPT2LikeEncoder(len(ck["vocab"]), **ck["config"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["vocab"]


@torch.no_grad()
def predict(model, ds, indices, tok2id, override_head=None, batch_size=512):
    """Return per-item (pred, correct, ce, p_correct, margin) for the LAST supervised position of each item.
    override_head: dict idx->entity id (replaces the first token). Items are batched with padding."""
    out = {}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        exs = []
        for i in chunk:
            ex = ds[i]
            if override_head is not None and i in override_head:
                ex = dict(ex)
                ids = ex["input_ids"].clone()
                ids[0] = override_head[i]
                ex["input_ids"] = ids
            exs.append(ex)
        b = collate_pad(exs)
        logits = model(b["input_ids"], pad_mask=b["pad_mask"])
        for n, i in enumerate(chunk):
            pos = b["loss_mask"][n].nonzero()[-1].item()
            lg = logits[n, pos]
            tgt = b["target_ids"][n, pos].item()
            logp = torch.log_softmax(lg, -1)
            top2 = torch.topk(lg, 2).values
            pred = lg.argmax().item()
            margin = (lg[tgt] - (top2[1] if pred == tgt else top2[0])).item()
            out[i] = dict(pred=pred, correct=int(pred == tgt), ce=-logp[tgt].item(),
                          p_correct=logp[tgt].exp().item(), margin=margin)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpts", default=None, help="comma-separated checkpoint files (default: last 3)")
    ap.add_argument("--types", default="d2_mixed_second,d2_mixed_first,d2_heldout,d2_leakpair_unseen,d2_train_unseen_pair,d2_train_unseen_instance",
                    help="comma-separated test types; role-control runs: rc2_newpair_role_unseen,rc2_newpair_role_seen,rc1_newpair_role_seen,rc1_newpair_role_unseen,hh_cov_both,hh_cov_other")
    ap.add_argument("--max_per_type", type=int, default=0, help="subsample each type to this many items (0 = all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = args.run.rstrip("/")
    cell = os.path.basename(run).replace("skills_", "").rsplit("_s", 1)[0]
    data = args.data or os.path.join(os.path.dirname(run), "..", "data", f"skills_{cell}")
    out_dir = args.out or os.path.join(run, "role_tags")
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(data, "meta.json")))
    rel = meta["rel_map"]
    ho = set(meta["heldout_relations"])
    train_rows = json.load(open(os.path.join(data, "train.json")))
    exact, pairs, role1, role2, atomic, exposures = training_sets(train_rows, rel)
    n_ent = meta["num_entities"]

    # role coverage per held-out relation
    coverage = {}
    for j in sorted(ho):
        coverage[j] = dict(role2_facts_seen=sum(1 for b in range(n_ent) if (b, j) in role2),
                           role1_facts_seen=sum(1 for h in range(n_ent) if (h, j) in role1),
                           partners_as_second=sorted({a for (a, b) in pairs if b == j}),
                           partners_as_first=sorted({b for (a, b) in pairs if a == j}))
    json.dump(dict(coverage=coverage, n_train_rows=len(train_rows), n_exact=len(exact), n_pairs=len(pairs)),
              open(os.path.join(out_dir, "role_coverage.json"), "w"), indent=1)

    ds = CompDataset(os.path.join(data, "test.json"), os.path.join(data, "vocab.json"), 64, expect_type=True)
    tok2id = {t: i for i, t in enumerate(ds.vocab)}
    ent_id = lambda e: tok2id[f"<e_{e}>"]
    types = set(args.types.split(","))
    items = []
    per_type = collections.Counter()
    for i, it in enumerate(ds.items):
        if it["type"] not in types:
            continue
        (h, chain, t), = parse_tasks(it["target_text"])
        if len(chain) != 2:
            continue
        per_type[it["type"]] += 1
        if args.max_per_type and per_type[it["type"]] > args.max_per_type:
            continue
        b = rel[chain[0]][h]
        items.append(dict(idx=i, type=it["type"], h=h, r1=chain[0], r2=chain[1], b=b, t=t,
                          exact_instance_seen=(h, chain) in exact, pair_seen=chain in pairs,
                          role1_fact_seen=(h, chain[0]) in role1, role2_fact_seen=(b, chain[1]) in role2,
                          atomic_seen=((h, chain[0]) in atomic and (b, chain[1]) in atomic),
                          pair_exposures=exposures.get(chain, 0),
                          r1_heldout=chain[0] in ho, r2_heldout=chain[1] in ho))
    # atomic probes for the two facts of every item (single-task rows, gold format)
    probe_rows = {}
    for it in items:
        for key, (hh, rr) in (("a1", (it["h"], it["r1"])), ("a2", (it["b"], it["r2"]))):
            probe_rows.setdefault((hh, rr), None)
    probe_list = sorted(probe_rows)
    probe_ds_items = [{"input_text": f"<e_{hh}><r_{rr}>", "target_text": f"<e_{hh}><r_{rr}><e_{rel[rr][hh]}>", "type": "probe"}
                      for hh, rr in probe_list]
    probe_ds = CompDataset.__new__(CompDataset)
    probe_ds.items = probe_ds_items; probe_ds.vocab = ds.vocab; probe_ds.tok2id = ds.tok2id
    probe_ds.id2tok = ds.id2tok; probe_ds.max_len = 64; probe_ds.expect_type = True
    probe_index = {k: n for n, k in enumerate(probe_list)}

    if args.ckpts:
        ckpts = [os.path.join(run, c) for c in args.ckpts.split(",")]
    else:
        allck = sorted(glob.glob(os.path.join(run, "epoch*.pt")), key=lambda f: int(re.search(r"epoch(\d+)", f).group(1)))
        ckpts = allck[-3:]
    rng = np.random.default_rng(args.seed)
    override = {it["idx"]: ent_id(int(rng.integers(n_ent))) for it in items}

    summary = {}
    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as fout:
        for ck_path in ckpts:
            model, _ = load_model(ck_path)
            ck_name = os.path.basename(ck_path)
            preds = predict(model, ds, [it["idx"] for it in items], tok2id)
            preds_ignore_head = predict(model, ds, [it["idx"] for it in items], tok2id, override_head=override)
            probe_preds = predict(model, probe_ds, list(range(len(probe_list))), tok2id)
            cells = collections.defaultdict(lambda: [0, 0])
            ign = collections.defaultdict(lambda: [0, 0])
            for it in items:
                p = preds[it["idx"]]
                a1 = probe_preds[probe_index[(it["h"], it["r1"])]]["correct"]
                a2 = probe_preds[probe_index[(it["b"], it["r2"])]]["correct"]
                rec = dict(ckpt=ck_name, **it, **p, atomic_ok1=a1, atomic_ok2=a2,
                           pred_ignore_head_correct=preds_ignore_head[it["idx"]]["correct"])
                fout.write(json.dumps(rec) + "\n")
                key = (it["type"], it["pair_seen"], it["role2_fact_seen"], it["role1_fact_seen"], bool(a1 and a2))
                cells[key][0] += p["correct"]; cells[key][1] += 1
                ign[it["type"]][0] += preds_ignore_head[it["idx"]]["correct"]; ign[it["type"]][1] += 1
            summary[ck_name] = {"|".join(map(str, k)): v for k, v in cells.items()}
            print(f"\n== {run} @ {ck_name}   (chance 1/N = {1 / n_ent:.4f})")
            print(f"  {'type':>24} {'pair':>5} {'role2':>5} {'role1':>5} {'atomic':>6}  {'acc':>6}  {'n':>5}")
            for k in sorted(cells):
                c, n = cells[k]
                print(f"  {k[0]:>24} {str(k[1]):>5} {str(k[2]):>5} {str(k[3]):>5} {str(k[4]):>6}  {c / n:6.3f}  {n:5d}")
            print("  ignore-head baseline (random h):", {t: f"{v[0] / v[1]:.3f}" for t, v in sorted(ign.items())})
    json.dump(dict(coverage=coverage, cells=summary), open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"\nrole coverage (held-out relations): " + ", ".join(f"r{j}: role2 {c['role2_facts_seen']}/{n_ent} role1 {c['role1_facts_seen']}/{n_ent} partners2={len(c['partners_as_second'])} partners1={len(c['partners_as_first'])}" for j, c in coverage.items()))
    print(f"outputs: {out_dir}/predictions.jsonl, role_coverage.json, summary.json")


if __name__ == "__main__":
    main()
