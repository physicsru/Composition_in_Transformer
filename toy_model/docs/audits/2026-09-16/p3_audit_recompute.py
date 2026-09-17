"""Read-only P3 audit. Execute on Miyabi with Python 3; writes JSON to stdout only."""
import collections
import hashlib
import json
from pathlib import Path

BASE = Path("/work/go39/b20033/code/toy_model/Analogy_in_Transformer/toy_model")
CELLS = ["F_long", "F_perrel1", "F_pairleak8",
         "RC_k1", "RC_k4", "RC_first4", "RC_both4"]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = {
    "audit_date": "2026-09-16",
    "remote_base": str(BASE),
    "method": "Read existing predictions only; no model execution or remote writes. pos=1 is the first relation token. Numerators are intervention draws, not independent targets.",
    "source_files": {
        str(p.relative_to(BASE)): digest(p)
        for p in [BASE / "scripts/patch_bridge.py", BASE / "src/model/gpt2.py",
                  BASE / "scripts/role_tags.py"]
    },
    "runs": {}
}

for cell in CELLS:
    run = BASE / "runs" / ("skills_" + cell + "_s1")
    data = BASE / "data" / ("skills_" + cell)
    pred_path = run / "patch_bridge/predictions.jsonl"
    meta = json.loads((data / "meta.json").read_text())
    vocab = json.loads((data / "vocab.json").read_text())
    tok = {v: i for i, v in enumerate(vocab)}
    rel = meta["rel_map"]
    train_relations = set(meta["train_relations"])
    heldout_relations = set(meta["heldout_relations"])
    # Recompute role2 exposure from actual serialized local tasks.
    import re
    pattern = re.compile(r"<e_(\d+)>|<r_(\d+)>")
    role2 = set()
    for row in json.loads((data / "train.json").read_text()):
        tokens = [(int(a) if a else None, int(b) if b else None)
                  for a, b in pattern.findall(row["target_text"])]
        i = 0
        while i < len(tokens):
            state = tokens[i][0]
            i += 1
            chain = []
            while i < len(tokens) and tokens[i][1] is not None:
                chain.append(tokens[i][1])
                i += 1
            i += 1  # Skip the supervised final entity of this local task.
            if len(chain) >= 2:
                for hop, relation in enumerate(chain):
                    if hop:
                        role2.add((state, relation))
                    state = rel[relation][state]

    selections = collections.defaultdict(lambda: {
        "r1": set(), "r2": set(), "pairs": set(), "items": set(),
        "heads": set(), "bridges": set(), "correct": 0, "n": 0})
    condition_counts = collections.defaultdict(lambda: {
        "n": 0, "correct": 0, "follow_n": 0, "follows": 0,
        "sum_p_target": 0.0, "sum_margin": 0.0,
        "source_r1_ids": set(), "source_r1_T": 0, "source_r1_H": 0})
    random_source_counts = collections.defaultdict(lambda: {
        "n": 0, "correct": 0, "follows_recomputed": 0,
        "source_r1_ids": set(), "source_r1_T": 0, "source_r1_H": 0})
    total_rows = 0
    for line in pred_path.open():
        x = json.loads(line)
        total_rows += 1
        if x["pos"] != 1:
            continue
        stem = (x["ckpt"], x["layer"], x["type"])
        if x["cond"] == "no_patch":
            a = selections[stem]
            a["r1"].add(x["r1"]); a["r2"].add(x["r2"])
            a["pairs"].add((x["r1"], x["r2"]))
            a["items"].add((x["h"], x["r1"], x["r2"]))
            a["heads"].add(x["h"]); a["bridges"].add(x["b"])
            a["correct"] += x["correct"]; a["n"] += 1
        ck = stem + (x["role2_seen"], x["cond"])
        a = condition_counts[ck]
        a["n"] += 1; a["correct"] += x["correct"]
        a["sum_p_target"] += x["p_t"]; a["sum_margin"] += x["margin"]
        if "follows_src" in x:
            a["follow_n"] += 1; a["follows"] += x["follows_src"]
        if x["cond"] != "no_patch":
            a["source_r1_ids"].add(x["src_r1"])
            a["source_r1_T"] += int(x["src_r1"] in train_relations)
            a["source_r1_H"] += int(x["src_r1"] in heldout_relations)
        if x["cond"] == "random_src":
            source_bridge = rel[x["src_r1"]][x["src_h"]]
            source_answer = rel[x["r2"]][source_bridge]
            source_role2_seen = (source_bridge, x["r2"]) in role2
            # Record both combined and disaggregated source-coverage pools.
            for pool in ["all", "seen" if source_role2_seen else "unseen"]:
                rk = stem + (x["role2_seen"], pool)
                a = random_source_counts[rk]
                a["n"] += 1; a["correct"] += x["correct"]
                a["follows_recomputed"] += int(x["pred"] == tok[f"<e_{source_answer}>"])
                a["source_r1_ids"].add(x["src_r1"])
                a["source_r1_T"] += int(x["src_r1"] in train_relations)
                a["source_r1_H"] += int(x["src_r1"] in heldout_relations)

    sel_rows = []
    for k, a in sorted(selections.items()):
        sel_rows.append(dict(ckpt=k[0], layer=k[1], type=k[2], n=a["n"],
            correct=a["correct"], accuracy=a["correct"]/a["n"],
            unique_r1=len(a["r1"]), r1_ids=sorted(a["r1"]),
            unique_r2=len(a["r2"]), r2_ids=sorted(a["r2"]),
            unique_pairs=len(a["pairs"]), pairs=sorted(a["pairs"]),
            unique_targets=len(a["items"]), unique_heads=len(a["heads"]),
            unique_bridges=len(a["bridges"])))
    cond_rows = []
    for k, a in sorted(condition_counts.items()):
        cond_rows.append(dict(ckpt=k[0], layer=k[1], type=k[2],
            target_role2_seen=k[3], condition=k[4],
            n=a["n"], correct=a["correct"], accuracy=a["correct"]/a["n"],
            follow_n=a["follow_n"], follows=a["follows"],
            follow_rate=a["follows"]/a["follow_n"] if a["follow_n"] else None,
            mean_p_target=a["sum_p_target"]/a["n"],
            mean_margin=a["sum_margin"]/a["n"],
            source_r1_ids=sorted(a["source_r1_ids"]),
            source_r1_T=a["source_r1_T"], source_r1_H=a["source_r1_H"]))
    random_rows = []
    for k, a in sorted(random_source_counts.items()):
        random_rows.append(dict(ckpt=k[0], layer=k[1], type=k[2],
            target_role2_seen=k[3], source_role2_pool=k[4],
            n=a["n"], original_target_correct=a["correct"],
            original_target_accuracy=a["correct"]/a["n"],
            follows_recomputed=a["follows_recomputed"],
            follow_rate_recomputed=a["follows_recomputed"]/a["n"],
            source_r1_ids=sorted(a["source_r1_ids"]),
            source_r1_T=a["source_r1_T"], source_r1_H=a["source_r1_H"]))
    result["runs"][cell] = {
        "training_seed": 1, "predictions_sha256": digest(pred_path),
        "meta_sha256": digest(data/"meta.json"),
        "vocab_sha256": digest(data/"vocab.json"),
        "total_prediction_rows": total_rows,
        "all_relations_permutations": all(sorted(r)==list(range(meta["num_entities"])) for r in rel),
        "target_selection_pos1": sel_rows,
        "condition_counts_pos1": cond_rows,
        "random_source_follow_recomputed_pos1": random_rows}
print(json.dumps(result, separators=(",", ":")))

