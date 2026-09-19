"""Read-only summary of the sixth batch (D = program sets R = d, H = learned halting, P = random budget, Hc = H + compute cost).

    python scripts/summarize_halt.py            # finished runs: final tables; unfinished runs: latest monitoring line (labelled)
Main metric per arm: D accuracy at R = d; H 'actively stopped AND correct' (timeouts and wrong stops are failures); P accuracy at the
common budget R = 8. Diagnostics on the same weights are printed separately and never replace the main metric.
"""
import glob, json, os, re

DEPTHS = (2, 3, 4, 8, 16, 32, 64, 128)
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
runs = sorted(r for r in glob.glob(os.path.join(root, "halt_*_s*")) if re.search(r"halt_[A-Za-z0-9]+_s\d+$", r) and "gtest" not in r and "bench" not in r)
lo = lambda blk, d, k="correct": blk[f"one_new|d{d}"][k]          # one_new = exactly one UNSEEN adjacent pair (random chains mix seen and unseen pairs)
has = lambda blk: [d for d in DEPTHS if f"one_new|d{d}" in blk]
print("== FINAL (full 5,000 per cell): one_new = chains with exactly one UNSEEN adjacent relation pair; fit = all_seen d2")
for r in runs:
    p = os.path.join(r, "final_eval.json"); name = os.path.basename(r)[5:]
    if not os.path.exists(p):
        continue
    j = json.load(open(p)); m = j["main"]; sh = j["shallow"]
    print(f"{name:9s} atomic {sh['atomic']['correct']:.3f} val_d2 {sh['val_d2']['correct']:.3f} fit {m['all_seen|d2']['correct']:.3f} | main " + " ".join(f"d{d}:{lo(m, d):.3f}" for d in has(m)))
    print(f"{'':9s} forced R=d                          | " + " ".join(f"d{d}:{lo(j['forced_R_eq_d'], d):.3f}" for d in has(m)))
    if "timeout" in m["one_new|d2"]:
        print(f"{'':9s} timeout / wrong stop / mean T      | " + " ".join(f"d{d}:{m[f'one_new|d{d}']['timeout']:.2f}/{m[f'one_new|d{d}']['wrong_stop']:.2f}/{m[f'one_new|d{d}']['T']:.1f}" for d in has(m)))
    g = j["fixed_grid_first500"]; print(f"{'':9s} best fixed R on one_new (R:acc)     | " + " ".join(f"d{d}:" + max(((int(k[1:]), v[f'one_new|d{d}']['correct']) for k, v in g.items()), key=lambda t: t[1]).__repr__() for d in has(m)))
print("== IN PROGRESS (monitoring set, 100 per cell -- not results)")
for r in runs:
    mp = os.path.join(r, "metrics.jsonl")
    if os.path.exists(os.path.join(r, "final_eval.json")) or not os.path.exists(mp) or not os.path.getsize(mp):
        continue
    l = json.loads(open(mp).read().strip().split("\n")[-1]); m = l["monitor"]; sh = l["shallow"]; name = os.path.basename(r)[5:]
    line = f"{name:9s} u{l['update']:7d} ({l['elapsed'] / 3600:.1f} h) atomic {sh['atomic']['correct']:.3f} val_d2 {sh['val_d2']['correct']:.3f} fit {sh['train_d2_fit']['correct']:.3f} | " + " ".join(f"d{d}:{lo(m, d):.2f}" for d in has(m))
    if "T" in m["one_new|d2"]:
        line += " | T " + "/".join(f"{m[f'one_new|d{d}']['T']:.0f}" for d in has(m)) + f" | timeout d2 {m['one_new|d2']['timeout']:.2f} d8 {m['one_new|d8']['timeout']:.2f}"
        if "monitor_forced_R_eq_d" in l:
            line += " | forced R=d " + " ".join(f"{min(l['monitor_forced_R_eq_d'][f'one_new|d{d}'], l['monitor_forced_R_eq_d'][f'random|d{d}']):.2f}" for d in DEPTHS[:4])
    print(line)
