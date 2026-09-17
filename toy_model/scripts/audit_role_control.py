"""
Independent audit of a role-control dataset (P1/P2 of docs/propose_experiment.md §8.2), computed from the
ACTUAL train.json / test.json rows rather than from the builder's bookkeeping.

    python scripts/audit_role_control.py data/skills_RC_k4 [--expect_cov 400 --expect_exposure 8]

Checks and prints: per held-out relation H the second-slot / first-slot covered facts, U-set violations,
reserved-partner violations, exposures per covered fact (min/max), H->H training rows, test/train overlap of
every composition test type, and dataset hashes. Exit code 1 on any violated assertion.
"""
import argparse, collections, hashlib, json, os, re, sys

TOK = re.compile(r"<e_(\d+)>|<r_(\d+)>")


def parse_tasks(text):
    toks = [(int(a) if a else None, int(b) if b else None) for a, b in TOK.findall(text)]
    out, i = [], 0
    while i < len(toks):
        h = toks[i][0]; i += 1; chain = []
        while i < len(toks) and toks[i][1] is not None:
            chain.append(toks[i][1]); i += 1
        out.append((h, tuple(chain), toks[i][0])); i += 1
    return out


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--expect_cov", type=int, default=None, help="expected covered facts per H per active slot (default: N - role_unseen)")
    ap.add_argument("--expect_exposure", type=int, default=None, help="expected supervised answers per covered fact (default: meta.role_exposure)")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    rel = meta["rel_map"]; E = meta["num_entities"]; H = list(meta["heldout_relations"])
    train = json.load(open(os.path.join(args.data, "train.json")))
    test = json.load(open(os.path.join(args.data, "test.json")))
    split = meta["role_split"]; slots = meta["role_slots"]
    exp_cov = args.expect_cov if args.expect_cov is not None else E - meta["role_unseen"]
    exp_exp = args.expect_exposure if args.expect_exposure is not None else meta["role_exposure"]

    role1 = collections.Counter(); role2 = collections.Counter(); pairs = set(); hh = 0
    atomic = collections.Counter(); comps = set(); n_answers = 0
    for r in train:
        for h, chain, t in parse_tasks(r["target_text"]):
            n_answers += 1
            if len(chain) == 1:
                atomic[(h, chain[0])] += 1
            elif len(chain) >= 2:
                comps.add((h, chain))
                x = h
                for k, j in enumerate(chain):
                    (role1 if k == 0 else role2)[(x, j)] += 1
                    x = rel[j][x]
                for a, c in zip(chain, chain[1:]):
                    pairs.add((a, c))
                    hh += int(a in H and c in H)
    ok = True
    print(f"{args.data}: slots={slots} k={meta['role_k']} exposure={meta['role_exposure']} unseen={meta['role_unseen']} reserved={meta['role_reserved']}")
    print(f"  train rows {len(train)}  supervised answers/epoch {n_answers}  unique comps {len(comps)}  H->H training rows {hh}")
    ok &= hh == 0
    print(f"  {'H':>4} {'slot':>6} {'covered':>8} {'U-viol':>6} {'exp min/max':>12} {'atomic U':>9} {'reserved viol':>13} partners")
    for j in H:
        sp = split[str(j)]
        for slot, cnt, S, U in (("second", role2, sp["S2"], sp["U2"]), ("first", role1, sp["S1"], sp["U1"])):
            active = slot in ({"none": [], "first": ["first"], "second": ["second"], "both": ["first", "second"]}[slots])
            covered = sum(1 for b in range(E) if cnt[(b, j)] > 0)
            uviol = sum(1 for b in U if cnt[(b, j)] > 0)
            exps = [cnt[(b, j)] for b in S]
            atomic_u = min(atomic[(b, j)] for b in U)
            partners = sorted({a for (a, c) in pairs if c == j}) if slot == "second" else sorted({c for (a, c) in pairs if a == j})
            rviol = sum(1 for t in sp["reserved"] if t in partners)
            print(f"  {j:>4} {slot:>6} {covered:>8} {uviol:>6} {min(exps):>5}/{max(exps):<6} {atomic_u:>9} {rviol:>13} {partners}")
            ok &= uviol == 0 and rviol == 0 and atomic_u > 0
            if active:
                ok &= covered == exp_cov and min(exps) == exp_exp and max(exps) == exp_exp and len(partners) == meta["role_k"]
            else:
                ok &= covered == 0
    # test/train overlap
    train_txt = {r["target_text"] for r in train}
    by_type = collections.Counter(); overlap = collections.Counter()
    for r in test:
        by_type[r["type"]] += 1
        if r["target_text"] in train_txt:
            overlap[r["type"]] += 1
    print("  test rows per type (overlap with training rows):")
    for t, n in sorted(by_type.items()):
        # width rows are recombinations of atomic facts; an exact repeat of a training grouping is benign (reported, not a violation)
        flag = "" if overlap[t] == 0 or t.startswith(("atomic", "d2_train_seen", "w")) else "   <-- VIOLATION"
        print(f"    {t:>26}: {n:6d}  overlap {overlap[t]}{flag}")
        if flag:
            ok = False
    hashes = {k: sha(os.path.join(args.data, f"{k}.json")) for k in ("train", "test", "meta", "vocab")}
    print(f"  hashes: {hashes}")
    # first-hop coverage of the rc2 new-pair tests by the base data (proposal 8.2: report, do not count a first-hop gap as transfer)
    rc2 = [r for r in test if r["type"].startswith("rc2_newpair")]
    r1cov = sum(1 for r in rc2 if role1[(parse_tasks(r["target_text"])[0][0], parse_tasks(r["target_text"])[0][1][0])] > 0)
    print(f"  rc2_newpair items whose first-hop fact (h, T) was a first hop in training rows: {r1cov}/{len(rc2)}")
    json.dump(dict(hashes=hashes, train_rows=len(train), answers_per_epoch=n_answers, unique_comps=len(comps), hh_training_rows=hh,
                   test_counts=dict(by_type), test_overlap=dict(overlap), rc2_newpair_role1_covered=[r1cov, len(rc2)], passed=bool(ok)),
              open(os.path.join(args.data, "audit.json"), "w"), indent=1)
    print("  AUDIT", "PASSED" if ok else "FAILED", f"(written to {os.path.join(args.data, 'audit.json')})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
