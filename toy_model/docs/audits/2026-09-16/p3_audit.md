# P3 code and existing-prediction audit

Date: 2026-09-16. Read-only audit; no training, model execution, or remote writes.
Scope: seven seed-1 runs in p3_audit.json; F_perrel4_ckpt timeline was read separately and is not included in this exported recomputation.

## Confirmed correction: random_src is an ordinary random bridge intervention

patch_bridge.py takes block-0 output at position 1 from [e_x,r_y,e_answer].
The causal mask and per-token LayerNorm/MLP in src/model/gpt2.py imply that this vector is exactly the same as the block-0 position-1 vector from [e_x,r_y,r2] with the same prefix.
It cannot establish that atomic-context states are unreadable in composition.
The script records src_b=None and reports accuracy against the ORIGINAL target, omitting the relevant alternative answer r2(r_y(x)).
Recomputing that alternative from existing predictions gives:

| Run / target | Follows recomputed source answer | Original target correct |
| --- | ---: | ---: |
| F_long / T->T | 897/900 = 0.9967 | 1/900 |
| RC_k4 / T->T | 900/900 = 1.0000 | 1/900 |
| RC_k4 / rc2 role seen | 822/900 = 0.9133 | 1/900 |
| RC_k4 / rc2 role unseen | 834/900 = 0.9267 | 1/900 |
| RC_first4 / rc1 role unseen | 898/900 = 0.9978 | 4/900 |

All random sources choose their first relation from T, never H.
Their bridges are uniform over the 500 entities because T relations are permutations.
For RC_k4 / target role-seen, random sources divide into source-role-seen 717/719 follows and source-role-unseen 105/181 follows.
For RC_k4 / target role-unseen, they divide into 714/716 and 120/184.
Thus the overall 0.91-0.93 rate mixes roughly 80% seen and 20% unseen source bridge facts; compare the unseen source stratum, not the mixture, to wrong_bridge_unseen.

## Confirmed sample-selection limitation

The script takes the FIRST max_items rows of each type; it does not randomly or evenly sample H relations.
The RC files are ordered by relation and partner. Existing pos1 no_patch records show:

| Run / type | Targets | Unique r1 | Unique r2 | Consequence |
| --- | ---: | ---: | ---: | --- |
| RC_k1 and RC_k4 / rc2 role seen | 300 | 1 | 1 | One H and one target T partner |
| RC_k1 and RC_k4 / rc2 role unseen | 300 | 3 | 1 | One H and three target T partners |
| RC_first4 / rc1 role seen | 300 | 1 | 1 | One H and one target T partner |
| RC_first4 / rc1 role unseen | 300 | 1 | 3 | One H and three target T partners |
| RC_both4 / hh_cov_both | 300 | 1 | 1 | Only one H1,H2 pair |
| RC_both4 / hh_cov_other | 300 | 1 | 2 | One first-hop H, two second-hop H relations |

The older F pilot samples cover multiple H relations; that does not extend the NEW RC patching results to all eight H relations.
Source-side T diversity is not target-side H diversity.
Exact IDs, counts, per-condition numerators, probabilities, margins and file hashes are in p3_audit.json.

## Supported interpretation

The block-0 first-relation residual causally controls the downstream answer in the tested examples.
Its position cannot see r2 or the final answer because the model is causal, so this is not trivial direct final-answer injection.
Cross-prefix same-bridge substitutions and wrong-bridge alternative-answer following support a reusable intermediate interface within the tested scope.
In RC_first4's sampled single-H slice, a T-derived bridge state rescues 14/300 to 900/900; in RC_both4's sampled single H1,H2 pair, rescue raises 191/300 to 843/900.
These localize a failure to computation BEFORE, OR COMPATIBILITY AT, that intervention boundary. They do not distinguish failure to encode the bridge from failure to encode it in a usable format.

## Overclaims to avoid

- random_src near-zero original accuracy does not show atomic states cannot be read; the corrected following metric shows the opposite for T->T.
- Whole-residual replacement does not establish exact head circuitry, a unique semantic bridge code, or complete context independence.
- A same-bridge patch that does not rescue does not by itself prove absence of second-hop factual knowledge; destination-side routing/gating and other preserved residual information remain possible.
- Position 0 and destination position 2 are both unchanged. Differences across target contexts cannot be uniquely assigned to position 2.
- The new RC patching is one training seed and one H (or H pair) per main slice.
- Three sources per target are repeated interventions, not 900 independent targets.
- Source sampling changes across checkpoints and positions because a single RNG advances throughout execution; the timeline is not paired on an identical source manifest.

## Reproduction

The adjacent p3_audit_recompute.py is a standalone, read-only remote analysis script.
From this local directory:

```sh
ssh -S /Users/ruwang/.ssh/miyabi-codex.sock -o BatchMode=yes miyabi python - < p3_audit_recompute.py > p3_audit.json
```

This reads existing remote artifacts and writes only the local redirected output.

