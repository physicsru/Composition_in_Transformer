# Paper notes: arXiv:2602.01992 vs. our compositional-grokking setup

## 1. Citation

Gouki Minegishi, Jingyuan Feng, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo. **"Emergent Analogical Reasoning in Transformers."** arXiv:2602.01992 (v1 2 Feb 2026, v5 27 May 2026); comments line: "Accepted to ICML2026 (spotlight)".
Read: abstract page `https://arxiv.org/abs/2602.01992` and the full HTML rendering `https://arxiv.org/html/2602.01992` (serves v5; all sections and appendices A–T were read from the downloaded HTML, figure curves from the HTML's SVG files). The PDF was downloaded but could not be rendered on this host.

This paper is the one whose official code is this repository (Sec. 3, footnote 4). Numbers marked **[code]** below are not printed in the paper; they were obtained by running this repo's `generate_data.py` with the paper's default configuration.

**Framing caveat.** The paper is about *analogical* reasoning; two-hop composition is its baseline/control condition and is never called grokking. The paper reports no delayed generalization for composition ("Compositional reasoning closely follows the training accuracy", Sec. 3.1).

## 2. Claim

A one-layer Transformer trained on a synthetic knowledge graph with two isomorphic entity categories learns in three stages: it memorizes in-distribution facts, then generalizes to held-out two-hop compositions, and only later infers the cross-category functor (analogy) for held-out entities (Fig. 1B). Composition is robust to data, optimizer and scale; analogy needs many relations, a dense graph, moderate weight decay and moderate width, and otherwise fails or is transient (Secs. 3.1–3.3). Analogy emerges after the two categories' entity embeddings become geometrically aligned (Dirichlet energy over the functor graph drops); the functor token then attends to the source entity and is added to it in the residual stream, `e_t ≈ e_s + f` (Sec. 4). Gemma2-2B/9B show the same signature across layers under in-context analogy prompts (Sec. 5).

## 3. Task / data setup

- **Entities.** `E` is split into two disjoint categories `E1`, `E2` with `|E1| = |E2|` (Sec. 2.1, fn. 2). Default `|E| = 20` (Sec. 2.2). Inconsistencies: Sec. 3 says Fig. 1B is "for the case of 10 entities"; Appendix P calls "2 categories (50 × 2)" with 100 entities "the main-text setting"; this repo's `configs/default.yaml` uses 10 entities, `sub_size 5`. Sweep values: 10, 20, 30, 40 entities (Fig. 2).
- **Relations = edge labels, not functions.** On `E1` a *directed complete graph*; each ordered pair `(e_i, e_j)`, `i ≠ j`, gets a label `r(e_i, e_j)` "sampled uniformly at random from R, with the constraint that each entity e_i ∈ E1 has distinct relation labels on its outgoing edges" (Sec. 2.1, fn. 3). Default `|R| = 10,000` (Sec. 2.2); sweep 100 / 1,000 / 10,000 / 100,000 (Fig. 2). Out-degree = `|E1| − 1 = 9`. `E2` is a mirror: `r(F(e_s), F(e_t)) = r(e_s, e_t)` for a random bijection `F: E1 → E2` (Sec. 2.1).
  **[code]** Default dataset: 90 distinct labels for 180 atomic edges, each label on exactly one `E1` edge plus its `E2` mirror; (head, second relation) determines the answer of 1,440/1,440 compositions. The paper's composition is thus a "(head's category, r2) → tail" lookup, not composition of many-to-many relations.
- **Atomic facts** `(e_s, r(e_s,e_t), e_t)`, 3 tokens (Table 1). Count **[code]**: 180 for `|E| = 20` (40 for `|E| = 10`). All atomic facts are in training (no atomic hold-out; Appendix D removes atomic facts only as a sparsity ablation).
- **Compositional facts**: *all* two-hop paths `(e_s, r(e_s,e_i), r(e_i,e_t), e_t)`, 4 tokens, depth 2 only (Sec. 2.1, Table 1). Count **[code]**: 1,440 for `|E| = 20` (paths with `e_t = e_s` excluded); 120 for `|E| = 10`. Ratio compositional/atomic (phi) is never stated; **[code]** it is 1440/180 = **8.0** (`|E| = 20`) and 120/40 = **3.0** (`|E| = 10`).
- **ID / OOD split.** "OOD ratio for compositional facts `|D_comp^OOD| / |D_comp|`", default **0.1** (Sec. 2.2), sweep 0.1/0.3/0.5/0.7/0.9 (Fig. 2). Def. 2.1: OOD compositional facts have their constituent atomic facts in training but not the quadruple; the split is uniform over quadruples **[code]**, so training holds 90 % of *all* 2-hop paths, with every relation pair and almost every head seen. No unseen-relation-pair or unseen-head split exists.
- **Analogical facts** `(e_s, f, F(e_s))`, 3 tokens, one per `e_s ∈ E1`; analogical OOD ratio 0.1 default, sweep 0.1–0.9 (Fig. 2). **[code]** 9 ID + 1 OOD for `|E| = 20`.
- **Sequence format.** Tokens concatenated, no separators, vocabulary `|E| + |R| + 1` (Sec. 2.2); cross-entropy "applied only to the final token of each sequence" (Sec. 2.2); intermediate entity never appears in input or output; max sequence length 64 (Table 2).
- **Training-set size [code].** `|E| = 20`: 180 + 1,297 + 9 = 1,486 rows → 24 steps/epoch at batch 64 → 100 epochs ≈ 2.4 k steps. `|E| = 10`: 150 rows → 3 steps/epoch.

## 4. Model (Sec. 2.2; Appendix A, Table 2)

"Causal Transformer, similar to GPT-2", RoPE; **1 layer**, **1 head**, `d_model = 128`, MLP width `4 × d_model`, dropout 0.0, max length 64. Pre-LN vs post-LN: not stated (repo code: pre-LN). Weight tying: not stated (repo code: untied). Parameter count: not stated. Sweeps: `d_model ∈ {64, 128, 256, 512}`, `n_layer ∈ {1, 2, 4}` (Fig. 4); 4-layer LR sweep (Fig. 11); 4-layer embeddings (Fig. 16).

## 5. Optimization (Sec. 2.2; Table 3)

Adam (coupled L2; AdamW never used), lr `1e-4`, weight decay 0.0, batch size **32** (Sec. 2.2) but **64** (Table 3) — inconsistent; 100 epochs; "linear warmup then constant (warmup steps = 0)"; AMP enabled; "averaged over three random seeds" (Sec. 2.2); one NVIDIA A100 (Appendix A). Sweeps: weight decay {0, 0.01, 0.1, 1} and batch {32, 64, 128, 256} (Fig. 3); lr {1e-4, 1e-3, 1e-2} for 1 layer (Fig. 14); lr {1e-5, 1e-4, 1e-3} for 4 layers (Fig. 11).

## 6. Timeline (step numbers read off log-scale figure axes; approximate)

- **Default (Fig. 1B, 10 entities):** train acc ≈ 1 by ~50–70 steps; compositional OOD ≈ 1 by ~100 steps; analogical jumps 0 → 1 at ~150–200 steps. "Three-stage progression" (Sec. 3). No "grokking step scales as …" statement anywhere.
- **Entities (Fig. 2, panel 1):** train and composition saturate between ~10² and ~5×10² steps for 10–40 entities; analogical at ~1.5×10² (10), ~3×10² (20), ~10³ (30), ~3×10³ (40): "the time required to acquire analogical reasoning grows substantially relative to compositional reasoning" (Sec. 3.1).
- **Relations (Fig. 2, panel 2):** composition ≈ 1 by ~3×10²–10³ for all `|R|`; analogy "fails to emerge" at `|R| = 100`, is "acquired but later lost" at `|R| = 1,000` (Appendix C), stable at 10⁴–10⁵.
- **Compositional OOD ratio (Fig. 2, panel 3):** even at 0.9 (10 % of paths in training) composition reaches ≈ 1 by ~4×10² steps; higher ratios "delay the emergence" (Sec. 3.1). Analogy unaffected ("do not interfere with each other").
- **Weight decay (Fig. 3 left):** composition ≈ 1 by ~60–100 steps for wd 0–1 ("remains robust even under strong weight decay"); analogy at ~90–150 steps for 0/0.01/0.1, earlier with larger wd; wd = 1 peaks ~0.4 then collapses to 0.
- **Batch (Fig. 3 right):** analogy at ~130 / ~110 / ~100 / ~60 steps for batch 32 / 64 / 128 / 256; composition ≈ 1 by ~50–100 steps.
- **Width (Fig. 4 left):** composition ≈ 1 by ~10² (256, 512), ~5×10² (128), ~10³ (64) steps; analogy: 256 at ~2–5×10², 128 at ~10³, 64 ≈ 0 through 3×10³, 512 ≈ 0.3 at 3×10³.
- **Depth (Fig. 4 right):** composition ≈ 1 by ~10² steps for 1, 2 and 4 layers; analogy: 1 layer ~2×10²–10³, 2 layers ~5×10²–2×10³ (reaching ~0.9–1.0), 4 layers ~0.55 at 3×10³. Appendix B / Fig. 11: 4 layers with lr 1e-3 reaches ~0.8 by ~4×10³ steps, lr 1e-4 ~0.5 by ~8×10³, lr 1e-5 ≈ 0 — "an optimization artifact rather than an architectural limitation".
- **Learning rate, 1 layer (Fig. 14):** lr 1e-4 and 1e-3 both reach composition = analogy = 1 by ~10² steps; lr 1e-2: composition stuck ≤ 0.5 and noisy, analogy 0 ("even compositional reasoning becomes difficult to learn", Appendix E).
- **Sparsity (Fig. 13b):** removing 0–70 % of atomic facts: composition ≈ 1 by ~2.5×10² steps at every ratio; analogy 1.0 at 0/0.1/0.3, ~0.8 at 0.5, ~0.4 at 0.7.
- **100 entities (Fig. 23):** 2 categories of 50 → analogy at ~2×10³ steps; 5 categories of 20 → ~2–7×10².
- Longest runs: Fig. 6a to ~5×10⁴ steps (Dirichlet-energy minimum near 10³, rising afterwards; Appendix S); Figs. 22/24 to ~10⁴.

## 7. Mechanism / circuit findings

All mechanistic analysis concerns analogy, not composition:
- Dirichlet energy `E = Σ A_ij ||h_ei − h_ej||²` over the functor graph on the *token embeddings* (Eq. 1) "substantially decreased" before analogical accuracy rises (Fig. 6a; Sec. 4.1). PCA before (step 0) vs after (10³ steps) shows category alignment (Fig. 7).
- Attention from `f` to `e_s` (Eq. 2) "tends to increase prior to observable improvements" (Fig. 6b), cited as a Nanda-style progress measure.
- Parallelism `cos(h'_t − h_s, h_f)` between unembedding of target minus embedding of source and the functor embedding (Eq. 3) rises concurrently with accuracy, ID and OOD (Fig. 6c) → `e_t ≈ e_s + f`.
- 4-layer model reaches low training loss without aligned embeddings (Fig. 16, Appendix H).
- Transient analogy at `|R| = 1,000`: target probability decays and Dirichlet energy rises "once the model begins to overly fit the training data", with wd = 0 (Appendix C).
- Bilinear probe on embeddings: 81.1 % accuracy; null-space ratio 0.624 → 0.108 (Appendix T).
- **Not in the paper:** which layer/head performs which hop; logit lens or probes of the intermediate entity in the toy model (used only on LLMs, Sec. 5, Appendix R); any analysis of compositional OOD failure (composition never fails except at lr 1e-2). Sec. 7 admits the evidence is "correlational".

## 8. Other findings relevant to our questions

- **Held-out relations (seen only atomically):** nothing. Every relation labels one edge per category and appears in compositions.
- **Depth > 2 compositions:** nothing; depth 2 only.
- **Width / co-occurrence of facts in one sequence:** nothing in the toy task (one fact per 3–4-token sequence). Multiple facts per context appear only in the LLM prompts (Sec. 5, Appendix J).
- **Curriculum / data ordering, RL:** nothing.
- Related: sparsity ablation (Appendix D), functor noise ratio 0–1 (Appendix O, Fig. 22), 2/4/5 categories with multiple functor tokens (Appendix P), implicit functor without `<f>` token (Appendix Q, saturates lower, ~0.85 vs ~1.0 in Fig. 24).

## 9. Comparison table

| Parameter | Paper (default) | Ours round 1 | Ours pilot | Mismatch? | Suggested change |
|---|---|---|---|---|---|
| Entities | 20 (10 per category) [Sec. 2.2]; 10 in Fig. 1B; 10–40 swept | 500 | 500 | Yes, 12–50× | None needed for our goal; note paper's step counts do not transfer |
| Relations | 10,000 edge labels for 180 edges [Sec. 2.2] | 20 (12 train + 8 held-out) | 20 | Yes | See rec. 1: paper's relations are near-unique edge IDs |
| Relation definition | Random label per directed edge, distinct per out-node; complete graph [Sec. 2.1] | Random permutation of all 500 entities | Same | Fundamental | Keep ours; do not expect paper's dynamics |
| Atomic facts | 180 [code] (all in train) | 10,000 | 10,000 | Yes | — |
| Compositions / ratio phi | 1,440 / 8.0 [code]; 90 % of all paths in train | 20,000 / 2; 35 % of (head, pair) instances | 40,000 / 4 | Partly | Raise instance coverage toward ≥ 50 %; phi itself is not a paper knob |
| Depth | 2 only | 2 | 2 | No | — |
| Comp OOD definition | Random 10 % of quadruples, atomics seen [Def. 2.1] | Unseen heads of seen pairs + unseen pairs | Same | Comparable to our "unseen head" split | — |
| Layers | 1 [Table 2]; 1/2/4 swept [Fig. 4] | 2 | 2 | No | Keep 2 |
| d_model | 128 [Table 2] | 128 | 128 / 256 | No | Keep 128 (paper: 64 too small, 512 hurts analogy) |
| Heads | 1 | 2 | 2 | Minor | — |
| MLP / dropout / tying | 4×, 0.0, not stated | 4×, 0, untied | Same | No | — |
| Optimizer | Adam, wd 0 [Table 3] | AdamW | AdamW | Minor | Either; paper shows composition insensitive to wd 0–1 |
| LR / schedule | 1e-4, warmup 0, constant | 1e-3, 100 warmup | 3e-4 or 1e-3 | Yes | Paper: 1e-3 also fine for 1 layer (Fig. 14), 1e-2 breaks composition |
| Weight decay | 0 (0.01–0.1 speeds analogy only) | 0.1 | 0.1 | Yes | Not the lever for composition per Fig. 3 |
| Batch | 32 (text) / 64 (Table 3); 32–256 swept | 256 | 256 | Yes | Paper: larger batch = fewer steps; harmless |
| Steps / epochs | 100 epochs ≈ 2.4 k steps [Table 3, code]; curves ≤ 5×10³ | 196 k steps / 2,000 epochs | 200–280 k | Yes, ~100× | Paper gives no support for longer training |
| Precision | AMP [Table 3] | fp32 | fp32 | Minor | — |
| Positional encoding | RoPE [Table 2] | RoPE | RoPE | No | — |
| Answer format | Last token only, no intermediate entity [Sec. 2.2] | Answer token(s) only | Same | No | — |

## 10. Recommendations

1. **Do not treat this paper as evidence about our regime.** Its relation tokens label one edge per category [code], so (head's category, r2) determines the answer, and compositional OOD is a random 10 % of quadruples with every relation pair and head seen (Def. 2.1, Sec. 2.2). That is why composition tracks training accuracy within ~10² steps (Figs. 1B, 2) even at 10 % path coverage (Fig. 2, panel 3). Our permutation relations (500 edges each) require true two-hop lookup, which the paper never measures.
2. **If the goal is to reproduce the paper's curves,** use its configuration verbatim: 20 entities, 10,000 relations, complete graph, comp/analogical OOD 0.1, 1 layer / 1 head / d 128, Adam 1e-4, wd 0, batch 32–64, 100 epochs (Secs. 2.2, Tables 2–3). This repo's `configs/default.yaml` already matches except `n_layer: 2` (paper: 1) and 10 entities. Expect composition ≈ 1 by ~10² steps and analogy by ~2×10² (Fig. 1B).
3. **Layers: no evidence that 2 layers are insufficient, and no minimum stated.** `n_layer ∈ {1, 2, 4}` all reach compositional accuracy ≈ 1 by ~10² steps (Fig. 4 right), and the default is 1 layer / 1 head. The only depth caveat is for analogy: deeper models "can underperform under our default (fixed) optimization settings" and recover with lr 1e-3 (Appendix B, Fig. 11). Keep 2 layers.
4. **Learning rate / weight decay are not the missing ingredient.** Composition is unaffected by wd ∈ {0, 0.01, 0.1, 1} (Fig. 3) and by lr 1e-4 vs 1e-3 (Fig. 14); only lr 1e-2 destroys it. Our lr 1e-3 / wd 0.1 is within the paper's working range; the loss spikes you see at 1e-3 are consistent with Fig. 14's instability at higher lr, so 3e-4 in the pilot is reasonable, but do not expect it to unlock generalization.
5. **Training length is not the lever either.** All of the paper's phenomena occur within ≤ 5×10³ steps on ≤ 1.5 k training rows (Table 3, Fig. 2–4); the longest runs (Fig. 6a, ~5×10⁴ steps) show the analogical signal *degrading*, and Appendix C reports capability loss with continued fitting. Our 196–280 k steps already exceed the paper's regime by ~100×.
6. **The one transferable data knob is coverage of the composition space.** The compositional OOD sweep (Fig. 2, panel 3) shows generalization surviving down to 10 % path coverage in the paper's lookup regime; our analogue is the fraction of (head, pair) instances in training (35 % in round 1). Raising it (the pilot's phi 2 → 4 does this) is the change most consistent with the paper, subject to Rec. 1. For permutation-style relations the right references are the grokking-of-composition papers it cites (He et al. 2024; Wang et al. 2024, Sec. 1), not this paper.
