# Paper notes: Grokked Transformers are Implicit Reasoners (Wang et al., 2024)

## 1. Citation and source

Boshi Wang, Xiang Yue, Yu Su, Huan Sun. "Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization." NeurIPS 2024, arXiv:2405.15071.

Read: full arXiv HTML **v3** (`https://arxiv.org/html/2405.15071v3`, fetched raw with curl and parsed locally; all sections and appendices A-F). Plot values were read off the paper's own SVG figures after rasterizing ("read from Fig. N"). Where the paper is silent, values come from the official repo `github.com/OSU-NLP-Group/GrokkedTransformer` (README command, `main.py`, `composition.ipynb`, the `simpletransformers`/`transformers` forks) and are marked "(repo)". "(derived)" = our arithmetic on the paper's numbers.

## 2. Claim

A GPT-2-style transformer trained from scratch on atomic facts plus two-hop inferred facts learns to compose only by grokking: training accuracy saturates at ~14K steps while ID test accuracy is 9.2%, and near-perfect ID accuracy arrives only after "extended optimization lasting around 50 times the steps taken to fit the training data" (Sec. 3.2). Grokking speed is set by phi = |train_inferred_ID| / |atomic_ID|, not by dataset size, and is accelerated by weight decay and by bigger models. Composition never generalizes OOD (even at 2M steps) because the generalizing circuit stores a *second copy* of second-hop facts in the upper layers and only for atomic facts that appeared as second hops in training; comparison, whose circuit retrieves both facts in parallel in the lower layers, does generalize OOD.

## 3. Task / data setup (composition; Sec. 2, 3.1; repo `composition.ipynb`)

- **Entities / relations.** |E| = 2000 (5K and 10K in Fig. 2b); |R| = 200.
- **Graph sampling.** "each entity (as the subject) has 20 random distinct relations that each connects to another random entity (as the object)" (Sec. 3.1). Repo: per subject, 20 of 200 relations without replacement, tail uniform over all entities. So (h, r) has at most one object (relations are functions), but each relation is a **partial, many-to-one** function defined on ~10% of entities (~200 subjects per relation), not a permutation.
- **Atomic facts.** 2000 x 20 = 40,000 (derived), split 95%:5% into atomic_ID (38,000) and atomic_OOD (2,000). **All atomic facts, ID and OOD, are in training** ("Our training set includes all the atomic facts", Sec. 2; repo `train.json = all_atomics + train_inferred`).
- **Inferred facts.** Rule (1): (h, r1, b) and (b, r2, t) => (h, r1, r2, t). Total queries 2000 x 20 x 20 = 800,000 (derived). Repo split: both hops OOD -> test_inferred_OOD (~2,000); exactly one hop OOD -> discarded; both hops ID -> 99.5% train pool, 0.5% (~3,600) test_inferred_ID. 3,000 probes per test type.
- **phi.** train_inferred_ID is downsampled to phi x 38,000; swept phi in {3.6, 5.4, 7.2, 9.0, 12.6, 18.0} (Fig. 2a; repo) = 136.8K / 205.2K / 273.6K / 342K / 478.8K / 684K facts, i.e. 19 / 28 / 38 / 47 / 66 / 95% of the ~722K ID-composable queries (derived). There are 200^2 = 40,000 relation pairs, so at phi = 7.2 a pair has ~6.8 training instances on average (derived). No relation pairs are held out.
- **ID vs OOD.** ID test = unseen (h, r1, r2) whose two atomic facts are in training and also occur inside other training compositions. OOD test = queries whose atomic facts are in training only in atomic form ("facts are only observed in the atomic form, not in the compositional form", Sec. 3.3). OOD is instance-level, not relation-level.
- **Sequence format** (repo `form_items`): input `<e_h><r_1><r_2>`, target `<e_h><r_1><r_2><e_t></a>` (atomics `<e_h><r_1>` -> `...<e_t></a>`). Decoder-only LM with prefix labels set to -100 ("no loss on prefix tokens"), so the loss covers the tail token and `</a>`. **The bridge entity b is never in the output.** One token per entity/relation (two-token entities still grok, more slowly; Appendix C). `max_seq_length 10`.
- **Comparison task** (Sec. 4): |E| = 1000, 20 attributes, 20 ordinal values; atomic (e, a, v) for every pair (20,000), 90:10 ID/OOD; inferred (a, e1, e2) -> {<, =, >}; phi in {3.6, 7.2, 9.0, 12.6} (Appendix E.3).

## 4. Model (Sec. 2; repo)

"standard decoder-only transformer model as in GPT-2 with 8 layers, 768 hidden dimensions and 12 attention heads"; Appendix B adds 24 layers/1024 dim and 36 layers/1280 dim (Fig. 7). Repo builds `GPT2LMHeadModel` from the `gpt2` config with `--init_weights --n_layer 8`: MLP 4x (3072), **learned absolute positions** (`wpe`), **tied** embeddings (`_tied_weights_keys = ["lm_head.weight"]`), **dropout 0.1** on attention/embedding/residual (HF default; `--no_dropout` exists but is not used in the README command), pre-LN. Vocabulary = GPT-2 BPE (50,257) + added entity/relation tokens. Parameter count: not stated (derived from repo config: ~57M in the 8 blocks + ~40M tied embedding, ~98M).

## 5. Optimization (Sec. 2; Appendix A; repo)

AdamW, lr 1e-4, batch 512, weight decay 0.1, 2000 warm-up steps (Sec. 2), then **constant** lr (repo `constant_schedule_with_warmup`). Decay applies to all parameters except biases and LayerNorm, so it includes embeddings (repo). Total steps: not stated as one number; curves run to ~3-4e5 (Fig. 1, 2a, 13), 1.5e6 (Fig. 14), and "2 million optimization steps" for the OOD check (Sec. 3.2); repo `--max_steps 1500000`. Precision: not stated (repo `--fp16` AMP). Seeds: not stated, no error bars (repo default `manual_seed 42`). A6000/A100, <= 96 h per run (Appendix A).

## 6. Grokking timeline (composition unless noted)

- **Train saturation.** phi = 7.2: "training performance saturates (over 99% accuracy on both atomic and inferred facts) at around 14K optimization steps, before which the highest ID generalization accuracy is merely 9.2%" (Sec. 3.2). "The training performances of all settings saturate within 25K steps, where larger phi takes more steps" (fn. 3).
- **ID rise, phi = 7.2 (Fig. 1 left, read):** train 0.72 @ 1e4, ~1.0 @ 2e4; ID test 0.06 @ 1.5e4, 0.17 @ 2e4, 0.50 @ 5e4, 0.77 @ 1e5, 0.96 @ 2e5, ~1.0 @ 3-4e5. OOD flat at 0; "We extend the training to 2 million optimization steps, and there is still no sign of OOD generalization."
- **Dependence on phi (Fig. 2a, ID test, read):** phi 3.6: 0.04 @ 1e4, 0.05 @ 1e5, **0.08 @ 3e5** (no grokking in window); 5.4: 0.14 @ 1e5, 0.31 @ 3e5; 7.2: 0.13 @ 2e4, 0.47 @ 1e5, 0.72 @ 2e5, 0.84 @ 3e5; 9.0: 0.18 @ 2e4, 0.48 @ 5e4, 0.77 @ 1e5, 0.95 @ 2e5; 12.6: 0.88 @ 2e4, 1.0 @ 1e5; 18.0: 0.99 @ 2e4. Text: "the ratio phi strongly correlates with the speed of generalization. A very large ratio can push generalization to improve at a similar pace as the model fits the training data"; "When phi = 18.0, the model achieves 96.7% accuracy before training performance saturates" (fn. 4). (The Fig. 1 curve labelled 7.2 matches Fig. 2a's 9.0 curve; use Fig. 2a as reference.) No closed-form scaling law for the grokking step is given.
- **Data size (Fig. 2b, phi = 9.0, |E| in {2K, 5K, 10K}, x = epochs):** curves coincide, ID ~0.48 @ 67 epochs, ~0.83 @ 187; "training data size does not qualitatively affect the model's generalization." At |E| = 2K, phi = 9 an epoch is 746 steps (derived).
- **Weight decay (Fig. 13, phi = 9.0, read):** ID @ 1e5 / 2e5 / 3e5: wd 0.03: 0.53 / 0.77 / 0.87; wd 0.1: 0.73 / 0.93 / 0.97; wd 0.3: 0.93 / 0.98 / 0.99. "a larger weight decay can improve the speed of grokking, and vice versa" (E.1).
- **Model size (Fig. 7, read):** phi 9.0 @ 3e4 steps: 8L/768 0.24, 24L/1024 0.83, 36L/1280 0.90; phi 5.4 @ 2.2e4: 0.04 / 0.09 / 0.10; phi 18.0: 8L 0.99 @ 2.2e4, larger @ 1.2e4. "larger models converge in fewer optimization steps, but have no qualitative changes" (App. B). Nothing shallower than 8 layers is tested.
- **Comparison (Fig. 1 right, phi = 7.2, read):** train 0.99 by 1e4; ID 0.86-0.90 until ~1e5, 0.98 @ 3e5, 1.0 @ 4e5; **OOD also rises**: 0.85 @ 1e4, 0.82 @ 5e4, 0.88 @ 2e5, ~1.0 @ 4e5.
- **Parameter sharing (Fig. 14, E.2, phi = 12.6):** tying layers 1-4 with 5-8 gives OOD 0.23 @ 2.5e5, 0.52 @ 5e5, 0.73 @ 1.5e6.

## 7. Mechanism (Sec. 3.3, Appendix D)

- **Circuit** (Fig. 4a; logit lens + causal tracing, 300 train_inferred_ID examples, phi = 9.0): states in layers 0, 5, 8. "the lower layers retrieve the first-hop fact (h, r1, b) from the input h, r1, store the bridge entity b in S[5, r1], and 'delay' the processing of r2 to S[5, r2]; the upper layers retrieve the second-hop fact (b, r2, t) from S[5, r1] and S[5, r2], and store the tail t to the output state S[8, r2]." Hop 1 = layers 0-5 at the r1 position; hop 2 = layers 5-8 at the r2 position.
- **During grokking** the causal strength S[5, r1] -> prediction grows from weak to strong (Fig. 4b, 9) and the logit-lens MRR of r2 at S[5, r2] rises (Fig. 4c); b is present in S[5, r1] throughout. Hence "before grokking, the model is very likely mostly memorizing the examples in train_inferred_ID by directly associating (h, r1, r2) with t, without going through the first hop."
- **Why phi matters** (circuit efficiency): C_mem stores atomic + inferred facts; C_gen stores atomic facts in the lower layers "and another copy of the atomic facts that appear as the second hop in the inferred facts in the upper layers", so N_gen "is always bounded by two times the total amount of atomic facts". Weight decay and implicit bias favour C_gen, "and the transition would happen faster as phi increases"; data size alone leaves N_mem/N_gen unchanged.
- **Why no OOD** (distributed storage): "it does not have any incentive to store atomic facts in the upper layers that do not appear as the second hop during training ... the OOD atomic facts are simply not stored in the upper layers when queried during the second hop." Fn. 8: in OOD queries S[5, r1] still encodes b and S[5, r2] still r2, i.e. hop 1 works, hop 2 fails. Attributed to the "non-recurrent design of the transformer architecture which forbids memory sharing across different layers".
- **Comparison circuit** (Fig. 5a): both values retrieved in parallel in layers 0-5 (S[5, e1], S[5, e2]), label space at S[7, a], comparison in layers 5-8; atomic facts live only in the lower layers, so OOD works.

## 8. Not covered by the paper

- **Held-out relations:** absent; every relation may appear in compositions. OOD = held-out atomic *instances*.
- **Depth > 2:** absent ("We focus on two-hop composition"). Sec. 5's complex task is multi-step comparison search, not deeper composition.
- **Minimum depth for two-hop:** not stated; smallest model is 8 layers. The 4-layer model in E.2 runs twice (recurrence), i.e. 8 effective layers. No 2-layer result.
- **Curriculum, RL, seeds/variance:** absent.

## 9. Comparison with our setup

| Parameter | Paper | Ours round 1 | Ours pilot | Mismatch? | Suggested change |
|---|---|---|---|---|---|
| Entities | 2000 (5K, 10K) | 500 | 500 | minor; size "does not qualitatively affect" (Fig. 2b) | keep |
| Relations | 200 | 20 (12 train + 8 held-out) | same | **yes**: 144 vs 40,000 pairs | >= 50-200 relations |
| Relation definition | partial many-to-one function on ~10% of entities | total bijection | same | **yes**: r2 o r1 is itself a bijection, memorizable as one more 500-row relation | sparse partial functions |
| Out-degree | 20 of 200 | 20 of 20 | same | yes | 20 of >= 100 |
| Atomic facts | 40,000, all in train | 10,000, all in train | same | no | - |
| Inferred in train | 136.8K-684K | 20,000 | 40,000 | see phi | - |
| phi | 3.6-18.0; groks within 3e5 steps only for >= 7.2 | 2.0 (3.3 vs the 6K train-relation atomics) | 4.0 (6.7) | **critical**: at/below paper's non-grokking values | phi >= 9 vs the atomics that appear in compositions |
| Coverage | 38% @ 7.2, 47% @ 9.0, 66% @ 12.6 | 35% of 115 pairs | 70% | similar, but the paper's argument is about the ratio to atomics | raise phi via fewer atomics per composition |
| Depth | 2-hop | 2 | 2 | no | - |
| ID/OOD | ID = unseen instance of hops seen in compositions; OOD = hops seen only as atomics | unseen_instance ~ ID; held-out relations/pairs ~ OOD (relation-level) | same | partial | judge grokking on unseen_instance only |
| Layers | 8 (24, 36); none smaller | 2 | 2 | **yes, untested**; circuit spans 0-5 and 5-8 | 4-8 |
| d_model | 768 | 128 | 128 / 256 | yes; bigger groks faster (App. B) | 256-512 |
| Heads | 12 | 2 | 2 | yes | 4-8 |
| MLP | 4x | 4x | 4x | no | - |
| Optimizer | AdamW | AdamW | AdamW | no | - |
| lr | 1e-4 | 1e-3 | 3e-4 / 1e-3 | **yes** (10x); our loss spikes 0.8-1.7 | 1e-4 |
| Weight decay | 0.1 (0.3 faster, 0.03 slower), incl. embeddings | 0.1 | 0.1 | no | try 0.3; decay embeddings |
| Batch | 512 | 256 | 256 | minor | 512 |
| Warmup / schedule | 2000, constant | 100, constant | same | minor | 2000 |
| Steps | shown to 3-4e5; budget 1.5-2M | 196K | 200-280K | **yes** at low phi | >= 5e5, or raise phi |
| Positional encoding | learned absolute (repo) | RoPE base 100 | same | yes, but 3-token inputs; unlikely to matter | keep |
| Tying / dropout | tied; 0.1 (repo) | untied; 0 | same | unknown; not ablated | optional dropout 0.1 |
| Precision | not stated (fp16 AMP in repo) | fp32 | fp32 | no | - |
| Answer format | tail + `</a>`, loss on answer only, no bridge | loss on answer only, no bridge | same | no | - |

## 10. Recommendations

1. **Raise phi to >= 9, counted against the atomic facts that actually serve as hops.** At phi = 3.6 the paper's ID accuracy is ~0.08 after 3e5 steps and at 5.4 only 0.31, whereas 9.0 reaches 0.95 by 2e5 and 12.6 reaches 1.0 by 1e5 (Fig. 2a). Round 1 is phi = 2.0 (3.3 vs the 6K train-relation atomics) and the pilot 4.0 (6.7), i.e. the regime where the paper sees nothing within our budget. In the paper's own currency, N_mem/N_gen is 1.9 (round 1) and 3.1 (pilot) versus 3.9 at phi 7.2 and 4.8 at phi 9. With 144 pairs x 500 heads = 72K possible compositions, phi >= 9 means >= 54K in training (75% coverage); the cleaner route is fewer atomics per composition: |R| = 100-200 with out-degree 20 as in the paper, which also makes each pair rare (~7 examples at phi 7.2) so per-pair memorization is no longer cheap.
2. **Use 4-8 layers and d_model >= 256.** The paper never goes below 8 layers; hop 1 occupies layers 0-5 and hop 2 layers 5-8 (Fig. 4a); depth/width speed grokking sharply (phi 9.0 @ 3e4 steps: 0.24 for 8L/768 vs 0.83 for 24L/1024, Fig. 7c). A 2-layer model may work but has no support in the paper, so it should not be the configuration used to judge whether the data regime is right.
3. **lr 1e-4, 2000 warmup steps, batch 512** (Sec. 2). Ours is 10x the paper's lr and the spikes to 0.8-1.7 suggest instability; the paper's train accuracy stays flat at ~100% for the whole grokking window (Fig. 1).
4. **Budget >= 5e5 steps and keep wd 0.1-0.3, applied to embeddings.** Rule of thumb: ~14K steps to fit then ~50x that (Sec. 3.2); wd 0.3 reaches 0.93 at 1e5 steps vs 0.73 (0.1) and 0.53 (0.03) (Fig. 13).
5. **Score grokking on the unseen-instance split only.** Held-out relations and held-out pairs are OOD in the paper's sense (facts never seen inside a composition are "not stored in the upper layers", Sec. 3.3; OOD stayed at 0 for 2M steps), so expect them at chance even after ID groks.

**Should 196K steps at phi = 2 already have grokked?** No. The paper's nearest run, phi = 3.6 with 8 layers/768 dim, wd 0.1, batch 512, sits at 0.04-0.08 ID accuracy over the whole 1e4-3e5 window (~870 epochs; Fig. 2a), and phi = 5.4 reaches only 0.31 at 3e5. Ours has lower phi, a far smaller and shallower model (slower per App. B), 10x the lr, and fewer presentations (196K x 256 = 50M vs 3e5 x 512 = 154M). No grokking is the expected outcome of an unfavorable regime, not evidence of a bug. One diagnostic to keep in mind: the paper's non-grokking phi = 3.6 run still sits ~100x above its 1/2000 chance level, and at phi >= 7.2 ID accuracy is already 0.13-0.18 at 2e4 steps, right after training saturates; ours is exactly at 1/500 chance. That is plausibly structural (bijections vs sparse partial functions), but if a pilot with phi >= 9 and >= 4 layers shows no upward drift at all by ~5e4 steps, then suspect the pipeline.
