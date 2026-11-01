# From diagnosis to solutions (draft, 2026-10-31)

The paper defines the problem: representational vs behavioral compositionality, measured on
fresh parts. Every result so far has the same shape. States keep the parts; the procedures that
read them fail. Each lesson below names the failing procedure, the fix it implies, and a test
we can run in a setup we already have.

## Lesson -> fix -> test

| # | Lesson (where measured) | Fix it implies | Test |
|---|---|---|---|
| S1 | A procedure fitted on composites of V parts is undetermined outside their span (Prop. span); glyph decoders trained on 8 glyphs fail on fresh glyphs, a new decoder trained on 64 glyphs reads them from the same frozen state (0.25 -> 0.75) | **Learn parts from singletons, relations from composites.** The span requirement is over parts, not pairs, so part diversity can come from cheap single-part data while composites stay few | Glyph nets: 8-glyph composites + 64 single glyphs (render one glyph alone). Predict fresh-glyph behavior near the V=64 end-to-end value (0.88). Ties to the diffusion-composition finding that single-object captions teach each object |
| S2 | Role-factorized states let one composite fix every reversal (Prop. systematic); writing a new glyph's code through B2 renders it at 0.75 with no damage, fine-tuning gives 0.62 and harms old pairs | **Add parts by writing codes, not by gradient steps.** Estimate c, B_i once; a new part needs only its code e(x), placed into any role | Already have the glyph result; generalize to a procedure (estimate B_i from the probe pool, write codes) and test continual addition of 1..32 parts vs fine-tuning and replay |
| S3 | Pooled states blend parts that token/patch states keep (images: mean 0.17-0.49 vs quadrant 0.81-0.94); LMs answer by meaning while pooled readouts follow word order | **Pool by structure, or pool by query.** Embedding models (CLIP text conditioning, retrieval) pass a pooled state downstream; replace the mean with position/slot-preserving pooling, or with the state at the end of a role question appended to the input | E11 (running) tells whether the query state carries the figure. If yes: role-query embeddings on an ARO-style retrieval test with fresh nouns and converse phrasings; quadrant pooling for image retrieval of swapped layouts |
| S4 | The recursive decomposer fails one level past training while the state stays invertible (0.26 vs 0.99) | **Train the procedure on its own outputs at every depth.** Weight-tied decomposer trained on a depth curriculum of the composer's own states, with a recoverability loss on fresh leaves | RAAM: add depth-3/4 self-generated trees to decomposer training only; predict decomposer within ~0.1 of the population readout at depth 3-5 |
| S5 | kappa in every Pythia run peaks at ~1% of training and then declines with behavior flat | **Monitor kappa on fresh parts; protect it.** Use it for checkpoint selection, or as an auxiliary loss on a held-in part pool scored on held-out parts | Cheapest: does the step-2000 checkpoint transfer better on downstream role tasks than the final one? The loss variant needs pretraining runs: speculative, last |

## One principle behind all five

Train and read through the factorization the state already has. The state's part axis (codes e)
and role axis (maps B_i) can each be learned from the data that covers them: parts from many
cheap singletons, roles from few composites, depth from the system's own outputs. Most failures
come from procedures learned jointly from composites, where only V parts in V^2 pairs pin them
down.

## Suggested order

1. S1 (CPU, hours; decisive for the "parts from singletons" principle).
2. S3 once E11 lands (GPU, a day; the most practical payoff: compositional embeddings).
3. S2 as a continual-learning procedure (CPU, a day; overlaps DCU's compose-vs-create work).
4. S4 (CPU, hours).
5. S5 only the checkpoint-transfer variant.

Paper placement: one new section "From diagnosis to remedies" with S1-S3 as experiments, S4-S5
in the discussion, or a follow-up paper if the main text must stay near 13 pages.

## S1 results (2026-10-31; e12_singletons.py, results/e12 e12b e12c, agg_e12.py, agg_e12b.py)

Fresh-glyph output, over relation, mean of MLP, CNN and role-filler nets, 9 seeds, m = 64.

| Training | fresh pair | one fresh + one trained | fresh glyph alone | held-out pairs |
|---|---|---|---|---|
| all pairs of 8 glyphs | 0.26 | 0.52 | -0.30 | 0.93 |
| + singletons of the same 8 | 0.22 | - | - | 0.96 |
| + singletons of 64 new glyphs | 0.47 | 0.64 | 0.88 | 0.92 |
| + singletons of 256 new glyphs | 0.47 | - | - | 0.93 |
| 140 pairs of 16 glyphs | 0.57 | 0.74 | 0.05 | 0.99 |
| + singletons of 128 new glyphs | 0.68 | 0.80 | 0.93 | 0.98 |
| 140 pairs of 32 glyphs | 0.74 | 0.83 | 0.36 | 0.95 |
| + singletons of 128 new glyphs | 0.79 | 0.86 | 0.93 | 0.95 |
| 140 pairs of 64 glyphs | 0.81 | 0.85 | 0.49 | 0.91 |
| + singletons of 128 new glyphs | 0.83 | 0.87 | 0.93 | 0.91 |

Singleton share 0.25 vs 0.5: no difference. Transformer gains least (-0.05 -> 0.15-0.24 at V = 8).

**Lesson.** Two procedures, two data needs. Rendering a part needs the part, and singletons teach it
(alone: -0.30 -> 0.88). Separating a composite into its parts is a procedure fitted on composites,
and it extends only to parts it saw combined: a few pairs of many distinct parts suffice (140 pairs
of 64 glyphs: 0.81), many pairs of few parts do not (all 56 pairs of 8: 0.26). Singletons substitute
for about half of the missing composite diversity (32 combined + singletons 0.79 ~ 64 combined 0.81)
and add little once composites cover many parts. Remedy S1 is therefore: **many parts alone, plus
sparse combinations of many distinct parts**, never dense combinations of few.

## S3 results (2026-11-01; e14_role_query.py, results/e14, agg_e14.py)

No readout is fitted. For a fresh pair, S = "a x on a y", P = its converse paraphrase "a y under a x",
R1 = "a y on a x", R2 = "a x under a y". Score `pair` = P is more cosine-similar to S than both
reversals (chance 1/3); `top1` = P is S's nearest neighbour among all 800 sentences of the relation.
Layer chosen on validation nouns; 3 seeds, 200 fresh pairs per relation.

| model | mean (E7 state) | last token | role-question state (figure + ground) | figure question only (top1) | behavior (E9) |
|---|---|---|---|---|---|
| phi-4 | 0.00 | 0.00 | **1.00** (top1 0.95) | 0.03 | 0.99 |
| Qwen3-8B | 0.00 | 0.01 | **0.92** (top1 0.85) | 0.43 | 0.99 |
| OLMo-2 13B | 0.00 | 0.00 | **0.83** (top1 0.81) | 0.39 | 0.92 |
| OLMo-2 7B | 0.00 | 0.00 | 0.38 (top1 0.36) | 0.10 | 0.77 |
| Pythia-1.4b | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| GPT-2-large | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |

Plain sentence states always place a role reversal nearer than the paraphrase (pair 0.00). The states
at the end of two role questions keep the paraphrase nearest in the models that answer by meaning,
ordered as their behavior; one question alone matches the paraphrase against the reversals but
retrieves poorly (top1), since many sentences share the figure. Caveat: the questions are written per
relation.

**Lesson.** Pool by query. The model builds semantic roles where it answers a role question, so an
embedding read there keeps roles that every pooled or final-token state loses. Free, and it works only
as far as the model's behavior does.

## S2 results (e15_continual.py, results/e15, agg_e15.py)

32 new glyphs added in sequence, each from five composites in role 1; scored after 32 additions
(BESIDE), mean of MLP/CNN/role-filler, 5 seeds, V = 16, m = 64.

| base training | method | new, role 1 | new, role 2 (never shown) | two new together | held-out old pairs |
|---|---|---|---|---|---|
| E2 | none (unchanged, encodes the image) | 0.73 | 0.72 | 0.54 | 0.99 |
| E2 | fine-tune | 0.26 | 0.10 | 0.14 | **0.18** |
| E2 | fine-tune + replay | 0.92 | 0.82 | 0.76 | 1.00 |
| E2 | write codes (16 numbers per part, frozen net) | 0.72 | 0.71 | 0.55 | 0.99 |
| + singletons of 128 glyphs (S1) | none | 0.83 | 0.81 | 0.70 | 0.99 |
| + singletons of 128 glyphs (S1) | fine-tune | 0.26 | 0.14 | 0.17 | 0.18 |
| + singletons of 128 glyphs (S1) | fine-tune + replay | 0.95 | 0.87 | 0.84 | 1.00 |
| + singletons of 128 glyphs (S1) | write codes | **0.83** | **0.81** | **0.72** | 0.99 |

**Lesson.** Writing codes adds parts at the network's own ceiling (write = none on every measure) with no
gradient, no stored data and no forgetting; the code needs no image at generation time and places a part
in a role it was never shown. The ceiling is the decoder's fresh-part rendering, which S1 raises, so the two
remedies compose (0.72 -> 0.83 role 1, 0.55 -> 0.72 for two new parts). Replay beats writing by 0.11-0.21 on the E2 base and 0.06-0.12 with singletons,
but needs the old data and 200 gradient steps per part; plain fine-tuning forgets (old pairs 0.99 -> 0.18).
k = 1 numbers are unreliable (R^2 over 8 composites of one glyph) and are not reported.

## S4 results (e13_depth_curriculum.py, results/e13, e13b, agg_e13.py)

RAAM, V = 16, 5 seeds, decomposer R^2 on fresh leaves vs the population readout of the same states.
e13b: 20,000 steps, 128 deep trees per step.

| w | condition | depth 3 | 4 | 5 | 6 (never trained) |
|---|---|---|---|---|---|
| 256 | base (E3), decomposer | 0.27 | -0.08 | -0.29 | -0.47 |
| 256 | base, population readout | 0.98 | 0.97 | 0.88 | 0.43 |
| 256 | decomposer-only training on depths 3-5 (composer frozen) | 0.74 | 0.29 | 0.09 | -2.57 |
| 256 | joint training on depths 3-5, decomposer | **1.00** | **1.00** | **0.97** | -0.14 |
| 256 | joint training, population readout | 1.00 | 1.00 | 0.99 | 0.48 |
| 64 | joint training, decomposer | 1.00 | 0.50 | 0.25 | 0.12 |
| 64 | joint training, population readout | 1.00 | 0.49 | 0.24 | 0.12 |

Fresh Gaussian leaves at depths 1-2 alone (fresh12, 6k steps) lift depth 3 only to 0.24-0.47.

**Lesson.** The decomposer reaches the state's ceiling (the population readout, and at w = 64 the rank
bound of Prop. 9) at every depth on which composer and decomposer train together; the decomposer alone
cannot learn to invert the composer's deep states, even though they stay linearly invertible. Nothing
transfers past the trained depths (depth 6 fails in every condition). The prediction "decomposer within
~0.1 of the population readout" holds only with joint training, and productivity beyond training depth
remains open.

## E11 token states (2026-11-01; e11_tokens.py, results/e11, agg_e11.py; complete 2026-11-01, notes/audit/agg_e11.txt)

xfer_role_conv (figure readout fitted on canonical sentences, scored on converse; 0 = word order, 1 = meaning), fresh nouns, 3 seeds:

| model | mean | last | nouns | query |
|---|---|---|---|---|
| phi-4 | 0.02 | 0.08 | 0.00 | **0.98** |
| Qwen3-8B | 0.04 | 0.13 | 0.00 | **0.87** |
| OLMo-2 13B | 0.00 | 0.05 | 0.00 | **0.73** |
| OLMo-2 7B | 0.01 | 0.05 | 0.00 | 0.20 |
| pythia-12b | 0.02 | 0.06 | 0.00 | 0.10 |
| pythia-1.4b | 0.06 | 0.16 | 0.00 | 0.08 |
| gpt2-large | 0.06 | 0.14 | 0.00 | 0.07 |

The nouns state gives kappa2 0.41-0.74 (mean 0.05-0.35).

Lessons:
- Query-state transfer follows the models' answer accuracy: phi-4 0.98/0.99, Qwen3 0.87/0.99, OLMo-13B 0.73/0.92, OLMo-7B 0.20/0.77.
- The models that answer by meaning compute the role at the question. Every state of the sentence alone binds the figure to word order.
- The nouns state carries more of the second noun, but it is bound to order, so it does not explain the behavior.
- Untrained Pythia scores about 0.5 transfer only because its canonical accuracy is at chance (0.52-0.59). Transfer counts only when the canonical fit holds.
- This motivates S3 (pool by query). E14 confirms S3 without any fitted readout.

## S5 results (checkpoint selection; E11 on pythia-1.4b@step1000/2000/8000, 3 seeds) — NEGATIVE

| ckpt | mean κ2 | nouns κ2 | xfer mean | xfer nouns | xfer query |
|---|---|---|---|---|---|
| step0 | -0.44 | -0.16 | (chance) | (chance) | (chance) |
| step1000 | 0.56 | 0.69 | 0.00 | 0.00 | 0.14 |
| step2000 | 0.57 | 0.75 | 0.01 | 0.00 | 0.19 |
| step8000 | 0.35 | 0.73 | 0.04 | 0.00 | 0.23 |
| final | 0.31 | 0.66 | 0.06 | 0.00 | 0.08 |

Lessons:
- Picking the peak-κ checkpoint buys recovery of the second noun but no semantic roles in any state.
- Behavior stays at chance (E9).
- S5 therefore does not repair the behavioral failure. Peak κ measures part recovery, not role meaning.

## E17 encoder behavior (2026-11-01; e17_encoder_behavior.py, results/e17, agg_e17.py, notes/audit/agg_e17.txt)

Image-text matching on E8's fresh classes, BESIDE and ABOVE, 1000 composites, 3 seeds.

| model | roles (correct vs swapped) | identity (vs one wrong object) |
|---|---|---|
| CLIP-B/32 | 0.50 | 0.92 |
| CLIP-H/14 | 0.51 | 0.88 |
| SigLIP | 0.51 | 0.97 |

Lesson: the image encoders dissociate in the glyph-network direction. Their quadrant states keep each photo's place (κ 0.81-0.94), but their own matching ignores the place. So the limitation "no behavior of the encoders" is now answered for the image-text encoders.

## E16 other prompts and the ground task (2026-11-01; e16_prompts.py, results/e16, agg_e16.py, agg_e16_layers.py)

Query state read at each model's E11 layer and code. Values are answer accuracy / transfer to converse sentences.

| model | reword | zeroshot | ground |
|---|---|---|---|
| phi-4 | 0.98/0.94 | 0.92/0.82 | 0.98/0.94 |
| Qwen3-8B | 0.94/0.77 | 0.76/0.34 | 1.00/0.89 |
| OLMo-2 13B | 0.76/0.47 | 0.90/0.70 | 0.91/0.83 |
| OLMo-2 7B | 0.72/0.17 | 0.72/0.18 | 0.80/0.21 |
| GPT-2 / Pythia | ~0.5/≤0.11 | ~0.5-0.64/≤0.18 | ~0.5/≤0.10 |

Lessons:
- Transfer tracks answer accuracy across 4 prompts and 7 models. The role-question result is a property of the answering, not of one prompt.
- GOTCHA: selecting the layer on canonical-only validation picks early, order-bound layers (phi-4 L10: 0.12, against L30: 0.94). Fix the layer from E11, or select with both phrasings.
