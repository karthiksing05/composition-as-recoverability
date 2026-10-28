# Review R1-C: "Defining Compositional Representations by Recoverability"

Reviewer C: empirical ML / mechanistic interpretability (probing, control tasks, causal interventions, DAS, binding in LMs, compositional-generalization benchmarks, T2I compositional failures).

I read the full paper (main.tex, all 22 rendered pages), the E2 and E4 code (`e2_glyph.py` plus the `_run`/`_run2` variants that produced `results/e2`, `e4_pretrained_run.py`, which produced `results/e4`, and `e4_pretrained.py`), the figure scripts (`fig_e2.py`, `fig_e3e4.py`) and the raw JSON. All numbers below that the paper does not print come from my own aggregation of `results/*/*.json` with grep/awk (no Python was run).

---

## 1. Summary

The paper defines representational compositionality through two properties. A relation is *recoverable* if a readout from a declared class U, fitted on composites of a finite vocabulary, returns the part codes of composites built from *fresh* parts (fresh-part R² ≥ 1−ε). It is *role-factorized* if the state is affine in the part codes, with one matrix per role shared across parts. The theory section proves simple facts: lookup decodes any relation that is injective on a vocabulary; for Gaussian parts the MMSE ceiling of a linear blend is tr(A_iᵀ(AAᵀ)⁺A_i)/d; a least-squares readout is determined only on the span of its training parts; and under exact role-factorization, role reversals and new role assignments are affine in one composite. Four experiments follow. E1 calibrates analytic operators against the ceiling. E2 trains small encoder–decoders on rendered glyph pairs, varying bottleneck width m and vocabulary V. E3 tests a RAAM-style recursive composer against HRR. E4 probes six pretrained text encoders with ridge readouts from sentence states ("a x on a y") to phrase states ("a x"). The headline empirical claims are:
- (i) networks generalize to unseen pairs whether or not their states keep parts recoverable ("imitation").
- (ii) fresh-part recovery from the state predicts fresh-part transfer (r = 0.94).
- (iii) part diversity, not combination coverage, governs transfer.
- (iv) pretrained encoders, especially CLIP, recover nouns of familiar sentences but not of new ones.

The conceptual point that fresh-part evaluation is what separates relational recoverability from vocabulary clean-up is clear and useful, and E1 is a clean calibration. The interpretability evidence for the paper's empirical claims (E2's "imitation" cell, the r = 0.94 correlation, the part-diversity claim, and all of E4) has a recurring problem: the probe's own limitations (span of its training targets, scale miscalibration of R², test-set layer selection, in-sample "familiar" scores, no control task) are not separated from properties of the system. Several stated conclusions do not survive a close reading of the code and raw results. Some of them are contradicted by metrics the authors already computed.

## 2. Scores

| | Score |
|---|---|
| Soundness | 2 / 4 |
| Presentation | 3 / 4 |
| Contribution | 2 / 4 |
| Overall | 4 / 10 (below the acceptance threshold in its current form) |
| Confidence | 4 / 5 |

---

## 3. Major weaknesses

### W1. In E2, the probe is span-limited by the same vocabulary as the network, so "imitation" (Table 1) is not established at small V.

**Location.** Sec. 4.2 Results ("The MLP, CNN, and role–filler networks trained on eight parts thus fall in the imitation cell"); `e2_glyph.py` lines 284–305.

**Problem.** The state readout (`probe_pix_*`) is a RidgeCV fitted on composites of the *network's own training vocabulary* (`vocab[train[:,0]]`, `vocab[train[:,1]]`), with glyph pixels (64-d) as targets. Ridge predictions always lie in the affine span of the training targets. At V = 8 the role-1 targets take at most 8 distinct glyphs and the role-2 targets at most 6 (two of the eight are held out of role 2). Predictions therefore live in a ≤7-dimensional (≤5 for role 2) affine subspace of the 64-d pixel space, whatever information h carries. This is exactly Proposition 4, and the paper's own Fig. 3c shows it: a perfect concatenation gives fresh R² = 0.05 at V = 8. At V = 64 the role-2 readout still sees only 48 distinct glyphs in a 64-d target space, so the pixel probe stays span-limited even at the largest V.

So the observed low "recovery from h" at V = 8 (pixel-probe R² ≈ 0.20–0.28) is what any state would give, including a perfectly invertible one. The authors' own calibration shows that this readout cannot distinguish "state loses the parts" from "readout saw too few parts". What *is* established is that the network's decoder fails on fresh glyphs. That fits the "failure of performance" cell (recoverable state, poor decoder) as well as the "imitation" cell. The paper does not test which one applies.

**Fix.** Decouple the readout's vocabulary from the system's training vocabulary. Freeze the trained encoder, encode composites of a *large, separate* pool of parts (e.g., 2,000 latents never used in training or in the fresh test), fit the ridge readout on those, and evaluate on the fresh test pairs. The definition permits this: Def. 1 fixes the readout's vocabulary as some V ~ D, not the system's V. Report the matching reference (concatenation of glyphs) fitted the same way. Only if recovery under this readout is low at V = 8 can you assign the imitation cell. Fitting the latent-target probe (`probe_lin_*`, 8-d targets, so the span saturates at V ≥ 9) would be a partial check; the code computes it, but the paper never reports it.

### W2. The r = 0.94 "recovery predicts transfer" correlation is pooled across conditions and mostly reflects the shared V and m manipulation.

**Location.** Sec. 4.2 ("Across all runs ... r = 0.94 ... r = 0.61"); Fig. 4d; `fig_e2.py` lines 103–110.

**Problems.**
- The correlation is computed over all 640 runs pooled across 4 architectures × 4 m × 4 V × 10 seeds, after clipping both variables at −0.2.
- Both the probe and the network's decoder are trained on the same V parts and are both span/coverage-limited (W1). Both are also bounded by m (Prop. 3). The correlation therefore mostly says that two estimators trained on the same small vocabulary fail together.
- Both quantities are computed on the *same 32 fresh composites* (`fr_pairs`, 32 pairs), so test-sample noise is shared, which inflates r further.
- The contrast with held-out pairs (r = 0.61) is mostly a range restriction: held-out behavior sits at 0.86–1.00 almost everywhere, so its correlation with anything is attenuated. That does not show the dissociation the text claims.
- Clipping at −0.2 changes r, and transformer runs reach −1.

**Fix.**
- Report within-condition (partial) correlations, i.e., residualize both variables on condition (arch × m × V) or correlate across seeds within a cell.
- Use separate fresh samples for the probe evaluation and the behavioral evaluation.
- Report rank correlations without clipping, with bootstrap CIs over seeds.
- Compare against the obvious baseline that "V alone predicts both" (e.g., the R² of a model with V and m as predictors versus V, m and probe recovery).

### W3. The part-diversity claim is confounded with the number of training pairs; an unreported control partly supports it but contradicts the strong form.

**Location.** Sec. 4.2 ("above all by the number of distinct training parts"); Discussion ("Coverage of combinations ... is not the variable that matters"); Fig. 4c.

**Problem.** In the main grid, V and the number of training pairs move together. Per `splits()`, n_train = 34 at V = 8, 144 at V = 16, 596 at V = 32 and 2,420 at V = 64, while training runs for a fixed 4,000 steps × batch 256. At V = 8, 34 composites are each seen roughly 30,000 times, so V also confounds overfitting and data-set size.

The code already has an `npairs` control (`results/e6pairs`, 140 pairs at V ∈ {16, 32, 64}). My aggregation of those runs, fresh output R² at m = 64:

| arch | V16, n = 140 | V32, n = 140 | V64, n = 140 | V64, full (2,420) |
|---|---|---|---|---|
| CNN | 0.63 | 0.77 | 0.81 | 0.90 |
| MLP | 0.57 | 0.75 | 0.81 | 0.90 |
| role–filler | 0.45 | 0.67 | 0.81 | 0.85 |

At fixed pairs, part diversity clearly matters (good, and it should be in the paper). Going from 140 to 2,420 pairs at V = 64 still adds 0.04–0.09 fresh R² and 0.04–0.08 held-out R² (CNN held-out 0.905 → 0.982). So "combination coverage is not the variable that matters" is too strong. The V = 8 point, which drives the imitation story, has no matched-pairs counterpart.

**Fix.**
- Report the fixed-pairs control in the main text (Fig. 4c with a second curve).
- Add a fixed-V, varying-pairs sweep.
- Equalize the number of distinct images *and* optimization steps per distinct image across V.
- Soften the Discussion claim accordingly.

### W4. E4's "fresh" nouns are not fresh to the system, the "training-noun" scores are in-sample, and the best layer is chosen on the test metric.

**Location.** Sec. 4.4; abstract ("recover the nouns of familiar sentences but not of new ones"); `e4_pretrained_run.py` lines 116–119; `fig_e3e4.py` line 64.

**Problems.**
- **Freshness.** Every one of the 250 nouns is in the vocabulary of every model, and the file's own docstring says so ("the model has of course seen every word"). By the paper's own definition (Sec. 3.1: a fresh part is a draw outside the vocabulary to which the *system* was exposed), E4 tests readout generalization across nouns. It does not test the system's handling of new parts. The abstract's phrase "new sentences" is accurate; "fresh parts" (Sec. 4.4) is not.
- **In-sample "training" scores.** `rec_train` is computed on `Htr[:500]`, which is part of the ridge fitting set. The 0.82–1.00 "training noun" bars are therefore in-sample fits of a 512–1,024-d ridge to 32 targets, not held-out pairs of familiar nouns. That contradicts the Sec. 4 protocol bullet, which says readouts are evaluated on held-out pairs of vocabulary parts. The train/fresh gap mixes overfitting with noun generalization.
- **Layer selection on the test metric.** `best` is selected by `argmax over layer of mean rec_fresh` on the same fresh-noun data that is then reported. With three candidate layers the optimism is modest, but it is selection on the test set, and the text does not disclose it.
- **Fixes.**
  - Add a held-out-pairs-of-training-nouns split.
  - Select the layer on a validation noun split disjoint from the reported fresh nouns, or report all layers.
  - Rename "fresh nouns" to "held-out nouns".
  - To test true freshness in pretrained models, use nonce words, novel subword strings or newly coined compounds (e.g., the recipe of Kim & Linzen-style lexical generalization or Lake & Baroni 2019).

### W5. E4's negative conclusion rests on an R² that is dominated by scale miscalibration; the authors' own identification metric contradicts it; and the "symmetric control" behaves exactly like the relations.

**Location.** Sec. 4.4 Results; Fig. 5c,d.

**Problems.** From the raw JSON, averaged over the three relations and three seeds:

| Model | Layer | Fresh R² | Fresh identification (1/80 chance) | Notes |
|---|---|---|---|---|
| CLIP-B/32 | pooled | −1.20 | 0.38 | |
| CLIP-B/32 | L6 | −4.2 | 0.12 | |
| SD CLIP-L | pooled | −0.92 | 0.44 | |
| T5-base | pooled | 0.43 | 0.84 | sum reference identification 0.48 |
| T5-small | pooled | 0.20 | 0.76 | |

- **R² against identification.** An R² of −4 alongside identification 10× above chance means the predictions point roughly in the right direction but have the wrong scale or offset. The usual cause is per-dimension standardization (`(h-mu)/sd` with `sd+1e-6`) inflating near-constant dimensions of CLIP states, combined with RidgeCV whose GCV (leave-one-*composite*-out) leaks noun identity across composites and so picks a small alpha. "CLIP falls far below even the symmetric-sum reference" is therefore a statement about R² calibration, not about information content. By identification, CLIP (0.38–0.44) is close to the sum reference (0.46–0.48), and both T5 models (0.76–0.84) exceed it substantially. The abstract's headline does not survive the authors' own second metric.
- **Symmetric control.** The "and" control gives the same results as "on" and "left" in every model (e.g., GPT-2 L6: 0.79 / 0.80 / 0.78; T5-base L6: 0.62 / 0.54 / 0.55; CLIP pooled: −1.32 / −1.27 / −1.03). The readouts therefore recover *serial position*, not relational role. The paper neither reports this nor discusses it. Since the readouts never see a role that differs from position, E4 says nothing about relational binding. The figure averages over all three relations, including the control.
- **Reversal similarity.** "Normalized reversal similarity 0.91" for CLIP is presented as bag-of-words evidence. At its own best layer T5-base has 0.87, T5-small L3 0.84 and SD CLIP-L 0.88, so the statistic does not single out CLIP. At mean-pooled layers, near-invariance to reversal is partly a property of mean pooling, not of the model.
- **Fixes.**
  - Report scale-invariant metrics (per-target correlation, CCA, or R² after a per-dimension affine recalibration fitted on training composites) next to raw R².
  - Choose alpha by leave-one-*noun*-out CV.
  - Drop per-dimension z-scoring, or whiten with a variance cutoff.
  - Report each relation separately and use the "and" control explicitly. A relational-role claim requires sentences in which role and position dissociate: passives ("a y is under a x" vs "a x is on a y"), or "x on y" vs "y under x" with role defined semantically. Show that role readouts follow semantic role, not position.
  - Add a control task in the Hewitt & Liang sense (readout of random noun-to-code assignments) to estimate selectivity.

### W6. E4's part code favours causal models, and contextual token states were not used.

**Location.** Sec. 4.4 ("The part code of a noun is the model's state for 'a x'"); `embed()`.

**Problem.** For causal models (GPT-2, and the CLIP text encoder, which uses a causal mask), the hidden states of the "a x" prefix in "a x on a y" are *identical* to those in the stand-alone "a x" (up to CLIP's BOS/EOS). The mean-pooled sentence state therefore contains the part code as an exact linear sub-sum, which mechanically favours role 1 for causal models. This may partly explain why GPT-2 alone exceeds the sum reference. For bidirectional models (BERT, T5) the code of "a x" in isolation is a different distribution from the in-context noun state, which mechanically penalises them. Mean pooling also mixes function-word and template tokens into every state.

The question "is a state of 'a x' a fair part code?" therefore has different answers for causal and bidirectional models, and the cross-model ranking partly reflects architecture × pooling × part-code interactions.

**Fix.**
- Use (a) the contextual token states at the noun positions and (b) a pooled state that excludes template tokens, and report both.
- Report role 1 and role 2 separately, and causal and bidirectional models separately.
- Add the obvious stronger reference built from the model itself: the mean-pooled sentence state with the other noun replaced by a fixed filler (a "context-matched" part code).
- Ideally, include a DAS/interchange-intervention check that the recovered subspace is causally used, since U and M are otherwise untested on pretrained models.

### W7 (moderate). Under-powered claims about the role–filler decoder and new roles.

**Location.** Sec. 4.2 ("The role–filler decoder ... is the best on parts in new roles at the smallest vocabulary ... as role-factorization predicts").

**Problem.** At V = 8 the strong split contains 2 held-out parts (14 pairs). Strong-split output R² (my aggregation, m ∈ {8, …, 64}): role–filler 0.31–0.33 (SD 0.26–0.31 across seeds); CNN 0.11–0.27 (SD 0.25–0.34); MLP 0.11–0.19 (SD 0.33–0.38). With 10 seeds the SE is about 0.09–0.11, so the role–filler vs CNN difference (≈0.05 at m ≥ 32) is well within noise. "Best" is not supported.

The role–filler model is also *worse* than MLP/CNN on fresh parts at V ≤ 32 (V = 16: 0.44 vs 0.57–0.61; V = 32: 0.70 vs 0.81–0.82). The text says it "transfers no better", but in fact it transfers worse. That undercuts the Discussion's "third requirement".

**Fix.** Report means ± 95% CIs (or a paired test across seeds that share vocabulary/world) in the text. Enlarge the strong split (e.g., V//2 held out of role 2, or several held-out-role draws per seed). State the fresh-part deficit of the role–filler decoder plainly.

### W8 (moderate). The OCCLUDE comparison confounds encoding with rendering a discontinuous output, and OCCLUDE is not "invertible in principle" as stated.

**Location.** Sec. 4.2; Fig. 4e; `World.render`, `SlotModel.decode`.

**Problems.**
- `b_fresh_in1` averages over output relations, so the OCCLUDE bar includes the decoder's ability to *render* an occlusion, which needs an argmax over host quadrants (discontinuous in z).
- The BESIDE/OVER bars come from networks trained on {BESIDE, OVER}, and the OCCLUDE bar from separately trained networks on {BESIDE, OCC}.
- The host glyph loses a whole 4×4 quadrant and the occluder is 2× average-pooled. Recoverability of the latents from that is plausible but not shown, and the pixel targets of the host are not recoverable from the composite at all.
- "States carry 1–4% interaction ... so the networks approximate the non-additive relation by an additive one" does not follow from a low two-way ANOVA interaction on a 10 × 10 grid alone.

**Fix.**
- Compare recovery from h (with the readout of W1), not output R².
- Restrict to the BESIDE output when comparing input relations.
- Compute an upper bound by fitting the same readout class to the OCCLUDE inputs themselves.
- Qualify the invertibility claim.

### W9 (moderate). Statistics and reporting.

- Every claim in Sec. 4 is a range across architectures, with no CI or test.
- E4's three "seeds" vary only the noun split and pair sampling of one fixed checkpoint per model, which should be stated.
- E2-OCCLUDE: the paper says 5 seeds, but `results/e2occ` currently holds 313 files with 5–10 seeds per cell and some cells still being written (e.g., `mlp_m64_V32` has 6). Make sure the figure and text are regenerated from a frozen, complete set.
- The fresh evaluation in E2 uses only 32 composites (`fr_pairs`), so per-run R² is noisy.
- `results/e4` was produced by `e4_pretrained_run.py` (3 relations, 2 layers + pooled, with a `ref_weighted` reference). The shipped `e4_pretrained.py` is a different, later version (6 relations, 4 layers, 3 part codes) that does not reproduce the reported figure. The released code must match the reported results.
- Several controls exist in `results/` (`e6pairs`, `e6aux`, `e2q`, `e1rob`, `e5`) but are not in the paper. Some are informative: auxiliary recoverability pressure raises the MLP's V = 8 fresh transfer only from 0.17 to 0.34, and q = 4 vs q = 16 confirms the width ceiling. Report them or remove them from the artifact.

### W10 (conceptual, moderate). Prop. 4 is invoked to explain trained nonlinear decoders beyond its scope.

**Location.** Sec. 4.2 ("the vocabulary effect is Proposition 4 acting on the network's own decoding procedure"); Discussion.

**Problem.** Prop. 4 concerns least-squares *affine* readouts of an exactly role-factorized relation and predicts saturation once the latents span R^q, i.e., at V ≥ q+1 = 9. In E2, fresh transfer keeps rising from V = 16 (0.57–0.63) to V = 64 (0.85–0.90), well past the span threshold in latent space. The quantitative pattern is better explained by nonlinear sample complexity (manifold coverage) than by span. The mechanism claim is therefore unsupported as stated.

**Fix.** Either show the E2 curve against a span-predicted curve (e.g., with an 8-d *linear* generator, where Prop. 4 applies), or rephrase this as an analogy.

---

## 4. Minor issues

1. Sec. 4 intro says readouts are "evaluated on held-out pairs of vocabulary parts, on parts in roles they never occupied, and on fresh parts". E4 has neither held-out pairs nor new-role splits.
2. Sec. 4.2 Readouts says "to each role's glyph". Def. 1 is about part codes e(x); say explicitly that the part code is the glyph's pixels, and report the latent-target probe as well.
3. Fig. 4d: label the axes as probe R² (pixel) vs output R². State that the values are clipped and that r is pooled.
4. RidgeCV with the default GCV is effectively leave-one-composite-out. With shared parts across composites this underestimates the penalty that matters for fresh parts, in both E2 and E4. Use group CV by part.
5. Per-dimension standardization with `+1e-6` (E2 and E4) can blow up near-constant state dimensions. Report the selected alphas and the conditioning of the design.
6. E4: the PCA-32 target space is fitted on training nouns only. Report the fraction of fresh-noun variance that falls outside it, because R² is computed only inside it.
7. The role-assignment metric's reference ("Every model assigns roles better than the sum reference") is trivially 0.5 by symmetry. Use a positional baseline instead (e.g., the readout's role accuracy on the symmetric "and" control).
8. Table 1's "imitation" cell cites Chaabouni et al. (2020), whose claim concerns emergent-language messages, not internal states. Make the mapping explicit.
9. Abstract: "Trained networks generalize to unseen pairs ... almost perfectly whether or not their states keep parts recoverable". The transformer at V = 8 reaches only 0.55–0.61 on held-out pairs, so qualify this.
10. Self-citations: "Anonymous (2026)" in a non-anonymous preprint, plus the TRELLIS citation, are unnecessary for the argument.
11. Related work: cite control tasks / selectivity (Hewitt & Liang 2019), amnesic or causal probing (Elazar et al. 2021), and probing critiques (Belinkov 2022). For E4, cite binding-ID work beyond Feng & Steinhardt (e.g., Prakash et al.; Dai et al.) and syntactic-role probing (Tenney et al. 2019; Hewitt & Manning 2019). For CLIP, add ARO/Winoground (Thrush et al. 2022) alongside Yuksekgonul et al.
12. Prop. 7's proof bounds the rank by min(w, 2^n d), but a composer whose width w exceeds d can hold rank w. The statement mixes the case "composites lie in the space of their parts" (w = d) with general w; state the two cases separately.
13. The "difficulty-matched baseline" in E2 (random held-out pairs) is never compared to the strong split in the text. Report both side by side with CIs.

---

## 5. What would move my score up by 2 points (to 6)

1. **Decoupled readouts in E2 (W1).** Re-fit the state readouts on a large independent part pool through the frozen encoder, with matching references. Re-derive the 2×2 cell assignment and the probe-to-behavior relation using within-condition correlations on separate test samples (W2).
2. **Fixed-pairs and fixed-V controls in the main text (W3).** Soften the "coverage does not matter" claim to match the data.
3. **E4 rebuilt as a proper probing study (W4–W6).**
   - Held-out pairs of training nouns.
   - Layer chosen on a validation split.
   - Scale-invariant metrics and part-grouped CV for alpha.
   - In-context token-state part codes alongside phrase states.
   - Per-relation reporting, with the "and" control and a role/position dissociation (passives or converse relations).
   - Nonce-word parts for genuine freshness.
   - A causal check (interchange intervention on the recovered subspace) for at least one model.

   If CLIP still sits clearly below the references under these conditions, the claim stands and is much more interesting. If not, the paper should say so.
4. **Statistical reporting (W7, W9).** CIs or paired tests for every comparative claim, a frozen and complete result set, and released code that matches the reported numbers.
