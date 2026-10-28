# Review R4-C: empirical audit of "Defining Compositional Representations by Recoverability"

Reviewer C (empirical audit). I read `recoverability-theory/main.tex` (mtime 2026-10-28 05:57, 596 lines) and checked it against these sources:

- `experiments/results/agg_r4_e4w.txt`, with three blocks:
  - `e4w`: the main analysis, on whitened codes;
  - `e4wd`: the same with the token count removed;
  - `e4wn`: the pool-size curve.
- Other result files:
  - `agg_r4_e2e5.txt`, `agg_e2h_partial.txt`;
  - `agg_r2.txt`, `agg_r3.txt`;
  - `fig_e3e4.txt`, `diag_e4_codes.txt`, `agg_e4b.txt`.
- Code:
  - `e4_pretrained.py`, `agg_r4.py`, `agg_e2h_partial.py`;
  - `fig_e3e4.py`, `fig_e2.py`, `diag_e4_codes.py`.
- The launch lists `tasks_e4e/f/g.txt`.
- The per-run JSONs, read with grep/awk. I used them for the penalty-edge check, file counts and the noun count.
- The rendered `notes/fig_depth_pretrained.png`.

I ran no Python, LaTeX or compute.

**Bottom line.** Every number in the rewritten pretrained-encoder section, abstract, Discussion and appendix matches `agg_r4_e4w.txt` to rounding. There are two exceptions, both in what the text claims or omits rather than in the arithmetic:

1. The abstract (and the last sentence of Sec. 4.5) still makes a claim about CLIP that CLIP-H contradicts.
2. A random-weight control result that cuts against L263 is still unreported.

The method descriptions match `e4_pretrained.py` and the launch flags. I found one internal inconsistency elsewhere (0.13 vs 0.14 for the RAAM ceiling gap) and a few unsupported or stale items.

---

## MAJOR

**M1. The abstract and L267 say that "the CLIP encoders that condition text-to-image models gain little from training beyond the length of each noun". CLIP-H is such an encoder, and it gains a lot.**

- `clip-h14` loads `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (`e4_pretrained.py:68`). This is the OpenCLIP ViT-H/14 text encoder of Stable Diffusion 2.x.
- Effect of training on CLIP-H in `agg_r4_e4w.txt`:

  | Codes | Trained R² (κ) | Random weights R² (κ) | Δκ |
  |---|---|---|---|
  | Main (e4w) | 0.664 (0.355) | 0.498 (0.051) | +0.30 |
  | Token count removed (e4wd) | 0.665 (0.358) | 0.499 (0.055) | +0.30 |

  CLIP-H keeps its whole gain once the token count is removed. Only CLIP-B/32 and SD-1.5 CLIP-L fall below random once length is removed (0.466 / 0.457 vs 0.561 / 0.526).
- CLIP-B/32 is not the conditioner of any major text-to-image model, so "the CLIP encoders that condition text-to-image models" picks out exactly the wrong pair.
- L282 in the Discussion already words this correctly ("the smaller CLIP encoders").
- Corrected abstract sentence: "…GPT-2 gets about halfway, and the two smaller CLIP encoders, including the one that conditions Stable Diffusion 1.5, gain little from training beyond the number of subword tokens in each noun."
- Corrected last sentence of L267: "…GPT-2 gets about halfway, and the two smaller CLIP encoders, including Stable Diffusion 1.5's, gain little from training beyond the token count of the noun."

**M2. Role assignment and identification are offered as positive evidence without the random-weight baseline, and that baseline matches or beats trained CLIP-B (L263, Fig. 6d). This is r3-B2 and is still open.**

- L263: "Every trained model assigns roles to fresh nouns far better than the sum reference (0.90–1.00 against 0.50) and identifies them … far above chance (0.65–0.94) and above its own sum reference (0.41–0.44)."
- Random weights scored on their own codes (`agg_r4_e4w.txt`, e4w block):

  | Encoder | Role, random | Role, trained | Identification, random | Identification, trained |
  |---|---|---|---|---|
  | CLIP-B | 0.899 | 0.895 | **0.744** | 0.709 |
  | CLIP-L | 0.839 | 0.957 | 0.687 | 0.786 |
  | CLIP-H | 0.817 | 0.977 | 0.653 | 0.824 |
  | GPT-2 (three sizes) | 0.72–0.74 | — | 0.51–0.53 | — |
  | BERT | 0.67–0.68 | — | 0.47–0.49 | — |
  | T5 | 0.50 | — | 0.44–0.46 | — |

  - Random CLIP-B identifies fresh nouns better than trained CLIP-B and assigns roles as well.
  - Random T5 is position-blind: role 0.50, rev_sim 1.000.
- Fig. 6d still has no random-weight markers.
- Corrected sentence for L263: "Every trained model assigns roles to fresh nouns better than the sum reference (0.89–1.00 against 0.50) and identifies them among 100 candidates above chance (0.65–0.94). Untrained architectures with positional information already do much of this: with random weights, GPT-2 assigns roles at 0.72–0.74 and the CLIP encoders at 0.82–0.90, and random CLIP-B identifies fresh nouns better than trained CLIP-B (0.74 against 0.71). Only R² and κ separate trained from random weights consistently."
- Also add random-weight marks to Fig. 6d.

---

## MINOR

**m1. The RAAM gap to the ceiling is stated two ways (L254 "within 0.14", L282 "within 0.13").**

- `agg_r3.txt` l.90: the largest pop_r2 shortfall is V=16, w=16, depth 1: 0.867 against 1.0, a gap of 0.133.
- "Within 0.13" is false strictly.
- Fix: use "within 0.14" at L282 as well.

**m2. "About 0.3 for T5 and CLIP-H" (L265, L282) hides T5-large.**

- Δκ: T5-small +0.304, T5-base +0.297, T5-large +0.257, CLIP-H +0.304.
- Fix at L265: "by 0.26–0.30 for T5 and CLIP-H". L282: "by about 0.3 for T5-small, T5-base and CLIP-H, and 0.26 for T5-large", or simply "0.26–0.30".

**m3. Sum-reference range (L263, "0.47–0.48").** The trained models' sum references are 0.476–0.482, all 0.48 to two decimals. The range "0.47–0.48" only fits if the random-weight rows are included. Write "0.48".

**m4. Role-assignment lower bound (L263, "0.90–1.00").** The minimum is 0.895 (CLIP-B), with BERT-large at 0.899. Write "0.89–1.00" (already in the M2 fix).

**m5. "Leaves the top of the ranking in place" (L267) is slightly loose.**

- In e4wd, CLIP-H (0.665, κ 0.358) moves above GPT-2 small (0.659, κ 0.350). The numbers the text gives are correct.
- Corrected: "Removing it lowers recovery for ten of the eleven encoders and keeps the same four on top (GPT-2 0.66–0.70, κ = 0.35–0.42; CLIP-H 0.67, κ = 0.36)…"

**m6. The dominant-direction claims (L261) need two qualifiers.**

- **"The first component holds 99% of the variance in GPT-2's early layers."**
  - `diag_e4_codes.txt` reports `share_pc1`, which is the share of the *top-16 components'* variance.
  - It is computed on all 305 nouns and on unwhitened codes.
  - Values: phrase L3 1.00, L6 0.99, L9 0.99.
  - Write "99% of the variance of the 16-component code".
- **The CLIP-B/32 token-count R² = 0.98 at L3** (matches `diag_e4_codes.txt`). It is carried by only **7 multi-token nouns out of 305** (`multi-token nouns=7`).
  - Roughly 2 of them fall among the 100 fresh nouns per split.
  - So the "length of the noun" that the CLIP conclusions rest on (abstract, L267) is in effect an indicator for about 7 nouns.
  - State this. For example, after "…explains R² = 0.98 of the code at layer 3": "a feature that separates the 7 of 305 nouns that CLIP splits into several tokens".
  - Consider whether "length of each noun" should be "token count" throughout.

**m7. The pool-size curve is incomplete.**

- `results/e4wn` holds 130 of the 132 tasks in `tasks_e4g.txt`. Missing: `clip-b32_n80_s0` and `gpt2-medium_n165_s0`.
- So GPT-2-medium's 165-noun endpoint in the curve is 0.728 (2 seeds), against 0.724 in the main table.
- The stated ranges still hold:
  - gains from 120 to 165 nouns: 0.023–0.048 for models, 0.007–0.024 for concatenation;
  - κ of GPT-2-large: 0.455 → 0.502.
- Fix: rerun the two tasks, or note "2 seeds" for those cells.

**m8. The "earlier run" claim (Appendix L572) rests on a confounded comparison.**

- The only support I found is `agg_e4b.txt` vs `agg_e4c.txt`.
  - e4b: new-pair R² about 0.87–0.89 while fresh R² is negative (for example BERT-base L3 'on': 0.886 vs −1.24).
- But e4b differs in more than the selection rule:
  - its concatenation reference on fresh nouns is 0.89–0.998, not 1.00, so the PCA or pool differed;
  - `agg_e4b.py` selects on 'on' and 'left' only.
- Corrected: "In an earlier version of the pipeline with unwhitened codes, which also differed in other respects, selecting the penalty on composites as in a standard probe fitted the fitting nouns more closely and transferred worse." Or drop the sentence.

**m9. One claim has no supporting output (L120).** The text says: "we report this share wherever a system-independent description exists (the glyph latents of Section 4.2)".

- No number in the text, and none in any aggregate I found, reports how well an affine map of the glyph part code recovers the latent z.
- Fix: either report it (for example, ridge R² of z from 64 glyph pixels on fresh latents), or change to "a system-independent description exists for glyphs (their latents) but not for nouns".

**m10. Two aggregators disagree on the E2h correlations.**

- `agg_r4_e2e5.txt` (from `agg_r4.py`): corr(vocab_glyph, beh) within 0.755 and pool 0.396, on 512 runs merged with e2v2, using `beh_fresh_1`.
- `agg_e2h_partial.txt` (576 runs, retrained behavior `beh_fresh_r11`): 0.501 and 0.300.
- The paper (L216, Appendix L564) correctly uses the latter, and all of these numbers match it:
  - 0.42 / 0.30 / 0.06;
  - 0.50 / 0.30 / 0.35;
  - 0.48 / 0.30 / 0.26;
  - β 0.45 / 0.53 / 0.73 and −0.09 / −0.04 / −0.31;
  - CI 0.09–0.32;
  - V=32 spread −0.40 to 0.43.
- But the E2h block of `agg_r4.py` is misleading and should be removed or switched to `beh_fresh_r11`.
- "Within 0.08 of the original runs" for non-transformers cannot be verified: the aggregate prints only the overall maximum, 0.161, which supports the transformer's 0.16. Print the maximum per architecture.

**m11. Stale and incorrect artifacts.**

- `results/fig_e2.txt` (10-27 23:14) predates `fig_e2.py` (10-28 01:54). It still prints r_vocab 0.957 and within 0.64 / 0.40 from e2v2. Rerun to log what the figure shows.
- `e4_pretrained.py` docstring (r3-B8, partly fixed):
  - it still says "identification of the part among the 40 fresh nouns"; the code uses 100;
  - it says "Readouts from a declared class (ridge, MLP)", but `RidgeCV` and `MLPRegressor` are imported and unused, and only ridge is run.

**m12. Protocol details the appendix should state.**

1. For "rand-X@X", the layer and code are chosen by the random model's own validation score, and the trained model's codes are taken at that same layer (`run()`: `codes[cname][layer]`), not at the trained model's selected layer. Add: "the layer and code of each random-weight control are chosen on its own validation nouns".
2. Role assignment is defined as the prediction being closer, in Euclidean distance in the whitened code, to the correct part than to the other part. The appendix never defines it.
3. Mean pooling includes special tokens: CLS/SEP, BOS/EOS, `</s>`.
4. Fig. 6c draws the trained-codes dots at max(value, 0). The true values are −0.013 to 0.035. The caption does not say this.

**m13. Hewitt citation (L261, r3-S7).** It still reads "in the spirit of control tasks". That is acceptable, but a random-weight model is an untrained-model baseline, not a control task.

---

## Method claims checked against `e4_pretrained.py` and the launch lists (all match)

| Claim in the paper | Where it is in the code or launch lists |
|---|---|
| Whitening: 16 PCs on the 165 fitting nouns, each divided by its standard deviation on those nouns | `pca_idx = range(N_FIT)`; `T /= sv/sqrt(n)` |
| e4w and e4wd were run whitened | `--whiten` on every task in `tasks_e4e/f.txt` |
| Token-count removal: a quadratic in the token count, fitted on the fitting nouns, removed before PCA | `XT = [1, ntok, ntok²]`; `lstsq` on `pca_idx`; every e4wd task has `--detok` |
| Penalty grid of 25 values from 1e-3 to 1e9 | `--wide` → `logspace(-3, 9, 25)` on every task |
| Penalty chosen on validation nouns (600 pairs of 40 nouns) | `RidgePath.select` |
| "No penalty selected for a trained model lies at the edge of the grid" | Verified both edges with awk over the per-run JSONs at each model's selected layer/code: none at 1e-3 or 1e9. The aggregate itself checks only the top edge. |
| Layer and code chosen on mean validation R² over the 5 asymmetric relations and the 3 seeds | `agg_r4.py` |
| codes-from: random weights scored with the trained model's codes, components and references | `part_codes(codes_from)` replaces `codes` and `ctok`; references are built from those codes |
| Random weights use a single initialization (seed 0) | `torch.manual_seed(0)` |
| Splits 165/40/100 of 305 nouns; 3,000 fit + 800 new-pair sentences, 600 validation pairs, 1,000 fresh pairs | 305 unique nouns, counted with awk |
| κ = (model − sum)/(concat − sum), computed on the means | `agg_r4.py` |
| δ = 1 − R² of an affine map from the whitened codes, on fresh pairs | `evaluate()`. The concatenation-reference δ range 0.044–0.810 matches "0.04 and 0.81". |
| rev_sim normalized so that 0 is unrelated sentences and 1 is identity | `evaluate()` |

---

## Numbers verified (±0.005 or rounding), Sec. 4.5, abstract, Discussion and appendix

- **Main results (e4w):**
  - New pairs 0.642–0.869.
  - Fresh nouns 0.510–0.741.
  - GPT-2 0.708 / 0.724 / 0.741, κ 0.443 / 0.470 / 0.502.
  - CLIP-H 0.664, κ 0.355.
  - Others 0.510–0.613, κ 0.063–0.259; BERT-large lowest.
  - Concatenation reference 1.000.
  - Identification 0.646–0.939; sum-reference identification 0.405–0.436.
- **Random weights:**
  - Scored on the trained model's codes: ≤ 0.035.
  - Scored on their own codes: GPT-2/T5/BERT 0.414–0.471 (κ −0.110 to −0.011); CLIP 0.498–0.556 (κ 0.051–0.151).
  - Δκ from training: GPT-2 0.51–0.61; BERT and CLIP-L 0.080–0.167; CLIP-B 0.037.
- **Caveats in L267:**
  - 'and' control, GPT-2-large: 0.751 / 0.996.
  - Causal role 1 vs role 2 with random weights: 0.737–0.780 vs 0.079–0.341.
  - Trained GPT-2: 0.732–0.801 vs 0.651–0.684.
  - Trained CLIP: 0.714–0.777 vs 0.438–0.550.
  - Token count removed (e4wd): GPT-2 0.659–0.698 (κ 0.350–0.419); CLIP-H 0.665 (0.358); CLIP-B / CLIP-L 0.466 / 0.457 vs random 0.561 / 0.526.
  - Pool curve: see m7.
  - rev_sim 0.611–0.913.
- **Diagnostics in L261:** share 0.99–1.00; CLIP-B L3 R² 0.98.
- **κ cross-reference:** "the networks of Section 4.2 reached κ = 0.73–0.83" matches agg_r4 E2h (0.728 / 0.826; references 0.98 / 0.484).
- **Fig. 6c/d (`fig_e3e4.txt`):** identical to agg; the caption fixes (seeds, clipping in panel a, "means over seeds and relations") are in place.
- **Spot checks elsewhere:**
  - E2: L214, L216, L218 (fixed pairs, aux, width), Appendix L564.
  - E5: L236 spills and L238 fine-tuning values against e5d, all matching.

## Status of the round-3 points

| Status | Points |
|---|---|
| Resolved | B3, B4, B5, B6, B7; S1, S2, S3, S4, S5, S6, S8 |
| Partly resolved | B1 (Discussion fixed, abstract and L267 not: M1); B8 (docstring: m11); S9 (moved to the appendix but confounded: m8); S7 (m13) |
| Open | B2 (M2) |
