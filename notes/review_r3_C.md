# Review R3-C: "Defining Compositional Representations by Recoverability" (third round)

Reviewer C: empirical ML and interpretability methodology (probing, control tasks, causal interventions).

**What I checked.** I read `main.tex` (md5 `5b71a30e`, mtime 2026-10-28 00:57) against:

- the aggregates `results/agg_r3.txt`, `agg_e4c.txt`, `agg_e5.txt`, `fig_e3e4.txt` and `fig_e2.txt`;
- the code in `e4_pretrained.py`, `e5_structure.py`, `e2_refs.py`, `e2_glyph.py` (World, splits, readouts), `agg_e4c.py`, `agg_r3.py`, `fig_e2.py`, `fig_e5.py` and `fig_e3e4.py`;
- the rendered figures `notes/fig_networks.png`, `fig_structure.png` and `fig_depth_pretrained.png`;
- the per-seed values in `results/e2refs/refs.json`, read with awk.

I ran no Python.

**Verdict: minor revision.** Score 7/10 (up from 6). Soundness 3/4, presentation 3/4, contribution 3/4. Confidence 4/5.

The revision did nearly everything I asked for in round 2:

- interchange donors that share neither glyph;
- same-norm random, range(B₁) and natural edits, with a spill measure;
- V = 16 reported;
- training-glyph and both-roles fine-tuning controls;
- input, concatenation and sum references for E2;
- penalty, layer and code all chosen on validation nouns, with PCA on the fitting nouns only;
- random-weight controls for every encoder;
- the corrected Fig. 4c.

Every number I spot-checked matches the aggregates (list in §4). What remains is mostly interpretation. One abstract claim is contradicted by the authors' own new control (CLIP-H). The random-weight controls contain an unreported result that matters (random CLIP identifies nouns better than trained CLIP). E2 and E4 still receive different verdicts for the same κ. And the E2 vocabulary readout uses the penalty selection that the paper itself now calls wrong in E4.

---

## 1. Confirmed mismatches and bugs (ranked by severity)

**B1. The abstract and Discussion claim that CLIP gains nothing from training, but CLIP-H does (L46, L265, L280).**

- **What the text says.**
  - Abstract (L46): "CLIP's text encoders do no better than the same architecture with random weights."
  - Discussion (L280): "training itself made the difference for GPT-2 and T5 and none for CLIP."
  - End of §4.5 (L265): "the CLIP encoders that condition text-to-image models do no better than random weights."
- **What the data show** (`agg_e4c.txt`, kappa table).
  - CLIP-H/14 reaches fresh R² **0.695**, against 0.495 for the same architecture with random weights (κ 0.41 vs 0.05). Training adds 0.20, about as much as it adds for T5-large (0.641 vs 0.423, +0.22) and far more than for BERT (≈0).
  - The claim holds only for CLIP-B/32 (0.508 vs 0.552) and SD-1.5 CLIP-L (0.524 vs 0.518).
  - OpenCLIP ViT-H/14 is the text encoder family of SD 2.x, so the L265 sentence is also wrong as worded.
- **Fix.** Write "the two smaller CLIP text encoders, including Stable Diffusion 1.5's", and in the Discussion "none for BERT and the two smaller CLIP encoders".

**B2. Random-weight CLIP identifies fresh nouns better than trained CLIP, and this is not reported (L263, Fig. 6d).**

- `agg_e4c.txt`, id_fresh:

  | Encoder | Random weights | Trained |
  |---|---|---|
  | CLIP-B | 0.725 | 0.549 |
  | CLIP-L | 0.687 | 0.531 |

- Role assignment shows the same pattern: random CLIP-B 0.887 vs trained 0.856. Random GPT-2 assigns roles at 0.73–0.77. Random T5 sits at exactly 0.50, and its rev_sim is 1.000, so random T5 is position-blind.
- **Why it matters.** L263 offers "assigns roles … far better than the sum reference (0.85–0.98 against 0.50) and identifies them … far above chance … and above its own sum reference" as positive evidence. The relevant baseline for both measures is the random-weight model, which Fig. 6d does not show.
- **Fix.**
  - Add random-weight markers to Fig. 6d.
  - State that role assignment and identification are largely available from untrained architectures with positional information (CLIP, GPT-2), and that only R²/κ separates trained from random.
  - Note that the random T5 control is a pure bag of words (role 0.50, rev_sim 1.00). "Far above random weights" for T5 therefore partly reflects T5's learned positional (relative-bias) information, and this baseline is not comparable to the others.

**B3. The "bag of words" reading of reversal similarity is not supported by the metric (L265).**

- **The claim.** "The composite states of the two smaller CLIP encoders are nearly invariant to role reversal (0.69 and 0.85), consistent with … bag of words."
- **The same metric for other models.**
  - GPT-2-large 0.686, BERT-base 0.683, CLIP-H 0.689, against CLIP-B 0.690.
  - T5-large 0.913 and T5-small 0.836, both higher than CLIP-B.
- **The metric is uncalibrated.** On the concatenation reference, which is perfectly order-sensitive, it ranges from −0.98 (GPT-2) to −0.01 (CLIP-L). The cause: `rev_sim` uses uncentred states and normalizes by the mean cosine of unrelated sentences (`e4_pretrained.py:167-170`), so it depends on each model's anisotropy.
- **Fix.** Drop the bag-of-words sentence, or recompute rev_sim on states centred by the fitting-set mean and report it next to the same quantity for each model's references.

**B4. The Fig. 4d caption still says behavior follows the vocabulary readout "and not the population readout" (L225).** Round-2 M6 is not fixed. The within-condition r for the population readout is 0.40 [0.30, 0.48]. Write "more closely than". Also state that the scatter and the pooled r are computed on values clipped at −0.2 (`fig_e2.py`: `cl = clip(lower=-0.2)`).

**B5. The HRR population readout is not ≥ 0.999 at every w ≥ 2ⁿd (L254).** At w = 64, depth 3 (w = 2ⁿd), `agg_r3.txt` gives pop_r2 = 0.997. Write "≥ 0.997".

**B6. The Fig. 6 caption says "bands show standard errors" for panels c and d, which have no error bars (L270).** `fig_e3e4.py` draws `ax.bar` without `yerr`. Similarly, L185 says "every number is a mean over seeds with standard errors in the figures", but no E4 number in the text or figure has an SE, nor does any E5 number at V = 16. Either add error bars across the 3 noun splits (and across seeds for E5), or qualify L185.

**B7. Fig. 5d "familiar" bars are mislabeled by implication (L243, `fig_e5.py`, `e5_structure.py:152,185-186`).**

- In the `oldctl` condition, `newpart_role1/role2_oldctl` evaluate the *training glyph* that was fine-tuned (`z1 = zold`), not the fresh glyph.
- The caption ("Old pairs and a fresh glyph in each role …") invites the reader to read the orange bar at "familiar" (0.86) as the fresh glyph's role-2 score.
- No "before" value is recorded for that training glyph, so its role-2 change cannot be read off.
- **Fix.** Relabel the panel, and record `zold`'s role-1/role-2 scores before fine-tuning (see S3).

**B8. Artifact hygiene.**

- The `e4_pretrained.py` docstring (L6-L12) still says 80 training nouns, 40 fresh nouns and identification among 40. The code uses 165/40/100.
- `agg_r3.txt` prints the role-2 spill as its reciprocal (`ctl_swapB2_ratio_r1` = 5.27, 12.36), while `fig_e5.py` and the text use dleft/dright (0.19). Print the quantity the text reports.
- The protocol (L570) says nouns are split "for each relation and seed". The code splits once per seed (`run()` shuffles before the relation loop).
- The random-weight control uses a single initialization (`torch.manual_seed(0)`) for all three noun splits, so its "seeds" do not sample initializations. State this.

---

## 2. Substantive issues (suggestions, ranked)

**S1. E2 and E4 give opposite verdicts for the same κ (L216 vs L263–L265; Def. 2 at L117).**

- **The two verdicts.**
  - E2 trained states: κ ≈ 0.82, and they are declared recoverable ("lose almost nothing").
  - GPT-2: κ = 0.81–0.83, and "no encoder we tested matches the concatenation reference".
- **Where the difference comes from.** E2 is judged against the *input* (0.91), and E4 against the concatenation reference. By Def. 2's own tolerance rule (κ ≥ 1 − τ), both are compositional at τ ≈ 0.2 and neither at τ = 0.1.
- **Fix.** Apply one rule to both experiments.
  - Either report κ relative to an input reference in E4 as well, for example the random-weight model or the static-embedding sequence as the "what the input makes available" baseline;
  - or describe both as "partly recoverable, κ ≈ 0.8".

  The upper-right-cell placement of the E2 networks does not depend on this. What depends on it is the contrast drawn in the Discussion (L280).

**S2. The E2 vocabulary readout uses the penalty selection the paper now says is wrong.**

- **The inconsistency.**
  - In E4 the paper concludes "the unit on which a readout is selected matters as much as the readout class" (L263): composite-level selection made CLIP R² −0.35.
  - Every E2 readout (`e2_glyph.py:309-311`, `e2_refs.py:15-17`) chooses alpha by RidgeCV GCV, i.e. leave-one-*composite*-out, on z-scored states. This is round-2 m4 and m5, still open.
- **Evidence that it bites.** In `refs.json` the raw-input vocabulary readout is non-monotone in V:

  | V | Mean | Per-seed range |
  |---|---|---|
  | 8 | 0.26 | |
  | 16 | 0.55 | 0.42–0.65 |
  | 32 | 0.22 | −0.40 to 0.49 |
  | 64 | 0.78 | |

  This is the signature of a double-descent peak: the number of distinct glyphs approaches the 64 effective pixels, and GCV picks a tiny penalty.
- **Two consequences.**
  1. The "input vocabulary readout correlates at only r = 0.10" control (L216) rests on a noisy, partly broken estimator. A noisy regressor correlates weakly with anything.
  2. The headline "vocabulary readout recovers only 0.21–0.24 at V = 8" may be partly an under-regularization artifact rather than the span limit of Prop. 5.
- **Fix.** Select the vocabulary-readout penalty on held-out *glyphs*, leave-glyphs-out, exactly as E4 does with nouns. Then recompute the vocab-readout curve, the within-condition correlations and the input control. Cost: see §5.

**S3. The fine-tuning controls do not isolate role-specific damage (L238, Table 4 row "New parts without forgetting").**

- **The both-roles control proves little.** It trains 3 composites in role 1 and 2 in role 2 (`e5_structure.py:157-159`), so its role-2 score of 0.91 says only that training role 2 helps role 2. "The damage to role 2 is specific to learning in one role" does not follow from it.
- **The informative contrast is not computed.** That contrast is role-2 decrement vs old-pair decrement in the same run:
  - fresh glyph, role 2: 0.749 → 0.630, Δ = −0.12;
  - old pairs: 0.979 → 0.905, Δ = −0.07.

  It needs a paired SE over the 40 runs (there is one new glyph per run).
- **The training-glyph role-2 change is unmeasured.** Its after-value is 0.856 and its before-value is missing; it is what would show whether one-role fine-tuning damages *any* glyph's other role.
- **Fix.**
  - Report the paired difference (role-2 Δ minus old-pair Δ) with an SE.
  - Record before-values for `zold`.
  - Soften the sentence to "fine-tuning in one role lowers the other role by more than it lowers old pairs (0.12 vs 0.07 ± SE)" if the SE supports it.
  - Better: test the paper's own prescription, adding the glyph's code through the fitted B₂ without gradient steps. That is a few lines of code and would make the Table 4 row an actual demonstration.

**S4. The random edit in range(B₁) is isotropic in code coordinates, so it is off the data manifold (L234, L236).**

- **The code.** `randB1 = unit(N(0, I_K) @ B1.T) * nrm` puts equal weight on all 16 PCA coordinates. Glyph codes have a steeply decaying spectrum: the top 16 PCs carry 98% of the variance.
- **Two readings of the result.** Spill 0.75 for this edit is read as "locality is a property of the role components … not of any edit in the range of B₁". It may instead reflect that the decoder never saw large excursions along low-variance PCs.
- **Why it matters.** Table 4 (L518) cites "Locality in the range of B_i (Prop. iv)" as the guarantee, and the randB1 result appears to contradict Prop. iv unless the off-manifold explanation holds.
- **Fix.** Add a covariance-matched control: B₁ applied to the difference of codes of two *random pool glyphs*, normalized to the same norm. If spill stays near 0.16, locality is a property of on-manifold code edits, which is consistent with Prop. iv. If it rises to 0.75, the role components the readout identifies are special. Either result sharpens the claim.
- **Related wording.** The Discussion (L282) contrasts spill 0.16 with "0.90 for a natural change of the same size". But the natural direction h(a′,b′) − h(a,b) changes *both* glyphs by design, so a spill near 1 is expected. Use the range(B₁) or covariance-matched control as the comparison there.

**S5. The E4 random-weight control compares recovery of different targets (L261, L263).**

- **The problem.** Each random model is scored against its *own* part code: the random network's state for "a x". Comparing trained CLIP-B (0.51) with random CLIP-B (0.55) therefore compares recovery of two different 16-d targets. A random network's "a x" state is close to a token-plus-position embedding, so recovering it from a mean-pooled sentence is near-trivial.
- **Fix.** Also map the random model's composite states onto the *trained* model's part code, and onto one system-independent code shared by all models (e.g., GloVe/fastText vectors of the nouns). With a shared target:
  - κ becomes comparable across models and across trained and random;
  - the paper fulfils its own promise at L120, "we report how much of a system-independent description of the parts each code retains", which no E4 number currently does.

**S6. Causal-prefix asymmetry in CLIP is unreported (L265, caveat 2).** Caveat 2 reports only that GPT-2 shows no role-1 advantage. The CLIPs do:

| Encoder | Role 1 | Role 2 |
|---|---|---|
| CLIP-H, trained | 0.80 | 0.60 |
| CLIP-B, trained | 0.55 | 0.46 |
| CLIP, random weights (3 models) | 0.74–0.77 | 0.25–0.34 |

Report these numbers. They show that much of what random CLIP "recovers" is role 1 through the causal prefix.

**S7. Citation of Hewitt & Liang (L261).** Control tasks randomize the *labels* (word type → random class) while keeping the model. A random-weight model is a different baseline (cf. Zhang & Bowman 2018; Conneau et al. 2018). Cite accordingly, or describe it as an "untrained-model baseline".

**S8. Unreported condition in Fig. 6a (L270).** Panel a plots V = 16 only, and its curves are also clipped at −0.5 (`fig_e3e4.py`). The caption states clipping only for panel b. Add "V = 16; values below −0.5 are clipped" to panel a.

**S9. The previous-version number in §4.5 (L263) cannot be checked.** "In an earlier version of this experiment … R² of −0.35 and 0.01" is not reproducible from the current pipeline. Either rerun composite-level selection with the current PCA on the fitting nouns and report that number, or drop the sentence.

**S10. Minor figure issues.**

- Fig. 4d: the inset text still overlaps points near (0, 0).
- Fig. 5d: the legend sits on the 1.1 tick.
- Fig. 6c: the legend entry "random weights, fresh" runs past the axes.

---

## 3. Status of round-2 points

**Fixed:**

- **Number and figure mismatches:** M1, M2, M3, M4, M5/N3 (Fig. 4c now has per-architecture untrained lines and an input line), M7, M8, M9 (e5c has n = 40 per V in both relations), M10, M11, M12 (per-model sum-reference marks), M13, M14, M15, M17.
- **N2:** donors now share neither glyph; there are same-norm random, range(B₁) and natural edits; spill is measured; V = 16 is reported; "destroys" is removed.
- **N8:** PCA is now fitted on the fitting nouns only.
- **N9:** random-weight controls were added.
- **N10:** the role–filler fresh-part deficit is now stated.
- **N12:** the Prop. 4 wording is softened.
- **N16:** GPT-2 carrier-code robustness is now reported.
- **N17:** Prop. 7 now has A, B ∈ R^{w×w}.
- **W7:** "best" is now claimed at V = 64 only.

**Partly fixed:**

- **M6:** the Fig. 4d caption (B4).
- **M16:** the Fig. 6a clipping (S8).
- **N1:** references were added, but the verdicts are inconsistent (S1).
- **N4:** controls were added, but they do not isolate role-specific damage (S3).
- **N5:** a caveat was added ("bounds, rather than measures"). Acceptable.
- **N6:** the CI was added and the input-vocabulary control was run, but that control is unreliable (S2).
- **N7:** the per-metric ranking caveat was added and noun-level selection removes the negative CLIP R². The rev_sim and role metrics remain uncalibrated (B2, B3).
- **N11:** the "invertible" claim is qualified. The BESIDE-output-only OCCLUDE number is still unused.
- **N13:** SEs are missing for E4 and for the V = 16 E5 numbers (B6).
- **N14:** figure readability (S10).
- **N15:** the E4 seeds are now described, but the docstrings are stale (B8).
- **m11:** Hewitt is now cited, but miscast (S7). Elazar et al., Belinkov and Tenney et al. are still absent.

**Open:**

- **m4/m5:** composite-level GCV and z-scoring in E2. This now matters because of the paper's own E4 lesson (S2).
- **m10:** the TRELLIS self-citation remains (L290).

---

## 4. Numbers verified (±0.005)

- **E1:**
  - lookup 1.00 / 0.52;
  - fresh R² at V = 128: 1.00, 0.71, 0.47, 0.80, 0.50, 0.44, 0.47;
  - OLS 0.38 / 0.93 / 1.00 vs closed form 0.37 / 0.93 / 1.00;
  - MLP 0.11 vs linear 0.47;
  - robustness 0.47 / 0.47 / 0.38 / 0.64;
  - kernel 0.01 vs 0.47;
  - ridge 0.27 vs OLS 0.38 at V = 8.
- **E2:**
  - input 0.91, concatenation reference 0.98, sum reference 0.48, κ 0.82;
  - within r 0.64 / 0.40, CI on the difference [0.14, 0.36];
  - input-control r 0.10;
  - regression coefficients 0.56 / −0.04.
- **E5 (`agg_r3.txt`, e5c), OVER:**
  - δ and reversal-state ranges (from e5b);
  - reversal decoding 0.86 / 0.86 and 0.48 / 0.49; new role 0.81 / 0.81;
  - donor 0.78 vs 0.81; untouched glyph 0.78 vs 0.80;
  - spill 0.16; natural 0.90; range(B₁) 0.75; random 0.96;
  - V = 16: spill 0.35, 0.28 without the transformer; natural 0.85.
- **E5, OCCLUDE:**
  - reversal 0.56 / 0.70 and 0.17 / 0.25;
  - spill 0.32 / 0.61, natural 1.06 / 1.15;
  - new role 0.76 / 0.68.
- **E5, fine-tuning:**
  - before 0.81 / 0.75; held-out 0.98 → 0.905 (rounds to 0.90 or 0.91);
  - role 2 0.63, role 1 0.92;
  - training-glyph held-out 0.935 (Δ 0.04 vs 0.07);
  - both roles 0.89 / 0.91;
  - V = 16: 0.97 → 0.63 vs 0.85;
  - replay 0.98 / 0.77.
- **E3:**
  - RAAM: w ≥ 32 gives ≥ 0.95; at w = 16, 0.56–0.76; depth 3 at w = 256: 0.26 and 49%; population readout 0.99; within 0.13 of the ceiling.
  - HRR: 0.96 / 0.75 / −0.13 vs 0.97 / 0.78 / 0.03; identification 1.00 / 0.97 / 0.66.
- **E4:**
  - references: concatenation 1.00, sum 0.46–0.50; new pairs 0.58–0.98;
  - GPT-2 0.90–0.92 (κ 0.81–0.83); T5-base 0.82 (κ 0.66); random 0.46–0.54 and 0.43;
  - middle group 0.59–0.71 (κ 0.21–0.47); CLIP-B/L 0.51 / 0.52 (κ 0.07 / 0.09) vs random 0.55 / 0.52;
  - BERT 0.64 / 0.59 vs 0.69 / 0.59;
  - role assignment 0.85–0.98; identification 0.50–0.85; sum-reference identification 0.16–0.39;
  - "and" control 0.92 / 0.94 vs 0.91 / 0.94; GPT-2 roles 0.90–0.92; carrier code 0.89–0.91;
  - noun-curve gain 0.01–0.06; CLIP-B 0.50 → 0.51;
  - T5-base fourth → last, identification 0.50; rev_sim 0.69 / 0.85.

Mismatches: B1 (claim vs data), B5, B6. All other values agree.

---

## 5. Additional experiments I consider necessary, with cost

1. **E2 vocabulary readouts with leave-glyphs-out penalty selection** (S2).
   - Recompute the vocab readout, the input-vocabulary control and the within-condition correlations.
   - If states are not cached, this needs the 576 + 480 E2 runs again. These are CPU-only, a few minutes each, so about 2–4 CPU-node hours on the existing sbatch array. Only the readouts change.
2. **E4 shared-target control** (S5, B2).
   - Fit readouts from trained and random-weight states to (a) the trained model's code and (b) a system-independent noun embedding (GloVe 300-d to PCA-16, fitted on the fitting nouns).
   - Add random-weight identification and role assignment to Fig. 6d.
   - The embeddings are already computed per run; one extra pass over 22 models × 3 splits is about 1–2 GPU-hours.
3. **E5 covariance-matched range(B₁) edit, `zold` before-values, paired SEs, and code-addition through B₂** (S3, S4).
   - These are additions to `e5_structure.py`; rerun the 160-run e5c array, about 1–2 CPU-node hours.

Not necessary but useful: report OCCLUDE behavior on the BESIDE output only (`beh_fresh_r10`, already in the JSON; no compute).
