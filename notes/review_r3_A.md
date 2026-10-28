# Review R3 -- Reviewer A (ML theory / statistics)

Paper: "Defining Compositional Representations by Recoverability" (main.tex, 593 lines).
Line numbers refer to `recoverability-theory/main.tex`. Code paths are relative to `experiments/`.
I did not run any code. I checked every proof in Appendix A by hand. I checked numbers against `results/agg_r3.txt`, `agg_r2.txt` and `agg_e4c.txt`, and I read the code that produces them (`e1_calibration.py`, `e3_depth.py`, `e4_pretrained.py`, `e5_structure.py`, `agg_r3.py`).

## Verdict: **Minor revision** (close to accept)

The mathematics is now correct, and all of the major r2 items have been substantively addressed:
- the converse of Prop. `span`, now with the B^T B metric and the centring term;
- the calibration pool and the population denominator;
- the population readouts in the recursion experiment;
- the graded Def. 2;
- the interchange controls.

I verified more than 60 numbers in the text against the aggregates. All but six match to rounding; the six are listed in A5.

What remains:
- **One definitional gap.** The informativeness clause of Def. 2 does not close the loophole it was added for.
- **One overclaim about CLIP**, in the abstract and the Discussion.
- **Def. 2 is never fully applied** to the pretrained or recursive systems.
- **A handful of small statement and number errors.**

Scores: Soundness 3.5/4, Presentation 3/4, Contribution 2.5-3/4, Overall 6-7/10.

---

## 1. Verification log

| Result | Status |
|---|---|
| Lemma `psi`, Lemma `lookup`, Prop. `blend`, Prop. `fiber` | Correct. The remark now says "discontinuous injections exist" (r2 N9 fixed). |
| Prop. `ceiling` (l.358-368) and the covariance form (l.371) | Correct. With block-diagonal Σ, E[x\|z] = ΣAᵀ(AΣAᵀ)⁺z. The explained covariance of role i is Σ_i A_iᵀ(AΣAᵀ)⁺A_iΣ_i, so the trace ratio is right. |
| Prop. `ey` (l.373-384) | Correct. "Finite second moments" and Izenman have been added. |
| Prop. `span` (l.389-396) | Correct, with the range caveat in A4. The B^T B metric and the centring term are right. The minimum-norm W vanishes on (BT)^⊥ (b is unpenalised). The intercept compensation b' = -W'(c+Bx̄) is right. |
| Closed form R² = (V-1)/d - (d-V+1)/(Vd) (l.399) | Correct for V ≤ d+1. For Gaussian parts the sample mean of the V vocabulary parts is independent of the centred deviations, so it is independent of S_i. Hence E‖(I-Π_S)x̄‖² = (d-V+1)/V, and the cross term with a fresh x_i vanishes. Values: V=8 gives 0.367, V=12 gives 0.661, V=16 gives 0.934. These match `agg_r3.txt` ols_raw (0.376, 0.665, 0.927). Caveat: the code's x̄ is the mean over 6,000 sampled pairs, not the vocabulary mean, which explains the ≤0.01 gap. |
| Prop. `separable`, Prop. `context` | Correct (Prop. `context` is trivial). |
| Prop. `systematic` (i)-(iv) | Correct. |
| Prop. `approx` (l.433-443) | Correct. ĥ_π - h_π = BQ(ê - x) - η_π with Q a permutation, so ‖BQ‖ = ‖B‖. E‖η_π‖² = E‖η‖² by i.i.d. roles. Then 2a²+2b². The remark that ‖B_πB⁺‖ ≤ ‖B‖‖B⁺‖ is right. |
| Prop. `depth` (l.448-455) | Correct as restated. The rank of the composite map is ≤ min(w, 2^n d). Prop. `ceiling` gives average R² = rank/(2^n d). The affine clause follows from Prop. `ey`. |
| HRR population readout reaching ≈1 when w ≥ 2^n d | Consistent with theory. In the Fourier domain each row of the depth-n map is a rank-one tensor (⊗_levels role spectra) ⊗ (lifted row). Generic points of the Segre variety span the ambient space, so the map has full rank 2^n d generically. |
| Crosstalk heuristic 1 - (2^n-1)d/w | Right to first order. Each crosstalk term C L x keeps about d/w of its energy after L⁺. It is stated without derivation (S6). |
| Leak remark (l.470) | Plausible but unproved, and "interaction" is undefined (S8). |
| Def. 2: κ well-posed? | Yes. A remark could strengthen it: for i.i.d. parts and any U containing s ↦ s/2, the symmetric blend has ε_blend = 1/2 exactly. By symmetry E[x\|x+y] = (x+y)/2, so MMSE = E‖x-y‖²/4 = Var/2. So in population κ = (R² - ½)/(R²_ref - ½), well defined whenever ε_ref < ½. The measured sum references (0.46-0.50) then directly measure estimation bias (S4). |

### Numbers spot-checked against aggregates (match unless flagged)

**E1 (`agg_r3.txt`)**
- V=128: concat, tpr, proj_m1 and proj_m1.5 all 1.00; proj 0.71/0.47; weighted 0.80; sum 0.50; HRR 0.44; MAP 0.47.
- MLP 0.11 vs linear 0.47; lookup 1.00/0.52.
- OLS 0.38/0.93/1.00 vs closed form 0.37/0.93/1.00.
- Robustness: 0.47/0.47/0.38/0.64; kernel 0.01 vs 0.47; ridge 0.27 at V=8. The variance-weighted ceiling 0.80 = H_8/H_16 = 0.804. ✓

**E2 (`agg_r2.txt`, `agg_r3.txt` refs)**
- Population readout 0.83-0.86 at V=8 and 0.89 at V=64; untrained 0.82-0.83; input 0.91; vocabulary readout 0.21-0.24; transformer 0.59/0.41.
- Behaviour 0.20-0.34 → 0.85-0.91; held-out 0.85-1.00, transformer at V=8 0.53-0.61.
- Fixed pairs 0.47-0.64 → 0.81-0.82; transformer 0.32 → 0.45; held-out drops of 0.05-0.10 and 0.32; a further 0.04-0.09.
- Auxiliary loss ≤ 0.14.
- Width: 0.67-0.71 vs 0.79; gaps 0.08-0.14.
- Occlude: 0.75-0.83 and 0.65-0.79; interaction 2-5% vs 14%.
- Role-filler new-role 0.94 vs 0.88-0.93; ≤ 0.21 at V=8; SD 0.23-0.29 (0.90) → 0.01-0.02.
- Within-condition r 0.64/0.40/0.10; CI 0.14-0.36; regression coefficient -0.044. ✓

**E5 (`agg_r3.txt` E5c, `agg_r2.txt` E5)**
- δ 0.02-0.04 / 0.06-0.13; occlude 0.11-0.22 (0.34).
- Reversal outputs 0.86/0.86 and 0.48/0.49; new role 0.81/0.81.
- Donor 0.78 vs 0.81; kept 0.78 vs 0.80.
- Spill 0.16, natural 0.90, randB1 0.75, rand 0.96; V=16 0.35 (0.28 without the transformer), natural 0.85.
- Occlude 0.56/0.70, 0.17/0.25, 0.76/0.68, spills 0.32/0.61 vs 1.06/1.15.
- All fine-tuning numbers (0.81/0.75, 0.98 → 0.91, 0.63, 0.92, 0.94, 0.89/0.91, 0.63 vs 0.85, 0.98/0.77). ✓

**E3 (`agg_r3.txt`)**
- Decomposer ≥ 0.95 for w ≥ 32; 0.56-0.76 at w=16; 0.26 and 49% at depth 3.
- Population readout 0.99 at depth 3.
- HRR 0.96/0.75/-0.13 vs 0.97/0.78/0.03; identification 1.00/0.97/0.66. ✓

**E4 (`agg_e4c.txt`)**
- GPT-2 0.90-0.92 (κ 0.81-0.83); T5-base 0.82 (κ 0.66); random 0.46-0.54 and 0.43.
- Middle group 0.59-0.71 (κ 0.21-0.47); CLIP 0.51/0.52 (κ 0.07/0.09) vs random 0.55/0.52; BERT 0.64/0.59 vs 0.69/0.59.
- Role 0.85-0.98; identification 0.50-0.85 vs sum 0.16-0.39; new pairs 0.58-0.98.
- "and" control 0.92/0.94 vs 0.91/0.94; carrier code 0.89-0.91.
- Pool curve +0.01-0.06, with CLIP B/32 0.50 → 0.51; T5-base identification last (0.50); reversal similarity 0.69/0.85. ✓

---

## 2. Confirmed errors (ranked by severity)

### A1 (moderate; definitional). The informativeness clause does not close the projection loophole (l.117, l.120)

Def. 2 calls a code informative "if parts can be identified from it among held-out candidates, **or** if it retains a declared share of the variance of a system-independent description".

- **The first disjunct is satisfied by the counterexample it targets.** For e(x) = Px with any rank r ≥ 1 and continuous D, the codes of distinct candidates are almost surely distinct. Nearest-neighbour identification from e(x) (or from an exact readout u(h) = Px) is therefore perfect, so the rank-1 projection blend passes.
- **The second disjunct sets no threshold.** "A declared share" can be 1%.

As written, the clause is a reporting convention, not a constraint. So the sentence "Requiring the part code to be informative closes a second loophole" (l.120) is false.

*Fix.* Make informativeness quantitative and tie it to the tolerance. Replacement for the "A part code is informative ..." sentence of Def. 2:

> A part code is *informative at tolerance τ* if an affine map of $e(x)$ recovers a declared, system-independent description $z(x)$ of the part with population $R^2 \ge 1 - \tau$ (identification among held-out candidates is reported but does not suffice, since any injective code identifies noise-free candidates).

Alternatively, define ε against z directly: recover z(x_i) from the state, which composes the two losses. Then report this share ρ_e for every code used:
- Sec. 4.2 (pixels): ρ = 1.
- Sec. 4.3 (16 PCs): ρ = 0.98.
- Sec. 4.5: for example, the R² of the static input embedding's PCs, or of a fixed external word-vector description, from each selected code.

### A2 (moderate; claim vs data). "CLIP's text encoders do no better than random weights" is false for CLIP ViT-H/14 (l.46, l.280)

`agg_e4c.txt`: clip-h14 reaches rec_fresh 0.695 against rand-clip-h14 0.495 (κ 0.41 vs 0.05). Only clip-b32 (0.508 vs 0.552) and sd15-clip-l (0.524 vs 0.518) match their random controls. Sec. 4.5 (l.263, l.265) states this correctly; the abstract and Discussion do not.

*Fix.*
- l.46: "...and CLIP's text encoders do no better than..." should read "...and the two CLIP text encoders that condition text-to-image models do no better than...".
- l.280: "training itself made the difference for GPT-2 and T5 and none for CLIP" should read "...for GPT-2, T5 and the largest CLIP encoder, and none for the CLIP encoders of text-to-image models or for BERT".

### A3 (moderate). Def. 2 is stated as a verdict procedure but never fully applied (l.117, l.216, l.254, l.263)

Def. 2 says a system is compositional at tolerance τ if κ ≥ 1-τ **and** δ ≤ τ, with an informative code, and productive if the same holds with child states as part codes.
- **Sec. 4.2.** κ is reported, but δ and τ are not. "κ ≈ 0.82" holds only at V=64; at V=8, (R² - 0.484)/(0.980 - 0.484) gives 0.69-0.75 for the 0.83-0.86 states.
- **Sec. 4.3.** δ is reported, κ is not.
- **Sec. 4.5.** κ is reported, δ is not. `e4_pretrained.py` already computes an ANOVA interaction (`inter`, 0.007-0.19), which is not δ. No informativeness figure is given, although l.120 promises one.
- **Sec. 4.4.** "Under Definition 2 both relations are productive up to the width ceiling" (l.254) is not what was measured. The productivity clause uses the child composite's state as the part code. The experiment recovers all 2^n leaves from the depth-n state, and no κ or τ is given.

*Fix.*
- Report (κ, δ, ρ_e) in one row per system in Secs. 4.2-4.5, plus the κ range at V=8 and V=64 for Sec. 4.2.
- Compute δ for the pretrained encoders by fitting Eq. (2) from the 16-PC codes to the states on fitting nouns and evaluating on fresh nouns.
- For Sec. 4.4, either measure recovery of the two depth-(n-1) child states from the depth-n state (the literal clause), or reword the clause to "...when the parts are the leaves of composites of depth up to n" and drop "Under Definition 2" from l.254 in favour of "Under Definition 1".
- State τ wherever verdict language is used ("recoverable whatever their vocabulary", l.216).

### A4 (minor-moderate; statement). The closed form of Prop. `span` is stated without its range (l.391, l.399)

"dim S_i = V - 1" holds only for V ≤ d + 1. For V > d + 1 the displayed formula exceeds 1 (V=24, d=16 gives 1.45), while l.196 quotes the closed form as 1.00 at V=24.

*Fix.* At l.399: "For concatenation of $V \le d + 1$ isotropic parts ..., $R^2 = (V-1)/d - (d-V+1)/(Vd)$; for $V \ge d+1$, $S_i = \R^d$ and $R^2 = 1$." The proof's last line should also say that the formula assumes x̄_i equals the vocabulary mean. The code's x̄ is the mean over sampled pairs, which accounts for the 0.01 gap.

### A5 (minor; numbers that do not match the aggregates)

1. **l.216.** "r = 0.96" for the pooled vocabulary-readout correlation: `agg_r3.txt` gives pooled = 0.953, so write 0.95.
2. **l.254.** HRR "$R^2 \ge 0.999$ whenever $w \ge 2^n d$": at w=64, depth 3 (64 = 2³·8) the value is 0.997. Write "≥ 0.997".
3. **l.220.** "it is the lowest under occlude at V=64 (0.68 against 0.76-0.79)": the transformer is lower (0.646, `agg_r2.txt` OCCLUDE tf V=64 m=64). Likewise, "0.47 against 0.58-0.64 at V=16" omits the transformer (0.355). Write "against 0.58-0.64 for the MLP and CNN" and "the lowest of the non-transformer architectures under occlude (0.68 against 0.76-0.79; transformer 0.65)".
4. **l.236.** "spill 0.19" for role 2 is 1/mean(dright/dleft) = 1/5.27, because `agg_r3.py:59` defines the ratio as dright/dleft for every edit. The ratio of means is 0.058/0.286 = 0.20. Define the spill per edit as (untouched half)/(edited half) and compute it the same way for role 1 and role 2.
5. **l.254.** "within 0.13 of the ceiling": the worst cell is 0.133 (V=16, w=16, depth 1). Write "within 0.14", or "about 0.13".
6. **l.216.** "the trained states score κ ≈ 0.82" (see A3): write "κ = 0.69-0.75 at V = 8 and 0.82-0.83 at V = 64".

### A6 (minor). r2 N7 is left over at l.153

"a 'memorization leak' that corresponds to our residual η" still stands, and it contradicts the corrected remark at l.470: the leak is a design-dependent projection of η, not η itself. Replace with: "...a 'memorization leak' that arises from the interaction term η when the training design is incomplete (Remark after Proposition `context`)".

### A7 (minor; statements)

- **l.146.** "A nested linear blend of depth n loses at least $(2^n-1)d$ dimensions" holds only at width d. Prop. `depth` gives $2^n d - w$ in general. Add "of width $d$".
- **l.153.** "role-factorization asks that $[B_1 \cdots B_k]$ have full column rank": Eq. (2) asks no such thing; it is the hypothesis of Prop. `systematic`. Write: "Proposition `systematic` needs only that $[B_1\cdots B_k]$ have full column rank, not that the role subspaces be orthogonal".
- **l.446.** κ(B) for the condition number clashes with the normalized score κ of Def. 2. Use cond(B).
- **l.446.** "jointly predict a **behavioral** quantity, and Section 4.3 measures it": the bound concerns the *state* error of a predicted reversal. Sec. 4.3 reports neither ε of the code readouts nor ‖B‖, so the bound is never evaluated. Report ε, δ, ‖B‖², the bound, and the measured E‖ĥ_π - h_π‖² for one condition (I expect it to be loose by the factor-2 terms; say so), or delete "and Section 4.3 measures it".
- **l.559.** "standardization changes the metric in which the fit projects, as Proposition `span` describes": for standardized concatenation, BᵀB = D⁻² is block-diagonal, so the fit does not mix roles (the case Prop. `span` describes). It projects obliquely within each role. Write "...changes the metric of the projection within each role (Proposition `span` with $B = D^{-1}$)".

---

## 3. Suggestions (not errors)

- **S1. E4 ridge penalty at the grid boundary.** `agg_e4c.txt` "selected alphas": t5-small, t5-base and t5-large all select 10⁵, the top of `ALPHAS = np.logspace(-3, 5, 17)` (`e4_pretrained.py:115`); rand-t5-large selects 10⁴. The states are not standardized (`RidgePath` on raw centred states). T5's κ values (0.33-0.66) may therefore be limited by the grid. Extend the grid to 10⁸, or standardize states, and report whether the T5 numbers move.
- **S2. The concatenation reference in E4 is exact by construction.** The targets are 16 PCs fitted on the fitting nouns, and those PCs are linear functions of the part state lying in the span of the fitting nouns' states. So the concatenation readout is exactly determined for fresh nouns (1.000 at n_fit=165). It therefore calibrates the readout class, not the estimation difficulty the model's nonlinear state faces (cf. l.120, "we calibrate ... since the best attainable values depend on the parts"). Consider adding a nonlinear role-filler reference, e.g. tanh(W[e(x); e(y)]) with W random of width D, which faces the same span problem.
- **S3. The randB1 control (l.234; `e5_structure.py:132`)** draws isotropic coefficients in the 16-dimensional code space, so most of its energy sits on low-variance PCs that the decoder never saw. Draw r from the code covariance, r ~ N(0, Cov(e)), rescaled to the same norm. That separates "role components identified by the readouts" from "in-distribution edits inside range(B₁)". The Discussion's "in-distribution changes only" caveat already hints at this.
- **S4.** Add the two-line remark that ε_blend = 1/2 exactly for i.i.d. parts (Sec. 1). It shows that κ is well posed whenever ε_ref < 1/2, and it turns the measured sum references (0.46-0.50) into a direct read-out of estimation bias.
- **S5.** At l.559, compute the covariance-form ceiling for the 1/j spectrum half-rank projection numerically and print it next to 0.64, instead of asserting that "its exact value is given by" the formula (r2 N6 asked for the comparison).
- **S6. Crosstalk formula.** Give a one-line derivation: each of the 2^n - 1 crosstalk terms is a unitary-circulant image of a lifted leaf, of which L⁺ keeps a share of about d/w. Note that it is first order: observed values fall below it by 0.09-0.23 at intermediate widths (w=64, depth 3: -0.10 vs 0.125; w=256, depth 5: -0.13 vs 0.03).
- **S7.** In the leak remark (l.470), define "interaction": the residual of the two-way additive decomposition on the complete grid of the same parts. Add the three-line argument: the least-squares fit on O returns additive + Π_O η; on a complete balanced grid η ⊥ additive, so the leak is zero; otherwise the parameter error is ≤ ‖η‖_O / λ_min⁺(design Laplacian)^{1/2} up to constants.
- **S8. l.155.** "What we add is the requirement that the readouts be evaluated on parts outside the vocabulary from which they **and the system** were fitted" contradicts l.94 and Sec. 4.5, where nouns are fresh only to the readout. Align it with the r2 N2 reframing (l.151): "...evaluated on parts outside those used to fit them, and compared with the system's own procedures".
- **S9. Remark after Prop. `fiber`, and the notebox (l.123).** Non-injectivity of a continuous map rules out exact recovery of every composite. It does not by itself bound the population ε under unrestricted readouts. Say "exactly" in the notebox ("cannot exactly and continuously encode ...").
- **S10.** Schott et al. 2022 is still absent. `\label{app:human}` (l.527) still sits outside a sectioning command; it resolves to Appendix C, which is harmless but fragile. E4's `r2` uses the sample spread of the fresh targets while Eq. (1) uses the population variance; state this in App. D.

---

## 4. Status of r2 points

| r2 | Status |
|---|---|
| N1a: converse of Prop. `span` false | **Fixed** (orthonormal B / BᵀB metric / centring term; intercept compensation). The range caveat remains (A4). |
| N1b-c: V=8 mismatch; fresh pool of size V | **Fixed.** Pool of 2,000 and population denominator (`e1_calibration.py:89-97`, `r2_pop`); OLS on centred composites is 0.376 vs closed form 0.367; Fig. 3c caption reads V ≥ d+1. |
| N2: fresh parts as estimator; abstract, l.151, l.278 | **Fixed** in the abstract, l.151, l.278; hewitt2019 is cited. **Remaining**: l.155 (S8). |
| N3a: graded Def. 2 | **Fixed** in the definition (κ, τ). **Partly open** in application (A3). |
| N3b: existential code loophole | **Not closed**: the clause is added but is ineffective (A1). |
| N3c: e for composites | **Fixed** in the definition. Not what Sec. 4.4 measures (A3). |
| N4: recursion verdicts | **Fixed.** Population readouts for both systems, the crosstalk formula, the rewritten verdict, Fig. 2 and Table 3 qualified to m < kd. Minor numbers in A5.2 and A5.5. |
| N5: random-edit control | **Fixed** ("destroys" removed; randB1 and natural edits added; V=16 spill reported). A refinement remains (S3). |
| N6: Prop. `depth`, Prop. `approx`, `span` intercept, `ey` | **Fixed.** Covariance form added, but the 0.64 is not compared with it numerically (S5). Notebox citation fixed. |
| N7: leak bound | **Fixed** at l.470 (but unproved: S7). **Open** at l.153 (A6). |
| N8: pretrained references, pool curve, PCA on fitting nouns, penalty on nouns | **Fixed.** New issues are the alpha grid boundary (S1) and the concatenation reference being exact by construction (S2). |
| N9 minor items | Fixed: l.136 count (2 lemmas + 10 propositions, correct), l.61, l.114, l.355, l.216/399 wording, l.194, hadley citation, l.236 "suggests". **Open**: `\label{app:human}`, Schott 2022, comparison table (optional). |

## 5. What would move this to accept

1. A1: a quantitative informativeness condition, reported per code.
2. A2: correct the CLIP claim in the abstract and Discussion.
3. A3: one (κ, δ, ρ_e) row per system; the productivity wording in Sec. 4.4.
4. A4-A7: the one-line fixes above.

S1 (the alpha grid) is a quick rerun that I would like to see before camera-ready, because it could change the T5 ranking.
