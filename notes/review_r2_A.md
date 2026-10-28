# Review R2 -- Reviewer A

Paper: "Defining Compositional Representations by Recoverability" (revised main.tex, 575 lines; rendered p01-p27).
Line numbers refer to `recoverability-theory/main.tex`. Where I cite code, paths are relative to `experiments/`.
I did not run any code. Every claim below comes from reading the math, the code, and the aggregated results (`results/agg_r2.txt`, `agg_e5.txt`, `agg_e4b.txt`, `agg_e1rob.txt`).

---

## 0. Summary of the revision

The revision does real work. Def. 1 is now a population quantity with an explicit probe-pool estimator (l.100-111). A new Eckart-Young proposition covers arbitrary relations under affine readouts (Prop. `ey`, l.373-384). Prop. `span` now requires all role spans (l.391). Prop. `systematic` has an approximate-error bound (l.431). The networks section separates a population readout from a vocabulary readout and reports within-condition correlations (l.212-216). The pretrained analysis has validation-selected layers, dimension-matched references, and identification and role accuracy in the main text. The HRR baseline has been rebuilt.

What remains:
- The converse half of Prop. `span` is false as stated.
- The small-V calibration numbers still disagree with the paper's own closed form, and part of the reason is a design flaw in `e1_calibration.py`.
- Def. 2 is still binary in practice, and it has a new loophole through the existential choice of part code.
- The recursion verdict is not obtained from Def. 1/2.
- Several remarks still assert bounds that are not proved.

---

## 1. Status of round-1 issues

| R1 issue | Status | Comment |
|---|---|---|
| **W1** Def. 1 depends on \|V\| and the fitting rule | **Fixed** | Def. 1 is now an existential population risk (l.101-111), and l.114 separates the definition from its estimation. This creates a framing problem (new issue N2). |
| **W2** For affine U, "fresh" reduces to "enough parts" | **Partly fixed** | l.114 now says fresh parts are "the unit of evidence" because they remove an estimation bias. That concedes the R1 point. The abstract (l.46: "the requirement of fresh parts carries the definition") and the Discussion (l.278: "is what gives the definition content") still make the stronger claim (N2). |
| **W3a** Prop. `span` all-role condition | **Fixed** in the necessity half. The new converse is wrong (N1). |
| **W3a'** Min-norm prediction vs. the reported 0.05 at V=8 | **Not fixed** | The paper now *states* the min-norm prediction (R^2 = dim S_i/d, l.391). That makes the discrepancy with 0.05 (l.196) explicit instead of resolving it (N1). |
| **W3b** Prop. `depth` dimensions | **Not fixed** | l.435 still fixes A, B in R^{d x d} and then speaks of "states of width w" (N6). |
| **W3c** Prop. 3 used for nonlinear networks | **Fixed** | Prop. `ey` was added and l.212/218 now compare with it. I checked `e2_glyph.py:247-253`: the ceiling is the top-m share of the stacked two-role pixel spectrum. The probes are averaged over roles (`agg_r2.py:26`) and `r2` is variance-weighted (`e2_glyph.py:188`), so the comparison is the one the proposition bounds. |
| **W3d** Remark after Prop. 2 (space-filling maps) | **Partly fixed** | l.387 now says nonlinear readouts are not bounded. l.355 still says "discontinuous injections ... are not stable under any noise" without support, and l.114 still says every blend into a smaller space "must" discard information without saying "under affine readouts". |
| **W4** Prop. 6 tautological; learning version | **Partly fixed** | The claims are toned down (l.144: "a statement about what the states determine"), and a perturbation bound was added. There is still no learning version. The bound uses B^+, while the experiments use a ridge readout (N6). |
| **W5** Novelty / comparison | **Partly fixed** | TPDN, Uselis, Schug, Lippl and Hupkes are now discussed. There is no comparison table and no new identifiability result. Two of the stated correspondences are inaccurate (N2, N7). Hewitt and Liang (2019) is still in the bib but uncited, and Schott et al. 2022 is absent. |
| **W6** Pretrained confounds | **Mostly fixed** | Layer and code are selected on validation nouns; 305 = 165+40+100; identification and role accuracy are in the text. The references are now built from the *unreduced* part states (`e4_pretrained.py:178-180`), which partly controls the span effect. The text does not say this (N8). There is still no curve over the number of fitting nouns. |
| **W7** Def. 2 not applied to networks; circular correlation | **Mostly fixed** | Sec. 4.3 declares e (16 PCs, 98% of variance) and reports delta. Within-condition correlations are reported (r = 0.40, CI 0.30-0.48, matching `agg_r2.txt`). The networks are moved to the upper-right cell of Table 1. One misreading in the intervention control remains (N5). |
| **W8** Def. 2 collapses to exact; e for composites | **Not fixed** | See N3. |
| **W9** HRR width-independence | **Fixed** | The unitary per-level roles now behave as capacity analysis predicts. The verdict drawn from them does not follow (N4). |
| Minor: abstract "ceiling exactly by rank" | Fixed | |
| Minor: elmoznino year, 305 nouns, CLIP clipping, Table 4 "(conjecture)", "general position" | Fixed | |
| Minor: "proves eight results" | **Not fixed** | The appendix now has 2 lemmas and 9 propositions (11 results), l.136. |
| Minor: notebox cites Prop. `depth` for continuous parts | **Not fixed** | l.123. |
| Minor: Hadley uncited at the strong-systematicity mention | **Not fixed** | l.210. |
| Minor: HRR ceiling "averages 1/2" | Not fixed (cosmetic) | l.371. |
| Minor: `\label{app:human}` on no sectioning command | Not fixed | l.509. |
| Minor: "the residual delta bounds" the leak | **Not fixed** | l.455; see N7. |
| Anonymity | n/a | The `preprint` style suggests an arXiv preprint; ignore if not double-blind. |

---

## 2. New issues, ranked by severity

### N1 (major). The converse of Prop. `span` is false for general B, and the V=8 calibration value still contradicts it (l.391, l.395, l.196; `e1_calibration.py`)

**(a) The converse is false.** Take the minimum-norm exact fit and centre so that b absorbs the means. The constraint is W(Bv) = v for v in T, and the min-norm W is zero on (BT)^perp. For a fresh composite it therefore returns the vector v in T that minimises ||B(x - xbar) - Bv||. That is the projection of x - xbar onto T **in the metric B^T B**, not the Euclidean projection.

The two coincide only when T is invariant under B^T B, for instance when B has orthonormal columns. Otherwise the role-i prediction depends on the other roles' codes.

*Counterexample* (k = 2, d = 1). Let B = [[1, 1], [0, 1]], so B_1 = (1, 0)^T and B_2 = (1, 1)^T, with full column rank. Train with role 1 varying and role 2 fixed, so S_1 = R, S_2 = {0}, and T = S_1 (+) S_2 = span{e_1}. The min-norm W is [[1, 0], [0, 0]]. A fresh composite that changes only role 2 by +1 gives B(0, 1)^T = (1, 1)^T. W maps this to (1, 0): the role-1 prediction moves by 1 although the role-1 code did not change. The proposition predicts the projection onto S_1 of a zero change, which is zero.

The proof's sentence "the minimum-norm solution is W = B^+ followed by the orthogonal projection onto T" is also wrong. In the same example, Pi_T B^+ = [[1, -1], [0, 0]] has squared Frobenius norm 2, while the min-norm solution has norm 1.

The stated R^2 also ignores the centring offset. Fresh codes have mean 0, but the fit is centred at xbar, so the error is (I - Pi)(x - xbar) and R^2_i = dim S_i/d - E||(I - Pi_{S_i}) xbar_i||^2/d.

*Replacement text for the last sentence of the statement:*
> Conversely, if $B$ has orthonormal columns (as for concatenation) and the fitted pairs vary each role independently enough that $T = S_1 \oplus \cdots \oplus S_k$, the minimum-norm fit returns $\bar{x}_i + \Pi_{S_i}(x_i - \bar{x}_i)$ for each role, so that for isotropic codes $R^2_i = \dim S_i/d - \mathbb{E}\lVert(I - \Pi_{S_i})\bar{x}_i\rVert^2/d$. For general $B$ the minimum-norm fit returns the projection of $x - \bar{x}$ onto $T$ in the metric $B^\top B$, which mixes roles.

*Replacement for the last proof sentence:*
> For the converse, the minimum-norm $W$ vanishes on $(BT)^\perp$ and inverts $B$ on $BT$, so $u(h) - \bar{x}$ is the $B^\top B$-orthogonal projection of $x - \bar{x}$ onto $T$; when $B^\top B = I$ this is the Euclidean projection, whose $i$-th block is $\Pi_{S_i}$ when $T$ is the direct sum.

Note that the calibration standardises composites (`e1_calibration.py`, `mu, sd = Xtr.mean(0), Xtr.std(0)`). So even for concatenation the fitted B is diagonal but not orthonormal, and the clean formula holds only approximately.

**(b) The V=8 number still disagrees with the closed form.** For concatenation with d = 16 and V = 8, dim S_1 = 7.
- With the population denominator of Eq. (1), the corrected formula gives R^2 of about 7/16 - (9/16)(2/16), which is about 0.37.
- With the estimator the code uses (denominator = spread of the fresh test parts about their own mean), it gives about 1 - (7.9 + 2.25)/14, which is about 0.28.
- The paper reports 0.05 (ridge; l.196, `agg_e1rob.txt`, gauss/linear/V=8) and 0.20 for OLS on one seed (l.541).

**(c) Root cause in the code.** In `e1_calibration.py:run`, the fresh pool is `fresh = sample_parts(rng, V, d, dist)`, and fresh pairs index into it (`fr = ... rng.integers(0, V, ...)`). At V = 8, "fresh" R^2 is therefore computed on **8 fresh parts**, and `r2` normalises by their sample spread. That is a very noisy, downward-biased estimate of the population quantity in Def. 1, which the paper says is estimated on "composites of other parts drawn from D" (l.111). The same pattern occurs in `e3_depth.py:run_hrr` (`fresh = rng.normal(size=(V, D_PART))`).

*Fix.*
- Draw a fixed fresh pool (e.g. 2,000 parts) independent of V.
- Use the population variance (d for isotropic parts) in the denominator, matching Eq. (1).
- Overlay the corrected closed form on Fig. 3c.
- Change the caption of Fig. 3c from "V >= d = 16" to "V >= d + 1", as Prop. `span` itself says (l.391).

### N2 (major). Under the population definition, fresh parts are held-out evaluation, not the content of the definition. The abstract, Discussion and novelty claims must be brought in line (l.46, l.114, l.151, l.155, l.278)

With Def. 1 as written, epsilon = inf over u in U of the population risk under D^k. For affine U this is the population linear-probe R^2 of each role's code. Delta is the population residual of an affine (TPDN-style) fit. Fresh parts enter only through the estimator: the probe pool is drawn from D, so test parts are *in-distribution* for the readout.

This is the standard held-out-evaluation principle, and Hewitt and Liang (2019), which is in the bib, is the canonical argument for restricting U. The out-of-distribution question, "trained on a finite V, tested on new parts", is now about the **system's own procedures** (the vocabulary readout and the decoder in Sec. 4.2), not about Def. 1.

The paper's best empirical result follows exactly this dissociation: states are recoverable regardless of V, decoders are not (l.216). But the abstract still says "the requirement of fresh parts carries the definition", and l.151 claims that requiring fresh parts is one of "two things [that] are new".

Relative to Lachapelle et al. 2023 / Wiedemer et al. 2024: fresh draws from D lie inside each slot's marginal support of D. So the population criterion is an affine analogue of their Cartesian-product target, not an extrapolation beyond it. The correspondence at l.151, "our recoverability is their consistency", is inaccurate. Compositional consistency is a constraint that the encoder inverts the *decoder's* outputs on recombined latents. Def. 1 is a readout of encoder states with no decoder involved.

*Fix.*
- Abstract, replace "We show that the requirement of fresh parts carries the definition." with:
  > "We show that recovery must be measured over the distribution of parts, and estimated on parts not used to fit the readout: on any finite vocabulary ..."
- l.278, replace "The requirement of fresh parts is what gives the definition content" with:
  > "Measuring recovery over the part distribution, on parts not used for fitting, is what gives the definition content"
- l.151, replace "our recoverability is their consistency" with:
  > "our recoverability is a readout-side, affine analogue of their consistency condition"
- Replace "we require recovery of fresh parts, which separates what the relation guarantees from what a vocabulary supplies" with:
  > "we separate recovery by a population readout from recovery by procedures learned on the system's own vocabulary, and show that the two dissociate in trained networks"
- Cite hewitt2019 at l.120 next to sutter2025.

### N3 (major). Def. 2 is still binary in practice, and the existential part code admits lossy blends (l.117, l.120, l.236, l.254)

**(a) Binary in practice.** The reference role-filler operator applied to the same codes has delta = 0 exactly, and epsilon of about 0 (concatenation: 0.89-1.00 in `agg_e4b.txt`, 1.00 in E1). "No worse than the reference" therefore demands delta = 0 and epsilon of about 0. No trained system in the paper passes:
- delta = 0.02-0.04 (`agg_e5.txt`);
- GPT-2 large has R^2 = 0.84 against 1.00.

The text nonetheless uses verdict language: "close to role-factorized" (l.236), "HRR is productive" (l.254).

*Fix.* Make the graded reading explicit in the definition. Replacement for l.117 from "with epsilon and delta no worse than ..." to the end of the sentence:
> "and we report it by the normalized scores $\kappa_\varepsilon = (\varepsilon_{\mathrm{blend}} - \varepsilon)/(\varepsilon_{\mathrm{blend}} - \varepsilon_{\mathrm{ref}})$ and $\kappa_\delta = 1 - \delta/\delta_{\mathrm{blend}}$, where the references are a role--filler operator and a symmetric blend applied to the same part codes; it is compositional at tolerance $\tau$ if $\kappa_\varepsilon, \kappa_\delta \ge 1 - \tau$."

Then report the kappas in Fig. 5c and in Sec. 4.3.

**(b) Loophole.** "for some part code found under M" is existential. Take g(x, y) = [Px; Py] with P a rank-r < d orthogonal projection. This is a blend that discards d - r dimensions of each part.
- With e = id, epsilon = 1 - r/d, so it fails.
- With e(x) = Px, epsilon = 0 and delta = 0, matching the concatenation reference on those codes, so it **passes**.

Every verdict about blends (Table 3, Fig. 2) is therefore relative to e = identity. Sec. 4.3 handles this well by declaring e independently of the system and reporting its variance share (0.98). Pretrained Sec. 4.5 uses system-dependent codes.

*Fix.* Add to Def. 2:
> "...for some part code found under $\M$ that is informative about the part, in that parts are identifiable from $e(x)$ among held-out candidates (or $e$ retains a declared share of a system-independent description of the part)."

Report this for each code used.

**(c) Productivity clause.** "the same readouts applied recursively" still does not define e for a composite. State "with $e(\text{composite}) := g_r(\text{composite})$, the composite's state".

### N4 (major-moderate). The recursion verdicts are not obtained from Def. 1/2 (l.252-254; `e3_depth.py`)

- **Learned composer.** The RAAM's "fresh-leaf R^2" (0.26 at depth 3) is the R^2 of its *own* decomposer applied three times (`run_raam`: `model.decode(model.encode(...), n)`). That is the system's procedure, not a readout from U. Sec. 4.2 shows that these two dissociate. So "the composer learned a relation that is invertible on the composites it was trained on, not on its own outputs" (l.254) is not established. The depth-3 *state* may still be affinely recoverable (Prop. `ey` ceiling = 1 at w = 256), with the failure lying in the decomposer.
- **HRR.** The readout is correlation unbinding followed by the pseudo-inverse of the lift. With unitary per-level roles, the depth-n state is a linear map of rank min(w, 2^n d). By Prop. `ceiling`, the best affine readout recovers fresh leaves with R^2 = 1 whenever w >= 2^n d. Under Def. 1, HRR at w = 256 is therefore epsilon of about 0 at depth 3 (and at depth 5, where 2^5 * 8 = 256).
- The observed HRR values are what correlation unbinding gives. Each of the 2^n - 1 crosstalk terms keeps a fraction d/w of its energy after the pinv of the lift, so R^2 is about 1 - (2^n - 1)d/w. That gives 0.78 at depth 3, w = 256 (reported 0.73), 0.03 at depth 5, w = 256 (reported -0.21), and 0.76 at depth 5, w = 1024 (reported 0.69).
- "Graceful degradation" is thus a property of the chosen readout, not of the relation. This also contradicts Fig. 2 and Table 3, which place HRR among non-recoverable modes. That placement holds only when w < kd (or w < 2^n d).

*Fix.*
- Fit a ridge population readout (probe pool of fresh trees) from the depth-n state to the 2^n leaves for both systems, and report it next to the decomposer and unbinding numbers.
- State the crosstalk formula.
- Rewrite the last two sentences of l.254 as:
  > "Under an affine population readout, HRR with per-level unitary roles is exactly recoverable whenever $w \ge 2^n d$; correlation unbinding loses the crosstalk, $R^2 \approx 1 - (2^n - 1)d/w$. Whether the learned composer's failure at depth 3 lies in its state or in its decomposer is decided by the population readout."
- Qualify Fig. 2 and Table 3: "fixed-width (w < kd) superposition".

### N5 (moderate). The random-edit control in the interchange intervention is misread (l.236; `e5_structure.py:120-125`)

`swap_rand_left` is the R^2 between the left half of the randomly edited output and the **donor** glyph a'. For two independent glyphs from the same distribution, E||g - g'||^2 = 2 Var, so R^2 is about -1 by construction. The reported -1.12 is what one sees if the random edit leaves the output **unchanged** (still showing a). "the random edit of the same norm destroys the output" is therefore unsupported. The data may even indicate the opposite: the decoder ignores norm-matched off-subspace edits.

*Fix.*
- For the random edit, report R^2 of the left half against the *original* glyph a, and of the right half against b.
- Add a second control: a random edit of the same norm *inside* range(B_1). This separates "the B_1 direction carries role 1" from "any change in range(B_1) changes role 1".
- Replace "destroys the output ($R^2 = -1.12$...)" with the measured change.
- The 15% leak (l.236) is the V = 64 value. At V = 16 it is 0.27-0.33 for non-transformer architectures and 0.50-0.56 for the transformer (`agg_e5.txt`, `swap_right_change_r1`), so state both.

### N6 (moderate). Proposition fixes

**Prop. `depth` (l.435-440).** The w-dependent bound concerns a lifted composer. Replacement statement:
> Let leaves in $\R^d$ be lifted to $\R^w$ by a linear map and composed by $g(x, y) = Ax + By$ with $A, B \in \R^{w \times w}$. The map from the $2^n$ leaves of a balanced composite of depth $n$ to its state has rank at most $\min(w, 2^n d)$, so its kernel has dimension at least $2^n d - w$ (at least $(2^n - 1)d$ when $w = d$), and for Gaussian leaves the average fresh-leaf $R^2$ of any readout is at most $\min(1, w/(2^n d))$. For any relation, linear or not, the same bound holds for affine readouts by Proposition~\ref{prop:ey}.

The last sentence also repairs l.443. The nonlinear composer *is* covered for affine readouts, which is one more reason for N4's population readout. In the notebox (l.123), cite the Remark after Prop. `fiber` (no continuous injection from R^{kd} into R^m when m < kd, by invariance of domain) for "continuous parts", not Prop. `depth`.

**Prop. `systematic`, approximate bound (l.431).** The inequality is correct: ĥ_π - h_π = B_π B^+ η - η_π, and ||B_π B^+|| <= κ(B). But the experiments do not use B^+ (l.546); they use a ridge readout ê. A bound in the paper's own two quantities is just as short and covers what was run. Suggested replacement for the sentence beginning "When the relation is only approximately role-factorized":
> When the relation is only approximately role-factorized, $h = c + Bx + \eta$, and the parts are read by any readouts $\hat{e}_i$ (not necessarily $P_i$), the predicted permuted state $\hat{h}_\pi = c + \sum_i B_{\pi(i)}\hat{e}_i(h)$ satisfies $\hat{h}_\pi - h_\pi = B_\pi(\hat{e}(h) - x) - \eta_\pi$, so that, for exchangeable roles, $\mathbb{E}\lVert\hat{h}_\pi - h_\pi\rVert^2 \le 2\lVert B\rVert^2 k\,\varepsilon\,\mathbb{E}\lVert e - \mathbb{E}e\rVert^2 + 2\delta\,\mathbb{E}\lVert g - \mathbb{E}g\rVert^2$. With $\hat{e} = B^+(h - c)$ the first term is replaced by $\lVert B_\pi B^+\rVert^2 \le \kappa(B)^2$ times $\mathbb{E}\lVert\eta\rVert^2$.

I checked this. ||B_π|| = ||B|| because a block permutation is an orthogonal right factor. E||η_π||^2 = E||η||^2 when the parts are i.i.d. across roles. Both terms then follow from ||a + b||^2 <= 2||a||^2 + 2||b||^2. This is the one place where Def. 1's epsilon and delta jointly predict a behavioural quantity, and it deserves to be a numbered statement.

**Prop. `span`, first half (l.395).** "Any W' with W'BT = 0 can be added without changing the fit" needs a compensating intercept, b' = -W'(c + B xbar). Otherwise W'(c + B x_t) need not vanish.

**Prop. `ey` (l.375).** Add "with finite second moments". The bound does not depend on g at all; it bounds any affine function of any m-dimensional vector. Say so, and cite reduced-rank regression (Izenman 1975) along with Eckart-Young. The proof is correct.

**Prop. `ceiling`, non-isotropic codes.** The robustness paragraph (l.541) explains the correlated-parts value 0.61 by "the variance-weighted ceiling of Prop. `ey`". For the 1/j spectrum, that ceiling is sum_{j<=8} (1/j) / sum_{j<=16} (1/j), about 0.80. It is an upper bound, not the predicted value for a random projection. The exact value follows from the Gaussian MMSE with covariance Sigma:
> R^2_i = tr(Σ_i A_i^T (A Σ A^T)^+ A_i Σ_i) / tr Σ_i.

Add this one-line generalisation to Prop. `ceiling` and compare 0.61 with it.

### N7 (moderate). Unproved bounds in remarks (l.153, l.455)

l.455: "The residual delta of Definition 1 bounds it [the memorization leak]." Write g = (additive part) + η on the observed design O. The least-squares additive fit is additive part + Π_O η, so the leak in φ̂_1(x) is a component of Π_O η. That gives:
- on a complete balanced grid, η is orthogonal to the additive space and the leak is **zero whatever delta is**;
- on an incomplete design, |leak| <= C_O ||η||_O, where C_O depends on the connectivity of the design (the smallest nonzero eigenvalue of its bipartite Laplacian).

Also, delta in Def. 1 is a residual from a role-factorized fit B_i e(x_i), not the role-specific φ_i of Prop. `separable`. Replacement:
> "The leak is a component of the projection of the interaction onto additive functions under the observed design; it vanishes for a complete balanced design and is otherwise at most a design-dependent constant times the interaction's norm on the observed pairs."

Correspondingly, at l.153 replace "a 'memorization leak' that corresponds to our residual η" with "a 'memorization leak' that arises from the interaction term η when the training design is incomplete".

### N8 (moderate). Pretrained encoders: say what the references control, and add the pool-size curve (l.261-265)

- **Say what the references are.** They are built from the *unreduced* part states (concatenated, dimension 2D, or summed, dimension D; `e4_pretrained.py:178-180`), while targets are 16 PCs. The reference readouts therefore face the same span problem as the model's. This is visible in concatenation reaching only 0.887 for BERT-large and 0.912 for T5-large (static codes). This is a good control, and the text should say so: "The references are built from the unreduced part states, so their readouts face the same extrapolation from 165 fitting nouns as the model's." It answers most of R1 W6.
- **Add the curve.** The held-out risk of a fitted readout overestimates epsilon at a finite pool. The bias is largest when the composite's noun-dependent variance spans more directions than the 165 fitting nouns can cover, which a linear reference cannot mimic. "CLIP ... furthest" (l.265) needs rec_fresh as a function of N_FIT (e.g. 40, 80, 120, 165) for the model and the references. If the CLIP curve is still rising at 165, the verdict is limited by estimation, by the paper's own Prop. `span`.
- **Minor.** The 16 PCs are fitted on all 305 nouns, fresh ones included (l.552). That is transductive with respect to the target definition. It probably hurts rather than helps, but fit the PCs on the fitting nouns, or state the choice as part of e.
- **Minor.** The ridge penalty is chosen by GCV over composites, which share nouns. That is leave-one-composite-out, not leave-one-noun-out, and favours memorisation. The same pipeline is applied to the references, so it is a matched confound, but say so.

### N9 (minor)

- **l.136.** "proves eight results; we state the four" should read "proves eleven results (two lemmas and nine propositions); we state the four".
- **l.61.** "give ceilings on recovery for linear and for arbitrary relations" should read "... and, under affine readouts, for arbitrary relations". Same at l.46 ("by the width of any state" should read "and, for affine readouts, by the width of any state").
- **l.114.** "as every blend into a space smaller than its parts must" should read "as, under affine readouts, every blend into a space smaller than its parts must (Proposition `ey`)".
- **l.355.** Drop "but are not stable under any noise", or cite a result.
- **l.216, l.399.** "as Proposition `span` predicts for any procedure learned from composites" overreaches: the proposition concerns affine procedures of an affine g. Write "consistent with the mechanism of Proposition `span`".
- **l.236.** "which shows that the network's encoder, not the readout, is the less systematic of the two" should read "which suggests ...". The comparison does not isolate the encoder.
- **l.194.** "behave exactly like a half-rank projection". The values (0.42, 0.45) sit 0.05-0.08 below their ceilings, as the projection does. Say "within 0.08 of the same ceiling, like the half-rank projection". The finite fresh pool (N1c) is a likely cause.
- **l.210.** Cite hadley1994 at "Hadley's strong systematicity".
- **l.509.** Move `\label{app:human}` after a `\subsection*{...}`, or onto the table caption.
- **Related work.** Cite hewitt2019 (control tasks) and Schott et al. 2022. A five-row comparison table (assumption / object / conclusion / regime) against Lachapelle, Wiedemer, Brady, Uselis and McCoy would settle the novelty question in one place.

---

## 3. Verification log (what I checked against the math)

- **Lemma `psi`, Lemma `lookup`, Prop. `blend`, Prop. `fiber`, Prop. `ceiling`.** Correct. The Gaussian conditional mean, the residual I - P and the block traces all check out.
- **Prop. `ey`.** Correct: pointwise projection onto L, then PCA optimality over affine L of dimension at most m. Isotropic case: min(1, m/(kd)). Implementation (`e2_glyph.py:247-253`) and usage (role-averaged, variance-weighted R^2) are consistent with the statement. The numbers at l.218 match `agg_r2.txt` (m = 8: 0.67-0.71 vs 0.787; m >= 16: gap 0.08-0.14).
- **Prop. `span`.** The necessity half is correct (after adding the intercept compensation). The converse is false in general (N1a).
- **Prop. `separable`.** Correct; I verified the telescoping sum term by term.
- **Prop. `systematic` (i)-(iv).** Correct (B^+B = I gives P_j B_i = 0 for j != i). The perturbation inequality is correct.
- **Prop. `depth`.** The first statement is correct for w = d; the second is mis-dimensioned (N6).
- **Prop. `context`.** Correct and trivial. The remark after it is unproved (N7).
- **Numbers spot-checked against aggregates.** l.214, l.216 (0.83-0.86, 0.89, 0.82-0.83, transformer 0.59/0.41), l.218, l.236 (delta 0.02-0.04 / 0.06-0.13; reversal 0.86/0.86 and 0.48/0.49; leak 15% at V = 64), and l.263 (CLIP -0.35/0.01; GPT-2 0.78-0.84; role 0.86-0.98; identification 0.29-0.75) all match `agg_r2.txt`, `agg_e5.txt` and `agg_e4b.txt`. The within-condition correlation (0.405, CI 0.30-0.48) matches.

---

## 4. Verdict

**Major revision**, though a much smaller one than in round 1. Scores: Soundness 3/4 if N1 and N4 are fixed (2.5 now); Presentation 3/4; Contribution 2.5/4; Overall 5/10 now, 6-7/10 after the fixes below.

To reach accept:
1. **N1.** Correct the converse of Prop. `span` (orthonormal B or the B^T B metric, plus the centring term). Fix the fresh pool in `e1_calibration.py` to a fixed large size with the population denominator. Show that Fig. 3c matches the corrected closed form at V = 4-16.
2. **N2.** Reword the abstract, l.151 and l.278 so that fresh parts are the estimator of a population quantity. Present the population-vs-vocabulary dissociation as the contribution. Correct "recoverability is their consistency".
3. **N3.** Make Def. 2 graded (normalised scores and a tolerance), add an informativeness condition on e, and define e for composites.
4. **N4.** Add affine population readouts to the recursion experiment for both systems, state the HRR crosstalk formula, and redo the verdicts.
5. **N5-N7.** Fix the random-edit control, the Prop. `depth` statement and the unbinding bound (the (epsilon, delta) version), and either prove or drop the "delta bounds the leak" claims.

N8 (the pool-size curve for the pretrained encoders) is strongly recommended. Without it, the CLIP claim should be stated as an upper bound on epsilon at 165 fitting nouns.

I still see no identifiability result that goes beyond the affine case of Lachapelle et al. and Wiedemer et al. With N2 done, the paper can be honestly positioned as a definitional and methodological contribution with a clean empirical dissociation. In that framing I would not require one.
