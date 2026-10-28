# Review R1 -- Reviewer A

Paper: "Defining Compositional Representations by Recoverability" (main.tex, 523 lines; rendered pages p01-p22).
Reviewer expertise: identifiability, object-centric learning theory (additive decoders, provable compositional generalization), representation learning theory.

Line numbers below refer to `recoverability-theory/main.tex`.

---

## 1. Summary

The paper proposes to call a representation compositional when (a) a readout from a declared class U, fitted on composites of a finite vocabulary, recovers each part's code in its role from composites of *fresh* parts (epsilon-recoverability, Eq. 1), and (b) the composite state is affine in a single, role-independent part code up to a residual (delta-role-factorization, Eq. 2), with both values compared against a role-filler reference built from the same part codes (Def. 2). The appendix proves ten short results: unrestricted additivity and lookup decoding are vacuous (Lemmas 1-2), linear blends have a Gaussian MMSE ceiling of tr(A_i^T (AA^T)^+ A_i)/d (Prop. 3), linear readouts are pinned down only on the span of the training parts (Prop. 4), and an exactly role-factorized, full-column-rank relation makes role reversal and role transfer affine functions of one composite (Prop. 6). Experiments cover 14 analytic operators, small encoder/decoder networks on synthetic glyphs, a RAAM-style recursive composer against HRR, and six pretrained text encoders probed with ridge readouts.

## 2. Scores

| | Score |
|---|---|
| Soundness | 2 / 4 |
| Presentation | 3 / 4 |
| Contribution | 2 / 4 |
| Overall | 4 / 10 (borderline reject) |
| Confidence | 4 / 5 |

The writing is clear, and the paper is honest that its conditions "are not new as mathematics" (l.149). The problems are these. (i) The central definition is not a property of the relation alone: as written it depends on the experimenter's vocabulary size and fitting procedure. (ii) Under the paper's own declared affine readout class, the claim that "fresh parts carry the definition" reduces to "use enough distinct parts". (iii) Two propositions are misstated, and several empirical claims invoke propositions outside their hypotheses. (iv) The pretrained-encoder conclusion is confounded by the readout-span effect that the paper itself proves.

---

## 3. Major weaknesses

### W1. Definition 1 is not well-posed as a property of a relation: it depends on |V| and the fitting rule (l.100-111, l.117)

*Location.* Def. 1 says "some u_i in U, fitted on composites of parts in a finite vocabulary V ~ D". Neither |V| nor the fitting procedure is fixed, and the quantifier ("some u_i ... fitted on") is ambiguous.

*Why it matters.*
- Read existentially ("there exists u_i in U with low fresh-part risk"), the fitting clause does nothing. epsilon is then the population risk inf_{u in U} E||u(g(x)) - e(x_i)||^2 / Var e. For affine U this is the standard linear-probe R^2 under D. Fresh and familiar parts do not differ in that case, because for continuous D every test draw is a.s. outside V anyway.
- Read procedurally ("the u_i the procedure fits on V"), the paper's own Fig. 3c shows that the *perfect* operator, concatenation, fails the definition at V=8 (fresh R^2 = 0.05) and passes at V=32. In that reading epsilon-recoverability is a property of (relation, |V|, estimator), not of the relation.
- The networks section then takes a V=8 failure as evidence about the networks' states (l.212, the "imitation cell"). Prop. 4 says that no learner could do better there: with V=8 latents in R^8 (q=8), the centred span has dimension at most 7. The role-filler decoder fails identically (Fig. 4c), which is the giveaway.

*Fix.* Define epsilon as a population quantity, inf over U of the fresh-part risk, possibly as a limit |V| -> infinity. Then state separately a sample-complexity result (Prop. 4 made quantitative) for how many distinct parts a fitted readout needs. Every experimental verdict should be reported at a V past the span threshold, or as a curve in V with the threshold marked.

### W2. Under the declared affine class U, "fresh parts" adds little beyond "enough parts"; the familiar-part argument rests on a readout outside U (l.96, l.140, l.192, Fig. 3a,c)

*Location.* The abstract and l.252 claim that "the requirement of fresh parts is what gives the definition content". The support is Lemma 2 and Fig. 3a, both of which use a *lookup given candidates*, a readout not in any restricted U that the paper declares.

*Why it matters.* For affine readouts with a shared vocabulary across roles, familiar-part recovery already discriminates once V is moderately large.
- For g = Ax + By with m < kd, an exact affine fit on vocabulary composites needs W A = I on the role-1 difference span S and W B = 0 on the role-2 span. That is impossible once 2(V-1) > m. The paper's own Fig. 3c shows it: the dashed (familiar) curve for the half-rank projection stays at 1.0 up to V=8 and then falls toward about 0.55 at V=128, converging on the fresh value of 0.45.
- For x + y the familiar curve is low at every V (role collision).

So the familiar/fresh dissociation is a small-vocabulary phenomenon (V <= m/k + 1) for affine U. What the paper shows is that lookup and MLP readouts over-fit vocabularies, which is the well-known reason to restrict probe capacity (Hewitt and Liang 2019, which is in the bib). It does not show that fresh parts are conceptually necessary.

*Fix.* State the result for affine U: familiar-part affine recovery of a rank-r linear relation is exact iff kV - k <= r (or a similar condition), and it converges to the fresh ceiling as V grows. Either (a) reposition "fresh parts" as the condition that removes the vocabulary-size confound, or (b) give a case where affine familiar recovery stays high at large V while fresh recovery is low. A natural candidate is a nonlinear injective-on-V but non-invertible relation, e.g. a random tanh layer at m/kd = 0.5. Its fresh value is missing from Fig. 3b/3d and should be reported.

### W3. Two propositions are misstated, and Prop. 3/7 are applied outside their hypotheses (l.345-352, l.389-399, l.212, l.244)

- **Prop. 4 (span), l.347.** The claim that "every affine readout that fits the training composites exactly agrees with the true inverse on composites whose role-i code lies in S" is false as stated. The fitted W satisfies W B_j = 0 only on the role-j span S_j. A test composite with role-i code in S but a role-j code outside S_j gives an error W B_j (e_j - mean) that need not vanish. The statement must require *every* role's code to lie in its training span. The "connected design" assumption appears only in the proof's parenthetical and must be moved into the statement.
  - "Can differ arbitrarily" is also the wrong conclusion for the estimator actually used. Ridge/least squares picks the minimum-norm solution, which predicts the projection onto S. For concatenation with isotropic Gaussian parts that gives a fresh R^2 of about (dim S)/d minus a small mean-offset term: roughly 0.38-0.44 at V=8, d=16. The reported value is 0.05 (l.194, Fig. 3c). Either the experiment departs from min-norm ridge (GCV selection, standardization) or there is a bug. Please reconcile; this is checkable in closed form.
- **Prop. 7 (depth), l.391.** The statement fixes A, B in R^{d x d} (state width d) and then bounds R^2 by min(1, w/(2^n d)) for "states of width w". The kernel bound (2^n - 1)d holds only for w = d. The two parts need separate statements with consistent dimensions.
- **Prop. 3 used for nonlinear trained networks.** l.212 says "the width effect is the ceiling of Proposition 3 acting in a trained network". The hypotheses are not met: the encoder and decoder are nonlinear, the targets are pixels rather than isotropic Gaussian latents, and the reported output R^2 at m=8 < kq=16 is 0.58-0.68, *above* the would-be linear ceiling of 0.5.
  - There is a correct statement the paper could use instead. For any g and any *affine* stacked readout, the average R^2 over roles is at most m/(kd) for isotropic codes, by Eckart-Young, since the stacked prediction lies in an m-dimensional affine set. That would also cover Prop. 7 for the nonlinear composer, but it applies only to the ridge readout, not to the decoder's output R^2.
  - The remark after Prop. 2 (l.327) should also note that continuous non-injective maps (Shannon-Kotelnikov-type space-filling maps) can reach R^2 near 1 with nonlinear readouts. So "every blend into a smaller space must discard information" is true only exactly, not in the graded epsilon sense.

### W4. Prop. 6 ("systematicity is necessary") is tautological under its hypotheses and does not address the Fodor-Pylyshyn/McLaughlin necessity (l.142, l.157, l.377-387)

*Why it matters.* Prop. 6 assumes delta = 0 over the full distribution of parts *in all role assignments*. That means g is already assumed affine on the unseen role assignments it then "predicts". The content of (i)-(iv) is that a full-column-rank linear map has a left inverse, which is Smolensky's (1990) unbinding with linearly independent roles. The scientifically interesting question is whether training on a subset of assignments yields a g that is role-factorized on the rest, and the proposition is silent on it.

Under role-factorization, B_j must also be learned from parts observed in role j. A part whose code has a component outside the role-j training span gets an undetermined B_j e(x). So a span or connectivity condition re-enters, just as in Prop. 5. The contrast drawn at l.142 ("without role-factorization ... depends on the training data") compares an assumption-about-all-inputs result with a learning-from-data result. That is not a like-for-like comparison.

*Fix.* Either (a) state and prove a learning version: if g is role-factorized on the training assignments with codes spanning R^d in each role, then B (and hence the unseen assignments) is identified, with an error bound in delta and the condition number of B (the "holds approximately" claim at l.387 is currently unproved). Or (b) tone down "necessity" and the claimed answer to Fodor and McLaughlin (1990).

### W5. Novelty does not survive comparison with prior identifiability and decomposition work (l.149-155)

- **Lachapelle et al. 2023 / Wiedemer et al. 2024.** Additive decoders give identifiability up to block-wise transformations and Cartesian-product extrapolation for *nonlinear* per-slot functions. Wiedemer et al. add compositional consistency (the encoder inverts the decoder on recombined slots). The paper's conditions are the affine special case: one shared code, linear role maps, and a linear encoder-side readout. Prop. 5 is the discrete, affine shadow of Cartesian-product extrapolation on a connected support.
  - The first claimed difference, using the conditions as a definition, is a reframing.
  - The second, fresh parts, is extrapolation *outside* the training support of each slot. The paper proves nothing about it beyond the trivial affine case and the span limitation. This is exactly where Lachapelle et al. need their assumptions, and where a novel identifiability result could have been.
- **McCoy et al. 2019 (TPDN).** delta is essentially the TPDN approximation error: learned filler embeddings times linear role maps (f ⊗ r = (I ⊗ r) f = B_r f). epsilon is a linear probe evaluated on held-out fillers. That should be said explicitly.
- **Uselis et al. 2026.** They argue linear *and orthogonal* factors are necessary. Def. 1 needs only full column rank of [B_1 ... B_k]. The paper should reconcile the two: is orthogonality needed for learnability of the readout from few parts, or for robustness? The sentence "close to both necessary and sufficient" (l.151) is unsupported.
- **Schug et al. 2024 / Lippl and Stachenfeld 2025.** Schug et al. give identification of modules with a sample-complexity analysis under connected/compositional support. Lippl and Stachenfeld give a kernel theory, including when *data statistics* produce or prevent conjunction-wise additivity, plus the memorization leak. Props. 5 and 8 are much weaker restatements. The claim at l.411 that "the residual delta bounds" the leak is unproved.
- **Missing:** Hupkes et al. 2020 (in the bib, never cited; substitutivity, productivity and localism are directly relevant), Schott et al. 2022 (extrapolation to unseen factor values), Andreas 2019 TRE (cited, but TRE with held-out primitives is close to the fresh-part test), and Hewitt and Liang 2019 control tasks (in the bib, uncited; the argument for restricting U is theirs).

*Fix.* Add a precise comparison table (assumption / object / conclusion / extrapolation regime) against these five papers. Either prove one genuinely new result, for example an identifiability or sample-complexity statement for *fresh-part* recovery under nonlinear additive models, or recast the paper explicitly as a position or methodology paper.

### W6. The pretrained-encoder conclusion is confounded by readout extrapolation, and the reference is not matched (l.235-239, Fig. 5c,d)

- The noun code is "a x" projected onto 32 PCs fitted on training nouns. So the fresh targets are forced into the training-noun subspace, and the concatenation reference (64-d, spanned by 170 nouns) passes by construction.
- The model states are 512-768-d mean-pooled vectors fitted from about 170 nouns per role. By the paper's own Prop. 4, the readout is undetermined on state directions not spanned by training composites. Fresh nouns excite those directions, so the extrapolation error is unbounded, which explains R^2 = -1.21 and -0.92 for CLIP.
- Fig. 5d contradicts the reading that CLIP has lost the nouns. CLIP-B/L identify the fresh noun among 80 at 38-44% (chance 1.25%), comparable to the sum reference (45%), and they assign roles at 84-89% against the reference's 50%. The R^2 ranking (CLIP worst) and the identification ranking disagree.
- "Fresh" nouns are not fresh to the *system*: every noun was seen in pretraining. That violates the definition's own requirement (l.86, "parts the readout and the system have never seen").
- 170 + 80 = 250, not 305; what happened to the other 55 nouns?
- "Best layer" appears to be selected on the reported fresh-noun metric. That is selection on test.

*Fix.*
1. Reduce composite states with PCA fitted on training composites, or use ridge with a penalty chosen by noun-held-out CV, and report R^2 as a function of the number of training nouns.
2. Build the references in the model's own state dimensionality: embed concat/sum of the unreduced noun states with a random rotation into R^768.
3. Report identification and role accuracy alongside R^2 in the main text.
4. Select layers on a validation noun split.
5. Rename "fresh" to "readout-held-out" for pretrained models, or use genuinely novel tokens (nonce words with defined embeddings).

### W7. The network experiments do not apply Definition 2, and the key correlation is close to circular (l.206-214, Fig. 4d)

- No trained network is ever scored by Def. 2. There is no declared e and M, no delta, and no comparison to a role-filler reference on the same codes. The "1-4% interaction" (l.214) is an ANOVA on a 10x10 grid whose quantity (which coordinates of h? which variance decomposition?) is unspecified.
- The r = 0.94 between ridge fresh recovery from h and decoder fresh output R^2 correlates two readouts of the same state trained on the same parts, pooled across conditions that vary V and m over a wide range. The correlation is largely driven by the design factors. Report the partial correlation controlling for V and m, or within-condition correlations across seeds.
- Placing the V=8 networks in the "imitation" cell of Table 1 is inconsistent with Table 1's own column header, "Behavior succeeds on new combinations **and new parts**". These networks fail on new parts behaviourally (output R^2 at most 0.30).
- The role-filler decoder's advantage on new roles is small (0.32-0.33 against 0.11-0.27 at V=8) and disappears by V=16. That is weak support for "as role-factorization predicts".

### W8. Definition 2's "graded" calibration collapses to an exact criterion (l.117, l.120)

The required reference is a role-filler operator on the same codes. For concatenation the reference has delta = 0 identically and epsilon of about 0 once V > d (0.97-1.00 in Sec. 4.4). "No worse than the reference" therefore means delta = 0 and epsilon close to 0, so every real system fails, GPT-2 at 0.79 included. The definition is binary in practice, not graded. Also unspecified: which role-filler operator (concatenation? TPR with which roles?), and what the part code of a composite is in the productivity clause ("the same readouts applied recursively").

*Fix.* Report (epsilon, delta) as coordinates placed between the blend and role-filler references, e.g. a normalized score (epsilon_sum - epsilon)/(epsilon_sum - epsilon_concat), and define e for composites explicitly, e.g. e(composite) := g.

### W9. HRR baseline in the recursion experiment looks inconsistent with standard capacity analysis (l.228-230, Fig. 5b)

HRR identification is essentially identical at w=256 and w=1024 (Fig. 5b). With unitary roles, crosstalk after unbinding has energy of about (n-1)||x||^2 spread over w dimensions. Projecting back through the pseudo-inverse of an 8-d lift keeps a fraction of about 8/w, so identification among V=16 should *improve* with w and be near-perfect at depth 3 with w=256. A width-independent 49% at depth 3 suggests a normalization or lift bug, or an undocumented noise source. Please verify. Also, the learned composer is scored by R^2 and HRR only by clean-up identification; report both measures for both systems.

---

## 4. Minor issues

- **Anonymity (l.37, l.262, l.485).** The author is named, and two self-citations (singaravadivelan2026; anonymous2026seen with an OpenReview link) are de-anonymizing for a double-blind venue.
- **l.46 (abstract).** "Its ceiling is set exactly by the rank of the relation": only the *average* over roles is rank/(kd). The per-role ceiling is the trace formula.
- **l.57 vs rendered p.6.** The key elmoznino2024 renders as "Elmoznino et al. (2025)". Check the year.
- **l.94.** "A readout class U, such as affine maps". Lemma 2's lookup is not in any such class, so state that Fig. 3a deliberately steps outside U.
- **l.114 and l.149.** Terminology drifts between "role-factorized" (Def. 1), "separable" (Lemma 1, Prop. 5) and "additive". l.149 says "our separability is their additivity", but Def. 1 never defines separability.
- **l.123.** The notebox cites Prop. 7 for "a state of fixed dimension cannot invertibly encode trees of unbounded depth over continuous parts". Prop. 7 is linear only; the continuous claim needs the remark after Prop. 2.
- **l.136.** "Appendix A proves eight results": it contains 2 lemmas and 8 propositions.
- **l.188, l.494.** "Two-layer MLP" vs "two hidden layers of 256 units". Be precise.
- **l.192.** "Behave exactly like a half-rank projection": 0.42/0.45 against a ceiling of 0.5, and projections sit 0.05-0.06 under their ceilings too (Fig. 3b). Compute the per-instance ceiling for each sampled A (and for HRR, the role-1 value per sampled role pair) instead of the expectation, then say "within x".
- **l.199 caption (a).** "V = 32", but the text says lookup is over "a candidate set". Say what the candidates are.
- **l.208.** q is never defined (the latent dimension, 8). Elsewhere the part dimension is d.
- **l.208.** Hadley's strong systematicity is named but hadley1994 is not cited there.
- **l.212.** "With the state as wide as m = 64" reads oddly; say "at m = 64".
- **l.214.** "Interaction 1-4% in the states": define the ANOVA quantity (per coordinate of h? summed variance?).
- **l.226.** "Gaussian parts in R^8" while Prop. 7 uses d. Say d = 8.
- **l.235.** 305 nouns vs 170 + 80.
- **l.237.** Mean pooling of "a x on a y" is nearly a symmetric sum by construction for some models. The pooling choice drives the result and should be ablated (e.g. token-specific states at the noun positions).
- **l.239.** "Every model looks compositional on the training nouns": is that measured on held-out *pairs* of training nouns or on the fitting sentences? Fig. 5c says "training nouns".
- **l.244 caption.** "Values below -1 are clipped": CLIP-B is -1.21, so state it in the text.
- **l.327.** "Discontinuous injections ... are not stable under any noise" is unsupported; give a reference or drop it.
- **l.343.** The HRR ceiling "averages 1/2" over draws of the roles. For a fixed role pair, role 1's ceiling is the frequency average of |r1_f|^2/(|r1_f|^2+|r2_f|^2).
- **l.351.** "The span of V centered points has dimension at most V - 1" is fine, but the statement says "at least d + 1 distinct parts in role i", which also requires general position.
- **l.387.** "Holds approximately, with error controlled by delta and the conditioning of B": give the bound. It is a two-line perturbation argument.
- **l.411.** "The residual delta bounds it": unproved.
- **l.459.** Table 4 lists a "(conjecture)" in a column titled "Guarantee required".
- **l.465.** \label{app:human} sits on no sectioning command, so the reference resolves to Appendix C via the enclosing section. That works, but it is fragile.
- **l.491.** "Code is provided with the paper": no code was available to this reviewer.
- **Table 1 (l.159-171).** The column "Behavior fails" conflates failure on new combinations with failure on new parts. The networks in Sec. 4.2 succeed on the first and fail on the second.
- **Fig. 3d.** The HRR MLP bar is missing (zero or negative?). Show it.

---

## 5. What would move my score up by 2 points (4 -> 6)

1. **Repair the definition (W1, W8).** State epsilon as a population quantity over D for a declared U, independent of |V|. Give a separate, proved sample-complexity statement (the corrected Prop. 4 with all-role spans and a quantitative min-norm prediction that matches Fig. 3c). Report a normalized graded score between the blend and role-filler references. Define e for composites.
2. **Fix the misstatements (W3).** Correct Props. 4 and 7. Replace the uses of Prop. 3 for nonlinear networks with the Eckart-Young bound for affine readouts of arbitrary g. Remove the claim that the width effect in trained networks "is" Prop. 3.
3. **One genuinely new theorem (W4, W5).** Either a learning version of Prop. 6 (training assignments plus spanning codes identify B and hence all unseen assignments, with an error bound in delta), or a fresh-part extrapolation result for nonlinear additive models that goes beyond Lachapelle et al. / Wiedemer et al. Include a precise comparison table against those papers, Uselis, Schug, Lippl and Stachenfeld, and McCoy (TPDN).
4. **Redo the pretrained analysis (W6)** with dimension-matched references, state-space PCA or CV-tuned ridge, a curve over the number of training nouns, validation-selected layers, and identification/role accuracy in the main text. If CLIP's negative R^2 survives these controls, the headline claim about text-to-image conditioning becomes credible.
5. **Apply Def. 2 to the trained networks (W7):** declared e, delta, epsilon at V past the span threshold, partial correlations controlling for V and m, and a check of the HRR width-independence (W9).
