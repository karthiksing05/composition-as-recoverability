# Review R4, Reviewer A (theory and philosophy of compositionality)

Paper: "Defining Compositional Representations by Recoverability". `recoverability-theory/main.tex` has 596 lines (mtime 2026-10-28 05:57), and all line numbers below refer to it.

**What I did.**
- Read the whole file and checked every proof in Appendix A by hand.
- Checked the numbers that the definitions' verdicts rest on against `experiments/results/agg_r4_e4w.txt` (E4, whitened), `agg_r4_e2e5.txt` and `agg_e2h_partial.txt` (E2/E5).
- Read the model identifiers in `experiments/e4_pretrained.py:64-72`.
- Ran no code and did not compile.

Points from the r3 reviews that are now resolved are listed in Section 4 and not repeated.

## Verdict: major revision, though small in effort

The mathematics in Appendix A is correct (verification log, Section 3). The main text now applies Definition 2 explicitly: κ, δ and τ verdicts in Sec. 4.2, a whitened κ in Sec. 4.5, and an honest scope note in Sec. 4.4. That makes it possible, for the first time, to test the verdicts against the definition. Six problems follow.

**Claims that do not follow from the results:**
- Two abstract claims do not follow from the results (M1).
- The verdicts of Secs. 4.2 and 4.5 are not computed in the way Definition 2 prescribes (M2-M4).

**Definitional issues** (both verifiable in the paper's own numbers):
- The residual δ cannot separate a perfect role-filler operator from a non-compositional one when the part code is a reduced description. The paper's own concatenation reference has δ up to 0.81 (M2).
- κ is not invariant to invertible linear changes of the part code. Whitening alone moved GPT-2 from κ ≈ 0.82 (r3) to 0.44-0.50, so the metric must be part of what is declared (M3).

**Positioning:** van Gelder's functional compositionality is closer to this definition than the paper says. Under affine readouts, Definition 2 excludes classical concatenative strings whose constituents vary in length (M6).

Most fixes are text changes. M2, M5 and possibly M4 need one cheap re-analysis each, on states that are already cached.

---

## 1. MAJOR

### M1. Two abstract claims do not follow from the results, and one theorem is stated without its hypothesis (l.46, l.61, l.267, l.282)

**(a) The recursive composer (l.46).** The abstract says: "In networks trained on glyph composites, and in a learned recursive composer, the states keep fresh parts recoverable, but the systems' own decoders read them only after training on many distinct parts."
- For the recursive composer this is false. Sec. 4.4 (l.254) reports that "Within the trained depths the learned composer's own decomposer recovers fresh leaves almost perfectly whenever $w \ge 32$ ($R^2 \ge 0.95$)" with $V = 16$.
- Its failure is at the first untrained *depth*, not a shortage of distinct parts.
- Replacement: "In networks trained on glyph composites, and in a learned recursive composer, the states keep fresh parts recoverable, but the systems' own decoders fail where training did not reach: the glyph networks read fresh glyphs only after training on many distinct glyphs, and the composer's decomposer fails one level beyond its training depths."

**(b) The CLIP encoders (l.46, l.267).** "the CLIP encoders that condition text-to-image models gain little from training beyond the length of each noun" is wrong in both directions.
- `e4_pretrained.py:70-72` loads `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`. That is the OpenCLIP ViT-H/14 text encoder family used by Stable Diffusion 2.x, so it does condition a text-to-image model.
- It gains about 0.30 in κ from training (0.355 vs 0.051, `agg_r4_e4w.txt`), and it keeps κ = 0.36 after the token count is removed (l.267).
- CLIP ViT-B/32, which falls below its random-weight control, is not described anywhere in the paper as conditioning a text-to-image model.
- Sec. 4.5 itself says "the two smaller CLIP encoders" (l.267, caveat 3), and the Discussion (l.282) says "the smaller CLIP encoders". Only the abstract and the last sentence of l.267 use the wrong description.
- Replacement for both places: "...and the two smaller CLIP encoders, including the one that conditions Stable Diffusion 1.5, gain little from training beyond the length of each noun."

**(c) Reversals need full column rank (l.46, l.61).** The abstract says "a shared code fixes every role reversal from a single composite", and l.61 says "under a role-factorized relation every role reversal ... is fixed by a single composite".
- Proposition `systematic` needs δ = 0 **and** full column rank of $[B_1\cdots B_k]$.
- $x + 0.5y$ is role-factorized with a shared code (Table 5 says "Yes"), but its reversal $y + 0.5x$ is not determined by $x + 0.5y$: $[I,\ 0.5I]$ has rank $d < 2d$.
- l.144 states the hypothesis correctly. The abstract and the Introduction drop it.
- Replacement (abstract): "...and when parts are recoverable and enter every role through one shared code, a single composite fixes every role reversal."
- Replacement (l.61): "...and show that under a role-factorized relation whose role maps are jointly injective, every role reversal ... is fixed by a single composite."

### M2. The residual δ penalizes a perfect role-filler operator whenever the part code is a reduced description (Eq. 2, l.108; l.263; l.572)

**The problem.** Eq. (2) normalizes $\mathbb{E}\lVert\eta\rVert^2$ by the *total* state variance, and it requires the state to be affine in the declared code $e$. When $e$ is a low-dimensional summary of what the system has about a part (16 PCs of a 768-1280-dim phrase state), the state legitimately carries part information that is not an affine function of $e$. All of that information lands in η.

**The paper's own numbers show it.**
- In `agg_r4_e4w.txt` (column `delta_fresh`, rows `ref_concat`), the concatenation reference has δ from 0.044 (GPT-2-medium) to 0.810 (CLIP-H). The concatenation reference is the paper's role-filler operator.
- l.572 acknowledges this ("the concatenation reference itself has δ between 0.04 and 0.81 ... so the verdicts rest on κ").
- But Definition 2's verdict requires δ ≤ τ. As defined, concatenation is therefore not compositional at τ < 0.81 for CLIP-H.
- So δ conflates two things: non-additivity (the classical failure, prototype overrides, l.114) and incompleteness of the declared code (not a failure of composition at all).

**The same issue affects Sec. 4.2/4.3.** There, δ = 0.02-0.04 only because the 16 glyph PCs carry 98% of pixel variance.

**This is also an inconsistency with Definition 2.** Definition 2 says the references are "a role–filler operator and a symmetric blend applied to the same part codes". In E4 the reference concatenates the full states, not the declared 16-PC code. If it were applied to $e$ itself, its δ would be 0 by construction.

**Fix (definition).** Allow the factorization to be stated in a richer code from which $e$ is read. Replacement for the role-factorization sentence of Def. 1:

> The relation is *δ-role-factorized* if, for some code $e'$ from which $e$ is an affine function (possibly $e' = e$), there exist $c$ and matrices $B_1, \ldots, B_k$, one per role and shared by all parts, such that $g_r = c + \sum_i B_i e'(x_i) + \eta$ with $\mathbb{E}\lVert\eta\rVert^2 \le \delta\,\mathbb{E}\lVert g_r - \mathbb{E} g_r\rVert^2$.

With $e'$ = the full phrase state (or a high-dimensional PCA of it), the concatenation reference has δ = 0, as it should.

**Alternative fix.** Report the interaction share on a complete grid of fresh parts, which the paper already computes. It is "Interaction is the share of variance left by a two-way additive fit on a 10×10 grid" at l.564, and `inter` in `agg_r4_e4w.txt`, where it is 0.000 for every reference and 0.002-0.193 for the models. Define δ through it, with a separate term for the role-sharing part.

**Fix (text).** At l.263, replace "whatever its residual δ" with the δ (or interaction) values. Note at l.572 that, under the corrected definition, δ for the reference is zero.

### M3. κ depends on the metric of the part code, the definition is silent on it, and the paper compares κ across two metrics (Eq. 1, l.117, l.261, l.263, l.286)

**The math.**
- Eq. (1) is invariant under $e \mapsto sQe + b$ ($Q$ orthogonal, $s > 0$). It is not invariant under a general invertible linear map $T$.
- The best affine readout of $Te$ is $T$ times the best readout of $e$ only when errors are weighted in the $T^{\top}T$ metric. Whitening is such a $T$.
- So ε, and with it κ, is a property of the pair (code, metric), not of the code.

**The evidence is in the paper's own history.**
- The r3 pipeline used the same three part codes, layer and penalty chosen on validation nouns, and PCA on the fitting nouns, but unwhitened components. It gave GPT-2 κ = 0.81-0.83 (`agg_e4c.txt`, verified in review_r3_A §1).
- The present whitened pipeline gives κ = 0.44-0.50 (l.263; `agg_r4_e4w.txt`: 0.443, 0.470, 0.502).
- Under Def. 2's tolerance rule the first gives "compositional at τ ≈ 0.2" and the second "not compositional at any τ < 1/2".
- Definition 2 quantifies existentially over codes ("for some part code $e$ found under $\M$"). Unless the metric is declared as part of $\M$, both codes are admissible and the existential picks the generous one.

**Inconsistency 1 (l.263).** "the networks of Section 4.2 reached κ = 0.73–0.83" compares E2's κ, computed on *unwhitened pixel* codes, with E4's κ on *whitened* 16-PC codes.
- Glyph pixels are also concentrated in few directions: 16 PCs carry 98% of the variance (l.234), and the generator has an 8-dim latent.
- So E2's κ is computed under exactly the convention that l.261 rejects for E4 ("Without whitening, $R^2$ would mostly score that one direction").

**Inconsistency 2 (l.286).** The Terminology paragraph asks reports to "state how the part codes are scaled". Sec. 4.2 does not.

**Fix.**
1. Add to Def. 1, after Eq. (1): "The error is measured in a declared inner product on part codes, which is part of the declaration of $e$; ε is invariant to rotations and rescalings of $e$ but not to other linear changes of coordinates."
2. In Sec. 4.5, state that the verdict is relative to the whitened metric, and give the unwhitened κ in one clause (or in Appendix D) so the reader sees the sensitivity.
3. Either recompute E2's κ on the whitened 16-PC code that Sec. 4.3 already defines, or delete the cross-section comparison at l.263.

### M4. The Sec. 4.2 verdict combines quantities from different codes and conditions, and its informativeness clause is not checked (l.216, l.117, l.120)

The text at l.216 reads: "Their role-factorization residual, measured in Section 4.3, is δ = 0.02–0.04, so by Definition 2 these states are compositional at a tolerance of about τ = 0.27 at V = 8 and 0.17 at V = 64."

Definition 2 requires one code $e$, informative at τ, for which both κ and δ hold. Three things fail:
1. **No δ at V = 8.** Sec. 4.3 retrained only $V \in \{16, 64\}$ and $m \in \{32, 64\}$ (l.232). No δ exists for V = 8, so the V = 8 verdict is not supported.
2. **Two different codes.** κ uses the 64-pixel glyph code (l.212), while δ uses the 16-PC code (l.234). This one is repairable by an argument the paper does not give: the PCs are a linear function of the pixels, so in population the best affine fit on the pixel code has residual no larger than on the PC code. The PC-code δ is therefore an upper bound on the pixel-code δ.
3. **Informativeness never reported.** l.120 says "we report this share wherever a system-independent description exists (the glyph latents of Section 4.2)", but no R² of the latents from either code appears anywhere. The generator is a random tanh layer, a smoothed linear map and a sigmoid (l.563), so an affine map of pixels need not recover the latents at R² ≥ 0.73.

**Fix.**
- Declare $z$ = the rendered glyph pixels, which are system-independent. Informativeness of the pixel code is then 1 and that of the 16-PC code is 0.98. In l.120, change "(the glyph latents of Section 4.2)" to "(the rendered glyphs of Sections 4.2 and 4.3)".
- Replacement for the l.216 sentence:

> Their role-factorization residual, measured in Section 4.3 on the 16-component code at $V \in \{16, 64\}$, is $\delta = 0.02$–$0.04$, which bounds the residual of the pixel code because the components are linear in the pixels. By Definition 2 the states at $V = 64$ are representationally compositional at a tolerance of about $\tau = 0.17$ and not at $\tau = 0.1$; at $V = 8$, where δ was not measured, κ alone places the tolerance at no less than 0.27.

- Also write "representationally compositional" here. By the Terminology paragraph (l.286) the bare term is reserved for systems whose own procedures use the parts, and these networks' decoders do not (see m10).

### M5. The pretrained reference is exact by construction, and by the paper's own Prop. `span` a perfectly role-factorized state with non-orthogonal role maps would score below it (l.263, l.267, abstract "none comes near a role–filler reference")

This is r3 A S2. It is still open, and I raise it to major because the theory in the paper now predicts the bias.

**Why the reference is exact.**
- At the full pool, the 16 PCs and the readout are fitted on the same 165 nouns. Each target is a fixed linear map of the part state whose rows lie in the span of the fitting nouns' centred states.
- For the concatenation reference ($B = I$), the minimum-norm readout is therefore exact on fresh nouns: `ref_concat` rec_fresh is 1.000 for all 11 models and all random controls in `agg_r4_e4w.txt`.

**Why a model with the same structure scores lower.**
- Prop. `span` (l.393, last sentence) states that for general $B$ the fitted procedure returns the projection onto $T$ in the metric $B^{\top}B$, "which mixes roles".
- Consider a state that is *exactly* role-factorized in the full phrase code, with full-rank but non-orthogonal $B_i$. It loses, on fresh nouns, the cross-terms from components of the fresh code outside the span of the 164 fitting directions. Here $D = 768$-$1280 \gg 164$.
- The concatenation reference loses nothing.

So κ < 1 for such a state reflects the estimate, not the relation. Caveat 4 (l.267) concedes that the values are lower bounds, but the reference does not share the bias, so κ is biased downward for every model.

**Fix.**
- Add a role-filler reference that faces the same estimation problem, for example $h = A_1 s(x) + A_2 s(y)$ with random Gaussian $A_i \in \R^{D\times D}$ applied to the full phrase states (optionally followed by a random tanh). Report κ against it.
- Or state the comparison as "relative to an orthogonal role–filler reference whose recovery is exact by construction" in l.263, l.267 and the abstract.
- This is one extra pass over cached states.

### M6. Positioning: van Gelder is closer than stated, and under affine readouts the definition excludes the classical concatenative paradigm (l.59, l.114, l.123, l.159)

**(a) van Gelder.** l.59 says the proposal "also bears on van Gelder's distinction".
- van Gelder (1990) *defines* functional compositionality as the existence of "general, effective and reliable processes for (a) producing an expression given its constituents, and (b) decomposing the expression back into those constituents".
- Recoverability is an operationalization of clause (b):
  - "general" corresponds to transfer across $\D$, including parts fresh to the procedure;
  - "effective" corresponds to a declared, restricted $\U$;
  - "reliable" corresponds to the graded ε.
- Role-factorization has no counterpart in van Gelder. It is what the paper adds, together with the fresh-part estimator.
- Replacement for the second sentence of l.59:

> It is closest to \citet{vangelder1990}'s *functional* compositionality, which requires general, effective and reliable procedures for producing a complex from its constituents and for decomposing it back into them; our recoverability makes "general" and "effective" precise as transfer to parts fresh to a procedure of a declared class, and role-factorization adds the requirement, absent from van Gelder, that each part contribute one code to every role. \citet{chalmers1990} showed that structure-sensitive operations can act on such representations directly: a representation can be functionally compositional without being concatenative, with parts that are tokened.

**(b) Concatenative strings.** l.114 calls role-factorization "the classical requirement that a constituent contribute the same thing to every complex in which it occurs", and l.159 says the criterion "takes Smolensky's side". Under the declared affine $\U$, though, Definition 2 does not certify the paradigm classical representation, a symbol string whose constituents vary in length.
- In a positional code of "a $x$ on a $y$", the tokens of $y$ start at an offset that depends on the length of $x$.
- So the role-2 contribution is a shift of $e(y)$ that depends on $x$. That is not $B_2 e(y)$ for a fixed $B_2$: δ > 0, and an affine readout of $y$ fails.
- This is the same mechanism as the token-count confound the paper itself diagnoses in Sec. 4.5 (caveats 2-3, l.267).
- Under affine readouts, Definition 2 therefore recognizes role-indexed codes (fixed slots, tensor products) and treats concatenation with variable-length constituents as a blend. That is the opposite of what Fodor and Pylyshyn would say, and the paper does not mention it.
- Fix: add to the note box (l.123) or to Sec. 3.4:

> Under affine readouts the definition recognizes role-indexed codes, such as fixed slots and tensor products. A concatenative string whose constituents vary in length places each constituent at an offset that depends on the others, so it meets the definition only under a declared readout class that includes alignment, such as a parser; the class must then be declared and calibrated like any other.

**(c) Performance limits (l.123).** The note box reads: "productivity is reported as a curve over depth, and its limits, as \citet{fodor1988} argued, are matters of performance."
- Fodor and Pylyshyn treat human depth limits as performance limits *on an architecture whose competence is unbounded*: a classical system's representations grow with the expression.
- The limit shown by the Remark after Prop. `fiber` is a limit of the representational *format* (a fixed-width continuous state). On their view that is a limit of competence, and such a system would lack productivity in the strict sense.
- Citing them for the opposite conclusion misreads them.
- Replacement:

> ...so productivity is reported as a curve over depth. \citet{fodor1988} treated the depth limits of people as limits of performance on an unbounded competence; a state of fixed width imposes its limit in the representational format itself, so on their view such a system is at most approximately productive, and the depth curve measures how closely.

---

## 2. MINOR

**m1. "Vacuous" overstates (l.61, l.120, Lemma title l.315).**
- Lemma `psi` shows that unrestricted separability *reduces to injectivity*. It is not vacuous: $x + y$ still fails it on continuous $\D$, and l.96 says this correctly ("adds nothing to invertibility").
- l.120, "with unrestricted readouts ... any injective system passes", is asserted without proof. For measurable readouts it needs the Lusin-Souslin theorem: a Borel injection between Polish spaces has a Borel inverse on its image (Kechris 1995, Thm 15.1).
- Replacement (l.61): "We prove that unrestricted versions of both properties reduce to invertibility, and that on a finite vocabulary even invertibility is checked by lookup."
- Add one sentence after Lemma `lookup` with the Lusin-Souslin citation.

**m2. The symmetric-blend normalization is specific to the sum and to k = 2 (l.117, l.120).**
- Definition 2 says "a symmetric blend". l.120 says that for it "the best affine readout returns the mean of the two parts, so $\varepsilon_{\mathrm{blend}} = 1/2$ ... whatever the code". That holds for the sum. For $x \odot y$ with zero-mean parts, the best affine readout is constant and ε = 1. For the sum of $k$ parts, ε = 1 − 1/k.
- One more fact is worth stating: for *any* symmetric $g$ and *any* readout, $\mathbb{E}[x\mid g] = \mathbb{E}[y\mid g]$ gives ε ≥ 1/2. So the sum is the best symmetric blend.
- Replacement in Def. 2: "...from concatenation and from the sum $e(x_1) + \cdots + e(x_k)$ applied to the same part codes".

**m3. State the population form of κ (l.120).** With affine $\U$, i.i.d. parts and concatenation, ε_ref = 0 and ε_blend = 1/2. So in population κ = 1 − 2ε, and "compositional at τ" means ε ≤ τ/2 and δ ≤ τ.
- One sentence makes clear that the references only correct estimation bias.
- It also shows that the tolerance on recovery is twice as strict as the tolerance on δ. Say whether that asymmetry is intended.

**m4. Wording of the informativeness clause (l.117).**
- "since almost any continuous code separates its parts" should be "since any injective code separates its parts". A constant code is continuous and separates nothing.
- The verdict sentence should name the tolerance: "...compositional at tolerance τ if, for some $e$ found under $\M$ that is informative at τ, κ ≥ 1 − τ and δ ≤ τ."
- r3 B13 is still open: the box defines four notions. Consider moving informativeness and productivity into a third box.

**m5. Table 1, lower-left cell, contradicts its column header (l.167, l.170).** The column is "Behavior succeeds on new combinations **and new parts**", but the cell reads "Vocabulary-bound generalization" and cites Chaabouni et al., who test new combinations of familiar values.
- A system whose behavior succeeds on fresh parts while an affine readout cannot recover them is "recoverable only by the system's own nonlinear procedure", not vocabulary-bound.
- Fix: change the header to "Behavior succeeds on new combinations of familiar parts". Alternatively, relabel the cell "Generalization without affinely recoverable parts".

**m6. HRR is understated (l.194).**
- "Vector-symbolic superposition of dense parts is therefore not invertible in the sense of Definition 1 under affine readouts" has two problems. Definition 1 defines *recoverable*, not invertible. And HRR/MAP with fixed roles in $d$ dimensions is linear of rank ≤ $d$, so by Prop. `ceiling` *no* readout exceeds R² = 1/2 for Gaussian parts.
- Replacement: "...is therefore not recoverable in the sense of Definition 1: for Gaussian parts no readout of any kind exceeds the ceiling of 1/2 (Proposition `ceiling`); it is decodable over a vocabulary."

**m7. Prop. `depth` versus the HRR experiment (l.452, l.252, l.254).**
- Prop. `depth` assumes one $g(x,y) = Ax + By$ at every node, while the HRR system uses one pair of role vectors *per level* (l.252). The proof uses only that the depth-n map is linear, so state the proposition for "a composition that is linear at every node, possibly with level-specific maps".
- At l.254, "as Proposition `ceiling` requires for a linear relation of full rank" should be "allows". The ceiling is an upper bound and does not require that it be reached.

**m8. Inconsistent number and term (l.282 vs l.254).**
- l.254 says the composer's population readout stays "within 0.14 of the ceiling"; l.282 says "within 0.13 of the rank ceiling". Use 0.14.
- For a nonlinear composer the bound is the Eckart-Young ceiling for affine readouts (Prop. `ey`), not a rank ceiling. Write "within 0.14 of the width ceiling".

**m9. "At any tolerance below one half" (l.263).**
- GPT-2-large has κ = 0.502 and δ_fresh = 0.208 (`agg_r4_e4w.txt`). Read literally, Definition 2 makes it compositional at τ = 0.499 < 1/2.
- Write "With κ ≤ 0.50, no encoder is compositional under Definition 2 at a tolerance below about one half".
- Also say "under the declared part codes and metric" (see M3).

**m10. Two notions of "compositional" need names (l.216, l.284, l.286, Discussion l.280).**
- Definition 2 is representational. The Terminology paragraph reserves "the bare term" for systems whose parts are recoverable and "whose own procedures use them". The five requirements end with "has earned the term in our sense", although requirements 2, 4 and 5 are not part of Definition 2.
- Say explicitly: "representationally compositional (Definition 2)" versus "compositional system (Definition 2 plus requirements 2, 4 and 5)".
- Use the first at l.216 ("these states are representationally compositional ...").

**m11. Requirement 1 still lacks the qualifier from r3 B3 (l.284).** "a state narrower than the parts it combines loses them" is false for glyphs: the state of width 64 recovers two 64-pixel glyphs at 0.89, because glyphs occupy far fewer dimensions.
- Write "a state narrower than the dimensions its parts occupy loses them".
- Make the same change at l.114 ("every blend into a space smaller than its parts").

**m12. "Behavioral quantity" (l.448; r3 A7, still open).** Prop. `approx` bounds the error of a *predicted state*. Replacement: "This is the one place where the two quantities of Definition 1 jointly bound the error of a predicted state".

**m13. Lippl and Stachenfeld (l.153 vs l.472).** l.153 calls the memorization leak "a behavioral counterpart of the residual η", while l.472 says it is "a component of the projection of the interaction onto additive functions under the observed design".
- Replacement at l.153: "...a behavioral counterpart of the part of η that a least-squares additive fit absorbs under an incomplete design (Remark after Proposition `context`)".
- The bound at l.472 ("at most a design-dependent constant times the interaction's norm") is still unproved (r3 A S7). Either add the three-line argument or write "and otherwise depends on the design".

**m14. Elmoznino et al. (l.157).** "a fixed-width superposition over a small vocabulary compresses well" is asserted, not shown (r3 B, §C). Write "would plausibly compress well".

**m15. Uselis et al. (l.57, l.153).** Their title concerns *vision* embedding models (r3 B, §C). Write "in vision embedding models".

**m16. Prop. `systematic` (iv) wording (l.425).** "leaves every readout $P_j$, $j \ne i$, unchanged" should read "leaves every readout value $P_j(h - c)$, $j \ne i$, unchanged". $P_j$ is a fixed matrix.

**m17. Lemma `psi` (l.316).** "for some invertible ψ": ψ is defined only on the image of $g$ (l.320). Write "for some ψ injective on the image of $g$". If $m > kd$, no ψ on all of $\R^m$ is both invertible and equal to $g^{-1}$ on the image.

**m18. Random-weight identification and role assignment are not reported (l.263; r3 C B2, still open).** The sentence "identifies them among 100 candidates far above chance (0.65–0.94) and above its own sum reference (0.41–0.44)" is offered as positive evidence, but untrained models do as well or better:
- CLIP-B/32 with random weights identifies at 0.744 and assigns roles at 0.899, against 0.709 and 0.895 trained (`agg_r4_e4w.txt`).
- Add the random-weight values or drop the sentence.

**m19. Leftovers from r3.**
- `\label{app:human}` (l.529) still sits outside a sectioning command.
- l.561, "standardization changes the metric in which the fit projects, as Proposition `span` describes": for standardized concatenation, $B^{\top}B = D^{-2}$ is block-diagonal. Write "changes the metric of the projection within each role (Proposition `span` with $B = D^{-1}$)".
- The crosstalk formula $1 - (2^n-1)d/w$ (l.254, Fig. 6b) still has no one-line derivation (r3 A S6).
- l.261: "in the spirit of control tasks \citep{hewitt2019}" — a random-weight model is an untrained-model baseline, not a control task (r3 C S7).

**m20. Optional, philosophical.** l.114 could note that Fodor and Pylyshyn (1988, p. 42) require only "*approximately* the same semantic contribution". That licenses the graded δ and helps the pet-fish paragraph (l.501). Context-dependent modifiers (*red* hair vs *red* apple; Kamp & Partee 1995's recalibration) are the textbook non-role-factorized case and deserve one clause at l.114 (r3 B, still open).

---

## 3. Verification log (Appendix A)

| Result | Status |
|---|---|
| Lemma `psi` | Correct, with the domain caveat in m17. |
| Lemma `lookup`, Prop. `blend` | Correct (trivial). |
| Prop. `fiber` and Remark | Correct. Rank-nullity; for $m < kd$ no continuous injection exists (Borsuk-Ulam / invariance of domain). |
| Prop. `ceiling`, covariance form | Correct. $\mathbb{E}[x\mid z] = A^{\top}(AA^{\top})^+z$; the block traces of the projection give $R^2_i$. |
| Prop. `ey` | Correct. A fixed affine subspace of dimension ≤ m; Eckart-Young. It bounds the variance-weighted mean over roles, which is what l.140 and l.284 claim. |
| Prop. `span` and closed form | Correct, including the range statement now added for $V > d+1$. The $B^{\top}B$ clause is what M5 relies on. |
| Prop. `separable` | Correct. I checked the telescoping identity for ℓ = 2 by hand. |
| Prop. `systematic` (i)-(iv) | Correct under δ = 0 and full column rank; see m16. |
| Prop. `approx` | Correct. Block permutation is an orthogonal right factor. Exchangeability gives $\mathbb{E}\lVert\eta_\pi\rVert^2 = \mathbb{E}\lVert\eta\rVert^2$. The cond(B) remark is right. |
| Prop. `depth` | Correct for any linear depth-n map (m7). |
| Prop. `context` | Correct (trivial). The l.472 bound is unproved (m13). |
| ε_blend = 1/2 (l.120) | Correct for the sum with i.i.d. parts and affine readouts. More generally, ε ≥ 1/2 for any symmetric g under any readout (m2). |

Numbers behind the verdicts, checked against the aggregates. All match except where an issue is flagged:
- **E2 κ** (`agg_r4_e2e5.txt`): 0.728 at V = 8 and 0.826 at V = 64, with references 0.98 and 0.484.
- **E2 readout figures** (`agg_e2h_partial.txt`, glyph rule): untrained 0.825-0.826; vocabulary readout 0.23-0.29; within-condition r 0.50 / 0.30, CI 0.09-0.32; input r 0.35; coefficients 0.53 / −0.04.
- **E4 recovery and κ** (`agg_r4_e4w.txt`): GPT-2 0.708 / 0.724 / 0.741 (κ 0.44-0.50); CLIP-H 0.664 (κ 0.355); others 0.510-0.613 (κ 0.06-0.26).
- **E4 random-weight controls:** κ −0.11 to −0.01 for GPT-2, T5 and BERT, and 0.05-0.15 for CLIP. Training gains are 0.51-0.61 (GPT-2), 0.26-0.31 (T5), 0.30 (CLIP-H), 0.08-0.17 (BERT, CLIP-L) and 0.04 (CLIP-B).
- **E4 caveat 2 role splits:** random 0.74-0.78 / 0.08-0.34; trained 0.73-0.80 / 0.65-0.68 (GPT-2) and 0.71-0.78 / 0.44-0.55 (CLIP).

---

## 4. Status of r3 points (A, B, C), theory and positioning items

| r3 item | Status |
|---|---|
| A1 informativeness clause ineffective | **Fixed in the definition** (quantitative, tied to τ). **Not applied**: no informativeness value is reported anywhere (M4). |
| A2 / C-B1 CLIP claim | **Partly fixed.** l.265 and l.282 are right; the abstract and the end of l.267 are wrong in a new way (M1b). |
| A3 Def. 2 never applied | **Mostly fixed.** Sec. 4.2 now gives κ, δ and τ, but they are assembled inconsistently (M4). Sec. 4.4 wording is fixed (l.254). E4 δ is reported but is invalid as defined (M2). |
| A4 closed-form range | Fixed (l.401). |
| A5 numbers | Fixed (0.997; occlude transformer values; role-2 spill 0.21; κ ranges). l.282 still says 0.13 (m8). |
| A6 Lippl leak at l.153 | Partly fixed (m13). |
| A7 | l.146, l.153 and cond(B) fixed. "Behavioral quantity" (m12) and the l.561 metric wording (m19) still open. |
| A S1 alpha grid | Fixed (grid to 10⁹, "no penalty ... at the edge", l.572). |
| A S2 reference exact by construction | **Open; raised to M5** with the Prop. `span` argument. |
| A S3 / C S4 covariance-matched edit | Fixed (0.30, l.236). |
| A S4 ε_blend = 1/2 remark | Added (l.120); scope needs m2. |
| A S5 covariance-form value | Fixed (0.66, l.561). |
| A S6 crosstalk derivation; A S7 leak bound | Open (m19, m13). |
| A S8 l.155 | Fixed. |
| A S9 note box "continuously" | Fixed. A new positioning issue in the same sentence (M6c). |
| A S10 Schott 2022 | Fixed (l.155). `app:human` label still open. |
| B A1 Kamp & Partee; B A2 Westerståhl | Fixed (l.501, l.57). |
| B B1 pet fish presented as neutral | Fixed: "takes a position in this debate rather than standing outside it". |
| B B2, B4-B10, B12 | Fixed. |
| B B3 "only"/dense | Fixed in the Fig. 2 caption; still open in Discussion requirement 1 (m11). |
| B B11 abstract length | Fixed (282 words). |
| B B13 Def. 2 overloaded | Open (m4). |
| B W1(b) Chalmers; Hummel & Holyoak | Fixed (l.59, l.290). The van Gelder positioning needs M6a. |
| B: Uselis "vision"; Elmoznino "compresses well" | Open (m15, m14). |
| C S1 E2 vs E4 verdicts | Directionally resolved by whitening, but the cross-section comparison is not like-for-like (M3). |
| C B2 random-weight identification | Open (m18). |
| C S7 Hewitt citation | Open (m19). |

## 5. What would move this to accept

1. **M1:** three text fixes to the abstract and Introduction.
2. **M2:** restate δ relative to a code from which $e$ is read, or via the grid interaction already computed. Report the corrected δ for E4.
3. **M3:** declare the metric in Def. 1. Give the unwhitened κ once. Recompute E2 κ on the whitened 16-PC code, or drop the cross-section comparison.
4. **M4:** rewrite the l.216 verdict with the V = 8 limitation, the pixel-code bound and $z$ = pixels.
5. **M5:** add a non-orthogonal role-filler reference in E4, or qualify "none comes near".
6. **M6:** add the van Gelder sentence, the concatenative-string scope sentence and the corrected note-box sentence.

M2, M5 and the E2 κ recomputation in M3 are re-analyses of cached states. Everything else is text.
