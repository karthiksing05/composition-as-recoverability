# Mathematical audit: "Distinguishing Representational and Behavioral Compositionality via Recoverability"

Source audited: `recoverability-theory/main.tex` (930 lines, read in full). This is pure reasoning; nothing was run.
Severity: **fatal** means the definition or claim as stated is vacuous or false in the setting the paper targets. **Major** means a stated result or classification is false, or does not follow without an added assumption. **Minor** means wording, typing, or scope.
Each item is tagged [CONFIRMED] (a counterexample or derivation is given) or [INTERPRETATION] (it depends on how an undefined term is read).

---

## 0. Summary of the logical structure as written

- Separable (Def. "Separable relation"): there exist an invertible psi and maps phi_i with psi(g_r(x)) = sum_i phi_i(x_i) on "admissible parts".
- Invertible: there exist u_i with u_i(g_r(x)) = x_i on admissible parts.
- Def 4: the system is RC w.r.t. r relative to M if some decomposition found under M makes r separable and invertible. "It then exhibits" R1–R4.
- psi, phi_i and u_i are **unrestricted** (no continuity, no class, no sharing across relations or depths). The "admissible parts" are **undefined**: they could be a finite vocabulary, R^d, or a subset.

Almost every issue below comes from these two gaps: unrestricted maps and an unspecified domain.

---

## FATAL

### F1. Separability is implied by invertibility, so the two-axis taxonomy collapses [CONFIRMED]

**Claim.** If g_r is invertible (injective) on the admissible set D_1 x ... x D_k, then g_r is separable.

**Proof.** Let Im be the image of g_r. Let phi_i(x) = (0,...,x,...,0) be the embedding of x into the i-th block of the direct sum (R^d)^k, so sum_i phi_i(x_i) = (x_1,...,x_k), the concatenation. Define psi = concat ∘ g_r^{-1} on Im. psi is a bijection from Im onto D_1 x...x D_k, so it is invertible, and psi(g_r(x)) = sum_i phi_i(x_i). QED. In the finite case, use one-hot phi_i into R^{sum |D_i|}.

**Consequences.**
- "Separable and invertible" is equivalent to "invertible". Def 4 adds nothing through separability.
- The Appendix B row for conceptual blending ("Separable: No; Invertible: Partially") is impossible on any sub-domain where the mode is invertible.
- The paper's argument that head–modifier psychological forms are "not even separable, since what a modifier contributes depends on its head" does not work. Counterexample: g(x,y) = (x, y + x⊙x⊙y). In raw coordinates, y's contribution depends on x, yet g is separable via psi(z1,z2) = (z1, z2/(1+z1⊙z1)), which gives (x, y) = phi_1(x) + phi_2(y). A head-dependent contribution in raw coordinates never shows non-separability when psi is free.
- Separability without invertibility is not vacuous. It is the additive-conjoint-measurement condition (Krantz, Luce, Suppes & Tversky 1971; Thomsen/cancellation axioms). For example, on {0,1}^2 a g with g(0,0)=g(1,1)=g(0,1) ≠ g(1,0) is not separable for any injective psi. However, the paper never uses separability without invertibility, apart from describing Andreas 2019 in Related Work.

**Restricting psi only partly helps.** If psi must be a homeomorphism or diffeomorphism onto its image, the construction still works for any topological embedding g. A concrete case is a one-layer MLP g(x,y) = tanh(W1 x + W2 y + b) with [W1 W2] injective: psi = atanh(.) − b gives linear phi_i. If psi must be linear, then psi^{-1} distributes over the sum and separability means g itself is additive. At that point psi does no work.

**Minimal fix.** Fix psi before the data, from a declared small class Psi that is shared across all relations and depths. Examples are the identity, an affine map, or one fixed elementwise nonlinearity. Also require phi_i to be **role-factorized**: phi_i = B_i ∘ e, with one filler code e shared by all roles and a fixed role operator B_i (see M1). Then a TPR, Sum x_i ⊗ rho_i = Sum (I ⊗ rho_i) x_i, is the canonical case, and separability carries real content.

### F2. With finite discrete inputs, invertibility is generic, so Def 4 is satisfied by unstructured codes. With continuous inputs, Def 4 does not apply to the networks the paper wants to test [CONFIRMED]

**Finite case.** Take admissible parts = a vocabulary of size V and let u_i be any function. Then "invertible" just means g_r is injective on V^k. A memorized random codebook g(a,b) = w_ab, with w_ab i.i.d. Gaussian in R^m, is injective almost surely, so it is invertible (u_i = table lookup) and, by F1, separable. **The random codebook is therefore RC under Def 4.** The same argument covers:
- EPAM-style whole-chunk storage, which the paper calls "not compositional" (Sec 3.4).
- Weighted addition alpha x + beta y over a generic finite vocabulary. It is injective a.s. because each collision is a nontrivial linear equation on the generic vectors. The paper says "Invertible: No" (App B), and Sec 3.4 says only role–filler composition satisfies Def 4.
- Even x + y on a finite vocabulary, up to the order of the pair. Only its role-swap collisions (Prop 1) fail.

The Remark after Prop 2 sees this ("a non-invertible relation may happen to be injective on that vocabulary") and calls it borrowed recoverability. However, neither the invertibility definition nor Def 4 excludes it, so the formal criterion contradicts the paper's own classification.

**Continuous case.** If admissible parts = R^d (or an open set), then Prop 2 and the "no infinite fibers" argument apply. But a network with token inputs defines rep(a) only on V points. Its g_r on R^d is not determined by the network: it depends on an extension that the experimenter chooses, such as injecting off-manifold embedding vectors. Def 4 then becomes a property of that chosen extension, not of the system.

**Why this is fatal.** The paper's stated targets are trained networks with finite discrete inputs (Sec 4.3, App D). In that setting Def 4 is either trivially satisfied (finite domain) or undefined (continuous domain).

**Minimal fix.** Make recoverability relation-guaranteed and out-of-sample, with readouts from a restricted class:
> r is (U, Psi, eps)-recoverable on a part distribution P if readouts u_i in a declared class U, **fit on composites of a training set S of tuples**, recover the part in role i with error ≤ eps on held-out tuples H. H must include (a) unseen pairs, (b) unseen part–role assignments, and (c) parts that never appeared in any training composite.

Condition (c) is what removes "borrowed" recoverability. A table-lookup u is not in U, or it cannot be fit on S and still generalize to H. A quantitative version can use bi-Lipschitz inverses. If u is L-Lipschitz on composites of norm ≤ R and the parts are δ-separated, a packing argument gives V^k ≤ (C L R / δ)^m, so m ≥ k log V / log(C L R / δ). This is the rigorous form of the Remark's informal "degrades as the vocabulary grows relative to m".

---

## MAJOR

### M1. R3 (new role) does not follow from separability. Prop 3 needs every part in every role [CONFIRMED]

Separability uses **role-specific** maps phi_i. Suppose part x* has never been observed in role 2, which is exactly the split in Fig. 7 and Exp 3 ("a part is never observed in role 2"). Then x* is not a role-2 vertex of the bipartite graph in Prop 3, and phi_2(x*) is completely unconstrained.

Counterexample: g(x,y) = phi_1(x) + phi_2(y), with phi_2(x*) set to any value. The system is separable and invertible, and its composites with x* in role 2 are arbitrary.

The "2V−1 pairs determine all V^2" count (Sec 3.5 item 3) holds only when the spanning tree covers all 2V vertices, which means every part has appeared in **both** roles. That is weak systematicity (Hadley) and the opposite of R3.

**Statements that are wrong as a result:** Table 2 row 2 ("Known part, new role — Separability (Prop 3)"), App C Case 2 ("A separable relation guarantees that a part seen in one role can be used in another"), and the claim in Def 4 that R3 follows.

**Fix.** Require role factorization phi_i = B_i ∘ e. Then x* needs to be observed once, in any role, to fix e(x*), and B_2 e(x*) is determined. Weighted addition (B_i = alpha_i I) is role-factorized but not invertible. TPR is both. So role-factorization, not separability, is what separates "new role" success. The new-role test then tests role-factorization.

### M2. Props 3, 4(iii) and 6 assume psi = id (or a known psi). Under the paper's own definition they have no predictive content [CONFIRMED]

Prop 3 assumes psi = id, but Def 4 lets M choose psi after seeing all the states. Given any injective assignment of values to unobserved pairs, F1 builds a psi that makes the whole table separable. So under the definition as stated, observed composites constrain unobserved ones **exactly as little as under an unrestricted g_r**, which is the paper's own baseline for "contingent". The random codebook from F2 is RC, yet its unobserved composites are unconstrained.

"Whatever the training data" (Sec 3.5 item 4; text after Prop 4) is also false even with psi = id. The determination needs a connected observation graph with every part in both roles (M1).

**Fix.** Require psi in Psi to be fixed before the data, or estimated from S alone, as in F1. State Props 3, 4(iii) and 6 for "psi known, or estimable from O", and list the coverage condition explicitly.

### M3. Prop 4(iii), "every role reversal is determined", is false in general. A correct replacement exists [CONFIRMED]

The reversal (b,a) is determined only if b is a role-1 vertex and a is a role-2 vertex. If a has only appeared in role 1, g(b,a) is free (see M1). The Fodor–Pylyshyn sentence the paper paraphrases, "the ability to build and read one composite entails ... its reversal", needs a **single** composite, not a connected graph.

**Correct replacement (provable).** Let g(x_1,...,x_k) = Sum_i B_i e(x_i), with the B_i known and g invertible on e-codes (for example, the B_i have complementary ranges). From one observed composite z = g(a,b), the readouts give e(a) and e(b). Then g(b,a) = B_1 e(b) + B_2 e(a) is determined, and so is every permutation or substitution of parts seen anywhere. This is the formal necessity the paper is after, and it needs role-factorization plus invertibility. Separability alone does not give it.

### M4. R4 / productivity does not follow from Def 4 and is unsatisfiable as stated [CONFIRMED]

1. Def 4 does not say that composites are admissible parts. Invertibility on D^k → R^m says nothing about g_r(D^k) ⊂ D. Recursion needs a closure assumption.
2. If D ⊂ R^d has nonempty interior and g is continuous with g(D x D) ⊂ D, then g cannot be injective (invariance of domain, dim 2d > d). So R4 "to arbitrary depth" is impossible for continuous parts in fixed dimension. The paper concedes this after Prop 5 but still lists R4 as a consequence of Def 4.
3. Without continuity the claim "no representation of fixed size can invertibly encode trees of unbounded depth" is false. The Cantor pairing N x N → N is a bijection, so trees over a finite vocabulary are invertibly coded in **one** dimension to any depth. With finite precision, pigeonhole rules out every real network. So R4 is either trivial (discontinuous, exact arithmetic) or impossible (finite precision). "Performance" absorbs every failure, which makes R4 unfalsifiable.
4. Prop 5's title, "requires", is not proved. It only shows that a **linear** blend into the same space fails and that per-level invertibility suffices. The necessity direction is trivial: the composed map is injective only if each level is injective on the image of the level below.

**Fix.** Replace R4 with depth-n stability. Readouts from the same class U, applied recursively, recover leaves at depth n with error ≤ eps_n. Bound the growth of eps_n by per-level bi-Lipschitz constants, eps_n ≤ Σ_j L^j eps_level. Test depth beyond training against that predicted curve instead of against an unbounded "competence".

### M5. R2 locality ("altering one part alters only its own contribution") holds only in psi-coordinates [CONFIRMED]

Prop 4(ii) shows two things. First, psi(g) changes only in the phi_i term, which is true by definition of separability. Second, the readouts u_j for j ≠ i do not change, which follows from invertibility alone. With nonlinear psi, changing x_i can change **every coordinate** of rep(c) (see the counterexample in F1). Interchange interventions (Exp 3: "fraction of each intervention's effect confined to its role") measure locality in network coordinates, and Def 4 does not imply that. Table 2 row 4 labels Prop 4 "Localism", which it is not.

**Fix.** Define locality in a declared coordinate system (psi in Psi, affine). Then R2 means the role subspaces range(B_i) are complementary, and interchange interventions on those subspaces test it.

### M6. The taxonomy contradicts Def 4 and Sec 3.4's own rule [CONFIRMED]

- **Superposition with disjoint regions** (App B: separable yes, invertible "only if regions disjoint", roles "by region") satisfies Def 4 when the regions are disjoint. It is a TPR with one-hot (positional) roles. That contradicts "role–filler composition ... is the only mode that satisfies Def 4" and the Fig. 3 caption ("the other modes can produce compositional behavior at most"). It also makes BESIDE in the test domain RC by construction.
- **Elhage-style superposition** of k-sparse features is invertible over the sparse admissible domain (compressed sensing, m ≳ k log(n/k)). By Sec 3.4's rule ("blend only when no admissible choice of primitives makes it invertible") it is not a blend, yet it is listed as a non-compositional mode.
- **EPAM**: see F2. Finite whole-unit storage is injective, so Def 4 is met.
- **"Distinct roles" column**: it is implied by invertibility, since injective g gives g(a,b) ≠ g(b,a) for a ≠ b. The column is redundant except where it is "No", and those rows are already non-invertible.
- **Prop 2 vs App B**: Prop 2 lists superposition as non-invertible over R^d. Even disjoint-region projections P_i discard the off-region components of x_i in R^d. App B says invertible if disjoint. The two agree only if the admissible domain is restricted to each region, which is again the undefined-domain issue.
- **"Partially"** (conceptual blending) has no meaning for binary properties defined with "for all admissible parts".

**Fix.** Make the taxonomy domain-relative and say so. Superposition with disjoint supports **is** a role–filler code. Classify by (role-factorized?, psi-class, invertible out of sample?) instead of the current columns.

### M7. "Necessary vs contingent" and content–context dissociation do not follow as claimed, and the proposed tests do not discriminate them [CONFIRMED / INTERPRETATION]

- **Category mismatch [INTERPRETATION].** Fodor & Pylyshyn's and Aizawa's necessity concerns what the *architecture* forces. Def 4 is a property of the trained states, so a network that ends up with a separable-invertible code got there contingently, through training. Necessity needs a statement about the hypothesis class, for example "every parameter setting of the architecture has phi_i = B_i ∘ e". It cannot come from properties of one learned solution.
- **Test 5 (seed variance) [CONFIRMED].** It does not discriminate necessity from contingency. A deep linear network trained by gradient descent from small init converges near-deterministically to a min-norm solution, giving low variance with no architectural guarantee. A role-factorized model with an under-determined e (a part seen in no composite) can have high variance.
- **beta (data exponent) [CONFIRMED].** beta ≈ 1 holds both for role-specific separable codes (2V−1 pairs, every part in both roles) and for role-factorized codes (about V pairs). A non-separable learner with a good inductive bias can also reach beta ≈ 1. So beta measures the learner's sample efficiency, not the mode of combination. The R3 split discriminates them; beta does not.
- **Prop 6 / Test 3 (context leakage) [CONFIRMED].** Prop 6 holds only if g_r is **exactly** separable with known psi, and then it is a restatement of Prop 3's identifiability up to constants. It does not stop context leakage under learning. Counterexample: fit an additive model phi_1(x) + phi_2(y) by least squares to a non-additive target t(x,y), where x is only ever paired with host h. The fit gives phi_1(x) = t(x,h) − phi_2(h), which absorbs the (x,h) interaction, so h's features leak into every new composite with x. The architecture is separable, yet leakage occurs. Test 3 therefore tests target additivity × coverage, not the definition.

**Fix.** State necessity as a property of the architecture's hypothesis class, and state Prop 6 as an identifiability result ("if the data are exactly separable..."). Replace the seed-variance criterion with a capacity test: does the system compose a **fresh** part, introduced after training (Exp 6), in all roles without retraining the relation?

### M8. The grading in Sec 4.3 contradicts Table 1 and Def 4 [CONFIRMED]

Sec 4.3 says: fails behavior → "blends"; behavior without decomposition → "imitates"; meets all criteria → RC. Table 1 calls "parts recoverable + behavior fails" a *performance failure*, not a blend. Def 4 says RC is independent of behavior. Under Sec 4.3, an RC system with a poor decoder is graded "blends", and RC requires behavioral success.

**Fix.** Use a 2x2 grade: recoverability (out-of-sample, per F2) x behavior.

### M9. The rebuttal to unfalsifiability and the abstract's "infinitely many combinations" assume a continuous domain [CONFIRMED]

Sec 5 says "Prop 2 ensures that no method will find parts that a relation has discarded". Prop 2 is about linear maps on R^d. On a finite vocabulary, alpha x + beta y discards nothing (F2), so a strong enough M will find parts for any generic blend. The abstract says a blend "maps infinitely many combinations of parts onto each composite". That is true only for continuous parts and rank-deficient (or scale-invariant, e.g. x⊙y) maps. For finite inputs, the fiber is generically a singleton.

**Fix.** Qualify both claims to continuous parts, or restate them with the out-of-sample definition from F2. Under that definition the claim becomes: "a blend's readouts, fit on familiar parts, fail on fresh parts", which Prop 2 does support, because the fiber over a fresh composite is an affine subspace.

---

## MINOR

- m1. **Types in Eq. 1**: u_i returns a_i (a part). In the invertibility definition, u_i returns x_i = rep(a_i). If rep is not injective on primitives, these differ.
- m2. **R2 wording**: "r(a,b) and r(b,a) are distinct" should say rep(r(a,b)) ≠ rep(r(b,a)). As compositions they are always distinct. For semantically symmetric relations (is BESIDE ordered?), R2 should not be required.
- m3. **Prop 1 wording**: "not injective under exchange of roles" should say g_r(a,b) = g_r(b,a) for some a ≠ b. The proof is correct, and its last sentence is a tautology (R1 is equivalent to invertibility).
- m4. **Prop 2**: correct. The note "every blend into the space of its parts" (m = d < kd for k ≥ 2) is fine. The Remark's invariance-of-domain claim is correct. The Remark's "degrades as the vocabulary grows" is false for exact injectivity and needs a readout class or margin (see the packing bound in F2).
- m5. **Prop 3 proof**: it shows only a 3-edge path. The general case is the alternating (telescoping) sum along an odd-length path, which is easy but should be stated. The constant-shift remark is correct.
- m6. **Prop 5**: correct as linear algebra, with kernel dimension ≥ (2^n − 1)d. "Fixed size cannot invertibly encode unbounded depth" needs "continuous" (M4.3).
- m7. **App B**: "x⊙y separable under log" needs positive entries. log is not invertible on R^d.
- m8. **Def 4 "over the system's internal states"**: which states (layer, token position), and what rep(a) is for a primitive inside an image, are unspecified. Define rep(a) operationally, for example as the embedding, or as the state of a in a neutral frame.
- m9. **The "5 percentage points" criterion** is arbitrary. Tie it to seed confidence intervals.
- m10. **Sec 3.3**: "parts that shorten the description of experience while preserving its distinctions" is an unformalized constraint on M. It is the only thing that blocks the trivial decomposition primitive = composite, so it should be a formal condition.
- m11. **Leftover editorial note**: `\note{Insert the initial results of TRELLIS here...}` (line 563).
- m12. **Table 2 row 7** ("no forgetting") has no supporting proposition. It holds only for role-factorized codes where adding e(x*) leaves the B_i and other e(·) unchanged, which is an architectural assumption.

---

## What each proposition actually establishes

| Prop | Status | Correct scope |
|---|---|---|
| 1 Role collisions | Correct, near-tautological | Any g_r |
| 2 Non-identifiability | Correct | Linear g_r, continuous parts on R^d; false/inapplicable for finite vocabularies |
| 3 Substitutivity | Correct with psi = id | Needs a connected graph with every part in both roles; no new-role guarantee |
| 4 (i)(ii) | Correct; (i) needs only invertibility, (ii) is definitional | Locality only in psi-coordinates |
| 4 (iii) | **False as stated** ("every reversal") | True only for reversals whose parts are vertices on both sides; the role-factorized version (M3) is true from one composite |
| 5 Depth | Correct linear algebra; "requires" not shown | Positive half impossible for continuous fixed-dim, trivial for discontinuous |
| 6 Context | Correct but restates Prop 3 | Identifiability under exact separability; no anti-leakage guarantee under learning |

---

## Operational definitions for a trained network with finite discrete inputs (recommended replacement)

Fix in advance: the part vocabulary per role, a layer/position for states, a readout class U (for example linear, or a 1-hidden-layer MLP with a norm bound), a coordinate class Psi (identity, affine, or one fixed elementwise map), and the method class M.

1. **Recoverability (R1, relation-guaranteed).** Fit u_i ∈ U on states for tuples in S. Report recovery on held-out H_pair (unseen pairs), H_role (parts unseen in role i), and H_part (parts introduced after training, e.g. Exp 6 tokens, or embeddings sampled from the fitted embedding distribution, flagged as off-manifold). Familiar-vocabulary recovery alone is uninformative, since any injective code passes it.
2. **Role-factorized separability.** Fit Sum_i B_i e(x_i) in psi-coordinates on S, with e shared across roles. Report prediction error of states on H_role and H_part (Andreas TRE-style, but out of sample). Role-specific phi_i should be reported as a separate, weaker level.
3. **Locality (R2).** Check that range(B_i) are complementary. Interchange interventions restricted to range(B_i) should move only role-i readouts and outputs.
4. **New role (R3).** The H_role split, which tests role-factorization (M1).
5. **Depth (R4).** Recursive application of the same U. Report eps_n against the bi-Lipschitz prediction from depth-1 and depth-2 constants.
6. **Necessity.** A property of the architecture (are all parameter settings role-factorized and invertible?), checked analytically. Empirically, check zero-shot composition of fresh parts (H_part) without updating the relation. Seed variance and beta are reported, not used as criteria.

Under these definitions, the paper's intended classification follows. Weighted addition is role-factorized but fails 1 on H_part. A random codebook fails 1 and 2 out of sample. Superposition with disjoint supports and TPR pass. Elhage-sparse superposition passes 1 only within the sparse regime.
