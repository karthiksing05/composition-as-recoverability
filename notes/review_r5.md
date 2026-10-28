# Review R5: verification of the round-4 fixes

**What I checked.**
- File: `recoverability-theory/main.tex`, 598 lines, mtime 06:27. All line numbers refer to it.
- The full diff against `notes/main_pre_r4fix.tex`.
- The result files:
  - `agg_r4_e4w.txt`
  - `agg_r4_e4mix.txt`
  - `agg_e4c.txt`, for the unwhitened κ
  - `e2_informative.py` and its `.json` output
  - `e4_pretrained.py`, for the interaction grid
- The regenerated figures: `fig_taxonomy`, `fig_fibers` and `fig_depth_pretrained`, all timestamped 06:19–06:20.
- The bib entries of every new citation.

I ran nothing.

## 1. Status of the round-4 items

### Reviewer A

| Item | Status | Where |
|---|---|---|
| M1a recursive composer in abstract | Resolved | l.46 |
| M1b CLIP / text-to-image claim | Resolved | l.46, l.272, l.274 |
| M1c full column rank for reversals | Resolved | l.46 ("enter every role through one shared code"), l.61 |
| M2 δ penalizes reduced codes | Resolved in text: e′ added to Def. 1 (l.113, l.118); E4 reports the grid interaction as a lower bound (l.270, l.574). **The e′ clause creates a new definitional hole (N1).** | l.113–118, l.270, l.574 |
| M3 metric dependence | Mostly resolved. The metric is declared (l.118), the unwhitened κ is given (l.270), and the E2-vs-E4 κ comparison is deleted. Sec. 4.2 still does not state how its pixel code is scaled, which the Terminology paragraph asks for (l.293). | l.118, l.270, l.223 |
| M4 Sec. 4.2 verdict | Resolved. Informativeness is reported (0.92 / 0.89, matching `e2_informative.json`: means 0.918 / 0.893); V = 8 is limited to κ; the pixel-code bound argument is given. | l.223 |
| M5 reference exact by construction | Resolved. A mixed role–filler reference is added (0.92–0.98; κ_mix GPT-2 0.47–0.52, CLIP-H 0.42; this matches `agg_r4_e4mix.txt`). | l.270, l.574 |
| M6a van Gelder | Resolved | l.59 |
| M6b concatenative strings | Resolved | l.130 |
| M6c Fodor & Pylyshyn performance | Resolved | l.130 |
| m1 "vacuous" | Partly. Fixed at l.61 and the Lusin–Souslin citation is added (l.127). The Lemma title at l.316 still reads "Unrestricted coordinates make separability vacuous"; replace with "Unrestricted coordinates reduce separability to invertibility". | l.316 |
| m2 sum, not "a symmetric blend" | Resolved | l.124, l.127 |
| m3 κ = 1 − 2ε | Resolved. The τ/2-vs-τ asymmetry is still not commented on. | l.127 |
| m4 informativeness wording; box overload | Partly. "Any injective code" and τ in the verdict are fixed. Def. 2 still packs definition, κ, τ, informativeness and productivity into one box. | l.124 |
| m5 Table 1 header | Resolved as asked, **but it creates a contradiction (N3).** | l.174 |
| m6 HRR wording | Resolved | l.201 |
| m7 Prop. depth level-specific maps; "allows" | Resolved | l.454, l.261 |
| m8 0.13/0.14 | Resolved | l.289 |
| m9 "below one half" | Resolved | l.270 |
| m10 two names | Resolved in usage, **but defined three different ways (N2).** | l.127, l.291, l.293 |
| m11 "dimensions its parts occupy" | Resolved | l.121, l.291 |
| m12 "predicted state" | Resolved | l.450 |
| m13 Lippl | Partly. l.160 is fixed; the l.474 bound ("at most a design-dependent constant times …") is still unproved. | l.474 |
| m14 Elmoznino "plausibly" | Resolved | l.164 |
| m15 Uselis "vision" | Resolved | l.57, l.160 |
| m16 readout values | Resolved | l.427 |
| m17 ψ injective on image | Resolved | l.318 |
| m18 random-weight id/role | Resolved, with an overstatement (N5). | l.270 |
| m19 leftovers | Partly. The l.563 metric wording and the Hewitt "control tasks" phrasing are resolved. `\label{app:human}` still sits outside a sectioning command (l.531). The crosstalk formula still has no derivation (l.261). | l.531, l.261 |
| m20 "approximately"; red hair | Partly. "Approximately" is added, but with broken grammar (N4). The Kamp & Partee context-dependent-modifier clause is not added. | l.121 |

### Reviewer B

| Item | Status | Where |
|---|---|---|
| M1 two notions | Resolved, but see N2 | l.46, l.127, l.291, l.293 |
| M2 composer in abstract | Resolved | l.46 |
| M3 CLIP / SD claim | Resolved | l.46, l.272, l.274 |
| M4 x + y running example; Fig. 8 placement | Resolved. The text and caption now say lookup returns the pair but not the roles, and the figure was moved into Sec. 2. | l.80, l.82–87 |
| M5 taxonomy figure text | Resolved. All six points are fixed in the regenerated PDF/PNG: no "never seen", blending "fresh parts not recoverable; roles collide when symmetric", "Reversals fixed by the state", slots/templates removed, head–modifier under superposition, layout clean. Remaining tension: Fig. 2 places "prototypes" under Blending, while Table 5 (l.492) and l.503 treat prototype combination as approximately role-factorized, with recoverability "Not determined". | Fig. 2 |
| M6a Lake 2019 | Resolved | l.297, l.542 |
| M6b neural analogue | Resolved | l.297 |
| M6c conjunctions / LISA | Resolved | l.297 |
| M6d Evans | Resolved | l.297, l.546 |
| M7 random-weight baselines | Resolved; Fig. 6d now has crosses. See N5. | l.270, Fig. 6d |
| M8 informativeness | Resolved | l.223 |
| M9 Marcus and in-bib works | Partly. Marcus 1998 (l.149), Park 2024 (l.176) and Frankland 2020 (l.297) are added. `yuksekgonul2023`, `lepori2023`, `russin2024` and `quiltydunn2023` are still in the bib but uncited. | l.149, l.176, l.297 |
| minor 1–5 | Resolved | l.289, l.291 |
| minor 6 | Resolved | l.295 |
| minor 7, 8 | Resolved | l.492, l.506 |
| minor 9 | Resolved, but the abstract grew from 282 to 330 words (see N9). | l.46 |
| minor 10 | Resolved | l.57, l.164 |
| minor 11 | Partly. Uselis and SLOG are fixed. Griggs & Cox is still the only content-effect source. | l.546 |
| minor 12 | Open (Def. 2 box) | l.124 |
| minor 13 | Resolved | l.217 |
| minor 14 | Open. The Method paragraph is still one block. | l.241 |
| minor 15 | Resolved | l.127, l.192 |
| minor 16 | Partly. The Fig. 3d caption is fixed (l.208). Fig. 5a's y-axis is unchanged, since `fig_structure` was not regenerated. Fig. 6c's legend is still at the top of the panel. | — |
| minor 17 | Resolved | l.270, l.574 |

### Reviewer C

| Item | Status | Where |
|---|---|---|
| M1 CLIP-H | Resolved | l.46, l.274 |
| M2 random-weight id/role | Resolved, with the overstatement in N5 | l.270, Fig. 6d |
| m1 0.14 | Resolved | l.289 |
| m2 0.26–0.30 | Resolved | l.272, l.289 |
| m3 sum ref 0.48 | Resolved | l.270 |
| m4 0.89–1.00 | Resolved | l.270 |
| m5 "top of ranking" | Resolved | l.274 |
| m6 share of 16-component code; 7 of 305 nouns | Resolved. One leftover: l.289 says "length of the noun" where the rest of the paper says "token count". | l.268, l.274, l.289 |
| m7 pool curve missing 2 tasks | Open. No "2 seeds" note. | l.274 |
| m8 "earlier run" | Resolved (sentence dropped) | l.574 |
| m9 latent informativeness | Resolved | l.223 |
| m10 aggregator; "within 0.08" | Paper side fine. The claim at l.566 is still unverifiable from the printed aggregate, and the `agg_r4.py` E2h block is not fixed. | l.566 |
| m11 stale docstring | Open. `e4_pretrained.py` l.5 still says "(ridge, MLP)", l.13 still says "40 fresh nouns", and l.29–30 still import RidgeCV and MLPRegressor without using them. | code |
| m12 protocol details | Resolved | l.574, l.279 |
| m13 Hewitt | Resolved | — |

## 2. New problems introduced by the revision

**N1 (MAJOR). The new e′ clause empties role-factorization of its "shared code" content (l.113).**

Definition 1 now quantifies over any code e′ of which e is an affine function. Take e′(x) = (e(x), φ₁(x), φ₂(x)) with B₁ = [0 I 0] and B₂ = [0 0 I]. Then any additively separable relation c + φ₁(x) + φ₂(y), including two unrelated role-specific lookup tables, is 0-role-factorized.

That contradicts three passages:
- l.121: "each part contribute the same code to every whole, transformed only by its role";
- l.423: separability with role-specific φᵢ "says nothing ... that case requires role-factorization";
- l.151: "Without role-factorization, observed composites constrain unobserved ones only when …".

Prop. `systematic` still needs full column rank, so the theorem is safe. The definition and Definition 2's δ verdict are not.

Fix: require each Bᵢ to be injective. That keeps x + y and x + 0.5y role-factorized, keeps the concatenation reference with e′ = the full state at δ = 0, and makes a part's contribution to one role determine its contribution to every other. Replacement for l.113:

> The relation is *δ-role-factorized* if, for some code e′ of dimension d′ of which e is an affine function (possibly e′ = e), there exist c ∈ ℝ^m and injective matrices B₁, …, B_k ∈ ℝ^{m×d′}, one per role and shared by all parts, such that …

Add to l.118:

> Injectivity of each Bᵢ is what makes the code shared: a part's contribution to one role then determines its contribution to every other. Without it, a code that stacks role-specific features would make every additively separable relation role-factorized.

Also, Prop. `approx` (l.439) mixes ε, which is measured on e, with h = c + Bx + η, where x stacks e. Add "with e′ = e".

**N2. "Compositional system" is defined three different ways.**
- l.127: "one whose own procedures also read its parts from composites of fresh parts, a *compositional system*".
- l.291: "one that also meets all five requirements, verified on fresh parts …, is a compositional system". Requirements 1 and 5 concern width and depth, not the system's own reading.
- l.293: "'compositional system' for systems whose own procedures also use them".

Make l.127 the single definition and point the others to it. For l.291, replace with:

> A network whose states meet Definition 2 is representationally compositional; one whose own procedures also read its parts from composites of fresh parts, which requirements 2, 4 and 5 test, is a compositional system; one that meets only the behavioral tests should be described as generalizing compositionally.

Also, Def. 2 itself (l.124) says "the system is compositional at tolerance τ". Write "representationally compositional at tolerance τ".

**N3. The Table 1 header change (l.174) contradicts the placement of the glyph networks (l.223) and the claim at l.127.**
- The behavioral column is now "Behavior succeeds on new combinations". The V = 8 networks *succeed* on new combinations (0.85–1.00, l.221) and fail only on fresh glyphs.
- Yet l.223 places them in "the upper right cell of Table 1" (Behavior fails).
- l.127 says Table 1 "crosses" representational compositionality with "compositional system". The latter is defined by fresh parts, which the new header dropped.
- Fix: restore the fresh-part header and use A's alternative label for the lower-left cell. l.174: "Behavior succeeds on new combinations, including fresh parts | Behavior fails on fresh parts". l.177, lower-left: "Generalization without affinely recoverable parts; systematicity is contingent on the data \citep{chaabouni2020}".

**N4. Broken sentence at l.121.** The current text reads:

> …which is the classical requirement that a constituent contribute the same thing to every complex in which it occurs \citep{fodor1988}, who asked for approximately the same contribution and so license a graded δ, with the coordinates restricted to affine maps so that the requirement has content.

With `\citep` this renders "(Fodor and Pylyshyn, 1988), who asked…", and the final "with the coordinates…" clause then dangles. Replace with:

> …which is the classical requirement that a constituent contribute approximately the same thing to every complex in which it occurs \citep{fodor1988}; their "approximately" licenses a graded δ. We restrict the coordinates to affine maps so that the requirement has content.

**N5. The random-weight sentence overgeneralizes (l.270).**
- The sentence: "Every trained model assigns roles to fresh nouns better than the sum reference … but so does the same architecture with random weights … (roles 0.50–0.90 …)". Random T5 assigns roles at 0.497–0.501, which equals its sum reference (`agg_r4_e4w.txt`), so it does not assign roles better.
- "Only recovery, and hence κ, separates trained from random weights" conflicts with caveat 3 (l.274). There, CLIP-B and CLIP-L fall *below* random on recovery once the token count is removed, and CLIP-B's training gain is 0.04 (l.272).
- Replacement: "…but so, except for T5, does the same architecture with random weights, scored on its own part codes (roles 0.66–0.90, identification 0.44–0.74; random T5 is at chance on roles, 0.50); random CLIP ViT-B/32 even identifies more fresh nouns than the trained model (0.74 against 0.71). Of the three measures, only recovery, and hence κ, separates trained from random weights, and for the two smaller CLIP encoders it does so only through the token count (third caveat below)." The non-T5 random range is 0.665–0.899.

**N6. "Invariant to rotations and rescalings" (l.118) is wrong if "rescalings" is read per coordinate.**
- Whitening is a rotation followed by a per-coordinate rescaling, and the paper's own E4 shows that it changes κ (0.81–0.83 to 0.44–0.50, l.270).
- Write: "ε is invariant to rotations and uniform rescalings of e, but not to other linear changes of coordinates, including rescaling single coordinates."

**N7. The grid interaction is an in-sample estimate on fitting nouns (l.270, l.574).**
- The grid uses the first 12 fitting nouns (`e4_pretrained.py:227`).
- l.118 says δ is evaluated "never on the parts used for fitting".
- The share is still a valid lower bound on the in-sample residual, but it is a sample estimate, not a bound on the population δ.
- l.270: write "the interaction share of a two-way additive fit on a complete 12 × 12 grid of fitting nouns, an in-sample estimate of a lower bound on δ for every code, is …".

**N8. The unwhitened κ comes from a pipeline that differs in more than whitening (l.270).**
- "on the same components without whitening … GPT-2 reaches κ = 0.81–0.83" matches `agg_e4c.txt` (0.833 / 0.817 / 0.807).
- But e4c also used the narrow penalty grid; the file itself labels it "e4c (narrow grid)".
- Write "with the same components unwhitened (and a narrower penalty grid), …", or recompute with `--wide`.

**N9. Smaller items.**
- l.127, "we report this share wherever …": the antecedent is now the projection loophole, not a share. Write "we report this informativeness wherever …".
- l.289, "beyond the length of the noun": write "beyond the token count of the noun", to match l.46 and l.274.
- The abstract grew from 282 to 330 words. B's r3 B11 asked for about 280. Candidate cut: the sentence "Analytic modes of combination match these bounds, and vector-symbolic superposition … like a lossy blend."
- l.268, "the ones that condition Stable Diffusion 1.5 and 2 \citep{rombach2022}": Rombach et al. (CVPR 2022) describe latent diffusion. The SD 1.5 (CLIP-L) and SD 2 (OpenCLIP-H) text encoders come from the public releases, not that paper. Write "the latent-diffusion models of \citet{rombach2022} released as Stable Diffusion 1.5 and 2", or cite the model cards.

**Numbers checked and correct:**
- informativeness 0.92 / 0.89 over 10 generators;
- mixed reference 0.92–0.98, κ_mix GPT-2 0.47–0.52 and CLIP-H 0.42, with a largest rise of 0.062 (CLIP-H);
- interaction: GPT-2 0.007–0.012, CLIP ≤ 0.019, BERT/T5 0.108–0.193, references 0.000;
- random roles 0.497–0.899 and identification 0.443–0.744; trained roles 0.895–0.996 and identification 0.646–0.939;
- unwhitened GPT-2 κ 0.807–0.833.

All new citations resolve to bib entries that support their sentences, except the rombach2022 point above. Kechris Thm 15.1 is the Lusin–Souslin theorem.

## 3. Verdict

**Not yet ready to circulate.** One item blocks it, and two more should be fixed together with it. All three are text changes, about 30 minutes in total:

1. **N1.** The e′ clause makes role-factorization equivalent to additive separability, which undercuts the paper's central distinction. Add "injective Bᵢ" and the one-sentence justification.
2. **N3.** Table 1's new header puts the paper's headline networks in the wrong cell. Restore the fresh-part header.
3. **N2 and N4.** Unify the three definitions of "compositional system" and repair the l.121 sentence. These are what a careful first reader will trip on.

After these, N5–N9 and the remaining open minors are polish. The open minors are the Def. 2 box split, the Lemma title, `app:human`, the crosstalk derivation, the l.474 bound, the uncited in-bib works, the Fig. 5a axis, the pool-curve seeds and the docstring. None of them blocks circulation. Every number in the revised text matches its output file.
