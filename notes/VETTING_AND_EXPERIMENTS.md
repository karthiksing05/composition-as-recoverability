# Recoverability paper: vetting, experiments, and what changed (2026-10-27)

Original draft: `notes/original_paper_snapshot/` (untouched copy of the zip contents).
Revised paper: `recoverability-theory/main.tex` (+ `preprint.sty`, new figures, extended `references.bib`).
Agent reports: `notes/math_audit.md`, `notes/literature_novelty.md`, `notes/style_guide.md`.

## 1. Confirmed problems in the original draft (and how the revision handles them)

| # | Problem | Severity | Resolution |
|---|---|---|---|
| F1 | "Separable" with unrestricted invertible psi is implied by invertibility (psi = block-embed o g^-1), so the 2-axis taxonomy collapses | fatal | Lemma 1 states it; Def 1 fixes affine coordinates and requires a shared part code (role-factorization) |
| F2 | On a finite vocabulary any injective relation is invertible (lookup); Def 4 was satisfied by random codes, EPAM chunks, weighted sums | fatal | Lemma 2; Def 1 requires recovery of FRESH parts by readouts of a declared class; E1a confirms lookup decodes every injective operator |
| M1 | R3 (new role) does not follow from role-specific separability | major | Prop 6 (role-factorized) gives new-role assignment and reversal from one composite |
| M2/M3 | Props 3/4(iii)/6 assumed psi = id; "every reversal determined" false | major | restated with affine coordinates; reversal now from Prop 6 |
| M4 | R4 unsatisfiable for continuous fixed-width states | major | productivity is a depth curve (Def 2, Note 1, Prop 7); E3 |
| M6 | Taxonomy inconsistent: HRR listed as approx. invertible but Prop 2 makes it a rank-d/2d blend | major | taxonomy redrawn: "fixed-width superposition" class; E1b: HRR fresh R2 = 0.42 = half-rank projection |
| new | No quantitative link between blend and recovery | -- | Prop 3: exact ceiling tr(A_i^T (AA^T)^+ A_i)/d, mean rank/kd, MMSE over all readouts for Gaussian parts; matches E1 to within 0.06 |
| new | Readout/procedure must be learned from enough parts | -- | Prop 4 (span); E1c and E2c |

Citation fixes (verified by the literature agent): Goodale first name Michael (+DOI); elmoznino2024 -> ICML 2025 version; elmoznino2026 description; Schapiro 2025 (failure at abstraction, not mapping); Conwell (DALL-E 2, 22% of images by human raters); SCAN 20.8% = best architecture tested; SLOG contrast is with COGS; Liu 2022 does not report the "inside" failure (now attributed only to anonymous2026seen). Added: Wiedemer 2023, Lachapelle 2023, Brady 2023, Schug 2024, Lippl & Stachenfeld 2025, Uselis 2026, Redhardt 2025, Bernardi 2020, Ito 2022, Geiger 2021/2024, Sutter 2025, Soulos 2020, Montero 2021, Chaabouni 2020, McCoy 2020, Luce & Tukey 1964, Plate 2003, Frady 2018, Robbins 2005, COGS, Park 2024, Yuksekgonul 2023, model papers.

Novelty after the audit: separable+invertible as mathematics is Wiedemer/Lachapelle (additive decoder + consistency). What is new: using the conditions as a DEFINITION, the fresh-part requirement separating relation-borne from vocabulary-borne recovery, the exact ceiling (Prop 3), the span requirement (Prop 4), the calibration-against-own-part-codes protocol, and the empirical results below.

## 2. Experiments (all CPU, SLURM overcap; code in `experiments/`)

| Exp | Script | Grid | Key result |
|---|---|---|---|
| E1 calibration | e1_calibration.py | 14 operators x V 4..128 x 5 seeds, d=16 | lookup = 1.00 for every injective operator (0.50 symmetric); fresh linear R2 matches Prop 3 ceiling (proj .75: 0.69, .5: 0.45, HRR 0.42, MAP 0.45, x+.5y 0.80, sum 0.50, full rank 1.00); V=8 < d: concat fresh R2 0.05; MLP readouts transfer worse |
| E2 networks | e2_glyph.py | {slot, MLP, CNN, transformer} x m {8..64} x V {8..64} x 10 seeds (+ OCCLUDE, V 16..64, 5 seeds) | held-out pairs ~0.9-1.0 everywhere; fresh parts 0.12-0.30 at V=8 vs 0.85-0.90 at V=64; m=8<kq caps fresh at <=0.68; state recovery predicts fresh transfer r=0.94 (held-out r=0.61); shared role decoder best on new roles only at V=8 (CNN matches by V=16), no help on fresh parts; OCCLUDE lowers fresh transfer (0.84-0.91 -> 0.61-0.80); seed spread large only at small V |
| E3 depth | e3_depth.py | RAAM w 16..256 x V {16,64} x 5 seeds; HRR unitary | learned recursion perfect at trained depths, fails at depth 3 with ample width (R2 0.26, ID 49%); HRR ID 1.00 -> 0.49 -> 0.20 (depth 1/3/5) |
| E4 pretrained | e4_pretrained.py | CLIP-B/32, SD1.5 CLIP-L, T5-s/b, BERT, GPT-2 x 3 relations x 3 seeds; 305 nouns | training nouns R2 0.82-1.0; fresh nouns: CLIP -1.21/-0.92 (below sum ref 0.47), BERT 0.12, T5-s 0.20, T5-b 0.57, GPT-2 0.79; concat ref from own part codes 0.97-1.0; CLIP reversal similarity 0.91 |

Bugs found and fixed during the work: E2 first relations were exactly additive at the input (ANOVA interaction 1e-30), fixed-region occlusion too, so OCCLUDE places the part by a host-dependent quadrant (input interaction 11-14%); E3 HRR first used non-unitary roles (noise blow-up) -> unitary roles, raw R2 not meaningful for HRR so ID with clean-up is reported; E4 first version had no calibration (target dim >> #nouns) -> PCA-32 targets + reference operators on each model's own part codes + 305 nouns.

## 3. Open items for the author

- `anonymous2026seen` (own ICLR 2027 submission) is cited in a signed paper: check ICLR anonymity policy. It is now the only support for the "object inside another" diffusion failure.
- TRELLIS placeholder removed; TRELLIS now listed as a future test candidate.
- E4 conclusions are relative to the declared part code (state of "a {x}") and linear readouts; discovered codes (ROLE/TPDN, DAS) are the obvious next M.
- Not run: experiment 6 of the original plan (new primitive / forgetting), interchange interventions, human data.

## 4. Review rounds 2-6 (2026-10-27 to 2026-10-28)

Reviews are in `review_r{1..5}_*.md`; each round's edits are scripted in `scripts/fix_*.py`, with the pre-edit text in `main_pre_*.tex`.

- **E4 (pretrained encoders) redone.** Part codes are whitened (one direction held up to 99% of the variance), penalties are chosen on validation nouns, and random weights are scored on their own codes and on the trained model's. Results: κ ≤ 0.52 for every encoder (GPT-2 0.44-0.50, CLIP-H 0.36); without whitening, GPT-2 reaches 0.81-0.83. Training raises κ by 0.51-0.61 (GPT-2), 0.26-0.30 (T5, CLIP-H) and ≤ 0.17 elsewhere. A mixed role-filler reference (`e4_refmix.py`) moves κ by at most 0.06.
- **E2 verdict.** The verdict is stated at V = 64 only (τ ≈ 0.17). The glyph code is informative: R² 0.92 from pixels and 0.89 from 16 PCs (`e2_informative.py`).
- **Definitions.**
  - Def 1: role-factorization goes through a richer code e′ with injective B_i, and ε is invariant only to rotations and uniform rescalings.
  - Def 2: κ against the concatenation and sum references.
  - Def 3 (round 6): informativeness and productivity, split out of Def 2.
- **Round 6 (polish).**
  - Text: the Method paragraph of Sec 4.3 is split into run-ins; the HRR crosstalk formula and the memorization-leak bound are derived in App. A; Sec 4.2 states the pixel scale.
  - Citations: the Kamp & Partee modifier clause and the content-effect sources (Johnson-Laird et al. 1972; Cheng & Holyoak 1985) are added; the four uncited bib entries are now cited; the stray `app:human` label is removed.
  - Figures: Fig 5a's axis is tightened, and the Fig 6c/d legends are moved below the axes.
