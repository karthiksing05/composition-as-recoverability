# Review R2-C: "Defining Compositional Representations by Recoverability" (second round)

Reviewer C: empirical ML and mechanistic interpretability (probing, control tasks, causal interventions/DAS, binding in LMs, compositional-generalization benchmarks).

**Scope of this check.** I read the revised `main.tex` and checked it against the aggregated outputs (`agg_r2.txt`, `agg_e5.txt`, `agg_e4b.txt`, `agg_e1rob.txt`, `fig_e3e4.txt`, `fig_e2.txt`), the evaluation code (`e2_glyph.py::run`, `e5_structure.py::run`, `e4_pretrained.py::evaluate/run`, `agg_*.py`, `fig_*.py`) and the figure PNGs. I aggregated the raw JSON with awk only for numbers that no aggregate file prints: RAAM depth from `results/e3`, E1 concatenation and lookup from `results/e1`, per-seed OCCLUDE input interaction from `results/e2v2occ`, and best layer per part code from `results/e4b`. I ran no Python.

**Version note.** `main.tex` was being edited while I reviewed it. Its mtime changed to 23:34 during my pass. Line numbers refer to the version with md5 `dca09e16`; the results paragraphs are at L214–L220 (E2), L236–L238 (E5), L254 (E3) and L263–L265 (E4).

---

## 1. Status of round-1 issues

| R1 issue | Status | Comment |
|---|---|---|
| **W1.** E2 probe span-limited by the network's vocabulary | **Fixed** | The population readout is fitted on 400 pool glyphs, with an untrained-network control and the Eckart–Young ceiling. This is the experiment I asked for, and it changes the cell assignment. The paper now places the networks in the upper-right cell (failure of performance), which is the correct reading. |
| **W2.** Pooled r = 0.94 driven by V/m; shared test noise; clipping | **Mostly fixed** | Behavior and readouts now use disjoint fresh sets (`fresh_b` and `fresh_p`), and within-condition correlations are reported (0.64 vocabulary, 0.40 population). The pool r has a bootstrap CI; the vocabulary r and the *difference* between the two do not. The pooled r in Fig. 4d is still computed on clipped values (0.957 clipped vs 0.953 unclipped; immaterial). The "V and m alone" baseline model is still not reported. |
| **W3.** Part diversity confounded with number of pairs | **Partly fixed** | The fixed-pairs control is in the main text (Fig. 4e). There is still no fixed-V, varying-pairs sweep, and the Discussion claim is unchanged. See mismatch rows M10–M11. |
| **W4.** E4 "fresh" nouns, in-sample training scores, test-set layer selection | **Fixed** | New pairs of fitting nouns are now held out. The layer and part code are chosen on 40 validation nouns, and an oracle-sensitivity check is reported (≤ 0.01, verified). The text now says "fresh to the readout". No nonce words were used, which the Limitations acknowledge. |
| **W5.** R² scale miscalibration, the "and" control, reversal similarity | **Partly fixed** | Done: z-scoring dropped (centred states only), identification reported, the "and" control reported, and the conclusion rephrased as *order*, not relational semantics (good, honest). Not done: GCV is still leave-one-composite-out rather than noun-grouped, and there is no scale-invariant metric. CLIP-B32 still has R² = −0.35 alongside identification 0.29 (29× chance), the signature I flagged. There is no role/position dissociation (passives or converses) and no control task. |
| **W6.** Part code favours causal models; contextual token states | **Partly fixed** | Three part codes are now compared and role 1/role 2 are reported. From the raw `e4b` data, GPT-2's lead does *not* depend on the phrase code: with the "there is a x here" code GPT-2 s/m/l reach 0.75/0.81/0.82 against 0.78/0.83/0.84. That partly answers my concern, and **the paper should say so**. Not done: contextual noun-token states, a context-matched code, and any causal check on a pretrained model. |
| **W7.** Underpowered role–filler / new-role claims | **Partly fixed** | "Best" is now claimed only at V = 64 (0.94 vs 0.88–0.93; consistent across m = 16–64, so acceptable). The strong split is still 2 held-out parts at V = 8, and its seed SD is now used as positive evidence (see N5). The role–filler decoder's *fresh-part deficit* is still not stated (N11). There are no CIs in the text. |
| **W8.** OCCLUDE confounds | **Partly fixed** | Recovery from the state (population readout) is now reported. Behavior is still averaged over output relations, which includes rendering an occlusion; the BESIDE-output-only number `beh_fresh_r10` exists in the JSON but is unused. "All three are invertible in principle" (L208) is still unqualified, and "approximate the non-additive relation by a nearly additive one" still rests on the ANOVA alone. |
| **W9.** Statistics and artifact hygiene | **Mostly fixed** | Seeds are complete (OVER 9 × 64 = 576; OCC 10 × 48 = 480; controls 10), and controls `e6pairs`, `e6aux` and `e2q` are reported. Remaining: stale docstrings (`e4_pretrained.py` says 80/40 nouns; `agg_e4b.py` says selection on "on" and "left" but uses all five relations). `agg_r2.txt` still prints the superseded E5 v1 block (`results/e5`, values down to −1.4e7) next to the real E5b results. The text still does not say that E4 "seeds" are noun splits of one checkpoint. |
| **W10.** Prop. 4 invoked for nonlinear decoders | **Partly fixed** | The appendix now hedges ("a nonlinear decoder is subject to the same lack of constraint"). L216 still says "as Proposition 4 predicts for any procedure learned from composites". Fresh behavior keeps rising from V = 16 to V = 64, well past the latent-span threshold V ≥ q + 1 = 9, so this is an analogy, not a prediction. |
| m1 Protocol bullet vs E4 splits | Fixed | |
| m2 Part code = glyph pixels | Fixed | |
| m3 Fig. 4d axis/clipping/pooled | Partly fixed | Neither clipping nor the fact that the printed r is pooled is stated in the caption. |
| m4 GCV leaks across composites sharing parts | Not fixed | Affects E2 and E4. |
| m5 Standardization `+1e-6` | Partly fixed | Fixed in E4. E2 still z-scores h. |
| m6 PCA target space | Changed | PCA is now fitted on all 305 nouns *including the fresh ones* (transductive). See N8. |
| m7 Role-assignment reference trivially 0.5 | Partly fixed | The "and" caveat addresses it, but L263 still compares against 0.50 and concludes "role information survives". |
| m8 Table 1 / Chaabouni mapping | Fixed | Cell renamed "Vocabulary-bound generalization". |
| m9 Abstract "almost perfectly" | Fixed | |
| m10 Self-citations | Partly fixed | The TRELLIS citation remains. |
| m11 Probing literature | Partly fixed | Geiger and Feng are now cited. Hewitt & Liang (control tasks), Elazar et al. (amnesic probing), Belinkov (probing critique), Tenney et al. and Thrush et al. (Winoground) are still absent. |
| m12 Prop. 7 statement mixes w = d and general w | Not fixed | The statement fixes A, B ∈ R^{d×d} (so w = d) but bounds by min(1, w/(2^n d)). |
| m13 Difficulty-matched baseline vs strong split | Partly fixed | Both are in the tables, but the text never contrasts them. |

---

## 2. Number check

I checked every number in Sections 4.1–4.5, the abstract, the Discussion and the appendix. **Verified correct** (with a tolerance of ±0.005 for rounding) unless listed below:

- **E1:** lookup 1.00, blends 0.49; fresh R² 1.00/0.69/0.45/0.80/0.50/0.42/0.45; MLP 0.10 vs 0.45; concatenation 0.05/0.90/1.00 at V = 8/16/32.
- **E1 robustness:** 0.45/0.47/0.37/0.61 and kernel −0.00 vs 0.61.
- **E2:**
  - held-out ≥ 0.85, transformer at V = 8 0.53 (m = 64);
  - fresh behavior 0.20–0.34 → 0.85–0.91;
  - population readout 0.83–0.86 and 0.89; untrained 0.82–0.83; vocabulary readout 0.21–0.24;
  - correlations: r = 0.96/0.64 and 0.73/0.40 [0.30, 0.48];
  - transformer 0.59/0.41;
  - auxiliary loss +0.14 at most (MLP 0.21 → 0.35);
  - width: m = 8 gives 0.67–0.71 vs ceiling 0.79; gap 0.08–0.14;
  - OCCLUDE population 0.75–0.83 and behavior 0.65–0.79 vs 0.85–0.89 and 0.85–0.91; state interaction 2–5%;
  - role–filler 0.94 vs 0.88–0.93; strong split ≤ 0.21 at V = 8; SD 0.23–0.29 (0.90) → 0.01–0.02.
- **E5:**
  - δ 0.02–0.04 / 0.06–0.13; reversal state R² 0.96–0.99 / 0.84–0.92;
  - reversal decoding 0.86/0.86 and 0.48/0.49; new role 0.81/0.81;
  - interchange 0.78/0.81, 0.79/0.80, random edit −1.12, 15% (V = 64);
  - OCCLUDE δ 0.11–0.22 (0.34), reversal 0.56/0.70, 31%, new role 0.76/0.68 (V = 64);
  - fine-tuning 0.98 → 0.90, 0.84 → 0.71, 0.97 → 0.60, 0.22 → −0.46; code variance 98%.
- **E3:** 0.26, 49%, HRR identification 1.00/0.97/0.66, R² 0.95/0.73/−0.21, w = 1024 0.69.
- **E4:**
  - references: concatenation 0.89–1.00, sum 0.41–0.50;
  - new pairs 0.82–1.00; GPT-2 0.78–0.84; T5-base 0.71; middle group 0.29–0.52; small CLIPs −0.35/0.01;
  - role assignment 0.86–0.98; identification 0.29–0.75; selection sensitivity ≤ 0.01;
  - "and" control for GPT-2-large 0.87/0.96 vs 0.84/0.95;
  - CLIP-H role 1/role 2 0.63/−0.06; reversal similarity 0.85–0.88.

**Mismatches and unstated conditions:**

| # | Location | Paper says | Aggregated output says | Correct statement |
|---|---|---|---|---|
| M1 | L208 | OCCLUDE input interaction "11–14% of composite variance" | per-seed `inter_input_fresh_r1` ranges 0.116–0.161, mean 0.141 (`e2v2occ`, 10 seeds); L220 itself says 14% | "12–16% (mean 14%)" |
| M2 | L254 | RAAM "R² ≥ 0.95 for w ≥ 2^n d" | depth 1, w = 16 = 2·8: fresh R² 0.56 (V = 16) and 0.76 (V = 64). Depth 2, w = 32: 0.95/1.00. Depth 1 at w ≥ 32: ≥ 0.99 | "R² ≥ 0.95 whenever w ≥ 32 (w > 2^n d at depth 1, w ≥ 2^n d at depth 2); at w = 2^n d = 16, depth 1 reaches only 0.56–0.76" |
| M3 | L236 | Interchange 0.78 vs 0.81, 0.79 vs 0.80, random edit −1.12, leakage 15% (no V given) | These are **V = 64 only**. At V = 16: 0.32 vs 0.40, 0.33 vs 0.32, random edit −1.01, leakage **33%** | Add "at V = 64" and give the V = 16 values; Fig. 5c caption must also say V = 64 (the script plots `d64`) |
| M4 | L236 | OCCLUDE: 0.56 vs 0.70; 31%; 0.76 vs 0.68 | V = 64 only. At V = 16: 0.17 vs 0.25; leakage **59%**; 0.23 vs 0.11 | Add "at V = 64" |
| M5 | Fig. 4c, dotted line; L216 | Text: untrained readout 0.82–0.83 (MLP/CNN/role–filler), 0.41 (transformer) | `fig_e2.py` plots `d.groupby("V")["init_fresh_1"].mean()`, i.e. the mean over all four architectures, ≈ 0.72 | The figure suggests training *raises* recoverability by about 0.12, which the text denies. Plot one untrained line per architecture (or a band at 0.82–0.83 plus the transformer line) |
| M6 | Fig. 4d caption | "follows the vocabulary readout … and not the population readout" | within-condition r for the population readout = 0.40 [0.30, 0.48], clearly > 0 | "follows the vocabulary readout more closely (within-condition r 0.64 vs 0.40)"; state that the printed r values are pooled over conditions and computed on values clipped at −0.2 |
| M7 | L218 / Fig. 4e | "raises fresh behavior from 0.47–0.64 to 0.81–0.82" | excludes the transformer (0.32 → 0.45) without saying so; Fig. 4e also omits it | Say "MLP, CNN and role–filler"; give the transformer values |
| M8 | L218 / Fig. 4e caption | held-out behavior "falls slightly" / "small cost" | 0.99–1.00 at V = 16 → 0.89–0.94 at V = 64 (CNN 0.987 → 0.893, MLP 0.987 → 0.885); transformer 0.91 → 0.60 | "falls by 0.05–0.10 (by 0.32 for the transformer)" |
| M9 | L232, Fig. 5 caption | "5 seeds each" | `agg_e5.txt` reports OCC n = 4–5 (one cell has 4 seeds) | Say "4–5 seeds", or complete the cell |
| M10 | L282 (Discussion) | "Coverage of combinations … is not the variable that matters" | At V = 64, m = 64: 140 pairs vs 2,420 pairs gives fresh 0.81–0.82 vs 0.85–0.91 (+0.04 to +0.09) and held-out 0.89–0.94 vs 0.96–0.99 | "matters less than the number of distinct parts, but still adds 0.04–0.09 on fresh parts at V = 64" |
| M11 | L218 | "Two controls confirm that distinct parts, not combinations …" | same data as M10 | "…that distinct parts matter beyond combinations" |
| M12 | Fig. 6d caption | "the dashed line is identification from the sum reference" | the line is the mean over 11 models (≈ 0.33). Per-model sum-reference identification ranges 0.12 (GPT-2) to 0.41 (CLIP-B). CLIP-B (0.29) and CLIP-L (0.34) fall **below their own** sum references (0.41, 0.39) | Mark per-model sum references (as in 6c), or say "mean over models" and report the CLIP comparison in the text; this actually strengthens the CLIP claim |
| M13 | L263 | "so role information survives even where the noun's code does not" | L265 shows the "and" control gives the same role assignment (0.80–0.97) | "order information survives" |
| M14 | L265 | causal encoders "favor role 1 under mean pooling" | GPT-2 s/m/l (all mean-pooled, causal) show no role-1 advantage: 0.77/0.79, 0.85/0.82, 0.84/0.84 | Note that GPT-2 does not show it; the asymmetry is specific to CLIP-H |
| M15 | Appendix L545 | q = 4 control: fresh behavior "saturating from m = 16" | `e2q`: m = 16 gives 0.82–0.86, m ≥ 32 gives 0.84–0.89 | "nearly saturating from m = 16 (within 0.04)" |
| M16 | Figs. 5c, 5d, 6a | (not stated) | bars and curves are clipped at −0.5 (random edit −1.12/−0.99; RAAM w = 32 at depth 3 = −1.03; at depth 4–5 down to −4.2) | State the clipping in the captions |
| M17 | L214 | held-out 0.85–1.00 "in every condition except" the transformer at V = 8 "(0.53)" | correct for OVER at m = 64; the transformer at V = 8 spans 0.53–0.61 over m | "(0.53–0.61)" |

---

## 3. New issues, ranked by severity

### Major

**N1. "Recoverable" is judged by different standards in E2 and E4, and E2 has no reference (L216, L263–L265; Def. 2 at L117).**

- *The inconsistency.* Def. 2 asks that ε be "no worse than those of a reference role–filler operator applied to the same part codes". E4 applies this test: GPT-2 at 0.78–0.84 is "below concatenation" and so not compositional. E2 does not: glyph-network states at 0.83–0.89 are declared "recoverable whatever their vocabulary", without any reference.
- *What a reference would show.* The only yardstick in E2 is the Eckart–Young ceiling, which sits 0.10–0.14 above the population readout at every m ≥ 16. The untrained network reaches 0.82. The plateau near 0.86–0.89 is plausibly the linear ceiling of the OVER *input* itself (two glyphs overlaid in the same 8×8 region). That would explain both the constant gap to the ceiling and why training adds almost nothing.
- *Fix.* Report two E2 references with the same population readout:
  1. the raw input composite (identity encoder);
  2. a role–filler reference (concatenated glyph pixels, or their random projection to width m).

  Then state the E2 verdict relative to these references, as Def. 2 requires, and use the same rule in E4. If the input reference is about 0.86, the claim becomes "the states keep everything the input makes linearly available", which is cleaner and still supports the failure-of-performance cell.

**N2. The interchange-intervention controls do not test what the text claims (L234, L236, Fig. 5c).**

- *(a) The random-edit number is a no-edit baseline.* The random edit is scored by the R² of the *donor* glyph a′. A random direction carries no information about a′, and two independent glyphs give R² ≈ −1. So −1.12 is essentially the no-edit value: it says nothing about whether the output was "destroyed" or whether the edit was local. The informative random-edit numbers are the untouched role's R² after the random edit and the right/left change ratio (cf. 15%). Neither is computed (`e5_structure.py` only stores `swap_rand_left`).
- *(b) The donor shares the untouched glyph.* The donor composite is h(a′, b), with the *same* role-2 glyph b (`h_a2 = enc(a2, fb, ...)`). Any leakage of b into ê₁ therefore cancels in the edit. Standard interchange interventions use a donor (a′, b′) with b′ ≠ b.
- *(c) The result is close to guaranteed.* With δ ≈ 0.03, B₁(ê₁(h(a′,b)) − ê₁(h(a,b))) ≈ h(a′,b) − h(a,b). The intervention is then nearly "replace h by the donor state". The new sentence at L236 concedes this and names leakage as the informative quantity. That makes it more important that leakage be compared against a calibrated control.
- *Fix.*
  - Use donors with b′ ≠ b.
  - Report the untouched role's R² and the change ratio for (i) a random edit of the same norm, (ii) a random edit *within range(B₁)*, and (iii) an edit along range(B₂). Report the no-edit baseline for the donor glyph.
  - Report V = 16, where leakage is 33% (OVER) and 59% (OCCLUDE), not 15%/31%.
  - Delete "destroys the output".
  - The Discussion's "fourth requirement" and the abstract's "the role components we recover are the ones the decoders use" should be qualified to V = 64 until this is done.

**N3. Fig. 4c misrepresents the untrained control (M5).** This is the key panel for the paper's main E2 claim, and its dotted line (≈ 0.72) contradicts the text (0.82–0.83). A reader will conclude that training improves recoverability by about 0.12 for MLP/CNN/role–filler, which is false at m = 64. Fix the figure.

**N4. The fine-tuning result lacks a control for generic forgetting (L238, Fig. 5d, Table 4 row "New parts without forgetting").**

- *What the data show.* At V = 64, role-2 rendering of the new glyph falls by 0.13 while old held-out pairs fall by 0.08. Five composites seen for 200 steps without replay is a textbook recipe for catastrophic interference whatever the representation. The data therefore do not show that the damage is *role-specific*.
- *A wrong premise.* "Even though their pre-trained decoder had rendered it in both roles" holds only at V = 64. At V = 16 the "before" values are 0.21/0.22.
- *Power.* There is one new glyph per seed and 5 seeds per cell.
- *Fix.* Add a control that fine-tunes on the new glyph in *both* roles (same number of composites and steps), and report role-2 damage *relative to* old-pair damage. Alternatively, test the paper's own prescription: add the glyph's code through the fitted B₂ without gradient steps and decode. Until then, soften "gradient descent on one role does not respect that structure" to "fine-tuning without replay damages both old pairs and the new part's untrained role".

### Moderate

**N5. The seed-SD "signature of contingent systematicity" (L220) is confounded by the size of the test set.**

- At V = 8 the strong split holds V//4 = 2 glyphs out of role 2, so `beh_strong` is a mean over 14 pairs built from 2 random glyphs.
- Its seed SD (0.23–0.29) mostly reflects *which two glyphs* were held out (test-set sampling), not training contingency in McCoy et al.'s sense, which used a fixed, large test set.
- At V = 64 the split has 16 glyphs and 1,008 pairs, so the SD would shrink for that reason alone.
- *Fix:*
  - Hold the number of held-out-role test glyphs fixed across V, or average several held-out-role draws per trained network.
  - Compare against the SD of the difficulty-matched held-out split at matched size.
  - Otherwise drop the citation to McCoy et al.

**N6. The comparison of within-condition correlations has no test, and its common cause is not separated (L216).**

- *Missing test.* 0.64 vs 0.40 is reported with a CI only for the 0.40. Give a bootstrap CI on the difference (paired over runs).
- *Common cause.* Both the vocabulary readout and the decoder are fitted on the same V glyphs, so their seed-level covariation may come from the *vocabulary draw* (how well 8 glyphs span the glyph manifold), not from the state.
- *Fix.* Add a vocabulary readout fitted on the *untrained* network, or on the raw input, over the same training composites. If it also correlates about 0.6 with behavior, the right reading is "the decoder's reach is set by the vocabulary's geometry", which is the paper's point stated more precisely.

**N7. E4 rankings depend on the metric, and the R² calibration problem remains (L263–L265; abstract).**

- *Calibration.* CLIP-B32 has R² −0.35 alongside identification 0.29 (29× chance). This is still the R² miscalibration from round 1. Noun-grouped CV for alpha and one scale-invariant metric (e.g., R² after an affine recalibration fitted on validation nouns, or mean per-target correlation) would settle it.
- *Ranking by identification.*
  - T5-base falls from 2nd tier (R² 0.71) to near the bottom (0.39).
  - T5-large rises to the top (0.75).
  - CLIP-H (0.60) ties GPT-2-small.
- *What still holds.* The abstract's "CLIP's text encoders recover them least" holds for CLIP-B and CLIP-L under both metrics, and holds *relative to their own sum references* (M12), but not for CLIP-H under identification.
- *Fix.* State the ranking per metric, and restrict the CLIP claim to the two smaller encoders, or to the comparison against the model's own sum reference.

**N8. E4 PCA target space is fitted on all 305 nouns, including the 100 fresh ones (appendix L553; `e4_pretrained.py` L175–177).** This is transductive. It guarantees that fresh-noun variance lies inside the 16-d target space, so it biases fresh R² *upward* for every system. It is probably small, but it contradicts "parts that played no role in fitting the readouts". Fit the PCA on fitting nouns only, or disclose it and report the fraction of fresh-noun variance captured under both choices.

**N9. E4 has no untrained (random-init) model control and no causal check.** E2's untrained-network control was the most informative addition this round. The same control for E4 (randomly initialized weights of each architecture, same tokenizer) is cheap and would show how much of the "new pairs 0.82–1.00" and "fresh 0.3–0.8" comes from the input embeddings and positional structure alone. It is also the Hewitt & Liang selectivity control I asked for. Given the conceded "order, not semantics" result, this baseline matters: a random transformer with positional embeddings may recover noun order well.

**N10. The role–filler decoder's fresh-part deficit is still unstated (L220, Discussion L282/L286).** At m = 64 its fresh behavior is 0.47 vs 0.58–0.64 (V = 16) and 0.71 vs 0.81–0.84 (V = 32), and its OCCLUDE fresh behavior is lowest at V = 64 (0.68 vs 0.76–0.79). The Discussion's "third requirement" and "Architectures that build in role-factorized slots move part of the guarantee into the architecture" read as endorsements. State the deficit; it is informative, because the architectural prior helps new roles and hurts new parts.

**N11. OCCLUDE is still described as "invertible in principle" (L208).** The host loses a 4×4 quadrant and the occluder is 2×-pooled. Pixel-level host recovery is impossible, and latent-level recovery is unshown. Qualify it ("the latents are plausibly recoverable; the pixels of the host are not"), and compare input relations on the BESIDE output only (`beh_fresh_r10`), which separates encoding from rendering an occlusion.

### Minor

- **N12. Prop. 4 at L216.** Replace "as Proposition 4 predicts for any procedure learned from composites" with "in the way Proposition 4 describes for affine procedures"; see the W10 row.
- **N13. Section 4 intro (L185).** "every number is a mean over seeds with standard errors in the figures": several in-text numbers (Fig. 4e transformer, E5 at V = 16, E4 per-code) appear in no figure. Add a supplementary table with means ± SE for every in-text number.
- **N14. Figure readability.**
  - Fig. 6d: the legend overlaps the tops of the green bars.
  - Fig. 5d: the annotation "solid V=64, dashed V=16" collides with the panel title.
  - Fig. 4d: the r annotation overlaps data points.
  - Fig. 4e: the legend (shared figure legend) lists the transformer although it is not plotted.
  - Fig. 6c: the CLIP-B bar runs into the legend.
- **N15. Artifact hygiene.**
  - Remove the stale E5 v1 block from `agg_r2.txt`, or label it superseded.
  - Fix the docstrings in `e4_pretrained.py` (80/40 nouns) and `agg_e4b.py` ("on" and "left").
  - State in Section 4.5 that the 3 E4 seeds are noun splits of one checkpoint per model.
- **N16. GPT-2 code robustness.** Report the GPT-2 part-code robustness (carrier code 0.75/0.81/0.82). It answers the natural objection that causal masking hands GPT-2 its own phrase states.
- **N17. Prop. 7.** The statement still mixes w = d with general w (round-1 m12).

---

## 4. Overall verdict

**Recommendation: minor revision, borderline major.** It becomes a clear accept-level paper (from my side) if the items marked must-fix are done. Score 6/10 (up from 4); soundness 3/4, presentation 3/4, contribution 3/4; confidence 4/5.

The revision addresses the core of my round-1 critique.

- **E2.** The population readout, untrained control, disjoint fresh sets and within-condition correlations reverse the paper's earlier "imitation" conclusion into a "failure of performance" conclusion, and this one is supported by the data.
- **E4.** It is now a properly split probing study, with held-out pairs, validation-based selection with a sensitivity check, multiple part codes and the "and" control. It honestly concedes that the readouts recover order, not relational semantics.
- **E5.** It adds a genuinely useful structural test.

Nearly all numbers match the aggregates. The mismatches are unstated conditions (V = 64, transformer excluded), one wrong range (M1), one overstated threshold (M2) and two figure/caption problems (M5, M12).

**Must change:**

1. Fix Fig. 4c's untrained line (M5/N3) and the Fig. 4d and 6d captions (M6, M12).
2. Qualify the E5 interchange numbers to V = 64, report V = 16, and replace or augment the random-edit control with locality-relevant measures (untouched-role R² and change ratio under random and range(B₂) edits), using donors with b′ ≠ b (N2). Remove "destroys the output".
3. Add an input/role–filler reference for E2 and apply Def. 2's relative standard to E2 and E4 alike (N1).
4. Soften the Discussion's coverage claim and the held-out "falls slightly" (M8, M10, M11). State the transformer exclusion (M7) and the role–filler fresh-part deficit (N10).
5. Add a both-roles control or soften the fine-tuning conclusion (N4). Fix or drop the seed-SD/McCoy argument (N5).
6. Correct M1, M2, M9, M13, M14 and M16.

**Should change:** noun-grouped CV plus a scale-invariant metric in E4, with per-metric rankings (N7); PCA on fitting nouns only (N8); a random-init E4 control (N9); a CI on the difference of the within-condition correlations and a vocabulary-readout-on-input control (N6); OCCLUDE on BESIDE output only (N11).
