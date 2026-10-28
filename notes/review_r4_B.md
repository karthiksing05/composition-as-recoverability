# Review R4-B: "Defining Compositional Representations by Recoverability" (fourth round)

Reviewer B: cognitive science and ML, judged against the venue style of Elmoznino et al. (2026, arXiv 2610.20922).

**What I read.**
- `recoverability-theory/main.tex` (596 lines, mtime 05:57). Line numbers below refer to this file.
- `references.bib`.
- The figure files `figures/fig_taxonomy.pdf`, plus `notes/fig_calibration.png`, `fig_fibers.png`, `fig_networks.png`, `fig_structure.png` and `fig_depth_pretrained.png`.
- The aggregates the E2, E4 and E5 numbers come from: `experiments/results/agg_r4_e4w.txt` (E4 with whitened codes, and the e4wd token-count variant), `agg_e2h_partial.txt` and `agg_r4_e2e5.txt`.
- `experiments/e4_pretrained.py`, lines 60–80, to see which checkpoints were loaded.

I ran nothing. The rendered page PNGs in `notes/pages` date from 00:58, which is before the current PDF, so I make no page-layout claims beyond the figures themselves.

## Verdict

**Minor revision, with three abstract-level corrections.** Score 7.5/10, unchanged.

Every required item from my round 3 is fixed in the text:
- A1 (Kamp & Partee) and A2 (Westerståhl);
- the pet-fish position (B1);
- "never seen before" (B2), but only in the text and captions;
- the taxonomy caption (B3), App. B (B4), the seed spread (B5), the cross-reference (B6) and the Discussion counts (B7).

The new E4, with whitened codes, random weights scored on both sets of codes, and a token-count variant, is a clear improvement. Its numbers match `agg_r4_e4w.txt` wherever I checked.

What remains falls into three groups:
- three claims at the level of the abstract or the story that the paper's own data contradict (M1–M3);
- a running example and two figures that contradict the paper's own results (M4, M5);
- a human-evidence paragraph that misclassifies its best fresh-part study and proposes a human test for the wrong column of Table 1 (M6).

---

## MAJOR

### M1. The paper uses two definitions of "compositional" and never says which one the title means (l.46, l.117, l.284, l.286)

- **What the abstract and Definition 2 say.** Compositionality is a property of states: "a representation is compositional when every part ... can be read back in its role by a fixed procedure of a declared kind" (l.46).
- **What the Discussion says.** It reserves the word for state plus procedure:
  - l.286: "the bare term should be reserved for systems whose parts are recoverable from composites of fresh parts **and whose own procedures use them**";
  - l.284: "A network that meets all five ... has earned the term". Requirements 2, 4 and 5 concern the system's own decoder.
- **Why this matters.** Under Definition 2, the glyph networks at V = 8 are compositional to κ = 0.73 (l.216). Under the Discussion, they have not "earned the term", because their decoder fails on fresh glyphs. A reader who finishes the paper cannot say whether those networks have compositional representations. The 2×2 of Table 1 is the paper's best device for this distinction, but the terminology does not follow it.
- **Fix.** Name the two notions once in Section 3 and use the names throughout. For example, after Definition 2 (l.120):
  > We call a system whose states meet Definition 2 *representationally compositional*, and one whose own procedures also read its parts on fresh parts a *compositional system*; Table 1 crosses the two.

  Then:
  - l.284, last sentence: "A network that meets all five ... is a compositional system in our sense; one whose states meet Definition 2 but whose procedures do not is representationally compositional only; one that meets only the behavioral tests generalizes compositionally."
  - l.46, last sentence: "We close with five requirements a network must meet before both its states and its procedures can be called compositional."

### M2. The abstract misattributes the recursive-composer result (l.46)

- **What the abstract says.** "In networks trained on glyph composites, and in a learned recursive composer, the states keep fresh parts recoverable, but the systems' own decoders read them only after training on many distinct parts."
- **What Section 4.4 shows.** The composer's decomposer recovers fresh leaves "almost perfectly whenever w ≥ 32" within the trained depths, with V = 16 (l.254). It fails at depth 3 whatever the vocabulary. Its limit is **depth**, not the number of distinct parts.
- **Fix:**
  > In networks trained on glyph composites the states keep fresh parts recoverable, but the networks' own decoders read them only after training on many distinct parts; a learned recursive composer keeps its leaves recoverable beyond its training depths, but its own decomposer fails one level deeper.

### M3. The CLIP and text-to-image claim is wrong for the one CLIP encoder that gains from training (l.46, l.267)

- **What the abstract says** (l.46, and almost verbatim at l.267): "the CLIP encoders that condition text-to-image models gain little from training beyond the length of each noun."
- **What the code loads** (`e4_pretrained.py:64–72`):
  - `clip-h14` is `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, the OpenCLIP text encoder of Stable Diffusion 2.x;
  - `clip-b32` is OpenAI ViT-B/32, which conditions none of the text-to-image models the paper cites.
- **What the data show** (`agg_r4_e4w.txt`).
  - CLIP-H gains about 0.30 in κ from training (0.355 against 0.051).
  - It keeps that gain when the token count is removed (κ 0.358 in e4wd).
  - The claim holds only for CLIP-L (Stable Diffusion 1.5) and CLIP-B/32. These are the two that fall below their random counterparts in the token-count variant: 0.457 and 0.466 against 0.526 and 0.561, which matches l.267.
- **Fix:**
  - l.46: "... GPT-2 gets about halfway, and the text encoder of Stable Diffusion 1.5 and the smallest CLIP encoder gain little from training beyond the length of each noun."
  - l.267, last clause: the same, or "the two smaller CLIP encoders, including that of Stable Diffusion 1.5, gain little ...". You may add that CLIP-H, the encoder of Stable Diffusion 2, gains about as much as T5. That sentence would be informative for the text-to-image community.

### M4. The running example uses the one blend that lookup cannot decode in role (l.78–80, Fig. 8 caption l.310)

- **What the Intuition section says** (l.80): "a lookup over the vocabulary recovers the parts of almost every sum. The sum then looks recoverable."
- **Why it is wrong under the paper's own measure.**
  - Under x + y the composites of (a, b) and (b, a) are identical. Lookup therefore returns the unordered pair, not the part in its role.
  - Proposition 1 says this, and Fig. 3a measures it: identification 0.52 for x + y (l.194).
  - Fig. 8a shows it too. The two dark vocabulary points on the line x + y = z, at about (−0.38, 0.66) and (0.66, −0.38), are a role-reversal pair. Even the familiar composite has two vocabulary decompositions.
- **Where the claim is repeated.** The caption (l.310) says "Few pairs of a finite vocabulary (dark points) fall on any one line, so the parts of familiar composites can be decoded". Its own panel contradicts it.
- **Two easy fixes:**
  - Run the example on weighted addition x + ½y, which is decodable by lookup (Fig. 3a, 1.00) and has a fresh-part ceiling of 0.8. Redraw Fig. 8a with the line x + ½y = z and a vocabulary in which only one pair lies on it.
  - Or keep x + y and write at l.80: "a lookup over the vocabulary recovers which two parts were summed, though not which role each filled; with unequal weights, x + ½y, it recovers both."
- **Placement.** Fig. 8 carries the Section 2 argument but sits in Appendix A (l.307). Move it into Section 2, for example as panels beside Fig. 1. As it stands, a reader of Section 2 has to jump to page ~20.

### M5. The taxonomy figure itself (Fig. 2) still contradicts the text (`fig_taxonomy.pdf`)

The caption was fixed in round 3, but the figure's own text was not. I read the PDF text layer.
1. **"Role–filler composition: parts recoverable, including parts never seen before".** This is the r3 B2 wording, and l.94 rejects it. Write "including parts the readout was not fitted on".
2. **"Blending: parts not recoverable, even from a vocabulary, or roles collide".**
   - Fig. 3a and Lemma 2 show the opposite for every blend that is injective on the vocabulary, for example the narrowest tanh layer at 1.00. Only symmetric blends fail on a vocabulary.
   - Write "fresh parts not recoverable; roles collide for symmetric blends".
3. **"Systematicity is necessary"** (the bottom-right box). This contradicts:
   - Table 1, where recoverable states coexist with failed behavior;
   - l.144: "a statement about what the states determine, not about what a system does with them";
   - l.288: "whether the network learns such states and a decoder that uses them remains contingent".

   Write "Reversals and new role assignments are fixed by the state".
4. **"slots and templates" under role–filler composition.** l.499 says that "whether the slots of a template are recoverable for fresh fillers is the test our definition proposes". Move them to a "?" or drop them.
5. **The placement of x + αy is inconsistent.**
   - Fig. 2 lists x + 0.5y under fixed-width superposition and "head–modifier" under blending.
   - Table 5 (l.489) defines head–modifier blending as x + αy.
   - Pick one. The calibration (l.190) treats weighted addition as superposition.
6. **Layout.** The bottom-left box's text runs past its left border, and the "Chunking" title overlaps its subtitle.

### M6. The human-evidence paragraph misclassifies its best fresh-part study and proposes a test for the wrong column (l.290, Table 7 l.538–544)

The paragraph is accurate in most details. Four points need correcting.

**(a) Lake et al. (2019) is a fresh-part study by construction.**
- Its primitives are pseudowords, new to every participant, taught from a few examples and tested in unseen combinations. That is freshness to the person's composition procedure in the sense of l.94.
- The paragraph instead opens with "almost none of it was gathered with parts that are fresh in our sense". It then files Lake et al. under "recombination of known parts", and Table 7 (l.540) labels it "Known part, new role".
- **Fix.** Move it to the paradigms sentence: "The paradigms that come closest teach a new item and test it in new combinations: the wug test ..., rule learning over novel syllables ..., syntactic bootstrapping of novel verbs ..., and few-shot learning of pseudoword instructions \citep{lake2019}." Relabel the Table 7 row "New part, new combinations".

**(b) The proposed human analogue tests the system's own procedure, not Definition 1.**
- l.290: "A human analogue of our test would teach a new word, object, or face and ask whether it can be read back from a scene or sentence in a role it was never shown in." When a person reads something back, the reading is done by the person's own procedure, which is the behavioral column of Table 1.
- The analogue of Definition 1 is a neural decoder fitted on some items and evaluated on items held out from fitting. That is cross-condition generalization, already cited at l.155 (Bernardi et al. 2020).
- The Frankland & Greene design could do this with nouns held out from decoder training. The paragraph notes that it did not ("over a closed set of nouns and verbs").
- **Fix.** Add: "Definition 1 has a direct neural analogue: a decoder of agent and patient fitted on some nouns and tested on nouns withheld from fitting; behavioral read-back tests the person's own procedure, the second column of Table 1."

**(c) "Fresh conjunctions" mixes the paper's two notions of novelty.**
- Illusory conjunctions and the binding limits of working memory concern new conjunctions of familiar features. In the paper's terms these are held-out pairs of familiar parts, not fresh parts.
- Hummel & Holyoak (1997) is a model (LISA), not evidence of costs.
- **Fix:**
  > illusory conjunctions \citep{treisman1982} and binding limits in working memory \citep{wheeler2002} show that people bind even familiar features into new conjunctions with limited capacity, as LISA's account of dynamic role binding predicts \citep{hummel1997}.

**(d) "Belief overrides validity" (l.290; Table 7 l.544) overstates Evans, Barston & Pollard (1983).**
- They found main effects of both logic and belief, and an interaction. Validity was not overridden.
- **Fix:** "beliefs about a conclusion bias judgments of its validity \citep{evans1983}".

### M7. The random-weight baselines for role assignment and identification are still unreported, and the text offers both measures as evidence (l.263, Fig. 6d)

- **What the text says** (l.263): "Every trained model assigns roles to fresh nouns far better than the sum reference (0.90–1.00 against 0.50) and identifies them among 100 candidates far above chance (0.65–0.94) and above its own sum reference (0.41–0.44)."
- **What the random-weight controls give** (`agg_r4_e4w.txt`, own codes):

  | Encoder | Measure | Random weights | Trained |
  |---|---|---|---|
  | CLIP-B | identification | **0.744** | 0.709 |
  | CLIP-B | role assignment | 0.899 | 0.895 |
  | CLIP-L | role assignment | 0.839 | — |
  | CLIP-H | role assignment | 0.817 | — |
  | GPT-2 (three sizes) | role assignment | 0.715–0.737 | — |

- **What this means.** Role assignment and identification are largely supplied by an untrained architecture with positional information. The relevant baseline is the random model, not the sum reference.
- This is round-3 C-B2. It still stands with the whitened codes.
- **Fix.**
  - Add random-weight markers to Fig. 6d.
  - Replace the sentence with: "Every trained model assigns roles and identifies fresh nouns far above its sum reference, but so do the same architectures with random weights (roles 0.72–0.90, identification 0.47–0.74); only recovery, and hence κ, separates trained from random."

### M8. Definition 2's informativeness clause is never applied, yet the paper issues a verdict at a tolerance (l.117, l.120, l.216)

- **What Definition 2 requires.** A system is compositional at τ only for a part code that is "informative at tolerance τ" (l.117).
- **What l.120 promises.** "we report this share wherever a system-independent description exists (the glyph latents of Section 4.2)".
- **What is missing.** No such number appears anywhere. I grepped for "informative" and "latent": the only reported share is that the 16 pixel principal components carry 98% of *pixel* variance (l.234), which is not recovery of z.
- **The verdict that depends on it** (l.216): "these states are compositional at a tolerance of about τ = 0.27 at V = 8".
- **Fix.** Either report the affine R² from the declared glyph code (pixels, or the 16 PCs) to the latent z on fresh glyphs, which is one line of computation from data the paper already has, or change l.120 to "we report this share in future work" and make l.216 "meet the recoverability and factorization clauses of Definition 2 at a tolerance of about 0.27".

### M9. The closest classical antecedent of the central argument is missing: Marcus on generalization outside the training space

- **The paper's central mechanism.** A procedure learned from composites is determined only on the span of what it was fitted on, so the number of distinct parts governs transfer to new ones (Proposition 5, l.142, requirement 2 at l.284).
- **The antecedent.** This is Marcus's (1998, *Cognitive Psychology*; 2001, *The Algebraic Mind*) "training space" argument: networks trained by backpropagation fail to generalize a universally quantified relation to items whose features lie outside the training space. The paper cites Marcus only for infant rule learning (l.290).
- **Why it matters.** For a cognitive-science reader, Proposition 5 and requirement 2 will read as a formal, readout-side version of Marcus's argument. Not citing it invites the charge that the point is old.
- **What is new here.** The paper adds the dissociation: the state is outside Marcus's verdict, because it keeps the fresh parts, while the decoder falls under it.
- **Fix.** One sentence in Section 3.3 after Proposition 5 (l.142):
  > This is a readout-side form of \citet{marcus1998}'s argument that networks trained by backpropagation do not extend a relation to items outside their training space; our networks show that the argument can hold of a decoder while failing for the state it reads (Section 4.2).

Several directly relevant works are **already in `references.bib` but uncited**:
- `yuksekgonul2023` (vision-language models behave like bags of words). This is the obvious citation for E4's CLIP results and for caveat 1 at l.267.
- `park2024hidden` (hidden capabilities present internally before they can be elicited). This is an ML precedent for the upper-right cell of Table 1.
- `lepori2023` (structural compositionality via subnetworks). This belongs in "Decodability and generalization".
- `frankland2020`, `russin2024` and `quiltydunn2023` belong in the People paragraph and the classical-debate paragraph for a Cognitive Science audience.

---

## MINOR

1. **l.282 and l.254 give different numbers.** "within 0.13 of the rank ceiling" against "within 0.14 of the ceiling". Use 0.14 in both.
2. **l.282 "were recoverable whatever their training" needs an exception.** Write "... whatever their training, except the transformer at small vocabularies (0.59 at V = 8)". l.216 states this exception.
3. **l.284, first requirement** ("a state narrower than the parts it combines loses them"). This is still unqualified (r3 B3, open), and l.194 and l.478 say sparse parts may escape it. Write "a state narrower than the dense parts it combines loses them under affine readouts".
4. **l.284, second requirement** ("a network trained on eight glyphs solves unseen pairs"). The transformer does not: 0.53–0.61 (l.214). Write "the MLP, CNN, and role–filler networks trained on eight glyphs solve unseen pairs".
5. **l.284 "each tied to a proposition and an experiment".** Requirements 4 and 5 cite no proposition. Add "(Proposition 7(iv))" to the fourth and "(Proposition 9)" to the fifth.
6. **l.238 and l.288: the code-writing result shows no harm, not that the code teaches anything.** "Writing the fresh glyph's code into role 2 ... renders it exactly as well as the untouched network does (0.75 against 0.75)". `agg_r4_e2e5.txt` gives `add_vs_before` = 0.000. Adding the code therefore costs nothing but teaches nothing the network did not already do; its advantage over fine-tuning comes from fine-tuning's damage. l.288's "a role-factorized state allows a part to be added by adding its code alone" should read "... allows a part to be placed in a new role by adding its code, without the damage that fine-tuning causes".
7. **Table 5, l.490 (r3 B1 leftover).** "Default inheritance only; overrides and emergent attributes form η" should match l.501: "Approximately, if default inheritance is role-factorized (assumed); overrides and emergent attributes form η".
8. **App. C title, l.504 (r3 B12 leftover).** "Capacities and the human reference profile" should read "Capacities and a qualitative human profile", to match l.290 and the Table 7 caption.
9. **Abstract jargon (l.46).** "a role–filler reference" and "part codes in which no single direction dominates" are undefined at that point. Write "a concatenation reference" and "whitened part codes". The abstract also omits caveat 1 of l.267, that the encoders' roles are word positions. One clause would prevent ML readers from assuming thematic roles: "... recover nouns in their sentence positions, but none comes near ...".
10. **l.57 and l.157, Elmoznino et al. (2026)** (r3 C-table, open).
    - Their definition concerns the compositional structure of an *object* (strings, images), not of a representation. At l.57 write "the novel structure in the best model of an object beyond the best models of its parts".
    - At l.157, "a fixed-width superposition over a small vocabulary compresses well" is asserted, not shown. Write "would plausibly compress well".
11. **Small r3 citation leftovers.**
    - Uselis et al. concern *vision* embedding models (l.57, l.153).
    - Table 6, l.518: write "at most 40.6% on SLOG", since SLOG is entirely structural.
    - Griggs & Cox (l.544) is not the canonical source for content effects. Add Johnson-Laird, Legrenzi & Legrenzi (1972) or Cheng & Holyoak (1985).
12. **Definition 2 now packs five notions into one box** (l.117): the definition itself, κ, tolerance τ, informativeness and productivity (r3 B13, open, and worse). Move informativeness and productivity into a third box. Elmoznino et al. state one notion per box.
13. **l.210: q is used before it is defined** ("around kq = 16, the dimension of a pair of latents"). Add "q = 8" at l.208.
14. **l.234 is a single Method paragraph of about 330 words.** Split it into the run-ins the paper uses elsewhere: *Role maps*, *Reversal and new role*, *Interchange and controls*, *Fine-tuning*.
15. **Prose details.**
    - l.120 opens with "Note that". Start with "Definition 2 is graded and relative."
    - l.185 "Methods are summarized below" is passive. Write "We summarize the methods below".
16. **Figures.**
    - **Fig. 3d.** The MLP bar for HRR is missing, presumably because it is at or below zero. Say so in the caption.
    - **Fig. 5a.** The y-axis runs to 1.5 although no bar exceeds 0.86, which wastes a third of the panel.
    - **Fig. 6c.** The legend crowds the top of the panel. A legend below the axis, as in Fig. 4, would free it.
17. **Section 4.5 never says that δ cannot be used.** The concatenation reference itself has δ up to 0.81 (l.572), so the δ ≤ τ clause of Definition 2 cannot be applied to the pretrained encoders. This is stated only in App. D. Add half a sentence after "whatever its residual δ" (l.263): "(the concatenation reference itself has δ up to 0.81 with 16 components, so δ is not informative here)".

---

## Claims checked against reported results

| Claim | Location | Verdict |
|---|---|---|
| Lookup decodes every injective operator; symmetric blends 0.52 | l.46, l.194 | Supported (Fig. 3a) |
| Fresh-part recovery matches the rank ceiling | l.46, l.194 | Supported |
| Dense VSA under affine readouts behaves like a lossy blend | l.46 | Supported, and correctly scoped |
| Glyph-network states recoverable; decoders need many parts | l.46, l.216 | Supported (transformer exception, minor 2) |
| Recursive composer's decoder needs many parts | l.46 | **Not supported** (M2) |
| None of eleven encoders near the reference; GPT-2 about halfway | l.46, l.263 | Supported (κ 0.44–0.50) |
| CLIP encoders of text-to-image models gain little beyond length | l.46, l.267 | **Wrong for CLIP-H, the SD 2 encoder** (M3) |
| Training raises κ by 0.51–0.61 (GPT-2), about 0.3 (T5, CLIP-H), 0.08–0.17 (BERT, CLIP-L), 0.04 (CLIP-B) | l.265 | Matches `agg_r4_e4w.txt` |
| Token-count variant: CLIP-B/L 0.47/0.46 against random 0.56/0.53 | l.267 | Matches e4wd, like for like |
| Causal-prefix numbers (0.74–0.78 / 0.08–0.34; 0.73–0.80 / 0.65–0.68; 0.71–0.78 / 0.44–0.55) | l.267 | Match |
| Requirement 4: spill 0.16 / 0.25 / 0.30 / 0.75 | l.284 | Match (`agg_r4_e2e5.txt`: 0.158, 0.252, 0.296, 0.752) |
| Fine-tuning excess 0.05 ± 0.02; V = 16, 0.22 ± 0.06 against 0.04 ± 0.02 | l.238 | Match |
| E2 vocabulary readout 0.23–0.29; r = 0.93 (0.50) against 0.73 (0.30); CI 0.09–0.32 | l.216 | Match `agg_e2h_partial.txt` (glyph rule) |
| Compositional at τ ≈ 0.27 / 0.17 | l.216 | Informativeness clause unreported (M8) |
| "Systematicity is necessary" | Fig. 2 | **Contradicted by Table 1 and l.144** (M5) |

## Status of round-3 points

| r3 item | Status |
|---|---|
| A1 Kamp & Partee | **Fixed** (l.501) |
| A2 Westerståhl | **Fixed** (l.57) |
| B1 pet fish | **Fixed** in text (l.501); Table 5 row open (minor 7) |
| B2 "never seen before" | **Fixed** in text and captions; **open inside `fig_taxonomy.pdf`** (M5) |
| B3 "only" in the taxonomy caption | **Fixed** (l.129); first requirement still unqualified (minor 3) |
| B4 App. B sparse claim | **Fixed** (l.478) |
| B5 seed spread | **Fixed** (l.288) |
| B6 dangling "second sense" | **Fixed** (l.259) |
| B7 Discussion summary | **Fixed**, except the 0.13/0.14 mismatch (minor 1) |
| B8 fourth requirement qualifier | **Fixed** (l.284) |
| B9 coverage claim scoped to glyphs | **Fixed** (l.284) |
| B10 reverse compositionality as fact | **Fixed** ("on many accounts") |
| B11 abstract length | **Fixed** (282 words, from about 360) |
| B12 "human reference profile" | **Mostly fixed**; App. C title open (minor 8) |
| B13 overloaded Definition 2 | **Open**, and worse (minor 12) |
| C-table: Uselis "vision", SLOG wording, Griggs, Elmoznino "object" / "compresses well" | **Open** (minors 10, 11) |
| W1(b) Chalmers 1990 | **Fixed** (l.59) |
| W3 Hummel & Holyoak | Cited, but as evidence rather than as a model (M6c) |

## Summary of required changes

1. **M1:** name the two notions (representationally compositional against compositional system) and use them in the abstract and the Discussion.
2. **M2, M3:** correct the abstract's recursive-composer and CLIP sentences.
3. **M4, M5:** fix the x + y running example and Fig. 8, move Fig. 8 into Section 2, and correct the text inside `fig_taxonomy.pdf`.
4. **M6:** reclassify Lake et al. (2019), give the neural-decoder analogue of Definition 1, and correct the "fresh conjunctions" and belief-bias wording.
5. **M7, M8:** report the random-weight role and identification baselines, and either report or drop the informativeness clause.
6. **M9:** cite Marcus (1998/2001) at Proposition 5, plus the in-bib Yuksekgonul et al. (2023) and Park et al. (2024).

None of these needs new compute, except optionally the one-line latent-recovery number for M8. With them done, I recommend acceptance.
