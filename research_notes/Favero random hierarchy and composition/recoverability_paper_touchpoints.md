# Recoverability paper: what an outside body of work must connect with, and candidate touchpoints for the Random Hierarchy Model line

Source keys, used in every citation below (all local files, read 2026-11-06; line numbers refer to the files as they stood that day):
- `RT` = /coc/flash10/ksingara3/compositionality-theory/recoverability-theory/main.tex ("Dissociating Representational and Behavioral Compositionality via Recoverability")
- `RB` = /coc/flash10/ksingara3/compositionality-theory/recoverability-theory/references.bib
- `SU` = /coc/flash10/ksingara3/diffusion-composition-how/paper/dataconds/main.tex ("Seen but Unnamed")
- `SB` = /coc/flash10/ksingara3/diffusion-composition-how/paper/dataconds/iclr2027_conference.bib
- `MEM/` = /nethome/ksingara3/.claude/projects/-coc-flash10-ksingara3/memory/ (treat as a possibly stale record)

Scope: this file describes only the user's side. It makes no claim about what RHM or Favero et al. show. The user's vocabulary rule: write "held-out" and "training" parts, never "fresh" or "familiar" (MEM/project_recoverability_theory.md:52).

## Definitions: recoverability, kappa, weak vs strong, question state

### Takeaway
Recoverability is the return of each part in its role from composites of parts held out from the reading procedure, scored under a declared readout class and calibrated between a sum and a concatenation of the same part codes. Behavioral (weak) and representational (strong) compositionality are the same recovery test read through the system's own procedure and through a declared readout.

### Cited Findings
- Recoverability, in prose: "We use \emph{recoverability}, the return of each part in its role from composites of parts held out from the reading procedure, to define the second property and to compare the two on the same parts." — `RT:70`
- Why held-out parts: "Held-out parts are essential, because a readout fitted on a few training parts decodes almost any combination of them, blends included." — `RT:70`
- Held out: "A part is \emph{held out} from a procedure when no composite containing it helped fit the procedure. A glyph can be held out both from a readout and from the network that encodes it, while a noun in a pretrained model can be held out only from the readout." — `RT:120`
- Parts: "A part is a draw from a distribution $\D$ over possible parts, such as glyphs, nouns or photographs." A part code is $e(x) \in \R^d$ and a composite's state is $h = g_r(x_1, \ldots, x_k) \in \R^m$. — `RT:118`
- Formal recoverability (Definition "Recoverable relations"): "Fix a part distribution $\D$, a part code $e$, and a readout class $\U$. A relation $r$, which a system realizes by $g_r$, is \emph{$\varepsilon$-recoverable} if for every role $i$ some $u_i \in \U$ satisfies, for parts $x_1, \ldots, x_k$ drawn independently from $\D$," `\frac{\mathbb{E}\,\lVert u_i(g_r(x_1,\ldots,x_k)) - e(x_i) \rVert^2}{\mathbb{E}\,\lVert e(x) - \mathbb{E}\,e(x) \rVert^2} \le \varepsilon`. "We estimate this quantity by fitting $u_i$ on composites of a \emph{fitting pool} of parts and evaluating it on composites of other parts, with errors in a declared inner product on part codes." — `RT:141-146`
- Score, as printed: `\kappa = \frac{\varepsilon_{\mathrm{blend}} - \varepsilon}{\varepsilon_{\mathrm{blend}} - \varepsilon_{\mathrm{ref}}}`, "where the same procedure gives $\varepsilon_{\mathrm{ref}}$ for the concatenation and $\varepsilon_{\mathrm{blend}}$ for the sum of the same part codes. The system is representationally compositional at tolerance $\tau$ if $\kappa \ge 1 - \tau$ for some such $e$ informative at $\tau$." — `RT:152-155`
- Link to $R^2$: "$1 - \varepsilon$ is the population $R^2$ of the best readout in $\U$." For a symmetric relation of identically distributed parts, "in population $\varepsilon_{\mathrm{blend}} = 1/2$, $\varepsilon_{\mathrm{ref}} = 0$ and $\kappa = 1 - 2\varepsilon$." — `RT:162`
- Scale: "$\kappa$ places a state's recovery on a scale set by its references, with 0 for the sum of the part codes and 1 for their concatenation." — `RT:235`. On pretrained models the pipeline recovers the concatenation reference with $R^2 = 1.00$ and the sum reference with 0.48. — `RT:450`
- Weak vs strong: "Behavioral compositionality is the weak version, in which a system composes parts, and any decomposition it performs comes from a procedure trained for that purpose. Representational compositionality is the strong version, in which a system composes parts and its composites also decompose. Recoverability is our heuristic for the strong version, and it asks whether parts read out easily from the internal states of a composite." — `RT:72`
- Behavioral definition: "A system is \emph{behaviorally compositional} with respect to a relation $r$ to the degree that its own procedures, such as a decoder, a decomposer or its answers to questions about the roles, give the correct output for composites of parts held out from those procedures." — `RT:130`
- Representational definition: "A system is \emph{representationally compositional} with respect to a relation $r$, relative to a declared readout class $\U$ and a declared class $\M$ of methods for finding part codes, to the degree that its realization of $r$ is $\varepsilon$-recoverable for some part code $e$ found under $\M$ that is informative about the parts" — `RT:150`
- Informative codes and productivity: "A part code $e$ is \emph{informative} at tolerance $\tau$ if an affine map of $e$ recovers a declared description $z(x)$ of the parts that does not depend on the system, with $R^2 \ge 1 - \tau$ on held-out parts. A relation is \emph{productive to depth $n$} if Definition~\ref{def:repcomp} holds when the parts are themselves composites of depth up to $n - 1$, with the state of each composite as its part code." — `RT:159`. A system-independent description "holds for glyph latents and fails for nouns and photographs." — `RT:164`
- Role-factorization, the mechanism and not a condition: `g_r = c + \sum_i B_i e'(x_i) + \eta` with $\mathbb{E}\lVert\eta\rVert^2 \le \delta\,\mathbb{E}\lVert g_r - \mathbb{E} g_r\rVert^2$ (`RT:197-201`); "$\delta$ diagnoses the mechanism behind recoverability and adds no condition to Definition~\ref{def:repcomp}." — `RT:207`
- Hypothesis: "We hypothesize that behavioral compositionality is a subset of representational compositionality, able to do some, and not all, of the things representational compositionality can do." — `RT:74`; proved only as Proposition "Behavior entails recoverability, and the converse fails" (`RT:555-562`), whose converse uses a constant procedure (`RT:561`).
- Naming collision the paper disclaims: Smolensky's weak and strong compositional structure is "a distinction among representations that differs from the weak and strong composition we introduced above." — `RT:94`
- Question state: "A test prompt ends with ``Sentence: a tulip on a barrel. Question: Which object is on top? Answer: the'', and the model's state at that final token is the \emph{question state}." — `RT:284`; read "at the layer and part code selected on validation nouns with both phrasings." — `RT:683`
- User's framing in memory: "behavioral = weak composition, "just composition"; representational = strong, composition and decomposition. Recoverability is the heuristic for strong." — `MEM/feedback_recoverability_overview.md:16`

### Inferences
- The brief's form $\kappa = (\mathrm{rec} - \mathrm{ref\_sum})/(\mathrm{ref\_concat} - \mathrm{ref\_sum})$ does not appear in the paper; the paper prints the error form. They agree algebraically if rec means $R^2$ and $R^2 = 1 - \varepsilon$ (my derivation from `RT:152` and `RT:162`). A report should quote the printed form.
- An outside model must supply four declarations before kappa exists: a part distribution, a part code, a readout class, and a held-out split (`RT:141-155`). Hierarchical data would also need a choice of which level counts as a "part".

### Gaps
- The Reply to Fodor and Pylyshyn (key `smolensky1987`) is recorded as unread; the weak/strong attribution rests on a secondary source (`MEM/project_recoverability_theory.md:54,62`).
- I did not run git (the assignment limits me to Read, grep and ls), so "uncommitted" rests on the brief and memory only.

## Experimental designs and main results

### Takeaway
Two designs: parts held out from the system (analytic operators, glyph networks, recursive composers) and parts held out only from the readout (pretrained text, language and image models). Behavior fails on held-out parts, additions, depth and roles, while states hold the parts. But the glyph networks' recoverability "belongs to the input", so the toy establishes only a state/decoder dissociation.

### Cited Findings
**Protocol.** Readouts fit on a fitting pool, validation parts choose penalties, layers and codes, and "held-out parts give every reported number"; every system also runs with random weights. — `RT:233`

**Analytic operators** (Gaussian parts in $\R^{16}$, 14 operators; `RT:259`)
- A lookup identifies role-1 with accuracy 1.00 for every operator injective on the vocabulary, including half-rank projections and HRR, and fails only for symmetric blends, 0.52. An affine readout of the half-rank projection recovers held-out parts at $R^2$ 0.47; an MLP readout at 0.11. — `RT:306`
- At $V = 128$, linear readouts give role-filler and full-rank projections 1.00, half-rank 0.47, $x + y$ 0.50, $x + 0.5y$ 0.80 (its ceiling), HRR 0.44 and MAP 0.47. — `RT:360`

**Glyph networks, plain rendering** (fixed random generator from $z \in \R^8$ to $8 \times 8$ glyphs; two glyphs on an $8 \times 16$ canvas; relations \textsc{beside}, \textsc{over}, \textsc{occlude}; MLP, CNN, transformer, role-filler decoder; $m$ 8 to 64; $V$ 8 to 64; 9 seeds; `RT:261-263`)
- New pairs of training glyphs: output $R^2$ 0.85-1.00, except the transformer at 8 glyphs. Held-out glyphs at $m = 64$: 0.20-0.34 at $V = 8$, rising to 0.85-0.91 at $V = 64$. — `RT:311`
- With 140 training pairs, moving from 16 to 64 glyphs raises held-out output from 0.47-0.64 to 0.81-0.82. — `RT:313`
- One MLP at $V = 8$ renders with $R^2 = -0.15$ while an affine readout gets 0.76. Population readout 0.83-0.86 at $V = 8$ and 0.89 at $V = 64$, $\kappa = 0.73$ and 0.83; latent recoverable from code at 0.92; transformer 0.59 at $V = 8$. — `RT:365`
- New decoder on a frozen $V = 8$ encoder, trained on 64 other glyphs: 0.25 to 0.75, against 0.88 end to end. — `RT:367`

**Pixel baseline (the key caveat).** "The input pixels themselves reach 0.91 and untrained networks 0.82--0.83, and a frozen untrained encoder supports an equally good decoder (0.75). In these small networks, the recoverability comes largely from the input, and training mostly shapes the decoder." — `RT:369`

**Entangled rendering** (three rounds of orthogonal rotation and $\tanh$ at gain 4, decoder targets stay in pixels; `RT:269`, `RT:689`)
- Input readout falls to 0.68. At $V = 8$ MLP, CNN and role-filler read at 0.54-0.55, "0.06--0.08 above untrained encoders and 0.13--0.15 below the input"; transformer 0.28, level with untrained; decoders $-0.42$ to 0.15. — `RT:371`
- Conclusion as written: "no condition we tried lifts a state above its input. The glyph networks therefore establish the dissociation between a state and its decoder, and the recoverability they keep belongs to the input." — `RT:371`
- Pre-fixed criterion failed: target of input readout at most 0.5 with output at least 0.6 at $V = 64$; calibration over gains up to 32, up to 6 rounds, widths 64 and 256 and 12,000 steps "found no mix that meets it". — `RT:689`
- Readout predicts a new decoder: $r = 0.87$ plain, 0.98 entangled. — `RT:373`

**Objective table** ($R^2$ on held-out glyphs at $V = 8$; MLP, CNN, role-filler; 5 seeds; `RT:425-435`)

| Objective | Plain: readout / own decoder / renderer | Entangled: readout / own decoder / renderer |
|---|---|---|
| None | 0.85 / 0.29 / --- | 0.54 / 0.13 / --- |
| Training glyphs | 0.84 / 0.36 / 0.52 | 0.54 / 0.17 / 0.22 |
| 64 other glyphs | 0.90 / 0.47 / 0.92 | 0.66 / 0.39 / 0.70 |

- Stated confound: "The 64 extra glyphs also give the encoder more data, and we built the renderer by hand, so this result leaves open whether the objective beats plain training on the same parts." — `RT:422`

**Additions and role codes**
- Fine-tuning on a held-out glyph in role 1 at $V = 64$: role 2 falls 0.75 to 0.62, new pairs 0.98 to 0.90; sequential 32 glyphs from $V = 16$ drops old pairs to 0.18. — `RT:318`
- Under \textsc{over}, $\delta$ 0.02-0.04 (0.06-0.13 transformer); predicted reversals render 0.86 vs true 0.86; interchange 0.78 vs 0.81, spill 0.16 vs 0.90 natural direction; under \textsc{occlude} $\delta$ 0.11-0.22, spill 0.31. — `RT:389-391`
- Written codes: old pairs 0.99 vs 0.18, new glyph in new role 0.71; replay beats by 0.11-0.21 on new glyphs. — `RT:416`

**Depth** (balanced binary trees, Gaussian leaves in $\R^8$, RAAM trained on depths 1 and 2, HRR control; `RT:271`)
- Decomposer at depth 3, $w = 256$: 0.26; population readout of the same state: 0.99; depths 4-5 within 0.14 of the ceiling. HRR unbinding falls 0.96 to $-0.13$ across depths 1 to 5. — `RT:323`, `RT:377`
- Joint depth training: 1.00, 1.00, 0.97 at depths 3-5; decomposer alone 0.74, 0.29, 0.09; "no condition transfers to depth 6." — `RT:418`

**Pretrained models** (305 nouns split 165/40/100; Tiny-ImageNet 200 classes split 120/30/50; `RT:276`)
- Text encoders: GPT-2 $\kappa$ 0.44-0.50, CLIP-H 0.36, others 0.06-0.26; random $-0.11$ to 0.15. Random-weight causal models already recover the first noun at $\kappa$ 0.50-0.58; "a $x$ and a $y$" gives the same $\kappa$, "so the readouts recover the order of the nouns." — `RT:450-452`
- Language models: Pythia $\kappa$ 0.27 at 160M to 0.56-0.58 at 6.9B and 12B; Qwen3-8B, OLMo-2, phi-4 0.40-0.49. Pythia-1.4b "peaks at 0.71 after 1{,}000 steps, 0.7\% of training, and declines to 0.47", and six runs of Pythia-410m peak at about the same point. — `RT:454`
- Image encoders: $\kappa$ 0.17-0.49 token-averaged, 0.81-0.94 quadrant; \textsc{overlap} 0.85-0.94 vs 0.29-0.72 random. — `RT:456`
- Behavior: CLIP-B/32, CLIP-H/14, SigLIP identify objects 0.88-0.97, places at chance 0.50-0.51; text encoders match converse paraphrases 0.02-0.15. — `RT:328`. Pythia and GPT-2 answer role questions at 0.41-0.56, at every checkpoint of Pythia-1.4b; Qwen3-8B and phi-4 0.99, OLMo-2 13B 0.92, 7B 0.77. — `RT:340-342`
- Transfer: token-averaged state 0.00-0.05 vs 0.15 at initialization, "so training strengthens the binding to word order"; kernel and MLP readouts 0.00-0.06. — `RT:470-474`. Question state: 0.98 phi-4, 0.87 Qwen3-8B, 0.73 OLMo-2 13B; 0.07-0.10 GPT-2 and Pythia. — `RT:479`. $r = 0.94$ (95% interval 0.82-0.99), within-model 0.88. — `RT:481`

**Input baselines, pretrained.** Averaged input embeddings $\kappa$ $-0.21$ to $-0.49$; concatenated noun embeddings $-0.12$ to 0.49, token-averaged states exceed it in five of seven models; raw pixels $-0.6$ to $-0.9$ vs quadrant 0.81-0.94. — `RT:465`

**Interventions** (target "a $x$ on a $y$", source "a $x$ under a $y$"; `RT:288`)
- Whole final-token patch flips 0.99 phi-4, 0.84 Qwen3-8B, 0.80 OLMo-2 13B; same-figure source leaves answers 0.78-1.00; first half of layers at most 0.21. — `RT:483`
- 16-dimensional readout subspace flips at most 0.06; everything except it flips 0.67-0.98; top 64 difference directions 0.76, 0.75, 0.51; random 64 at most 0.03; everything except them at most 0.12. — `RT:492`

**Invented words** (93 nonce words; `RT:288`, `RT:691`). Answers 1.00, 0.98, 0.79, 0.71 (phi-4, Qwen3-8B, OLMo-2 13B, OLMo-2 7B) vs 0.54 GPT-2 and Pythia; transfer 0.75, 0.66, 0.53, 0.41; token-averaged $\kappa$ below 0 in every model; an invented word takes about 2.5 tokens vs 1.1. — `RT:494`

**Lower-left cell.** OLMo-2 7B answers 0.77, ridge transfer 0.20; whole-state patches flip 0.77 and 0.85; readouts at most 0.51 (0.56 with test-set hyperparameters). — `RT:496`

**Repair of sentence embeddings.** Role-question embedding places the converse paraphrase nearest for 100%, 92%, 83% of held-out pairs; fails in Pythia-1.4b and GPT-2-large. — `RT:498`

**Internal inconsistencies I noticed (flag, not resolved).**
- RAAM depth 3 at $w = 256$ is 0.26 in `RT:323` and `RT:377` and 0.27 in Table `RT:408`.
- The $V = 8$ own decoder is 0.25 in `RT:367`, 0.29 in `RT:433` and 0.20-0.34 in `RT:311`. These may be different averages, as `RT:433` excludes the transformer; the text does not reconcile them.
- `RT:82` says "up to 15 billion parameters"; the largest sizes the text states or names are Pythia's 12 billion (`RT:340`, `RT:454`) and OLMo-2 13B (`RT:280`), and no line gives a 15 billion model.

### Inferences
- Any outside comparison must respect the paper's own ceiling on toy claims: glyph networks may be cited for the state/decoder dissociation and not for learned recoverability (`RT:371`; `MEM/project_recoverability_theory.md:57` "User decided: toy = dissociation only").
- Memory numbers differ slightly from the tex (0.76 vs 0.75 new decoder, `MEM/project_recoverability_theory.md:27` vs `RT:367`). Use the tex.

### Gaps
- Figures and the compiled PDF were not read; `Read` cannot render PDFs here because pdftoppm is not installed, so figure-only values are not covered. Raw result files were not re-checked: numbers are quoted from the text.

## Current related-work coverage

### Takeaway
The recoverability paper cites none of Favero, Wyart, Cagnetta, Sclocchi, Kamb and Ganguli, or Okawa. Kamb and Okawa exist as orphan entries in its bib. Seen but Unnamed cites Kamb and Okawa but does not cite Favero or Cagnetta: their entries remain in its bib as uncited, after a recorded user decision to drop them.

### Cited Findings
**Recoverability paper**
- A case-insensitive grep of `RT` for favero, wyart, cagnetta, sclocchi, kamb, ganguli, okawa, biroli, mezard, ambrogioni, and random hierarchy returned nothing. The only matches among diffusion-related terms are `RT:96` (hinton1990 "part--whole hierarchies"), `RT:280` (rombach2022, for text encoders that condition Stable Diffusion) and `RT:687` (the Stable Diffusion 1.5 text encoder). — `RT`
- Grep of `RB` for favero, wyart, cagnetta, sclocchi: no entries. Grep of the project's notes, experiments and paper directories (py, md, tex, bib, txt, sh): no files. — `RB`; /coc/flash10/ksingara3/compositionality-theory/notes, experiments
- In `RB` but uncited in `RT` (grep count 0): `kamb2025` (`RB:215`), `okawa2023` (`RB:570`), `liu2022composable` (`RB:305`), `park2024hidden` (`RB:838`), `uselis2026` (`RB:728`).
- Earlier versions did cite them: notes/main_pre_flow.tex:63 cites okawa2023 and notes/appendix_archive.tex:59 cites kamb2025. — /coc/flash10/ksingara3/compositionality-theory/notes
- Pruning is on record: "Citations pruned 110 -> 71". — `MEM/project_recoverability_theory.md:54`
- Text-to-image claims in `RT` use ramesh2021, conwell2022 and rassin2022 only (`RT:64`, `RT:76`).
- Closest cited recovery work: probes, tensor-product decompositions, interchange interventions, compression measures (andreas2019, elmoznino2024, elmoznino2026), identifiability (lachapelle2023, wiedemer2024), and montero2021. — `RT:98`. The two-by-two table cites vegner2025. — `RT:178`
- The paper's own novelty note on diffusion: "Park 2024 separates *capability* (shown by latent intervention) from *behavior* (shown by prompting). That is an empirical version of the behavioral/representational split." — /coc/flash10/ksingara3/compositionality-theory/notes/literature_novelty.md:34

**Seen but Unnamed**
- Cited: `kamb2025creativity` (`SU:53`, `SU:126`), `okawa2023compositional` (`SU:58`, `SU:130`), `bradley2025local` (`SU:54`, `SU:99`, `SU:123`, `SU:136`, `SU:457`), `bradley2025projective` (`SU:122`, `SU:1233`), `liu2022composable` (`SU:91`, `SU:114`, `SU:1097`).
- Not cited: grep counts for favero or cagnetta are 0 in main.tex, main.aux and main.bbl of /coc/flash10/ksingara3/diffusion-composition-how/paper/dataconds.
- Entries present and uncited: `favero2025compositional` (`SB:33`), `cagnetta2024rhm` (`SB:40`), `liang2024factorize` (`SB:46`), under the comment "structured-data theories of composition" (`SB:32`).
- Recorded decision: "Dropped as low-value: smolensky, marcus, ... cagnetta, ... park, liang, favero, udandarao (entries remain in the .bib, uncited)." — `MEM/feedback_paper_prose_rules.md:40` (user, 2026-09-23)
- An earlier draft cited them for identifiability: "related work asks when compositional generalization is identifiable from the data and the decoder at all \citep{liang2024factorize,favero2025compositional,cagnetta2024rhm,wiedemer2024compositional}." — /coc/flash10/ksingara3/diffusion-composition-how/docs/review1_rewrite.py:131-133
- Repo plan's own summary of Favero (user-side record, unverified here): "Diffusion on PCFGs learns composition rules with the sample complexity of clustering features with statistically similar contexts, hierarchically; intermediate data sizes yield locally but not globally coherent samples; coherence length grows with data. Explicit renormalization-group analogy." — /coc/flash10/ksingara3/diffusion-composition-how/PLAN.md:110. The same list has "Sclocchi et al. 2025 (PNAS). Phase transition in diffusion reveals hierarchical data structure." — PLAN.md:111
- Sclocchi also appears in the synthesis: "This refines rather than contradicts the phase-transition picture (Sclocchi et al. 2025; Biroli et al.)". — /coc/flash10/ksingara3/diffusion-composition-how/docs/SYNTHESIS.md:134. There is no Sclocchi-first key in `SB`.
- The mechanism paper cites Kamb, Okawa, Park and Wiedemer and not Favero. — /coc/flash10/ksingara3/diffusion-composition-how/paper/mechanism/main.tex:459, 486, 496, 512
- Favero's paper is on disk: /coc/flash10/ksingara3/composition-creativity.pdf has PDF title "How Compositional Generalization and Creativity Improve as Diffusion Models are Trained" (grep of PDF metadata). Memory says the ICLR paper was "structured after Favero et al. (composition-creativity.pdf)". — `MEM/project_diffusion_composition.md:32`
- A further user repo mentions the terms: /coc/flash10/ksingara3/diffusion-concept-trees/docs/ALTERNATIVES.md matches favero or "random hierarchy" (file name from grep only; content not read).

### Inferences
- Adding Favero to Seen but Unnamed reverses a recorded user choice, so it needs a reason beyond completeness. The older draft's "identifiable" gloss for these papers was never verified in the material I read and should not be reused without the other researcher's check.
- The memory note that the ICLR paper is "structured after" Favero while citing nothing of it is a mismatch a reviewer could notice, though it concerns format and the structure note dates from August.

### Gaps
- The submitted ICLR PDF (/coc/flash10/ksingara3/_ICLR_2027__Diffusion_Composition_Generalization.pdf) could not be read. Its bibliography is inferred from the current tex, which may differ from the submission. Title evidence shows drift: `SU:14` reads "How Caption Coverage Affects Composition", while memory records "How Caption Completeness Affects Composition" as final (`MEM/project_diffusion_composition.md:19`).
- A grep of the top-level docs of diffusion-cobweb-updates and cobweb-forest found nothing, but I did not sweep the other repos under /coc/flash10/ksingara3 (no recursive search of the whole tree, on a shared login node).

## Open gaps the paper names, and claims without a theory of when recoverability arises

### Takeaway
The paper's theorems give ceilings and necessary conditions (what a state or an affine procedure cannot do). It has no theory predicting when training produces recoverable states, how much data that takes, or how it changes over learning. Its data have no generative hierarchy with known sample complexity, and its diffusion content stops at text encoders.

### Cited Findings
**Limitations as stated.** "The inclusion of behavior in representation is a logical result for a readout class that contains the system's procedure, and the evidence that it is strict comes from the systems we tested. Our evidence for design comes from repairs and from a recovery objective that also adds data, and separating the objective from the data is the next test. Our trained networks are small and their domains rendered, chosen so that we could hold out parts, and their recoverability comes largely from their inputs. In pretrained models the parts are held out only from the readout, apart from invented words. Richer methods, such as a discovered tensor-product decomposition \citep{mccoy2019}, may find structure our readouts miss. A human study could ask whether people show the same gap, by testing whether they recover the parts of composites built from newly learned parts." — `RT:510`

**What is proved (scope matters).**
- Non-identifiability of linear blends: `RT:567-574`.
- Exact ceiling, Gaussian parts and linear relations: `RT:580-590`.
- Eckart-Young ceiling, any relation, affine readouts only: `RT:595-609`. "it does not bound nonlinear readouts" (`RT:609`).
- Span: fitted affine procedures are determined only on the span of fitted composites; held-out recovery by every exact fit needs "at least $d + 1$ distinct parts in every role". — `RT:611-613`. This is the paper's only sample-count result.
- Unbinding under role-factorization: `RT:627-634`; approximate unbinding, whose bound is measured but "does not evaluate" against data: `RT:639-652`.
- Depth: linear composition, rank at most $\min(w, 2^n d)$, plus affine bound for any relation: `RT:654-664`. Recursive HRR crosstalk is a calculation with stated approximations: `RT:667`.

**What is empirical only, with no explaining theory in the paper.**
- Whether training creates recoverability: "no condition we tried lifts a state above its input" (`RT:371`), and the failed criterion (`RT:689`).
- Nonlinear decoders' dependence on distinct parts is argued by analogy: "a nonlinear decoder learned from composites is likewise unconstrained outside the region its training composites cover, and the mechanism is the same." — `RT:621`. Evidence is empirical (`RT:313`).
- The Pythia kappa peak at 0.7% of training and its decline, replicated over seeds, gets no explanation (`RT:454`). The paper also says recovery "tracks neither size nor length of training" (`RT:454`).
- Training strengthening word-order binding (`RT:470`), and role transfer at the question state following accuracy (`RT:479-481`).
- Strictness of the inclusion: the proof uses a constant procedure (`RT:561`); the substantive evidence "comes from the systems we tested" (`RT:510`).
- Recovery objective benefit: confounded with data (`RT:422`).
- Depth failure of a trained decomposer, and no transfer to depth 6 (`RT:418`). Proposition `prop:depth` gives a ceiling, and the decomposer falls far below it.
- The five "requirements for a compositional network" are labelled as suggestions: "Our results suggest five requirements" (`RT:506`).

### Inferences
Where a generative model with known hierarchy and known sample-complexity theory could help on the paper's side (my reading; whether RHM can deliver any of this is for the other researcher):
- It would give a ground-truth description $z(x)$ for the informative-code requirement, which nouns and photographs lack (`RT:159`, `RT:164`).
- It would give an input whose recoverability is not trivially high, addressing the failure of the entangled mix (`RT:371`, `RT:689`), which is the paper's sharpest open problem.
- It would replace the $d + 1$ affine span count (`RT:613`) with a data-scaling prediction to test against kappa.
- It would give an a priori expectation for learning-time curves, currently unexplained at `RT:454`.

### Gaps
- The paper does not state a sample-complexity question as an open problem; the data-scaling framing is my inference from `RT:313` and `RT:621`. Its named next test is separating objective from data (`RT:510`).

## Candidate touchpoints, as neutral questions with the paper's side answered

### Takeaway
On the paper's side, the most open slots are: a generative data model with varied hierarchy, parts defined at several levels, diffusion denoisers scored with kappa, and joint state-and-behavior curves over learning time and data size in systems trained from scratch. The main obstacles are the paper's finite-vocabulary loophole and its continuous part codes.

### Cited Findings
| # | Question | Paper's side |
|---|---|---|
| 1 | Does it have a controlled generative model where hierarchy depth varies? | Partly. Recursive composers use balanced binary trees of varying depth $n$, but depth lives in the composer, with Gaussian leaves and no rule structure in the data (`RT:271`). Glyph composites are flat, two glyphs on a canvas, and the generator is one fixed random map (`RT:261`, `RT:677`). |
| 2 | Does the data contain structure in which several lower-level configurations share one higher-level identity? | Not in the data. Many-to-one appears only as a property of the relation, as in blends (`RT:122`, `RT:568-574`). Parts are drawn independently (`RT:141`). |
| 3 | Are intermediate constituents scored, or only leaves? | Only leaves in experiments: "a population readout maps the same root state to all $2^n$ leaves" (`RT:271`). The definition of productivity does cover intermediate composites as parts (`RT:159`). Depth results are reported in $R^2$, not $\kappa$ (`RT:377`). |
| 4 | Does it test diffusion models? | No. No denoiser or sampler is scored. Stable Diffusion appears only through its text encoders (`RT:280`, `RT:687`), and text-to-image failures are only cited (`RT:76`). Grep of experiments/*.py for unet, denois, diffusers, ddpm found no files. |
| 5 | Does it test across learning time? | State across time: yes, for Pythia-1.4b (ten checkpoints), Pythia-410m and five PolyPythia runs (nine each) (`RT:280`, `RT:454`). Behavior across time: only "every checkpoint of Pythia-1.4b", flat at chance (`RT:340`). Systems trained from scratch report no time curves, with fixed 4,000 and 6,000 steps (`RT:677`, `RT:679`). |
| 6 | Does it test across training-set size, with a fitted scaling? | Sweeps $V$ 8 to 64 and a fixed-pairs comparison (`RT:263`, `RT:313`), with no fitted exponent or threshold in the text. |
| 7 | Is there a theoretical expectation for recoverable states before training? | Only ceilings and the span condition (`RT:580-613`). |
| 8 | Is there a known ground-truth part description? | Glyph latents, $R^2 = 0.92$ (`RT:365`). None for nouns or photographs (`RT:164`). |
| 9 | Are parts held out from the system, not only the readout? | Yes for toys (`RT:224`, `RT:120`). For pretrained models only invented words, with a tokenization caveat (`RT:494`, `RT:510`). |
| 10 | Can parts be discrete, from a finite vocabulary? | Eq. 1 uses squared error on continuous codes (`RT:143`). Lemma: a lookup decodes any relation injective on a finite vocabulary (`RT:541-548`). Span needs at least $d + 1$ parts (`RT:613`). |
| 11 | Is the readout class fixed to something a hierarchy could need? | Protocol is ridge (`RT:233`); kernel and MLP readouts appear for role transfer (`RT:474`, `RT:685`). "Behavior beyond the declared readouts" is an explicit table cell (`RT:186`, `RT:496`). |
| 12 | Is the role structure asymmetric with order and meaning separated? | Yes in language via converse paraphrases (`RT:282`). The analogous operation for hierarchical data is not defined. |
| 13 | Does noise level or scale enter? | No. There is no noise-level or length-scale variable in `RT`. Seen but Unnamed has noise levels (`SU:1236`). |

**Related paper: Seen but Unnamed**
- Location: /coc/flash10/ksingara3/diffusion-composition-how/paper/dataconds/main.tex; memory places it at ICLR 2027, submission 45331, reviews out, "revision and rebuttal close Nov 18, 10-page limit". — `MEM/project_diffusion_composition.md:11`
- Central claim: "Our claim is that a model follows a prompt that names several objects only when enough of its captions name several objects together." — `SU:239`. Caption coverage is "the fraction of multi-object training images whose caption names at least two of their objects". — `SU:29-30`
- Hypotheses H1-H3 and the control: `SU:248-256`. Opposing view: the "independence hypothesis" (`SU:244`).
- Numbers (three-object prompts, low / high / ceiling): ShapeWorld tokens 0.119 / 0.932 / 0.994; text 0.054 / 0.357 / 0.994; CLEVR 0.725 / 0.857 / 0.867; MS-COCO 0.390 / 0.512 / 0.693. — `SU:304-307`
- Summing single-object predictions places 0.951 vs 0.119. — `SU:433-435` (subsection at `SU:429`). "Its captions therefore taught it each object, and it fails to read several from one prompt."
- MS-COCO: captions name 27.6% of categories, 13.9% name two or more. — `SU:83-84`
- Scope limit: $64\times64$ rendered scenes of at most three objects "with position as the only relation". — `SU:487-488`
- Mechanism study, memory only and paused: "Composition = addressing x binding"; cross-attention carries it; key holds position, value holds colour. — `MEM/project_diffusion_mechanism.md:16-40, 84`
- User stance: the user rejected narrowing "necessary", and "composition is representing several things together, so keep the broad claim". — `MEM/project_diffusion_composition.md:11`
- Open issue, memory only and not re-derived here: the Figure 7c seed null "holds only at low sigma", with Spearman at sigma=0.5 of locality -0.286 and at sigma=5 of +0.864, and "main.tex:1236 says properties are averaged over noise levels, but make_figure_null.py uses sigma=0.5 only". `SU:1236` does say "average it over noise levels". — `MEM/project_diffusion_composition.md:13`
- Self-citation constraint: "anonymous2026seen (under-review self-cite) REMOVED from paper+bib 2026-10-27 for anonymity" (`MEM/project_recoverability_theory.md:20`); grep of `RT` and `RB` for it and for "seen but" finds nothing.

### Inferences
- Seen but Unnamed scores prompt accuracy, which in the recoverability vocabulary is behavior (`RT:130`). I found no state readout scored with $\kappa$ there. Memory records attention routing and key/value swaps, which are closer to the interchange tests of `RT:391`. So the two papers meet at one untested question: does a low-coverage denoiser state hold the objects it fails to place? Summing predictions (`SU:433-435`) is suggestive of this but is a procedure, not a declared readout of a state.
- The toy's blocker (recoverability inherited from the input, `RT:371`) and a data model with known depth are the most direct meeting point. Whether the input of such a model is hard enough is a question to check on the RHM side, and table row 10 is the likely technical friction.
- Any noise-level touchpoint touches the unresolved sigma issue above, so it should not lean on the seed-null claim until the user decides.
- Timing: the ICLR revision closes Nov 18 with the user aiming for "minimal paper changes" (`MEM/project_diffusion_composition.md:11`), so the recoverability paper is the lower-friction place for new integration.

### Gaps
- I did not read RHM or Favero papers, by instruction, so no row claims a match; each row only states what the user's paper has.
- `SU` was read in part (title, abstract, introduction, related work, hypotheses, part of results, conclusion, selected appendix lines). I did not read figure content or the full appendix.
- `docs/PREREG_REBUTTAL_2026-11-06.md` was only grepped (T1 row and thresholds), so running-job results are unknown.
