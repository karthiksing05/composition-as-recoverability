# Story restructure plan (2026-11-11)

User's direction: the paper is technically lost. Tell one story: there are two kinds of composition,
recoverability dissociates them, and every test affirms that story in the terms we use to describe it.
Graphics follow the story's progression, colors are standardized, and there are more case studies.

## The story in five beats

1. **Two kinds of composition.** Weak (behavioral) composition: a system composes parts, and whatever
   decomposition it does comes from a procedure trained for that task. Strong (representational)
   composition: the composite also decomposes, so the parts stay recoverable from the state.
2. **Recoverability is the measure.** Can a declared readout return each part, in its role, from
   composites of parts held out from the readout? kappa puts the state on a scale from a blend (0) to
   a role-filler code (1). Held-out parts are essential, because a lookup passes on training parts.
3. **Recoverability dissociates the two.** The hypothesis is that behavior sits inside representation.
   We fill all four cells of the 2x2 with real systems, one case study per cell:
   - both: phi-4 / Qwen3 answer role questions, and the question state recovers the role
     (r = 0.94; patches flip the answer).
   - strong without weak: glyph MLP at V = 8 (state 0.85, own decoder 0.29); recursive composer at
     depth 3 (readout 0.99, decomposer 0.26); image encoders (quadrants 0.81-0.94, caption swaps at chance).
   - weak without strong under the declared readout: OLMo-2 7B (answers 0.77, ridge transfer 0.20,
     whole-state patches flip 0.77-0.85).
   - neither: GPT-2 / Pythia (answers follow word order, readout returns the last-mentioned noun).
4. **Decomposition is what the strong form adds.** Strong composition lets other procedures reuse the
   parts: a new decoder reads frozen states, role-code edits swap one part, written codes add parts
   without forgetting, depth training repairs the decomposer, and the question state gives a
   paraphrase embedding. Each item is one row of the repair table and one case.
5. **Where recoverability comes from.** Honest limits: in the glyph toys the input supplies it
   (entangled inputs, E20). Pretrained models build it beyond their input embeddings (E24/E24b), and it
   rises and falls in training (Pythia). New: Random Hierarchy Model input floor (E27).

## Section order (replaces the current behavior-section / representation-section split)

1. Introduction: the beats above in prose, plus Figure 1 (the map).
2. Related work: trimmed, unchanged in substance.
3. Two kinds of composition: definitions, kappa, behavior inside representation, role-factorization,
   the three consequences (theory stays as is).
4. Measuring recoverability: protocol, systems table, calibration (lookup vs held-out readout, ceilings).
5. Recoverability dissociates the two kinds: one subsection per cell of the 2x2, in the order
   both -> strong only -> weak only -> neither. Each subsection opens with its case study.
6. Decomposition is what strong composition adds: repairs and edits.
7. Where recoverability comes from: input baselines, scale and training curves, RHM.
8. Conclusion and limitations.

The current "behavior fails" results (lookup, new pairs vs held-out glyphs, adding parts, depth,
encoders, role answers) do not disappear: each one becomes the "weak" half of a case in sections 4-5.

## Figures, one step of the story each

| # | Figure | Beat | Panels |
|---|---|---|---|
| 1 | The map | 3 | 2x2 grid, each cell one real case: inputs, what the procedure returned, what the readout returned, with numbers |
| 2 | Two kinds of composition | 1 | avocado on/in armchair (existing), plus weak = compose only and strong = compose + decompose schematic |
| 3 | Measuring recoverability | 2 | protocol (existing, trimmed) + calibration: lookup passes every operator, held-out readout separates them at the ceiling |
| 4 | Strong without weak | 3 | glyph renders (decoder vs readout), state vs decoder over V, depth trees (decomposer vs readout), image quadrants case |
| 5 | Weak traces to strong | 3 | phi-4 case sentences, transfer vs accuracy (r = 0.94), patching flips |
| 6 | Weak without strong, and neither | 3 | OLMo-2 7B readouts vs patches; Pythia / GPT-2 word-order case |
| 7 | Decomposition at work | 4 | role-code edit renders, written codes vs fine-tuning, new decoder, paraphrase embedding at the question |
| 8 | Where recoverability comes from | 5 | input vs trained (glyphs, entangled), input embeddings vs LM states, pixels vs encoders, Pythia over training, RHM input floor |

Tables kept: systems (methods), repairs (beat 4), objective (beat 4). The 2x2 table becomes Figure 1.

## One color per kind of measurement, in every figure

- **orange**: weak composition, i.e. the system's own procedure (decoder, decomposer, answers, caption match)
- **blue**: strong composition, i.e. a declared readout of the state (R^2, kappa, transfer)
- **green**: decomposition put to use (new decoder, edits, written codes, patches, question-state embedding)
- **grey**: controls and references (random weights, input, chance, ceilings), dotted for reference lines
- **red**: only the blend endpoint and word-order outcomes in case studies

Secondary distinctions never use a new hue:
- filled marker = held-out parts; open marker = training parts or new pairs
- dark shade = trained weights; light shade = a second series of the same kind

## Wording fixes found on the way

- fig_case_real says "fresh nouns" / "fresh photographs": change to held-out.
- fig_beh_real (b, c) and fig_rep_codes (d) use grey for a non-control series: recolor.
- fig_beh_toy (a) colors the operator class: switch to filled/open and blue/grey by role-filler vs not.

## Open checks before drawing

- Per-example outputs needed for the case studies (phi-4 answers on specific sentences, OLMo-2 7B,
  Pythia): check whether the E18/E21 results store per-item predictions; otherwise rerun those items only.
