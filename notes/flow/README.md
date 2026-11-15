# Flow rewrite, 2026-11-05

Order: why composition matters → behavior shows only some benefits → related work → definitions (recovery by own procedures; recovery by a declared readout) → design → behavior fails (toy, then pretrained) → recoverable states succeed (toy, then pretrained) → conclusion.

## Vocabulary (one term per concept, in text, figures, appendix)
- **held-out part** (not "fresh"): held out from a procedure when no composite containing it helped fit that procedure. Sources: "primitive holdout" splits (Furrer et al.), "outside the training space" (Marcus 1998).
- **training part** (not "familiar"): a part in the vocabulary.
- **new pair**: unseen pair of training parts (not "held-out pair", which means a pair of held-out parts).
- **own procedure**: decoder, decomposer, answers, matching. **readout**: declared procedure. **procedure** covers both.
- **population readout / vocabulary readout**, **fitting pool** (not "probe pool"). "probe" only for prior work.
- **token-averaged state**, **noun states**, **quadrant state**, **question state** (final token of a prompt asking about a role); "sentence states" = states of the sentence alone.
- **canonical / converse**, **figure / ground**, **transfer**.
- **text-only encoders, image--text encoders, image encoders, language models**.
- **repair** (not "remedy"). **recovery / recoverable / recoverability**, metric **κ**; **δ** is the mechanism diagnostic only.

## Cut from the paper (kept in notes/appendix_archive.tex and notes/main_pre_flow.tex)
Modes of combination, capacities and human profile, threats-to-validity table, case-study figure, per-model protocol detail, three propositions (role collisions, substitutivity, content/context), old figures fig_calibration/networks/structure/depth_pretrained/real/design_real.
