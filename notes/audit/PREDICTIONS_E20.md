# Predeclared criteria for the strengthening tests (2026-11-05, written before any run)

Seed-level statistics; "clear" = the difference exceeds 2 standard errors of the paired seed difference.

## E20: toy with entangled inputs (test 1)
Inputs pass through a fixed nonlinear mix, chosen in calibration so that an affine readout of the
input recovers little (target at most 0.5) while a network trained on 64 glyphs can still render
held-out glyphs (target at least 0.6). Decoder targets stay in pixels.
- Validity: gain 0 must reproduce the paper (own output about 0.25, readout 0.83-0.86, input 0.91,
  new decoder 0.75 for trained and untrained encoders).
- REPRESENTATION LEARNED if, at V = 8, m = 64, the trained state's population readout clearly beats both
  the input and the untrained encoder by at least 0.1, while the own decoder stays at most 0.5.
- DECODER ONLY if the trained state does not beat the input and untrained encoders. The paper then says
  the toy shows decoder failure, and representation has to rest on the real models.

## E25: does recovery predict a better procedure? (test 5)
Across architecture, width and V, trained and untrained encoders, at gain 0 and the entangled gain:
correlate the state's population readout with the new decoder's output on held-out glyphs.
- SUPPORTED if within-gain Pearson r >= 0.7 over conditions, and the transformer falls below the others
  on both axes at V = 8.

## E21: intervene at the question state (test 2)
Target "a x on a y" (figure x). Sources use the same two nouns:
same order, reversed figure ("a x under a y"); swapped ("a y on a x"); same figure, reversed order
("a y under a x", answer unchanged if the state codes meaning). Patch at the final token.
- CAUSAL if, in phi-4, Qwen3-8B and OLMo-2 13B at the E11 layer, patching the figure readout's subspace
  moves answers to the source's answer on at least 0.7 of held-out pairs, with random and top-PC subspaces
  of the same dimension clearly lower, and removing the subspace lowers accuracy clearly more than removing
  a random one. Same-figure sources must not move the answer.
- Sufficiency of the WHOLE state is the weaker first step. If the whole state does not flip answers,
  later layers read the answer from earlier tokens, and the paper's "answers trace to this state" must change.
- r = 0.94: report 95% bootstrap interval over models and leave-one-model-out range.

## E23: nonce words (test 3)
Readouts fit on real nouns; held-out parts are invented words, so they are held out from the system too.
- HOLDS if the models that answer by meaning keep answer accuracy above 0.75 on nonce words, with transfer
  still ordered with accuracy. Pythia and GPT-2 stay near 0.5.
- Interesting failure: answers hold and readouts fall (or the reverse). Either goes in the paper.

## E24: input baseline for pretrained models (test 4)
States: concatenated input embeddings of the two nouns (input keeps the roles by position) and the
average of input embeddings over the sentence.
- The objection applies if trained kappa does not exceed the input concatenation's kappa. Input
  concatenation scoring above trained states means models lose recoverable parts relative to their input,
  the same pattern as pixels in the toy.

## E26: the lower-left cell (test 6)
OLMo-2 7B: answers 0.77, ridge transfer 0.20. Kernel and MLP readouts of its question state, and whole-state
patching from E21.
- CLASS LIMIT if a richer readout or the whole-state patch reaches near the answer accuracy.
- Otherwise the answers do not come from this state, and the paper must stop placing the model in that cell.

## E22: recoverability as an objective
Prior result, not in the paper: results/e6aux (vocabulary aux loss on pixels), +0.14 at most at V = 8.
Arms at V = 8, m = 64, gain 0 and entangled: base; auxvocab (linear head to both glyphs on training pairs);
auxpool (head also fits composites of 64 other glyphs, decoder still only on the 8); through (the system's
output is the fixed render of its linear head).
- Prediction: auxvocab stays limited by the span proposition, near the vocabulary readout, at most +0.15.
- OBJECTIVE HELPS if auxpool clearly raises the state's readout over base on the entangled task by at least
  0.1. PROCEDURE HELPS if `through` clearly beats base's own output by at least 0.2 and matches the new decoder.
- Auxpool's own decoder is predicted to stay near base, which would mean a better state alone does not fix
  the decoder.

## Addendum, 2026-11-05 night (written after E21 and E20 calibration, before E21b and the sweep)
- E20 calibration: no cell of gain up to 32, layers up to 6, m 64/256, steps 4000/12000 meets input <= 0.5 with V=64 own >= 0.6, and trained readouts never exceed the input. The sweep runs at gain 0 and at gain 4, layers 3 (input 0.68, V=64 own 0.67), the strongest mix that clears the own-output floor. The input bar is NOT met; report this.
- E21: whole-state patches flip answers at the picked layer (phi-4 0.99, Qwen3 0.84, OLMo-13B 0.82); the readout subspace does not (<= 0.06), so the CAUSAL criterion as written fails on the readout.
- E21b criteria: if the patched state does not read out as the source's figure, the patch is buggy and E21 is void. REDUNDANT if readout or diff subspaces of growing k reach flip >= 0.7 well below the size at which random does. UNUSED if readout_128 stays near 0 while diff_k or whole flips and allbut_readout_16 still flips: the readout then reads a copy that the answer does not need.
