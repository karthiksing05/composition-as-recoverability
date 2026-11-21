# Refactor outline and opening draft (2026-11-20, for the user's reaction; main.tex untouched)

## Spine (each section opens by naming the step it serves)
1. Tests of composition score outputs on new combinations of known parts.
2. A lookup table passes them, and so does a model whose state has merged its parts.
3. These tests miss the way back from a whole to its parts: decomposition.
4. A fair test of the way back needs a linear probe fixed in advance and parts the probe never saw.
5. With it we can ask whether a model's state holds more than its outputs use.
6. Toy systems show that it often does, and that a procedure reading the state recovers the rest;
   larger image, diffusion and language models show the same pattern.

## New order
1. Introduction (spine; toy-first plan in its own paragraph; bulleted contributions)
2. Related work (unchanged topics, trimmed)
3. Weak and strong compositionality
   3.1 Parts, wholes and roles (example first: x + y against side by side)
   3.2 What a whole keeps (NEW, from the user's note: parts and roles / parts only / neither; most wholes keep part of each)
   3.3 Weak compositionality recombines known parts
   3.4 Decomposition makes parts reusable (the spectrum of capabilities, Table 1)
   3.5 Behavior and representation (where each form is measured; the hypothesis)
4. Measuring decomposition
   4.1 Recoverability (linear probe fixed in advance; parts new to the probe; kappa, anchored by examples)
   4.2 Shared role codes (one paragraph; residual)
   4.3 Toy operators: tests on training parts cannot see decomposition (moved from 4.4; first toy result)
5. Experimental setup (systems from toy to large: operators, glyph networks, recursive composers,
   image encoders, diffusion models, language models; protocol; metrics; H1-H4 kept)
6. Results, by system, each opening with why that system
   6.1 Glyph networks: parts new to the model (behavior lags state; new decoder; code swaps; written codes)
   6.2 Recursive composers: the same gap at depth
   6.3 Image encoders: places kept, matching blind to them
   6.4 Diffusion models (NEW; D1-D4)
   6.5 Language models: the question state carries the role; word order; OLMo-2 7B as the limit of a linear probe
   6.6 Where recoverability comes from (inputs, random weights, scale, training, across all systems)
   Main-result figure (the ladder) moves to page 2 as the teaser.
7. A shared language for compositionality (moved spectrum; five named sets of capabilities; Table)
8. Conclusion and limitations
Appendix: proofs; informative codes and productivity; the three limits in full; random hierarchy;
recovery-objective table; width and occlusion; protocol details.

## Vocabulary
| old | new |
|---|---|
| rung | (dropped) "the set of capabilities an account asks for"; the sets are named: blend, weak, strong, shared role codes, explicit copies |
| affine readout, readout class, declared | linear probe, fixed in advance (bias included, said once) |
| part code | part vector |
| fitting pool | the probe's training parts |
| held out from the readout / system | new to the probe / new to the model |
| own procedure | the model's own output (its decoder, answers or images) |
| population / vocabulary readout | probe trained on many parts / probe trained on the model's training parts |
| tokened constituents | explicit copies |
| role-factorized, delta | shared role codes, role residual |
| interchange edit | code swap |
| transfer | role transfer |
| epsilon-recoverable, informative, productive to depth | appendix only |

## Draft abstract (~260 words)
Anyone who can think that John loves Mary can also think that Mary loves John. Fodor and Pylyshyn
called this regularity systematicity and traced it to representations built from reusable parts.
Yet the tests we run on neural networks check only the way in: they ask a model to act correctly on
new combinations of known parts. A lookup table passes these tests, and so does a model whose
internal state has merged its parts beyond repair. We argue that compositionality is a spectrum of
capabilities, from recombining known parts to taking in parts never seen together, placing a new
part in any role, adding parts without forgetting and nesting wholes inside wholes. The spectrum
divides into two forms. Weak compositionality combines known parts into new wholes and acts on them
correctly. Strong compositionality keeps every weak capability and adds decomposition, the return
of each part in its role from a whole, and every harder capability passes through that return. We
measure decomposition by recoverability, a calibrated score of how well a linear probe, fixed in
advance, returns parts it never saw in training. We hypothesize that behavioral compositionality is
a subset of representational compositionality, able to do some, and not all, of the things
representational compositionality can do. We prove that tests on training parts cannot detect
decomposition and that a model's state bounds the behavior that reads it. Toy networks, where parts
can be new to the model itself, show the gap cleanly: their states keep parts that their own
decoders lose. The gap recurs in pretrained image encoders and [diffusion models, pending D3],
while language models that answer by meaning carry each role in the state they answer from.
Procedures that read these states close the gap, and the same terms place fifteen accounts of
compositionality on one spectrum.

## Draft introduction
[P1, user's, verbatim] Composition is the study of parts and their combination through some
operation. A compositional system builds wholes from reusable parts, so that the meaning of a
sentence follows from the meanings of its words and the way they combine, and a scene follows from
its objects and their places.

[P2, kept] Composition matters because finite experience must support an unbounded number of new
situations. People understand sentences they have never heard. A person who learns the verb dax
understands dax twice at once, and anyone who can think that John loves Mary can think that Mary
loves John. Composition earns its value through the abilities that follow from reuse. A
compositional system handles new combinations of known parts, and it can also use a new part in
every role, add knowledge without damaging old knowledge, nest wholes inside wholes and keep each
part in its role.

[P3, rewritten around the user's note] Wholes differ in how much of their parts they keep, and in
the way they keep them (Figure 1). Two photographs side by side keep both pictures and their places,
so the whole comes apart again. A sum of two vectors keeps the pair yet loses its order, because
x + y equals y + x, and an armchair shaped like an avocado keeps neither object whole. Most wholes
sit between these cases and keep some of each part. We call the building of a whole composition,
and the return of the same parts in the same roles decomposition.

[P4, kept, sharpened] Current tests score composition through behavior. Benchmarks ask a model to
act on new combinations of training parts, and we judge text-to-image models by the scenes they
compose. These tests reward recombination, yet a lookup over a finite vocabulary passes them, and
success can come from fragments seen in training. Notably, text-to-image models that compose new
scenes still misplace objects and attach attributes to the wrong object. Behavioral tests cannot
say whether such a model lost the parts or merely failed to read them.

[P5, kept] We propose that compositionality is a spectrum of capabilities, and that the spectrum
divides into two forms, one containing the other. Weak compositionality combines known parts into
new wholes and acts on them correctly. Strong compositionality keeps every weak capability and adds
the capabilities that need wholes to come apart. Every strongly compositional system is therefore
weakly compositional, whereas a weakly compositional system may lack decomposition entirely.

[P6, user's, verbatim] The two notions form a weak and a strong version of composition. Behavioral
compositionality is the weak version, in which a system composes parts, and any decomposition it
performs comes from a procedure trained for that purpose. Representational compositionality is the
strong version, in which a system composes parts and its composites also decompose. Recoverability
is our heuristic for the strong version, and it asks whether parts read out easily from the internal
states of a composite. Decomposition matters because it supports the efficient use of parts and
reveals what each part donates to the whole.

[P7] Decomposition needs a careful test, because a loose one passes every system. A lookup table
over the training parts takes apart any whole that keeps them distinct, and a flexible enough probe
decodes almost anything. Our test therefore makes two demands. The probe that takes a whole apart is
linear and fixed before we look at the data, and it must succeed on parts it never saw in its own
training. Recoverability scores decomposition under both demands, from 0 for a sum of the parts to 1
for a side-by-side copy of each.

[P8, hypothesis verbatim + one sentence] We hypothesize that behavioral compositionality is a subset
of representational compositionality, able to do some, and not all, of the things
representational compositionality can do. A model whose own outputs handle new parts must have a
state that holds them, while a state can hold parts that the model's outputs miss. A recoverable
state therefore tells us which capabilities a new procedure could reach, before anyone builds it.

[P9, NEW: why toys first] We start with toy systems because only there can we keep parts away from
the model itself. Small networks that render pairs of glyphs, trained on eight glyphs, show the
whole pattern on one page: their decoders fail on new glyphs, a linear probe of their state recovers
those glyphs, and a new decoder trained on the frozen state renders them. We then carry the same
test to pretrained image encoders, diffusion models and language models with up to 15 billion
parameters, where parts are new to the probe, and the toy pattern recurs.

[Contributions, bulleted, bold heads; unchanged in substance, renamed terms]
