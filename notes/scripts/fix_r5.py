p = "/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/main.tex"
s = open(p, encoding="utf-8").read()
a0 = s.index("People understand sentences they have never heard")
a1 = s.index("can be called compositional.", a0) + len("can be called compositional.")
ABS = (r"People understand sentences they have never heard by building their meaning from familiar parts, and on many accounts they can take that meaning apart again into the same parts. Machine learning now calls almost any successful recombination of familiar material \emph{compositional}, but what a representation must be like to count as compositional remains unsettled. We propose that a representation is compositional when every part, \emph{including one the reading procedure was never fitted on}, can be read back in its role by a fixed procedure of a declared kind, such as an affine map, and when each part enters every role through one shared code. On a finite vocabulary almost any mode of combination can be decoded by lookup; recovery of fresh parts is bounded by the rank of a linear relation and, for affine readouts, by the width of the state; and when parts are recoverable and enter every role through one shared code, a single composite fixes every role reversal. Analytic modes of combination match these bounds, and vector-symbolic superposition of dense parts behaves like a lossy blend. In networks trained on glyph composites and in a learned recursive composer, the states keep fresh parts recoverable, but the systems' own decoders fail where training did not reach: on new glyphs after training on few glyphs, and one level beyond the composer's training depths. Among eleven pretrained text encoders, scored on whitened part codes, none comes near a role--filler reference: GPT-2 gets about halfway, and the two smaller CLIP encoders, including the one that conditions Stable Diffusion 1.5, gain little from training beyond the number of tokens in each noun. We close with five requirements a network must meet before both its states and its own procedures can be called compositional.")
s = s[:a0] + ABS + s[a1:]
R = [
 # N1
 (r"without counting what the reduction omits against the relation.",
  r"without counting what the reduction omits against the relation. Each $B_i$ must be injective; otherwise $e'$ could hold a separate code for each role, and any additively separable relation, even two unrelated lookup tables, would count as role-factorized."),
 # N6
 (r"$\varepsilon$ is invariant to rotations and rescalings of $e$", r"$\varepsilon$ is invariant to rotations and uniform rescalings of $e$"),
 # N2
 (r"the system is compositional at tolerance $\tau$ if, for some such", r"the system is representationally compositional at tolerance $\tau$ if, for some such"),
 (r"one that also meets all five requirements, verified on fresh parts against references built from its own part codes, is a compositional system in our sense;",
  r"one whose own procedures also read its parts from composites of fresh parts, as the second, fourth, and fifth requirements spell out, is a compositional system;"),
 (r"and ``compositional system'' for systems whose own procedures also use them.", r"and ``compositional system'' for systems whose own procedures also read them."),
 (r"and we report this share wherever a system-independent description exists", r"and we report the informativeness of the code wherever a system-independent description exists"),
 # N3
 (r"    & Behavior succeeds on new combinations & Behavior fails \\", r"    & Behavior succeeds on new combinations and new parts & Behavior fails \\"),
 (r"Vocabulary-bound generalization; systematicity is contingent on the data \citep{chaabouni2020} &",
  r"Generalization without parts recoverable by the declared readouts: the system's own procedure succeeds where they fail, and the readout class may be too weak &"),
 # N4
 (r"in which it occurs \citep{fodor1988}, who asked for approximately the same contribution and so license a graded $\delta$, with the coordinates restricted to affine maps so that the requirement has content.",
  r"in which it occurs \citep{fodor1988}. They asked for approximately the same contribution, which licenses a graded $\delta$, and we restrict the coordinates to affine maps so that the requirement has content."),
 # N5
 (r"but so does the same architecture with random weights, scored on its own part codes (roles 0.50--0.90, identification 0.44--0.74); random CLIP ViT-B/32 even identifies more fresh nouns than the trained model (0.74 against 0.71). Only recovery, and hence $\kappa$, separates trained from random weights.",
  r"and random weights, scored on their own part codes, come close for most architectures (roles 0.67--0.90, except 0.50 for T5; identification 0.44--0.74); random CLIP ViT-B/32 even identifies more fresh nouns than the trained model (0.74 against 0.71). Recovery, and hence $\kappa$, separates trained from random weights more reliably than either measure."),
 # N7
 (r"the interaction share of a two-way additive fit, a lower bound on $\delta$ for every code,", r"the interaction share of a two-way additive fit, an in-sample estimate of a lower bound on $\delta$ for every code,"),
 (r"we report the lower bound given by the interaction share", r"we report an in-sample estimate of the lower bound given by the interaction share"),
 # small
 (r"gained little beyond the length of the noun", r"gained little beyond the token count of the noun"),
 (r"including the ones that condition Stable Diffusion 1.5 and 2 \citep{rombach2022};", r"including the ones that condition the latent diffusion models Stable Diffusion 1.5 and 2 \citep{rombach2022};"),
 (r"\begin{lemma}[Unrestricted coordinates make separability vacuous]", r"\begin{lemma}[Unrestricted coordinates reduce separability to injectivity]"),
]
for a, b in R:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok", len(ABS.split()))
