import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
R = [
 # A2: Westerstahl disputes Zadrozny
 (r"Any system that computes a function of its inputs satisfies it \citep{zadrozny1994,westerstahl1998}, so it cannot distinguish, in representational form, an avocado placed \emph{on} a chair from an armchair in the shape of an avocado (Figure~\ref{fig:ops}).",
  r"Any system whose state is a function of its inputs satisfies it, so it cannot distinguish, in representational form, an avocado placed \emph{on} a chair from an armchair in the shape of an avocado (Figure~\ref{fig:ops}); \citet{zadrozny1994} argued for a stronger triviality result, which \citet{westerstahl1998} disputed."),
 # A1 + B1: pet fish
 (r"\citet{kamp1995} and \citet{hampton2012} replied that prototypes compose by default with overrides. In our terms, default inheritance is a role-factorized component, and overrides and emergent attributes are the residual $\eta$; the debate is about the size and structure of $\delta$, not about whether composition occurs.",
  r"\citet{kamp1995} largely accepted the objection and kept prototypes as a non-compositional component beside a compositional semantics, while \citet{hampton1987,hampton2012} proposed that prototypes compose by default inheritance with overrides and emergent attributes. Our graded definition takes a position in this debate rather than standing outside it: if default inheritance is role-factorized, which we assume and do not show, overrides and emergent attributes fall in the residual $\eta$, and the question becomes how large and how structured $\delta$ is. Fodor and Lepore would reject this reading, since for them a residual of any size means the concept does not compose."),
 # B2/B3: taxonomy caption
 (r"keeps roles distinct but returns parts only by clean-up over a known vocabulary;",
  r"keeps roles distinct but returns dense parts only by clean-up over a known vocabulary;"),
 (r"Only role--filler composition, which includes superposition in disjoint regions, recovers parts never seen before and so satisfies",
  r"Among the modes shown, which act on dense parts, only role--filler composition, which includes superposition in disjoint regions, recovers parts fresh to the readout and so satisfies"),
 # B2: discussion opening
 (r"from which every part, including a part never seen before, can be read back",
  r"from which every part, including a part the procedure was never fitted on, can be read back"),
 # B4
 (r"such a relation is recoverable on its sparse domain and should be tested with fresh parts drawn from that domain.",
  r"under a readout class that includes sparse recovery, such a relation may be recoverable on its sparse domain. We did not test this case; it should be tested with fresh parts drawn from that domain."),
 # B5
 (r"remains contingent on training, as the seed spread at small vocabularies shows.",
  r"remains contingent on training, as the seed spread at small vocabularies suggests, although part of that spread reflects which glyphs are withheld."),
 # B6
 (r"the fresh nouns are fresh to the readout, the second sense of Section~\ref{sec:definition}.",
  r"the fresh nouns are fresh to the readout, in the sense of freshness relative to a procedure (Section~\ref{sec:definition})."),
 # B7
 (r"The states of small networks trained on glyph composites were recoverable, and nearly role-factorized, whatever their training, largely because an invertible input survives a random encoder;",
  r"The states of small networks trained on glyph composites were recoverable whatever their training, largely because an invertible input survives a random encoder, and in trained networks they were nearly role-factorized;"),
 (r"its state kept every leaf one and two levels beyond its training, while its decomposer did not.",
  r"its state kept every leaf at the untrained depths, within 0.13 of the rank ceiling, while its decomposer failed at the first untrained depth."),
 (r"training itself made the difference for GPT-2 and T5 and none for CLIP.",
  r"training itself made a large difference for GPT-2 and the T5 models (0.25--0.45 above random weights), a smaller one for CLIP-H, and none for CLIP-B, CLIP-L, or BERT."),
 # B8
 (r"Figure~\ref{fig:structure}c), although this holds for in-distribution changes only.",
  r"Figure~\ref{fig:structure}c). The separation holds for components taken from real composites; a random direction within the same role subspace spills (0.75), so the decoder respects the role code only on the distribution of codes it was trained on."),
 # B9
 (r"Coverage of combinations, which benchmarks reward, matters less than the number of distinct parts.",
  r"In our glyph domain, coverage of combinations, which benchmarks reward, mattered less than the number of distinct parts; whether this holds in other domains is untested."),
 # ---- reviewer C / A
 (r"CLIP's text encoders do no better than the same architecture with random weights.",
  r"the two smaller CLIP text encoders, including Stable Diffusion 1.5's, do no better than the same architecture with random weights."),
 (r"every number is a mean over seeds with standard errors in the figures.",
  r"every number is a mean over seeds; figures show standard errors where their captions say so."),
 (r"The network's own fresh-glyph behavior follows the vocabulary readout (colored) and not the population readout (grey).",
  r"The network's own fresh-glyph behavior follows the vocabulary readout (colored) more closely than the population readout (grey); values below $-0.2$ are clipped."),
 (r"($R^2 \ge 0.999$) whenever $w \ge 2^n d$", r"($R^2 \ge 0.997$) whenever $w \ge 2^n d$"),
 (r"it stays within 0.13 of the ceiling", r"it stays within 0.14 of the ceiling"),
 (r"(spill 0.19)", r"(spill 0.20)"),
 (r"with $r = 0.96$ ($r = 0.64$ within conditions)", r"with $r = 0.95$ ($r = 0.64$ within conditions)"),
 (r"Under Definition~\ref{def:repcomp} both relations are productive up to the width ceiling;",
  r"Both relations keep every leaf recoverable from the root state up to the width ceiling, the leaf-level form of productivity (we did not test the clause of Definition~\ref{def:repcomp} that takes child states as part codes);"),
 (r"As a selectivity control \citep{hewitt2019}, we run the same pipeline",
  r"To separate what training contributes from what the architecture, tokenizer, and readout supply, in the spirit of control tasks \citep{hewitt2019}, we run the same pipeline"),
 (r"jointly predict a behavioral quantity, and Section~\ref{sec:structure} measures it.",
  r"jointly predict a behavioral quantity; Section~\ref{sec:structure} measures both quantities but does not evaluate the bound."),
 (r"$\lVert B_\pi B^{+} \rVert^2 \le \kappa(B)^2$", r"$\lVert B_\pi B^{+} \rVert^2 \le \mathrm{cond}(B)^2$"),
 (r"Our condition is weaker than theirs: role-factorization asks that $[B_1 \cdots B_k]$ have full column rank, not that the role subspaces be orthogonal, and a full-rank but ill-conditioned $B$ is recoverable with amplified noise.",
  r"Our condition is weaker than theirs: role-factorization asks only for an additive decomposition into role components, recovery of fresh parts then needs $[B_1 \cdots B_k]$ to have full column rank, not orthogonal role subspaces, and a full-rank but ill-conditioned $B$ is recoverable with amplified noise."),
 (r"with a ``memorization leak'' that corresponds to our residual $\eta$.",
  r"with a ``memorization leak'' in their outputs, a behavioral counterpart of the residual $\eta$ that we measure in states."),
 (r"What we add is the requirement that the readouts be evaluated on parts outside the vocabulary from which they and the system were fitted.",
  r"What we add is the requirement that the readouts be evaluated on parts outside the vocabulary from which they were fitted, and, where the system can be trained from scratch, outside the vocabulary it was trained on."),
 (r"Decodable, disentangled factors do not by themselves ensure that a decoder generalizes \citep{montero2021},",
  r"Decodable, disentangled factors do not by themselves ensure that a decoder generalizes \citep{montero2021,schott2022},"),
 (r"It also bears on \citet{vangelder1990}'s distinction:",
  r"It also bears on \citet{vangelder1990}'s distinction, and on \citet{chalmers1990}'s demonstration that structure-sensitive operations can act on distributed representations directly:"),
 (r"illusory conjunctions \citep{treisman1982} and binding limits",
  r"illusory conjunctions \citep{treisman1982}, the costs of dynamic role binding \citep{hummel1997}, and binding limits"),
 (r"A part code is \emph{informative} if parts can be identified from it among held-out candidates, or if it retains a declared share of the variance of a description of the parts that does not depend on the system.",
  r"A part code is \emph{informative} at tolerance $\tau$ if an affine map of $e$ recovers a declared description $z(x)$ of the parts that does not depend on the system, with $R^2 \ge 1 - \tau$ on fresh parts. Identification among candidates does not suffice, since almost any continuous code separates its parts."),
 (r"and we report how much of a system-independent description of the parts each code retains.",
  r"and we report this share wherever a system-independent description exists (the glyph latents of Section~\ref{sec:networks}); for nouns we have none, which limits the verdicts of Section~\ref{sec:pretrained}."),
 (r"For each relation and seed, the 305 nouns", r"For each seed, the 305 nouns"),
 (r"A nested linear blend of depth $n$ loses", r"A nested linear blend of depth $n$ into a state of width $d$ loses"),
 (r"For concatenation of $V$ isotropic parts in $\R^d$, $\dim S_i = V - 1$ and",
  r"For concatenation of $V \le d + 1$ isotropic parts in $\R^d$, $\dim S_i = V - 1$ and"),
 (r"on fresh parts, which Figure~\ref{fig:calibration}c compares with fitted readouts.",
  r"on fresh parts; for $V > d + 1$, $S_i = \R^d$ and $R^2 = 1$. Figure~\ref{fig:calibration}c compares the closed form with fitted readouts."),
]
for a, b in R:
    if b == "PLACEHOLDER":
        continue
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
