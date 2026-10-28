p = "/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/main.tex"
s = open(p, encoding="utf-8").read()
R = [
 # ---------------- abstract (A-M1, B-M1/M2/M3, minor 9)
 (r"by the width of the state, and a shared code fixes every role reversal from a single composite.",
  r"by the width of the state; and when parts are recoverable and enter every role through one shared code, a single composite fixes every role reversal."),
 (r"In networks trained on glyph composites, and in a learned recursive composer, the states keep fresh parts recoverable, but the systems' own decoders read them only after training on many distinct parts.",
  r"In networks trained on glyph composites, and in a learned recursive composer, the states keep fresh parts recoverable, but the systems' own decoders fail where training did not reach: the glyph networks read fresh glyphs only after training on many distinct glyphs, and the composer's decomposer fails one level beyond its training depths."),
 (r"Among eleven pretrained text encoders, scored on part codes in which no single direction dominates, none comes near a role--filler reference: GPT-2 gets about halfway, and the CLIP encoders that condition text-to-image models gain little from training beyond the length of each noun. We close with five requirements a network must meet to earn the term.",
  r"Among eleven pretrained text encoders, scored on whitened part codes, readouts recover nouns in their sentence positions, but none comes near a concatenation reference: GPT-2 gets about halfway, and the two smaller CLIP encoders, including the one that conditions Stable Diffusion 1.5, gain little from training beyond the number of tokens in each noun. We close with five requirements a network must meet before both its states and its own procedures can be called compositional."),
 # ---------------- intro
 (r"the novel structure in the best model of a whole beyond the best models of its parts \citep{elmoznino2026}",
  r"the novel structure in the best model of an object beyond the best models of its parts \citep{elmoznino2026}"),
 (r"linear, orthogonal factors are necessary for compositional generalization in embedding models \citep{uselis2026}",
  r"linear, orthogonal factors are necessary for compositional generalization in vision embedding models \citep{uselis2026}"),
 (r"It also bears on \citet{vangelder1990}'s distinction, and on \citet{chalmers1990}'s demonstration that structure-sensitive operations can act on distributed representations directly: a representation can be functionally compositional, with parts that are recoverable, without being concatenative, with parts that are tokened.",
  r"It is closest to \citet{vangelder1990}'s \emph{functional} compositionality, which asks for general, effective, and reliable procedures that produce a complex from its constituents and decompose it back into them. Recoverability makes ``general'' and ``effective'' precise, as transfer to parts fresh to a procedure of a declared class, and role-factorization adds a requirement van Gelder did not make, that each part contribute one code to every role. \citet{chalmers1990} showed that structure-sensitive operations can act on such representations directly, so a representation can be functionally compositional without being concatenative, with parts that are recoverable but not tokened."),
 (r"We prove that unrestricted versions of both properties are vacuous,", r"We prove that unrestricted versions of both properties reduce to invertibility,"),
 (r"and show that under a role-factorized relation every role reversal", r"and show that under a role-factorized relation whose role maps are jointly injective, every role reversal"),
 # ---------------- running example (B-M4)
 (r"few vocabulary pairs lie on any one line $x + y = z$, and a lookup over the vocabulary recovers the parts of almost every sum. The sum then looks recoverable.",
  r"few vocabulary pairs lie on any one line $x + y = z$, and a lookup over the vocabulary recovers which two parts were summed in almost every case, though not which role each filled; with unequal weights, $x + 0.5y$, it recovers both. The sum then looks nearly recoverable."),
 (r"Few pairs of a finite vocabulary (dark points) fall on any one line, so the parts of familiar composites can be decoded, but fresh pairs (diamonds) can lie anywhere on it.",
  r"Few pairs of a finite vocabulary fall on any one line, and the two that fall on this one (dark points) are a role reversal of each other, so a lookup over familiar composites returns which two parts were summed but not their roles; fresh pairs (diamonds) can lie anywhere on the line."),
 # ---------------- definition 1 (A-M2, A-M3)
 (r"The relation is \emph{$\delta$-role-factorized} if there exist $c \in \R^m$ and matrices $B_1, \ldots, B_k \in \R^{m \times d}$, one per role and shared by all parts, such that the state is affine in the part codes up to a residual,",
  r"The relation is \emph{$\delta$-role-factorized} if, for some code $e'$ of dimension $d'$ of which $e$ is an affine function (possibly $e' = e$), there exist $c \in \R^m$ and matrices $B_1, \ldots, B_k \in \R^{m \times d'}$, one per role and shared by all parts, such that the state is affine in these codes up to a residual,"),
 (r"g_r(x_1, \ldots, x_k) = c + \textstyle\sum_{i} B_i\, e(x_i) + \eta(x_1, \ldots, x_k),",
  r"g_r(x_1, \ldots, x_k) = c + \textstyle\sum_{i} B_i\, e'(x_i) + \eta(x_1, \ldots, x_k),"),
 (r"evaluating on composites of other parts drawn from $\D$, never on the parts used for fitting.",
  r"evaluating on composites of other parts drawn from $\D$, never on the parts used for fitting. Errors are measured in a declared inner product on part codes, which is part of the declaration of $e$: $\varepsilon$ is invariant to rotations and rescalings of $e$ but not to other linear changes of coordinates. The richer code $e'$ lets $e$ be a reduced description, such as a few principal components, without counting what the reduction omits against the relation."),
 (r"as, under affine readouts, every blend into a space smaller than its parts must (Proposition~\ref{prop:ey})",
  r"as, under affine readouts, every blend into a space smaller than the dimensions its parts occupy must (Proposition~\ref{prop:ey})"),
 (r"which is the classical requirement that a constituent contribute the same thing to every complex in which it occurs \citep{fodor1988}, with the coordinates restricted",
  r"which is the classical requirement that a constituent contribute the same thing to every complex in which it occurs \citep{fodor1988}, who asked for approximately the same contribution and so license a graded $\delta$, with the coordinates restricted"),
 # ---------------- definition 2
 (r"where $\varepsilon_{\mathrm{ref}}$ and $\varepsilon_{\mathrm{blend}}$ are obtained by the same procedure from a role--filler operator and a symmetric blend applied to the same part codes; the system is compositional at tolerance $\tau$ if $\kappa \ge 1 - \tau$ and $\delta \le \tau$.",
  r"where $\varepsilon_{\mathrm{ref}}$ and $\varepsilon_{\mathrm{blend}}$ are obtained by the same procedure from concatenation and from the sum of the same part codes; the system is compositional at tolerance $\tau$ if, for some such $e$ informative at $\tau$, $\kappa \ge 1 - \tau$ and $\delta \le \tau$."),
 (r"Identification among candidates does not suffice, since almost any continuous code separates its parts.",
  r"Identification among candidates does not suffice, since any injective code separates its parts."),
 (r"Note that Definition~\ref{def:repcomp} is graded and relative. The normalization is well posed: for a symmetric blend of two parts with identical, independent distributions, no readout can tell the roles apart, and the best affine readout returns the mean of the two parts, so $\varepsilon_{\mathrm{blend}} = 1/2$ in units of part variance, whatever the code.",
  r"We call a system whose states meet Definition~\ref{def:repcomp} \emph{representationally compositional}, and one whose own procedures also read its parts from composites of fresh parts a \emph{compositional system}; Table~\ref{tab:twobytwo} crosses the two. Definition~\ref{def:repcomp} is graded and relative. The normalization is well posed: for the sum of two parts with identical, independent distributions, no readout can tell the roles apart, and the best affine readout returns the mean of the two parts, so $\varepsilon_{\mathrm{blend}} = 1/2$ in units of part variance, whatever the code. No readout does better on any symmetric relation, since the two parts then have the same conditional mean given the composite, and in population concatenation gives $\varepsilon_{\mathrm{ref}} = 0$, so $\kappa = 1 - 2\varepsilon$; the references correct the estimates for finite samples and readout fitting."),
 (r"with unrestricted readouts, as with unrestricted causal alignment maps \citep{sutter2025} and unrestricted probes \citep{hewitt2019}, any injective system passes.",
  r"with unrestricted readouts, as with unrestricted causal alignment maps \citep{sutter2025} and unrestricted probes \citep{hewitt2019}, any injective system passes (for measurable readouts by the Lusin--Souslin theorem: a Borel injection between Polish spaces has a Borel inverse on its image; \citealt[Theorem~15.1]{kechris1995})."),
 # note box (A-M6b,c)
 (r"so productivity is reported as a curve over depth, and its limits, as \citet{fodor1988} argued, are matters of performance.",
  r"so productivity is reported as a curve over depth. \citet{fodor1988} treated the depth limits of people as limits of performance on an unbounded competence; a state of fixed width imposes its limit in the representational format itself, so on their view such a system is at most approximately productive, and the depth curve measures how closely. Under affine readouts the definition recognizes role-indexed codes, such as fixed slots and tensor products. A concatenative string whose constituents vary in length places each constituent at an offset that depends on the others, so it meets the definition only under a readout class that includes alignment, such as a parser, which must then be declared and calibrated like any other."),
 # Marcus (B-M9)
 (r"A procedure therefore has to be learned from enough distinct parts before it can handle new ones.",
  r"A procedure therefore has to be learned from enough distinct parts before it can handle new ones. This is a readout-side form of \citet{marcus1998}'s argument that networks trained by backpropagation do not extend a relation to items outside their training space; our networks show that the argument can hold of a decoder while failing for the state it reads (Section~\ref{sec:networks})."),
 # related work
 (r"\citet{uselis2026} argue that compositional generalization in embedding models requires",
  r"\citet{uselis2026} argue that compositional generalization in vision embedding models requires"),
 (r"with a ``memorization leak'' in their outputs, a behavioral counterpart of the residual $\eta$ that we measure in states.",
  r"with a ``memorization leak'' in their outputs, a behavioral counterpart of the part of $\eta$ that a least-squares additive fit absorbs under an incomplete design (Remark after Proposition~\ref{prop:context})."),
 (r"a fixed-width superposition over a small vocabulary compresses well and is decodable",
  r"a fixed-width superposition over a small vocabulary would plausibly compress well, and it is decodable"),
 # Table 1 (A-m5) + park
 (r"    & Behavior succeeds on new combinations and new parts & Behavior fails \\",
  r"    & Behavior succeeds on new combinations & Behavior fails \\"),
 (r"(Section~\ref{sec:networks}; cf.\ \citealt{montero2021}) \\",
  r"(Section~\ref{sec:networks}; cf.\ \citealt{montero2021,park2024hidden}) \\"),
 # experiments
 (r"Methods are summarized below;", r"We summarize the methods below;"),
 (r"Vector-symbolic superposition of dense parts is therefore not invertible in the sense of Definition~\ref{def:recoverable} under affine readouts; it is decodable over a vocabulary.",
  r"Vector-symbolic superposition of dense parts in a fixed width is therefore not recoverable in the sense of Definition~\ref{def:recoverable}: for Gaussian parts no readout of any kind exceeds the ceiling of $1/2$ (Proposition~\ref{prop:ceiling}); it is decodable over a vocabulary."),
 (r"around $kq = 16$, the dimension of a pair of latents,", r"around $kq = 16$, the dimension of a pair of latents ($k = 2$, $q = 8$),"),
 (r"Their role-factorization residual, measured in Section~\ref{sec:structure}, is $\delta = 0.02$--$0.04$, so by Definition~\ref{def:repcomp} these states are compositional at a tolerance of about $\tau = 0.27$ at $V = 8$ and $0.17$ at $V = 64$, and not at $\tau = 0.1$.",
  r"The glyph code is informative: an affine map of a glyph's pixels recovers its latent with $R^2 = 0.92$ on fresh glyphs (0.89 from the 16 principal components used in Section~\ref{sec:structure}; 10 generators). The role-factorization residual, measured in Section~\ref{sec:structure} on the 16-component code at $V \in \{16, 64\}$, is $\delta = 0.02$--$0.04$, which bounds the residual for the pixel code because the components are linear in the pixels. By Definition~\ref{def:repcomp} the states at $V = 64$ are therefore representationally compositional at a tolerance of about $\tau = 0.17$ and not at $\tau = 0.1$; at $V = 8$, where $\delta$ was not measured, $\kappa$ alone puts the tolerance at no less than 0.27."),
 (r"as Proposition~\ref{prop:ceiling} requires for a linear relation of full rank", r"as Proposition~\ref{prop:ceiling} allows for a linear relation of full rank"),
 # E4
 (r"including the one that conditions Stable Diffusion 1.5 \citep{rombach2022};",
  r"including the ones that condition Stable Diffusion 1.5 and 2 \citep{rombach2022};"),
 (r"the first component holds 99\% of the variance in GPT-2's early layers,",
  r"the first component holds 99\% of the variance of the 16-component code in GPT-2's early layers,"),
 (r"To separate what training contributes from what the architecture, tokenizer, and readout supply, in the spirit of control tasks \citep{hewitt2019}, we run",
  r"To separate what training contributes from what the architecture, tokenizer, and readout supply, we run"),
 (r"and from the sum reference with 0.47--0.48,", r"and from the sum reference with 0.48,"),
 (r"With $\kappa \le 0.50$, no encoder is compositional under Definition~\ref{def:repcomp} at any tolerance below one half, whatever its residual $\delta$; the networks of Section~\ref{sec:networks} reached $\kappa = 0.73$--$0.83$.",
  r"With $\kappa \le 0.50$, no encoder is representationally compositional under these part codes and this metric at a tolerance below about one half. The residual does not change the verdict: on a complete $12 \times 12$ grid of fitting nouns, the interaction share of a two-way additive fit, a lower bound on $\delta$ for every code, is 0.007--0.012 for GPT-2, at most 0.02 for CLIP, and 0.11--0.19 for BERT and T5, and zero for both references. The verdict does depend on the metric: on the same components without whitening, where one direction dominates, GPT-2 reaches $\kappa = 0.81$--$0.83$."),
 (r"Every trained model assigns roles to fresh nouns far better than the sum reference (0.90--1.00 against 0.50) and identifies them among 100 candidates far above chance (0.65--0.94) and above its own sum reference (0.41--0.44).",
  r"Every trained model assigns roles to fresh nouns better than the sum reference (0.89--1.00 against 0.50) and identifies them among 100 candidates far above chance (0.65--0.94, against 0.41--0.44 from its sum reference), but so does the same architecture with random weights, scored on its own part codes (roles 0.50--0.90, identification 0.44--0.74); random CLIP ViT-B/32 even identifies more fresh nouns than the trained model (0.74 against 0.71). Only recovery, and hence $\kappa$, separates trained from random weights."),
 (r"by about 0.3 for T5 and CLIP-H, by 0.08--0.17 for BERT", r"by 0.26--0.30 for T5 and CLIP-H, by 0.08--0.17 for BERT"),
 (r"Third, the part codes carry the length of the noun in tokens. Removing it lowers recovery for most encoders and leaves the top of the ranking in place",
  r"Third, the part codes carry the number of tokens into which the tokenizer splits a noun. CLIP's tokenizer splits only 7 of the 305 nouns, yet that distinction dominates the codes of the smaller CLIP encoders. Removing a quadratic in the token count lowers recovery for most encoders and keeps GPT-2 and CLIP-H ahead of the rest"),
 (r"GPT-2 gets about halfway, and the CLIP encoders that condition text-to-image models gain little from training beyond the length of the noun.",
  r"GPT-2 gets about halfway; CLIP-H, the text encoder of Stable Diffusion 2, gains from training about as much as T5; and the two smaller CLIP encoders, including the one that conditions Stable Diffusion 1.5, gain little beyond the token count of the noun."),
 (r"crosses show fresh nouns scored on its own part codes and dots the same scored on the trained model's part codes.",
  r"crosses show fresh nouns scored on its own part codes and dots the same scored on the trained model's part codes (drawn at zero when negative)."),
 (r"dark grey marks show identification from each model's own sum reference.",
  r"dark grey marks show identification from each model's own sum reference, and crosses both measures for the same architecture with random weights, scored on its own part codes."),
 # discussion
 (r"The states of small networks trained on glyph composites were recoverable whatever their training,",
  r"The states of small networks trained on glyph composites were recoverable whatever their training, except the transformer's at small vocabularies,"),
 (r"within 0.13 of the rank ceiling", r"within 0.14 of the width ceiling"),
 (r"it raised $\kappa$ by 0.5--0.6 for GPT-2 and by about 0.3 for T5 and CLIP-H,", r"it raised $\kappa$ by 0.5--0.6 for GPT-2 and by 0.26--0.30 for T5 and CLIP-H,"),
 (r"a state narrower than the parts it combines loses them,", r"a state narrower than the dimensions its dense parts occupy loses them under affine readouts,"),
 (r"a network trained on eight glyphs solves unseen pairs and fails on new glyphs although its state contains them, because its decoder, like any procedure fitted to composites, is determined only where its training composites constrain it",
  r"the MLP, CNN, and role--filler networks trained on eight glyphs solve unseen pairs and fail on new glyphs although their states contain them, because their decoders, like any procedure fitted to composites, are determined only where their training composites constrain them"),
 (r"(spill 0.16 against 0.25; Figure~\ref{fig:structure}c)", r"(Proposition~\ref{prop:systematic}(iv); spill 0.16 against 0.25; Figure~\ref{fig:structure}c)"),
 (r"failed one level deeper (Figure~\ref{fig:depth}a)", r"failed one level deeper (Proposition~\ref{prop:depth}; Figure~\ref{fig:depth}a)"),
 (r"A network that meets all five, verified on fresh parts against references built from its own part codes, has earned the term in our sense; one that meets only the behavioral tests should be described as generalizing compositionally.",
  r"A network whose states meet Definition~\ref{def:repcomp} is representationally compositional; one that also meets all five requirements, verified on fresh parts against references built from its own part codes, is a compositional system in our sense; one that meets only the behavioral tests should be described as generalizing compositionally."),
 (r"the bare term should be reserved for systems whose parts are recoverable from composites of fresh parts and whose own procedures use them.",
  r"``representationally compositional'' should be reserved for states whose parts are recoverable from composites of fresh parts, and ``compositional system'' for systems whose own procedures also use them."),
 (r"although a role-factorized state allows a part to be added by adding its code alone (Figure~\ref{fig:structure}d).",
  r"although a role-factorized state allows a part to be placed in a new role by adding its code, without the damage that fine-tuning causes (Figure~\ref{fig:structure}d)."),
 # people (B-M6)
 (r"but almost none of it was gathered with parts that are fresh in our sense.", r"but little of it was gathered with parts that are fresh in our sense."),
 (r"over a closed set of nouns and verbs \citep{frankland2015};", r"over a closed set of nouns and verbs \citep{frankland2015}, as reviewed by \citet{frankland2020};"),
 (r"people represent novel objects as compositions of known elements \citep{schwartenbeck2023}, use a newly learned word in combinations they have never seen \citep{lake2019}, and endorse inferences across arbitrary contents \citep{rips1994}.",
  r"people represent novel objects as compositions of known elements \citep{schwartenbeck2023} and endorse inferences across arbitrary contents \citep{rips1994}."),
 (r"and belief overrides validity \citep{evans1983}.", r"and beliefs about a conclusion bias judgments of its validity \citep{evans1983}."),
 (r"A human analogue of our test would teach a new word, object, or face and ask whether it can be read back from a scene or sentence in a role it was never shown in. The paradigms that come closest are the wug test \citep{berko1958}, rule learning over novel syllables \citep{marcus1999}, and syntactic bootstrapping of novel verbs \citep{gleitman1990,naigles1990}; illusory conjunctions \citep{treisman1982}, the costs of dynamic role binding \citep{hummel1997}, and binding limits in working memory \citep{wheeler2002} suggest that human recovery of fresh conjunctions is capacity-limited and graded.",
  r"The paradigms that come closest teach a new item and test it in new combinations: the wug test \citep{berko1958}, rule learning over novel syllables \citep{marcus1999}, syntactic bootstrapping of novel verbs \citep{gleitman1990,naigles1990}, and few-shot learning of pseudoword instructions \citep{lake2019}. These test the person's own procedure, the behavioral column of Table~\ref{tab:twobytwo}. Definition~\ref{def:recoverable} has a direct neural analogue: a decoder of agent and patient fitted on some nouns and tested on nouns withheld from fitting, cross-condition generalization \citep{bernardi2020} applied to new parts. Illusory conjunctions \citep{treisman1982} and binding limits in working memory \citep{wheeler2002} show that people bind even familiar features into new conjunctions with limited capacity, as models of dynamic role binding predict \citep{hummel1997}."),
 # appendices
 (r"$\R^d$ be lifted to $\R^w$ by a linear map and composed by $g(x, y) = Ax + By$ with $A, B \in \R^{w \times w}$.",
  r"$\R^d$ be lifted to $\R^w$ by a linear map and composed by $g(x, y) = Ax + By$ with $A, B \in \R^{w \times w}$, possibly different at each level."),
 (r"leaves every readout $P_j$, $j \ne i$, unchanged.", r"leaves every readout value $P_j(h - c)$, $j \ne i$, unchanged."),
 (r"for some invertible $\psi$ and some maps $\varphi_i$.", r"for some $\psi$ injective on the image of $g$ and some maps $\varphi_i$."),
 (r"jointly predict a behavioral quantity;", r"jointly bound the error of a predicted state;"),
 (r"standardization changes the metric in which the fit projects, as Proposition~\ref{prop:span} describes,",
  r"standardization changes the metric of the projection within each role (Proposition~\ref{prop:span} with $B$ the inverse of the diagonal matrix of standard deviations),"),
 (r"Default inheritance only; overrides and emergent attributes form $\eta$",
  r"Approximately, if default inheritance is role-factorized (assumed); overrides and emergent attributes form $\eta$"),
 (r"    Head--modifier blending & $x + \alpha y$ &", r"    Head--modifier weighting (fixed-width superposition) & $x + \alpha y$ &"),
 (r"\section{Capacities and the human reference profile}", r"\section{Capacities and a qualitative human profile}"),
 (r"Transformers reach at most 40.6\% on the structural cases of SLOG,", r"Transformers reach at most 40.6\% on SLOG, whose cases are all structural,"),
 (r"    Known part, new role & A newly learned word applied correctly in unseen combinations & \citet{lake2019} \\",
  r"    New part, new combinations & A newly learned pseudoword applied correctly in unseen combinations & \citet{lake2019} \\"),
 (r"Belief overrides validity;", r"Beliefs about a conclusion bias judgments of its validity;"),
 (r"The residual $\delta$ is the share of state variance that an affine map of the two whitened part codes does not predict on fresh pairs; the concatenation reference itself has $\delta$ between 0.04 and 0.81, because 16 components omit part of each code, so the verdicts rest on $\kappa$. In an earlier run with unwhitened codes, selecting the penalty on composites instead of nouns, as in a standard probe, fitted the fitting nouns more closely and transferred worse.",
  r"On the 16-component code, the residual of an affine fit of the state is 0.04--0.81 even for the concatenation reference, because 16 components omit part of each code, so it gives no useful upper bound on $\delta$; we report the lower bound given by the interaction share of a two-way additive fit on the $12 \times 12$ grid of fitting nouns, which any additive fit in any code must leave. For the random-weight model scored on the trained model's codes, layer and code are chosen on its own validation score. A role is assigned correctly when the readout's prediction is closer to the code of the noun in that role than to the code of the other noun. Mean pooling covers every token, including special tokens."),
]
for a, b in R:
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
# move the fibers figure into Section 2, after the running example
i = s.index(r"\begin{figure}[h]" + "\n" + r"  \centering" + "\n" + r"  \includegraphics[width=0.9\linewidth]{fig_fibers.pdf}")
j = s.index(r"\end{figure}", i) + len(r"\end{figure}")
fig = s[i:j].replace(r"\begin{figure}[h]", r"\begin{figure}[t]", 1)
s = s[:i] + s[j:]
k = s.index(r"The sum then looks nearly recoverable.")
k = s.index("\n\n", k)
s = s[:k] + "\n\n" + fig + s[k:]
open(p, "w", encoding="utf-8").write(s)
print("ok", len(R))
