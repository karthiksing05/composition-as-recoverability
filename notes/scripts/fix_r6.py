"""Round-6 polish: Def 2 split, Method run-ins, crosstalk derivation, leak bound, citations, pixel scale."""
import sys
P = "/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/main.tex"
s = open(P, encoding="utf-8").read()
R = []

# 1. Definition 2 split into a definition of the score and a definition of informativeness and productivity
R.append((r"""A system is \emph{representationally compositional} with respect to a relation $r$, relative to a declared readout class $\U$ and a declared class $\M$ of methods for finding part codes, to the degree that, for some part code $e$ found under $\M$ that is informative about the parts, its realization of $r$ is $\varepsilon$-recoverable and $\delta$-role-factorized. We report the degree by $\delta$ and by the normalized score $\kappa = (\varepsilon_{\mathrm{blend}} - \varepsilon)/(\varepsilon_{\mathrm{blend}} - \varepsilon_{\mathrm{ref}})$, where $\varepsilon_{\mathrm{ref}}$ and $\varepsilon_{\mathrm{blend}}$ are obtained by the same procedure from concatenation and from the sum of the same part codes; the system is representationally compositional at tolerance $\tau$ if, for some such $e$ informative at $\tau$, $\kappa \ge 1 - \tau$ and $\delta \le \tau$. A part code is \emph{informative} at tolerance $\tau$ if an affine map of $e$ recovers a declared description $z(x)$ of the parts that does not depend on the system, with $R^2 \ge 1 - \tau$ on fresh parts. Identification among candidates does not suffice, since any injective code separates its parts. The relation is \emph{productive to depth $n$} if the same holds when the parts are themselves composites of depth up to $n - 1$, with the state of each composite as its part code.
\end{defbox}""",
r"""A system is \emph{representationally compositional} with respect to a relation $r$, relative to a declared readout class $\U$ and a declared class $\M$ of methods for finding part codes, to the degree that, for some part code $e$ found under $\M$ that is informative about the parts (Definition~\ref{def:informative}), its realization of $r$ is $\varepsilon$-recoverable and $\delta$-role-factorized. We report the degree by $\delta$ and by the normalized score
\begin{equation}
  \kappa = \frac{\varepsilon_{\mathrm{blend}} - \varepsilon}{\varepsilon_{\mathrm{blend}} - \varepsilon_{\mathrm{ref}}},
  \label{eq:kappa}
\end{equation}
where $\varepsilon_{\mathrm{ref}}$ and $\varepsilon_{\mathrm{blend}}$ are obtained by the same procedure from concatenation and from the sum of the same part codes. The system is representationally compositional at tolerance $\tau$ if, for some such $e$ informative at $\tau$, $\kappa \ge 1 - \tau$ and $\delta \le \tau$.
\end{defbox}

\begin{defbox}{Informative part codes and productivity}{informative}
A part code $e$ is \emph{informative} at tolerance $\tau$ if an affine map of $e$ recovers a declared description $z(x)$ of the parts that does not depend on the system, with $R^2 \ge 1 - \tau$ on fresh parts. A relation is \emph{productive to depth $n$} if Definition~\ref{def:repcomp} holds when the parts are themselves composites of depth up to $n - 1$, with the state of each composite as its part code.
\end{defbox}"""))
R.append((r"""yet with the code $e(x) = Px$ it is exactly recoverable and role-factorized. A verdict""",
          r"""yet with the code $e(x) = Px$ it is exactly recoverable and role-factorized. Identification among candidates cannot stand in for informativeness, since any injective code separates its parts. A verdict"""))
R.append((r"""(we did not test the clause of Definition~\ref{def:repcomp} that takes child states as part codes)""",
          r"""(we did not test the clause of Definition~\ref{def:informative} that takes child states as part codes)"""))

# 2. Method paragraph of Section 4.3 split into run-ins
R.append((r"""\textbf{Method.} We declare a part code""", r"""\textbf{Role maps.} We declare a part code"""))
R.append((r"""Each test of Proposition~\ref{prop:systematic} then replaces a code by the corresponding prediction and is evaluated on fresh glyphs. \emph{Reversal} predicts""",
          r"""Each test of Proposition~\ref{prop:systematic} then replaces a code by the corresponding prediction, passes the predicted or edited state to the network's own decoder, and is evaluated on fresh glyphs.

\textbf{Reversal and new role.} \emph{Reversal} predicts"""))
R.append((r"""from a composite in which it occupies role 1. The \emph{interchange intervention} edits""",
          r"""from a composite in which it occupies role 1.

\textbf{Interchange and controls.} The \emph{interchange intervention} edits"""))
R.append((r"""which changes both parts; and the network's own change when only the role-1 glyph changes, $h(a', b) - h(a, b)$. Each predicted or edited state is passed to the network's own decoder. Finally, we fine-tune each network""",
          r"""which changes both parts; and the network's own change when only the role-1 glyph changes, $h(a', b) - h(a, b)$.

\textbf{Fine-tuning.} We fine-tune each network"""))

# 3. Crosstalk derivation
R.append((r"""A learned nonlinear composer is covered by the bound for affine readouts, and Section~\ref{sec:depth} compares it with a population readout of the composer's state and with the composer's own decomposer.""",
          r"""A learned nonlinear composer is covered by the bound for affine readouts, and Section~\ref{sec:depth} compares it with a population readout of the composer's state and with the composer's own decomposer.

\noindent
\textbf{Crosstalk in recursive HRR.} The prediction $1 - (2^n - 1)d/w$ of Section~\ref{sec:depth} follows from a short calculation. Leaves $x \in \R^d$ are lifted to $xP$ by $P \in \R^{d \times w}$ with i.i.d.\ $\mathcal{N}(0, 1/w)$ entries, and two subtrees $a, b$ at height $h$ are composed as $(a \circledast \rho^h_1 + b \circledast \rho^h_2)/\sqrt{2}$, with unitary role vectors, for which circular convolution is an orthogonal map that circular correlation inverts exactly. Unbinding one path from the root, by correlation with that path's role vectors and multiplication by $\sqrt{2}$ at each level, returns the target leaf's lift plus, for each of the other $2^n - 1$ leaves, its lift transformed by a product of convolutions with unitary vectors. Each of these crosstalk terms has the norm of the leaf's lift, and the role vectors randomize the phases of its spectrum; since the spectrum of a lifted Gaussian leaf is close to flat, each term is close to isotropic and uncorrelated with the target and with the other terms. The read-out applies the pseudo-inverse of $P$, which for $w \gg d$ is close to an isometry composed with the orthogonal projection onto the $d$-dimensional row space of $P$, and that projection keeps on average a share $d/w$ of an isotropic vector's squared norm. With leaf variance $d$, the expected squared error is therefore $(2^n - 1)d \cdot d/w$, and $R^2 \approx 1 - (2^n - 1)d/w$. The calculation ignores the departure of $P$ from orthonormal rows, of relative order $\sqrt{d/w}$, and is least accurate at small $w$, where the observed values fall below it (Figure~\ref{fig:depth}b)."""))
R.append((r"""recursive unbinding by correlation, which treats crosstalk between superposed bindings as noise, loses about $(2^n - 1)d/w$ of the variance:""",
          r"""recursive unbinding by correlation, which treats crosstalk between superposed bindings as noise, loses about $(2^n - 1)d/w$ of the variance (the calculation after Proposition~\ref{prop:depth}):"""))

# 4. Leak bound, stated and proved
R.append((r"""The leak is a component of the projection of the interaction onto additive functions under the observed design; it vanishes for a complete balanced design and is otherwise at most a design-dependent constant times the interaction's norm on the observed pairs.""",
          r"""To make this precise, take finite filler sets $X$ and $Y$, let $\lVert f \rVert_S^2$ be the mean of $\lVert f(x, y) \rVert^2$ over the pairs in $S \subseteq X \times Y$, and write $g = c + \varphi_1 + \varphi_2 + \iota$, with the interaction $\iota$ orthogonal to every additive function $a(x) + b(y)$ under the uniform measure on $X \times Y$. Let $\Pi_O$ be the least-squares projection onto additive functions under the uniform measure on a connected design $O \subseteq X \times Y$. Because $\Pi_O$ fixes additive functions, the additive fit is $c + \varphi_1 + \varphi_2 + \lambda$, with the leak $\lambda = \Pi_O \iota$ itself additive. (i) If $O = X \times Y$, then $\lambda = 0$, since $\iota$ is orthogonal to the additive functions under the fitting measure. (ii) $\lVert \lambda \rVert_O \le \lVert \iota \rVert_O$, because $\Pi_O$ is an orthogonal projection in $\lVert \cdot \rVert_O$. (iii) $\lVert \lambda \rVert_{X \times Y} \le C_O \lVert \iota \rVert_O$, with $C_O = \sup_f \lVert f \rVert_{X \times Y} / \lVert f \rVert_O$ over nonzero additive $f$. The constant is finite because an additive function that vanishes on a connected design vanishes on every pair (Proposition~\ref{prop:separable}), so $\lVert \cdot \rVert_O$ is a norm on the finite-dimensional space of additive functions; it depends only on the design and grows as the design approaches disconnection."""))

# 5. Kamp and Partee: context-dependent modifiers
R.append((r"""and fails the second when what a part contributes depends on its co-parts, as in prototype combination \citep{smith1988,hampton1987}.""",
          r"""and fails the second when what a part contributes depends on its co-parts, as in prototype combination \citep{smith1988,hampton1987} and in context-dependent modifiers, where \emph{red} picks out different colors in \emph{red hair} and \emph{red apple} \citep{kamp1995}."""))

# 6. Content effects: canonical sources
R.append((r"""\citet{evans1983}; \citet{wason1968}; \citet{griggs1982}""",
          r"""\citet{evans1983}; \citet{wason1968}; \citet{johnsonlaird1972,griggs1982,cheng1985}"""))

# 7. app:human label outside a sectioning command
R.append((r"""Appendix~\ref{app:human} collects this evidence""", r"""Table~\ref{tab:human} in Appendix~\ref{app:cases} collects this evidence"""))
R.append(("\n\\label{app:human}\n", "\n"))

# 8. Uncited bib entries
R.append((r"""binding identifiers in language models \citep{feng2024}, and interchange interventions""",
          r"""binding identifiers in language models \citep{feng2024}, structural subnetworks \citep{lepori2023}, and interchange interventions"""))
R.append((r"""argue that meta-learning \citep{lake2023} has not yet produced human-like systematicity.""",
          r"""argue that meta-learning \citep{lake2023} has not yet produced human-like systematicity. \citet{russin2024} review how the debate bears on deep networks, and \citet{quiltydunn2023} argue that evidence for structured, language-like representations is reemerging across the cognitive sciences."""))
R.append((r"""and the two smaller CLIP encoders, including the one that conditions Stable Diffusion 1.5, gain little beyond the token count of the noun.""",
          r"""and the two smaller CLIP encoders, including the one that conditions Stable Diffusion 1.5, gain little beyond the token count of the noun, consistent with the argument that contrastive training gives vision--language models little incentive to encode word order \citep{yuksekgonul2023}."""))

# 9. Pixel code scaling in Section 4.2
R.append((r"""\textbf{Readouts.} Ridge readouts map $h$ to each role's glyph.""",
          r"""\textbf{Readouts.} Ridge readouts map $h$ to each role's glyph, coded by its 64 pixel intensities in $[0, 1]$ under the Euclidean inner product; $R^2$ pools squared error over pixels, so each pixel weighs in proportion to its variance."""))

bad = 0
for old, new in R:
    n = s.count(old)
    if n != 1:
        print("MATCH COUNT", n, "for:", old[:90]); bad += 1
    else:
        s = s.replace(old, new)
if bad:
    sys.exit(1)
open(P, "w", encoding="utf-8").write(s)
print("applied", len(R))
