p = "main.tex"
s = open(p, encoding="utf-8").read()

def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)

# --- Section 3 intro
rep("In this section, we define recoverable and role-factorized relations and state the results that follow from them. Appendix~\\ref{app:proofs} gives the proofs.",
    "In this section, we define behavioral and representational compositionality, introduce recoverability as the property that separates them, and state the results that follow. Appendix~\\ref{app:proofs} gives the proofs.")

# --- Definitions block: from "\subsection{The definition}" to the two-by-two table
i = s.index("\\subsection{The definition}")
j = s.index("\\begin{table}[tbp]\n  \\caption{\\textbf{Behavioral and representational compositionality vary independently}")
old = s[i:j]
eq_factor = old[old.index("\\begin{equation}\n  g_r(x_1"):]
eq_factor = eq_factor[:eq_factor.index("\\end{equation}") + len("\\end{equation}")]
eq_recover = old[old.index("\\begin{equation}\n  \\frac{\\mathbb{E}"):]
eq_recover = eq_recover[:eq_recover.index("\\end{equation}") + len("\\end{equation}")]
eq_kappa = old[old.index("\\begin{equation}\n  \\kappa"):]
eq_kappa = eq_kappa[:eq_kappa.index("\\end{equation}") + len("\\end{equation}")]
inf = old[old.index("\\begin{defbox}{Informative part codes and productivity}"):]
inf = inf[:inf.index("\\end{defbox}") + len("\\end{defbox}")]

new = r"""\subsection{Two notions of compositionality}
\label{sec:notions}

The behavioral tradition judges a system by its outputs on new combinations, and the representational tradition judges it by the way its states are built. We define each without reference to the other.

\begin{defbox}{Behavioral compositionality}{behcomp}
A system is \emph{behaviorally compositional} with respect to a relation $r$ to the degree that its own procedures, such as a decoder, a decomposer or its answers to questions about the roles, give the correct output for composites of parts fresh to those procedures.
\end{defbox}

\begin{defbox}{Representational compositionality}{repcomp}
Fix a part distribution $\D$ and a declared class $\M$ of methods for finding part codes. A system is \emph{representationally compositional} with respect to a relation $r$, which it realizes by $g_r$, to the degree that its realization of $r$ is \emph{$\delta$-role-factorized} for some part code $e$ found under $\M$ that is informative about the parts (Definition~\ref{def:informative}). The relation is $\delta$-role-factorized if, for some code $e'$ of dimension $d'$ of which $e$ is an affine function, some $c \in \R^m$ and injective matrices $B_1, \ldots, B_k \in \R^{m \times d'}$, one per role and shared by all parts, satisfy
""" + eq_factor + r"""
The system is representationally compositional at tolerance $\tau$ if $\delta \le \tau$ for some such $e$ informative at $\tau$.
\end{defbox}

""" + inf + r"""

Behavioral compositionality scores the system's procedures together with its states, and representational compositionality scores the states alone. Role-factorization asks each part to contribute the same code to every whole, transformed only by its role, the classical demand that a constituent contribute the same thing to every complex containing it \citep{fodor1988}. The maps $B_i$ must be injective, because otherwise $e'$ could hold a separate code for each role and two unrelated lookup tables would pass. A relation fails role-factorization when a part's contribution depends on its co-parts, as \emph{red} does in \emph{red hair} and \emph{red apple} \citep{kamp1995}. The informativeness clause closes a second loophole. The blend $(Px, Py)$, with $P$ a projection of rank $r < d$, discards $d - r$ dimensions of each part, yet with the code $e(x) = Px$ it is role-factorized with $\delta = 0$. We report informativeness wherever a description of the parts exists apart from the system, which holds for glyph latents and fails for nouns and photographs. The definition depends on $\M$ because only a search can find a decomposition, and a negative result reports that the declared methods found none.

\subsection{Recoverability separates the two notions}
\label{sec:recoverability}

Neither definition says how the parts come back out of a whole, yet both imply that they do. A role-factorized state with full-rank role maps returns every part's code by an affine map, with an error that grows with $\delta$ (Propositions~\ref{prop:systematic} and~\ref{prop:approx}), and a behaviorally compositional system returns the parts through its own procedures. We call this shared property recoverability.

\begin{defbox}{Recoverable relations}{recoverable}
Fix a part distribution $\D$, a part code $e$, and a readout class $\U$. A relation $r$, which a system realizes by $g_r$, is \emph{$\varepsilon$-recoverable} if for every role $i$ some $u_i \in \U$ satisfies, for parts $x_1, \ldots, x_k$ drawn independently from $\D$,
""" + eq_recover + r"""
We estimate this quantity and $\delta$ by fitting $u_i$, $c$ and $B_i$ on composites of a \emph{probe pool} of parts and evaluating them on composites of other parts, with errors in a declared inner product on part codes. We report recovery by
""" + eq_kappa + r"""
where the same procedure gives $\varepsilon_{\mathrm{ref}}$ for the concatenation and $\varepsilon_{\mathrm{blend}}$ for the sum of the same part codes.
\end{defbox}

Recoverability puts the two notions on one scale. A readout that the experimenter declares measures the representation, and a representationally compositional system with well-conditioned role maps gives $\kappa$ near 1. The system's own procedure measures behavior, scored by the error of Eq.~\ref{eq:recover} when it outputs parts and by accuracy when it chooses among them. Both act on composites of the same fresh parts, so the gap between them dissociates the two properties (Table~\ref{tab:twobytwo}). A system that meets Definitions~\ref{def:behcomp} and~\ref{def:repcomp} is a \emph{compositional system}.

Recoverability asks a fixed procedure of a declared kind to return each part in its role across the whole part distribution, and $1 - \varepsilon$ is the population $R^2$ of the best readout in $\U$. A readout fitted on a small vocabulary is undetermined outside the codes it saw (Proposition~\ref{prop:span}), and scoring it on that vocabulary inflates it (Lemma~\ref{lem:lookup}). Fitting on a large probe pool and scoring on fresh parts removes both biases. A relation fails recoverability when it discards information about its parts. For a symmetric relation of identically distributed parts, no readout can tell the roles apart, so in population $\varepsilon_{\mathrm{blend}} = 1/2$, $\varepsilon_{\mathrm{ref}} = 0$ and $\kappa = 1 - 2\varepsilon$. The references correct the estimates for finite samples and readout fitting. Unrestricted readouts let any injective system pass, as unrestricted probes \citep{hewitt2019} and causal alignment maps \citep{sutter2025} do, so the readout class is declared in advance.

"""
s = s[:i] + new + s[j:]

# references to the factorization equation and the table of modes
rep("Definition~\\ref{def:recoverable} therefore fixes the coordinates to be affine (Eq.~\\ref{eq:factor}).",
    "Definition~\\ref{def:repcomp} therefore fixes the coordinates to be affine (Eq.~\\ref{eq:factor}).")
rep("\\caption{Modes of combination and the properties of Definition~\\ref{def:recoverable}.}",
    "\\caption{Modes of combination and the properties of Definitions~\\ref{def:repcomp} and~\\ref{def:recoverable}.}")
rep("Table~\\ref{tab:forms} classifies widely discussed modes of combination by the properties of Definition~\\ref{def:recoverable}.",
    "Table~\\ref{tab:forms} classifies widely discussed modes of combination by the properties of Definitions~\\ref{def:repcomp} and~\\ref{def:recoverable}.")
open(p, "w", encoding="utf-8").write(s)
print("ok")
