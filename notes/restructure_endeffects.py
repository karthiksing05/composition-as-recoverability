p = "main.tex"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)

# ---------------- Introduction: origins and end effects first
i = s.index("People understand sentences they have never heard by building their meaning from familiar parts")
j = s.index("Answers to the representational question usually start from a principle about meaning.")
s = s[:i] + r"""Compositionality entered cognitive science and machine learning through two observations about what a compositional system does. The first concerns behavior. People understand sentences they have never heard by building their meaning from familiar parts \citep{frege1923}, and anyone who can think that John loves Mary can think that Mary loves John \citep{fodor1988}. Deep learning inherited this observation as compositional generalization, the production of correct outputs for novel combinations of familiar atoms \citep{lake2018,keysers2020,kim2020cogs,hupkes2020}. The end effect lies in the system's actions, which stay correct on combinations it never met. The second observation concerns what a system keeps. A classical representation contains its constituents, so any process can find the parts of a whole in it \citep{fodor1988}, and connectionist representations keep constituents that general procedures can read back out \citep{smolensky1990,vangelder1990}. The end effect lies in the state, from which every part of a whole returns in its role.

We call the first property \emph{behavioral} and the second \emph{representational} compositionality. The two can come apart (Table~\ref{tab:twobytwo}). A network can keep every part of a new combination in its state and fail to produce the combination, and a network can produce correct outputs from states from which no declared procedure reads the parts. Text-to-image models compose concepts into new scenes \citep{ramesh2021,okawa2023}, yet they often misplace objects \citep{conwell2022} and attach an attribute to the wrong object \citep{rassin2022}, and these failures say nothing definite about whether their states keep the objects apart. Both properties concern the return of parts from a whole, through the system's own procedures in one case and through a procedure that the experimenter declares in the other. We therefore use \emph{recoverability}, the return of each part in its role from composites of parts fresh to the reading procedure, to define the representational property and to separate it from the behavioral one.

""" + s[j:]

rep(r"\textbf{Contributions.} We define behavioral compositionality by a system's own outputs on composites of fresh parts and representational compositionality by a state in which each part enters every role through one shared code, and we introduce recoverability by a declared readout as the property that separates them.",
    r"\textbf{Contributions.} We define representational compositionality as the recovery of every part in its role, by a declared readout, from composites of parts fresh to the readout, and behavioral compositionality as the same success of the system's own procedures, so the two can be measured and dissociated on the same parts.")

# ---------------- Section 3: definitions by end effect, then mechanics
rep("In this section, we define behavioral and representational compositionality, introduce recoverability as the property that separates them, and state the results that follow. Appendix~\\ref{app:proofs} gives the proofs.",
    "In this section, we define behavioral and representational compositionality by their end effects, describe the structure that produces recoverable states, and state the results that follow. Appendix~\\ref{app:proofs} gives the proofs.")

i = s.index("\\subsection{Two notions of compositionality}")
j = s.index("\\subsection{Four consequences}")
old = s[i:j]
def blk(start, end):
    a = old.index(start); b = old.index(end, a) + len(end); return old[a:b]
eq_factor = blk("\\begin{equation}\n  g_r(x_1", "\\end{equation}")
eq_recover = blk("\\begin{equation}\n  \\frac{\\mathbb{E}", "\\end{equation}")
eq_kappa = blk("\\begin{equation}\n  \\kappa", "\\end{equation}")
inf = blk("\\begin{defbox}{Informative part codes and productivity}", "\\end{defbox}")
behbox = blk("\\begin{defbox}{Behavioral compositionality}", "\\end{defbox}")
table = blk("\\begin{table}[tbp]\n  \\caption{\\textbf{Behavioral and representational compositionality vary independently}", "\\end{table}")

new = r"""\subsection{Two notions of compositionality}
\label{sec:notions}

We define both notions by their end effects. Behavioral compositionality concerns the system's own procedures, and representational compositionality concerns its states, read by a procedure that the experimenter declares.

""" + behbox + r"""

\begin{defbox}{Recoverable relations}{recoverable}
Fix a part distribution $\D$, a part code $e$, and a readout class $\U$. A relation $r$, which a system realizes by $g_r$, is \emph{$\varepsilon$-recoverable} if for every role $i$ some $u_i \in \U$ satisfies, for parts $x_1, \ldots, x_k$ drawn independently from $\D$,
""" + eq_recover + r"""
We estimate this quantity by fitting $u_i$ on composites of a \emph{probe pool} of parts and evaluating it on composites of other parts, with errors in a declared inner product on part codes.
\end{defbox}

\begin{defbox}{Representational compositionality}{repcomp}
A system is \emph{representationally compositional} with respect to a relation $r$, relative to a declared readout class $\U$ and a declared class $\M$ of methods for finding part codes, to the degree that its realization of $r$ is $\varepsilon$-recoverable for some part code $e$ found under $\M$ that is informative about the parts (Definition~\ref{def:informative}). We report the degree by
""" + eq_kappa + r"""
where the same procedure gives $\varepsilon_{\mathrm{ref}}$ for the concatenation and $\varepsilon_{\mathrm{blend}}$ for the sum of the same part codes. The system is representationally compositional at tolerance $\tau$ if $\kappa \ge 1 - \tau$ for some such $e$ informative at $\tau$.
\end{defbox}

""" + inf + r"""

Recoverability puts the two notions on one scale. A readout that the experimenter declares measures the representation, and the system's own procedure measures behavior, scored by the error of Eq.~\ref{eq:recover} when it outputs parts and by accuracy when it chooses among them. Both act on composites of the same fresh parts, so the gap between them dissociates the two properties (Table~\ref{tab:twobytwo}). A system that meets Definitions~\ref{def:behcomp} and~\ref{def:repcomp} is a \emph{compositional system}.

Recoverability asks a fixed procedure of a declared kind to return each part in its role across the whole part distribution, and $1 - \varepsilon$ is the population $R^2$ of the best readout in $\U$. A readout fitted on a small vocabulary is undetermined outside the codes it saw (Proposition~\ref{prop:span}), and scoring it on that vocabulary inflates it (Lemma~\ref{lem:lookup}). Fitting on a large probe pool and scoring on fresh parts removes both biases. A relation fails recoverability when it discards information about its parts. For a symmetric relation of identically distributed parts, no readout can tell the roles apart, so in population $\varepsilon_{\mathrm{blend}} = 1/2$, $\varepsilon_{\mathrm{ref}} = 0$ and $\kappa = 1 - 2\varepsilon$. The references correct the estimates for finite samples and readout fitting. Unrestricted readouts let any injective system pass, as unrestricted probes \citep{hewitt2019} and causal alignment maps \citep{sutter2025} do, so the readout class is declared in advance. The informativeness clause closes a second loophole. The blend $(Px, Py)$, with $P$ a projection of rank $r < d$, discards $d - r$ dimensions of each part, yet with the code $e(x) = Px$ it is exactly recoverable. We report informativeness wherever a description of the parts exists apart from the system, which holds for glyph latents and fails for nouns and photographs. The definition depends on $\M$ because only a search can find a decomposition, and a negative result reports that the declared methods found none.

""" + table + r"""

\subsection{Shared codes produce recoverable states}
\label{sec:mechanics}

Recoverability describes an end effect, and a state can produce it in several ways. The simplest gives each part one code that enters every whole, changed only by its role.

\begin{defbox}{Role-factorized relations}{factor}
A relation $r$, which a system realizes by $g_r$, is \emph{$\delta$-role-factorized} if, for some code $e'$ of dimension $d'$ of which the part code $e$ is an affine function, some $c \in \R^m$ and injective matrices $B_1, \ldots, B_k \in \R^{m \times d'}$, one per role and shared by all parts, satisfy
""" + eq_factor + r"""
We estimate $\delta$ by fitting $c$ and $B_i$ on composites of the probe pool and evaluating them on composites of other parts.
\end{defbox}

A role-factorized state with full-rank role maps returns every part's code by an affine map, with an error that grows with $\delta$, and one composite then fixes every role reversal (Propositions~\ref{prop:systematic} and~\ref{prop:approx}). Role-factorization is the classical demand that a constituent contribute the same thing to every complex containing it \citep{fodor1988}. The maps $B_i$ must be injective, because otherwise $e'$ could hold a separate code for each role and two unrelated lookup tables would pass. A relation fails role-factorization when a part's contribution depends on its co-parts, as \emph{red} does in \emph{red hair} and \emph{red apple} \citep{kamp1995}. The residual $\delta$ therefore describes how a state is built. A state that gives each role its own code can be recoverable with a large $\delta$, so $\delta$ diagnoses the mechanism behind recoverability and adds no condition to Definition~\ref{def:repcomp}.

"""
s = s[:i] + new + s[j:]

rep("Definition~\\ref{def:repcomp} therefore fixes the coordinates to be affine (Eq.~\\ref{eq:factor}).",
    "Definition~\\ref{def:factor} therefore fixes the coordinates to be affine (Eq.~\\ref{eq:factor}).")
rep("\\caption{Modes of combination and the properties of Definitions~\\ref{def:repcomp} and~\\ref{def:recoverable}.}",
    "\\caption{Modes of combination and the properties of Definitions~\\ref{def:recoverable} and~\\ref{def:factor}.}")
rep("Table~\\ref{tab:forms} classifies widely discussed modes of combination by the properties of Definitions~\\ref{def:repcomp} and~\\ref{def:recoverable}.",
    "Table~\\ref{tab:forms} classifies widely discussed modes of combination by the properties of Definitions~\\ref{def:recoverable} and~\\ref{def:factor}.")
open(p, "w", encoding="utf-8").write(s)
print("ok")
