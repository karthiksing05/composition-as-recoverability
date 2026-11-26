B = "/coc/flash10/ksingara3/compositionality-theory/icml2027/"
def apply(fn, pairs):
    t = open(B + fn, encoding="utf-8").read()
    for a, b in pairs:
        n = t.count(a); assert n == 1, (fn, n, a[:80]); t = t.replace(a, b)
    open(B + fn, "w", encoding="utf-8").write(t); print(fn, "ok", len(pairs))

apply("main.tex", [
(r"so the image tower loses the places before any text enters (Appendix~\ref{app:images}).",
 r"so the image tower loses the places before any text enters. A matcher that reads the quadrant state assigns the two photographs to their roles on 1.00 of canvases, which repairs the judgment (H3; Appendix~\ref{app:images})."),
(r"the trained encoders fall to $\kappa = 0.24$--$0.45$ and random weights fall below zero (Figure~\ref{fig:places}a),",
 r"the trained encoders fall to $\kappa = 0.24$--$0.45$ and random CLIP-B/32 and DINOv2-L weights fall below zero (Figure~\ref{fig:places}a),"),
(r"and on at most 0.22 in the first half of the layers",
 r"and on at most 0.21 in the first half of the layers"),
(r"In pretrained models, parts are new only to the probe, apart from invented words, image encoders lack a tested repair, and attribute binding remains untested (Appendix~\ref{app:limits}).",
 r"In pretrained models, parts are new only to the probe, apart from invented words, and image systems lack a tested attribute binding (Appendix~\ref{app:limits})."),
(r"We test four hypotheses on parts new to every fitted procedure.",
 r"We test four hypotheses on parts new to every fitted procedure, and Appendix Table~\ref{tab:verdicts} gives one verdict per hypothesis and system."),
])

VERD = r"""

\begin{table*}[t]
  \caption{\textbf{One verdict per hypothesis and system.} H1: correct behavior on new parts implies a linearly recoverable state. H2: behavior lags the state. H3: a procedure that reads the state lifts behavior. H4: training builds recoverability beyond the input and the random-weight architecture, each model scored against its own part vectors. ``Vacuous'': the system's own procedure never succeeds, so H1 has nothing to test.}
  \label{tab:verdicts}
  \centering\footnotesize
  \begin{tabularx}{\textwidth}{@{}lYYYY@{}}
    \toprule
    System & H1 & H2 & H3 & H4 \\
    \midrule
    Glyph networks & Holds: $V = 64$ decoders 0.85--0.91, probe 0.89 & Holds: $V = 8$ decoders 0.20--0.34, probe 0.83--0.86 & Holds: new decoder 0.75 & Fails: input pixels 0.91, untrained encoders 0.82--0.83 \\
    Recursive composers & Holds: decomposers trained deeper match the probe & Holds: depth 3, 0.26 against 0.99 & Holds: decomposer trained with the composer, 0.97--1.00 & Untested \\
    Image encoders & Vacuous on swapped places & Holds; the pooled embedding loses the places & Holds: state matcher 1.00 fixed, 0.92--0.97 moved & Holds: 0.81--0.94 against 0.29--0.72 \\
    U-Net denoisers & Vacuous & Holds: chance against $\kappa$ 0.43--0.70 & Untested & Fails: random 0.79--0.83 \\
    Transformer denoisers & Holds: 0.59--0.68 with $\kappa$ 0.75--0.89 & Holds & Untested & PixArt-$\Sigma$ holds (0.58 against 0.75), Lumina fails (0.87 against 0.89) \\
    LMs answering by meaning & Holds at main-prompt layers; fails for linear probes in OLMo-2 7B and where validation picks early layers & Not observed & Holds for harnesses whose code assigns the roles & Holds: trained $\kappa$ above random \\
    LMs answering by word order & Vacuous & Not observed & Holds: parse harness 0.51 to 0.98 & Holds: Pythia $\kappa$ grows with scale \\
    \bottomrule
  \end{tabularx}
\end{table*}
"""
t = open(B + "parts/results.tex", encoding="utf-8").read()
first = t.index("\n\n")
t = t[:first] + VERD + t[first:]
pairs = [
(r"    CLIP, SigLIP & 0.88--0.97 & 0.50--0.51 & 0.81--0.94 & 0.29--0.72 & 0.17--0.49 &",
 r"    CLIP, SigLIP & 0.88--0.97 & 0.50--0.51 & 0.81--0.94 & 0.29--0.72 & 0.26--0.49 &"),
]
for a, b in pairs:
    assert t.count(a) == 1, a[:60]; t = t.replace(a, b)
open(B + "parts/results.tex", "w", encoding="utf-8").write(t); print("results ok")
