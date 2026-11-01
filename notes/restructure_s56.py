p = "main.tex"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)

rep("\\section{Results}\n\\label{sec:results}\n\nIn this section, we report nine results.",
    "\\section{Representation and behavior come apart}\n\\label{sec:results}\n\nIn this section, we report eight results.")
i = s.index("  \\item \\textbf{R9.}"); j = s.index("\n", i) + 1
s = s[:i] + s[j:]
rep("R5--R7 measure representation alone in models whose outputs the tests do not score, and R9 tests remedies.",
    "and R5--R7 measure representation alone in models whose outputs the tests do not score.")

i = s.index("\\subsection{Reading through the state's factorization repairs the failing procedures}")
j = s.index("%=====================================================================\n\\section{Conclusion and limitations}")
s = s[:i] + r"""%=====================================================================
\section{Repairing the failing procedures}
\label{sec:remedies}

Each failure above leaves the parts in the state and a procedure that cannot read them, so each suggests training or reading that procedure through the factorization the state already has. In this section, we test four such repairs on fresh parts, one for the decoder that fails on fresh glyphs, one for a network that must add parts, one for the decomposer that fails at depth, and one for the sentence states that follow word order (Table~\ref{tab:remedies}).

\paragraph{Single parts teach the decoder to render parts.} Networks trained on all pairs of 8 glyphs and on 64 other glyphs shown alone, each in one role with the other role blank, render a fresh glyph alone with $R^2 = 0.88$, against $-0.30$ without them, and fresh pairs with 0.47, against 0.26.

\paragraph{Written codes add parts without forgetting.} A network adds 32 new glyphs in sequence by reading each glyph's code from five composites and writing it through the role maps. Old pairs stay at 0.99, and the decoder renders each new glyph in a role it never filled at 0.71 and two new glyphs together at 0.55, matching the unchanged network. Fine-tuning on the same composites drops old pairs to 0.18, and replay of old pairs keeps them at 1.00 with new-glyph values 0.11--0.21 higher than the written codes, at the cost of storing old data and taking gradient steps.

\paragraph{Training at depth repairs the decomposer.} A decomposer that trains with its composer on trees of depth 3 to 5 with Gaussian leaves recovers fresh leaves at 1.00, 1.00 and 0.97 at width 256, the values of the population readout, and at width 64 it meets the bound of Proposition~\ref{prop:depth}. Trained alone on the composer's states, the decomposer reaches 0.74, 0.29 and 0.09, and no condition transfers to depth 6.

\paragraph{Role questions give sentence embeddings that keep roles.} A sentence embedding read at the end of two questions, one asking for the figure and one for the ground, places the converse paraphrase nearer than both role reversals for 100\% of fresh pairs in phi-4, 92\% in Qwen3-8B and 83\% in OLMo-2 13B, with no fitted readout. Token-averaged and final-token states never do, and the same embedding fails in Pythia-1.4b and GPT-2-large, whose answers follow word order.

""" + s[j:]

rep("We studied the conditions under which a representation counts as compositional, requiring that a fixed, declared readout recover every part in its role from composites of parts the readout never saw.",
    "We separated behavioral from representational compositionality and measured both by recoverability, the return of every part in its role from composites of parts fresh to the reading procedure, by a declared readout for the states and by the system's own procedure for behavior.")
open(p, "w", encoding="utf-8").write(s)
print("ok")
