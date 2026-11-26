B = "/coc/flash10/ksingara3/compositionality-theory/icml2027/main.tex"
t = open(B, encoding="utf-8").read()
P = [
(r"Across widths 16 to 256, wherever the bound of Proposition~\ref{prop:depth} falls below 1, the probe meets it within 0.04, for instance 0.49 against 0.50 at width 64 and depth 4 (Appendix~\ref{app:depth}).",
 r"Wherever the bound of Proposition~\ref{prop:depth} falls below 1, the probe meets it within 0.04 (Appendix~\ref{app:depth})."),
(r" (prompts verbatim; the first gallery pair of Appendix Figure~\ref{fig:case_generation} whose phrasings share a layout and misplace the figure)",
 r" (Appendix Figure~\ref{fig:case_generation})"),
(r"When each prompt selects its own layer, $r$ falls to 0.77--0.83, because validation then picks early layers for phi-4 on two prompts, where transfer falls to 0.08--0.12.",
 r"Per-prompt layer selection lowers $r$ to 0.77--0.83, because it picks early layers for phi-4 on two prompts (0.08--0.12)."),
(r"(iii) Evidence from toy networks that a reader fitted on few parts lags the state it reads, from image encoders and diffusion models that their states keep fixed places their own judgments ignore, and from language models up to 14 billion parameters that behavior can outrun a linear probe.",
 r"(iii) Evidence that readers fitted on few parts lag their states in toy networks, that image systems ignore places their states keep, and that language models up to 14 billion parameters can answer beyond a linear probe."),
(r"Toy and image systems favor a strict inclusion (Figure~\ref{fig:ladder}): networks trained on eight glyphs fail to draw new glyphs that a linear probe of their state recovers, and image systems ignore places that their states keep. Language models that answer by meaning sit on or above the diagonal, so H1 fails for linear probes there.",
 r"Toy and image systems favor a strict inclusion: their behavior lags states that a linear probe reads (Figure~\ref{fig:ladder}). Language models that answer by meaning sit on or above the diagonal, so H1 fails for linear probes there."),
]
for a, b in P:
    assert t.count(a) == 1, a[:60]; t = t.replace(a, b)
open(B, "w", encoding="utf-8").write(t); print("ok")
