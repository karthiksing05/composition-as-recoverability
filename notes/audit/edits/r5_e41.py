D = "/coc/flash10/ksingara3/compositionality-theory/icml2027/"
def apply(path, R):
    t = open(D + path, encoding="utf-8").read()
    for a, b in R:
        assert t.count(a) == 1, (path, a[:80], t.count(a))
        t = t.replace(a, b)
    open(D + path, "w", encoding="utf-8").write(t)
apply("main.tex", [
 (r"and on at most 0.21 in the first half of the layers (Figure~\ref{fig:both}c,d).",
  r"and on at most 0.21 in the first half of the layers (Figure~\ref{fig:both}c,d). Where the answer goes wrong, the question state returns the first-mentioned noun in either phrasing (0.66--0.99 of error items), so the state falls back to order of mention wherever the procedure fails (Appendix~\ref{app:behavior})."),
])
P = (r"\textbf{On error items the question state follows order of mention.} For phi-4, Qwen3-8B and OLMo-2 7B and 13B we score the figure question on 1{,}000 new-noun pairs per relation and phrasing, over 3 seeds, and read the figure probe of the question state at the layer of Table~\ref{tab:lm}. "
     r"Where the model answers wrongly on a canonical sentence, it names the ground, while the probe still returns the figure, the first-mentioned noun, on 0.88 (Qwen3-8B, 199 items), 0.82 (OLMo-2 13B, 84) and 0.99 (OLMo-2 7B, 2{,}811) of these items. "
     r"Where it answers wrongly on a converse sentence, it names the first-mentioned ground, and the probe follows it on 0.69 (phi-4, 48 items), 0.66 (Qwen3-8B, 64), 0.81 (OLMo-2 13B, 1{,}245) and 0.85 (OLMo-2 7B, 1{,}386). "
     r"On correct items the probe returns the figure on 0.98--0.99 (phi-4), 0.88--0.98 (Qwen3-8B) and 0.84--1.00 (OLMo-2 13B) of items, while in OLMo-2 7B it returns the figure on only 0.21 of correct converse items, so the layer that its validation picks reads order of mention even where the model answers by meaning. "
     r"In error items, order of mention and the wrong answer coincide on converse sentences and diverge on canonical ones, and the probe follows order of mention in both." "\n\n")
apply("parts/results.tex", [
 (r"\textbf{OLMo-2 7B decomposes inside its answering layers.}", P + r"\textbf{OLMo-2 7B decomposes inside its answering layers.}"),
])
print("ok")
