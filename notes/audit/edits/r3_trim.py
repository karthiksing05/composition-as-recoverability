M = "/coc/flash10/ksingara3/compositionality-theory/icml2027/main.tex"
t = open(M, encoding="utf-8").read()
R = [
 (r"The states inherit the glyphs from the input, whose pixels a probe recovers at 0.91, and untrained encoders keep 0.82--0.83 (Appendix~\ref{app:origins}).",
  r"The states inherit the glyphs from the input, and untrained encoders keep them (Appendix~\ref{app:origins})."),
 (r"yet the U-Nets score 0.49--0.51 against swapped places, also SDXL at its native 1{,}024 pixels, at every timestep and the transformers only 0.59--0.68",
  r"yet the U-Nets score 0.49--0.51 against swapped places at every timestep, SDXL also at its native 1{,}024 pixels, and the transformers only 0.59--0.68"),
 (r" places twenty-two; an account whose decomposer sees only training parts is weak, because a lookup takes apart every whole of training parts.",
  r" places twenty-two."),
 (r"including known parts in roles they never filled", r"including known parts in new roles"),
 (r"the probe included, because a lookup takes apart held-out combinations of training parts (Lemma~\ref{lem:lookup}).",
  r"the probe included (Lemma~\ref{lem:lookup})."),
]
for a, b in R:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open(M, "w", encoding="utf-8").write(t); print("ok")
