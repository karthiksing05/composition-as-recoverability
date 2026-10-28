p = "/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/main.tex"
s = open(p, encoding="utf-8").read()
R = [
 (r"With $\kappa \le 0.50$, no encoder is representationally compositional under these part codes and this metric at a tolerance below about one half.",
  r"The concatenation reference is recovered exactly by construction, because its readout lies in the span of the fitting nouns' states. A role--filler reference that mixes the two roles' codes through a random, non-orthogonal map of full rank, and so faces the same estimation problem as a model, is recovered with 0.92--0.98; against it, $\kappa$ rises by at most 0.06 (GPT-2 0.47--0.52, CLIP-H 0.42). With $\kappa \le 0.52$ against either reference, no encoder is representationally compositional under these part codes and this metric at a tolerance below about one half."),
 (r"For the random-weight model scored on the trained model's codes, layer and code are chosen on its own validation score.",
  r"The mixed role--filler reference is $[s(x), s(y)]\,R$, with $s$ the full part state and $R$ a random Gaussian matrix of size $2D \times 2D$ (one per seed), evaluated at each model's selected layer and code with the same nouns, pairs, and readouts. For the random-weight model scored on the trained model's codes, layer and code are chosen on its own validation score."),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
