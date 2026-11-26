M = "/coc/flash10/ksingara3/compositionality-theory/icml2027/main.tex"
t = open(M, encoding="utf-8").read()
R = [
 (r" Per-prompt layer selection lowers $r$ to 0.77--0.83, because it picks early layers for phi-4 on two prompts (0.08--0.12).", ""),
 (r"and in phi-4 and Qwen3-8B wherever validation picks an early layer.",
  r"and in phi-4 and Qwen3-8B wherever per-prompt validation picks an early layer, which lowers $r$ to 0.77--0.83."),
]
for a, b in R:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open(M, "w", encoding="utf-8").write(t); print("ok")
