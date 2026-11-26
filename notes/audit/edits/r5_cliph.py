D = "/coc/flash10/ksingara3/compositionality-theory/icml2027/"
def apply(path, R):
    t = open(D + path, encoding="utf-8").read()
    for a, b in R:
        assert t.count(a) == 1, (path, a[:80], t.count(a))
        t = t.replace(a, b)
    open(D + path, "w", encoding="utf-8").write(t)
apply("main.tex", [
 (r"keeps less than the sum of the parts ($\kappa = -0.30$ to $-0.15$), and matching a canvas against other canvases picks the same arrangement at chance and the right objects on 0.78--0.88,",
  r"keeps less than the sum of the parts ($\kappa = -0.45$ to $-0.15$), and matching a canvas against other canvases picks the same arrangement at chance and the right objects on 0.70--0.88,"),
])
apply("parts/results.tex", [
 (r"\begin{tabular}{@{}lcccccc@{}}" + "\n    \\toprule\n     & \\multicolumn{2}{c}{CLIP-B/32} & \\multicolumn{2}{c}{DINOv2-L}",
  r"\begin{tabular}{@{}lcccccccc@{}}" + "\n    \\toprule\n     & \\multicolumn{2}{c}{CLIP-B/32} & \\multicolumn{2}{c}{CLIP-H/14} & \\multicolumn{2}{c}{DINOv2-L}"),
 (r"     & Fixed & Moved & Fixed & Moved & Fixed & Moved \\", r"     & Fixed & Moved & Fixed & Moved & Fixed & Moved & Fixed & Moved \\"),
 (r"Order state, $\kappa$ & 1.00 & 1.00 &", r"Order state, $\kappa$ & 1.00 & 1.00 & 1.00 & 1.00 &"),
 (r"Position state, $\kappa$ & 1.00 & $-0.05$ &", r"Position state, $\kappa$ & 1.00 & $-0.05$ & 1.00 & $-0.03$ &"),
 (r"Pooled embedding, $\kappa$ & $-0.26$ & $-0.67$ &", r"Pooled embedding, $\kappa$ & $-0.26$ & $-0.67$ & $-0.45$ & $-0.57$ &"),
 (r"Patch quadrants, $\kappa$ & 0.81 & 0.24 &", r"Patch quadrants, $\kappa$ & 0.81 & 0.24 & 0.89 & 0.24 &"),
 (r"Image matching, arrangement & 0.50 & 0.50 &", r"Image matching, arrangement & 0.50 & 0.50 & 0.51 & 0.53 &"),
 (r"Image matching, objects & 0.78 & 0.60 &", r"Image matching, objects & 0.78 & 0.60 & 0.70 & 0.56 &"),
])
print("ok")
