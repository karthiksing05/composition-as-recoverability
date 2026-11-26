B = "/coc/flash10/ksingara3/compositionality-theory/icml2027/parts/results.tex"
t = open(B, encoding="utf-8").read()
a = r"random weights at 1{,}024 pixels remain untested."
assert t.count(a) == 1
t = t.replace(a, a + r" The transformer classifiers change with the timestep. Lumina prefers the wrong caption in both tests at $t \le 575$ (0.31--0.43 against swapped places, 0.34--0.45 against a wrong object) and the right one from $t = 688$ on (0.54--0.65 and 0.56--0.72), and PixArt-$\Sigma$ rises from 0.47 to 0.59 against swapped places, so the means over 25 timesteps mix a caption bias at low noise with the judgment at high noise.")
i = t.index(r"\label{tab:harness}")
j = t.rindex(r"\caption{", 0, i)
k = t.index("}\n", j)  # end of caption line
cap = t[j:k]
assert cap.count("\n") == 0, "caption spans lines"
t = t[:k] + r" SD 1.5 in the bottom block uses the harness prompts (about 102 per cell), so its range differs from Table~\ref{tab:generation}." + t[k:]
open(B, "w", encoding="utf-8").write(t); print("ok")
