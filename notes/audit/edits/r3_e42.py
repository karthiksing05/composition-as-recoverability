M = "/coc/flash10/ksingara3/compositionality-theory/icml2027/main.tex"
B = "/coc/flash10/ksingara3/compositionality-theory/icml2027/parts/results.tex"
t = open(M, encoding="utf-8").read()
R = [
 ("Every sentence and image state we tested binds roles to position, by order of mention",
  "Under a linear probe, every sentence and image state we tested binds roles to position, by order of mention"),
 ("every sentence and image state we tested bound roles to position, and harnesses",
  "every sentence and image state we tested bound roles to position for a linear probe, and harnesses"),
 ("Every sentence and image state we tested passes under positional roles and fails under converse or moved ones,",
  "Every sentence and image state we tested passes under positional roles and fails under converse or moved ones for a linear probe,"),
 (r"and every encoder falls below zero once a probe must find the figure from two box features in random order.",
  r"and every encoder falls below zero once a linear probe must find the figure from two box features in random order. An MLP probe recovers 0.43--0.68 there, against below zero for random weights, and 0.40--0.49 from a state that tags each photograph with its absolute horizontal position alone, so training puts position into the box features, and a nonlinear reader turns it into roles (Appendix~\ref{app:images})."),
 (r"and one that binds it to absolute position fails at about 0.", r"and one that binds it to absolute position fails at about 0 under a linear probe."),
]
for a, b in R:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open(M, "w", encoding="utf-8").write(t)
s = open(B, encoding="utf-8").read()
anchor = r"\textbf{Controls for the image results.}"
assert s.count(anchor) == 1
P = (r"\textbf{Nonlinear probes on moved photographs.} We refit the readouts of the relative-placement protocol with an MLP probe (one hidden layer of 256 units, weight decay chosen on validation) and score it against the linear references, at the selected layer, over 3 seeds. "
     r"For quadrant averages the MLP adds nothing: moved $\kappa$ stays at 0.08--0.36 for the trained encoders. "
     r"For box features in random order it lifts moved $\kappa$ from $-0.17$ to $-0.08$ (linear) to 0.43 for CLIP-B/32 and 0.67--0.68 for DINOv2-L and SigLIP, below the 0.76--0.86 that it reaches at fixed places, while random CLIP-B/32 and DINOv2-L weights stay at $-0.63$ and $-0.29$. "
     r"Three constructed states with shuffled slots calibrate the reading. Without any order or position tag, both probes stay at 0.02--0.07. With an additive left-to-right tag, the linear probe stays at chance and the MLP reaches 1.00. With an additive code of each photograph's absolute horizontal position, the linear probe stays at chance and the MLP reaches 0.40--0.49, with role accuracy 0.92--0.94. "
     r"CLIP-B/32's recovery therefore matches what absolute position alone gives a nonlinear reader, while DINOv2-L and SigLIP exceed it, and in every encoder the order lies beyond a linear probe, which our test requires. "
     r"At the final layer every encoder keeps less: fixed quadrants fall to 0.45--0.64 and moved ones to $-0.46$ to 0.16, so the selected middle layers give the upper end of Table~\ref{tab:places}." "\n\n")
s = s.replace(anchor, P + anchor)
open(B, "w", encoding="utf-8").write(s); print("ok")
