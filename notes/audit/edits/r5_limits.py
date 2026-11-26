D = "/coc/flash10/ksingara3/compositionality-theory/icml2027/parts/limits.tex"
t = open(D, encoding="utf-8").read()
R = [
 (r"\textbf{Probes and definitions.} The inclusion of behavior in representation is near-definitional for a probe that contains the system's procedure, so the evidence that the inclusion is strict comes from the systems we tested.",
  r"\textbf{Probes and definitions.} The inclusion of behavior in representation holds by construction for a probe that contains the system's procedure. Under our linear probe it becomes empirical, and its verdicts rest on two choices fixed in advance: a linear reader, and the state built before any question as the whole. An MLP probe recovers more from moved photographs (Appendix~\ref{app:images}) and from the question state of OLMo-2 7B (Appendix~\ref{app:behavior}), so an account that fixes a richer reader would place these systems higher and would credit the decomposition to the state where we credit it to a trained procedure."),
 (r"and our relative placement loosens that tie without removing it,",
  r"and our relative placement loosens that tie without removing it, because an MLP finds the figure from each photograph's absolute horizontal position alone (Appendix~\ref{app:images}),"),
]
for a, b in R:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open(D, "w", encoding="utf-8").write(t); print("ok")
