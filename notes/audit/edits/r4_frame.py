M = "/coc/flash10/ksingara3/compositionality-theory/icml2027/main.tex"
B = "/coc/flash10/ksingara3/compositionality-theory/icml2027/parts/results.tex"
def apply(path, R):
    t = open(path, encoding="utf-8").read()
    for a, b in R:
        assert t.count(a) == 1, (path, a[:70])
        t = t.replace(a, b)
    open(path, "w", encoding="utf-8").write(t)
apply(M, [
 ("and language models that answer by meaning hold only the asked part by meaning, at the question, where one of them answers beyond every linear probe we fit.",
  "and language models that answer by meaning build the asked part by meaning only at the question, inside the procedure that answers, which is the weak form."),
 ("Language models that answer by meaning sit on or above the diagonal, so H1 fails for linear probes there.",
  "Language models that answer by meaning show the weak form instead: their sentence states bind roles by order of mention, and the question sets off a procedure that builds the asked role by meaning, at times beyond any linear probe of its state."),
 ("and that language models up to 14 billion parameters can answer beyond a linear probe.",
  "and that language models up to 14 billion parameters decompose by meaning only inside the procedure that answers."),
 ("and where behavior outruns a linear probe we report a limit of the probe.",
  "and where behavior outruns a fixed linear probe, the decomposition happens inside the system's own procedure, which is the weak form."),
 ("and every system would pass.",
  "and every system would pass. A linear reader also performs no decomposition of its own: it returns a part only where the state keeps that part along fixed directions for its role, which lets any later procedure reach the part, while a nonlinear reader that compares features to assign the roles is itself a trained decomposer."),
 ("so training puts position into the box features, and a nonlinear reader turns it into roles",
  "so training puts position into the box features, and only a trained nonlinear reader, a decomposer of the weak form, turns it into roles"),
 ("H1 therefore fails for linear probes in OLMo-2 7B, and in phi-4 and Qwen3-8B wherever per-prompt validation picks an early layer, which lowers $r$ to 0.77--0.83.",
  "H1 therefore fails for linear probes in OLMo-2 7B, and in phi-4 and Qwen3-8B wherever per-prompt validation picks an early layer ($r$ falls to 0.77--0.83), so the decomposition sits in the later layers that answer, the weak form."),
 ("models that answered by meaning held the asked part at the question, sometimes beyond a linear probe,",
  "models that answered by meaning built the asked part at the question, inside the procedure that answers,"),
])
apply(B, [
 ("H1: correct behavior on new parts implies a linearly recoverable state.",
  "H1: correct behavior on new parts implies a linearly recoverable state; a failure places the decomposition inside the system's own procedure, the weak form."),
 (r"\textbf{OLMo-2 7B marks the limit of a linear probe.}", r"\textbf{OLMo-2 7B decomposes inside its answering layers.}"),
 (r"and Observation~\ref{prop:subset} still holds for a probe that contains the model's own later layers, so recoverability depends on the kind of probe we fix.",
  r"and Observation~\ref{prop:subset} still holds for a probe that contains the model's own later layers, so the decomposition happens in the procedure that answers, the weak form of Definition~\ref{def:behcomp}."),
 ("and in every encoder the order lies beyond a linear probe, which our test requires.",
  "and in every encoder only a trained nonlinear reader finds the order, a decomposer of the weak form."),
])
print("ok")
