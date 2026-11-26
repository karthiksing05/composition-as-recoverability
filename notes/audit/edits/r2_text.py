import sys
B = "/coc/flash10/ksingara3/compositionality-theory/icml2027/"
def apply(fn, pairs):
    t = open(B + fn, encoding="utf-8").read()
    for a, b in pairs:
        n = t.count(a)
        assert n == 1, (fn, n, a[:80])
        t = t.replace(a, b)
    open(B + fn, "w", encoding="utf-8").write(t)
    print(fn, "ok", len(pairs))

apply("main.tex", [
# abstract
(r"score only whether a model acts correctly on new combinations of known parts, a test that a lookup table passes.",
 r"score only whether a model acts correctly on new combinations of known parts, and a lookup over the known parts takes apart every such whole."),
(r"Every pretrained state we tested binds roles to position, by order of mention in sentences and by absolute place in images, and a role by meaning appears only at the question, in language models that answer by meaning.",
 r"Every sentence and image state we tested binds roles to position, by order of mention and by absolute place, and language models that answer by meaning hold only the asked part by meaning, at the question, where one of them answers beyond every linear probe we fit."),
# intro
(r"chain facts that a language model knows one at a time \citep{press2023}. A lookup over a finite vocabulary passes these tests (Section~\ref{sec:definition}).",
 r"chain facts that a language model knows one at a time \citep{press2023}. Every whole in these tests is built from training parts, and a lookup over a finite vocabulary takes apart every such whole (Section~\ref{sec:definition})."),
(r"The evidence favors a strict inclusion (Figure~\ref{fig:ladder}). Networks trained on eight glyphs fail to draw new glyphs that a linear probe of their state recovers, and pretrained states bind roles to position, so they pass our test only where position and meaning coincide.",
 r"Toy and image systems favor a strict inclusion (Figure~\ref{fig:ladder}): networks trained on eight glyphs fail to draw new glyphs that a linear probe of their state recovers, and image systems ignore places that their states keep. Language models that answer by meaning sit on or above the diagonal, so H1 fails for linear probes there."),
(r"(iii) Evidence from toy networks to language models with 14 billion parameters that a reader fitted on few parts lags the state it reads, and that pretrained states keep fixed places their own judgments ignore.",
 r"(iii) Evidence from toy networks that a reader fitted on few parts lags the state it reads, from image encoders and diffusion models that their states keep fixed places their own judgments ignore, and from language models up to 14 billion parameters that behavior can outrun a linear probe."),
# 6.4 H4 for U-Nets
(r"Random weights keep 0.58--0.87 against their own part vectors, above the trained U-Nets, so the architecture lays out most of these places and H4 fails for the U-Nets (Appendix~\ref{app:diffusion}).",
 r"Random weights keep 0.58--0.87 against their own part vectors, above the trained U-Nets, and 0.09--0.12 against the trained ones, so the architecture lays out most of these places, training changes their code, and H4 fails for the U-Nets (Appendix~\ref{app:diffusion})."),
# 6.5
(r"transfer tracks accuracy at $r = 0.94$ (0.77--0.83 when each prompt selects its own layer), mostly through the gap between models that answer by meaning and models that answer by word order (Figure~\ref{fig:both}b).",
 r"transfer tracks accuracy at $r = 0.94$, and at 0.96 among the models that answer by meaning (Figure~\ref{fig:both}b). When each prompt selects its own layer, $r$ falls to 0.77--0.83, because validation then picks early layers for phi-4 on two prompts, where transfer falls to 0.08--0.12."),
(r"\textbf{The question builds the shared role code.} At the period",
 r"\textbf{The question state holds the asked part.} At the period"),
(r"and the unasked role at 0.44--0.61. The answer runs",
 r"the unasked role at 0.44--0.61, and $\kappa$ over both roles stays at or below 0.05, so the question state holds the answer by meaning and fails Definition~\ref{def:decomp}, which asks for every role. The answer runs"),
(r"In every language model, a figure probe of the token average transfers at 0.00--0.05. OLMo-2 7B marks the limit of a linear probe: it answers by meaning on 0.77 of pairs and patches carry its answer, yet its question state transfers at only 0.07--0.35 across selection rules, and nonlinear probes reach at most 0.51 (Appendix~\ref{app:behavior}). H1 fails for these probes, and Observation~\ref{prop:subset} holds only for a probe that contains the model's later layers.",
 r"In every language model, a figure probe of the token average transfers at 0.00--0.06. OLMo-2 7B answers by meaning on 0.77 of pairs and patches carry its answer, yet its question state transfers at only 0.07--0.35 across selection rules, and ridge, kernel and MLP probes at other layers reach at most 0.51 on validation nouns (Appendix~\ref{app:behavior}). H1 therefore fails for linear probes in OLMo-2 7B, and in phi-4 and Qwen3-8B wherever validation picks an early layer."),
# section 7
(r"an account whose decomposer sees only training parts is weak, because a lookup passes.",
 r"an account whose decomposer sees only training parts is weak, because a lookup takes apart every whole of training parts."),
(r"SCAN, COGS and VISOR combine training parts \citep{lake2018,kim2020cogs,gokhale2022visor}.",
 r"SCAN, COGS and VISOR combine training parts, including known parts in roles they never filled \citep{lake2018,kim2020cogs,gokhale2022visor}."),
(r"    Question states$^\ast$ & Strong & Strong, asked role \\",
 r"    Question states$^\ast$ & Low ($\kappa \le 0.05$) & Asked part only \\"),
(r"\textbf{Modern systems bind roles to position.} Every pretrained state we tested passes under positional roles and fails under converse or moved ones (Table~\ref{tab:tested}), as binding and ordering IDs would \citep{feng2024,dai2024binding,gurarieh2026mixing}.",
 r"\textbf{Modern systems bind roles to position.} Every sentence and image state we tested passes under positional roles and fails under converse or moved ones, and question states hold only the asked part (Table~\ref{tab:tested}), as binding and ordering IDs would \citep{feng2024,dai2024binding,gurarieh2026mixing}."),
# section 8
(r"because a lookup passes held-out combinations of training parts (Lemma~\ref{lem:lookup}).",
 r"because a lookup takes apart held-out combinations of training parts (Lemma~\ref{lem:lookup})."),
# conclusion
(r"models that answered by meaning built the asked role at the question, every pretrained state we tested bound roles to position,",
 r"models that answered by meaning held the asked part at the question, sometimes beyond a linear probe, every sentence and image state we tested bound roles to position,"),
])

apply("parts/results.tex", [
(r"Lumina's partial success comes from the joint attention of its denoiser, which can read the relation word between the two nouns.",
 r"Lumina's partial success may come from the joint attention of its denoiser, which can read the relation word between the two nouns; we leave the ablation untested."),
(r"and within a single model the correlation is 0.88. Re-selecting the layer for each prompt on validation nouns lowers $r$ to 0.77--0.83, and within the models that answer by word order transfer does not track accuracy ($r = 0.06$), so the correlation mostly separates the two groups.",
 r"and within the models that answer by meaning it is 0.96 (16 model--prompt pairs). Re-selecting the layer for each prompt on validation nouns lowers $r$ to 0.77--0.83, because validation then picks layers 9--10 for phi-4 on the reworded and zero-shot prompts and for Qwen3-8B zero-shot, where transfer is 0.08--0.12 although layer 30 transfers at 0.82--0.94; under that rule H1 fails for linear probes in those cells. Within the models that answer by word order transfer does not track accuracy ($r = 0.06$)."),
(r"while Qwen3-8B, OLMo-2 and phi-4 reach 0.40--0.49 (Figure~\ref{fig:origins}e). Pythia-1.4b peaks at 0.71 after 1{,}000 steps, 0.7\% of training,",
 r"while Qwen3-8B, OLMo-2 and phi-4 reach 0.38--0.47 on one seed (Figure~\ref{fig:origins}e). Pythia-1.4b peaks at 0.70 after 2{,}000 steps, 1.4\% of training,"),
(r"    SD 1.5 & 0.86--0.88 & 0.49--0.50 & 0.70 & 0.79 & 0.13 & $-0.06$ & 0.35 & -- \\",
 r"    SD 1.5 & 0.86--0.88 & 0.49--0.50 & 0.70 & 0.79 & 0.13 & $-0.06$ & 0.35 & $-0.36$ \\"),
])
