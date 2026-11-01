p = "main.tex"
s = open(p, encoding="utf-8").read()
i = s.index("In this section, we present the protocol every experiment shares")
j = s.index("%=====================================================================\n\\section{Results}")
old = s[i:j]
def grab(start, end_marker="\\end{figure}"):
    a = old.index(start); b = old.index(end_marker, a) + len(end_marker)
    return old[a:b]
fig_protocol = grab("\\begin{figure}[tbp]\n  \\centering\n  \\includegraphics[width=\\linewidth]{fig_protocol.pdf}")
fig_design = grab("\\begin{figure}[tbp]\n  \\centering\n  \\includegraphics[width=\\linewidth]{fig_design_real.pdf}")
glyph = old[old.index("\\paragraph{Glyph networks.}"):]
glyph = glyph[:glyph.index("\n\n")]

new = r"""In this section, we present the protocol that every experiment shares and two designs. Systems trained from scratch let a part be fresh to the system itself, so their own procedures can be scored on parts outside their training. Pretrained models saw their parts in pretraining, so a part is fresh only to the readout, and random weights and training checkpoints supply the controls.

""" + fig_protocol + r"""

\paragraph{Protocol.} We split the parts before fitting anything. Readouts and role maps fit composites of a probe pool, validation parts choose ridge penalties, layers and part codes, and fresh parts give every reported number. A ridge readout of the system's state returns each role's part code, and the identical pipeline, applied to the concatenation and to the sum of the same part codes, calibrates its error into $\kappa$ (Eq.~\ref{eq:kappa}). Role maps fitted on the pool give the residual $\delta$. The system's own procedure, whenever it has one, is scored on the same fresh parts. Each system also runs with random weights, which keep its architecture, tokenizer and input format and remove learning, so the gap between trained and random weights measures what training contributes. Every number is a mean over seeds, and Appendix~\ref{app:protocol} gives full details.

\subsection{Systems trained from scratch}
\label{sec:design_scratch}

\paragraph{Analytic operators} calibrate the measure. Parts are Gaussian vectors in $\R^{16}$, and fourteen operators combine them in pairs. They include role--filler binding (concatenation and a tensor product), fixed-width superposition (random projections into $0.5$ to $1.5$ times $kd = 32$ dimensions, HRR and MAP binding, and $x + 0.5y$), symmetric blends ($x + y$ and $x \odot y$), and a random $\tanh$ layer. We fit ridge, least-squares and MLP readouts of the role-1 part on composites of a vocabulary of $V$ parts, with 5 seeds for each $V$.

""" + glyph + r""" Behavior is the decoder's rendering of composites of fresh glyphs. We compare it with a \emph{population readout} fitted on the probe pool and with a \emph{vocabulary readout} fitted on the network's own training glyphs, which stands in for a typical probe.

\paragraph{Recursive composers.} Balanced binary trees of depth $n$ have Gaussian leaves in $\R^8$. A recursive auto-associative memory \citep{pollack1990} applies one learned composer at every node and one learned decomposer recursively, and we train both on depths 1 and 2. Behavior is the decomposer's recovery of fresh leaves, and a population readout maps the same root state to all $2^n$ leaves. Holographic reduced representations (HRR) of the same width $w$ use one pair of unitary role vectors per level, with 5 seeds for each $w$ from 16 to 256.

\subsection{Pretrained models}
\label{sec:design_pretrained}

""" + fig_design + r"""

\paragraph{Parts and states.} Sentences place two nouns in the roles of five asymmetric relations, such as ``a $x$ on a $y$'', and of the control ``a $x$ and a $y$''. We split 305 concrete nouns into 165 for fitting readouts, 40 for validation and 100 fresh to the readout. Photographs come from Tiny-ImageNet \citep{le2015tiny}, whose 200 classes we split into 120 for fitting, 30 for validation and 50 fresh, and two photographs share a $224 \times 224$ canvas \textsc{beside} or \textsc{above} each other, or \textsc{overlap}ping with the second covering a corner of the first. A part code is the model's state for one part alone (``a $x$'', ``there is a $x$ here'', or the photograph), reduced to 16 principal components and whitened, because a few directions dominate transformer states \citep{ethayarajh2019,timkey2021}. The composite's state averages over tokens or patches. Image states also concatenate the mean patch state of each quadrant, which keeps coarse position, and language-model states are also read at single tokens. Each split is redrawn for each of 3 seeds.

\paragraph{Models.} We test eleven text encoders, three from CLIP \citep{radford2021clip}, including those that condition Stable Diffusion 1.5 and 2 \citep{rombach2022}, three from T5 \citep{raffel2020t5}, two from BERT \citep{devlin2019bert} and three from GPT-2 \citep{radford2019gpt2}. The language models are Qwen3-8B \citep{yang2025qwen3}, OLMo-2 7B and 13B \citep{olmo2024olmo2}, phi-4 \citep{abdin2024phi4}, and seven Pythia models from 160 million to 12 billion parameters trained on the same data in the same order \citep{biderman2023pythia}. We follow ten checkpoints of Pythia-1.4b from initialization, and nine checkpoints each of Pythia-410m and of five PolyPythia runs that retrain it from other random seeds \citep{vanderwal2025polypythias}. The image encoders are CLIP ViT-B/32 and ViT-H/14 \citep{radford2021clip,cherti2023}, DINOv2 small, large and giant \citep{oquab2024dinov2}, and SigLIP so400m \citep{zhai2023siglip}. Every model also runs with random weights, and Pythia's step 0 serves as its own control. Causal models let the first noun's tokens see only earlier tokens, so we also report $\kappa$ for each role.

\paragraph{Roles by meaning.} Each relation has a converse paraphrase that reverses the order of mention and keeps the meaning, such as ``a tulip on a barrel'' and ``a barrel under a tulip'' (Figure~\ref{fig:design_real}c). The tulip is the \emph{figure} in both. A readout of the figure, fitted on canonical sentences and scored on converse ones, has a transfer of 1 if the state codes semantic roles and 0 if it codes the order of mention.

\paragraph{Behavior.} The language models answer questions about the same sentences that name neither noun, such as ``Which object is on top?'', after four demonstrations with fitting nouns. Half of the questions ask for the figure and half for the ground, in canonical and converse phrasings, so a model that answers by word order scores 0.5, and we score whether the model prefers the correct noun on the 100 fresh nouns. The image--text encoders match each composite of fresh photographs against a caption that names its objects in their places and against the caption with the places swapped. The text-only and image-only encoders produce no output of their own, so their tests measure only the representation.

"""
s = s[:i] + new + s[j:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
