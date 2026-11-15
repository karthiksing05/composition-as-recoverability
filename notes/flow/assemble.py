import re
ROOT="/coc/flash10/ksingara3/compositionality-theory/"
old=open(ROOT+"notes/main_pre_flow.tex",encoding="utf-8").read()
A=open("body_a.tex",encoding="utf-8").read(); B=open("body_b.tex",encoding="utf-8").read(); P=open("protocol.tex",encoding="utf-8").read()
def env(name,key):
    m=re.search(r"\\begin\{%s\}(?:(?!\\end\{%s\}).)*?\\label\{%s\}.*?\\end\{%s\}"%(name,name,re.escape(key),name),old,re.S)
    # last begin before the label
    i=old.index("\\label{%s}"%key); s=old.rfind("\\begin{%s}"%name,0,i); e=old.index("\\end{%s}"%name,i)+len("\\end{%s}"%name)
    return old[s:e]
def defbox(key):
    m=re.search(r"\\begin\{defbox\}\{[^}]*\}\{%s\}.*?\\end\{defbox\}"%key,old,re.S); return m.group(0)
CAP={
"fig:protocol":r"""\textbf{One protocol reads the same held-out parts through three procedures.} \textbf{a.} We split the parts before fitting anything (1). The system encodes composites of two parts (2, 3), and affine readouts fitted on the fitting pool return each role's part from the state, with error $\varepsilon$ on held-out parts (4). The same steps, run on the concatenation and on the sum of the same part codes, calibrate $\varepsilon$ into the score $\kappa$ (5). \textbf{b.} A population readout measures the representation, a vocabulary readout stands in for a typical probe, and the system's own procedure measures behavior. \textbf{c.} A converse paraphrase keeps the figure and reverses the order of mention, so a readout of the figure fitted on one phrasing returns the tulip on the other if roles follow meaning and the barrel if they follow word order.""",
"fig:beh_toy":r"""\textbf{Procedures that pass on training parts fail beyond them.} \textbf{a.} A lookup over composites of 32 training parts (open circles) identifies the role-1 part of every operator except the symmetric blend, while an affine readout separates the operators on held-out parts (filled circles; blue marks role--filler operators, grey superposition and orange a symmetric blend). \textbf{b.} Glyph networks reproduce new pairs of training glyphs (dashed) and fail on held-out glyphs (solid) when trained on few glyphs ($m = 64$). \textbf{c.} Fine-tuning on a new glyph in role~1 lowers old pairs and the new glyph in role~2 ($V = 64$). \textbf{d.} A recursive composer's own decomposer fails one level beyond its training depths (shaded). Means over 5--9 seeds, with standard errors.""",
"fig:beh_real":r"""\textbf{Pretrained models lose the roles of parts they tell apart.} \textbf{a.} Accuracy of each language model's answers to questions about roles in sentences of held-out nouns (dots), and the range over three other prompts (bars). Answering by word order scores 0.5 (dotted). \textbf{b.} Image--text encoders match a composite of held-out photographs to a caption over one with a wrong object (grey) and over one with the two places swapped (orange). \textbf{c.} Text-only encoders match a sentence to its paraphrase over one with a wrong noun (grey) and over the same paraphrase with the nouns exchanged (orange), where matching word order scores 0 and matching meaning scores 1. Means over 3 seeds.""",
"fig:rep_toy":r"""\textbf{Recoverable states keep the parts that the system's own procedures lose.} \textbf{a.} Trained on 8 glyphs, an MLP renders composites of held-out glyphs poorly (black frames), while an affine readout of its state recovers them (blue frames). Trained on 64 glyphs, it succeeds at both. Numbers give $R^2$ to the target. \textbf{b.} Affine recovery of held-out parts meets the ceiling of Proposition~\ref{prop:ceiling} across analytic operators. \textbf{c.} A population readout of the state (blue) stays high where the network's own decoder (black) fails, and a new decoder trained on 64 other glyphs reads the frozen states of networks trained on 8 (square). \textbf{d.} A population readout of a recursive composer's state follows the bound of Proposition~\ref{prop:depth} (dotted) beyond the training depths (shaded), where the decomposer fails. Means over 5--9 seeds, with standard errors.""",
"fig:rep_codes":r"""\textbf{Decoders use the role codes, and writing a code adds a part without damage.} Role maps $h \approx c + B_1 e(x) + B_2 e(y)$, fitted on a pool of glyphs, predict and edit states that each network's own decoder renders (\textsc{over}, $V = 64$). \textbf{a.} An edit replaces the role-1 code with a donor's code, and the decoder renders the donor in role 1 and keeps the role-2 glyph. The natural direction $h(a', b') - h(a, b)$, scaled to the same norm, lands far from the target. Numbers give $R^2$ to the target. \textbf{b.} The decoder renders states predicted by the role maps as well as the network's own states. \textbf{c.} An edit changes the untouched half of the output less than other changes of the state do. \textbf{d.} Writing a new glyph's code through $B_2$ matches replay for old pairs and for the new glyph in role~2, with no gradient step and no stored data. Means over 5 seeds, four architectures and two widths, with standard errors.""",
"fig:rep_real":r"""\textbf{Pretrained states reach partial recovery, set by the state we read.} The score $\kappa$ is 1 for the concatenation of the part codes (dotted) and 0 for their sum. \textbf{a.} Text encoders with trained and random weights, on held-out nouns. \textbf{b.} Language models against the number of parameters, for the Pythia suite (blue) and four other models (orange), over both nouns (filled) and for the second-mentioned noun alone (open). \textbf{c.} Training of Pythia-1.4b (blue), where step 0 is the initialization, and of Pythia-410m from six random seeds (green line and range). \textbf{d.} Image encoders on held-out photographs, for the token-averaged state (bars) and the quadrant state (diamonds). Means over 3 seeds, with standard errors.""",
"fig:rep_roles":r"""\textbf{Roles live in the state at the question, and that state tracks the answers.} \textbf{a.} A readout of the figure fitted on canonical sentences and scored on converse ones, in phi-4, Qwen3-8B and OLMo-2 13B (left to right in each cluster). A readout of meaning scores 1, a readout of word order 0, and one without role information 0.5 (dotted). \textbf{b.} The same transfer from the question state against answer accuracy, for seven language models (blue: Qwen3-8B, phi-4, OLMo-2; orange: GPT-2, Pythia) on the main prompt (large dots) and three others (small dots). \textbf{c.} With no fitted readout, the share of pairs of held-out nouns whose converse paraphrase lies nearer than both role reversals, where chance is $1/3$ (dotted). The token-averaged state never succeeds. Means over 3 seeds.""",
}
FILE={"fig:protocol":"fig_protocol","fig:beh_toy":"fig_beh_toy","fig:beh_real":"fig_beh_real","fig:rep_toy":"fig_rep_toy","fig:rep_codes":"fig_rep_codes","fig:rep_real":"fig_rep_real","fig:rep_roles":"fig_rep_roles"}
def fig(key):
    if key=="fig:ops": return env("figure",key)
    return "\\begin{figure}[tbp]\n  \\centering\n  \\includegraphics[width=\\linewidth]{%s.pdf}\n  \\caption{%s}\n  \\label{%s}\n\\end{figure}"%(FILE[key],CAP[key],key)
body=A+B
body=re.sub(r"@@FIG:([a-z_:]+)@@",lambda m:fig(m.group(1)),body)
body=re.sub(r"@@TAB:([a-z_:]+)@@",lambda m:env("table",m.group(1)),body)
body=re.sub(r"@@DEF:([a-z]+)@@",lambda m:defbox(m.group(1)),body)
body=body.replace("Remedy & Failure it targets","Repair & Failure it targets")
# front matter
pre=old[:old.index("\\begin{abstract}")]
pre=pre.replace("\\usepackage{xcolor}  % or \\usepackage{color}\n\n\\newcommand{\\note}[1]{\\textcolor{red}{#1}}\n","")
# proofs, without the cut propositions
s=old.index("\\section{Formal results}"); e=old.index("\\section{Modes of combination}")
proofs=old[s:e]
for lab in ("prop:blend","prop:separable","prop:context"):
    i=proofs.index("\\label{%s}"%lab); a=proofs.rfind("\\begin{proposition}",0,i)
    z=proofs.index("\\end{proof}",i)+len("\\end{proof}")
    proofs=proofs[:a]+proofs[z:]
proofs=re.sub(r"\n\\noindent\nWhen the relation is only approximately separable.*?\n(?=\n|%)","\n",proofs,flags=re.S)
proofs=proofs.replace("; Figure~\\ref{fig:calibration}a confirms it for every injective operator we tried"," (Figure~\\ref{fig:beh_toy}a)")
tail="\\FloatBarrier\n\\bibliographystyle{apalike}\n\\bibliography{references}\n\n%=====================================================================\n\\clearpage\n\\appendix\n"
mid=old[old.index("\\FloatBarrier\n\\bibliographystyle"):old.index("\\section{Formal results}")]
out=pre+body+mid+proofs.rstrip("=% \n")+"\n\n%=====================================================================\n"+P
# vocabulary
rep=[("held-out pairs of training","new pairs of training")]
def sub(pat,r):
    global out
    out=re.sub(pat,r,out)
sub(r"\bprobe pool\b","fitting pool")
sub(r"\bfresh to the (readout|reading procedure)","held out from the \\1")
sub(r"\bfresh to those procedures","held out from those procedures")
sub(r"\bfresh to a procedure","held out from a procedure")
sub(r"\bfresh both to a readout and to","held out both from a readout and from")
sub(r"\bfresh only to the readout","held out only from the readout")
sub(r"\bfresh to the system itself","held out from the system itself")
sub(r"\ba fresh\b","a held-out"); sub(r"\bA fresh\b","A held-out")
sub(r"\bfresh\b","held-out"); sub(r"\bFresh\b","Held-out")
sub(r"\bfamiliar parts\b","training parts"); sub(r"\bfamiliar\b","training")
sub(r"\bremedy conditions\b","repair conditions")
open(ROOT+"recoverability-theory/main.tex","w",encoding="utf-8").write(out)
print(len(out.split()))
