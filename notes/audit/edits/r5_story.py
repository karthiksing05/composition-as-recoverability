import re
D = "/coc/flash10/ksingara3/compositionality-theory/icml2027/"
def apply(path, R):
    t = open(D + path, encoding="utf-8").read()
    for a, b in R:
        assert t.count(a) == 1, (path, a[:80], t.count(a))
        t = t.replace(a, b)
    open(D + path, "w", encoding="utf-8").write(t)
    return t

main = open(D + "main.tex", encoding="utf-8").read()
# extract Def 5 box to move it
i = main.index(r"\begin{defbox}{Shared role codes}{factor}")
j = main.index(r"\end{defbox}", i) + len(r"\end{defbox}")
factor_box = main[i:j]
main = main[:i] + main[j:].lstrip("\n")
open(D + "main.tex", "w", encoding="utf-8").write(main)

apply("main.tex", [
 # abstract
 ("Image encoders and diffusion models keep fixed places, largely laid out by their architecture, that their own judgments ignore.",
  "Image encoders and diffusion models keep fixed places that their own judgments ignore, and in U-Net denoisers the architecture alone lays out most of these places."),
 ("Under a linear probe, every sentence and image state we tested binds roles to position, by order of mention and by absolute place, and language models",
  "Under a linear probe, every sentence state before the question binds roles to order of mention and every image state binds them to absolute place, and language models"),
 # intro after protected hypothesis
 ("A model whose own procedure handles new parts must have a state that holds them (Observation~\\ref{prop:subset}), while a state can hold parts that the model's outputs miss, and the strictness of the inclusion is the empirical hypothesis that we test.",
  "For a kind of probe that contains the system's own procedure, the containment holds by construction (Observation~\\ref{prop:subset}). Our probe is linear and fixed in advance, so we test both halves: whether states keep parts that the system's outputs miss, and whether behavior on new parts outruns the probe."),
 ("at times beyond any linear probe of its state.",
  "which a linear probe of the question state reads in phi-4 and Qwen3-8B and misses in OLMo-2 7B."),
 # contributions
 (re.search(r"\\textbf\{Contributions\.\}.*?for the strong form\.", main, re.S).group(0),
  r"\textbf{Contributions.} (i) Definitions of weak and strong compositionality by the capabilities each supports, and recoverability $\kappa$ as a test of decomposition, with short proofs that tests on training parts cannot detect decomposition and that rank, width and depth each cap it. (ii) Evidence that toy networks, image encoders and diffusion models keep parts in their states that their outputs miss, and that language models up to 14 billion parameters build roles by meaning only at the question. (iii) A diagnosis of positional binding, a spectrum of twenty-two accounts of compositionality and the components of a benchmark for the strong form."),
 # ladder caption title
 (r"\textbf{Behavior lags the state in toy and image systems and outruns the linear probe in language models.}",
  r"\textbf{Behavior lags the state in toy and image systems; in language models it tracks the question state and outruns it in OLMo-2 7B.}"),
 # Def 1
 ("give the correct output for new wholes built from training parts.",
  "give the correct output for new wholes, and any decomposition comes from a procedure trained for that purpose. On new combinations of training parts, a lookup passes this test (Lemma~\\ref{lem:lookup})."),
 ("and a lookup table then inverts it (Lemma~\\ref{lem:lookup}).",
  "and a lookup table then inverts it (Lemma~\\ref{lem:lookup}; Appendix~\\ref{app:calibration} checks fourteen toy operators)."),
 # Def 2
 ("The wholes of a relation $r$ \\emph{decompose} under a kind of probe $\\U$, fixed in advance, if for every role",
  "The wholes of a relation $r$ \\emph{decompose} under a kind of probe $\\U$, fixed in advance, and a role assignment, if for every role"),
 ("returns the part in role $i$ from wholes built from parts new to it. The relation's meaning fixes its roles; for sentences, role 1 is the figure in either phrasing. A system is \\emph{strongly compositional} with respect to $r$ when the states it builds for wholes of $r$ decompose.",
  "returns the part in role $i$ from wholes built from parts new to it. A system is \\emph{strongly compositional} with respect to $r$ when the states it builds for wholes of $r$, before any question about them, decompose under the roles that the relation's meaning fixes; for sentences, role 1 is the figure in either phrasing."),
 # roles paragraph
 ("reads a \\emph{positional binding}.",
  "reads a \\emph{positional binding}; for an image relation such as \\emph{left of}, the meaning is the relative place. We fix roles by meaning because only then does one reader serve every phrasing of the relation, and a state that binds by position needs a further procedure, the weak form, to find the figure."),
 # behavior and representation
 (re.search(r"\\textbf\{Behavior and representation\.\}.*?which is the weak form\.", main, re.S).group(0),
  r"\textbf{Behavior and representation.} A state that decomposes supports every weak capability, because a procedure that starts from the returned parts can act on any new combination, and the system exhibits such a capability once its own procedure reads those parts. A second inclusion concerns measurement: a procedure that returns the parts of new wholes is itself a probe, so behavior is included in representation for every kind of probe that contains the procedure (Observation~\ref{prop:subset}). A linear probe contains neither a nonlinear decoder nor the layers of a language model that answer, so under it the inclusion becomes an empirical question, and where behavior outruns the probe, the system builds the parts inside its own procedure, which is the weak form."),
 # probe choice
 ("Throughout the paper the probe is linear, with a bias term. We fix it in advance because, with unrestricted coordinates, every invertible relation becomes additive (Lemma~\\ref{lem:psi}), and every system would pass. A linear reader also performs no decomposition of its own: it returns a part only where the state keeps that part along fixed directions for its role, which lets any later procedure reach the part, while a nonlinear reader that compares features to assign the roles is itself a trained decomposer.",
  "Throughout the paper the probe is linear, with a bias term, because a linear reader performs no decomposition of its own: it returns a part only where the state keeps that part along fixed directions for its role, which lets any later procedure reach the part, while a nonlinear reader that compares features to assign the roles is itself a trained decomposer. We fix it in advance because, with unrestricted coordinates, every invertible relation becomes additive (Lemma~\\ref{lem:psi}), and every system would pass."),
 # Def 4 tolerance -> appendix
 (" The system is \\emph{representationally compositional} at tolerance $\\tau$ if $\\kappa \\ge 1 - \\tau$ for a part vector that carries information about the parts (Appendix~\\ref{app:defs}).",
  " Every score depends on the declared part vector (Appendix~\\ref{app:defs})."),
 ("Every score depends on the part vector that its protocol declares (Appendix~\\ref{app:defs}).\n", ""),
 # Def 5 inline
 ("The simplest route to recoverability gives each part one vector that enters every whole and changes only with its role.",
  "The simplest route to recoverability gives each part one vector that enters every whole and changes only with its role: a relation has \\emph{shared role codes} with residual $\\delta$ when its state equals $c + \\sum_i B_i\\, e'(x_i)$ up to a residual of relative size $\\delta$, with one injective matrix $B_i$ per role shared by all parts (Definition~\\ref{def:factor}, Appendix~\\ref{app:defs})."),
 # H4 out of body
 ("We test four hypotheses on parts new to every fitted procedure, and Appendix Table~\\ref{tab:verdicts} gives one verdict per hypothesis and system.",
  "We test three hypotheses on parts new to every fitted procedure."),
 (" \\textbf{H4}: training builds recoverability beyond the input and the random-weight architecture. A system whose own procedure handles new parts that the fixed probe cannot recover refutes H1 for that probe class, and a random-weight model that matches the trained one refutes H4.",
  " A system whose own procedure handles new parts that the fixed probe cannot recover refutes H1 for that probe and places the decomposition in its procedure. Appendix Table~\\ref{tab:verdicts} gives one verdict per hypothesis and system, with a fourth, H4, on whether training builds recoverability beyond the input and the random-weight architecture."),
 ("so the architecture lays out most of these places, training changes their code, and H4 fails for the U-Nets (Appendix",
  "so the architecture lays out most of these places and training changes their code (Appendix"),
 # Lumina speculation
 (", although its Gemma caption state binds by order, as do CLIP-L and T5, so its joint attention over caption and image tokens may account for the gain.", "."),
 # question state sentence
 ("so the question state holds the answer by meaning and fails Definition~\\ref{def:decomp}, which asks for every role.",
  "so the question state carries the answer of the procedure and no decomposition of the whole."),
 # spectrum
 ("Five conditions place any account, and Appendix Table~\\ref{tab:spectrum} places twenty-two.",
  "Appendix Table~\\ref{tab:spectrum} places twenty-two accounts."),
 ("Every sentence and image state we tested passes under positional roles and fails under converse or moved ones for a linear probe,",
  "Every sentence and image state we tested keeps roles better by position and fails under converse or moved ones for a linear probe,"),
 # conclusion
 (re.search(r"Readers fitted on few parts lost parts.*?assigned the roles\. A strongly compositional network therefore", main, re.S).group(0),
  "Toy and image systems kept parts in their states that their outputs missed, language models built roles by meaning only inside the procedure that answers, and harnesses repaired behavior as far as the component that assigned the roles. A strongly compositional network therefore"),
])
# harness para fold
t = open(D + "main.tex", encoding="utf-8").read()
h = re.search(r"\n+\\textbf\{A harness moves the decomposer and leaves our test in force\.\}.*?code included\.", t, re.S).group(0)
t = t.replace(h, " Our test applies to whichever component assigns the roles, a lexicon included (Lemma~\\ref{lem:lookup}; Appendix~\\ref{app:harness}).")
# drop calibration paragraph
c = re.search(r"\\textbf\{Tests on training parts cannot see decomposition\.\}.*?\(Appendix~\\ref\{app:calibration\}\)\.\n", t, re.S).group(0)
t = t.replace(c, "")
open(D + "main.tex", "w", encoding="utf-8").write(t)

# appendix: Def 5 box + tolerance
apply("parts/appendix_orig_defs.tex", [
 ("\\begin{defbox}{Informative part vectors and productivity}{informative}",
  factor_box + "\n\nA system is \\emph{representationally compositional} at tolerance $\\tau$ if $\\kappa \\ge 1 - \\tau$ (Definition~\\ref{def:repcomp}) for an informative part vector.\n\n\\begin{defbox}{Informative part vectors and productivity}{informative}"),
 ("A relation is \\emph{productive to depth $n$} if Definition~\\ref{def:repcomp} holds when",
  "A relation is \\emph{productive to depth $n$} if $\\kappa \\ge 1 - \\tau$ holds when"),
])
apply("parts/results.tex", [
 ("Holds for harnesses whose code assigns the roles", "Untested; the harnesses read the text and leave the state unread"),
 ("Holds: parse harness 0.51 to 0.98", "Untested; a parse harness lifts answers from 0.51 to 0.98 from the text"),
])
print("ok")
