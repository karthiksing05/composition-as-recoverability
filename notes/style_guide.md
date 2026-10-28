# Style guide: arXiv 2610.20922 as the model for the recoverability paper

**Reference paper.** Elmoznino, Kasetty, Agüera y Arcas, Bengio, Lajoie, *Defining and Generating Compositional Structure*, arXiv:2610.20922v1 [cs.AI; cs.IT, cs.LG], 14 Oct 2026. This is `\citet{elmoznino2026}` in our bib.

**Sources used.** The PDF is at `notes/style_ref.pdf`. I extracted its text with pypdf to `notes/style_ref.txt`, along with per-page color operators, line widths and font names. I also looked at the arXiv abs page and the HTML page (`notes/ref_html/index.html`).

**What is measured and what is inferred.** These come straight from the PDF:
- page count
- fonts
- hex colors
- line widths
- word counts

These are inferred and marked "(inferred)" where they appear:
- the template's exact geometry
- whether caption labels are bold

The PDF could not be rendered to images, because poppler is missing.

---

## 1. Venue template, length, bibliography

| Item | Reference |
|---|---|
| Template | **Not a conference template.** It is the generic arXiv preprint style (kourgeorge `arxiv.sty`, inferred), with these features: the title is set in caps/small caps (`DEFINING AND GENERATING COMPOSITIONAL STRUCTURE`); every page has the running head "Defining and Generating Compositional Structure" with **PREPRINT** at the right, above a thin rule; the abstract heading reads `ABSTRACT`; a `Keywords` line follows the abstract; and a date line ("October 16, 2026") sits under the affiliations. |
| Page / text | US letter (612x792 pt), single column. Lines run about 120 characters of 10 pt Times, so the text block is roughly 6.5 in wide (inferred). That is noticeably wider than our 5.5 in NeurIPS-like block. |
| Fonts | Body is Times (NimbusRomNo9L Regular/Medium/Italic) with Computer Modern math. Typewriter text uses SFTT. |
| Authors | Names carry superscript affiliation numbers, with one affiliation line below. Page 1 has a footnote with "∗Correspondence to: …" and "Code is provided at <github url>". |
| Length | **34 pages**, broken down below. |
| Bibliography | natbib author-year in round parentheses, e.g. "(Leibniz, 1898)". The entry format is `Surname, I. (Year). Title. *Venue*, vol(issue):pages.`, which is the **apalike** look; the year sits in parentheses after the authors and the venue is italic. Entries are sorted alphabetically. Citations are hyperlinked in dark navy **#001473**, as text color with no boxes, and the same color is used for internal references. |

Page breakdown:

| Part | Pages |
|---|---|
| Main text | 1 to mid-15 |
| References | 15-16 (~1.3 pages, ~45 entries) |
| Appendices | 17-34 (~18 pages) |

## 2. Title and abstract

- **Title.** "Defining and Generating Compositional Structure": five words, two parallel gerunds plus a noun phrase. It has no colon, no subtitle, no method name and no claim.
- **Abstract.** One paragraph of **248 words** (ours is 182). It moves through seven steps:
  1. **Hook from nature, with a concrete chain**: "atoms combine into molecules, molecules into cells, and cells into organisms".
  2. **Broad definition in an em-dash aside**: "compositionality — the reuse and recombination of parts into novel structures —".
  3. **Gap**: "Yet despite its importance, … mathematical accounts … remain limited."
  4. **Pivot**: "**Here, we fill this gap by formalizing a simple idea:** compositional structure emerges when parts, considered together, reveal shared structure that is not captured when those parts are modeled separately."
  5. **Tool**: algorithmic information theory, glossed in another em-dash aside.
  6. **What the definition gives**, then **inversion/experiments**, then **what the results suggest** ("suggesting that the definition captures compositional structure rather than merely complexity, regularity, or shared information").
  7. **"Finally, we discuss implications for machine learning, …"**
- The abstract has no numbers, no list and no citations.
- **Keywords line**: "compositionality, complexity, structure, compression, machine learning".

## 3. Section list and page budget (main text ≈ 14.3 pp)

| # | Section | Pages | Share |
|---|---|---|---|
| — | Title, abstract, keywords | 0.6 | |
| 1 | Introduction (opens with a Leibniz **epigraph**) | ~1.5 (p1–2) | 10% |
| 2 | Intuition for compositional structure (running example, **no math until the end**) | ~1.2 (p2–3) | 8% |
| 3 | Defining compositional structure | ~5.8 (p3–9) | 40% |
| 3.1 | Preliminaries: algorithmic information theory | 1.1 | |
| 3.2 | Formal definition (Def 1, Def 2, Fig 1, conjecture) | 2.0 | |
| 3.3 | Approximating compositional structure | 0.5 | |
| 3.4 | Relationship to other measures of compositionality (**this is the related work**) | 2.2 | |
| 4 | Generating compositional structure (experiments) | ~4.8 (p9–14) | 34% |
| 4.1 | Compositional strings (+Fig 2) | 2.3 | |
| 4.2 | Compositional images (+Fig 3) | 2.2 | |
| 5 | Discussion (summary + implications + "Future work and limitations") | ~1.2 (p14–15) | 8% |
| | References | ~1.3 | |
| App. A | Practical approximation | 1.3 | |
| App. B | The maximal decomposition (argument for the conjecture) | 0.8 | |
| App. C | Proofs (C.1–C.4; Theorems 1–4 with corollaries) | ~3.5 | |
| App. D | Experiment details (D.1 strings, D.2 images; Algorithms 1–5; Tables D.1–D.2; Figs D.1–D.3) | ~11 | |

The paper has **no** separate Background section, **no** Related Work section and **no** Conclusion section; related work lives in Section 3.4 and in the second paragraph of the Introduction. Headings are numbered sections and subsections, plus many **bold run-in paragraph headings ending with a period** ("Kolmogorov complexity.", "Results.", "The success of deep learning."). There are no subsubsections.

## 4. How contributions are stated

**The paper has no "Contributions" list and no bullets in the introduction.** The introduction is five prose paragraphs:

1. A **motivating paragraph** of examples across scales and domains. It cites Arthur and Agüera y Arcas and ends with "arguably one of the principal sources of interesting complexity in the universe".
2. **What existing definitions miss.** This paragraph is the related work in miniature and ends with the open question in italics-free prose: "for an arbitrary object, what intrinsic compositional structure emerges across its parts, how much of it is present, and how is it organized across scales?"
3. **"Our starting point is that …"** states the core idea in one sentence, then gives a cell→tissue example and closes with "Section 2 develops this idea and tests it against several intuitive cases."
4. **The formal tool.** It closes with "Section 3 formalizes the definition, …".
5. **The inverse problem and experiments** ("We demonstrate this in Section 4 …"), followed by **implications** ("We close by developing these connections … in Section 5.").

So each contribution appears as a paragraph that ends by pointing to the section that delivers it. The key term is introduced in italics at first use ("This is what we call *novel compositional structure*.").

## 5. How theory and experiments interleave

The order is **Intuition → Preliminaries → Definition → Illustration → Approximation → Relation to others → Experiments**. Theory comes entirely before the experiments; the experiments are framed as "a concrete test of the definition".

- **Section 2 (Intuition)** builds one running example (programs and their optimal *library*) in plain language and walks through **three contrasting cases**: shared function (compositional), more uses of the same library (reuse without novelty), and unrelated programs. Notation for parts appears only once the example needs it (`x_a, x_b, m_{x_a}`). It ends with the generalisation ("For an arbitrary object, a *model* plays the role of the library") and a forward pointer: "The next section makes 'best model' and 'cannot be recovered' precise".
- **Definitions are few and boxed.** The main text has only **two** numbered definitions, each in a **titled, tinted box** (light teal frame ~**#b2d9d9**, near-white fill ~**#f7fbfb**). The box title reads `Definition 1. Novel compositional structure`, with the name bold and no parentheses. The box contains the setup, the assumptions and one displayed, numbered equation.
- **Each box is followed by "This quantity measures …"** and then **"Intuitively, … is large when …"**, an edge case (≈0 when …), and a scope note ("Note that Definition 1 measures only *novel* …").
- **"Note" boxes for technical caveats.** These use the same box style, e.g. `Note 1. Suppression of lower-order terms` (O(1)/O(log n) terms are suppressed throughout) and `Note 2. Technical status of the mutual information result` (in the appendix). They keep the caveats out of the main argument without hiding them.
- **No theorems or propositions in the main text.** One relation is shown inline as a displayed implication, with "(proof in Appendix C.1)" and an "Intuitively, two objects can share AMI for one of two reasons: (i) … or (ii) …" paragraph. A **conjecture** is stated in prose ("we conjecture that an object's intrinsic decomposition is the one that maximizes …") and defended in Appendix B.
- **Appendix theorems** use amsthm plain style: `Theorem 1 (Compositionality implies algorithmic mutual information).` with an italic statement. Each is followed by an **"Intuitively, the theorem states …" paragraph before the proof**, and corollaries are numbered 2.1–2.4.
- **Related work as theory.** Section 3.4 contrasts the definition with six neighbours (AMI, PID, emergence/structure function, formal grammars, assembly theory, symbiogenesis). Each gets a **bold run-in heading**, often a one-line displayed equation of the competing measure, and a closing sentence on how it relates ("The two approaches are thus complementary: …"; "Thus, symbiogenesis offers a plausible mechanism for …").

## 6. How each experiment is introduced

**There are no boxed findings, no bolded takeaways and no "Q:" or "Finding:" callouts.** The template is fixed and repeated across both experiments:

1. **Section opener** (Section 4 intro). It restates the inverse question in plain words with italic emphasis ("given a pre-specified compositional structure profile, can we *generate* objects that satisfy it?"). It defines the reward from the profile (two numbered equations) and gives a **3-item bullet list**: "In sum, each of our below experiments is fully characterized by:" (domain; decomposition tree + coding schemes + functional; optimizer).
2. **Subsection opener** (2–3 sentences): it states the domain concretely with exact sizes ("the domain is X={0,…,7}^128; strings of length 128 with a vocabulary of 8 symbols"), then gives the signpost "Methods are briefly described below and illustrated in Figure 2a; Appendix D.1 contains further details."
3. **Identical run-in paragraphs in both experiments**: **MDL algorithm. / Conditional model compression. / Reward. / Optimization. / Results.** Each methods paragraph says what was done and *why*, e.g. "To fit a short grammar without introducing unjustified structure, we use …"
4. The **Results** paragraph follows a fixed sequence:
   - what was run ("We ran the evolutionary optimization for 2,000 generations across 10 seeds")
   - the baseline and **why it is the right control** ("This comparison serves as an empirical test of our claim that …")
   - sanity checks ("Figure 2b shows … indicating they were successfully optimized")
   - one paragraph per remaining panel, written "Figure 2c … We find that …"
   - a closing interpretive sentence tying back to the definition ("This illustrates the core property of our compositionality definition."; "These differences are precisely what our definition of compositional structure is intended to capture.")
5. **Honest limits inline.** Examples: "only 3 seeds, given the higher computational cost"; "although it is difficult to visually establish the presence of feature hierarchies in an objective way"; "rather than claiming to cover every possible form of image structure".

## 7. Figure style

- **Count and placement.** The main text has **three figures**, all **full text width**. **There is no teaser figure on page 1**; Figure 1 appears on page 6 inside Section 3.2. Figures sit at the top of the page.
- **Fig 1** is a **conceptual schematic**. It shows four small "profile" diagrams in a 2x2 arrangement (Uncompositional / Flat / Hierarchical / Open-ended), each subtitled "e.g., …", plus an inset that defines the increment K(m_x | m_xa, m_xb) on a merge tree. It is single-hue (muted blue **#5975a4** with greys **#bfbfbf/#ebebeb**) and uses thick strokes (**2 pt** lines, 4–6 pt accents).
- **Figs 2 and 3** are **four-panel composites with lettered panels**, using the same layout in both experiments:
  - **a**: method schematic with a pipeline diagram and small example objects
  - **b**: "Reward over time" line plots
  - **c**: a grid of small-multiple statistic line plots over time (grammar or CPPN statistics)
  - **d**: qualitative examples (dot plots or images)

  Panel labels are **bold lowercase letters with a period inside the figure, followed by a panel title** ("b. Reward over time", "c. Generated string grammar statistics").
- **Plots.** Lines show the mean over seeds, with a **shaded SEM band** across seeds. The y-axis is in arbitrary units (a.u.), normalised to each reward's top score, with no legends crowded into the panels; color is explained in the caption and in panel a. Plot line widths are ~0.75–1.2 pt and the axes are thin (0.4 pt).
- **Palette** (seaborn "deep"):

  | Use | Color |
  |---|---|
  | Proposed method (compositionality) | blue **#4c72b0** |
  | Baseline (structure) | orange **#dd8452** |
  | Lighter tints | #5471ab, #cc8963 / #d1885c |
  | Neutrals | #d5d5d5, #929292, #5e5e5e |
  | Accent, Fig 3 network schematic only | purple **#662e7d / #ceb3d3** |

  The same blue and orange code each condition consistently across every figure, including the appendix figures (D.1 borders are blue for compositionality and orange for structure; D.2 and D.3 are blue- and orange-framed galleries).
- **Fonts in figures.** Labels are in **Helvetica Neue** (regular, bold and italic; the composites were probably assembled in a vector editor), and math is **STIX** (matplotlib `mathtext.fontset='stix'`). Math labels in figures match the notation in the text exactly (`C(x_a,x_b→x)`, `K(m_x)`).
- **Captions** sit below the figure in body-size type.
  - **Form.** `Figure 2:` is followed by a **bold title phrase** ("Generating compositional strings."), then **bold panel letters** "**a.** … **b.** …" with 1–3 sentences each. The panel sentences mix description *and result* ("Grammars fit to strings generated by the compositionality reward are significantly deeper, …"). The caption ends with a **bold run-in "Seeds."** sentence on statistics ("Shaded regions in panels (b, c) reflect the standard error of the mean across 10 seeds. Panel (d) shows results for one representative seed only; dot plots for all seeds are provided in Appendix D.1.").
  - **Length.** Figs 2 and 3 have ~175–250 words, and the conceptual Fig 1 has ~95 words. Color references appear inline: "(blue; Equation (14))", "(orange; K(m_x))".
- **Appendix figures** are numbered by appendix (Figure D.1, D.2, D.3) and have one-line captions.

## 8. Table style

- **The main text has no tables.** There are two appendix tables (Table D.1 NEAT hyperparameters, Table D.2 CPPN statistics). Both are **two-column text tables** (Setting | Value; Metric | Description) with booktabs-style rules (0.8 pt top/bottom, 0.5 pt mid) and no vertical rules.
- **Captions go above the table**: "Table D.1: …" followed by one or two sentences. Numbering follows the appendix letter.
- **Algorithms** (Algorithm 1–5, algorithm2e style, with `Input:` and `return`) appear only in Appendix D.

## 9. Writing voice

- **First person plural** throughout ("We now make this idea precise", "we conjecture", "We leave this as an open problem"). Present tense describes the framework and past tense describes runs ("We ran …").
- **Sentences are long and balanced**, about 25–35 words, built on colons, semicolons and spaced em dashes (" — "). Parallel lists of three are common ("reuse, hierarchy, and part–whole organization").
- **Characteristic discourse markers** open paragraphs: "Intuitively,", "Importantly,", "Concretely,", "Formally,", "More broadly,", "In sum,", "Thus,", "Conversely,".
- **Hedging is calibrated, not heavy.** Typical hedges are "suggesting that", "we conjecture", "seem to differ", "a plausible mechanism", "arguably". Limitations are named plainly in the main text.
- **Key terms** are set in italics at first definition (*novel compositional structure*, *library*, *model*, *intrinsic*), and italics are also used for contrastive emphasis (*novel*, *different*, *generate*, *kind*).
- **The paper restates its definitions in prose**, both after each formal statement and in the Discussion ("Formally, it is the information in the best model of a whole that cannot be recovered from the best models of its parts.").
- **Spelling is mixed**: British in places ("favour", "colour", "normalise" occasionally) and American elsewhere.
- **The Discussion has no bullets.** Open questions are posed as direct questions inside prose ("However, is there any sense in which an object admits an 'intrinsic' decomposition, and if so how can it be defined?").

## 10. Appendix organisation

The appendices run in the same order as the main text needs them. Their headings take the form "Appendix A Title" (with the word "Appendix" in the heading).

- **A.** Practical approximation (expands Section 3.3; numbered list of "three distinct difficulties")
- **B.** The maximal decomposition (prose argument for the conjecture: "First … Second … Third … Fourth …", each pointing to a theorem in C)
- **C.** Proofs (C.1–C.4): a theorem, its intuition paragraph, an optional Note box, then the proof
- **D.** Experiment details: D.1 and D.2 mirror 4.1 and 4.2, with bold run-in headings (e.g., "Grammar representation and codelength.", "MDL grammar induction.", "Additional results."), Algorithms, Tables D.x and extra-figure galleries

Every main-text claim that skips detail points to a specific appendix subsection.

## 11. Verbatim examples (characteristic sentences, captions and boxes)

1. **Starting-point sentence (introduction):** "Our starting point is that compositional structure emerges when parts, considered together, reveal shared structure that is not captured when those parts are modeled separately."
2. **Clarifying "Importantly" move:** "Importantly, the claim is not that the raw whole contains information absent from its parts: by construction, it is simply their concatenation. Rather, combining the parts can change the best account of their structure."
3. **Boxed note (the closest thing to a takeaway box):** "**Note 1. Suppression of lower-order terms** — Kolmogorov-complexity relationships generally hold only up to small additive terms. … Throughout the paper, we suppress these terms and write the corresponding equalities and inequalities directly."
4. **Caption opening plus the Seeds sentence:** "**Figure 2: Generating compositional strings. a.** We evolve a population of strings using either (i) a reward that maximizes multi-scale compositional structure using our definition (blue; Equation (14)) or (ii) an alternative reward that just maximizes structure without any pressure for compositionality (orange; K(m_x)). … **Seeds.** Shaded regions in panels (b, c) reflect the standard error of the mean across 10 seeds."
5. **Results framing and closing tie-back:** "This comparison serves as an empirical test of our claim that compositionality captures a specific *kind* of structure based on the interactions between parts at multiple scales, rather than simply the total amount of structure." … "These differences are precisely what our definition of compositional structure is intended to capture."

---

## 12. Comparison with our draft (`recoverability-theory/main.tex`, `positionpaper.sty`)

### 12.1 Where the draft already matches

- It uses "we" throughout, author-year natbib with round parentheses, and definitions followed by "Three features of the definition deserve comment".
- Proofs are in the appendix, there is a running two-images contrast (avocado on / avocado chair), and the conceptual taxonomy figure is analogous to their Fig 1.
- The draft cites `elmoznino2026` already. Note that the reference *is* that paper, so our Section 3.x relationship paragraph should describe it accurately: it measures *novel structure in the model of the whole given models of the parts*, not "the information a whole adds beyond its parts" in a raw sense.

### 12.2 Concrete changes, in order of impact

**A. Template and front matter**

1. Replace `positionpaper.sty` with an arXiv preprint style (kourgeorge `arxiv.sty` or an equivalent). The style should provide:
   - a caps/small-caps title
   - a running head "<Short title> — PREPRINT" with a rule
   - an `ABSTRACT` heading
   - a `Keywords:` line
   - a `\date{}` line
   - a ~6.5 in text width

   Drop the 4 pt/1 pt title rules and the "Working draft." footer.
2. Author block: superscript affiliation numbers, a correspondence footnote and a "Code is provided at …" footnote on page 1.
3. Hyperlinks: change `teal!70!black` to `\definecolor{refnavy}{HTML}{001473}`, and use it for `linkcolor`, `citecolor` and `urlcolor`.
4. Bibliography: change `plainnat` to **`apalike`** with natbib (`Surname, I. (Year). Title. *Venue*.`), keeping `round,authoryear`.
5. Title: make it shorter and parallel, with no "via". Candidates in the reference's register: "Defining Compositional Representations by Recoverability"; "Recoverability and the Compositionality of Representations"; "Defining and Testing Representational Compositionality". The current title is eight words, which is acceptable but breaks the pattern.
6. Abstract: grow it from 182 to ~230–250 words and follow the reference arc:
   - hook with a concrete chain
   - gap ("Yet … remain …")
   - **"Here, we … by formalizing a simple idea: a representation is compositional when …"**
   - what follows (systematicity/productivity necessary vs contingent)
   - tests and evidence
   - **"Finally, we discuss implications for …"** (terminology; ML practice)

   Currently it ends on "a plan for evaluating", which is weak. End on implications instead, and add a Keywords line.

**B. Introduction**

7. Delete the **"Contributions." bulleted list**. Replace it with prose paragraphs that each end in a section pointer ("Section 2 develops this intuition …", "Section 3 formalizes …", "We test this in Section 4 …", "We close … in Section 5.").
8. Move the three introductory definitions (colloquial, behavioral, representational-informal) **out of `definition` environments**. Give them as italicized terms in prose, as the reference does with "This is what we call *novel compositional structure*", so that only the formal criterion is numbered.
9. Optional, but characteristic: open with an **epigraph**, e.g. a short Frege line, or the Fodor & Pylyshyn line on systematicity, with attribution and citation in the reference's format ("— Gottfried Wilhelm Leibniz, (Leibniz, 1898)").
10. **No teaser on page 1.** Move `fig:ops` (avocado) out of the introduction into the new Section 2 (Intuition) as the running example, or merge it as panel **a** of the conceptual Figure 1 alongside the taxonomy.

**C. Restructure sections to the reference skeleton**

| Ours now | Target |
|---|---|
| 1 Introduction (with defs, fig, contributions) | **1 Introduction** (5 prose paragraphs; related work in miniature in paragraph 2) |
| 2 Background (5 run-in paragraphs) | **2 Intuition for recoverable composition**: one running example (avocado *on* table vs avocado-chair blend, or a tensor-product binding vs a sum), three contrasting cases (role–filler: recoverable; superposition in disjoint regions: recoverable by vocabulary/region only; blend: not recoverable), ending with "The next section makes … precise." Fold the Philosophy/CogSci history into Intro paragraph 2 and Section 3.4. |
| 3 Recoverability: A Definition … (3.1–3.5) | **3 Defining representational compositionality** — 3.1 Preliminaries (parts, relations, `comp`/`rep`, Eq. binding); 3.2 Formal definition (**boxed** Definition 1 = separable + invertible relation, **boxed** Definition 2 = representational compositionality relative to 𝓜; Figure 1 = taxonomy/fibers schematic); 3.3 What the definition accounts for (prose, at most 1–2 displayed implications with "(proof in Appendix C.k)"); 3.4 **Relationship to other accounts** — run-in headings: *Behavioral vs representational systematicity* (Vegner), *Compression-based definitions* (Elmoznino 2024/2026), *Tree reconstruction error* (Andreas), *Compositional generalization benchmarks* (Hupkes, Lake), *Tensor products and VSAs* (Smolensky, Plate), *The classical objection* (Fodor & McLaughlin) — each ending with a one-sentence relation ("complementary", "a special case", "a plausible mechanism for …"). |
| 4 Testing Recoverability (suite, human profile, systems) | **4 Testing recoverability**: an opener restating the question with italics, plus a 3-bullet "each experiment is fully characterized by:" (domain; relation/operator and readout class; measures). Then 4.1, 4.2… one per experiment, each with **identical run-in headings**: **Domain. / Composites. / Readouts. / Measures. / Results.** Results follow the reference sequence (what was run, seeds; baseline and why it is the right control; sanity check; panel-by-panel "Figure 2c … We find that …"; a closing tie-back sentence to the definition). |
| 5 Related Work | **Delete**; contents go to 3.4 and Intro paragraph 2. |
| 6 Concluding Remarks + bulleted open questions | **5 Discussion**: a summary paragraph restating the definition in one sentence ("Formally, …"), then bold run-in implication paragraphs (e.g., **Terminology.**, **Architectures and training.**, **Continual learning and editing.**), then **Future work and limitations.** in prose with the open questions posed as direct questions. **No bullets.** |

**D. Definitions, propositions and boxes**

11. Add a tcolorbox style that mimics the reference: frame `#B2D9D9`, background `#F7FBFB`, bold title "Definition N. Name" with no parentheses. Use it for **Definition (separable and invertible relation)** and **Definition (representational compositionality)**. A sketch:
    ```latex
    \usepackage[most]{tcolorbox}
    \definecolor{boxframe}{HTML}{B2D9D9}\definecolor{boxfill}{HTML}{F7FBFB}
    \newtcbtheorem[number within=none]{defbox}{Definition}{colback=boxfill,colframe=boxframe,
      fonttitle=\bfseries,coltitle=black,separator sign={.},description delimiters none,
      sharp corners,boxrule=0.5pt}{def}
    \newtcbtheorem{notebox}{Note}{...same...}{note}
    ```
12. Follow every box with "This criterion requires …" and then **"Intuitively, … fails when …"** and a scope sentence ("Note that Definition 2 says nothing about …").
13. Replace the **enumerated list of six propositions** in Section 3.5 with 2–3 prose paragraphs. Keep at most one or two displayed results, e.g. non-identifiability as "m < kd ⇒ dim fiber ≥ 1 (proof in Appendix C.2)" and systematicity-as-necessity, then add "Intuitively, …" paragraphs. All six propositions stay in the appendix, renamed **Theorem/Proposition N (Name).** Each gets an **"Intuitively, the result states …" paragraph before its proof**, which our appendix lacks.
14. Put caveats in **Note boxes**:
    - **Note 1. Recoverability is relative to 𝓜** (negative results are absences under declared methods)
    - **Note 2. Finite vocabularies** (decodability borrowed from the data, from the Remark after Prop. 2)
    - **Note 3. Performance vs competence** (finite depth/noise limits)
15. State one **conjecture** in prose if we have one, e.g. "we conjecture that the primitives that maximize recoverability across relations are the intrinsic ones". Defend it in an appendix, mirroring their Appendix B.

**E. Tables → appendix; figures → multi-panel results**

16. The main text should carry **zero or one table**. Move `tab:cases` (8 capacities), `tab:human` (human profile), `tab:experiments` (test suite) and `tab:forms` to the appendix as **two- or three-column text tables** with captions above, numbered by appendix (Table B.1…). If the 2x2 `tab:twobytwo` stays, consider redrawing it as a panel of Figure 1.
17. For each experiment, build one **full-width four-panel figure** in the layout below. Use **blue #4c72b0** for role–filler/recoverable and **orange #dd8452** for blend/baseline (add greys #929292/#d5d5d5 and a third seaborn-deep color only if needed), shaded **SEM over seeds**, Helvetica-style sans labels, STIX math, ~1 pt lines, bold lowercase panel letters with titles ("b. Recovery vs. representation dimension"). The panels:
    - **a**: schematic of domain, operator and readout
    - **b**: main curve (recovery accuracy vs. dimension ratio / depth / vocabulary), familiar vs. novel parts
    - **c**: small multiples of secondary measures (context leakage, seed variance, error types)
    - **d**: qualitative examples (decoded composites, intervention maps)
18. Rewrite captions to the reference form: `Figure k:` with a **bold title phrase**, then **bold "a."…"d."** sentences that state *what is shown and what it shows*, color keys inline "(blue; role–filler)", and a final **bold "Seeds."** sentence on error bars and representative-seed panels with an appendix pointer. Aim for 150–250 words for result figures and ~100 for the conceptual figure. The current captions (30–70 words, descriptive only) are too short.
19. Recolor existing figures (`fig_taxonomy.pdf`, `fig_fibers.pdf`, `fig7_test_domain.pdf`) to the shared palette. Make `fig_taxonomy` single-hue muted blue (#5975a4) with 2 pt strokes, like their Fig 1, and give every figure full width.

**F. Content gaps that the reference format exposes**

20. The reference's Section 4 reports **run results**, but ours describes a *plan* ("We plan to evaluate…", "We describe the design only") and has a red `\note{Insert the initial results of TRELLIS here…}`. To match the reference, Section 4 needs at least one or two experiments **with results and figures**. The **Calibration** test (analytic operators × familiar vs novel parts × readout class) is the natural 4.1, because it needs no trained network and directly tests Props 2 and 5. One small trained-network experiment (MLP/transformer, ≥10 seeds, reversal + new-role) would be 4.2. Otherwise the paper reads as a position paper, which the reference is not.
21. Move the human reference profile to an appendix table plus one paragraph in the Discussion, or to Section 3.4. The reference has no separate "reference profile" subsection.
22. Experiment details (`app:protocol`) become **Appendix D-style** subsections mirroring 4.1/4.2, with the same run-in headings plus **Additional results.** and extra figure galleries. The threats-to-validity table can stay there.

**G. Voice and line-level edits**

23. Lengthen and balance sentences. Many of ours are 12–20 words; the reference averages ~25–35, joined by colons, semicolons and spaced em dashes. Use the reference's openers ("Intuitively,", "Importantly,", "Concretely,", "Formally,") and one **"Importantly, the claim is not that … Rather, …"** move to pre-empt the main misreading. Ours would be: "the claim is not that networks cannot behave systematically; rather, …".
24. Replace `{\it …\/}` with `\emph{}`. Keep italics for first-use terms (*recoverability*, *separable*, *invertible*, *blend*, *role–filler*) and for contrastive emphasis.
25. Turn the open-question bullets in the conclusion into direct questions inside prose.
26. Our reference cites (`\citet{...}` with the 2026 date) must use the reference's own framing of its quantity, C(x_a,x_b→x) = K(m_x | m_xa, m_xb). The draft's current line ("measure the compositional structure of objects by the information that a whole adds beyond its parts") inverts the reference's own clarification: "the claim is not that the raw whole contains information absent from its parts". Rephrase it as "the information in the best model of the whole that cannot be recovered from the best models of its parts". This also sharpens our contrast: their *recoverability* is of models from models; ours is of parts from the composite.

**H. Length targets for our version** (main text ~12–14 pp at 6.5 in width)

| Section | Pages |
|---|---|
| Intro | 1.5 |
| Intuition | 1.0 |
| Definition section | 5.0 (Def/boxes 2.0, consequences 1.0, relationship to other accounts 2.0) |
| Experiments | 4–5 (2 subsections × ~2.2 pp, each with one full-width figure) |
| Discussion | 1.0–1.3 |

Then references, then appendices A (formal setup / admissible primitives), B (conjecture or modes-of-combination tables), C (proofs with intuition paragraphs and Note boxes), D (experiment details, tables and galleries), E (capacities and human profile tables).
