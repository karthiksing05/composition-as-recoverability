# Review R2-B: "Defining Compositional Representations by Recoverability" (revised)

Reviewer B: cognitive science and philosophy of mind.

I read the revised `recoverability-theory/main.tex` (575 lines, including all appendices), the bib entries for every philosophical and psychological citation, and the rendered pages p01 and p07. I checked each round-1 point against the text. Line numbers below refer to the revised main.tex.

## 0. Summary of the revision

The revision fixes the central philosophical problem from round 1:
- The abstract no longer claims "necessity". Prop. 6 is now presented as the unbinding argument of tensor-product representations and as "a statement about what the states determine, not about what a system does with them" (l.144, l.431).
- Smolensky is credited, and the explanatory challenge (Aizawa; Goodale & Mascarenhas; "contingent on training") is conceded explicitly (l.144, l.159, l.286).

The new population-readout design in Sec. 4.2 (pool of 400 glyphs the network never saw, plus an untrained-network control) separates "the state contains the part" from "the system's own decoder reads it". That is a better answer to my W2 encoder/relation confound than the frozen-encoder experiment I asked for.

The new interchange-intervention experiment (Sec. 4.3) is a welcome way to engage Fodor & McLaughlin. However, it is now overclaimed (N1 below).

The cognitive-science sections improved less than the formal and ML sections:
- the "People" paragraph and Table 6;
- the concept-combination row of Table 5;
- the missing human fresh-part paradigms.

## 1. Status of round-1 issues

### Major weaknesses

| R1 item | Status | Notes |
|---|---|---|
| **W1** Prop. 6 restates Smolensky; it does not answer F&M; drop "necessity" | **Mostly fixed** | The abstract and Sec. 3.3 are reframed correctly (l.144). Smolensky is credited (l.144, l.159, l.431). The explanatory/architectural half of F&M is conceded (l.144 last sentence; l.286). Two problems remain. (a) A new overclaim: the interchange intervention is said to "answer" F&M's causal-efficacy objection (l.144, l.159, l.236); see N1. (b) The post-1990 literature I listed is still absent: Smolensky 1995, Chalmers 1990, Hadley 1997, McLaughlin 2009, Calvo & Symons 2014. `smolensky1987` and `smolensky1988` are in the bib but uncited, although they, not `smolensky1990`, are the papers F&M 1990 actually replied to. |
| **W2** "Fresh part" undefined for concepts; encoder generalization vs relation invertibility | **Partly fixed** | Encoder vs relation: **fixed**. The population readout plus the untrained-network control (l.212, l.216) cleanly shows the state is recoverable while the decoder is vocabulary-bound. Freshness: l.94 now gives three senses (fresh to the system, fresh to the readout, and "learned after the composition procedure was in place" for people). That is progress, but the senses are not aligned with Def. 1, and the abstract and intro use the wrong one (N2). Still missing: what D is for concepts; the tension with Fodor–Lepore reverse compositionality and atomism; the human paradigms that test each sense (Jabberwocky/syntactic bootstrapping, the wug test, Marcus et al. 1999). |
| **W3** Human evidence described selectively | **Partly fixed** | Frankland & Greene is now correctly hedged ("over a closed set of nouns and verbs", l.288, Table 6). Lake et al. 2019 is now correctly classed as "known part, new role". Wisniewski's relation-linking is now credited as role–filler, but a new inaccuracy appears ("just as often", N3). Chase & Simon no longer claims "recoverable parts", but the random-position caveat is still missing, even though `gobet1996` (Recall of random and distorted positions) is in the bib and is cited elsewhere for the wrong claim (N5). Illusory conjunctions, visual working-memory binding limits, and Hummel & Holyoak are still absent (`hummel1997` is in the bib, uncited). The question of whether the human profile favours a graded criterion is not discussed. |
| **W4** Concept-combination models mapped onto Gaussian ceilings | **Not fixed** | Table 5 (l.472) still puts selective modification and composite prototypes in a row whose exemplar is x + αy with "No; ceiling 1/(1+α²)". The new App. B text (l.461) now describes selective modification correctly as acting on "those attributes of the head to which it applies", i.e., slot-wise. That makes the table row internally inconsistent with the paper's own gloss. The pet-fish debate is still not discussed, although `osherson1981`, `fodor1996`, `kamp1995` and `hampton2012` are all in the bib and none is cited. |
| **W5** Inaccurate lineage citations | **Partly fixed** | Robbins: **fixed** (now "criticized"; Fodor & Lepore 2002 "defended", l.59). van Gelder: **partly fixed**. The functional/concatenative gloss is fine, but the holistic-processing branch (structure-sensitive operations without decomposition; Chalmers 1990) is still not addressed, and that is exactly where van Gelder and this criterion diverge. Vegner et al.: **not fixed** (N9). Hadley: **partly fixed**. `hadley1994` is now cited in Sec. 3.4 (l.159), but l.210 still labels a flat novel-role split "Hadley's strong systematicity" without citation. Hadley's strong systematicity requires novel positions in embedded as well as simple sentences. |
| **W6** Claims beyond evidence; title; pretrained "new sentences"; reference circularity | **Partly fixed** | Pretrained relabelling: **fixed** in the abstract, l.259 and l.290, but not in l.179 (N7). Structured distributions: **partly fixed**. The App. D robustness paragraph adds sparse, uniform and correlated parts, but with linear readouts only, so it does not test whether sparse superposition is recoverable (N6). Contextual-token part codes: deferred to limitations (acceptable). Title: given that Def. 2 and l.120 now state the relativity to U and M plainly, I withdraw the request to retitle. The abstract should still name the declared readout class (N2). The reference operator in Def. 2 is still unspecified (minor 4). |

### Minor issues

| R1 minor | Status |
|---|---|
| 1 Abstract's opening empirical claim, uncited | Not fixed (l.46). Acceptable in an abstract if the Introduction supports it; at present the Introduction supports only the first half (Frege), not "take the meaning apart again". |
| 2 Avocado "cannot distinguish" | Not fixed (l.57). See N10. |
| 3 F&P context-independence vs affine code | Partly fixed. The affine restriction is now flagged as a choice "so that the requirement has content" (l.114). The type/code distinction and context-sensitive contributions (*red* hair vs *red* apple) are still unmentioned. |
| 4 Which reference operator | Partly fixed. Sec. 4.4 uses concatenation and the sum. Def. 2 does not say which reference. |
| 5 Prop. 7 used for Geirhos / Kamb & Ganguli | Not fixed (Table 3, l.501). |
| 6 "Imitation", "Blending that fails" | Not fixed (Table 1, l.170). |
| 7 Role–filler advantage at small V | Fixed in substance (l.220). "As role-factorization predicts" for 0.94 vs 0.88–0.93 is still generous. |
| 8 Seed spread as "signature of contingent systematicity" | Not fixed (l.220, l.286). At V = 8 every architecture is ≤ 0.21 on new roles, so the high SD at a low mean is just as well explained by an under-determined problem. |
| 9 RAAM depth-3 failure is a train/test shift | Not fixed (l.254, l.282). See N13. |
| 10 GPT-2 causal pooling | Partly fixed (second caveat, l.265). |
| 11 Blind review | N/A (preprint with named author). |
| 12 Prop. 4 connected design | Fixed (l.391). |
| 13 "Not stable under any noise" | Not fixed (l.355). Minor. |
| 14 Canva images | Disclosed in the caption; acceptable for a preprint. |
| 15 Missing citations | Mostly not fixed: Marcus 2001, Hummel & Holyoak, Szabó's SEP entry, Pylyshyn & Fodor's productivity argument. Piantadosi 2016 is still appendix-only. |

## 2. New issues, ranked by severity

### N1 (major). The interchange intervention is presented as answering Fodor & McLaughlin; it answers only an interventionist reconstruction of the objection, and the result is largely entailed by small δ (l.144, l.159, l.236)

F&M's objection is not empirical in their own framing. They argued that tensor-product constituents are "imaginary": the complex vector can be decomposed in indefinitely many ways, and processing that is causally sensitive to the whole vector is not thereby sensitive to any one decomposition's components *as constituents*. Their demand was that constituents be tokened and that structure-sensitivity follow from the architecture.

The intervention shows something real: selective counterfactual dependence of the decoder's role-1 output on the range of B₁, which is causal relevance in Woodward's interventionist sense and causal abstraction in Geiger's sense. That is a fair, useful reconstruction, but it is a reconstruction. Two further points matter.

1. **Entailment.** With h ≈ c + B₁e(a) + B₂e(b) and δ = 0.02–0.04, the edit B₁(ê₁(h(a′,b)) − ê₁(h(a,b))) is approximately h(a′,b) − h(a,b). The edited state is therefore approximately the donor state, and rendering the donor at 0.78 vs 0.81 follows from factorization plus decoder accuracy. The informative quantity is the 15% leakage into role 2, which the text mentions only in passing.
2. **The control is weak.** A random direction of the same norm leaves the state manifold, and almost any decoder breaks off-manifold, so R² = −1.12 is expected. Add two in-manifold controls:
   - the same edit placed in range(B₂), which should change role 2 and spare role 1;
   - an edit of the same norm along a random *natural* direction, h(c,d) − h(a,b) for random parts c, d, which shows that a role-agnostic in-distribution change of the same size leaks into both roles.

**Fix (text):**
- l.144: replace "That objection is empirical for a given system, and we test it directly:" with "Read in interventionist terms, as a claim that the components need not make a difference to processing, the objection becomes testable for a given system:".
- l.159: replace "and leaves Fodor and McLaughlin's objection as an empirical question for each system, which interchange interventions answer \citep{geiger2021}" with "and recasts Fodor and McLaughlin's causal-efficacy objection in interventionist terms, which interchange interventions can test for a given system \citep{geiger2021}; their stronger demands, that constituents be tokened and that their efficacy follow from the architecture, are not addressed by any criterion on states".
- l.236: replace "which answers, for these networks, the objection that recoverable constituents need not be causally efficacious \citep{fodor1990}" with "so, for these networks, the recovered constituents make the selective difference to processing that an interventionist reading of \citet{fodor1990}'s objection asks for. Because the states are nearly role-factorized, this result is close to what Proposition~\ref{prop:systematic}(iv) predicts; the informative quantity is the leakage into the untouched role".
- Add the in-manifold controls if feasible. Otherwise, at least report leakage for the random-edit control, so that 15% has a reference point.

### N2 (major). "Fresh" is used in a sense that Definition 1 does not use (abstract l.46; Intro l.59; l.94)

Def. 1 is a population quantity over D. It is estimated with readouts fitted on a probe pool and evaluated on other draws. The only freshness it needs is **freshness to the readout**. Freshness to the *system's training* matters only for the behavioural column of Table 1 and for the system's own decoder (Sec. 4.2).

Yet the abstract says "a part never seen before", and l.59 says "the decisive test is recovery of parts the system has never encountered". This matters for the philosophy as well. Under the paper's own definition, the pretrained probe (readout-fresh nouns) is a legitimate estimate of Def. 1 for D = the 305 nouns, so its caveat is weaker than l.290 implies. Meanwhile the human sense ("learned after the composition procedure was in place") concerns the person's *own* procedure, which corresponds to the system's decoder, not to Def. 1.

**Fix:** replace the senses sentence in l.94 ("What counts as fresh ... say which.") with:

> Freshness is always relative to a procedure: a part is fresh to a procedure if no composite containing it was used to fit that procedure, even if the part itself is known. Definition~\ref{def:recoverable} needs only parts fresh to the readout. Whether a system's \emph{own} decoding procedure handles parts fresh to it is a separate, behavioral question (Table~\ref{tab:twobytwo}). For glyphs and objects, a part can be fresh to both. For a word in a pretrained model it can be fresh only to the readout. For a concept in a person, the relevant sense is fresh to the person's composition procedure: a word or object learned in isolation, or in one role, and never yet combined, since a concept the person does not possess cannot be a constituent of a thought they entertain \citep{fodor2002}. What $\D$ is for concepts is an idealization; for words it can be operationalized by nonce forms with taught meanings.

Further edits:
- Abstract: replace "including a part never seen before" with "including a part the procedure was never fitted on". Replace "by the same fixed procedure" with "by the same fixed procedure of a declared kind, such as an affine map".
- l.59: replace "recovery of parts the system has never encountered" with "recovery of parts on which the reading procedure was never fitted".

### N3 (moderate). Wisniewski is now mischaracterized in the other direction (l.288; Table 6 l.523)

"just as often by mapping a property ... or by hybridizing them" is not what Wisniewski (1997) reports. Property mapping is common, especially for similar concepts. Hybridization is comparatively rare.

It is also doubtful that property mapping is a "blend" in this paper's sense. In "zebra clam" the head is intact and recoverable, and only one property of the modifier survives. That is a lossy modifier filling a slot of the head, closer to selective modification than to x + y.

**Fix (l.288):**

> Novel noun--noun compounds are often interpreted by linking the two concepts through a thematic relation, a role--filler mode; often by mapping one property of the modifier onto the head, which keeps the head and only a fragment of the modifier; and less often by hybridizing them, a blend \citep{wisniewski1997}.

Change the Table 6 row to match.

### N4 (moderate). Wason (1968) does not show that "content alters selection" (Table 6, l.524)

Wason (1968) used abstract materials and showed that people fail to select the falsifying card. Content effects come from later work: Johnson-Laird, Legrenzi & Legrenzi 1972; Griggs & Cox 1982; Cheng & Holyoak 1985.

**Fix:** either cite one of these for "content alters selection", or change the cell to "Belief overrides validity; people fail to seek falsifying cases of an abstract rule".

### N5 (moderate). Chess citations attached to the wrong claims (l.481; Table 6 l.521)

- `gobet1996` in the bib is "Recall of random and distorted chess positions" (Memory & Cognition). At l.481 it is cited for "CHREST adds templates with slots for variable fillers". The template paper is Gobet & Simon (1996), "Templates in chess memory: A mechanism for recalling several boards", *Cognitive Psychology* 31:1–40. Add it and cite it there.
- Use `gobet1996` where it belongs, for the Chase & Simon caveat. Table 6 row "Parts kept within chunks": append "; the expert advantage largely disappears for random positions, so recall depends on familiar configurations \citep{gobet1996}". This row is the paper's clearest human example of vocabulary-bound recovery. Saying so strengthens the argument.

### N6 (moderate). The superposition verdict depends on U and D, and the robustness check does not test the case that would overturn it (abstract; Fig. 2 caption l.129; l.194; App. B l.461; App. D l.541)

With sparse parts, a sparse-recovery readout (L1 or thresholded) recovers fresh parts from an m < kd projection once m grows on the order of s log(d/s). The sparse robustness run used linear and Gaussian-kernel readouts only. So the 0.37 does not test what App. B asserts about Elhage et al., and whether superposition "counts" turns on whether U includes sparse decoders.

That is a consequence of the definition's relativity, and the paper should say so where the verdict is stated:
- Abstract: replace "and vector-symbolic superposition behaves like a lossy blend" with "and fixed-width vector-symbolic superposition of dense parts behaves, under affine readouts, like a lossy blend".
- After l.194, add: "With sparse parts and a readout class that includes sparse recovery, a projection into fewer dimensions can recover fresh parts; whether superposition is compositional is then a question about the declared $\U$ and $\D$, as Definition~\ref{def:repcomp} intends."

### N7 (minor). Leftover "new sentences" (l.179)

Replace "and do pretrained encoders recover the parts of new sentences?" with "and do pretrained encoders recover nouns on which the readout was not fitted?"

### N8 (minor). Attribution of "tokening" and of Smolensky's reply (l.159)

- The requirement that constituents be tokened whenever the complex is tokened is F&M (1990)'s explicit formulation. F&P (1988) argue from combinatorial syntax and semantics and from structure-sensitive processes. Write "\citet{fodor1988} argued that systematicity requires constituent structure, and \citet{fodor1990} that classical constituents are tokened whenever the complex is".
- F&M's targets were Smolensky (1987, 1988). Cite `smolensky1987` and `smolensky1988` (both in the bib) alongside `smolensky1990`.

### N9 (minor). Vegner et al. (2025) characterization (l.57)

Their paper is an opinionated survey of behavioural vs. representational **systematicity**. "Showed" overstates what a survey does, and "left its criterion to future work" should be checked against the text. Suggested replacement:

> \citet{vegner2025} distinguished \emph{behavioral} from \emph{representational} systematicity, the first a property of outputs on unseen combinations and the second of the internal states, and argued that benchmarks of the first do not establish the second.

### N10 (minor). The triviality of the colloquial principle (l.57)

"Any system that computes a function of its inputs satisfies it" is the Zadrozny (1994) / Westerståhl (1998) triviality result for meaning functions. Cite it; it makes the point authoritative rather than rhetorical. Also add "in representational form" so the avocado contrast is not read as a claim about meaning determination: "so it cannot distinguish, in representational form, an avocado placed \emph{on} a chair from ...".

### N11 (minor). Hadley label (l.210)

Replace "(Hadley's strong systematicity)" with "(a novel-role split in the spirit of \citeauthor{hadley1994}'s strong systematicity, \citeyear{hadley1994}, without embedding)".

### N12 (minor). Table 1 characterizations (l.169–170)

- Montero et al. (2021) held out *combinations* of factor values, not parts. "A decoder learned from too few parts" is the authors' own Sec. 4.2 finding. Write "Failure of performance: the parts are in the state but the decoder does not read them (Section~\ref{sec:networks}; cf.\ \citealt{montero2021})".
- Rename "Imitation" to "Vocabulary-bound generalization".
- Rename "Blending that fails" to "Neither: parts lost and behavior fails".

### N13 (minor). Depth wording (l.254, l.282)

Depth-2 trees are already composites of the composer's own outputs, so "invertible on the composites it was trained on, not on its own outputs" is inaccurate. Write "invertible at the depths it was trained on, not one level deeper". In the fifth requirement, add "which may reflect training only at depths 1 and 2 as much as the relation itself" unless a depth-1–3/test-4 run is added.

### N14 (minor). Pet fish (W4) as a short paragraph after l.461

The bib already holds what is needed. Suggested text:

> The framework also locates the pet-fish debate. \citet{osherson1981} and \citet{fodor1996} argued that prototypes do not compose, because the typicality of a conjunction is not fixed by the typicalities of its parts; \citet{kamp1995} and \citet{hampton2012} replied that prototypes compose by default with overrides. In our terms, default inheritance is a role-factorized component, and overrides and emergent attributes are the residual $\eta$; the debate is about the size and structure of $\delta$, not about whether composition occurs.

Then either drop the ceiling cell for selective modification and composite prototypes in Table 5, or split the row so that the ceiling applies only to x + αy.

### N15 (minor, prose). The abstract is ~330 words against Elmoznino et al.'s 248

The two experimental sentences are hard to parse. Suggested replacements:
- "In small networks trained on glyph composites, the states keep fresh glyphs recoverable at every vocabulary size; the networks' own decoders read them only after training on many distinct glyphs, like readouts fitted on the training glyphs alone; and the role components we recover are the ones the decoders use."
- "Pretrained text encoders only partly recover nouns on which the readout was not fitted, and CLIP's text encoders recover them least."

The People paragraph (l.288) should open with what the evidence shows, not with "points to recoverable composition". Suggested opening: "Human evidence shows role-separated codes and recombination of known parts, but almost none of it was gathered with parts that are fresh in our sense." After the paragraph's existing test proposal, add the paradigms that come closest:

> The paradigms that come closest are the wug test \citep{berko1958}, rule learning over novel syllables \citep{marcus1999}, and syntactic bootstrapping of novel verbs \citep{gleitman1990,naigles1990}; illusory conjunctions \citep{treisman1982} and binding limits in working memory \citep{wheeler2002} suggest that human recovery of fresh conjunctions is capacity-limited and graded.

This is the single most useful addition for a cognitive-science reader.

## 3. Structure, style, and accuracy of the remaining lineage

- **Structure.** The paper now follows Elmoznino et al. (2026) closely:
  - epigraph-led Introduction;
  - a math-light "Intuition" section with a running example;
  - "Defining ..." with Preliminaries, Definition, consequences, and "Relationship to other accounts" serving as the related work;
  - experiments with bold run-in Composites/Readouts/Results headings;
  - a Discussion ending in "Future work and limitations";
  - no separate Related Work or Conclusion.

  This is good.
- **Characterizations I checked and consider accurate:**
  - Fodor & Pylyshyn epigraph and the performance limits on productivity (l.123);
  - F&M's architectural demand (l.286);
  - Robbins 2005 as a critic of reverse compositionality;
  - Fodor & Lepore 2002 as defenders;
  - van Gelder's functional vs concatenative distinction (as far as it goes);
  - Frankland & Greene 2015 (now properly hedged);
  - Schwartenbeck et al. 2023;
  - Lake et al. 2019 (as "known part, new role");
  - Rips 1994; Piantadosi et al. 2016; Evans et al. 1983;
  - Smith et al. 1988 and Hampton 1987 in the App. B prose;
  - Aizawa 2003 as the source of the "contingent" framing.
- **Could not verify, please check against the source:** Goodale & Mascarenhas (2026, TACL) as restating the challenge "as one about what an architecture guarantees by default"; Woydt et al. (2025) on meta-learning (the title is consistent with the claim).
- **Reverse compositionality (l.59).** "Close to" is acceptable. Add one clause saying that Fodor & Lepore's thesis concerns concept possession and understanding, not read-back by a fixed procedure. This also motivates the "fresh to the composition procedure" sense for people (N2).

## 4. Overall verdict

**Minor revision (text only), borderline major on the cognitive-science sections.** My score moves from 5/10 to 6.5/10.

The formal core and the ML experiments now carry the argument honestly, and the F&M framing is close to right. What blocks acceptance is a set of textual corrections, none of which requires new compute:

1. **N1.** Scope the interchange-intervention claim to an interventionist reading of F&M (three sentence replacements). Optionally add the range(B₂) and natural-direction controls.
2. **N2.** Align "fresh" with Definition 1 in the abstract, l.59 and l.94, using the replacement paragraph above.
3. **N3, N4, N5.** Correct the Wisniewski, Wason and Gobet & Simon citations, and add the random-position caveat to the chess row.
4. **N6.** Scope the superposition verdict to dense parts under affine readouts.
5. **W4 / N14.** Fix the Table 5 concept-combination row and add the pet-fish paragraph.
6. **N15.** Add the human fresh-part paradigms (wug, Marcus et al. 1999, syntactic bootstrapping) and the binding-limit literature to the People paragraph.

With items 1–6 done, I would rate the paper 7.5/10 and recommend acceptance. Items N7–N13 are quick fixes I would expect to see, but they would not block acceptance.
