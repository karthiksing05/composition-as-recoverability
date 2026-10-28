# Review R3-B: "Defining Compositional Representations by Recoverability" (third round)

Reviewer B: cognitive science and philosophy of language and mind.

I read the current `recoverability-theory/main.tex` (593 lines, including all appendices), the bib entries for every cited work in the related-work, Discussion and appendix tables, and rendered pages p01 and p04. Line numbers refer to the current main.tex.

I re-checked citations against their sources: by web search for Westerståhl 1998, Kamp & Partee 1995, Vegner et al. 2025 and Goodale & Mascarenhas 2026; from the arXiv text for Elmoznino et al. 2026 (`notes/style_ref.txt`); and from my knowledge of the remaining works.

## Verdict

**Minor revision.** Score: 7.5/10, up from 6.5.

Almost every blocking item from round 2 is fixed:
- **N1 (interchange intervention vs. Fodor & McLaughlin):** fixed, and the requested in-manifold controls were added: role 2, a random direction in range(B₁), and the natural direction.
- **N2 (freshness):** fixed in the definition section and in the abstract.
- **N3–N5 (Wisniewski, Wason, Gobet & Simon):** fixed.
- **N6 (superposition scope):** fixed at l.194 and in the abstract.
- **N14 (pet fish):** added.
- **N15 (human fresh-part paradigms):** added.

What remains:
- two confirmed misattributions in text that was added this round, one of which I introduced myself in round 2;
- one place where the pet-fish paragraph takes a side in the debate while presenting itself as neutral;
- a few leftover "never seen before" phrasings that contradict the new freshness paragraph;
- a dangling cross-reference;
- one overclaim each in the Discussion and in the taxonomy caption.

None of these needs new compute.

---

## A. Confirmed misattributions (fix before posting)

### A1. Kamp & Partee (1995) did not reply that prototypes compose (l.499)

The current text reads: "\citet{kamp1995} and \citet{hampton2012} replied that prototypes compose by default with overrides".

Kamp & Partee's paper largely concedes the standard objection. As Fodor & Lepore (1996) summarize it, they hold that the objection "is probably right in the long run" and that the failures are confined to particular construction types. Their positive proposal is a hybrid: prototypes act as a separate, non-compositional component alongside a supervaluation semantics, with context-dependent recalibration. "Default inheritance with overrides" is Hampton's composite-prototype view (Hampton 1987; Jönsson & Hampton 2006). Prinz (2002) gives the same view as retrieval–composition analysis.

**Fix:**

> \citet{osherson1981} and \citet{fodor1996} argued that prototypes do not compose, because the typicality of a conjunction is not fixed by the typicalities of its parts; \citet{kamp1995} largely conceded the point and kept prototypes as a non-compositional component beside a compositional semantics, while \citet{hampton1987} and \citet{hampton2012} argued that prototypes combine by inheriting attributes by default, with overrides and emergent attributes.

Optionally add Prinz 2002, *Furnishing the Mind*, Ch. 11, for the default/override view.

### A2. Westerståhl (1998) argues against the vacuity result it is cited for (l.57)

The current text reads: "Any system that computes a function of its inputs satisfies it \citep{zadrozny1994,westerstahl1998}".

Zadrozny (1994) argues that any semantics can be re-encoded compositionally, so the bare principle is formally vacuous. Westerståhl's paper is titled "On mathematical proofs of the vacuity of compositionality", and it is a critique of that claim and of two earlier ones in the same vein. Lappin & Zadrozny (2000) wrote a rejoinder to it. Citing Westerståhl as support inverts his position.

This was my suggestion in round 2 (N10), and I retract it.

The paper's own point is also more elementary than Zadrozny's: if the state is a function of the parts' codes and the mode of combination, the colloquial principle holds by construction. Neither paper is needed for that point.

**Fix:**

> Any system that computes its state as a function of its inputs satisfies it, and with freedom to choose the meanings of the parts even the formal homomorphism requirement can be met by any semantics \citep{zadrozny1994}; but see \citet{westerstahl1998}.

Alternatively, cite Zadrozny alone, or cite Szabó's SEP entry "Compositionality" for the triviality worry.

---

## B. Substantive issues, ranked by severity

### B1 (moderate). The pet-fish paragraph decides the debate while presenting itself as neutral (l.499, Table 5 l.488)

The paragraph ends: "the debate is about the size and structure of $\delta$, not about whether composition occurs."

For Fodor & Lepore (1996) that is exactly what is at issue. Their argument is that the emergent content of PET FISH comes from world knowledge about pet fish, not from the constituents. On their view, any residual η of that kind is a failure of compositionality, not a small quantity of it. Recasting the debate as one of degree is a substantive, anti-Fodorian move. It is defensible, and it fits the paper's graded framing, but it should be stated as the paper's position.

There is a second problem. The paper does not show that "default inheritance is a role-factorized component" for modifier–head combination. That combination is not a fixed k-role relation with part-independent B_i. In selective modification the effect of the modifier depends on which attribute of the head it targets, and that is exactly the co-part dependence that l.114 calls a failure of role-factorization.

**Fix.** Replace the last two sentences of l.499 with:

> In our terms, default inheritance can be read as an approximately role-factorized component and overrides and emergent attributes as the residual $\eta$. This turns part of the debate into a question of degree, the size and structure of $\delta$, which is the graded position of this paper; for \citet{fodor1996}, any residual supplied by world knowledge rather than by the constituents already counts against compositionality.

In Table 5 (l.488), change "Default inheritance only; overrides and emergent attributes form $\eta$" to "Approximately, via default inheritance; overrides and emergent attributes form $\eta$".

### B2 (moderate). "Never seen before" survives in two places and contradicts the freshness paragraph (l.129, l.278)

The l.94 paragraph now says correctly that Def. 1 needs only parts fresh to the readout. Two places still say otherwise:
- **Fig. 2 caption (l.129):** "recovers parts never seen before". The figure appears on p.4–5, right where readers form their understanding of the definition.
- **Discussion opening (l.278):** "including a part never seen before". This is the paper's one-sentence restatement of its thesis, and it uses the sense the abstract was revised to avoid.

**Fix:**
- l.129: "recovers parts on which the readout was never fitted".
- l.278: "including a part on which the reading procedure was never fitted".

### B3 (moderate). The taxonomy caption overstates "only" and conflicts with the sparse-superposition caveat (l.129)

The caption makes two unqualified claims:
- fixed-width superposition "returns parts only by clean-up over a known vocabulary";
- "Only role–filler composition ... satisfies Definition 2".

The paper's own l.194 says that with sparse parts and a sparse-recovery readout class, a narrower projection can recover fresh parts. App. B (l.476) says superposition of sparse features "is recoverable on its sparse domain".

Definition 2 also requires role-factorization and an informative part code, not recovery alone.

**Fix:** "... keeps roles distinct but, for dense parts under affine readouts, returns parts only by clean-up over a known vocabulary ... Under affine readouts, only role–filler composition ... recovers parts on which the readout was not fitted."

Also qualify the first requirement in the Discussion (l.282), "a state narrower than the parts it combines loses them", with "dense".

### B4 (moderate). App. B asserts what the paper did not test (l.476)

The sentence is: "such a relation is recoverable on its sparse domain and should be tested with fresh parts drawn from that domain".

The only sparse run (App. D, l.559) used linear and kernel readouts and gave 0.38 for the half-rank projection. As written, the sentence asserts recoverability and recommends testing it in the same breath.

**Fix:** "such a relation may be recoverable on its sparse domain under a readout class that includes sparse recovery; our sparse-part calibration (Appendix D) used only linear and kernel readouts and does not test this."

### B5 (minor–moderate). The seed-spread overclaim returns in the Discussion (l.286)

l.220 is now correctly hedged: the spread "bounds, rather than measures, the contingency on training". l.286 still says "remains contingent on training, as the seed spread at small vocabularies shows".

**Fix:** "... remains contingent on training, which the seed spread at small vocabularies is consistent with (Section 4.2)."

### B6 (minor). Dangling cross-reference (l.259)

The sentence is: "the fresh nouns are fresh to the readout, the second sense of Section 3". The l.94 paragraph no longer enumerates senses.

**Fix:** "the fresh nouns are fresh only to the readout (Section~3.1), which is all Definition~1 requires."

The limitations paragraph (l.290) should then drop the implication that this is a weakness of the estimate of Def. 1. It is a weakness only for the behavioral column.

### B7 (minor). Discussion summary claims more than was measured (l.280)

1. "The states ... were recoverable, and nearly role-factorized, whatever their training". δ was measured only on trained networks at V ∈ {16, 64} (Sec. 4.3), not on untrained ones or at V = 8. Drop "and nearly role-factorized" or add "after training".
2. "Its state kept every leaf one and two levels beyond its training". The population readout was run at depths 3, 4 and 5, which are one, two and three levels beyond the trained depths 1 and 2. Write "up to three levels beyond its training depths".
3. "Training itself made the difference for GPT-2 and T5". Random-weight numbers are reported only for GPT-2 and T5-base (l.263). Write "for GPT-2 and T5-base".

### B8 (minor). The fourth requirement ends with an unclear qualifier (l.282)

The clause is "although this holds for in-distribution changes only". Read literally, it says locality holds only for in-distribution edits. But the out-of-distribution random edit is the one that spills, and that is the control, not a limitation.

The intended caveat is presumably that the intervention was tested only with donor codes from the part distribution. If so, write: "... although we tested this only with donor components drawn from the part distribution."

### B9 (minor). The "Coverage of combinations ... matters less than the number of distinct parts" claim generalizes from one synthetic domain (l.282)

This is the paper's most quotable takeaway for ML readers, and it rests on one domain: glyphs with an 8-dimensional latent, plus a fixed-pairs control. It also echoes Proposition 5, which holds for affine procedures only.

**Fix:** "In our networks, coverage of combinations, which benchmarks reward, mattered less than the number of distinct parts."

### B10 (minor). The abstract's first sentence states reverse compositionality as fact (l.46)

The sentence is: "they can take the meaning apart again into the same parts". l.59 correctly calls this a thesis that Fodor & Lepore defended and Robbins criticized. The opening of the abstract asserts it as an empirical given.

**Fix:** "... and, on an influential view, take it apart again into the same parts."

### B11 (minor). The abstract has grown (≈360 words; r2 ≈330; Elmoznino et al. 2026 ≈250)

The freshness and superposition qualifiers were needed, but other material can be cut:
- The sentence "On any finite vocabulary ... single composite" packs three results. Keep the lookup result and the ceiling, and move "a shared part code fixes every role reversal from a single composite" to the Introduction.
- Shorten the final sentence to "Finally, we state five requirements a network must meet to earn the term."

The page-1 render (p01) shows the abstract filling roughly half the page, which is noticeably heavier than the reference paper.

### B12 (minor). The "human reference profile" is a qualitative list, not a profile against which systems can be compared (l.288, l.505, Table 7 caption)

Table 7 has no quantities, conditions or effect sizes. A system cannot be "compared against" it except by checklist.

**Fix:** call it "a summary of human signatures" (caption and l.505). In l.288, write "Appendix C summarizes this evidence; turning it into a quantitative reference profile requires the fresh-part paradigms above."

### B13 (minor). Definition 2 packs four definitions into one box (l.116–118; p.4)

The box defines representational compositionality, κ, informativeness, and productivity to depth n. Elmoznino et al. (2026) state one notion per box.

**Fix:** split informativeness and productivity into a third box ("Informative part code; productivity"), or make productivity a numbered definition after Prop. 9. This is purely presentational, but it would make the graded definition easier to cite.

---

## C. Citation spot-check (14 checked closely, more lightly)

| Claim (line) | Source | Verdict |
|---|---|---|
| Vegner et al. distinguish behavioural from representational systematicity and argue that benchmarks test the first (l.57) | ACL 2025 abstract | **Accurate.** They also propose mechanistic-interpretability measures for the second, which the paper could mention in one clause. |
| Goodale & Mascarenhas restate the challenge "as one about what an architecture guarantees by default" (l.159) | arXiv 2606.14512 full text: "classical symbolic systems are systematic by default"; connectionism "does not guarantee by default"; MLC "beholden to what precisely appeared in the meta-learning phase" | **Accurate.** My r2 "could not verify" is resolved. |
| Woydt et al.: meta-learning has not yet produced human-like systematicity (l.159) | Title and abstract | Accurate. |
| Elmoznino et al. 2026: novel structure in a whole beyond the models of its parts (l.57, l.157) | arXiv 2610.20922 abstract: "parts, considered together, reveal shared structure that is not captured when those parts are modeled separately" | Accurate. However, their account concerns compositional structure of *objects/data* (strings, images), not representations specifically. Add "of an object" at l.57 so it is not listed as a rival definition of *representational* compositionality. At l.157, "a fixed-width superposition ... compresses well" is asserted, not shown; write "would plausibly compress well". |
| Fodor & Pylyshyn 1988: a constituent contributes the same thing to every complex (l.114) | F&P p.42: "approximately the same semantic contribution" | Accurate. Consider quoting "approximately": it licenses the paper's graded δ, which is a point in the paper's favour. |
| Fodor & McLaughlin 1990: constituents tokened whenever the complex is; causal efficacy (l.144, l.159) | | Accurate. |
| Hadley 1994 weak/strong systematicity; Hupkes et al. 2020 tests (l.159, l.210) | | Accurate. |
| van Gelder 1990 functional vs concatenative (l.59) | | Accurate as far as it goes. |
| Uselis et al. 2026 (l.57, l.153) | Title: "... in **Vision** Embedding Models" | Slightly broadened. Write "vision embedding models". |
| Lippl & Stachenfeld: conjunction-wise additivity, "memorization leak" (l.153, l.470) | | Accurate. |
| Sutter et al. 2025: unrestricted alignment maps make causal abstraction vacuous (l.120) | | Accurate. |
| Montero et al. 2021; Chaabouni et al. 2020 as off-diagonal cells (l.155, Table 1) | | Accurate. The "cf." is appropriate. |
| Wisniewski 1997 (l.288, Table 7) | | Accurate now. |
| Conwell & Ullman: about 22% of DALL-E 2 images matched, 15 relations (Table 6) | | Accurate. |
| Li et al. 2023 SLOG: "at most 40.6% on the structural cases" (Table 6) | Abstract: transformers "up to 40.6%" overall | Essentially accurate. SLOG consists entirely of structural cases, so write "at most 40.6% on SLOG". |
| Gobet & Simon 1996a/b; Chase & Simon (Table 7, l.497) | | Accurate now. |
| Griggs & Cox 1982 for "content alters which cases they select" (Table 7) | Title: "The *elusive* thematic-materials effect": most thematic contents did not help, but the drinking-age rule did | Supported, but not the canonical source. Add Johnson-Laird, Legrenzi & Legrenzi (1972) or Cheng & Holyoak (1985). |
| Smith et al. 1988; Hampton 1987 (l.476) | | Accurate. |
| Hampton & Jönsson 2012 (l.499) | | Plausible as a defender of attribute inheritance; I could not check the chapter text. |
| Kamp & Partee 1995 (l.499) | | **Misattributed (A1).** |
| Westerståhl 1998 (l.57) | | **Misattributed (A2).** |

---

## D. Status of round-2 points

| r2 item | Status |
|---|---|
| N1 interchange intervention / F&M | **Fixed.** Text at l.144, l.159 and l.236 is scoped to an interventionist reading. The requested controls were added: role-2, range(B₁) random, and natural direction. The finding that range(B₁)-random spills at 0.75 is a genuinely informative addition. |
| N2 freshness | **Mostly fixed** (l.94, abstract, l.59). Open: l.129, l.278 (B2) and l.259 (B6). |
| N3 Wisniewski | Fixed. |
| N4 Wason | Fixed (see C for a canonical-source suggestion). |
| N5 Gobet & Simon / random positions | Fixed. |
| N6 superposition scope | Fixed at l.194 and in the abstract. Open in the Fig. 2 caption (B3) and App. B (B4). |
| N7 "new sentences" | Fixed. |
| N8 tokening attribution; Smolensky 1987/88 | Fixed. |
| N9 Vegner | Fixed and verified. |
| N10 triviality citation | Done, but my suggested Westerståhl citation was wrong (A2). |
| N11 Hadley label | Fixed. |
| N12 Table 1 labels / Montero | Fixed. |
| N13 depth wording | Fixed at l.254. l.280 now miscounts levels (B7.2). |
| N14 pet fish / Table 5 row | Added and the row was split. Open: A1 and B1. |
| N15 abstract length; People paragraph; paradigms | People paragraph and paradigms fixed. Abstract longer (B11). |
| W1(b) post-1990 literature (Smolensky 1995, Chalmers 1990, Hadley 1997, McLaughlin 2009, Calvo & Symons 2014) | **Still open.** Non-blocking now that F&M is correctly scoped. One sentence citing Chalmers 1990 (structure-sensitive operations on holistic, non-concatenative representations) belongs at l.59 next to van Gelder, since that is exactly where the criterion and van Gelder diverge. |
| W3 Hummel & Holyoak (`hummel1997`, in the bib, uncited) | **Still open.** DORA is cited (l.290); LISA's role–filler binding by synchrony is the natural human-model precedent for "role-factorized". |
| Reverse compositionality: F&L concerns possession/understanding, not read-back by a fixed procedure | Partly addressed through l.94's use of `fodor2002`. Adding one clause at l.59 would complete it. |
| r2 minor 3: context-sensitive contributions (*red* hair vs *red* apple) | Open. This is now more relevant given B1, since it is the same phenomenon as selective modification. One sentence at l.114 would do. |
| r2 minor 4: which reference operator | Mostly fixed in practice (l.216, l.261). Def. 2 could name concatenation, projected to the system's width, as the default. |
| r2 minor 5: Prop. 7 mapped to Geirhos / Kamb & Ganguli (Table 6) | Open, minor. |
| r2 minor 7: "as role-factorization predicts" for 0.94 vs 0.88–0.93 (l.220) | Open, minor. Write "consistent with". |
| r2 minor 8: seed spread | Fixed at l.220. Reappears at l.286 (B5). |
| r2 minor 13: "not stable under any noise" | Fixed (phrase removed). |

---

## E. Style relative to Elmoznino et al. (2026)

The structure matches the reference paper closely:
- epigraph;
- an Intuition section with a running example;
- boxed definitions followed by a plain-language gloss;
- a "what the definition does not require" note box;
- related work folded into "Relationship to other accounts";
- uniform Composites / Readouts / Results run-ins;
- a Discussion with requirements, terminology, and limitations.

Claims are measured almost everywhere, and the graded framing (κ, δ, tolerance τ) is now explicit.

The remaining stylistic gaps are:
- the abstract's length (B11);
- the overloaded Definition 2 (B13);
- a few sentences in Sec. 4.2 that run past 60 words. For example, l.216 "A readout fitted on the network's own training composites ... (Figure 4d)" should be split after "behavior".

The prose is otherwise clear and active, and the hedges are placed where the evidence ends.

## Summary of required changes

**Required:**
1. **A1, A2:** fix the Kamp & Partee and Westerståhl misattributions.
2. **B1:** state the pet-fish recasting as the paper's own graded position, and soften "role-factorized" for default inheritance.
3. **B2, B3, B6:** remove the leftover "never seen before" wording, qualify "only" in the taxonomy caption, and repair the "second sense" cross-reference.
4. **B4, B5, B7:** align App. B and the Discussion with what was measured.

**Optional:** B8–B13, W1(b), and Hummel & Holyoak.

With items 1–4 done, I recommend acceptance.
