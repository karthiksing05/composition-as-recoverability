# Review R1-B: "Defining Compositional Representations by Recoverability"

Reviewer B: cognitive science / philosophy of mind (classical systematicity debate, concept combination, human compositional learning, neural codes).

I read the full source (main.tex, including all appendices) and checked the rendered pages.

## 1. Summary

The paper proposes a criterion for *representational* compositionality. A relation r, realized in a system as a state map g_r, is (i) epsilon-recoverable if a readout from a declared class U, fitted on composites of a finite vocabulary, returns each part's code in its role for composites of **fresh** parts drawn from the same part distribution; and (ii) delta-role-factorized if the state is affine in the part codes, with one matrix per role shared by all parts. A system is compositional relative to U and a class M of part-finding methods when it does no worse than a reference role-filler operator applied to the same part codes. The formal results are elementary but clean:
- unrestricted separability is vacuous (Lemma 1);
- lookup decodes any relation that is injective on a vocabulary (Lemma 2);
- for Gaussian parts, the fresh-part R^2 ceiling of a linear relation is rank A / kd (Prop. 3);
- readouts are determined only on the span of the training parts (Prop. 4);
- under exact role-factorization, role reversals and new role assignments are affine functions of one composite (Prop. 6, presented as the answer to Fodor & McLaughlin).

The experiments cover four settings:
- 14 analytic operators on 16-d Gaussians, which match the ceilings; HRR/MAP behave like half-rank projections;
- small MLP/CNN/transformer encoder-decoders on 8x8 random-generator "glyphs". Held-out pairs are near ceiling while fresh-glyph transfer depends on state width and vocabulary size; state recovery correlates r = 0.94 with fresh-part transfer;
- a RAAM-style recursive composer that fails at depth 3;
- six pretrained text encoders probed with ridge readouts to 32-PC "part codes" of "a x". All of them fail on fresh nouns, and CLIP fails worst.

The core move, separating "decodable over a vocabulary" from "invertible over the part space", is useful. It is a crisp way to state something VSA and neuroscience practitioners know informally. My concerns are mostly about how the paper positions this criterion in the cognitive-science and philosophical literature, and about what it claims follows from it.

## 2. Scores

- Soundness: 2/4. The math is correct but narrow. The philosophical claims drawn from it (necessity, the answer to F&M) and some of the empirical claims (pretrained "new sentences"; human evidence) do not follow as stated.
- Presentation: 3/4. Clear writing and a good running example. Some key terms ("fresh", "reverse compositionality", "representational compositionality" attributed to Vegner et al.) are used loosely.
- Contribution: 2/4. The fresh-part criterion is a useful methodological sharpening. As a *definition of compositionality* it is underargued and partly conflicts with the human case it invokes.
- Overall: 5/10 (borderline; a good workshop/short paper in current form, and potentially a solid Cognitive Science or CogSci-proceedings contribution after revision).
- Confidence: 4/5.

## 3. Major weaknesses

### W1. "Necessity from the relation" does not answer Fodor & McLaughlin; it restates Smolensky's reply, which they had already rebutted (Sec. 2.4 "Under role-factorization, systematicity is necessary"; Sec. 2.5 "The classical debate"; Prop. 6; abstract)
F&M (1990) is a reply *to Smolensky's tensor-product proposal*. Smolensky's own defence was that tensor-product states are role-factorized and exactly unbindable. Prop. 6 is a linear-algebra restatement of that defence. F&M's objection was not that constituents cannot be recovered. It was twofold:
- (a) Tensor-product constituents are not tokened when the complex is tokened, so they cannot be the causes of structure-sensitive processing.
- (b) Nothing in connectionist architecture *requires* that a network represent via such a relation rather than via one that is not systematic. Systematicity is therefore at best compatible with the architecture, and not explained by it.

Prop. 6 shows that an affine map to the permuted state *exists*. It does not show that the system can compute it, has it, or must use it. Existence of a map is not a capacity, and F&P's systematicity is a claim about capacities ("the ability to think aRb is intrinsically connected to the ability to think bRa"). On (b), the paper concedes the point in the Discussion ("whether it learns that relation remains contingent on training, as the seed spread at small vocabularies shows"). That is precisely the F&P/F&M explanatory challenge, restated in Aizawa (2003) and in Goodale & Mascarenhas (2026), which the paper itself cites. The abstract and Sec. 2.4 nevertheless say systematicity "follows from the relation instead of from the training data", and Sec. 2.5 says the Proposition "answers" F&M.

**Fix:**
- Reframe Prop. 6 as a *conditional*: given a state that meets the criterion, systematic recombination is available to any affine downstream process. Say explicitly that this is Smolensky's (1990, 1995) position.
- Engage F&M's causal-efficacy argument directly. Either argue that affine availability is enough for causal efficacy, as Smolensky and Chalmers (1990) did, and say why, or concede that the explanatory question stays open.
- Drop "necessity ... that Fodor & McLaughlin demanded" from the abstract and Sec. 2.4.
- Cite the post-1990 literature: Smolensky (1995), Chalmers (1990/1993), Hadley (1994, 1997), McLaughlin (2009, "Systematicity redux"), and the Calvo & Symons (2014) volume.

### W2. The fresh-part criterion is not clearly a criterion for *human* compositionality, and "fresh part" is undefined for concepts (Sec. 2.1, Def. 1, Sec. 5 "People", App. C)
For glyphs and Gaussian vectors, "a further draw from D" is well defined. For concepts it is not. On the classical view the paper aligns with, concepts are atoms. A concept the thinker does not possess cannot be a constituent of a thought the thinker entertains. That is exactly what reverse compositionality (Fodor & Lepore) asserts. So the human analogue of the paper's test is not obvious. Candidates include:
- novel *perceptual* parts in a scene, as in object-based scene parsing;
- novel *word forms* in syntactic frames (Jabberwocky; syntactic bootstrapping: Gleitman 1990, Naigles 1990), where people recover the role but not the "content code";
- concepts learned one-shot and then recombined (Lake et al. 2019).

Lake et al. 2019 is actually a *vocabulary* case in the paper's own terms. Participants were taught "dax" etc. before composing, so the new word is in the vocabulary that fixed their decoding procedure. Moreover, for continuous parts the criterion is essentially about interpolation over a part manifold: Prop. 4 shows that fresh-part recovery requires the training parts to span R^d. That is a property of the part encoder's generalization, and arguably not of compositional structure. A classical symbol system meets the criterion only because concatenation keeps a fresh token intact. A classical system cannot produce a "code" for a symbol outside its lexicon at all.

**Fix:**
- Add a subsection that says what a fresh part is for (i) perception, (ii) language/word learning, and (iii) concepts, and which human paradigms test each.
- Separate "the relation is invertible on the part space" (a property of g_r) from "the part encoder generalizes" (a property of e). At present the network experiments conflate them: readouts target pixels, so a failure of the encoder on new glyphs is scored as a failure of composition. One concrete experiment would freeze a part encoder known to generalize and train only the composer.

### W3. The human evidence is described selectively, and several citations do not support recoverability *of fresh parts* by the paper's own standard (Sec. 5 "People"; Table 6)
- **Frankland & Greene (2015).** They decode agent/patient over a small, closed set of nouns. By the paper's own Lemma 2 and Prop. 4, this is vocabulary-level decoding. It shows role-separated codes (and generalization across verbs), but not fresh-part recovery. The same holds for cross-condition generalization (Bernardi et al. 2020), which generalizes across familiar conditions.
- **Chase & Simon (1973)** is listed as "parts kept within chunks ... recoverable parts". The classic finding cuts the other way. Expert recall advantage largely vanishes for random positions (only a small residual advantage; Gobet & Simon 1996). Recovery depends on familiar chunks, which is the paper's "vocabulary" case.
- **Lake et al. (2019).** See W2: the new primitive was trained before composition.
- **Missing evidence that bears directly on fresh-conjunction recovery:**
  - Treisman & Schmidt's (1982) illusory conjunctions, a human *role collision* under load;
  - visual working-memory binding limits (Luck & Vogel 1997; Wheeler & Treisman 2002);
  - Jabberwocky/syntactic bootstrapping (role recovery for novel words);
  - Hummel & Biederman (1992) / Hummel & Holyoak (role-filler binding by synchrony).

  These suggest that humans recover parts of fresh composites in a *capacity-limited*, graded way, which looks more like superposition than like the paper's dichotomy.
- **Wisniewski (1997)** is summarized as "concepts are often blended". Wisniewski's taxonomy has three interpretation types: relation-linking ("robin snake" = snake that eats robins), property mapping, and hybridization. Relation-linking is common, and it is role-filler composition in the paper's sense. Only property mapping and hybridization are blend-like.

**Fix:** Rewrite Sec. 5 "People" and Table 6 so each row states whether the evidence concerns familiar or fresh parts. Correct the Chase & Simon and Wisniewski characterizations. Add the binding-limit and illusory-conjunction literature, and discuss whether it favours a graded criterion over a categorical one.

### W4. The treatment of concept combination maps psychological models onto Gaussian-vector ceilings without justification (Sec. 2.2 paragraph after Def. 1; Table 5)
Table 5 assigns Smith et al.'s (1988) selective modification model the exemplar "x + αy" and the ceiling 1/(1+α^2). Selective modification operates on attribute-value frames. The adjective shifts votes and diagnosticity on *its own attribute slot* of the head's frame. That is closer to a slot-filling (role-filler) operation than to vector addition, and the modifier is recoverable from which slot changed. Hampton's composite prototypes (non-additive attribute inheritance, emergent attributes) are a fair example of a failure of role-factorization. Even there, "recoverable for fresh parts: No; ceiling 1/(1+α^2)" is not derived from Hampton's model.

The table also omits the central philosophical debate the paper's framework speaks to: Fodor & Lepore's pet-fish argument that prototypes do not compose, and Osherson & Smith (1981). Both are in references.bib but are never cited in the text. Prinz (2002) and Robbins (2002) replied that prototypes compose by default with overrides, and that reply also needs addressing. The paper's framework could say something new here: prototype combination may be role-factorized up to an interaction residual delta, with emergent features as eta.

**Fix:** Remove the Gaussian ceilings from the psychological-model rows, or derive them from an explicit formalization of each model. Add a paragraph that places the pet-fish debate in the delta/eta vocabulary. That would be a genuine contribution to the concept-combination literature.

### W5. The citations and framing in the philosophical lineage are inaccurate (Sec. 1, para. 3; Sec. 2.5)
- **Robbins (2005)** is "The Myth of Reverse Compositionality". It argues *against* reverse compositionality. Reverse compositionality is due to Fodor & Lepore (2001; also Fodor 1998). It is also a thesis about *concept possession and understanding* (understanding "red apple" requires having RED and APPLE). It is not a thesis about read-back by a fixed procedure, let alone of fresh parts. If anything, it is in tension with the fresh-part demand (see W2).
- **van Gelder (1990).** The identification with functional compositionality is fair as far as it goes: general, effective, reliable procedures for both composing and decomposing. But it omits van Gelder's main point, that non-concatenative representations can support structure-sensitive processing *holistically, without decomposition* (cf. Chalmers 1990 on RAAM active-to-passive). Under the paper's definition, a system that transforms structures holistically but whose fresh parts cannot be read back by affine maps is non-compositional. A fair treatment would state this divergence. Plate (1995, 2003) also offered HRR as a functionally compositional scheme. The paper's verdict that HRR is a "lossy blend" therefore directly contradicts the van Gelder/Plate position, and that disagreement should be argued, not assumed.
- **Vegner et al. (2025)** is about behavioural vs. representational *systematicity*. The paper attributes a behavioural/representational *compositionality* distinction to them and says they "left its criterion to future work". Please check that against the paper and adjust.
- **Hadley's strong systematicity** is invoked (Sec. 4.2) without citation, although hadley1994 is in the bib. Hadley's strong systematicity also requires generalization to novel positions *in embedded clauses*, not just to an unseen role in a flat 2-place relation. Call the manipulation "a novel-role split in the spirit of Hadley".

**Fix:** Correct each citation, cite Fodor & Lepore for reverse compositionality, and add an explicit paragraph on the holistic-processing branch of the debate.

### W6. The claims about "neural networks", "pretrained encoders" and "new sentences" go beyond the evidence (title, abstract, Sec. 4.4, Sec. 5)
- **The networks are tiny and the parts are unlike concepts.** The trained systems have 8-d latents, an 8x8 pixel random generator, 2-layer transformers and 4,000 steps. Parts are i.i.d. isotropic Gaussians. The rank ceilings are worst-case results for isotropic Gaussians. With structured or sparse part distributions, which is what concept and word codes look like, superposition can be recoverable for fresh parts, as the paper concedes for Elhage et al. The negative verdict on VSA/superposition is therefore distribution-relative, but it is stated categorically in the abstract ("behaves like a lossy blend") and in Fig. 2.
- **In the pretrained experiment, "fresh" nouns are not fresh to the system.** GPT-2, BERT and CLIP have seen all 305 nouns. They are fresh only to the ridge readout. Definition 1 and the intuition in Sec. 2 require parts "the readout and the system have never seen". So the abstract's "recover the nouns of familiar sentences but not of new ones" mislabels the manipulation. The failure (R^2 = -1.21 for CLIP) is far more plausibly a mismatch between the declared part code (the isolated "a x" state, 32 PCs of 170 nouns) and the noun's contextual contribution, or ridge extrapolation. It need not be a property of the relation. The paper's own limitations paragraph says as much, but the abstract and Sec. 4.4 conclusion ("no encoder we tested meets the criterion") do not.
- **The title promises more than the paper delivers.** "Defining Compositional Representations" promises a general definition, but the definition is relative to a declared U, M and D. It is calibrated against a "reference role-filler operator", which is defined as compositional, so the criterion is circular at its base.

**Fix:**
- Scope the abstract and title, e.g. "A fresh-part recoverability criterion for compositional representations".
- Re-label the pretrained result as "nouns not used to fit the readout", and add a contextual-token part code (the noun's own token state in context), which removes the most likely confound.
- Add one structured part distribution (sparse or low-dimensional manifold) to show how the superposition verdict changes.
- State the reference-operator dependence as a feature, not as a definition of compositionality simpliciter.

## 4. Minor issues

1. **Abstract.** "People ... can take the meaning apart again into the same parts" is an empirical claim stated as fact, with no citation. It is contested for idioms, compounds and blends.
2. **Sec. 1, para. 2.** The claim that the colloquial principle "cannot distinguish an avocado placed on a chair from an armchair in the shape of an avocado" is too strong. Both are compositional under Frege/Montague if "avocado chair" has a compositional (if context-dependent) semantics. The point is about *representational* form, not meaning determination.
3. **Sec. 2.2.** Role-factorization is equated with "the classical requirement that a constituent contribute the same thing to every complex", citing F&P 1988. That is F&P's context-independence requirement. Note that it is about the *same symbol type*, not an affine code, and that compositional semantics tolerates context-sensitive contributions ("red" in red hair vs red apple). The affine restriction is a modelling choice and should be flagged as such.
4. **Def. 2.** The comparison "no worse than a reference role-filler operator" makes the definition depend on which reference is chosen (concatenation vs tensor product with which roles). Specify which.
5. **Sec. 2.4.** "eight results; we state the four that carry the argument" is fine, but Prop. 7 (content-context dissociation) is used in Table 4 for claims about Geirhos et al. and Kamb & Ganguli that it does not support. Prop. 7 concerns additive fits under a connected design, not shortcut learning.
6. **Table 2 (2x2).** The cell "Blending that fails" is uninformative. Also, the "Imitation" label imports a Searle/Chinese-room tone without argument. "Vocabulary-bound generalization" is more neutral.
7. **Sec. 4.2.** "The role-filler decoder ... is the best on parts in new roles at the smallest vocabulary (0.32-0.33 vs 0.11-0.27)". These are all low numbers, and the advantage vanishes by V = 16. The Discussion's third requirement ("made new role assignments most reliable") overstates this.
8. **Sec. 4.2.** The "seed spread" in Fig. 4f is interpreted as "the signature of systematicity that is contingent on training (McCoy et al. 2020)". It is equally the signature of an under-determined problem at small V. The proposed reading needs a control.
9. **Sec. 4.3.** The RAAM depth-3 failure is attributed to the relation not being invertible on its own outputs. But training used only depths 1-2, so this is a train/test distribution shift. Train at depth 3 and test at depth 4 to see whether the failure is structural.
10. **Sec. 4.4.** GPT-2 is causal. Mean-pooled states for "a x on a y" give role-1 information to every token but role-2 information only to later tokens. Report the last-token readout separately.
11. **Sec. 5 "Future work".** Self-citation of TRELLIS and an "anonymous2026seen" citation in App. C appear together with a named author line. That is inconsistent with blind review, if the paper is intended for one.
12. **App. A, Prop. 4.** "Requires at least d + 1 distinct parts in role i". The proof sketch assumes a connected design; state that assumption in the proposition.
13. **App. A, Remark after Prop. 2.** "No continuous map from R^{kd} to R^m with m < kd is injective" (invariance of domain) is correct. But the extension to "not stable under any noise" for discontinuous injections needs a precise statement or a citation.
14. **Fig. 1.** It uses Canva-generated images. Consider a schematic instead, for archival stability and licensing.
15. **Missing citations:**
    - Hummel & Holyoak (1997, LISA), and Doumas & Hummel on role-filler binding and relational generalization;
    - Marcus (2001), *The Algebraic Mind*, which argues for "free generalization of universally quantified one-to-one mappings" to novel items. This is essentially the fresh-part criterion and should be cited prominently;
    - Pylyshyn & Fodor's productivity argument;
    - Szabo's SEP entry on compositionality, for the distinctions between the senses of the principle;
    - Piantadosi et al. (2016, LoT learning) is cited only in the appendix table.

## 5. What would move my score up by 2 points

1. Recast the F&M section honestly. State that Prop. 6 formalizes Smolensky's reply and concerns availability, not capacity or causal efficacy. Remove "necessity" claims from the abstract and Sec. 2.4. Engage the post-1990 systematicity literature and the holistic-processing branch (van Gelder, Chalmers).
2. Add a principled account of what a fresh part is for perception, words and concepts, and identify the human paradigms that test fresh-part recovery: Jabberwocky/syntactic bootstrapping, illusory conjunctions, binding limits. Correct the Chase & Simon, Wisniewski, Robbins and Frankland & Greene characterizations. Ideally, argue whether the human profile favours a graded criterion.
3. Separate encoder generalization from relation invertibility experimentally, with a frozen part encoder plus a learned composer. Repeat the pretrained probe with contextual part codes and relabel "fresh" as "readout-held-out".
4. Rescope the title and abstract to the actual claim: a fresh-part recoverability criterion relative to declared readouts and part distributions. Add one structured, non-Gaussian part distribution so the verdict on superposition is not an artefact of isotropy.

With these changes, I would rate the paper 7/10.
