# Literature and novelty audit: "Distinguishing Representational and Behavioral Compositionality via Recoverability"

Audit date: 2026-10-27. Sources: `recoverability-theory/main.tex` and `references.bib`, both read in full. Every external claim below was checked on the web (arXiv, ACL Anthology, PMLR, NeurIPS proceedings, PMC). Where I could check only an abstract, the entry says so. Items marked **[unverified]** are leads I could not confirm from a primary source.

---

## 0. Bottom line

1. **The definition (separable + invertible) is not new as mathematics.** Wiedemer et al. (ICLR 2024) prove compositional generalization from an *additive* slot decoder, f(z) = sum_k phi_k(z_k), which is the paper's separability with psi = id. They also require *compositional consistency*, meaning the encoder inverts the decoder on recombined slots, which is the paper's invertibility. Lachapelle et al. (NeurIPS 2023) prove identifiability and "Cartesian-product extrapolation" from additive decoders. Wiedemer 2024 also states that additivity fails under occlusion. That is exactly the paper's BESIDE vs PART-OF distinction, and Wiedemer excludes occluding scenes for this reason. The paper cites Wiedemer 2024 only once, as a system to evaluate. As written, it presents separability + invertibility as its own definition without saying that this pair of conditions is the established sufficient condition in the identifiability literature. **This is the largest positioning gap.**
   - Novel elements that survive: (a) using separability + invertibility as a *definition* of representational compositionality, with a link to the Fodor/Pylyshyn and van Gelder debate (identifiability work uses them as *sufficient conditions for generalization*); (b) relativizing the definition to a declared discovery class M; (c) the taxonomy of modes of combination drawn from psychology; (d) the familiar-vs-freshly-drawn-parts calibration that separates relation-borne from vocabulary-borne recoverability.
2. **Most formal results have direct precedents.** Prop. 3 (2V-1 connected pairs determine all V^2 composites) is the classical estimability result for a two-way additive model under a connected design. Schug et al. (ICLR 2024) use the same "compositional support + connected support → linear number of tasks suffices" logic for modular hypernetworks. Prop. 2 is rank-nullity. The finite-vocabulary remark is the HRR/VSA capacity and clean-up-memory story, and also compressed sensing. Prop. 6 (content-context dissociation) is the additive special case of Lippl & Stachenfeld's (ICLR 2025) conjunction-wise additivity, and their "memorization leak" is the paper's "context leakage". Uselis, Dittadi & Oh (arXiv 2602.24264, 2026; ICML 2026 oral per the conference listing) prove that compositional generalization *requires* linear, orthogonal per-concept factors. That is a necessity result close to the paper's "systematicity is necessary only under separable + invertible relations".
3. **The test suite overlaps with existing work** on the CCGP/decodability side (Bernardi 2020; Ito 2022), on decodability predicting generalization (Redhardt, Akram & Schug, NeurIPS 2025; Uselis 2026; Xu et al. 2022; Montero 2021), on interchange interventions (Geiger et al., DAS), and on seed variance (McCoy, Min & Linzen 2020). None of these is cited. The calibration with freshly drawn parts and the human reference profile look new.
4. **Citation errors:** the Goodale first name is wrong (Michael, not Matthew). The Schapiro 2025 claim is wrong: the paper locates the blending failure at *abstraction* (finding a generic space), not at mapping. The Liu 2022 "inside another region" failure is not in Liu et al.: Liu reports failures when objects are in the image centre and confusion of attributes. The "same model" contradiction pairs Liu (Composable Diffusion) with Conwell (DALL-E 2), which are different models. The Conwell wording is slightly off. elmoznino2024 should cite its ICML 2025 version. No entry looks fabricated: every flagged arXiv/TACL entry resolves.
5. **The paper's claim that neural networks "mostly blend" conflicts with literature that is in references.bib but never cited.** Feng & Steinhardt (binding IDs), McCoy et al. 2019 (RNNs implicitly implement TPRs), Lepori et al. 2023 (structural compositionality), and Greff et al. 2020 (the binding problem) all report or discuss role-filler-like, recoverable structure in trained networks. A reviewer will notice that these are in the .bib and uncited.

---

## 1. Domain map (brief)

| Thread | Core works | Relation to this paper |
|---|---|---|
| Classical debate | Fodor & Pylyshyn 1988; Smolensky 1987/88/90; van Gelder 1990; Fodor & McLaughlin 1990; Hadley 1994; Aizawa 2003; **reverse compositionality** (Fodor & Lepore 2001/2002; Robbins 2005; Johnson 2006) | The paper covers the first group well. **Reverse compositionality** (understanding a complex requires understanding, i.e. recovering, its parts) is the philosophical precursor of "recoverability" and is missing. Goodale & Mascarenhas 2026 and Woydt et al. 2025 are the current restatements. |
| VSA / TPR / HRR | Smolensky 1990; Plate 1995/2003; Kanerva 2009; Frady, Kleyko & Sommer 2018; Thomas, Dasgupta & Rosing 2021 (JAIR); Kleyko et al. 2022 survey; Soft TPR (Sun et al. 2024, arXiv 2412.04671) | They supply the capacity theory behind the remark on Prop. 2 (recovery from a finite vocabulary degrades as V grows relative to m; clean-up memory). The paper cites only Smolensky and Plate 1995. |
| TPR discovery in trained nets | McCoy et al. 2019 (TPDN); Soulos et al. 2020 (ROLE, BlackboxNLP pp. 238-254) | These are direct implementations of "a decomposition found under M makes r separable and invertible". **Must cite.** |
| Compositional-generalization benchmarks | SCAN (Lake & Baroni 2018); COGS (Kim & Linzen 2020); CFQ (Keysers 2020); SLOG (Li 2023); gSCAN; Hupkes 2020; Hupkes et al. 2023 (GenBench); T2I-CompBench | Cited except COGS and GenBench. |
| Identifiability / additive-decoder theory | Brady et al. 2023 (ICML); Lachapelle et al. 2023 (NeurIPS oral); Wiedemer et al. 2023 (arXiv 2307.05596); Wiedemer et al. 2024 (ICLR oral); Dong & Ma 2023 (ICLR; arXiv 2211.11719); Schug et al. 2024 (ICLR); Lippl & Stachenfeld 2025 (ICLR); Uselis et al. 2026; Redhardt et al. 2025 (NeurIPS) | **The closest formal prior art.** Only Wiedemer 2024 is cited, and only in passing. |
| Additive conjoint measurement | Debreu 1960; Luce & Tukey 1964; Krantz et al. 1971 | The paper's definition of "separable", psi(g(x,y)) = phi1(x) + phi2(y) with psi invertible, *is* an additive conjoint representation. Existence is characterized by cancellation axioms (Thomsen / double cancellation). This would give a testable, necessary-and-sufficient condition for separability. |
| Neuroscience of factorized/abstract codes | Bernardi et al. 2020 (CCGP, parallelism score); Ito et al. 2022; Johnston & Fusi 2023; Courellis et al. 2024; Frankland & Greene 2015/2020; Schwartenbeck 2023; Whittington 2020 | CCGP is "a readout trained on some conditions recovers the variable on held-out conditions", which is the paper's Test 2/3 metric under another name. Not cited. |
| Binding in LMs | Feng & Steinhardt 2024 (binding IDs); Dai et al. 2024 (ordering ID); Gur-Arieh et al. 2025 ("Mixing Mechanisms", arXiv 2510.06182); Prakash et al. (lookback) | Binding IDs are additive, role-indexed vectors from which fillers are retrieved. This is empirical evidence of a separable, roughly invertible relation in transformers. It cuts against the "networks mostly blend" framing. |
| Causal abstraction / interventions | Geiger et al. 2021 (NeurIPS); Geiger et al. 2024 DAS (CLeaR, PMLR 236:160-187); Geiger et al. 2025 (JMLR theory); **Sutter et al. 2025 (NeurIPS) "non-linear representation dilemma"** | Exp. 3 uses "interchange interventions" with no citation. Sutter et al. prove that with unrestricted nonlinear alignment maps, any network can be mapped to any algorithm. This is the exact formal reason the paper needs a declared class M. Citing it would strengthen Section 3.3's second feature and the reply to the unfalsifiability objection. |
| Emergent-language metrics | Brighton & Kirby 2006 (topographic similarity); Andreas 2019 (TRE); Chaabouni et al. 2020 (posdis/bosdis; generalization does not require compositional languages); Kharitonov & Baroni 2020; Ren et al. 2020 (in .bib, uncited); Xu et al. 2022 | Chaabouni 2020 is direct empirical support for the "imitation" cell of Table 1 (generalization without compositional representation). |
| Disentanglement → generalization | Montero et al. 2021 (ICLR); Schott et al. 2022 (ICLR); DCI (Eastwood & Williams 2018) | DCI "informativeness" is a decodability score for each factor, i.e. an R1-type measure. Montero shows that decodable, disentangled factors do not ensure combinatorial generalization by the decoder. That bears directly on the Table 1 cell "parts recoverable, behavior fails" (failure of performance). |
| Compression-based definitions | Elmoznino et al. 2025 (ICML; arXiv 2410.14817); Elmoznino et al. 2026 (arXiv 2610.20922); Gauderis et al. 2026 "From Mechanistic to Compositional Interpretability" (arXiv 2605.08934, MDL + commuting syntax/semantics maps) | The paper contrasts itself with Elmoznino. Gauderis et al. is a new, close competitor in the interpretability framing (not cited). |
| Diffusion composition mechanics | Liu 2022; Okawa 2023; Park et al. 2024 (NeurIPS spotlight, "hidden capabilities"); Kamb & Ganguli 2025 | Park 2024 separates *capability* (shown by latent intervention) from *behavior* (shown by prompting). That is an empirical version of the behavioral/representational split. |

---

## 2. Novelty threats, by contribution

### Contribution 1: the recoverability definition (separable + invertible, relative to M)

**Closest prior work.**

- **Wiedemer, Brady, Panfilov, Juhos, Bethge, Brendel, "Provable Compositional Generalization for Object-Centric Learning", ICLR 2024 (oral), arXiv 2310.05327.** Definition 5 calls a decoder *additive* if it renders each slot through slot functions phi_k and sums the results ("rendering each slot to an intermediate image via slot functions, then summing these images"). Theorem 3 assumes: (i) the generator is a compositional, irreducible diffeomorphism; (ii) the slot-supported training set is convex; (iii) the decoder is additive; (iv) the decoder is compositional on the encoded training data; (v) L_rec + lambda * L_cons = 0, where L_cons is *compositional consistency*: the encoder must invert the decoder on novel slot recombinations. The conclusion is that the autoencoder "generalizes compositionally". On occlusion, the paper says: "we exclude images with occluding objects"; softmax masking "facilitates modeling occluding objects but violates additivity"; and additivity and compositionality "do not allow slots to interact during rendering and thus cannot adequately model general multi-object scenes".
  - *Overlap:* separable ≈ additive decoder; invertible ≈ compositional consistency plus an injective generator; the BESIDE/PART-OF prediction ≈ their occlusion caveat.
  - *New in this paper:* using the conditions as a *definition* and not as sufficient conditions; a general psi (theirs is psi = id); roles/arity and nesting (R4); relativization to M; the connection to the debate over necessity vs contingency.
- **Lachapelle, Mahajan, Mitliagkas, Lacoste-Julien, "Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation", NeurIPS 2023 (oral).** Additive decoders identify latent blocks up to permutation and block-wise invertible maps, and they generate novel factor combinations ("Cartesian-product extrapolation"). *Overlap:* separability → generalization to unseen combinations (the paper's Prop. 3/4). This is the same idea with stronger identifiability results.
- **Brady et al. 2023 (ICML), "Provably Learning Object-Centric Representations".** Slot identifiability from compositional + irreducible generators. Invertibility is measured by reconstruction and compositionality by a contrast that is zero exactly for compositional functions. (Found via search; abstract-level only.)
- **Wiedemer, Mayilvahanan, Bethge, Brendel 2023 (arXiv 2307.05596), "Compositional Generalization from First Principles".** Abstract: compositionality is "a property of the data-generating process rather than the data itself", with "mild conditions on only the support of the training distribution and the model architecture". *Overlap:* the paper's argument that "necessity comes from the relation" is the same move.
- **Uselis, Dittadi & Oh, arXiv 2602.24264 (Feb 2026; ICML 2026 oral per the conference listing), "Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models".** They formalize three desiderata (divisibility, transferability, stability) and argue that representations must decompose linearly into per-concept components that are orthogonal across concepts. *Overlap:* the strongest threat to the "necessary" framing. They derive additive and orthogonal structure (≈ separable + linearly invertible) as *necessary* for compositional generalization. This paper argues necessity in the other direction (separable + invertible ⇒ systematicity is necessary) and only contingency without it. The two should be reconciled explicitly.
- **van Gelder 1990** (cited) and **reverse compositionality** (Fodor & Lepore 2001/2002; Robbins 2005, *Phil. Studies* 125(2):251-275; Johnson 2006, *Erkenntnis* 64(1):37-60; not cited). Reverse compositionality, the thesis that each constituent of a complex expression contributes its whole content so that understanding the complex entails understanding the parts, is the classical form of "recoverability". It has a known critical literature (Robbins argues that no extant theory satisfies it; Johnson argues it is a processing fact). The paper should position R1 against it.
- **Additive conjoint measurement** (Debreu 1960; Luce & Tukey 1964, *J. Math. Psych.* 1:1-27; Krantz et al. 1971). psi(g(x,y)) = phi1(x) + phi2(y) for an invertible psi is precisely an additive conjoint representation. Its existence is characterized by double-cancellation and Thomsen conditions. This is an old, exact characterization of the paper's "separable relation", and it would give an assumption-light test of separability that needs no probe.
- **Elmoznino et al.** (ICML 2025 version; cited as the 2024 arXiv) and **Gauderis et al. 2026** (MDL + commuting maps): compression-based alternatives. The paper's contrast with Elmoznino (no recoverability requirement) is fair. Gauderis is new and should be added.

**What is genuinely new:** the definitional stance, separating competence (relation) from vocabulary-borne recoverability; the M-relativization as a falsifiability device; the taxonomy of modes of combination that maps psychological concept combination (Smith 1988; Hampton 1987; Wisniewski 1997; Fauconnier & Turner) onto separable/invertible properties. I found no 2025-2026 paper that *defines* representational compositionality as separable + invertible recoverability (extended search: closest are Redhardt 2025, Uselis 2026, Soft TPR 2024, Gauderis 2026, none with this definition).

### Contribution 2: formal results

| Result | Precedent | Assessment |
|---|---|---|
| P1 Role collisions | Trivial; standard in the TPR/HRR literature (commutative binding cannot encode order; Plate 1995 uses non-commutative/permuted binding for this reason) | Fine as a lemma; the second sentence ("no non-invertible relation satisfies R1") is true by definition. |
| P2 Non-identifiability of linear blends | Rank-nullity. The finite-vocabulary remark is the VSA capacity / clean-up-memory analysis (Plate 1995/2003; Frady, Kleyko & Sommer 2018; Thomas et al. 2021) and compressed sensing for sparse sums | The remark should cite capacity theory. That theory already gives the scaling the paper predicts informally ("degrades as V grows relative to m"). |
| P3 Substitutivity from 2V-1 connected pairs | (a) Classical estimability of a two-way additive (no-interaction) model: all cell means are estimable iff the design graph is connected (standard linear-models result, e.g. textbook treatments of connected designs). (b) **Schug et al. 2024, ICLR**: "compositional support" + "connected support" ("there exists a path between any two task families"), and "fitting a number of tasks linear in the number of modules suffices". (c) Lachapelle 2023 Cartesian-product extrapolation; Dong & Ma 2023 (additive models extrapolate when marginals are covered) | **Not new.** Present it as a restatement in this setting and cite (a)-(c). The 2V-1 count is the spanning-tree edge count of the complete bipartite graph K_{V,V}. Note also that the result is stated only for psi = id; with a general invertible psi it holds after applying psi, which needs psi known or identified. |
| P4 Systematicity as a necessary consequence | Corollary of P3 + invertibility. The necessity/contingency framing is Fodor & McLaughlin 1990 / Aizawa 2003. Goodale & Mascarenhas 2026 make the "by default" architectural argument (their wording: symbolic theories explain it "by default"; a connectionist net "does not guarantee by default the possibility of representing" a permuted sentence) | The formalization is a modest new contribution. Uselis 2026 is the converse-direction necessity result and must be discussed. |
| P5 Depth kernel (2^n - 1)d | Rank-nullity again. Known growth/noise trade-offs for TPR and HRR (Smolensky 1990; Plate 1995) | Fine. The text says "No representation of fixed size can invertibly encode trees of unbounded depth over continuous parts". That holds only for *continuous* encodings (discontinuous bijections R^{kd} → R^d exist). Add "continuous". |
| P6 Content-context dissociation | The additive special case of **Lippl & Stachenfeld 2025 (ICLR)**: kernel models with compositionally structured representations are "conjunction-wise additive" (Thm 4.2). When the full conjunction has nonzero salience, test predictions are distorted ("memorization leak"). | The paper's "context leakage" (Test 3) is their memorization leak. Cite it, and note that their theory says *when* leakage occurs for kernel/lazy networks, which sharpens the paper's predictions. |

### Contribution 3: test suite, human profile, evaluation plan

| Element | Closest prior | What is new |
|---|---|---|
| Readouts recovering parts on held-out combinations (Tests 1-3) | Bernardi et al. 2020 CCGP + parallelism score; Ito et al. 2022; Redhardt et al. 2025 (linear decodability of constituents tracks compositional generalization and correlates with text-to-image composition failures); Uselis 2026 (linear factorization in CLIP/SigLIP/DINO tracks generalization); Xu et al. 2022; Montero et al. 2021; Eastwood & Williams 2018 DCI | Testing on **freshly drawn parts** to isolate recoverability carried by the relation from recoverability supplied by the vocabulary. I found no direct precedent in compositionality probing; it is standard in VSA capacity tests. |
| Interchange interventions (Exp 3) | Geiger et al. 2021; DAS (Geiger et al. 2024); Feng & Steinhardt 2024 | Nothing new; must cite. Sutter et al. 2025 shows that the discovery class must be restricted, which is why the M-relativization is needed. |
| Seed variance as a contingency test (Test 5) | McCoy, Min & Linzen 2020 (BlackboxNLP), "BERTs of a feather do not generalize together" | New as an operationalization of necessity vs contingency; the empirical precedent should be cited. |
| Data exponent N*(V) ∝ V^beta | Schug 2024 (linear in modules); Redhardt 2025 (scaling); Okawa 2023 (multiplicative emergence); Park 2024 | The reference beta ≈ 1 matches Schug's linear bound; cite it. |
| Synthetic glyph domain, BESIDE vs PART-OF | Wiedemer 2024 sprites (no occlusion); Okawa 2023 concept graph; Lewis et al. 2024, "Does CLIP bind concepts?"; dSprites/3DShapes held-out combinations (Montero; Schott 2022) | Contrasting disjoint and overlapping roles as a *designed* manipulation is new. Wiedemer 2024 predicts the result, so frame it as a test of that prediction. |
| Human reference profile | Frankland & Greene 2015/2020; Lake et al. 2019; Schwartenbeck 2023; Bernardi 2020; Courellis et al. 2024 | Assembling these into a calibration profile is new. |

---

## 3. Citation verification

### Flagged entries

| Key | Status | Notes |
|---|---|---|
| vegner2025 | **OK** | ACL 2025 long, pp. 31842-31856, DOI 10.18653/v1/2025.acl-long.1537; authors match. The paper's characterization (Hadley taxonomy, mechanistic interpretability as the route) matches the abstract. |
| woydt2025 | **OK** | arXiv 2506.01820; authors match; v3 revised Jan 2026. Characterization ("meta-learning ... only ... under a very narrow and restricted definition of a meta-learning setup") is supported. |
| goodale2026 | **ERROR: author first name** | Author is **Michael** Goodale, not "Matthew". TACL 14:1619-1632; arXiv 2606.14512 correct; add DOI 10.1162/tacl.a.743. The characterization is acceptable, but they say "by default", not "necessarily". |
| elmoznino2026 | **OK** | arXiv 2610.20922, submitted 14 Oct 2026, authors match. The paper's description ("information a whole adds beyond its parts") matches the abstract (parts "reveal shared structure that is missed when each part is modeled separately"). |
| elmoznino2024 | **Update** | Published as "Towards a Formal Theory of Representational Compositionality", ICML 2025, PMLR 267:15266-15295, same four authors. Cite the archival version; the arXiv title differs. |
| schapiro2025 | **Entry OK; claim WRONG** | arXiv 2509.21043, 14 authors match. The text says "Language models mostly fail at the mapping step of conceptual blending". The paper locates the failure at **abstraction**, not mapping: LLMs "struggle to find a shared abstraction between inputs"; "conditioned on finding a valid generic space, models can describe a coherent blended concept"; the main bottleneck "was abstraction ability". Rewrite as: "language models find blends hardest of the combinatorial tasks, failing mainly to find a shared abstraction (generic space) between inputs". |
| anonymous2026seen | **Cannot verify** (OpenReview blocked by a browser check) | This is the author's own ICLR 2027 submission. Two risks: (1) citing your own anonymous under-review paper by OpenReview forum ID in a non-anonymous paper can deanonymize the ICLR review (check the ICLR 2027 dual-submission and anonymity policy); (2) it is the *only* support for the "one object inside another" failure (see Liu below). |
| singaravadivelan2026 | **OK** | arXiv 2609.30414, Singaravadivelan & Langley, "A Unified Account of Concepts and Chunks" (v1 24 Sep 2026, v2 3 Oct 2026). The description matches (Trellis records composition and behavior). The red `\note{}` placeholder for TRELLIS results is still in main.tex (line 563). |
| schwartenbeck2023 | **OK** | Cell 186(22):4885-4897.e14, DOI 10.1016/j.cell.2023.09.004; 9 authors match. Table 2 claim ("novel objects represented and inferred as compositions of known elements") is consistent (fMRI relational/compositional codes; MEG replay of candidate arrangements). |
| kamb2025 | **OK** | ICML 2025, PMLR 267:28795-28831. The "mosaics of training patches" paraphrase matches ("locally consistent patch mosaic"). Note: Kamb & Ganguli present this as the *mechanism of creativity*; citing it as a "deficit" is the author's interpretation and should say so. |
| cohen2024 | **OK** | TACL 12:283-298, DOI 10.1162/tacl_a_00644. The claim (edits fail to propagate to dependent facts) matches: "Prominent editing methods fail to make consistent changes". |
| li2023slog | **OK** | EMNLP 2023 pp. 3213-3232. See numeric check below. |
| conwell2022 | **OK** (arXiv only) | See numeric check below. |

### Numeric and empirical claims

| Claim in paper | Source wording | Verdict |
|---|---|---|
| "text-to-image models matched about 22 percent of prompts describing fifteen basic relations" (App. C, Case 1) | Conwell & Ullman test **DALL-E 2** only, 15 relations, 169 participants: "only ~22% of images matched basic relation prompts" | **Slightly off.** It is one model, and 22% of *images*, not of prompts. Fix: "DALL-E 2 images matched only about 22 percent of prompts describing fifteen basic relations, as judged by human raters". |
| SLOG: "transformers reached at most 40.6 percent while performing near ceiling when a new word appeared in a familiar frame" | "generalization accuracy of Transformer models, including pretrained ones, only reaches 40.6%"; a structure-aware parser reaches 70.8%; both are far below near-perfect accuracy on COGS | **Essentially correct.** The "near ceiling" contrast is with COGS (mostly lexical generalization), not with SLOG items. Say so. |
| SCAN length split: "the best networks of 2018 reached 20.8 percent" | Lake & Baroni 2018: "The best result (20.8% on average, again over 5 runs) is achieved by a GRU with attention"; "The overall-best model achieves 13.8% accuracy" | **Number correct, framing loose.** 20.8% is the best architecture *in Lake & Baroni's sweep*, not "the best networks of 2018". Fix: "the best architecture tested reached 20.8 percent". |
| Frankland & Greene 2015: agent/patient in separate codes, generalizing across verbs | Distinct lmSTC subregions track agent and patient. In Exp. 2, classifiers trained on four of five verbs were tested on the held-out verb | **Supported.** (I could not reach the PNAS page because of a 403; the held-out-verb design is from secondary reports of the paper.) |
| Liu 2022: summed scores work for separate regions, fail "when one must occupy a region of another"; also "the same model is reported as compositional ... and as non-compositional when one part must lie inside another (liu2022composable, conwell2022)" | Liu et al.'s own failure text: "Such failures usually happen when the objects are in the center of the images"; "the diffusion models confuse the objects' attributes". DALL-E 2 and GLIDE "fail to understand object relations". The paper says nothing about one object inside another | **Not supported by Liu.** (1) The inside/overlap failure rests only on anonymous2026seen. Better existing support is Wiedemer 2024 on additivity vs occlusion. (2) Liu (Composable Diffusion) and Conwell (DALL-E 2) study *different models*, so "the same model is reported as..." is false. Rewrite as "the same mode of combination". (3) Table 4 cites Liu as the source for "Superposition: invertible only if regions are disjoint"; that is the author's inference and should be labeled so, or cited to Wiedemer 2024. |
| Wiedemer 2024: "slot-wise decoders, which provably generalize to unseen slot combinations under additivity" | Theorem 3 needs additivity **and** compositionality of the decoder (each pixel depends on at most one slot), an irreducible compositional diffeomorphic generator, convex training slot-support, and a zero compositional-consistency loss | **Incomplete.** Add "and compositional consistency, in scenes without occlusion". This matters because it is exactly the paper's own invertibility condition and BESIDE/PART-OF prediction. |
| Goodale 2026 "argue that the systematicity challenge concerns what an architecture guarantees" | "Symbolic theories immediately do the job by default, while connectionist theories do not"; "The whole notion of systematicity is precisely about properties of the system that generates behavior" | Acceptable. |
| Schapiro 2025 "fail at the mapping step" | See above | **Wrong.** |

### Remaining entries

I checked the other cited entries against publisher or proceedings records, or recognized them as standard citations: lake2018 (PMLR 80:2873-2882), lake2019 (CogSci 2019 pp. 611-617), lake2023, hupkes2020, keysers2020, press2023 (Findings EMNLP pp. 5687-5711), dziri2023, okawa2023, rassin2022, huang2023t2i, ramesh2021, liu2022composable, smolensky1990, plate1995, pollack1990, vangelder1990, fodor1988, fodor1990, hadley1994, montague1970, frege1923, mitchell2010, smith1988, kamp1995, osherson1981, fodor1996, hampton1987, wisniewski1997, chase1973, gobet2000, feigenbaum1984, doumas2022, whittington2020, geirhos2020, elhage2022, koh2017, grosse2023, mccloskey1989, french1999, hewitt2019, kirby2015, piantadosi2016, evans1983, wason1968, rips1994, aizawa2003, fauconnier2002, andreas2019. No problems found. Minor points:

- smolensky1987: published in the 1988 *Southern Journal of Philosophy* supplement (Spindel Conference 1987). The year 1987 is a common convention.
- gobet2000: the main CHREST *template* paper is Gobet & Simon 1996, "Templates in chess memory: A mechanism for recalling several boards", *Cognitive Psychology* 31:1-40. Consider adding it where templates are cited.
- hampton2012: page range not independently verified **[unverified]**.
- 29 entries in references.bib are never cited (bibtex drops them, so this is harmless). The important ones are listed in Section 4: feng2024, mccoy2019, lepori2023, greff2020, russin2024, frankland2020, ren2020, hummel1997.

---

## 4. Missing citations, ranked

**Must add (a reviewer from this area would flag the omission):**
1. **Wiedemer et al. 2024** (already cited): discuss it in Related Work as the formal source of separability (additive decoder) + invertibility (compositional consistency) → compositional generalization, including its occlusion caveat.
2. **Lachapelle et al. 2023**, NeurIPS: additive decoders, identifiability, Cartesian-product extrapolation.
3. **Schug et al. 2024**, ICLR, "Discovering modular solutions that generalize compositionally": connected-support condition and linear task count; the direct analogue of Prop. 3.
4. **Lippl & Stachenfeld 2025**, ICLR: conjunction-wise additivity and memorization leak; the analogue of Prop. 6 and Test 3.
5. **Uselis, Dittadi & Oh 2026**, arXiv 2602.24264 (ICML 2026 oral per the conference listing): linear + orthogonal factors are *necessary* for compositional generalization.
6. **Redhardt, Akram & Schug 2025**, NeurIPS, "Scaling can lead to compositional generalization": linear decodability of constituents from hidden activations accompanies compositional generalization. This is a direct empirical precedent for Tests 1-2.
7. **Bernardi et al. 2020**, Cell (CCGP, parallelism score), and **Ito et al. 2022**, NeurIPS.
8. **Geiger et al. 2021/2024 (DAS)** and **Sutter et al. 2025**, NeurIPS (the non-linear representation dilemma): the source for interchange interventions, and the formal reason M must be declared.
9. **Feng & Steinhardt 2024** (in .bib) and follow-ups (Dai et al. 2024; Gur-Arieh et al. 2025): additive binding IDs in LMs; counterevidence to "networks mostly blend".
10. **McCoy et al. 2019** (TPDN; in .bib) and **Soulos et al. 2020** (ROLE): discovered TPR decompositions of trained networks; an existing instance of "a decomposition under M".

**Should add:**
11. Reverse compositionality: Fodor & Lepore 2001/2002; Robbins 2005; Johnson 2006.
12. Additive conjoint measurement: Luce & Tukey 1964 (and Debreu 1960 / Krantz et al. 1971). This gives an exact characterization of the paper's "separable".
13. VSA capacity: Plate 2003 (book); Frady, Kleyko & Sommer 2018; Thomas, Dasgupta & Rosing 2021; Kleyko et al. 2022 survey. These support the finite-vocabulary remark and the freshly-drawn-parts calibration.
14. Montero et al. 2021 (ICLR) and Xu et al. 2022 (NeurIPS): decodable factors do not guarantee generalization. These populate the "parts recoverable, behavior fails" cell of Table 1.
15. Chaabouni et al. 2020 (ACL) and Kharitonov & Baroni 2020: generalization without compositional languages. These populate the "imitation" cell.
16. McCoy, Min & Linzen 2020: seed variance in generalization; precedent for Test 5.
17. Dong & Ma 2023 (ICLR) and Wiedemer et al. 2023 (first principles); Brady et al. 2023.
18. Park et al. 2024 (NeurIPS): capability vs behavior separated by latent intervention.
19. Gauderis et al. 2026 (arXiv 2605.08934): MDL + commuting maps, a compositional-interpretability definition.
20. Greff et al. 2020 (in .bib), the binding problem; Russin et al. 2024 (in .bib), review of definitions; Kim & Linzen 2020 (COGS); Lepori et al. 2023 (in .bib); Lewis et al. 2024 ("Does CLIP bind concepts?").

---

## 5. Suggested positioning (one paragraph for Related Work)

"Our separability and invertibility conditions are not new as mathematics. Additive decoders combined with encoder-decoder consistency are the sufficient conditions under which object-centric models provably generalize to unseen slot combinations \citep{lachapelle2023,wiedemer2024}, and a linear number of connected module combinations suffices to identify modular solutions \citep{schug2024}. Conversely, linear, orthogonal factorization is necessary for compositional generalization in embedding models \citep{uselis2026}. Our contribution is to treat these conditions as the *definition* of representational compositionality, relative to a declared discovery class (necessary because unrestricted alignment maps make causal abstraction vacuous; \citealt{sutter2025}). We then derive from it the classical necessity/contingency distinction of \citet{fodor1990}, and we test it with readouts on freshly drawn parts, which separate recoverability guaranteed by the relation from recoverability supplied by a finite vocabulary."

---

## Sources (primary pages consulted)

- Vegner et al. 2025: https://aclanthology.org/2025.acl-long.1537 ; https://arxiv.org/abs/2506.04461
- Woydt et al. 2025: https://arxiv.org/abs/2506.01820v3
- Goodale & Mascarenhas 2026: https://aclanthology.org/2026.tacl-1.73/ ; https://arxiv.org/html/2606.14512
- Elmoznino et al. 2026: https://arxiv.org/abs/2610.20922 ; ICML 2025: https://proceedings.mlr.press/v267/elmoznino25a.html
- Singaravadivelan & Langley 2026: https://arxiv.org/abs/2609.30414
- Schapiro et al. 2025: https://arxiv.org/abs/2509.21043 ; https://arxiv.org/html/2509.21043
- Schwartenbeck et al. 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10914680
- Kamb & Ganguli 2025: https://proceedings.mlr.press/v267/kamb25a.html
- Conwell & Ullman 2022: https://arxiv.org/abs/2208.00005
- Li et al. 2023 (SLOG): https://aclanthology.org/2023.emnlp-main.194/
- Lake & Baroni 2018: https://ar5iv.labs.arxiv.org/html/1711.00350
- Cohen et al. 2024: https://aclanthology.org/2024.tacl-1.16/
- Liu et al. 2022: https://ar5iv.labs.arxiv.org/html/2206.01714
- Frankland & Greene 2015 (secondary): https://psychology.fas.harvard.edu/publications/architecture-encoding-sentence-meaning-left-mid-superior-temporal-cortex
- Wiedemer et al. 2024: https://arxiv.org/abs/2310.05327 ; https://arxiv.org/html/2310.05327v2
- Wiedemer et al. 2023: https://arxiv.org/abs/2307.05596
- Lachapelle et al. 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ef594af0d9a519db8fb292452c461fa-Abstract.html
- Dong & Ma 2023: https://arxiv.org/abs/2211.11719v2
- Lippl & Stachenfeld 2025: https://arxiv.org/html/2405.16391v3
- Schug et al. 2024: https://arxiv.org/html/2312.15001v2
- Redhardt et al. 2025: https://arxiv.org/abs/2507.07207v1
- Uselis et al. 2026: https://arxiv.org/abs/2602.24264
- Sutter et al. 2025: https://arxiv.org/pdf/2507.08802
- Geiger et al. 2024 (DAS): https://proceedings.mlr.press/v236/geiger24a.html
- Xu et al. 2022: https://arxiv.org/abs/2210.00482
- Ito et al. 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/d0241a0fb1fc9be477bdfde5e0da276a-Abstract.html
- Park et al. 2024: https://arxiv.org/abs/2406.19370v3
- Bernardi et al. 2020: https://pmc.ncbi.nlm.nih.gov/articles/8451959
- Soulos et al. 2020: https://arxiv.org/pdf/1910.09113
- Gauderis et al. 2026: https://arxiv.org/abs/2605.08934
- Feng & Steinhardt / follow-ups: https://arxiv.org/html/2310.17191v2 ; https://www.emergentmind.com/papers/2510.06182
- Robbins 2005 / Johnson 2006: https://link.springer.com/doi/10.1007/s11098-005-7785-x ; https://link.springer.com/doi/10.1007/s10670-005-0362-z
