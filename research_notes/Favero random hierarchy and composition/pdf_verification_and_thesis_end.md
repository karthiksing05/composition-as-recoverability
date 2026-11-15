# PDF verification of the Favero papers and the unread end of his thesis

**Provenance.** All quotes below come from text extracted with PyMuPDF on a compute node, from PDFs downloaded from arXiv on 2026-11-06. The text sits beside each PDF under `pdfs/<paper>/`. Line breaks and hyphens are repaired in quotes, and math is written in plain text. Sub/superscripts collapse in extraction, so I checked equations against their surrounding text and redid the algebra where marked. Figure values marked "read off" come from pages rendered at 110 dpi and are approximate.

Versions: thesis v1, ICML v3, Korchinski v1, PRX v5, Bigger v2. Links use `#page=` with the **PDF page**. Thesis printed page = PDF page − 16 in the body.

Labels: **VERIFIED** (as stated in the earlier notes), **DIFFERS** (with the correct version), **NOT FOUND**.

## Which statements in the earlier notes must be corrected?

### Takeaway
Thirteen corrections, listed below by notes file and line. The most consequential: Korchinski does evaluate latent recovery on held-out strings; the P_1 ~ vm conflict is not a conflict; the "routinely verify" sentence is in the main text and says nothing about novelty. All core equations were otherwise transcribed correctly.

### Cited Findings

**`random_hierarchy_model.md`**
1. **Lines 113, 120, 123, 126: DIFFERS.** The notes say no evaluation of new inputs was reported, and that clustering accuracy applies to ILC only. Korchinski App. C.1 reports "test set performance on a held-out test set of 2000 samples" (1000 in Fig. 12) — [p.25](https://arxiv.org/pdf/2605.27734#page=25). Fig. 10 plots "Test clustering accuracy" per layer at L=7, P=9797 — [p.26](https://arxiv.org/pdf/2605.27734#page=26). App. C.3 defines it by comparing learned hard labels with ground-truth latents under a Hungarian mapping — [p.26](https://arxiv.org/pdf/2605.27734#page=26). So held-out **strings** are scored for per-level latent identity in SLC. Held-out rules, tuples, or symbols are still absent (see the Korchinski section).
2. **Line 196, first half: DIFFERS, withdraw.** The notes list "labels P₂ vs P₃ for the m³/m⁴ thresholds" as an inconsistency in 2502.12089. The PDF is consistent: P_2 = (1−f)⁻¹vm³, P_3 = (1−f)⁻¹vm⁴ (Eq. 4), and Eq. 6 gives both at ℓ=2,3 — [ICML pp.6–7](https://arxiv.org/pdf/2502.12089#page=6).
3. **Line 196, second half: DIFFERS in framing.** PRX's 0.7 and 0.1 are separate criteria for separate architectures. The two-layer FC net uses ε ≤ 0.7 ε_rand (Fig. 2), and P* for deep nets uses ε_rand/10 (Eq. 7) — [PRX p.4](https://arxiv.org/pdf/2307.02129#page=4).
4. **Lines 75 and 181: DIFFERS.** "U-Net must have depth ≥ 2L" overstates a footnote: "we expect our results to remain valid for sufficiently expressive architectures, in particular, if the network depth is at least 2L" — [ICML p.4](https://arxiv.org/pdf/2502.12089#page=4). The experiments use L blocks in the encoder and L in the decoder.
5. **Lines 127, 207: DIFFERS.** "Sec. 49" is the HTML renderer's numbering. It is the first paragraph of Sec. 9.3 — [thesis pp.140–141](https://arxiv.org/pdf/2510.06106#page=156).
6. **Line 98: incomplete.** Korchinski App. A cites Parley, Cagnetta, Wyart, arXiv:2602.06065, saying it "obtains sample complexity for non-regular trees" — [p.16](https://arxiv.org/pdf/2605.27734#page=16). The notes do not list it. I did not read it.
7. **Line 113: incomplete.** The SLC probe target is the root label. Fig. 12: "linear probes trained to classify the root latent h^(L)_1 from the final SLC representation" — [p.27](https://arxiv.org/pdf/2605.27734#page=27). Data2vec also has linear probes (Figs. 13–14), besides the MLP — [pp.27–28](https://arxiv.org/pdf/2605.27734#page=27).

**`diffusion_composition_creativity.md`**

8. **Line 77: DIFFERS on location.** The "routinely verify" sentence is in Sec. 3.1 of the main text, not App. C — [ICML p.4](https://arxiv.org/pdf/2502.12089#page=4). App. C has only the early-stopping sentence — [p.16](https://arxiv.org/pdf/2502.12089#page=16).
9. **Lines 119–120, 215: conflict resolved, no error in the paper.** See the ICML section. Eq. 6 does not extend to ℓ=1, and the level indices differ by a fixed offset between papers.
10. **Line 26: DIFFERS in detail.** R_ℓ is defined by "resetting all choices of the production rules emanating from level ℓ", "equivalent to substituting all tuples at depth ℓ−1 with a synonym" — [ICML p.17](https://arxiv.org/pdf/2502.12089#page=17). The next sentences then say "at depth ℓ", so the paper itself is loose. PRX's P_l replaces "all the level-l tuples" — [PRX p.5](https://arxiv.org/pdf/2307.02129#page=5). PRX's l therefore equals ICML's ℓ−1. Do not compare S_{k,l} across the two papers by raw index.
11. **Lines 66, 136, 159: gaps closed.** Parameters are in figure titles and captions; see the ICML and Bigger sections.
12. **Line 209: DIFFERS.** The depth sweep L=3..7 does not use the L=4 parameters. Fig. 12 has s=3, m=8, v=16 — [p.27](https://arxiv.org/pdf/2605.27734#page=27); Fig. 3 right has L=4, s=3, v=10 — [p.6](https://arxiv.org/pdf/2605.27734#page=6).

**`thesis_and_bibliography.md`**

13. **Lines 72–83: DIFFERS on numbering.** "Theorem 9.1, 11.1, 11.1.1, 15.1, Lemma 17.1" are HTML numbers. The PDF has Theorem 2.4.1, Theorem 2.6.1, Corollary 2.6.1.1, Theorem 3.3.1 and Lemma 3.5.1 — [thesis pp.33–49](https://arxiv.org/pdf/2510.06106#page=49). "Corollary 17.0.1" is probably Corollary 3.5.0.1, matched by order and title only. Line 116's "Chapters 10 to 16" are lettered Appendices A–G in the PDF. Do not cite any of the old numbers.

**Earlier inferences now confirmed or upgraded**
- Thesis line 125, chapter-to-paper mapping: VERIFIED by each chapter's "Parts of this chapter" box. Ch. 2 = JSTAT 2022, Ch. 3 = JSTAT 2024, Ch. 4 = PNAS, Ch. 5 = ICLR, Ch. 6 = ICML 2025 — [p.87](https://arxiv.org/pdf/2510.06106#page=103); Ch. 7 = ICML 2025 workshop version of Bigger — [p.103](https://arxiv.org/pdf/2510.06106#page=119); Ch. 8 = NeurIPS 2023 — [p.119](https://arxiv.org/pdf/2510.06106#page=135). PRX has no chapter.
- Thesis line 128: confirmed. Ch. 6 has one Assumption and no theorem. Ch. 7 has a kernel scaling argument (App. F.5). Ch. 8 has one Proposition, with a "Simplified" statement — [p.129](https://arxiv.org/pdf/2510.06106#page=145).
- Tags in thesis lines 147 and 157: the RHM is used by name in both papers, so [RHM] applies — [Korchinski p.2](https://arxiv.org/pdf/2605.27734#page=2); [ICML p.1](https://arxiv.org/pdf/2502.12089#page=1).
- `random_hierarchy_model.md` line 167, the snippet on mildly context-sensitive models: VERIFIED — [thesis p.141](https://arxiv.org/pdf/2510.06106#page=157).
- Diffusion notes line 30, the "tool summary" on decoder layers: now VERIFIED verbatim (ICML section).

### Inferences
- The transcription errors were few. Most problems were in locations, numbering, and conclusions drawn from unread tails.

### Gaps
- I checked only the claims in the assignment plus what I met while reading. Not rechecked: PRE 2025, Ren et al., Delrocq et al., Liu, PNAS, ICLR, Kamb and Ganguli, Okawa, Liang, Bonnaire, and the PRX appendices.

## Thesis (arXiv 2510.06106): what do Chapters 6, 7, 8 and 9 say?

### Takeaway
Chapters 6 to 8 are lightly edited versions of ICML 2025, Bigger, and NeurIPS 2023. Chapter 9 adds the only new material: a comparison with Gaussian mixtures and Gaussian random fields, and four future directions. The thesis mentions interpretability three times in chapter text, always as outlook. It never decodes latents from internal representations, never holds out rules or parts, and names context dependence as open future work.

### Cited Findings

**Ch. 6, "A theory of creativity and compositionality" (printed pp.87–101).**
- Same text as ICML with new numbers: Assumption 6.3.1, Eqs. 84–89, Figs. 17, 18, 57, 58, Eq. 404 for S_{k,ℓ} — [pp.95–96](https://arxiv.org/pdf/2510.06106#page=111); [p.244](https://arxiv.org/pdf/2510.06106#page=260).
- **Typo to avoid quoting:** Eq. 89 is followed by "confirming Theorem 6.3.1", where ICML has "confirming Assumption 4.1". There is no such theorem — [p.96](https://arxiv.org/pdf/2510.06106#page=112); [ICML p.7](https://arxiv.org/pdf/2502.12089#page=7).
- Conclusions: "when learning from data generated by a simple context-free grammar, U-Nets reduce the dimensionality by assigning identical representations to groups of features that share similar contexts" — [p.99](https://arxiv.org/pdf/2510.06106#page=115).

**Ch. 7, "A race between memorization and generalization" (pp.103–115).**
- Empirical law τ_mem ∝ P (Eq. 90), with theoretical support from kernel regression: "the time to fit the empirical score scales as τ_mem ∝ P/σ^ν" — [p.109](https://arxiv.org/pdf/2510.06106#page=125).
- Summary of the RHM: "P* ∼ v m^(L+1)", "P_ℓ ∼ v m^(ℓ+1)", "the number of training steps τ_ℓ required to learn the rules at layer ℓ is proportional to P_ℓ" — [p.110](https://arxiv.org/pdf/2510.06106#page=126).
- The phase diagram is Fig. 27, with partial generalization, generalization and memorization. It carries a caveat: "in the simplest version of the RHM, learning proceeds by well-distinct steps, while it is smoother for natural data" — [p.115](https://arxiv.org/pdf/2510.06106#page=131).

**Ch. 8, "Task compositionality in weight space" (pp.119–133).**
- Topic: composing whole tasks by adding weight vectors in CLIP ViTs, with disjoint input supports. "the sole condition for task arithmetic is actually weight disentanglement" — [p.120](https://arxiv.org/pdf/2510.06106#page=136).
- Definition, Property 3 (Eq. 94): f(x; θ0 + Σ α_t τ_t) = Σ g_t(x; α_t τ_t) + g_0(x), with g_t vanishing outside D_t — [p.125](https://arxiv.org/pdf/2510.06106#page=141).
- Results: post-hoc linearization has lower absolute but higher normalized task-addition accuracy, and linearized fine-tuning beats nonlinear (ViT-B/32 76.5 vs 71.4, ViT-L/14 88.5 vs 85.1) — [Table 5, p.123](https://arxiv.org/pdf/2510.06106#page=139).
- Random-init ViTs fail task addition (multi-task 3.8 to 6.1%), so "task arithmetic is a property acquired during pre-training" — [Table 7, p.131](https://arxiv.org/pdf/2510.06106#page=147).
- Explicitly separated from representation work: "Departing from the static perspective of feature disentanglement, weight disentanglement connects weight space and function space transitions" — [p.132](https://arxiv.org/pdf/2510.06106#page=148).

**Ch. 9.2, "Comparison with other theoretical frameworks" (pp.139–140).** There are only two comparators, and only for Part III.
- Framing: "the Random Hierarchy Model – characterized by non-Gaussian statistics – predicts qualitatively different phenomena from other common theoretical data models." — [p.139](https://arxiv.org/pdf/2510.06106#page=155).
- GMMs: "Although this resembles the class phase transition we identify, it lacks compositionality. In the RHM, following the class change, low-level features from the original sample are preserved and recombined to form the new sample. This compositional process – a form of combinatorial creativity that we confirm empirically in natural data – cannot be explained by GMMs." — [pp.139–140](https://arxiv.org/pdf/2510.06106#page=155). Also: "as mean-field models with no inherent spatial structure, GMMs do not present the growing dynamical susceptibility or length scale at the transition" — [p.140](https://arxiv.org/pdf/2510.06106#page=156).
- GRFs: "While a GRF can be tuned to match the RHM's two-point statistics, it fundamentally lacks the higher-order correlations induced by the RHM's latent tree structure." — [p.140](https://arxiv.org/pdf/2510.06106#page=156).
- **The sentence that ties the mechanism to context-freeness:** "Crucially, the RHM's context-free nature reduces complex multi-body correlations among observable tokens to effective pairwise correlations involving the latent variables. This property allows a deep learning algorithm to efficiently infer the hierarchy by clustering token groups that correspond to the same latent variable, yielding the 'local-to-global' learning dynamics we predict and observe." — [p.140](https://arxiv.org/pdf/2510.06106#page=156).
- Spectral bias: "eigenmodes of data covariance are learned in order of decreasing variance, meaning that low-frequency (global) components are acquired first [Wan25]. This predicts an opposite phenomenology" — [p.140](https://arxiv.org/pdf/2510.06106#page=156).
- Forward-backward: "a GRF exhibits a dynamical correlation length that grows monotonically with the inversion time. Whereas, in the RHM, the correlation length peaks at a finite time" — [p.140](https://arxiv.org/pdf/2510.06106#page=156).
- Conclusion: "The concurrence of a 'local-to-global' learning dynamics and a susceptibility peak in real systems provides strong evidence that hierarchical compositionality, not just second-order statistics, is a key principle" — [p.140](https://arxiv.org/pdf/2510.06106#page=156).

**Ch. 9.3, "Limitations and future directions" (pp.140–142).** Four headed paragraphs.
1. *Beyond simplified models of hierarchy.* The stated simplifications are "fixed, regular tree topology and random production rules" — [p.141](https://arxiv.org/pdf/2510.06106#page=157). Proposed extensions:
   - "Irregular tree topologies: ... it would introduce a degree of ambiguity, which is also found in language, and is currently absent from the model." "The model would then need to learn to resolve these parsing uncertainties, likely by leveraging broader context to infer the most probable latent structure." — [p.141](https://arxiv.org/pdf/2510.06106#page=157).
   - "Context dependences: The RHM is fundamentally a probabilistic context-free grammar. However, it is established that natural language requires at least mildly context-sensitive models to capture all syntactic phenomena. Introducing more general latent models that involve context dependencies is thus a key challenge for future work. For instance, this could be achieved by allowing for more complex latent variables that encode additional information or considering models that gradually depart from a tree-like structure." — [p.141](https://arxiv.org/pdf/2510.06106#page=157).
2. *The dynamics of hierarchical feature learning.* "a complete, dynamical theory of how these hierarchical features are learned across layers via gradient-based optimization remains elusive." — [p.142](https://arxiv.org/pdf/2510.06106#page=158).
3. *Probing latent structures in natural data.* "using these experiments as a data-driven method to probe and extract the latent hierarchical structure of real data. One could, for instance, interpret the large, correlated blocks of changing tokens in language as revealing grammatical constituents or context variables." Then the RG generalization, with turbulence as a possible application — [p.142](https://arxiv.org/pdf/2510.06106#page=158).
4. *The origins of weight disentanglement.* "the question of how this emergence occurs is unanswered." — [p.142](https://arxiv.org/pdf/2510.06106#page=158).

**Ch. 9.4.** The last sentence names the goal: "AI systems that are robust, interpretable, and universally useful." — [p.143](https://arxiv.org/pdf/2510.06106#page=159).

**Keyword audit over the full 333 pages** (case-insensitive search of the extracted text; hit lists read in context)
- **Interpretability** (four hits: three in chapters, one bibliography title).
  - Ch. 5: "they can enhance the interpretability of deep networks, whose representations are believed to reflect the hierarchical structure of data" — [p.85](https://arxiv.org/pdf/2510.06106#page=101).
  - Ch. 6: "Performing explicitly a 'word2vec' procedure hierarchically ... would be better controlled and easier to interpret." — [p.100](https://arxiv.org/pdf/2510.06106#page=116).
  - Ch. 9.4, above.
  - All are outlook statements, with no method or measurement attached.
- **Reading latents from internal representations.** "linear probe" has zero hits, and "probe" never refers to a decoder. The one internal measure in Part III's trained diffusion models is S_{k,ℓ} — [p.93](https://arxiv.org/pdf/2510.06106#page=109); [pp.244–245](https://arxiv.org/pdf/2510.06106#page=260). The thesis positions itself against reconstruction: "we focused on understanding how modifications in the latent variables manifest in the data, in contrast with common approaches which attempt to reconstruct the latent representations from visible data." — [p.85](https://arxiv.org/pdf/2510.06106#page=101). It also leaves open "Investigating whether the function learned by U-Nets approximates BP" — [p.72](https://arxiv.org/pdf/2510.06106#page=88).
- **Held-out rules or parts.** NOT FOUND. All four "held-out" hits are data splits: forward-backward inputs, Bigger's RHM validation loss, a CLIP validation set, and hyperparameter selection — e.g. [p.111](https://arxiv.org/pdf/2510.06106#page=127). "Unseen" means generic test data or zero-shot tasks.
- **Reusable parts** appear as framing: "A compelling hypothesis ... these models might learn a 'library' of reusable parts" — [p.14](https://arxiv.org/pdf/2510.06106#page=30); "diffusion models naturally act as 'compositional samplers', building new data from known, reusable parts" — [p.19](https://arxiv.org/pdf/2510.06106#page=35). VERIFIED: "recovering the latent variables – an ability that fundamentally depends on feature learning" — [p.19](https://arxiv.org/pdf/2510.06106#page=35).
- **Context sensitivity.** Only Sec. 9.3 above and "more general latent models involving context dependencies" in Ch. 5 — [p.85](https://arxiv.org/pdf/2510.06106#page=101).

### Inferences
- Sec. 9.2's sentence on context-freeness locates the limit of the whole account. The clustering mechanism works because a tuple's context depends on it only through its parent. Korchinski's Lemma 1 proves the same point — [p.19](https://arxiv.org/pdf/2605.27734#page=19) — so context dependence would break the identity of synonym context vectors that the mechanism rests on. The thesis calls this a future challenge and does not draw that consequence.
- Sec. 9.3's "probing" direction reads structure from output changes. A latent readout from activations does not appear among the future directions, even though Fig. 57's premise needs it. A paper that decodes latents from internal states, or that holds out rules, is not anticipated by this thesis.
- Ch. 9 upgrades the wording. Sec. 9.1 says "We showed that the sample complexity for learning the grammar rules scales polynomially" — [p.139](https://arxiv.org/pdf/2510.06106#page=155). Ch. 6 derives this under Assumption 6.3.1 plus an empirical match, and the opening of Ch. 9 itself says "We argued" — [p.137](https://arxiv.org/pdf/2510.06106#page=153). Cite Ch. 6's form.
- The comparison section omits the closest rivals, such as Kamb and Ganguli's patch mosaics, which appear only in related work — [p.88](https://arxiv.org/pdf/2510.06106#page=104).

### Gaps
- I read Chapters 6 to 9 in full, plus the last page of Ch. 5 and the App. E.4 sensitivity pages. Appendices F and G were read only where cited. Kernel derivations in App. F.5 were not checked.

## Favero et al., ICML 2025: Eq. 6, P_1, S_{k,l}, parameters, "routinely verify"

### Takeaway
The local file is the arXiv v3 PDF, byte for byte. Eq. 6, Eq. 26 and the caption are as the notes state. P_1 ~ vm is what the paper says, from a separate memorization argument, and v m² appears nowhere. The "routinely verify" sentence gives no method and does not mention novelty.

### Cited Findings
- **Identity: VERIFIED.** `cmp` reports `/coc/flash10/ksingara3/composition-creativity.pdf` identical to arxiv.org/pdf/2502.12089, which is stamped "arXiv:2502.12089v3 [stat.ML] 4 Jun 2025" and carries authors Favero*, Sclocchi*, Cagnetta, Frossard, Wyart — [p.1](https://arxiv.org/pdf/2502.12089#page=1). The local file was not modified.
- **Eq. 6: VERIFIED.** "P_ℓ = (1 − m/v^(s−1))^(−1) v m^(ℓ+1)", described as "the sample complexity required to reconstruct level-(ℓ−1) latents" — [p.7](https://arxiv.org/pdf/2502.12089#page=7). Inputs: Eq. 5, C(ℓ) ≃ sqrt[(1 − m/v^(s−1)) / (v³ m^(ℓ+2))], and sampling noise (v² m P)^(−1/2). Equating them gives Eq. 6 (algebra checked).
- **P_1: VERIFIED as vm; v m² is NOT FOUND in the paper.** "Since these tuples lack internal structure, they can only be memorized. Each tuple can take vm possible configurations ... Thus, the sample complexity required to learn the local constraints scales as P_1 ∼ vm." — [p.6](https://arxiv.org/pdf/2502.12089#page=6). The next thresholds are P_2 = (1−f)⁻¹ v m³ and P_3 = (1−f)⁻¹ v m⁴ (Eq. 4) — [p.6](https://arxiv.org/pdf/2502.12089#page=6). Eq. 6 concerns correlations with tuples of latents "at level ℓ−2" — [pp.6–7](https://arxiv.org/pdf/2502.12089#page=6) — so it starts at ℓ=2. Its value at ℓ=1 is simply not a claim.
- **Assumption 4.1: VERIFIED.** "The U-Net learns to generate data that is consistent with the rules at level ℓ when the correlations between a visible token and a tuple of latents at level ℓ−2 become detectable from the training data." The paper later says Eq. 6's scaling match is "confirming Assumption 4.1" — [pp.6–7](https://arxiv.org/pdf/2502.12089#page=6).
- **Overall: VERIFIED.** "m^(L+1) = m d^(log m / log s)" — [p.7](https://arxiv.org/pdf/2502.12089#page=7).
- **Sec. 3.4: VERIFIED.** "we alter the subtree generated by a given latent variable, while keeping that latent variable itself fixed"; "the hidden representations of the U-Net become increasingly invariant to such perturbations" — [p.5](https://arxiv.org/pdf/2502.12089#page=5).
- **Eq. 26: VERIFIED.** "S_{k,ℓ} = E_x[‖a_k(x) − a_k(R_ℓ x)‖²] / E_{x,y}[‖a_k(x) − a_k(y)‖²]", with a_k the pre-activations — [App. D.1, p.17](https://arxiv.org/pdf/2502.12089#page=17). The diffusion time or noise level at which activations are measured, and whether x comes from training or test data, are not stated.
- **Interpretation: VERIFIED.** "A low value of S_{k,ℓ} indicates that the network is invariant to synonym substitutions at depth ℓ, implying that it has learned the corresponding compositional rule." — [p.17](https://arxiv.org/pdf/2502.12089#page=17).
- **Fig. 6 caption: VERIFIED.** "The U-Net learns to satisfy rules at level ℓ when it becomes insensitive to the synonyms of the variables at level ℓ − 1." Also "deeper layers of the encoder become less sensitive to deeper synonymic transformations" and "The decoder layers, instead, progressively regain the sensitivity to the synonyms layer-by-layer" — [p.16](https://arxiv.org/pdf/2502.12089#page=16). App. D.1 strengthens it: "the network begins to satisfy rules at level ℓ precisely when it becomes insensitive to synonymic variations at level ℓ − 1. This suggests that the U-Net learns to collapse lower-level synonyms into shared latent representations" — [p.17](https://arxiv.org/pdf/2502.12089#page=17). No statistic is attached.
- **RHM parameters.**

| Figure | Parameters | Status |
|---|---|---|
| Fig. 1 | L=5, s=2, v=16, m=3 | In panel titles — [p.4](https://arxiv.org/pdf/2502.12089#page=4) |
| Fig. 2 | L=2, s=3, v=24, m on the x-axis (about 6 to 30) | In the panel title — [p.5](https://arxiv.org/pdf/2502.12089#page=5) |
| Fig. 6 | Five levels and five encoder/decoder layers, so L=5. **s, v, m NOT FOUND** in caption, figure or App. D.1 | [p.16](https://arxiv.org/pdf/2502.12089#page=16) |
| Fig. 7 | L=3, s=3, v=24 | [p.17](https://arxiv.org/pdf/2502.12089#page=17) |

- **Fig. 6, read off.** In encoder layer 5, sensitivity to ℓ≤3 falls to about 0.05 to 0.1, and ℓ=4 stops near 0.5. Encoder layer 2 gives ℓ=1 about 0.35. Decoder layer 5 returns to about 0.5 (ℓ=1) and 0.85 (ℓ=2). Encoder layer 1 stays at about 1. So invariance is graded rather than complete — [p.16](https://arxiv.org/pdf/2502.12089#page=16).
- **"Routinely verify": exact text.** "Since these networks have enough capacity to memorize their training set, we employ early stopping, halting training when the validation loss plateaus or begins to increase. Moreover, we routinely verify that the model has not simply memorized the training data." — [Sec. 3.1, p.4](https://arxiv.org/pdf/2502.12089#page=4). No method, threshold, or number follows. **It says nothing about novelty.** The word "novel" occurs in the abstract, the introduction, a related-work sentence ("novel predictions"), and the conclusion ("novel viewpoint"), and never in a measurement — [p.1](https://arxiv.org/pdf/2502.12089#page=1). The thesis repeats the sentence unchanged — [p.91](https://arxiv.org/pdf/2510.06106#page=107).
- **A_ℓ: VERIFIED.** "the fraction of generated samples that satisfy level-ℓ rules", on 1024 samples — [p.4](https://arxiv.org/pdf/2502.12089#page=4). Sample complexity is A_L crossing 1/2 — [p.5](https://arxiv.org/pdf/2502.12089#page=5).
- **Interpretability line** (missing from the earlier notes): "Our analysis suggests opportunities to improve the interpretability of generative models." The proposal is an explicit hierarchical word2vec, "most likely inferior" but "better controlled and easier to interpret" — [p.9](https://arxiv.org/pdf/2502.12089#page=9).

### Inferences
- Index reconciliation (algebra checked, resolves the earlier open item). ICML's ℓ labels the rule level; Korchinski's ℓ labels the latent level being built. Then ℓ_K = ℓ_ICML − 1, and v m^(ℓ_K+2) = v m^(ℓ_ICML+1). Both reach v m^(L+1) at the top, and ICML's P_2 equals Korchinski's P^tok_1 = v m³.
- Fig. 6's axis reaches P = 10⁶, like the online panel Fig. 1(b), so it may be an online run with Fig. 1's parameters. The paper does not state this.
- The Fig. 6 evidence is one sweep compared by eye. "Precisely" in App. D.1 is stronger than what a figure with widely spaced P can establish.

### Gaps
- The memorization check's data for ICML is not in the paper. Bigger supplies the method for a later U-Net setup, and it is not the same optimizer (Bigger uses Adam, ICML SGD) — [Bigger p.8](https://arxiv.org/pdf/2505.16959#page=8); [ICML p.16](https://arxiv.org/pdf/2502.12089#page=16).

## Korchinski, Favero, Wyart (arXiv 2605.27734): theorem, baselines, probes, held-out evaluation, status

### Takeaway
The theorem exists as stated, but it is a sufficient-sample bound for the ILC algorithm alone, conditional on an assumed stable clustering module and on separation justified by a heuristic. The supervised and token-level baselines are not proved here. Neural results at vm³ are experimental. Held-out strings are evaluated; held-out rules, tuples and grammars are not. It is a preprint on every source I could read directly.

### Cited Findings

**Theorem: VERIFIED with scope limits.**
- Eq. 2 (informal, Thm 1) and Eq. 3 (formal, Thm 2): "P ≥ C( vm log(Lvm/δ) + (vm³/(1−f)) log(Lvm/δ) )", then ILC "recovers h^(1), h^(2), ..., h^(L−1) and all production-rule classes below the root with probability at least 1 − δ, up to independent permutations of the latent labels at each level." — [p.5](https://arxiv.org/pdf/2605.27734#page=5); [p.19](https://arxiv.org/pdf/2605.27734#page=19).
- **Sufficiency only.** The proof ends: "P ≳ vm³/(1−f) suffices for recovery, and we write P_ILC ≍ vm³/(1−f) for this threshold." — [p.22](https://arxiv.org/pdf/2605.27734#page=22). No lower bound is proved, so "≍" is notation for this threshold.
- **Depth still enters through log L.** The abstract says "constant in L, up to logarithmic factors" — [p.1](https://arxiv.org/pdf/2605.27734#page=1).
- **Assumption 1:** c_bal/(vm) ≤ P[T=ν] ≤ C_bal/(vm), and min ‖Φ_a − Φ_b‖₂ ≥ c_sep √(1−f)/m, for ℓ = 0..L−2 — [p.18](https://arxiv.org/pdf/2605.27734#page=18). Balance cites Lemma C.1 of Parley et al. "in the limit v → ∞ at fixed f ∈ (0,1)". Separation rests on "a random-direction heuristic" — [pp.18–19](https://arxiv.org/pdf/2605.27734#page=18).
- **Assumption 2:** if ε ≤ Δ/8 then Cluster_v "returns the true partition" — [p.18](https://arxiv.org/pdf/2605.27734#page=18). It is assumed. The experiments use k-means — [p.6](https://arxiv.org/pdf/2605.27734#page=6).
- **Inputs given to the algorithm:** "RHM parameters L, s, v" — [Alg. 1, p.4](https://arxiv.org/pdf/2605.27734#page=4).
- **The root is excluded and unrecoverable without labels:** "unsupervised learning can recover h^(1), ..., h^(L−1) and the support of valid top-level tuples, but not the root labels themselves." — [fn. 3, p.4](https://arxiv.org/pdf/2605.27734#page=4).
- **Scope of "prove".** The abstract says "we prove that latent prediction achieves this", and the conclusion says "We have proved that learning from one's own latents recovers the full non-root latent tree" — [p.1](https://arxiv.org/pdf/2605.27734#page=1); [p.9](https://arxiv.org/pdf/2605.27734#page=9). The theorem covers ILC only. SLC is experimental. Data2vec is "reasonable assumptions": (A1) targets carry the learned latents, and (A2) "the same correlation-learning hypothesis used in previous RHM analyses" — [pp.2, 8](https://arxiv.org/pdf/2605.27734#page=8).

**Baselines: VERIFIED as stated, and they are not proved in this paper.**
- Supervised: "P^sup_ℓ ≍ v m^(L−ℓ). The hardest step is therefore the first one, ℓ = 0" — [p.4](https://arxiv.org/pdf/2605.27734#page=4). This gives v m^L; fn. 5 says "∼vm^L = vm⁴" — [p.8](https://arxiv.org/pdf/2605.27734#page=8).
- Token-level: "P^tok_1 ≍ vm³", and Eq. 1, "if ℓ ≥ 1 denotes the latent level being learned above the leaves, P^tok_ℓ ≍ v m^(ℓ+2)". "The overall sample-complexity is gated behind reconstructing latents ℓ = L − 1 leading to P_tok ≍ vm^(L+1)." — [p.4](https://arxiv.org/pdf/2605.27734#page=4).
- Support: "Prior work shows", plus "Neural networks ... were empirically shown to saturate this staged scaling" — [pp.2, 4](https://arxiv.org/pdf/2605.27734#page=2). Ren et al. is credited with a rigorous supervised result — [p.16](https://arxiv.org/pdf/2605.27734#page=16).

**Linear probe: VERIFIED, target is the root.**
- Text: "a linear probe trained to recover the top-level latent from the frozen final-layer representations of an SLC network, for L = 4, s = 3, v = 10" — [p.7](https://arxiv.org/pdf/2605.27734#page=7). Fig. 3's caption: "Class reconstruction accuracy" — [p.6](https://arxiv.org/pdf/2605.27734#page=6). Fig. 12: "classify the root latent h^(L)_1 ... The pre-trained models were trained for 30 epochs, and the classifiers for the same duration and on the same dataset." — [p.27](https://arxiv.org/pdf/2605.27734#page=27).
- This probe receives root labels, which unsupervised learning cannot supply. Fn. 5 says "an additional O(vm) labeled examples then suffice to learn the root partition" — [p.8](https://arxiv.org/pdf/2605.27734#page=8).
- Data2vec: main-text MLP probe on mean-pooled final-layer features — [p.7](https://arxiv.org/pdf/2605.27734#page=7). Linear and MLP on offline runs — [Fig. 14, p.28](https://arxiv.org/pdf/2605.27734#page=28). Linear probes of h^(ℓ), ℓ=2,3,4, from the **teacher target** — [Fig. 13, p.27](https://arxiv.org/pdf/2605.27734#page=27).

**Synonym clustering score: VERIFIED, no formula printed.**
- Definition is verbal: compares a synonym swap, "which resamples every level-ℓ production rule and thus replaces each level-(ℓ−1) tuple by a synonym while leaving higher-level latents unchanged", with "a generic swap to an unrelated input". "normalized so that C_ℓ = 0 when synonym swaps move the representation as much as generic swaps (no clustering), and C_ℓ = 1 when synonyms are mapped to a common image". Measured "after the third block of the encoder" — [p.9](https://arxiv.org/pdf/2605.27734#page=9). The paper calls this "direct evidence" for the recursive clustering mechanism.
- **Magnitudes, read off.** Fig. 5 at the largest P: C_1 about 0.6 to 0.8, C_2 about 0.45, C_3 about 0.1 — [p.9](https://arxiv.org/pdf/2605.27734#page=9). Fig. 13 left (m=6): peak about 0.37, 0.28, 0.13 for ℓ=1,2,3, about 0 for ℓ=4, near encoder layer 2, falling to about 0.03 by the last layer — [p.27](https://arxiv.org/pdf/2605.27734#page=27). Text: "Deeper layers exhibit stronger clustering of low-level latents, up to layer 2. This inversion is reminiscent of an encoder-decoder architecture" — [pp.26–27](https://arxiv.org/pdf/2605.27734#page=26).
- **Absolute scale, read off.** Online data2vec transitions at P/(vm³) of about 10⁴ — [Fig. 4, p.7](https://arxiv.org/pdf/2605.27734#page=7); [Fig. 5, p.9](https://arxiv.org/pdf/2605.27734#page=9). Fig. 14's offline axis has ticks 10⁰ to 10² — [p.28](https://arxiv.org/pdf/2605.27734#page=28). SLC transitions near P/(vm³) of order 1 — [Fig. 3, p.6](https://arxiv.org/pdf/2605.27734#page=6). The m ranges are short: 3 to 7 (data2vec), 7 to 15 (SLC), 6 to 32 (ILC).
- **Random-init control** (Fig. 13 right): a dashed baseline sits around 0.75 at m=3, ℓ=2, well above chance — [p.27](https://arxiv.org/pdf/2605.27734#page=27).
- Fig. 11 is a useful negative control: collapse under m^(L+1) scaling is "significantly degraded" — [p.26](https://arxiv.org/pdf/2605.27734#page=26).

**New or held-out trees and rules.**
- Whole strings: yes, as in Correction 1.
- Rules, tuples, symbols: **NOT FOUND.** "probabilities are taken over fresh samples from a fixed grammar instance" — [p.18](https://arxiv.org/pdf/2605.27734#page=18). The proof's first step requires every grammatical tuple to be observed: a Chernoff and union bound over "all (L−1)vm pairs (ℓ, ν)" gives "N_{ℓ,ν} ≳ P/(vm) simultaneously for all ℓ and ν" — [p.21](https://arxiv.org/pdf/2605.27734#page=21). ILC estimates "the support of grammatical tuples by the observed tuple set" — [p.4](https://arxiv.org/pdf/2605.27734#page=4), and nothing labels an unobserved tuple.
- New grammar instance: NOT FOUND. "three independent realizations" means retraining per instance — [p.7](https://arxiv.org/pdf/2605.27734#page=7).
- Data-split scope of the probes: Fig. 12 trains the probe on the pretraining set — [p.27](https://arxiv.org/pdf/2605.27734#page=27). Disjointness of test from train is implied by "held-out", with no construction described.

**What the unread tail contains (Apps. B to E).** Assumptions and full proof (B), SLC architecture, losses, hyperparameters tuned at L=5, v=16, m=8, s=3, and local-rule ablations (C), data2vec layerwise and probe results (D), compute (E: about 10, 100, 3 H100-hours for SLC figures; about 1,000 for data2vec) — [pp.18–28](https://arxiv.org/pdf/2605.27734#page=18). Also App. A: Delrocq et al.'s CLAPP used "a contrastive update rule that is gated by an extra learning signal", at f=1 — [p.16](https://arxiv.org/pdf/2605.27734#page=16). That confirms the earlier CLAPP note.

**Limitations: VERIFIED.** "(i) variable tree topologies, (ii) recursive production rules that can call themselves, and (iii) explicitly context-dependent rules" — [p.10](https://arxiv.org/pdf/2605.27734#page=10).

**Status: preprint.** The first page says "Preprint." and is stamped "arXiv:2605.27734v1 [cs.LG] 26 May 2026" — [p.1](https://arxiv.org/pdf/2605.27734#page=1). The arXiv API (queried 2026-11-07 UTC) returns v1 as latest with no journal_ref — [API](https://export.arxiv.org/api/query?id_list=2605.27734). The author's page still says "arXiv preprint, 2026" — [publications](https://alesfav.github.io/publications/). The earlier notes' Scholar listing (NeurIPS 39) stays unconfirmed.

### Inferences
- The theorem gives parsing of any new string by table lookup: on the good event all vm tuples per level are labelled, and the grammar is unambiguous. So the earlier inference at `random_hierarchy_model.md` line 120 holds. The authors do not state this, though Fig. 10 tests the neural analog. It remains composition of fully seen rules, with no test of what the method does on a tuple it never saw.
- The vm term in Eq. 2 is tuple coverage, the same quantity as ICML's P_1 ~ vm. The vm³/(1−f) term is the concentration requirement.
- "Exponentially fewer" compares a sufficient bound for ILC with scalings measured or argued for token-level nets. The comparison is sound as an order-of-growth reading and lacks a proved gap, since neither side has a lower bound in this paper.
- The scaling claim leaves absolute counts open. Online data2vec's prefactor of about 10⁴ means P* of several 10⁶ at m=3 (Fig. 4 inset). Token-level v m⁵ at m=3 is about 3,900 before constants, whose size is unknown, and the two counts differ in regime (online vs a stated scaling). So at these small m the scaling result does not establish a lower absolute count, and the paper does not compute a crossover. Its proposed same-architecture comparison — [pp.9–10](https://arxiv.org/pdf/2605.27734#page=9) — has not been run there.
- Probes and clustering scores tell different stories in data2vec. Root probes on final-layer features reach 1.0 while the last-layer clustering score is about 0.03 at m=6. A high C_ℓ is therefore not what the final-layer probe depends on, and low C at the output does not mean the latent is unreadable. A representational claim should report both measures with the random-init baseline. This reading is mine.
- Fig. 13's left panel appears to contradict App. D's phrase "All layers remain invariant to ℓ = L = 4": a score near 0 means no clustering. Treat "invariant" there as a slip. The likely meaning, from the sentence's own reason ("there is not signal"), is "show no clustering".

### Gaps
- NeurIPS 2026 proceedings: my single unredirected request to the listing page returned HTTP 302, so I could not check them. Publication status beyond "no evidence of publication" is open.
- Fig. 10's RHM parameters other than L and P are not stated. Fig. 14 was read from extracted axis ticks only.

## Cagnetta et al., PRX 2024 (arXiv 2307.02129): Eqs. 7, 8, 10

### Takeaway
All three equations are VERIFIED against arXiv v5. The PRX journal copy was not compared.

### Cited Findings
- **Eq. 7: VERIFIED.** "P* ≃ n_c m^L ⇔ P*/n_c ≃ d^(ln(m)/ln(s)), independently of the vocabulary size v", where P* is "the smallest training set size P such that the test error ε(P) is smaller than ε_rand/10" — [p.4](https://arxiv.org/pdf/2307.02129#page=4). It sits in the deep-network empirical section, with Figs. 3, 4, 12, 13 as support.
- **Eq. 8: VERIFIED.** "S_{k,l} = ⟨‖f_k(x) − f_k(P_l x)‖²⟩_{x,P_l} / ⟨‖f_k(x) − f_k(y)‖²⟩_{x,y}". f_k is "the sequence of activations of the k-th layer", and P_l "replaces all the level-l tuples with one of their m − 1 synonyms chosen uniformly at random" — [p.5](https://arxiv.org/pdf/2307.02129#page=5).
- **Fig. 6 parameters: VERIFIED.** L=3, s=2, n_c=m=v=8, and "invariance to level-l exchanges is obtained from layer k = l + 1" — [p.6](https://arxiv.org/pdf/2307.02129#page=6).
- **Eq. 10: VERIFIED.** "For large m, n_c and P, P_c = n_c m^L, which coincides with the empirical sample complexity of deep networks" — [p.7](https://arxiv.org/pdf/2307.02129#page=7). It balances signal (m^L/(mv))^(−1/2) against noise [P/(n_c m v)]^(−1/2) (algebra checked). The heuristic label is also VERIFIED: "a simplified argument based on an additional independence assumption" — [p.6](https://arxiv.org/pdf/2307.02129#page=6).
- **arXiv metadata:** v5, 3 Jul 2024, journal_ref "Phys. Rev. X 14, 031001 (2024)" — [API](https://export.arxiv.org/api/query?id_list=2307.02129).
- **Lazy caveat: VERIFIED.** Fig. 14 is for "the maximal case n_c = v, m = v^(s−1)". This settles half of the `random_hierarchy_model.md` line 58 gap, for the main text — [p.4](https://arxiv.org/pdf/2307.02129#page=4).

### Inferences
- PRX defines S on activations f_k; ICML defines it on pre-activations a_k. The definitions also differ in the level index (Correction 10). They should be called related measures.

### Gaps
- The APS version's pagination and text were not compared. The appendix caption for Fig. 14 was not rechecked.

## Bigger Isn't Always Memorizing (arXiv 2505.16959): the sentence and the copy fraction

### Takeaway
The sentence is VERIFIED, for v=16, m=3, L=5, s=2, P=1,024. "Fraction of copies" has a stated threshold only for images. On the RHM the paper says "Hamming distance ... determining if they are copies or not" and gives no threshold.

### Cited Findings
- **Sentence: VERIFIED.** "This behavior implies that the model never learns the rules at the deeper levels ℓ = 4, 5 since the number of training data is smaller than the sample complexity, and generates data with global consistency only when it starts memorizing." — [Sec. 4.4, p.9](https://arxiv.org/pdf/2505.16959#page=9); same in the [thesis p.113](https://arxiv.org/pdf/2510.06106#page=129).
- **Parameters: gap closed.** "Data for RHM parameters v = 16, m = 3, L = 5, s = 2", P = 1,024, P*_L ∼ 10⁴ — [Fig. 6, p.9](https://arxiv.org/pdf/2505.16959#page=9). 16·3⁶ = 11,664 is consistent (my check).
- **Scope:** errors at "ℓ ≤ 3 decrease at training times depending on ℓ"; "for ℓ > 3, the fractions of errors reach small values only at the onset of memorization" — [p.9](https://arxiv.org/pdf/2505.16959#page=9).
- **Copy fraction, by modality.**

| Data | Definition, verbatim | Threshold | Source |
|---|---|---|---|
| Images (CIFAR-10) | "classify x as a copy if ‖x − x′‖₂/‖x − x″‖₂ < 1/3", with x′, x″ the two closest training images in Euclidean distance | 1/3 | [p.4](https://arxiv.org/pdf/2505.16959#page=4) |
| Text (text8) | "the Hamming distance between each generated text sample and the closest training set text, averaged over the generations and divided by the sequence length. This metric captures the fraction of exact token matches" | None stated | [p.5](https://arxiv.org/pdf/2505.16959#page=5) |
| Stable Diffusion | Cosine similarity of SSCD features to the nearest training image, "following the protocol of [17]" | Not in the passage read | [p.17](https://arxiv.org/pdf/2505.16959#page=17) |
| **RHM** | "generate 1,024 data points with the diffusion model and measure their Hamming distance with the training data, determining if they are copies or not" | **NOT FOUND** | [p.8](https://arxiv.org/pdf/2505.16959#page=8) |

- A search of the whole text for "copy", "copies" and "Hamming" finds no further RHM definition, including in App. A's RHM section and App. D — [pp.16–18](https://arxiv.org/pdf/2505.16959#page=16).
- **Novelty is stated here,** which ICML lacks. Generalization "corresponds to the model producing novel samples that are consistent with, but not identical to, the training data" — [p.2](https://arxiv.org/pdf/2505.16959#page=2). On the RHM: "For P > 4,096, instead, the fraction of errors decreases before the onset of memorization: the diffusion model is generating valid data that do not belong to the training set, and it is therefore generalizing." — [p.8](https://arxiv.org/pdf/2505.16959#page=8).
- **Rule coverage** (App. D): at P=16,384, v=16, m=4, L=3, s=2, τ=32,768, the model "samples all RHM rules with a mean occurrence that is approximately uniform", and rules are "sampled approximately independently" — [p.18](https://arxiv.org/pdf/2505.16959#page=18).
- **Other VERIFIED items:** P* ≈ 4,096 for v=16, m=4, L=3, s=2; "For P < 4,096 ..."; fraction of errors < 15% for τ > τ* — [p.8](https://arxiv.org/pdf/2505.16959#page=8). Copy rates from 1% to 100% — [p.4](https://arxiv.org/pdf/2505.16959#page=4).
- **Optimizer:** Adam on the RHM — [p.8](https://arxiv.org/pdf/2505.16959#page=8).
- **Status:** the PDF says "Preprint." and is stamped v2, 1 Sep 2025 — [p.1](https://arxiv.org/pdf/2505.16959#page=1); the API has no journal_ref — [API](https://export.arxiv.org/api/query?id_list=2505.16959). The thesis cites it as an ICML 2025 workshop paper — [p.103](https://arxiv.org/pdf/2510.06106#page=119). The earlier TMLR claim rests on the author's page.

### Inferences
- The text-copy sentence is internally inconsistent: a Hamming distance counts mismatches, and the sentence says it captures matches. The plotted axis is "fraction of copies", so its exact computation is not recoverable from the text.
- On the RHM, a binary copy decision without a threshold most plausibly means an exact match or a near-exact one. For a generalization claim, near copies would matter, and the paper does not settle this. An RHM string that recombines two training strings' subtrees passes as a non-copy.
- The P=4,096 case is in the "generalizing models" list and on the boundary of "P < 4,096" and "P > 4,096". With the (1−f)⁻¹ prefactor the threshold would be about 5,461 (my calculation), so P* ≈ 4,096 drops the prefactor.

### Gaps
- The RHM copy threshold would need the authors' code. The paper's text gives no repository for this experiment that I noticed, and I did not search further.
