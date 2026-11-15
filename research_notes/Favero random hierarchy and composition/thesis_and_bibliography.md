# Alessandro Favero: identity, PhD thesis, and bibliography (as of 2026-11-06)

Reading status, applying to the whole file:
- **Read through a summarising fetch tool:** thesis front matter, Chapter 1, Chapters 2 and 3, and Chapter 4 up to the start of its reconstruction results, from the arXiv HTML. Part of Chapter 5 from ar5iv. All formulas below come from that tool's output, not from my own reading of raw text, so check any formula against the PDF before quoting it.
- **Known only from the thesis's own Section 1.5 summary and table of contents:** Chapters 6, 7, 8 and 9 and the end of Chapter 5. I did not read the bodies of these chapters.
- **Not opened:** the thesis PDF. It is 29 MB, above the fetch tool's 10 MB limit.
- **Listing or abstract only:** every paper in the bibliography. The two 2026 abstracts were read on arXiv; the others were checked against listings only.

## Who is this researcher, and could another person be meant?

### Takeaway
The person is almost certainly **Alessandro Favero**, with an "e". He did his PhD at EPFL with Matthieu Wyart and Pascal Frossard and is now a Physics-AI Fellow at Cambridge. I found no ML researcher named "Alessandro Favaro" in one targeted search, which is weak evidence of absence rather than proof.

### Cited Findings
- Current posts: "inaugural Physics-AI Fellow at the University of Cambridge" and "Sansom Research Associate at Emmanuel College", and a member of the interpretability team at Polymathic AI. — [Personal homepage](https://alesfav.github.io/)
- Cambridge lists him as a Physics-AI Fellow in DAMTP, in the Relativity and Gravitation group. — [DAMTP profile](https://www.damtp.cam.ac.uk/person/af940)
- Career path: physics studies in Turin, Trieste and Paris; PhD at EPFL with Wyart and Frossard; a summer at Amazon's AI Labs in Stefano Soatto's group on multimodal LLMs; then Cambridge. — [Personal homepage](https://alesfav.github.io/)
- His 2020 master's thesis, "Spectral analysis of infinitely wide convolutional neural networks", is listed as Sorbonne Université and Politecnico di Torino. — [Publications page](https://alesfav.github.io/publications/)
- Award: the G-Research EPFL PhD prize in maths and data science for the thesis, in the news item dated 25 Sep 2025. — [Personal homepage](https://alesfav.github.io/)
- Identifiers: [homepage](https://alesfav.github.io/), [publications](https://alesfav.github.io/publications/), [Google Scholar E0sOorcAAAAJ](https://scholar.google.com/citations?user=E0sOorcAAAAJ&hl=en), [arXiv API author query](https://export.arxiv.org/api/query?search_query=au:%22Favero,%20Alessandro%22&sortBy=submittedDate&sortOrder=descending&max_results=50), [X](https://x.com/alesfav), [Bluesky](https://bsky.app/profile/alessandrofavero.bsky.social).
- A search for the exact name "Alessandro Favaro" surfaced Alessandro Favero plus similarly named but different people: Stefano Favaro (statistics, Torino), Alberto Favaro (data scientist at Faculty) and Paolo Favaro (computer vision, Bern). — [Stefano Favaro](https://carloalberto.org/person/stefano-favaro); [Alberto Favaro](https://conferences.oreilly.com/strata/strata-eu-2019/public/schedule/speaker/349681.html); [Paolo Favaro](https://www.bme.master.unibe.ch/about_us/faculty/prof_dr_favaro_paolo/index_eng.html)

### Inferences
- The topics (RHM, diffusion, composition) match Favero's publication list exactly, so "Favaro" is very likely a misspelling.
- None of the three Favaros above showed work on these topics in the results I saw. I did not review their publication lists.
- The arXiv API query matches on the name string, so it could include a namesake. All 18 entries look like one person: the co-authors and themes are continuous, and the 2026 forgetting paper fits his Cambridge and Polymathic position.

### Gaps
- I found no start date for the Cambridge fellowship.
- I found no CV with degree dates. The homepage gives none.

## PhD thesis: metadata and examiners

### Takeaway
*The Physics of Data and Tasks: Theories of Locality and Compositionality in Deep Learning*, EPFL thesis no. 11462, PhD in Physics, publicly defended 26 September 2025. DOI 10.5075/epfl-thesis-11462; arXiv 2510.06106.

### Cited Findings
- Title, author, and degree in Physics at EPFL; presented 26 September 2025; copyright September 2025. — [arXiv HTML of thesis](https://arxiv.org/html/2510.06106v1)
- Jury: Prof. C. G. Theiler (president), Prof. M. Wyart and Prof. P. Frossard (thesis directors), Prof. E. Vanden-Eijnden, Prof. S. Ganguli and Prof. E. Abbé (examiners). — [arXiv HTML of thesis](https://arxiv.org/html/2510.06106v1)
- Thesis number 11462; public defense 26 Sep 2025, 17:00, room CE 1 5 and Zoom; physics doctoral program. — [EPFL Memento](https://memento.epfl.ch/event/the-physics-of-data-and-tasks-theories-of-locali-2)
- DOI 10.5075/epfl-thesis-11462 and arXiv DOI 10.48550/arXiv.2510.06106; v1 submitted 7 Oct 2025; size 29,017 KB; primary class cs.LG, cross-listed in cond-mat.dis-nn, cond-mat.stat-mech and stat.ML. — [arXiv abstract](https://arxiv.org/abs/2510.06106)
- Infoscience handle, as given on the thesis title page and the author's publications page. — [Infoscience](https://infoscience.epfl.ch/handle/20.500.14299/254241) (fetch returned HTTP 429, see Gaps)
- EPFL's lab news says he "successfully passed his PhD exam on August 25th, 2025". — [EPFL actu](https://actu.epfl.ch/news/new-phd-thesis-57)

### Inferences
- 25 August (exam) and 26 September (public defense) most likely reflect EPFL's separate private exam and public defense. No source I read states this, so treat it as my reading.

### Gaps
- The Infoscience record returned HTTP 429 with Retry-After 900. I did not retry or work around it, so its metadata was not read directly. The jury comes from the thesis itself, and the number and DOI from Memento and arXiv.

## PhD thesis: structure and main claims

### Takeaway
The thesis follows one thread, locality and compositionality, across three levels: infinite-width CNNs (kernel regime), diffusion models on hierarchical grammars, and weight space of pretrained models. Its overall claim is that high-dimensional learnability requires latent structure, and each part identifies which structure helps and what it costs in samples.

### Cited Findings: overall framing
All from the [arXiv HTML of the thesis](https://arxiv.org/html/2510.06106v1) unless another source is given.
- Central puzzle: deep networks learn high-dimensional tasks that should be statistically intractable, so learnable data must have latent structure. — [arXiv abstract](https://arxiv.org/abs/2510.06106)
- Generalization is modelled as a power law, ℰ(P) ∼ P^(−β). Lipschitz regression gives β ∝ 1/d, which is the curse of dimensionality (Sec. 1.2).
- Stated structure: data are compositional, with objects built from parts and parts from sub-parts (Sec. 1.2).
- Sec. 1.4 sets up modern pretraining and task vectors, and asks whether task arithmetic is a consequence of near-linear (NTK-like) behaviour and whether modularity emerges during pretraining.

### Cited Findings: chapter by chapter
Table of contents and Sec. 1.5 are from the [arXiv HTML](https://arxiv.org/html/2510.06106v1).

**Part I. Chapter 1, Introduction.** Deep learning basics; generalization (classical capacity bounds, NTK and spectral bias, feature learning in the mean-field limit, the role of data structure); deep generative modelling (score-based SDEs, DDPM, discrete diffusion for text); deep learning today; thesis structure and results (1.5). *Read.*

**Part II. Statistical mechanics of convolutional networks at infinite width.**

*Chapter 2, Locality Defeats the Curse of Dimensionality.* *Read.*
- Setting: teacher-student kernel regression, with a Gaussian random field target, on uniform data on a torus.
- **Theorem 9.1:** for a t-dimensional teacher constituent with non-analyticity exponent α_t and an s-dimensional student with nested patches, ℬ(P) ∼ P^(−β) with β = α_t/s. This carries over to ℰ(P) under the condition α_t ≤ 2(α_s + s).
- β does not depend on the input dimension d. Locality sets the exponent, while translational invariance affects only prefactors. If the teacher's patches are not contained in the student's, the error does not vanish.
- **Theorem 11.1 and Corollary 11.1.1:** with ridge λ ∼ P^(−γ) and 0 < γ < 1/2, β = γ α_t/(α_s + s). This also stays non-zero as d→∞, and it depends on s rather than t.
- Definitions here: *locality* means the constituent kernel acts on s-dimensional patches; *compositionality* means the target is a sum of patch-level functions (Eq. 20). This is additive composition only.
- Stated limitation: the analysis ignores hierarchical composition of real data, which motivates Chapter 3.
- Experiments: Laplacian kernels on [0,1]^9 confirm β = 1/s when α_t = α_s = 1.

*Chapter 3, The Role of Depth and Spatial Adaptivity.* *Read.*
- Setting: hierarchical CNNs with non-overlapping patches and no pooling. Effective dimension of a meta-patch: d_eff(1) = s₁ − 1 and d_eff(l) = (s₁ − 1)∏_{l'=2}^{l} s_{l'}.
- **Theorem 15.1:** NTK and RFK eigenvalues split into one sector per layer, Λ_k^(l) ≈ C_{2,l} k^(−2ν − d_eff(l)), with ν_NTK = 1/2 and ν_RFK = 3/2. Fully connected kernels have one global sector.
- Rates: Corollary 17.0.1 gives a rigorous-style exponent, and a spectral-bias ansatz gives β = [2m/(2ν+d_eff(l))]·[(2ν+d_eff(L))/d_eff(L)] (Eq. 66). A fully connected kernel gives β = 2m/(2m+p), which vanishes as p grows.
- **Lemma 17.1:** outputs of random infinitely wide hierarchical networks are cursed, with β = 3/d_eff(L).
- Conclusion: deep CNNs adapt to the spatial scale of the target, but hierarchical functions generated by CNNs are too rich to learn efficiently in high dimension. Sample complexity for long-range nonlinear dependencies grows exponentially with d even in a Bayes-optimal matched setting (Sec. 1.5).
- This negative result motivates a stronger compositional form, f* = g(h₁(x₁₁), h₂(x₁₂)), feature learning, and the RHM, where feature-learning networks scale polynomially in d.

**Part III. Statistical mechanics of diffusion models.**

*Chapter 4, A Phase Transition in the Diffusion Process.* *Read, except for the last part of the results and the conclusions.*
- Empirical start: ImageNet forward-backward experiments. At small inversion time only low-level features change; at a characteristic time t* the class changes abruptly; after that, low-level features of the original persist in the new image.
- RHM as defined here: L levels, vocabulary size v, branching factor s, d = s^L, m unambiguous production rules per symbol ("synonyms"), m^((d−1)/(s−1)) data per class. Key parameter: **f = m/v^(s−1)**.
- Bayes-optimal denoising is exact belief propagation on the tree, with leaf noise parameter Δ_t = (1 − ᾱ_t)/√ᾱ_t.
- Mean-field: an annealed recursion p′ = F(p). When sf > 1 only the uninformative fixed point p = 1/v is stable. When **sf < 1**, a repulsive fixed point appears between uninformative and perfect recovery, which creates the transition. The transition sharpens with depth. The authors read sf < 1 as lower-level errors being corrected by neighbouring information.

*Chapter 5, Probing Hidden Hierarchies in Data.* *Partly read, from [ar5iv](https://ar5iv.labs.arxiv.org/html/2510.06106).*
- Token-change correlation C_ij(t) and dynamical susceptibility χ(t) = Σ_ij C_ij(t) / Σ_i C_ii(t), described as measuring "the volume of the blocks of tokens that change together".
- Linearising around p* gives Δp_ℓ = (F*′)^ℓ Δp_0, and hence a correlation length in tree depth that diverges at the class transition.
- Tested with masked diffusion language models on WikiText and DDPM on ImageNet, with qualitative agreement. My access ended before the explicit exponent.

*Chapter 6, A Theory of Creativity and Compositionality.* *Sec. 1.5 summary only.*
- Learning composition rules in the feature-learning regime needs the same sample complexity as clustering features with similar contexts, and this scales polynomially with context size. A renormalization-group analogy is drawn.
- The abstract states that data are modelled with PCFGs and that sample complexity scales polynomially with data dimension.
- Sections: how diffusion models learn a grammar; theoretical analysis; natural data.

*Chapter 7, A Race Between Memorization and Generalization.* *Sec. 1.5 summary only.*
- Generalization precedes memorization, memorization time scales linearly with dataset size, and a phase diagram is built from the RHM.

**Part IV. Task localization and weight disentanglement.**

*Chapter 8, Task Compositionality in Weight Space.* *Sec. 1.5 summary and section titles only.*
- The NTK alone does not fully explain task arithmetic (a section is titled "Task arithmetic is not a consequence of linear fine-tuning").
- Proposed mechanism: **weight disentanglement**, where different weight directions produce localized, non-interfering changes in function. This connects to spatial localization of NTK eigenfunctions, and the structure emerges during pretraining rather than at initialization.

**Part V. Chapter 9, Conclusions.** *Section titles only.* Key findings and their synthesis; comparison with other theoretical frameworks; limitations and future directions; concluding remarks.

**Part VI.** Appendices (Chapters 10 to 16) for Chapters 2 to 8, plus a glossary. *Only the opening parts of the list were seen; bodies not read.*

### Cited Findings: how the thesis treats each theme
- **Locality:** a constraint on what a kernel's constituent sees (s-dimensional patches). It is the source of the d-independent exponent, and it matters more than weight sharing (Chapter 2). It is then generalized to *task localization* in function space (Part IV, Sec. 1.5). — [arXiv HTML](https://arxiv.org/html/2510.06106v1)
- **Compositionality:** additive sums of local functions in Chapter 2, tree-structured grammars in Part III, and task vectors that add in weight space in Part IV. — [arXiv HTML](https://arxiv.org/html/2510.06106v1)
- **Hierarchy:** kernel sectors matching architecture depth (Chapter 3), then the RHM and PCFG tree with depth-dependent transitions (Chapters 4 to 7). — [arXiv HTML](https://arxiv.org/html/2510.06106v1)
- **Diffusion:** score learning treated as regression; denoising as BP on the latent tree; a class-level transition with correlated token blocks; composition rules learned at polynomial sample cost; generalization arriving before memorization. — [arXiv HTML](https://arxiv.org/html/2510.06106v1)

### Inferences
- Likely chapter-to-paper mapping, from matching titles and themes: Ch. 2 = 2106.08619; Ch. 3 = 2208.01003; Ch. 4 = 2402.16991; Ch. 5 = 2410.13770; Ch. 6 = 2502.12089; Ch. 7 = 2505.16959; Ch. 8 = 2305.12827. The visible thesis text has no such list; its acknowledgements only say the core comes from work with Francesco, Antonio, Noam and Guille.
- 2307.02129 (the RHM paper in PRX) seems to be a source for the RHM but not a chapter of its own. It is cited as later work in the Sec. 1.5 discussion of Chapter 3.
- The arc of the thesis is a progression in the meaning of "compositional": additive and local (learnable by kernels), then hierarchical with long-range nonlinearity (cursed for kernels), then grammar with synonyms (learnable with feature learning).
- The thesis's theorems sit mainly in Part II. Going by what I could read, Part III relies more on mean-field arguments and numerics. I could not confirm this for Chapters 6 and 7.

### Gaps
- **Chapters 5 (end), 6, 7, 8 and 9 were not read in the thesis.** Both HTML renderings were cut off by the fetch tool, at about 312k characters on arXiv (in Ch. 4) and 282k on ar5iv (in Ch. 5). The PDF exceeds the tool's size limit. Downloading it with a shell command was outside this task's constraint of web tools only.
- Consequently the "comparison with other theoretical frameworks" and "limitations and future directions" sections in Chapter 9 are unread. These are probably the most useful pages for this project. To read them, someone should open pp. near the end of Chapter 9 in the [arXiv PDF](https://arxiv.org/pdf/2510.06106).
- The papers behind Chapters 6 and 7 are covered by another researcher, and may fill in the theory.

## Bibliography, about 2021 to November 2026, newest first

### Takeaway
There are 18 arXiv entries plus several non-arXiv items. The most recent on these topics is **"Learn from your own latents and not from tokens" (May 2026)**, a sample-complexity theory on a PCFG. The newest of all is a forgetting paper from October 2026, which is off-topic for composition.

### Cited Findings
Sources: [arXiv API query](https://export.arxiv.org/api/query?search_query=au:%22Favero,%20Alessandro%22&sortBy=submittedDate&sortOrder=descending&max_results=50), [publications page](https://alesfav.github.io/publications/), [Google Scholar, sorted by date](https://scholar.google.com/citations?user=E0sOorcAAAAJ&hl=en&sortby=pubdate&view_op=list_works&pagesize=100), [DAMTP](https://www.damtp.cam.ac.uk/person/af940). Items are grouped by publication year; dates shown are the first arXiv submission, so a paper can sit in a later year than its date. An asterisk marks co-first authors as listed by the author.

Tags: **[RHM]** uses or defines the RHM per a source I saw. **[GRAMMAR]** hierarchical grammar or language model where I did not confirm the RHM by name. **[DIFF]** diffusion, composition or creativity. **[OTHER]**.

**2026**
1. "Frozen in Time: How a Physics Foundation Model Forgets Under Sim-to-Real Adaptation." H. Mian, M. Cranmer, A. Favero, P. Mukhopadhyay. arXiv:[2610.40857](https://arxiv.org/abs/2610.40857), 29 Oct 2026; accepted at the Sim2Science workshop, NeurIPS 2026. **[OTHER]** (forgetting in Walrus-1.3B; replaying 1% of simulated data recovers most lost performance.)
2. "Learn from your own latents and not from tokens: A sample-complexity theory." D. J. Korchinski*, A. Favero*, M. Wyart. arXiv:[2605.27734](https://arxiv.org/abs/2605.27734), 26 May 2026. **[GRAMMAR]** (Abstract: on a tractable PCFG, supervised and token-level self-supervised learning need samples exponential in hidden-tree depth, while latent prediction is about constant in depth up to log factors. Argues that explicit stacking such as H-JEPA may be largely unnecessary.) Venue conflict, see below.
3. "Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models." A. Favero, A. Sclocchi, M. Wyart. arXiv:[2505.16959](https://arxiv.org/abs/2505.16959), 22 May 2025; *TMLR* 2026, with J2C certification. **[DIFF]** **[RHM]** (the thesis says Ch. 7's phase diagram uses the RHM).

**2025**
4. "Task Addition and Weight Disentanglement in Closed-Vocabulary Models." A. Hazimeh*, A. Favero*, P. Frossard. arXiv:[2511.14569](https://arxiv.org/abs/2511.14569), 18 Nov 2025; earlier at the ICML 2024 Efficient Systems for Foundation Models workshop ([OpenReview](https://openreview.net/pdf?id=t6GZM7a3Nc)). **[OTHER]**
5. "Backdoor Unlearning by Linear Task Decomposition." A. Abdelraheem*, A. Favero*, G. Bovet, P. Frossard. arXiv:[2510.14845](https://arxiv.org/abs/2510.14845), 16 Oct 2025; ICML 2025 Workshop on Machine Unlearning for Generative AI, per the publications page. **[OTHER]**
6. "The Physics of Data and Tasks: Theories of Locality and Compositionality in Deep Learning." A. Favero. PhD thesis, EPFL. arXiv:[2510.06106](https://arxiv.org/abs/2510.06106), 7 Oct 2025. **[RHM]** **[DIFF]**
7. "MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs." K. Wang, Y. Qin, N. Dimitriadis, A. Favero, P. Frossard. arXiv:[2506.07899](https://arxiv.org/abs/2506.07899), 9 Jun 2025; NeurIPS 2025. **[OTHER]**
8. "Scaling Laws and Representation Learning in Simple Hierarchical Languages: Transformers vs. Convolutional Architectures." F. Cagnetta, A. Favero, A. Sclocchi, M. Wyart. arXiv:[2505.07070](https://arxiv.org/abs/2505.07070), 11 May 2025; *Physical Review E* 112, 065312 (2025); "Featured in Physics" and PRE ML Spotlight per the author. **[GRAMMAR]**
9. "Compositional Generalization and Creativity in Language Diffusion Models." A. Favero, A. Sclocchi, F. Cagnetta, P. Frossard, M. Wyart. XLLM @ ACL 2025 (Scholar listing only; no separate arXiv id found). **[DIFF]**
10. "How Compositional Generalization and Creativity Improve as Diffusion Models are Trained." A. Favero*, A. Sclocchi*, F. Cagnetta, P. Frossard, M. Wyart. arXiv:[2502.12089](https://arxiv.org/abs/2502.12089), 17 Feb 2025; ICML 2025, [PMLR 267](https://proceedings.mlr.press/v267/favero25a.html). **[DIFF]** **[GRAMMAR]** (PCFGs).
11. "LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging." K. Wang, N. Dimitriadis, A. Favero, G. Ortiz-Jimenez, F. Fleuret, P. Frossard. arXiv:[2410.17146](https://arxiv.org/abs/2410.17146), 22 Oct 2024; ICLR 2025. **[OTHER]**
12. "Probing the Latent Hierarchical Structure of Data via Diffusion Models." A. Sclocchi*, A. Favero*, N. I. Levi*, M. Wyart. arXiv:[2410.13770](https://arxiv.org/abs/2410.13770), 17 Oct 2024; ICLR 2025; also *JSTAT* 2025, 084005. **[DIFF]** **[RHM]**
13. "A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data." A. Sclocchi, A. Favero, M. Wyart. arXiv:[2402.16991](https://arxiv.org/abs/2402.16991), 26 Feb 2024; *PNAS* 122(1), e2408799121 (2025). **[DIFF]** **[RHM]**

**2024**
14. "Unraveling the Latent Hierarchical Structure of Language and Images via Diffusion Models." A. Sclocchi, N. I. Levi, A. Favero, M. Wyart. NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning (Scholar listing only). **[DIFF]**
15. "Computational complexity of deep learning: fundamental limitations and empirical phenomena." B. Barak, A. Carrell, A. Favero, W. Li, L. Stephan, A. Zlokapa. *JSTAT* 2024(10), 104008 ([IOP PDF](https://iopscience.iop.org/article/10.1088/1742-5468/ad3a5b/pdf)); not in the arXiv query results. **[OTHER]**
16. "Multi-Modal Hallucination Control by Visual Information Grounding." A. Favero, L. Zancato, M. Trager, S. Choudhary, P. Perera, A. Achille, A. Swaminathan, S. Soatto. arXiv:[2403.14003](https://arxiv.org/abs/2403.14003), 20 Mar 2024; CVPR 2024. **[OTHER]**
17. "How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model." F. Cagnetta, L. Petrini, U. M. Tomasini, A. Favero, M. Wyart. arXiv:[2307.02129](https://arxiv.org/abs/2307.02129), 5 Jul 2023; *Phys. Rev. X* 14, 031001 (2024). **[RHM]**

**2023**
18. "Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models." G. Ortiz-Jimenez*, A. Favero*, P. Frossard. arXiv:[2305.12827](https://arxiv.org/abs/2305.12827), 22 May 2023; NeurIPS 2023, oral (top 0.54%). **[OTHER]** (task composition in weight space.)
19. "What Can Be Learnt With Wide Convolutional Neural Networks?" F. Cagnetta*, A. Favero*, M. Wyart. arXiv:[2208.01003](https://arxiv.org/abs/2208.01003), 1 Aug 2022; ICML 2023, PMLR 202, 3347–3379; also *JSTAT* 2024(10), 104020. **[OTHER]** (hierarchical kernels; precursor to the RHM.)
20. "Statistical Mechanics of Infinitely-Wide Convolutional Networks." A. Favero, F. Cagnetta, M. Wyart. *Bulletin of the American Physical Society*, 2023 (Scholar listing only; presumably a meeting abstract). **[OTHER]**
21. "Diffeomorphisms invariance is a proxy of performance in deep neural networks." L. Petrini, A. Favero, M. Geiger, M. Wyart. *Bulletin of the American Physical Society*, 2023 (Scholar listing only). **[OTHER]**

**2021**
22. "Locality defeats the curse of dimensionality in convolutional teacher-student scenarios." A. Favero*, F. Cagnetta*, M. Wyart. arXiv:[2106.08619](https://arxiv.org/abs/2106.08619), 16 Jun 2021; NeurIPS 2021, 34, 9456–9467; also *JSTAT* 2022(11), 114012. **[OTHER]**
23. "Relative stability toward diffeomorphisms indicates performance in deep nets." L. Petrini, A. Favero, M. Geiger, M. Wyart. arXiv:[2105.02468](https://arxiv.org/abs/2105.02468), 6 May 2021; NeurIPS 2021, 34, 8727–8739; also *JSTAT* 2022(11), 114013. **[OTHER]**

Context, before the requested range: "Spectral analysis of infinitely wide convolutional neural networks." A. Favero. Master's thesis, 2020 ([PDF](https://webthesis.biblio.polito.it/15963/1/tesi.pdf)).

### Cited Findings: 2025-2026 talks
All from the [homepage](https://alesfav.github.io/) except where noted.
- May 2026: speaker at the first FLARE workshop, EPFL.
- 5 May 2026: "Editing AI Minds", Infosys-Cambridge AI Industry Symposium.
- July 2026 (listed as upcoming on the page): Machine Learning Physics, Okinawa; Youth in High Dimensions, ICTP Trieste.
- Nov 2025: diffusion models seminar at JHU; talk at Flatiron CCM; Simons Collaboration on the Physics of Learning kickoff, Stanford.
- April 2025: "Creativity by Compositionality in Generative Diffusion Models", Perimeter Institute Theory + AI workshop. — [PIRSA 25040088](https://pirsa.org/25040088); dated 9 April on PIRSA, and in a homepage news item dated 8 April.

### Cited Findings: source conflicts
- **Item 2's venue:** Google Scholar lists "Advances in Neural Information Processing Systems 39, 2026", while the [homepage](https://alesfav.github.io/publications/) says arXiv preprint and [arXiv](https://arxiv.org/abs/2605.27734) has no journal reference. Treat it as a preprint unless the NeurIPS 2026 proceedings confirm it.
- **Item 3's venue:** the homepage says TMLR with J2C certification; [Scholar](https://scholar.google.com/citations?user=E0sOorcAAAAJ&hl=en&sortby=pubdate&view_op=list_works&pagesize=100) says "TMLR (NeurIPS 2026 Journal Track)". These are probably compatible, but I did not check the NeurIPS site.
- **Item 5's title:** "Backdoor unlearning through linear task decomposition" on the homepage, "by" on arXiv. arXiv is used above.
- **Item 12's authorship:** the JSTAT and ICLR records appear to be the same paper. Scholar lists a third, similar workshop title, item 14, with a different author order.
- **Year for item 7:** Scholar shows 2026, the proceedings volume is NeurIPS 2025 (vol. 38), and arXiv records acceptance at NeurIPS 2025.

### Inferences
- On composition theory, the recent line is 2402.16991, 2410.13770, 2502.12089, 2505.16959, 2505.07070, then 2605.27734. The last moves the grammar framework from generative diffusion to representation learning by latent prediction.
- Since moving to Cambridge, his listed work has leaned toward model editing and forgetting (the "Editing AI Minds" talk, Polymathic interpretability, the Walrus paper), alongside the Wyart collaboration. This is a small sample and only suggestive.
- The arXiv API result does not contain "How Deep Networks Learn Sparse and Hierarchical Data: the Sparse Random Hierarchy Model" (2404.10727), which came up in a search. It does not appear to be a Favero paper; do not credit him for it.

### Gaps
- **DBLP:** the API returned a bot-protection denial (Anubis), which I did not try to get around. So DBLP was not used.
- **Semantic Scholar and OpenReview** were not queried. OpenReview appears only through the link on the publications page. Workshop items may be missing.
- I found no 2026 talk abstract or slides on composition or diffusion; only titles and news lines. The homepage news does not confirm that the July 2026 talks happened.
- I found no 2026 preprint on diffusion and composition by him in arXiv, the homepage, or Scholar. The date of the homepage's last update is unknown, and it omits the October 2026 paper, so it may lag.
- Scholar's listing was fetched as a public page once; DAMTP's list showed only page 1 of 3.
- Tags [RHM] and [GRAMMAR] for items 2, 8 and 10 rest on abstracts or listings. Whether each uses the RHM specifically should be taken from the researchers reading those papers.
