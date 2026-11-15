# The Random Hierarchy Model (RHM) and the theory built on it

**How this was read.** Sources are arXiv HTML full texts, read through a fetch tool that summarises with a small model. No PDF could be extracted. Equations below are therefore second-hand transcriptions with their equation numbers. Check them against the paper before quoting them in print. I re-derived the algebra where marked "(algebra checked)". The APS page for the PRX paper served only the abstract, outline and references to this tool, so all PRX content comes from arXiv v5. Appendix-level reading was partial for several papers, as noted under each Gaps section.

**Author spelling.** Favero is correct. Alessandro Favero is a co-author of the PRX paper, but the first author is Cagnetta.

---

## 1. Exact generative model, sample complexity, and how hierarchy learning is measured (Cagnetta et al., PRX 2024)

### Takeaway
The RHM is a fixed-tree, unambiguous, non-recursive random PCFG classification task. Its central result, P* ≃ n_c m^L for deep feature-learning networks against P ~ P_max (exponential in d) for two-layer FC nets, is an **empirical scaling with a heuristic correlation argument**. The only "proof" is a one-GD-step calculation on a simplified two-layer net. It is not a theorem about deep nets.

### Cited Findings

**Definition** (all from [arXiv:2307.02129v5](https://arxiv.org/html/2307.02129v5), Sec. I; journal ref. [PRX 14, 031001](https://arxiv.org/abs/2307.02129))
- Parameters: n_c classes; L levels; disjoint vocabularies V_ℓ of size v_ℓ = v at each level; arity s; m rules per symbol. Input dimension d = s^L. — [Source](https://ar5iv.labs.arxiv.org/html/2307.02129)
- Rules: Eq. (1) α ↦ μ_1^(L),…,μ_s^(L), with m such rules per class. Eq. (2) μ^(ℓ) ↦ μ_1^(ℓ−1),…,μ_s^(ℓ−1), with m such rules per feature, applied from ℓ=L to 1. — [Source](https://ar5iv.labs.arxiv.org/html/2307.02129)
- Non-ambiguity: m v_{ℓ+1} ≤ v_ℓ^s, with v_{L+1} ≡ n_c. Each s-tuple has at most one parent, so the bottom-up map is a function. Eq. (4) writes the label as a composition of L functions g_ℓ of s variables. — [Source](https://arxiv.org/html/2307.02129v5)
- Data count per class, Eq. (3): m·m^s·…·m^{s^{L−1}} = m^{(d−1)/(s−1)}. Eq. (6): P_max = n_c m^{(d−1)/(s−1)}. Maximal case: n_c = v, m = v^{s−1}, giving P_max = v^{s^L} (all possible inputs). (algebra checked) — [Source](https://arxiv.org/html/2307.02129v5)
- Rule sampling: uniformly at random over assignments of m s-tuples to each of the v parents. The code confirms the procedure: enumerate all s-tuples, shuffle, truncate to m·(number of parents), reshape to (−1, m, s). This is sampling without replacement, partitioned among parents. — [Paper](https://arxiv.org/html/2307.02129v5); [code](https://raw.githubusercontent.com/pcsl-epfl/hierarchy-learning/master/datasets/hierarchical.py)
- Synonyms: the m tuples sharing a parent. Rule probabilities are uniform. — [Source](https://arxiv.org/html/2307.02129v5)
- Encoding: one-hot, Eq. (5), giving a d×v input, whitened over channels (App. A.1). — [Source](https://ar5iv.labs.arxiv.org/html/2307.02129)
- The PRX paper uses neither "m_max" nor f as a named fill fraction. In it, f_j(α|μ) is a conditional class probability (Eq. 9) and f is also the CNN filter size (App. A.2). The ratio f = m/v^{s−1} is notation from later papers. — [Source](https://ar5iv.labs.arxiv.org/html/2307.02129); [later usage](https://arxiv.org/html/2605.27734v1)
- Uncorrelated variant (Sec. III.4): rules are modified so that a tuple at a given patch and level belongs to each class with probability 1/n_c. This generalises parity (m=v=n_c=2). Deep CNNs stay near ε_rand even for P/P_max > 0.9 (Fig. 9). The correlations are thus necessary in practice, not just convenient. — [Source](https://ar5iv.labs.arxiv.org/html/2307.02129)

**Sample complexity**
- Shallow (two-layer FC, shown in Fig. 2 for L=s=2, v=n_c=m, the maximal case): the data needed for test error ≤ 0.7 ε_rand is proportional to P_max, where ε_rand = 1 − 1/n_c. This is an **empirical** statement, and no lower bound is proved. The paper shows the curse for this maximal case only. — [Source](https://ar5iv.labs.arxiv.org/html/2307.02129); [Ren et al. note the shallow bound is heuristic](https://arxiv.org/html/2601.19756)
- Deep (CNN depth L+1, and FC depth > L), Eq. (7): P* ≃ n_c m^L ⇔ P*/n_c ≃ d^{ln m/ln s}, independent of v. P* is the smallest P with test error ≤ ε_rand/10. — [Source](https://arxiv.org/html/2307.02129v5)
- Regime: width H = (4–8)×v^s, maximal-update parametrisation. In the lazy/NTK regime deep nets keep the curse even with a matched architecture (Fig. 14). — [Source](https://arxiv.org/html/2307.02129v5)
- Deep FC nets carry an extra factor independent of v, n_c, m. The text says 2^L and the caption says s^L (Fig. 13). — [Source](https://arxiv.org/html/2307.02129v5)
- Fig. 12: L=s=2, v=256, m=23 confirms P* linear in n_c. — [Source](https://arxiv.org/html/2307.02129v5)

**Correlation argument (heuristic, Sec. III.2, App. C)**
- f_j(α|μ) = Pr{x∈α | x_j=μ} = N_j(μ;α)/N_j(μ) (Eq. 9). The signal is f_j − 1/n_c. — [Source](https://arxiv.org/html/2307.02129v5)
- Sec. III.2 is explicitly "a simplified argument based on an additional independence assumption": CLT over the m^L choices gives relative signal (m^L/(mv))^{−1/2}. Relative sampling noise is [P/(n_c m v)]^{−1/2}. Balance gives P_c = n_c m^L (Eq. 10). (algebra checked) — [Source](https://arxiv.org/html/2307.02129v5)
- App. C: recursive moments of the occurrence counts (Eqs. 28–62), concentration of U as v grows (Eq. 50), Var[f] → v/(n_c² m^L) (Eq. 73), f̂ ≈ (1/n_c)(1 + √(v/m^L) ξ_RHM + √(v n_c/P) ξ_P) (Eq. 77–78), crossover P_c = n_c m^L (Eq. 79; large-n_c,m form Eq. 25). The Gaussian limit of the hypergeometric (Eqs. 74–75) needs P ≫ 1 and P_max, N ≫ P. — [Source](https://arxiv.org/html/2307.02129v5)
- One-step analysis (Sec. III.3, Eqs. 11–13): two-layer ReLU net on the first patch only, x_1 one-hot of dimension v^s ("equivalent to an orthogonalization of the input points"), a_h fixed i.i.d. Gaussian, w_h initialised to ones, W→∞ so F^(0)=0, cross-entropy. Eq. (13): Δf_h(x_1) = Σ_α a_{h,α}( N̂_1(μ(x_1);α)/P − (1/n_c) N̂_1(μ)/P ). Empirical counts converge to true counts, which are synonym-invariant. So representations become invariant, and the hidden-weight rank tends to v. Fig. 8: S_{1,1} ≈ 1 for P≪P_c and ≈ 0 for P≫P_c. — [Source](https://arxiv.org/html/2307.02129v5)
- The Intro's claim: "We prove for a simplified architecture trained with gradient descent that (a) and (b) must indeed coincide", where (a) is the number of data allowing an invariant representation and (b) is the training-set size at which low-level feature/class-label correlations become detectable. The scope is one step, first composition rule. Weight sharing is argued not to change this in the one-step setting. No theory is given for layers beyond the first. — [Source](https://arxiv.org/html/2307.02129v5)
- App. D (maximal case, heuristic): layerwise one GD step + k-means works at P ≈ P_c/√n_c = √n_c m^L = v^{1/2+L(s−1)} (Eqs. 80–89). Fig. 10 (L=3, s=2) collapses with v^{L+1/2}, a √v advantage over end-to-end P* = v^{L+1}. (algebra checked) — [Source](https://arxiv.org/html/2307.02129v5)

**How hierarchy learning is measured**
- Primary: synonymic sensitivity, Eq. (8): S_{k,l} = ⟨‖f_k(x) − f_k(P_l x)‖²⟩ / ⟨‖f_k(x) − f_k(y)‖²⟩. f_k is the activations of layer k, P_l swaps each level-l tuple for one of its m−1 synonyms, x and y are test points. — [Source](https://ar5iv.labs.arxiv.org/html/2307.02129)
- Results: S_{2,1} vs P collapses under rescaling by n_c m^L (Fig. 5). For L=3, s=2, n_c=m=v=8, invariance to level-l exchanges appears from layer k = l+1 (Fig. 6). — [Source](https://ar5iv.labs.arxiv.org/html/2307.02129)
- Secondary: intrinsic dimension decreases monotonically with depth (Fig. 11, one instance with P_max = 6,232); the authors warn finite-sample estimates can spuriously rise. — [Source](https://arxiv.org/html/2307.02129v5)
- Not used: "Gaussian-equivalent" arguments, activation kernels or kernel PCA as a representation probe, and any linear probe decoding latent symbols. The kernel appears only as the failing lazy control. — [Source](https://arxiv.org/html/2307.02129v5)
- Causal language: "We conclude that the generalization ability of a network relies on the synonymic invariance of its hidden representations" (Sec. II.1). The evidence is coincidence of P* with the drop in S. Necessity and sufficiency are not stated. — [Source](https://arxiv.org/html/2307.02129v5)

**Protocol**
- SGD, cross-entropy, batch 128 (or P if smaller), LR 10⁻¹ cosine-annealed to 10⁻² over 100 epochs, stop at train loss 10⁻³, averaged over 10 RHM instances and initialisations. CNN filter = s, non-overlapping. Sensitivity test size min(P_max−P, 1,000). — [Source](https://arxiv.org/html/2307.02129v5)
- Code: [pcsl-epfl/hierarchy-learning](https://raw.githubusercontent.com/pcsl-epfl/hierarchy-learning/master/datasets/hierarchical.py); Zenodo DOI 10.5281/zenodo.12509435 per the [PRX reference list](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.031001).

### Inferences
- P_c is a detectability threshold for the first-patch-to-label correlation. It says when the signal exists in the data, and links to what a trained deep net does only through the one-step calculation plus empirical collapse.
- S_{k,l} is a relative collapse statistic that needs ground-truth rule tables to apply P_l. S→0 means synonyms are merged relative to generic pairs. It does not by itself show that the k-th layer's state decodes the symbol identity at level l, though invariance plus a low class error makes this likely.

### Gaps
- Fig. 14's parameter was returned once as m = v^s in appendix text and as m = v^{s−1} in the main text. The main-text form is the one consistent with the definition. I could not read the caption directly.
- The source conflicts on last-layer μP scaling: Sec. II says extra H^{−1/2}, App. A.2 says extra 1/H.

---

## 2. Follow-ups

### Takeaway
The chain runs: invariance by supervised correlations (2023) → sparsity/diffeomorphisms (2024) → token prediction with level-by-level thresholds (2024–25) → diffusion and creativity (2025) → a rigorous layerwise upper bound (2026) → a rigorous latent-recovery result conditional on a clustering assumption (2026). All thresholds are of the form "learned once a correlation is detectable", and most papers **assume** this rather than derive it.

### Cited Findings

| Paper | Task | Key result (as stated) | Status |
|---|---|---|---|
| [Tomasini & Wyart, ICML 2024, 2404.10727](https://arxiv.org/html/2404.10727v2) SRHM | classification | d=(s(s₀+1))^L. LCN Eq. (3): P* ~ C₀(s,L)(s₀+1)^L n_c m^L, C₀~s^{L/2} empirically. CNN Eq. (4): P* ~ C₁(s₀+1)² n_c m^L. Eq. (5) in terms of F=(s₀+1)^{−L} (algebra checked) | Empirical + heuristic (Sec. 6); App. C first-GD-step calculation; the CNN (s₀+1)² factor is "not yet understood" |
| [Cagnetta & Wyart, NeurIPS 2024, 2406.00048](https://ar5iv.labs.arxiv.org/html/2406.00048) | last-token prediction | C̃^(ℓ) ≃ √[(1−f)/(v³m^{2ℓ−1})] (Eq. 8), noise (v²P)^{−1/2} (Eq. 9), P_ℓ = v m^{2ℓ−1}/(1−f) (Eq. 12) (algebra checked). Loss ~ P^{−α}, α = log f/(2 log m) as printed in the fetched summary (Eq. 13); see the sign note under Inferences | Correlations analytic; learning conjectured; explicitly no proof linking it to GD |
| [Cagnetta, Favero, Sclocchi, Wyart, 2505.07070](https://arxiv.org/html/2505.07070); [PRE 112 (Dec 2025) per EPFL](https://infoscience.epfl.ch/handle/20.500.14299/257881) | last-token prediction | Transformer P_ℓ=(1−f)⁻¹ v m^{2ℓ+1} (Eq. 15); CNN with weight sharing P̄_ℓ=(1−f)⁻¹ v m^{ℓ+2} (Eq. 16). Loss exponents log f/(2 log m) vs log f/log m (Eqs. 23–24) | Assumption V.1 (learned once detectable) is an assumption; empirical fit |
| [Favero et al., ICML 2025, 2502.12089](https://arxiv.org/html/2502.12089) | discrete diffusion (D3PM, U-Net depth ≥ 2L) | P₁ ~ vm; P_ℓ = (1−m/v^{s−1})⁻¹ v m^{ℓ+1} (Eq. 6); overall m^{L+1} = m·d^{log m/log s}. Mechanism: k-means on mean context vectors E[x\|h=ν], word2vec-like | Correlations derived asymptotically; Assumption 4.1 assumed; one-step GD for a linear model at first level only |
| [Sclocchi, Favero, Wyart, 2402.16991](https://arxiv.org/html/2402.16991v3) | Bayes-optimal denoising | Phase transition in class reconstruction at ε*(sf) when sf<1 (up to O(1/v)); low-level features change smoothly | BP exact with known rules; mean-field approximate; ImageNet evidence is by proxy |
| [Sclocchi, Favero, Levi, Wyart, 2410.13770](https://arxiv.org/abs/2410.13770) | forward-backward | Changes occur in correlated chunks, length diverging at the transition | I read the abstract only |
| [Favero, Sclocchi, Wyart, 2505.16959](https://arxiv.org/abs/2505.16959) | diffusion memorization | Memorization time ∝ dataset size; generalization by progressive rule learning | I read the abstract only |
| [Ren, Dandi, Krzakala, Lee, 2601.19756](https://arxiv.org/html/2601.19756) | classification | **Thm 4.1**: L-layer conv net, RBF random features, layerwise ridge GD, fresh samples per stage, first patch only: poly(κ,m,V,s,L,log 1/δ) K_ρ⁻² m^L samples; informal O(m^{(1+o(1))L}) | **Rigorous** upper bound under Assumption 4.1 (verified w.h.p. only for L~log d/log log d, m,V,s~log d, Prop. B.1). No lower bound |
| [Korchinski, Favero, Wyart, 2605.27734](https://arxiv.org/html/2605.27734v1) | self-supervised on own latents | Supervised P^sup_ℓ ≍ v m^{L−ℓ}; token P^tok_ℓ ≍ v m^{ℓ+2} (Eq. 1), overall v m^{L+1}; latent ILC P ≍ v m³/(1−f), **depth-independent** (Thm 1/2, Eq. 2–3) | Theorem rigorous conditional on Assumptions 1–2 (see Sec. 3); SLC has no theorem; data2vec argued under A1, A2 |
| [Delrocq, Zihan, Bellec, Gerstner, 2605.18557](https://arxiv.org/html/2605.18557) | local rules | DFA, Strong-DFC, PEPITA fail; CLAPP, LPL, layerwise SimCLR reach P* ∝ n_c v^L for m=v=n_c | Empirical |
| [Liu, 2510.17469](https://arxiv.org/html/2510.17469) | transformers, in-context | Layer-specialisation scores for causal vs masked attention, with RHM + Zipf layers | Empirical, correlational, unreviewed preprint; the abstract's PCA claims are not shown in the body as far as the summary reports |

**Adjacent, not RHM**
- [Gaitonde, Koehler, Mossel, Shin, Sly, 2605.13687](https://arxiv.org/html/2605.13687): broadcast on trees with stochastic kernels (Ising and coloring). Thms 3.1–3.4 cover bounded-context autoregressive sampling; freezing implies Ω(n) context; reasoning with Θ(log n) memory samples exactly. The transformer link is an unproved k-gram ansatz. It cites the RHM line but gives no formal mapping.
- [2606.28103 "Random Language Model"](https://arxiv.org/abs/2606.28103): the page does not mention the RHM. Treat the [aggregator's](https://www.emergentmind.com/topics/random-hierarchy-model) description of it as adjacent, and do not count it as an RHM extension.
- [Dandi et al. line, 2602.10867](https://bytez.com/docs/arxiv/2602.10867/paper) uses Gaussian data and Hermite targets, per a search snippet only.
- [2606.17522](https://arxiv.org/abs/2606.17522) (bounded-depth grammars, expressivity): withdrawn v2, 17 Jul 2026, no RHM mention. [2607.03660](https://arxiv.org/abs/2607.03660): unrelated.

### Inferences
- Exponents differ across papers largely by what is being learned and by ℓ indexing, so they should not be compared by raw ℓ. The stable overall numbers are: supervised class ~ n_c m^L; token-level top ~ v m^{L+1} (CNN/diffusion); latent-predictive ~ v m³.
- The 2ℓ−1 (NeurIPS) and 2ℓ+1 (PRE) transformer exponents probably reflect an ℓ offset. I did not confirm this against the PRE text.
- Sign note: with f<1, log f<0, so P^{log f/(2 log m)} decays. The [2406.00048 summary](https://ar5iv.labs.arxiv.org/html/2406.00048) writes P^{−α} with α = log f/(2 log m), which would be a growth law if taken literally. The [2505.07070 form](https://arxiv.org/html/2505.07070) P^{log f/(2 log m)} is coherent. Check the original before quoting.
- In the Prop. B.1 regime, m^L ≈ d up to subpolynomial factors (L log m ≈ log d), so the "polynomial" in Ren et al. is not comparable to fixed-m PRX experiments without care.
- Ren's result says this architecture and training can reach m^L. It does not establish that end-to-end SGD does, and it does not prove the shallow-deep gap.

### Gaps
- Search coverage: I found no RHM paper with explicit context-sensitive, nonlocal, or Boolean-rule variants. Search returned nothing under those terms, and this is not a proof of absence. "Correlations in the rules" appears only as the uncorrelated control (PRX Sec. III.4) and the PRX future-work line about errors in rules and frequency inhomogeneity. Liu's Zipf layers change rule frequencies only.
- Ren et al. appendices A–C and the proof of Lemma A.3 were not read.

---

## 3. Synonymic invariance and recovery of parts from a new composite

### Takeaway
Synonymic invariance is a statement about merging m tuples with one parent, measured by S_{k,l} and predicted from label or context correlations. The theory is about **recovering the partition of tuples into parents** (the grammar) from data. Only one theorem gives exact latent recovery (Korchinski et al.), it is on the given sample, and it depends on an assumed clustering module. No paper I read poses or tests "parse a new composite into its latents" as a target. The nearest result follows as an inference from unambiguity.

### Cited Findings
- PRX position: a label depends only on symbols, so the task is invariant under synonym exchange at every level; S_{k,l} falls at P* (Figs. 5–6); this is empirical beyond the one-step setting. — [Source](https://arxiv.org/html/2307.02129v5)
- PRE 2025 separates two interventions, random **variable** replacement (S) and random **rule** replacement (R), via cosine similarity q_T (Eq. 26) on standardised activations (Eq. 25). Invariance to R with sensitivity to S is read as encoding the hidden symbol but not its subtree (Figs. 6–7). This is the most direct "represents the latent" probe in the line, and it is still not a decoder. — [Source](https://arxiv.org/html/2505.07070)
- Ren et al.: latent tokens are **not** recovered explicitly. Each level learns a proxy, the conditional label distribution q_μ^(l), which identifies synonyms. The authors argue that recovering a specific set of lower-level features is unnecessary for the chain. — [Source](https://arxiv.org/html/2601.19756)
- Korchinski et al. Thm 1/2: with probability ≥ 1−δ, ILC outputs h^(1),…,h^(L−1) and all production-rule classes below the root, exactly, up to a permutation of labels at each level. Sample bound P ≳ vm log(Lvm/δ) + (vm³/(1−f)) log(Lvm/δ). Assumption 1 (balanced, separated grammar; separation scale from a random-direction heuristic). Assumption 2 (stable clustering module) is assumed, not proved. — [Source](https://arxiv.org/html/2605.27734v1)
- Korchinski et al. ILC takes "the observed tuples as the support of grammatical tuples", so tuple labels are defined on the observed support. The summary reports no evaluation of parsing unseen tuples or new inputs. Probes are clustering accuracy (with permutation matching), a linear probe (SLC, top-level latent), and an MLP on mean-pooled features (data2vec, root label). The clustering score C_ℓ is normalised (0 = synonym swap moves the representation as much as a generic swap, 1 = synonyms have a common image). — [Source](https://arxiv.org/html/2605.27734v1)
- Delrocq et al. use linear readouts from intermediate layers and S_{k,l}; the decoder is trained on the same P samples and tested on unseen samples. Readout target at intermediate layers not confirmed beyond this. — [Source](https://arxiv.org/html/2605.18557)
- Favero et al.: the clustering mechanism computes mean context per tuple and k-means, then "P₁ ~ vm… Since these tuples lack internal structure, they can only be memorized." — [Source](https://arxiv.org/html/2502.12089v3)
- Favero's thesis describes "recovering the latent variables" as the key mechanism in the Chapter 6 overview; reusable parts appear only as a hypothesis in Sec. 3.3 of the fetched text. — [Source](https://arxiv.org/html/2510.06106)
- Sclocchi et al.: on ImageNet, low-level features of the source persist into a new-class sample (a wolf with leopard eyes, nose and ears). This is qualitative, a recombination observation rather than a parsing result. Bayes-optimal BP inference is the analytic part, and it assumes known rules. — [Source](https://arxiv.org/html/2402.16991v3)

### Inferences
- For a grammar with all tuples labelled, unambiguity makes parsing a table lookup. Any input whose tuples are all in the support is parsed exactly, even if the tree as a whole was never seen. So Korchinski's result implies recovery on new compositions of known rules, as a corollary of exact recovery of the rule classes. This is **my inference**; I did not find it stated or tested in the summary I read.
- A tuple never observed has no usable content signal, because the rule tables are uniformly random: the parent of a tuple is independent of its component symbols, apart from the without-replacement constraint. Only its context could identify the parent. So rule-level or symbol-level generalization by "internal structure" is ruled out by the model's construction. The phrase "can only be memorized" for P₁ says this for level 1. Composition below the rule level is therefore absent from the RHM.
- Synonym invariance gives composition in only one direction. Merging tuples into symbols is abstraction that discards which synonym was present. Reconstruction of the specific lower-level tuple is impossible from the invariant state alone (m-to-1). The RHM's classification measures reward this discarding. The token and diffusion tasks need generation back down, which is the setting where latents plus rule tables are both required.
- Behavioural vs representational: P* is behavioural. S_{k,l} and q_T are representational, but they test invariance, not identifiability of latent values. Clustering accuracy with permutation matching (Korchinski) is the only measure in the line that checks identity of a latent partition, and it is for explicit-clustering ILC, with C_ℓ for the trained encoders.

### Gaps
- I could not verify whether Korchinski's remaining 25k characters or appendices evaluate a held-out parse.
- Thesis Chapters 6–7 and Sec. 49 were not reachable: the arXiv HTML fetch ended within Sec. 23.

---

## 4. How held-out data and generalization are defined

### Takeaway
Test sets are **new samples from the same rule tables**, split at the level of whole inputs. I found no hold-out of rules, tuples, or symbols in any RHM paper read. Test inputs are de facto new compositions of already-seen rules, because P ≪ P_max at P*, but this is a by-product of sampling and is not enforced.

### Cited Findings
- PRX: P training points drawn uniformly at random without replacement from the RHM data, all distinct; test size min(P_max−P, 20,000); App. A.3. — [Source](https://arxiv.org/html/2307.02129v5)
- Code: both train and test are slices of one set of unique datapoint indices, so they are disjoint. `sample_hierarchical_rules` is called once with one `seed`, so rules are shared. There is **no option** to hold out rules, tuples, symbols. Default `testsize` = min(max_dataset_size//5, 20000). — [Code](https://raw.githubusercontent.com/pcsl-epfl/hierarchy-learning/master/datasets/hierarchical.py)
- PRE 2025: online training, fresh batch of 128 each step, test size 2¹⁵, 8 realisations. With online sampling, train/test disjointness is not by construction in the summary I read. — [Source](https://arxiv.org/html/2505.07070)
- Delrocq et al.: test = min(100,000, P_max−P) unseen samples. — [Source](https://arxiv.org/html/2605.18557)
- Favero et al. (diffusion): no RHM test set. The measure is level accuracy A_ℓ, the fraction of 1024 generated samples satisfying level-ℓ rules; P* at A_L = 1/2. "We routinely verify that the model has not simply memorized the training data" gives no metric. Novelty is not formally defined and is implicit. — [Source](https://arxiv.org/html/2502.12089v3)
- Gaitonde et al.: no split; valid-rate statistics on ≥ 1,000 generations. — [Source](https://arxiv.org/html/2605.13687)
- Liu 2510.17469: the closest to a composition split, with "gen-same" described as "novel hierarchical combinations of learned production rules" and transfer as different Zipf layers. The construction is not specified on the page. — [Source](https://arxiv.org/html/2510.17469)
- Ren et al.: i.i.d. samples, no separate test distribution discussed; the guarantee covers patches in the support of the representations. — [Source](https://arxiv.org/html/2601.19756)

### Inferences (worked example, my calculation)
- For L=3, s=2, n_c=m=v=8 (Fig. 6's setting): P_max = 8·8⁷ = 8⁸ ≈ 1.68×10⁷; P* ≈ n_c m^L = 4,096, about 2.4×10⁻⁴ of P_max. Each of the 64 bottom tuples occurs ~P·4/64 ≈ 256 times at P* (if marginals are roughly uniform). So almost surely every test input uses only seen bottom tuples, and its whole tree is new.
- This makes the RHM test "new composition of known parts" for whole trees, and intermediate for subtrees. The split is never controlled by combination type, so nobody can read off generalization to specific unseen compositions, such as a specific pair of symbols that never co-occurred in a parent.
- Since all m rule alternatives at a level are random and independent of siblings given the parent, a test tree uses rules whose joint assignment is novel but whose factors are all present. The factorised structure is exact. There is no novelty beyond the Cartesian product that the grammar defines.

### Gaps
- A controlled hold-out in the RHM would be a straightforward experiment: withhold parent-tuple pairs, or sibling-symbol pairs, from training. I found none reported.

---

## 5. Criticisms and limits; rigour

### Takeaway
I found no paper dedicated to criticising the RHM. The limits below are ones stated by the authors or evident from the construction. The rigour picture: proofs exist for a layerwise upper bound and a conditional clustering recovery; every statement about SGD on deep nets remains empirical or assumption-based.

### Cited Findings

**Author-stated realism limits**
- Korchinski et al. Sec. 6: fixed tree topology, unambiguous non-recursive rules, no explicit context dependence; extensions needed to approach natural language and images. — [Source](https://arxiv.org/html/2605.27734v1)
- Cagnetta & Wyart 2024: context-free rules capture most but not all syntactic forms; fixed tree geometry, uniform rule probabilities and unambiguity do not hold for real text and cause the stepwise correlations. — [Source](https://ar5iv.labs.arxiv.org/html/2406.00048)
- PRE 2025: fixed topology and unambiguity; context-free only; simplified Transformer with no feedforward layers; variable topologies proposed as where attention might help. — [Source](https://arxiv.org/html/2505.07070)
- A search snippet from Favero's thesis says natural language requires at least mildly context-sensitive models and names context-dependent latent models as a key challenge. I could not read this directly (it lies in the unreachable part of the thesis). — [Snippet source](https://arxiv.org/pdf/2510.06106)
- SRHM: no empirical way yet to measure synonym insensitivity on benchmark images, so the predicted synonym-diffeo link is untested on real images. — [Source](https://arxiv.org/html/2404.10727v2)
- Delrocq et al.: unclear relevance to vision, music, language. — [Source](https://arxiv.org/html/2605.18557)
- PRX future work: "errors in the composition rules", frequency inhomogeneities, testing invariance in language modelling, and "a formidable challenge is to obtain a detailed description of the gradient-descent dynamics of deep networks." — [Source](https://arxiv.org/html/2307.02129v5)

**Real-data claims**
- PRX CIFAR-10 "rule of thumb": n_c=10, m∈[5,15], L=3 gives P*∈[10³, 3×10⁴], "comparable with the sample complexity of modern architectures". The m and L values are chosen as "reasonable", not measured. — [Source](https://arxiv.org/html/2307.02129v5)
- Natural-data scaling: Shakespeare β≈1.4, collapse z≈2.8; WikiText z≈3.1, α≈0.095. These are fits to a conjecture. — [Source](https://ar5iv.labs.arxiv.org/html/2406.00048)
- Favero 2025: text and images are qualitative or correlational (LLaMA-2 perplexity vs context, MMD by ResNet depth); there is no theory for natural data; the paper's "strong empirical support" and its statement that Assumption 4.1 is "confirmed" by scaling match are its own framing. — [Source](https://arxiv.org/html/2502.12089v3)

**Architecture match**
- Experiments use depth > L, CNN filter = stride = s, non-overlapping patches, width ~v^s. So topology is known to the architecture in the CNN. Deep FC nets succeed with an extra 2^L or s^L factor. — [Source](https://arxiv.org/html/2307.02129v5)
- Ren et al. require known s and topology. — [Source](https://arxiv.org/html/2601.19756)
- Transformers do learn without the match, but with a worse exponent: m^{2ℓ+1} vs m^{ℓ+2}. — [Source](https://arxiv.org/html/2505.07070)
- U-Net must have depth ≥ 2L. — [Source](https://arxiv.org/html/2502.12089v3)

**Rigour ladder**
| Claim | Status |
|---|---|
| P_max and counting, rule statistics (Eqs. 3, 15–24) | Exact |
| Correlations scaling (PRX App. C; 2406.00048 Eq. 8; 2502.12089 Eq. 5) | Asymptotic derivations with an annealed/concentration argument, averaged over RHM realisations |
| P_c = n_c m^L (Sec. III.2) | Heuristic per the paper; App. C refines it |
| One-step invariance (Eq. 13) | Exact calculation in a simplified net; limit P→∞ of counts. Not a deep-net theorem |
| P* ≃ n_c m^L for deep nets, curse for shallow | Empirical. Upper bound for a layerwise variant: Ren Thm 4.1. No lower bound |
| P_ℓ thresholds for token or diffusion | Assumption ("learned when detectable") + empirics |
| ILC vm³/(1−f) | Theorem, conditional on Assumptions 1–2 |
| SLC, data2vec | Experiments; argument under A1–A2 |

**Other technical cautions**
- Paper internal inconsistencies: threshold for P* (0.1 vs 0.7 ε_rand depending on figure); 2502.12089's labels P₂ vs P₃ for the m³/m⁴ thresholds. — [PRX](https://arxiv.org/html/2307.02129v5); [2502.12089](https://arxiv.org/html/2502.12089)
- Diffusion BP analysis uses a single phase-diagram realisation and known rules. — [Source](https://arxiv.org/html/2402.16991v3)
- Korchinski: CLAPP (Appendix A) used a supervised-style top-level signal and recovered m^L scaling; whether biological architectures reach m³ is untested. Note [Delrocq et al.](https://arxiv.org/html/2605.18557) report CLAPP-type rules scaling with n_c v^L. — [Source](https://arxiv.org/html/2605.27734v1)

### Inferences
- The RHM's strength as a model and its main unrealism have the same source: i.i.d. random rule tables with no structure inside rules, so all generalization comes from tree factorisation. Symbols have no features, so there is no notion of similar symbols, of generalising a rule to a new symbol, or of partial matches.
- The RHM's tree is fixed by position. Parts are position-indexed, and a CNN's weight sharing merges positions although the rule tables are per level, not per position. There is no reuse of a part in a different structural role unless the tuple appears at another position at the same level. Reuse across levels is impossible, because vocabularies are disjoint across levels.
- Theory results compare scalings, up to constants and for large v, m. Fig. 12 and Korchinski's collapse plots provide empirical check, but L is small in almost all experiments (L≤3–7, typically s=2–3).

### Gaps
- I found no external critical paper on the RHM's realism in these searches. Two searches aimed at criticism returned only the RHM papers themselves. Absence should be read as "not surfaced", not "does not exist".
- Thesis Sec. 49 ("Beyond simplified models of hierarchy") could not be read through this tool. The [Infoscience record](https://infoscience.epfl.ch/handle/20.500.14299/254241) holds the official copy.
