"""E41: does the question-state probe (E11 'query' state) read the model's ANSWER or the true FIGURE?
Same pipeline as e11_tokens.py (nouns/splits, part codes, 16-PC whitened targets, ridge fitted on
canonical sentences of fitting nouns with penalty on validation nouns, E9 prompt with 4 demos), at the
(layer, code) selected by the E11 rule.  Per fresh-noun item (canonical and converse) we record the
model's answer (E9 two-way continuation choice, plus greedy next-token category), the true figure, and
which noun the probe prediction is nearer.  Items are the fresh pairs of the E11 draw (half per phrasing).
"""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E
from e7_roles import PAIRS
from e9_behavior import QUESTIONS, item, fmt, N_DEMO, seq_logprob
from e11_tokens import spans  # noqa (unused; keeps import path identical)

SEL = {"microsoft/phi-4": (30, "phrase"), "Qwen/Qwen3-8B": (27, "carrier"),
       "allenai/OLMo-2-1124-13B": (30, "phrase"), "allenai/OLMo-2-1124-7B": (16, "carrier")}


@torch.no_grad()
def query_and_greedy(tok, mod, texts, li, cand_first, dev, bs=16):
    """State at the final token of the layer li, and greedy next token's first-token id."""
    H, G = [], []
    head = mod.get_output_embeddings()
    for s in range(0, len(texts), bs):
        b = tok(texts[s:s + bs], padding=True, return_tensors="pt").to(dev)
        o = mod.base_model(**b, output_hidden_states=True)
        hs = o.hidden_states
        last = b["attention_mask"].sum(1) - 1
        ar = torch.arange(len(last))
        H.append(hs[li][ar, last].float().cpu().numpy())
        G.append(head(hs[-1][ar, last]).argmax(-1).cpu().numpy())
    return np.concatenate(H), np.concatenate(G)


def run(name, seed, dev, scale=1.0, rels=None):
    li, cname = SEL[name]
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
    fit_n, val_n, fr_n = nouns[:E.N_FIT], nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    code_txt = (lambda x: f"a {x}") if cname == "phrase" else (lambda x: f"there is a {x} here")
    P = E.embed(tok, mod, [code_txt(x) for x in nouns], dev, name)[f"L{li}"]
    idx = {x: i for i, x in enumerate(nouns)}
    pca = list(range(E.N_FIT))
    pmu = P[pca].mean(0)
    _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
    T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
    first_tok = {n: tok(" " + n, add_special_tokens=False)["input_ids"][0] for n in fr_n}
    items = []
    for rel, (canon, conv) in PAIRS.items():
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        fitp, valp, frp = sample(fit_n, int(3000 * scale)), sample(val_n, int(600 * scale)), sample(fr_n, int(1000 * scale))
        texts, meta = [], []
        for split, prs in (("fit", fitp), ("val", valp), ("fresh", frp)):
            for i, (x, y) in enumerate(prs):
                ph = i % 2
                texts.append((canon if ph == 0 else conv).format(x=x, y=y)); meta.append((split, ph, x, y))
        dn = rng.choice(fit_n, 2 * N_DEMO, replace=False)
        demo = [fmt(*item(rel, dn[2 * k], dn[2 * k + 1], p, q)[:2]) + f" {item(rel, dn[2 * k], dn[2 * k + 1], p, q)[2]}\n\n"
                for k, (p, q) in enumerate([(p, q) for p in (0, 1) for q in (0, 1)])]
        head = "Read each sentence and answer the question with one object.\n\n" + "".join(demo[i] for i in rng.permutation(len(demo)))
        if rels and rel not in rels:
            continue
        prompts = [head + fmt(t, QUESTIONS[rel][0]) for t in texts]
        H, G = query_and_greedy(tok, mod, prompts, li, None, dev)
        sp = np.array([m[0] for m in meta]); ph = np.array([m[1] for m in meta])
        fig = [m[2] for m in meta]; gnd = [m[3] for m in meta]
        sel = lambda *c: np.where(np.logical_and.reduce(c))[0]
        Fc, Vc, R = sel(sp == "fit", ph == 0), sel(sp == "val", ph == 0), sel(sp == "fresh")
        g = lambda L, rows: [L[i] for i in rows]
        mu = H[Fc].mean(0)
        reg = E.RidgePath(H[Fc] - mu, T[[idx[x] for x in g(fig, Fc)]]).select(H[Vc] - mu, T[[idx[x] for x in g(fig, Vc)]])
        p = reg.predict(H[R] - mu)
        yf, yg = T[[idx[x] for x in g(fig, R)]], T[[idx[x] for x in g(gnd, R)]]
        near_fig = ((p - yf) ** 2).sum(1) < ((p - yg) ** 2).sum(1)
        # model answer: E9 two-way continuation choice; greedy next-token category
        Rp = [prompts[i] for i in R]
        lf = seq_logprob(tok, mod, Rp, [" " + fig[i] for i in R], dev, bs=8)
        lg = seq_logprob(tok, mod, Rp, [" " + gnd[i] for i in R], dev, bs=8)
        for k, i in enumerate(R):
            gr = int(G[i])
            gcat = "fig" if gr == first_tok[fig[i]] else ("gnd" if gr == first_tok[gnd[i]] else "neither")
            items.append(dict(model=name, seed=seed, rel=rel, phrasing=["canonical", "converse"][int(ph[i])],
                              fig=fig[i], gnd=gnd[i], ans="fig" if lf[k] > lg[k] else "gnd", greedy=gcat,
                              probe="fig" if near_fig[k] else "gnd"))
        acc = np.mean([t["ans"] == "fig" for t in items if t["rel"] == rel])
        print(name, seed, rel, "acc", round(float(acc), 3), "probe", round(float(near_fig.mean()), 3), "alpha", reg.a, flush=True)
    del mod
    import gc; gc.collect(); torch.cuda.empty_cache()
    return dict(model=name, seed=seed, layer=li, code=cname, scale=scale, items=items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="0+1+2")
    ap.add_argument("--outdir", default="/coc/flash10/ksingara3/compositionality-theory/experiments/results/e41")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rels", default=None)
    a = ap.parse_args()
    E.ALPHAS = np.logspace(-3, 9, 25)
    E.WHITEN = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    for seed in map(int, a.seeds.split("+")):
        fn = os.path.join(a.outdir, f"{a.model.replace('/', '_')}_s{seed}.json")
        if os.path.exists(fn):
            continue
        r = run(a.model, seed, dev, a.scale, a.rels.split(",") if a.rels else None)
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(r, f)
