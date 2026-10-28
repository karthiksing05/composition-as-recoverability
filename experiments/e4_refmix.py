"""E4 addendum (round-4 review A, M5): a role-filler reference that faces the same estimation
problem as a model.  The concatenation reference [s(x), s(y)] is recovered exactly by
construction, because its readout lies in the span of the fitting nouns' states.  Here the
reference is h = [s(x), s(y)] R with R a random Gaussian 2D x 2D matrix: still role-filler
(full column rank, delta = 0) but with non-orthogonal role maps, which a readout fitted on
165 nouns must unmix.  Same nouns, splits, pairs, whitened part codes, and readout protocol as
e4_pretrained.run, at the layer and part code selected for the trained model on validation nouns."""
import argparse, json, os
import numpy as np
import torch
import e4_pretrained as E


def run(name, seed, layer, cname, dev):
    rng = np.random.default_rng(seed)
    nouns = list(dict.fromkeys(E.NOUNS))
    rng.shuffle(nouns)
    fit_n = nouns[:E.N_FIT]
    pca_idx = list(range(E.N_FIT))
    val_n, fr_n = nouns[E.N_FIT:E.N_FIT + E.N_VAL], nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
    used = nouns
    tk, md = E.load(name, dev)
    if cname == "static":
        with torch.no_grad():
            W = md.get_input_embeddings().weight
            P = np.stack([W[torch.tensor(tk(x, add_special_tokens=False)["input_ids"])].mean(0).float().cpu().numpy() for x in used])
    else:
        tmpl = {"phrase": "a {}", "carrier": "there is a {} here"}[cname]
        P = E.embed(tk, md, [tmpl.format(x) for x in used], dev, name)[layer]
    idx = {x: i for i, x in enumerate(used)}
    pmu = P[pca_idx].mean(0)
    _, sv, Vt = np.linalg.svd(P[pca_idx] - pmu, full_matrices=False)
    T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca_idx)))
    D = P.shape[1]
    R = np.random.default_rng(7_000 + seed).normal(size=(2 * D, 2 * D)) / np.sqrt(2 * D)
    res = []
    for rel in E.RELATIONS:  # consume the random stream exactly as e4_pretrained.run does
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        allfit = sample(fit_n, 3800)
        n_tr = min(3000, int(0.8 * len(allfit)))
        fit_pairs, pairs_pairs = allfit[:n_tr], allfit[n_tr:]
        val_pairs, fresh_pairs = sample(val_n, 600), sample(fr_n, 1000)
        grid = [(x, y) for x in fit_n[:12] for y in fit_n[:12]]
        rev_pairs = [(y, x) for x, y in fresh_pairs]
        order = [("fit", fit_pairs), ("pairs", pairs_pairs), ("val", val_pairs), ("fresh", fresh_pairs), ("grid", grid), ("rev", rev_pairs)]
        sets, allp, s0 = {}, [], 0
        for k, prs in order:
            sets[k] = slice(s0, s0 + len(prs)); s0 += len(prs); allp += prs
            sets[f"{k}_pairs"] = prs
        if rel == "and":
            continue
        A, B = P[[idx[x] for x, _ in allp]], P[[idx[y] for _, y in allp]]
        cat = np.concatenate([A, B], 1)
        for sname, Hs in (("ref_concat", cat), ("ref_mix", cat @ R)):
            row = dict(model=name, seed=seed, rel=rel, layer=layer, code=cname, system=sname)
            row.update(E.evaluate(Hs, T, idx, sets, {"val": val_n, "fresh": fr_n}))
            res.append(row)
        print(name, seed, rel, {r["system"]: round(np.mean([r["rec_fresh_role1"], r["rec_fresh_role2"]]), 3) for r in res[-2:]}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--seeds", default="0+1+2")
    ap.add_argument("--outdir", default="results/e4mix")
    a = ap.parse_args()
    E.ALPHAS, E.WHITEN = np.logspace(-3, 9, 25), True
    os.makedirs(a.outdir, exist_ok=True)
    for seed in map(int, a.seeds.split("+")):
        fn = os.path.join(a.outdir, f"{a.model}_s{seed}.json")
        if not os.path.exists(fn):
            json.dump(run(a.model, seed, a.layer, a.code, "cpu"), open(fn, "w", encoding="utf-8"), indent=1)
