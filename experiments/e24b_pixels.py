"""E24b: the input baseline for image encoders.

The same splits, part codes, references and readouts as E8 grid (the same seed gives the same draws), with the
composite's raw pixels as the state: 64 x 64 RGB, reduced to 1024 principal components fitted on the fitting
composites.  Pixels keep position by construction, so this is the baseline for the quadrant state: a readout
must map pixels to the encoder's own part code.
"""
import argparse, io, json, os
import numpy as np
import torch
from PIL import Image
import e4_pretrained as E
import e8_vision as V


def run(name, seed, dev, images, labels):
    rng = np.random.default_rng(seed)
    cls = rng.permutation(200)
    splits, s0 = {}, 0
    for k, n, per in zip(("fit", "val", "fresh"), V.N_CLS, V.PER_CLS):
        ids = []
        for c in cls[s0:s0 + n]:
            ids += list(rng.choice(np.where(labels == c)[0], per, replace=False))
        splits[k] = [int(i) for i in ids]; s0 += n
    used = splits["fit"] + splits["val"] + splits["fresh"]
    pil = {i: Image.open(io.BytesIO(images[i])).convert("RGB") for i in used}
    m = V.load(name, dev)
    codes = {"center": V.embed(m, [V.canvas([(pil[i], 112, (56, 56))]) for i in used], dev),
             "full": V.embed(m, [pil[i].resize((224, 224), Image.BICUBIC) for i in used], dev)}
    idx = {x: j for j, x in enumerate(used)}
    pca = list(range(len(splits["fit"])))
    res = []
    for rel in V.RELATIONS:
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
        allfit = sample(splits["fit"], 3800)
        fit_pairs, pairs_pairs = allfit[:3000], allfit[3000:]
        val_pairs, fresh_pairs = sample(splits["val"], 600), sample(splits["fresh"], 1000)
        g12 = splits["fit"][:12]
        grid = [(x, y) for x in g12 for y in g12]
        rev_pairs = [(y, x) for x, y in fresh_pairs]
        sets, allp, s0 = {}, [], 0
        for k, prs in (("fit", fit_pairs), ("pairs", pairs_pairs), ("val", val_pairs), ("fresh", fresh_pairs), ("grid", grid), ("rev", rev_pairs)):
            sets[k] = slice(s0, s0 + len(prs)); s0 += len(prs); allp += prs
            sets[f"{k}_pairs"] = prs
        X = np.stack([np.asarray(V.compose(rel, pil[x], pil[y]).resize((64, 64), Image.BICUBIC), np.float32).ravel() / 255 for x, y in allp])
        mu = X[:3000].mean(0)
        _, _, Vx = np.linalg.svd(X[:3000] - mu, full_matrices=False)
        Hp = (X - mu) @ Vx[:1024].T
        cands = {"val": splits["val"], "fresh": splits["fresh"]}
        for cname in V.CODES:
            for layer, P in codes[cname].items():
                pmu = P[pca].mean(0)
                _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
                A = P[[idx[x] for x, _ in allp]]; B = P[[idx[y] for _, y in allp]]
                for sname, Hs in {"pixels": Hp, "ref_concat": np.c_[A, B], "ref_sum": A + B}.items():
                    row = dict(model=name, seed=seed, rel=rel, layer=layer, code=cname, system=sname)
                    row.update(E.evaluate(Hs, T, idx, sets, cands))
                    res.append(row)
                print(name, rel, layer, cname, {r["system"]: round((r["rec_fresh_role1"] + r["rec_fresh_role2"]) / 2, 3) for r in res[-3:]}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", default="results/e24b")
    a = ap.parse_args()
    E.ALPHAS = np.logspace(-3, 9, 25)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    images, labels = V.load_images()
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            fn = os.path.join(a.outdir, f"{name}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(name, seed, dev, images, labels)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
