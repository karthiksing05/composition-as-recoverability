"""E8 grid: does mean pooling over patches explain the shortfall of image encoders?

E8 reads the mean of an image encoder's patch states, which discards where each patch lies.
Here the composite state is instead the concatenation of the mean patch state in each quadrant of
the patch grid ("G" layers), which keeps coarse position.  The part codes, references, splits and
readouts are those of E8 (mean-pooled states of the photograph alone), so kappa changes only
through the composite's state.  If position invariance of the pooling causes the shortfall, the
grid states approach the concatenation reference; if the encoder blends the photographs within
its patches, they do not.  The mean-pooled layers are recomputed as a check on E8.
"""
import argparse, io, json, os
import numpy as np
import torch
from PIL import Image
import e4_pretrained as E
import e8_vision as V

PREFIX = {"clip": 1, "dino": 1, "siglip": 0}  # class tokens before the patch tokens


@torch.no_grad()
def embed_grid(m, ims, dev):
    mod, size, mean, std, kind = m
    out = {}
    for s in range(0, len(ims), 128):
        x = np.stack([(np.asarray(im.resize((size, size), Image.BICUBIC), np.float32) / 255 - mean) / std for im in ims[s:s + 128]])
        hs = mod(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2).to(dev), output_hidden_states=True).hidden_states
        L = len(hs) - 1
        for li in sorted({L // 4, L // 2, (3 * L) // 4, L}):
            h = hs[li]
            out.setdefault(f"L{li}", []).append(h.mean(1).float().cpu().numpy())
            p = h[:, PREFIX[kind]:]
            g = int(round(p.shape[1] ** 0.5))
            p = p.reshape(len(p), g, g, -1)
            q = [p[:, r][:, :, c].mean((1, 2)) for r in np.array_split(np.arange(g), 2) for c in np.array_split(np.arange(g), 2)]
            out.setdefault(f"G{li}", []).append(torch.cat(q, 1).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


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
        H = embed_grid(m, [V.compose(rel, pil[x], pil[y]) for x, y in allp], dev)
        cands = {"val": splits["val"], "fresh": splits["fresh"]}
        for layer, Hall in H.items():
            for cname in V.CODES:
                P = codes[cname]["L" + layer[1:]]
                pmu = P[pca].mean(0)
                _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
                A = P[[idx[x] for x, _ in allp]]; B = P[[idx[y] for _, y in allp]]
                systems = {"model": Hall} if layer.startswith("G") else {}
                if layer.startswith("L"):
                    systems = {"model": Hall, "ref_concat": np.c_[A, B], "ref_sum": A + B}
                for sname, Hs in systems.items():
                    row = dict(model=name, seed=seed, rel=rel, layer=layer, code=cname, system=sname, dim=int(Hs.shape[1]))
                    row.update(E.evaluate(Hs, T, idx, sets, cands))
                    res.append(row)
                print(name, rel, layer, cname, round((res[-1]["rec_fresh_role1"] + res[-1]["rec_fresh_role2"]) / 2, 3), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outdir", required=True)
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
