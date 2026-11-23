"""E38: three controls for the image-encoder results (E34 moved photographs, E17 matching).

1. Probe-class controls.  Built from the same encoder's photograph features f(photo) (E34's whitened PCA-16 part
   code T of the photograph alone, pooled-output layer), two synthetic scene states of fixed width:
     CTL_ORDER  s = f(left) (x) r1 + f(right) (x) r2      roles by left-to-right ORDER (r1, r2 random unit, dim 8; Kronecker, 128 d)
     CTL_ABS    s = f(p1) (x) q(x1, y) + f(p2) (x) q(x2, y)   q = fixed sinusoidal code of the absolute position of the photograph centre
                (cos/sin 2 pi k x, k=1..4, plus cos/sin 2 pi y; 10 d), mimicking a positional encoder
   On E34's fixed (beside) and moved ("rand") placements, same probe and kappa as the encoders.
   On the fixed canvas the photographs sit at the box centres (.25, .5), (.75, .5).
2. POOL readout: the model's own pooled image embedding (CLIP image_embeds after the visual projection; SigLIP
   pooler_output = MAP head; DINOv2 pooler_output = layer-normed CLS), same protocol, splits, pairs, placements, probe
   and kappa as E34.  Part codes for POOL are the same pooled embedding of the photograph alone (center / full).
3. Image-only matching.  For each canvas with photographs A (left), B (right): reference canvases built from OTHER photographs
   of the same classes: same arrangement (A' left, B' right), swapped (B' left, A' right), wrong object (A' left, C' right).
   Share of canvases whose pooled embedding is closer (cosine) to same than to swapped (roles) and to same than to wrong
   (identity).  Conditions: fixed (E17's BESIDE canvas), moved (test canvas moved as in E34, references at independent
   random placements), moved_samepos (references at the test canvas's own two left edges and row).
"""
import argparse, io, json, os
import numpy as np
import torch
from PIL import Image
import e4_pretrained as E
import e8_vision as V
import e30_diffusion_photos as E30
import e34_relative_places as E34

CANVAS, PH = E34.CANVAS, E34.PH
SMOKE = False
N_MATCH = 1000


# ---------------------------------------------------------------- pooled embedding
def load_pool(name, dev):
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection, AutoModel, SiglipVisionModel
    kind, repo, size = V.SPECS[name]
    proc = AutoImageProcessor.from_pretrained(repo)
    cls = {"clip": CLIPVisionModelWithProjection, "dino": AutoModel, "siglip": SiglipVisionModel}[kind]
    mod = cls.from_pretrained(repo, torch_dtype=torch.float32)
    return (mod.to(dev).eval(), size, np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32), kind)


@torch.no_grad()
def pool_embed(m, ims, dev, size=None, bs=32):
    mod, size0, mean, std, kind = m
    size = size or size0
    kw = dict(interpolate_pos_encoding=True) if (kind in ("clip", "siglip") and size != size0) else {}
    out = []
    for s in range(0, len(ims), bs):
        x = np.stack([(np.asarray(im.resize((size, size), Image.BICUBIC), np.float32) / 255 - mean) / std for im in ims[s:s + bs]])
        o = mod(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2).to(dev), **kw)
        e = o.image_embeds if kind == "clip" else o.pooler_output
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- controls
def role_vecs(seed, d=8):
    r = np.random.default_rng(31000 + seed).standard_normal((2, d))
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def pos_code(x, y):
    """x, y: (N,) fractions of the canvas (photograph centre) -> (N, 10)."""
    ks = np.arange(1, 5)[None]
    ax = 2 * np.pi * ks * x[:, None]
    return np.c_[np.cos(ax), np.sin(ax), np.cos(2 * np.pi * y)[:, None], np.sin(2 * np.pi * y)[:, None]]


def kron_sum(fa, fb, qa, qb):
    return (fa[:, :, None] * qa[:, None, :]).reshape(len(fa), -1) + (fb[:, :, None] * qb[:, None, :]).reshape(len(fb), -1)


def centres(cname, pl):
    """fractional (xa, xb, y) of the two photograph centres."""
    if cname == "fixed":
        n = len(pl)
        return np.full(n, .25), np.full(n, .75), np.full(n, .5)
    a = np.array(pl, float)
    return (a[:, 0] + PH / 2) / CANVAS, (a[:, 1] + PH / 2) / CANVAS, (a[:, 2] + PH / 2) / CANVAS


# ---------------------------------------------------------------- matching
def run_match(m, dev, seed, labels, fresh_ids, images):
    rng = np.random.default_rng(20000 + seed)
    sc = 0.1 if SMOKE else 1.0
    N = int(N_MATCH * sc)
    by = {}
    for i in fresh_ids:
        by.setdefault(int(labels[i]), []).append(i)
    cls = sorted(by)
    pil_cache = {}
    def pil(i):
        if i not in pil_cache:
            pil_cache[i] = Image.open(io.BytesIO(images[i])).convert("RGB")
        return pil_cache[i]
    recs = []
    for _ in range(N):
        ca, cb, cc = (int(c) for c in rng.choice(cls, 3, replace=False))
        a, b = (int(i) for i in (rng.choice(by[ca]), rng.choice(by[cb])))
        a2 = int(rng.choice([i for i in by[ca] if i != a])); b2 = int(rng.choice([i for i in by[cb] if i != b]))
        c2 = int(rng.choice(by[cc]))
        recs.append((a, b, a2, b2, c2))
    res = {}
    for cond in ("fixed", "moved", "moved_samepos"):
        if cond == "fixed":
            mk = lambda l, r, k: V.compose("beside", pil(l), pil(r))
            size = None; pls = None
        else:
            size = CANVAS
            rr = np.random.default_rng(40000 + seed)
            pls = [[E34.place(rr) for _ in range(4)] for _ in recs]  # 0 test, 1 same, 2 swap, 3 wrong (independent placements)
        def canv(k, j, l, r):
            if cond == "fixed":
                return V.compose("beside", pil(l), pil(r))
            p = pls[k][0] if cond == "moved_samepos" else pls[k][j]
            if j == 0:
                p = pls[k][0]
            return E34.rand_canvas(pil(l), pil(r), *p)
        sims = []
        for j, sel in enumerate(((0, 1, 0, 1), (2, 3, 0, 1), (3, 2, 0, 1), (2, 4, 0, 1))):
            ims = []
            for k, (a, b, a2, b2, c2) in enumerate(recs):
                ph = (a, b, a2, b2, c2)
                ims.append(canv(k, j, ph[sel[0]], ph[sel[1]]))
            e = pool_embed(m, ims, dev, size=size)
            sims.append(e / np.linalg.norm(e, axis=1, keepdims=True)); del ims
        cs = lambda j: (sims[0] * sims[j]).sum(1)
        res[cond] = dict(n=len(recs), role_acc=float((cs(1) > cs(2)).mean()), id_acc=float((cs(1) > cs(3)).mean()))
        print("match", cond, res[cond], flush=True)
    return res


# ---------------------------------------------------------------- main
def run(name, seed, images, labels, outdir):
    fn = os.path.join(outdir, f"{name}_s{seed}.json")
    if os.path.exists(fn):
        return
    sc = 0.1 if SMOKE else 1.0
    dev = "cuda"
    rng, splits = E30.get_split(seed, labels)
    used = splits["fit"] + splits["val"] + splits["fresh"]
    pil = {i: Image.open(io.BytesIO(images[i])).convert("RGB") for i in used}
    idx = {x: j for j, x in enumerate(used)}
    pca = list(range(len(splits["fit"])))

    def sample(ns, k):  # identical draws to E34.run_system
        prs = [(x, y) for x in ns for y in ns if x != y]
        return [prs[i] for i in rng.choice(len(prs), min(int(k), len(prs)), replace=False)]
    pairs = {"fit": sample(splits["fit"], 3000 * sc), "val": sample(splits["val"], 600 * sc), "fresh": sample(splits["fresh"], 1000 * sc)}
    n = tuple(len(pairs[k]) for k in ("fit", "val", "fresh"))
    allp = pairs["fit"] + pairs["val"] + pairs["fresh"]
    r2g = np.random.default_rng(7000 + seed)
    pl = [E34.place(r2g) for _ in allp]
    r2g.random(len(allp))  # the shuffle draw of E34 (OBJr), kept so the stream matches
    m = load_pool(name, dev)
    codes = {"center": {"pool": pool_embed(m, [V.canvas([(pil[i], 112, (56, 56))]) for i in used], dev)},
             "full": {"pool": pool_embed(m, [pil[i].resize((224, 224), Image.BICUBIC) for i in used], dev)}}
    Ts = {cn: {l: E30.make_T(P, pca) for l, P in cd.items()} for cn, cd in codes.items()}
    sp = {k: [[idx[p[r]] for p in pairs[k]] for r in (0, 1)] for k in ("fit", "val", "fresh")}
    cidx = {"val": [idx[x] for x in splits["val"]], "fresh": [idx[x] for x in splits["fresh"]]}
    for k in ("val", "fresh"):
        sp[k + "_c"] = [np.array([splits[k].index(p[r]) for p in pairs[k]]) for r in (0, 1)]
    ia = [idx[x] for x, _ in allp]; ib = [idx[y] for _, y in allp]
    refs, rows = {}, []

    def add_rows(cname, rd, layer, X, codes_use):
        Tm = {c: Ts[c][layer] for c in codes_use}
        for c in codes_use:
            if (layer, c) not in refs:
                P = codes[c][layer]
                refs[(layer, c)] = {s: score(P, ia, ib, s, n, {c: Ts[c][layer]}, sp, cidx)[c] for s in ("ref_concat", "ref_sum")}
        Xs = X if isinstance(X, dict) else {c: X for c in codes_use}
        for c in codes_use:
            s_ = E34.score(torch.as_tensor(Xs[c]), n, {c: Tm[c]}, sp, cidx)[c]
            row = dict(model=name, seed=seed, cond=cname, readout=rd, layer=layer, code=c, dim=int(Xs[c].shape[1]))
            row.update(s_)
            rf = refs[(layer, c)]
            for r_, tag in ((1, "role1"), (2, "role2"), (0, "mean")):
                g = (lambda d, k: d[f"{k}_{tag}"]) if r_ else (lambda d, k: (d[f"{k}_role1"] + d[f"{k}_role2"]) / 2)
                c_, s2 = g(rf["ref_concat"], "rec_fresh"), g(rf["ref_sum"], "rec_fresh")
                row[f"kappa_{tag}"] = (g(row, "rec_fresh") - s2) / (c_ - s2)
                row[f"ref_concat_{tag}"], row[f"ref_sum_{tag}"] = c_, s2
            rows.append(row)
            print(name, seed, cname, rd, c, "kappa", round(row["kappa_mean"], 3), flush=True)

    def score(P, ia_, ib_, s, n_, Tm, sp_, cidx_):
        c = next(iter(Tm))
        X = np.c_[P[ia_], P[ib_]] if s == "ref_concat" else P[ia_] + P[ib_]
        return E34.score(torch.as_tensor(X), n_, Tm, sp_, cidx_)

    R = role_vecs(seed)
    for cname in ("rand", "fixed"):
        # 2. POOL readout
        if cname == "rand":
            ims = (E34.rand_canvas(pil[x], pil[y], *p) for (x, y), p in zip(allp, pl))
        else:
            ims = (V.compose("beside", pil[x], pil[y]) for x, y in allp)
        ims = list(ims)
        X = pool_embed(m, ims, dev, size=(CANVAS if cname == "rand" else None)); del ims
        add_rows(cname, "POOL", "pool", X, ("center", "full"))
        # 1. controls, from the photograph feature f = whitened PCA-16 part code of the same encoder's pooled output
        xa, xb, y = centres(cname, pl)
        Xo, Xa = {}, {}
        for c in ("center", "full"):
            f = Ts[c]["pool"]; fa, fb = f[ia], f[ib]
            Xo[c] = kron_sum(fa, fb, np.tile(R[0], (len(fa), 1)), np.tile(R[1], (len(fb), 1)))  # A is the left photograph in both conditions
            Xa[c] = kron_sum(fa, fb, pos_code(xa, y), pos_code(xb, y))
        add_rows(cname, "CTL_ORDER", "pool", Xo, ("center", "full"))
        add_rows(cname, "CTL_ABS", "pool", Xa, ("center", "full"))
    match = run_match(m, dev, seed, labels, splits["fresh"], images)
    rows.append(dict(model=name, seed=seed, cond="match", match=match, n=list(n)))
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="clip-b32-img+siglip-so400m+dinov2-l")
    ap.add_argument("--seeds", default="0+1+2")
    ap.add_argument("--outdir", default="results/e38")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    SMOKE = a.smoke
    E34.SMOKE = a.smoke
    E.ALPHAS = np.logspace(-3, 9, 25)
    os.makedirs(a.outdir, exist_ok=True)
    images, labels = V.load_images()
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            run(name, seed, images, labels, a.outdir)
