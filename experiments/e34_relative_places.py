"""E34: does a linear probe fixed in advance recover the ROLE when absolute location no longer gives it away?

Control for the quadrant-pooling objection of E8/E30.  Two photographs (figure A, ground B) go on a larger
canvas under one relation, "A is the LEFT one", at random absolute positions:
  canvas CANVAS x CANVAS (448; encoders read it at 448 px with interpolated position embeddings, SD at 512), photographs PH x PH (64, native); both on one row y ~ U[0, CANVAS-PH]; two left edges uniform on [0, CANVAS-PH],
  rejected if they overlap (|dx| < PH); the smaller edge is A (figure).  Role = relation (left one = figure).
Readouts of the same state (a linear probe fitted in advance, E4's ridge with the penalty chosen on validation nouns):
  M      whole-map mean (E8's mean; encoders include the class token)
  G2     mean in the 4 fixed quadrants of the canvas (E8_grid's readout; location-aligned pooling)
  Q4     the 4 x 4 grid of cell means, flattened (no object alignment; 16 C features)
  Q8     the 8 x 8 grid (encoders only; 64 C features)
  OBJs   "given locations, sorted": features pooled inside each photograph's known box, left box first
         (the slot gives the role away; upper reference for whether parts survive intact)
  OBJr   "given locations, random slot order": same two vectors, slots shuffled at random; the probe must
         say which one is the figure, so it has to read the relation out of the features
Reference rows: the same readouts on E8's fixed BESIDE canvas (224, A left half, B right half).
Part codes, targets (PCA-16, whitened), splits, ridge penalty grid, kappa and references are E8/E30's:
kappa = (R2 - R2_sum) / (R2_concat - R2_sum) on photographs of fresh classes; layer (and timestep, code)
chosen per readout on validation nouns.  Also reported: role accuracy (the probe's output is nearer the true
part than the other part), identity accuracy among fresh photographs.
"""
import argparse, io, json, os
import numpy as np
import torch
from PIL import Image
import e4_pretrained as E
import e8_vision as V
import e30_diffusion_photos as E30

CANVAS, PH = 448, 64
PREFIX = {"clip": 1, "dino": 1, "siglip": 0}
SMOKE = False


# ---------------------------------------------------------------- stimuli
def place(rng):
    while True:  # two left edges uniform on [0, CANVAS-PH], rejected if the photographs overlap; the smaller is the figure
        u, v = (int(rng.integers(0, CANVAS - PH + 1)) for _ in range(2))
        if abs(u - v) >= PH:
            break
    y = int(rng.integers(0, CANVAS - PH + 1))
    return min(u, v), max(u, v), y


def rand_canvas(a, b, xa, xb, y):
    c = Image.new("RGB", (CANVAS, CANVAS), V.BG)
    c.paste(a.resize((PH, PH), Image.BICUBIC), (xa, y))
    c.paste(b.resize((PH, PH), Image.BICUBIC), (xb, y))
    return c


def rand_box(xa, xb, y):  # (2, 4) fractional x0,y0,x1,y1 of figure and ground
    S = float(CANVAS)
    return [[xa / S, y / S, (xa + PH) / S, (y + PH) / S], [xb / S, y / S, (xb + PH) / S, (y + PH) / S]]


FIXED_BOX = [[0, .25, .5, .75], [.5, .25, 1, .75]]  # V.compose("beside"): 112 px photographs at y 56..168 of 224


# ---------------------------------------------------------------- readouts of a feature map
def readouts(f, boxes, shuf, grids):
    """f: (B, C, h, w) float; boxes: (B, 2, 4); shuf: (B,) bool.  -> dict readout -> (B, dim)."""
    B, C, h, w = f.shape
    out = {}
    for g in grids:
        out[f"G2" if g == 2 else f"Q{g}"] = torch.nn.functional.adaptive_avg_pool2d(f, g).flatten(1)
    dev = f.device
    bx = torch.as_tensor(np.asarray(boxes, np.float32), device=dev)
    def rng_mask(lo, hi, n):
        i0 = torch.floor(lo * n + 1e-4).long(); i1 = torch.ceil(hi * n - 1e-4).long()
        i1 = torch.maximum(i1, i0 + 1)
        ar = torch.arange(n, device=dev)[None, None]
        return ((ar >= i0[..., None]) & (ar < i1[..., None])).float()
    mx = rng_mask(bx[..., 0], bx[..., 2], w); my = rng_mask(bx[..., 1], bx[..., 3], h)
    mask = my[..., :, None] * mx[..., None, :]
    mask = mask / mask.sum((2, 3), keepdim=True)
    obj = torch.einsum("bkhw,bchw->bkc", mask, f)
    sw = torch.as_tensor(shuf, device=dev)[:, None]
    out["OBJs"] = torch.cat([obj[:, 0], obj[:, 1]], 1)
    out["OBJr"] = torch.where(sw, torch.cat([obj[:, 1], obj[:, 0]], 1), out["OBJs"])
    return out


@torch.no_grad()
def enc_extract(m, ims, boxes, shuf, dev, bs=32, size=None):
    mod, size0, mean, std, kind = m
    size = size or size0
    kw = dict(interpolate_pos_encoding=True) if (kind in ("clip", "siglip") and size != size0) else {}
    out = {}
    for s in range(0, len(ims), bs):
        x = np.stack([(np.asarray(im.resize((size, size), Image.BICUBIC), np.float32) / 255 - mean) / std for im in ims[s:s + bs]])
        hs = mod(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2).to(dev), output_hidden_states=True, **kw).hidden_states
        L = len(hs) - 1
        for li in sorted({L // 4, L // 2, (3 * L) // 4, L}):
            h = hs[li].float()
            p = h[:, PREFIX[kind]:]
            g = int(round(p.shape[1] ** 0.5))
            f = p.reshape(len(p), g, g, -1).permute(0, 3, 1, 2)
            d = readouts(f, boxes[s:s + bs], shuf[s:s + bs], (2, 4, 8))
            d["M"] = h.mean(1)
            for k, v in d.items():
                out.setdefault((k, f"L{li}"), []).append(v.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def sd_extract(D, ims, boxes, shuf, ts, bs=24):
    out, cap, hooks = {}, {}, []
    u = D.unets["trained"]
    for name, b in [("mid", u.mid_block)] + [(f"up{i}", b) for i, b in enumerate(u.up_blocks)]:
        hooks.append(b.register_forward_hook(lambda mm, i, o, name=name: cap.__setitem__(name, o[0] if isinstance(o, tuple) else o)))
    g = torch.Generator().manual_seed(E30.NOISE_SEED)
    h = D.res // 8
    eps = torch.randn(1, 4, h, h, generator=g).to(D.dev)
    pe, added = D.cond([""], bs)
    for s0 in range(0, len(ims), bs):
        x0 = D.latents(ims[s0:s0 + bs]); B_ = len(x0)
        for t in ts:
            xt = D.noisy(x0, t, eps).to(u.dtype)
            u(xt, torch.full((B_,), t, device=D.dev), encoder_hidden_states=pe[:B_].to(u.dtype),
              added_cond_kwargs={k: v[:B_].to(u.dtype) for k, v in added.items()} if added else None)
            for name, f in cap.items():
                f = f.float()
                d = readouts(f, boxes[s0:s0 + bs], shuf[s0:s0 + bs], (2, 4))
                d["M"] = f.mean((2, 3))
                for k, v in d.items():
                    out.setdefault((k, f"{t}_{name}"), []).append(v.half().cpu())
    for hk in hooks:
        hk.remove()
    return {k: torch.cat(v) for k, v in out.items()}


# ---------------------------------------------------------------- probes (E4's ridge, exact, via Gram / covariance eigendecomposition)
def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def cosm(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12); b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


@torch.no_grad()
def probe(X, n, targets):
    """X (n_fit + n_val + n_fresh, D) rows ordered fit, val, fresh.  n = (nfit, nval, nfresh).
    targets: list of dicts with ytr, yv (to fit / select on) and the fresh targets are scored by caller from the returned predictions.
    returns, per target, (alpha, pred_val, pred_fresh)."""
    nf, nv, nr = n
    dev = "cuda"
    Xt = X[:nf].to(dev).float(); mu = Xt.mean(0); Xt = Xt - mu
    Xv = X[nf:nf + nv].to(dev).float() - mu; Xr = X[nf + nv:].to(dev).float() - mu
    if Xt.shape[1] <= nf:
        l, Vv = torch.linalg.eigh((Xt.T @ Xt).double()); l = l.clamp(min=0)
        Av, Ar = (Xv.double() @ Vv), (Xr.double() @ Vv); proj = lambda Yc: Vv.T @ (Xt.T.double() @ Yc)
    else:
        l, U = torch.linalg.eigh((Xt @ Xt.T).double()); l = l.clamp(min=0)
        Av, Ar = (Xv @ Xt.T).double() @ U, (Xr @ Xt.T).double() @ U; proj = lambda Yc: U.T @ Yc
    res = []
    for tg in targets:
        ytr = torch.as_tensor(tg["ytr"], device=dev).double(); ym = ytr.mean(0)
        b = proj(ytr - ym)
        sc, best = [], None
        for a in E.ALPHAS:
            pv = (Av * (1.0 / (l + a))[None]) @ b + ym
            s = r2(pv.cpu().numpy(), tg["yv"])
            sc.append(s)
        a = E.ALPHAS[int(np.argmax(sc))]
        w = 1.0 / (l + a)
        res.append((float(a), ((Av * w[None]) @ b + ym).cpu().numpy(), ((Ar * w[None]) @ b + ym).cpu().numpy()))
    del Xt, Xv, Xr, Av, Ar
    torch.cuda.empty_cache()
    return res


def score(X, n, Tm, sp, cands_idx):
    """Tm: {code: T}.  For each code and role, ridge from X to the T-codes; returns {code: row with rec/role/id on val and fresh}."""
    tg, keys = [], []
    for code, T in Tm.items():
        for role in (0, 1):
            tg.append(dict(ytr=T[sp["fit"][role]], yv=T[sp["val"][role]])); keys.append((code, role))
    pr = dict(zip(keys, probe(X, n, tg)))
    rows = {}
    for code, T in Tm.items():
        row = {}
        for role in (0, 1):
            a, pv, pf = pr[(code, role)]; tag = f"role{role + 1}"
            row[f"alpha_{tag}"] = a
            for split, p in (("val", pv), ("fresh", pf)):
                y = T[sp[split][role]]; yo = T[sp[split][1 - role]]
                row[f"rec_{split}_{tag}"] = r2(p, y)
                row[f"role_{split}_{tag}"] = float((((p - y) ** 2).sum(1) < ((p - yo) ** 2).sum(1)).mean())
                C = T[cands_idx[split]]
                row[f"id_{split}_{tag}"] = float((cosm(p, C).argmax(1) == sp[split + "_c"][role]).mean())
        rows[code] = row
    return rows


# ---------------------------------------------------------------- main
def make_T(P, pca):
    return E30.make_T(P, pca)


def run_system(name, seed, images, labels, outdir, D=None):
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

    def sample(ns, k):
        prs = [(x, y) for x in ns for y in ns if x != y]
        return [prs[i] for i in rng.choice(len(prs), min(int(k), len(prs)), replace=False)]
    pairs = {"fit": sample(splits["fit"], 3000 * sc), "val": sample(splits["val"], 600 * sc), "fresh": sample(splits["fresh"], 1000 * sc)}
    n = tuple(len(pairs[k]) for k in ("fit", "val", "fresh"))
    allp = pairs["fit"] + pairs["val"] + pairs["fresh"]
    r2g = np.random.default_rng(7000 + seed)
    pl = [place(r2g) for _ in allp]
    shuf = r2g.random(len(allp)) < 0.5
    cond = {
        "rand": ([rand_canvas(pil[x], pil[y], *p) for (x, y), p in zip(allp, pl)], [rand_box(*p) for p in pl]),
        "fixed": ([V.compose("beside", pil[x], pil[y]) for x, y in allp], [FIXED_BOX] * len(allp)),
    }
    # location-only diagnostic: best single threshold on one photograph's absolute centre x predicting "figure"
    xs = np.array([[p[0] + PH / 2, p[1] + PH / 2] for p in pl])
    cx = np.r_[xs[:, 0], xs[:, 1]]; lab = np.r_[np.ones(len(xs)), np.zeros(len(xs))]
    thr = np.sort(cx); acc_loc = max(max(((cx < t) == lab).mean(), ((cx >= t) == lab).mean()) for t in thr[::max(1, len(thr) // 400)])
    meta_loc = dict(loc_only_role_acc=float(acc_loc), mean_gap=float(np.mean([p[1] - p[0] - PH for p in pl])))
    # part codes and which layers
    if D is not None:  # a diffusion model (sd15 here; E35 passes its transformer denoisers)
        ts = (300,) if SMOKE else E30.TS
        alone = {"center": [V.canvas([(pil[i], 112, (56, 56))]) for i in used], "full": [pil[i].resize((224, 224), Image.BICUBIC) for i in used]}
        codes = {}
        for cn, ims in alone.items():
            e = E30.extract(D, ims, ["trained"], ts)["trained"]
            codes[cn] = {k[1:]: v for k, v in e.items() if k.startswith("M")}  # key "<t>_<blk>"
        ext = lambda cn, ims, bx: sd_extract(D, ims, bx, shuf, ts)
    else:
        m = V.load(name, dev)
        codes = {"center": V.embed(m, [V.canvas([(pil[i], 112, (56, 56))]) for i in used], dev),
                 "full": V.embed(m, [pil[i].resize((224, 224), Image.BICUBIC) for i in used], dev)}
        ext = lambda cn, ims, bx: enc_extract(m, ims, bx, shuf, dev, size=(CANVAS if cn == "rand" else None))
    Ts = {cn: {l: make_T(P, pca) for l, P in cd.items()} for cn, cd in codes.items()}

    def sp_for(T_unused=None):
        sp = {}
        for k in ("fit", "val", "fresh"):
            sp[k] = [[idx[p[r]] for p in pairs[k]] for r in (0, 1)]
        cidx = {"val": [idx[x] for x in splits["val"]], "fresh": [idx[x] for x in splits["fresh"]]}
        for k in ("val", "fresh"):
            sp[k + "_c"] = [np.array([splits[k].index(p[r]) for p in pairs[k]]) for r in (0, 1)]
        return sp, cidx
    sp, cidx = sp_for()
    rows, refs = [], {}
    for cname, ims_boxes in cond.items():
        feats = ext(cname, *ims_boxes)
        print(name, seed, cname, "features", len(feats), flush=True)
        for (rd, layer), X in feats.items():
            Tm = {code: Ts[code][layer] for code in ("center", "full")}
            if layer not in refs:
                refs[layer] = {code: {} for code in Tm}
                for s in ("ref_concat", "ref_sum"):
                    for code in Tm:
                        P = codes[code][layer]
                        A = P[[idx[x] for x, _ in allp]]; Bm = P[[idx[y] for _, y in allp]]
                        refs[layer][code][s] = score(torch.as_tensor(np.c_[A, Bm] if s == "ref_concat" else A + Bm), n, {code: Tm[code]}, sp, cidx)[code]
            sc_ = score(X, n, Tm, sp, cidx)
            for code in Tm:
                row = dict(model=name, seed=seed, cond=cname, readout=rd, layer=layer, code=code, dim=int(X.shape[1]))
                row.update(sc_[code])
                rf = refs[layer][code]
                for r_, tag in ((1, "role1"), (2, "role2"), (0, "mean")):
                    g = (lambda d, k: d[f"{k}_{tag}"]) if r_ else (lambda d, k: (d[f"{k}_role1"] + d[f"{k}_role2"]) / 2)
                    c_, s_ = g(rf["ref_concat"], "rec_fresh"), g(rf["ref_sum"], "rec_fresh")
                    row[f"kappa_{tag}"] = (g(row, "rec_fresh") - s_) / (c_ - s_)
                    row[f"ref_concat_{tag}"], row[f"ref_sum_{tag}"] = c_, s_
                rows.append(row)
            print(name, seed, cname, rd, layer, "done", flush=True)
        del feats
    rows.append(dict(model=name, seed=seed, cond="meta", **meta_loc, n=list(n), canvas=CANVAS, photo=PH))
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--seeds", default="0+1+2")
    ap.add_argument("--outdir", default="results/e34")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    SMOKE = a.smoke
    E.ALPHAS = np.logspace(-3, 9, 25)
    os.makedirs(a.outdir, exist_ok=True)
    images, labels = V.load_images()
    D = E30.Diff("sd15", 512, "cuda") if a.system == "sd15" else None
    for seed in map(int, a.seeds.split("+")):
        run_system(a.system, seed, images, labels, a.outdir, D)
