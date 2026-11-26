"""E42: controls for the "moved photographs" image test (E34).
a. MLP probe (1 hidden layer, 256 ReLU, Adam, weight decay + early stopping on validation photographs) in place of ridge,
   on G2 and OBJr, conditions rand (moved) and fixed, at the E34-selected layer and the final layer.  References (R2_sum,
   R2_concat) are computed with the same probe; kappa_linref uses E34's linear references.  One network with 32 outputs
   (both roles' 16-d codes); weight decay and epoch chosen on mean validation R2.
b. Constructed order controls through OBJr (random slot order, role = left one): state built from f = whitened PCA-16
   part code of the photograph alone at the final layer (E34-selected code of rand/OBJr):
     ctl_sorted   [f_A ; f_B]  left-first slots (positive control: slot gives the role)
     ctl_noorder  [f_s1 ; f_s2] random slot order, no order tag
     ctl_add      [f + o r ; f + o r]  random slot order, o in {0,1} (left/right), r ~ N(0,I) fixed per seed
     ctl_mul      [f (x) e_o ; f (x) e_o]  random slot order, order one-hot tensor
   linear and MLP kappa, same references (concat / sum of the part codes at that layer).
c. Linear kappa at the E34-selected, middle (L//2) and final layers for G2 and OBJr, rand and fixed.
Splits, pairs, placements, codes, shuffles: identical to E34.run_system (same rng streams).
"""
import argparse, io, json, os
import numpy as np, pandas as pd
import torch
from PIL import Image
import e4_pretrained as E
import e8_vision as V
import e30_diffusion_photos as E30
import e34_relative_places as E34

CANVAS, PH = E34.CANVAS, E34.PH
SMOKE = False
CTL_ONLY = False
E34CSV = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e34/agg_per_seed.csv"
WDS = (1e-4, 1e-3, 1e-2, 1e-1)


@torch.no_grad()
def extract(m, ims, boxes, shuf, dev, bs=32, size=None):
    mod, size0, mean, std, kind = m
    size = size or size0
    kw = dict(interpolate_pos_encoding=True) if (kind in ("clip", "siglip") and size != size0) else {}
    out = {}
    for s in range(0, len(ims), bs):
        x = np.stack([(np.asarray(im.resize((size, size), Image.BICUBIC), np.float32) / 255 - mean) / std for im in ims[s:s + bs]])
        hs = mod(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2).to(dev), output_hidden_states=True, **kw).hidden_states
        L = len(hs) - 1
        for li in sorted({L // 4, L // 2, (3 * L) // 4, L}):
            p = hs[li].float()[:, E34.PREFIX[kind]:]
            g = int(round(p.shape[1] ** 0.5))
            f = p.reshape(len(p), g, g, -1).permute(0, 3, 1, 2)
            d = E34.readouts(f, boxes[s:s + bs], shuf[s:s + bs], (2,))
            for k in ("G2", "OBJr"):
                out.setdefault((k, f"L{li}"), []).append(d[k].cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def r2n(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def mlp_score(X, n, T, sp, seed, dev="cuda"):
    """X (N,D) rows fit,val,fresh; T target codes (all photographs); returns dict like E34.score row (rec_fresh_role1/2, role_fresh_*)."""
    nf, nv, nr = n
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    mu = X[:nf].mean(0); sd = X[:nf].std(0) + 1e-6
    X = ((X - mu) / sd).to(dev)
    Y = {k: np.concatenate([T[sp[k][r]] for r in (0, 1)], 1) for k in ("fit", "val", "fresh")}
    Ytr = torch.as_tensor(Y["fit"], dtype=torch.float32, device=dev)
    Xtr, Xv, Xr = X[:nf], X[nf:nf + nv], X[nf + nv:]
    k = T.shape[1]
    def val_r2(p):
        return np.mean([r2n(p[:, r * k:(r + 1) * k], Y["val"][:, r * k:(r + 1) * k]) for r in (0, 1)])
    best = (-1e9, None, None)
    for wd in WDS:
        torch.manual_seed(1234 + seed)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 2 * k)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
        bsz, bad, loc = 128, 0, -1e9
        for ep in range(150):
            perm = torch.randperm(nf, device=dev)
            for i in range(0, nf, bsz):
                j = perm[i:i + bsz]
                loss = ((net(Xtr[j]) - Ytr[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pv = net(Xv).cpu().numpy()
            s = val_r2(pv)
            if s > loc + 1e-4:
                loc, bad = s, 0
                with torch.no_grad():
                    pr_loc = net(Xr).cpu().numpy()
                pv_loc = pv
            else:
                bad += 1
                if bad >= 15:
                    break
        if loc > best[0]:
            best = (loc, pr_loc, dict(wd=wd, ep=ep))
    pr = best[1]
    row = {"mlp_wd": best[2]["wd"], "val_r2": best[0]}
    for r in (0, 1):
        tag = f"role{r + 1}"
        p = pr[:, r * k:(r + 1) * k]; y = Y["fresh"][:, r * k:(r + 1) * k]; yo = Y["fresh"][:, (1 - r) * k:(2 - r) * k]
        row[f"rec_fresh_{tag}"] = r2n(p, y)
        row[f"role_fresh_{tag}"] = float((((p - y) ** 2).sum(1) < ((p - yo) ** 2).sum(1)).mean())
    return row


def lin_score(X, n, T, sp, cidx):
    row = E34.score(torch.as_tensor(np.asarray(X)), n, {"c": T}, sp, cidx)["c"]
    row["rec_val_mean"] = (row["rec_val_role1"] + row["rec_val_role2"]) / 2
    return row


def mean_rec(row):
    return (row["rec_fresh_role1"] + row["rec_fresh_role2"]) / 2


def run(name, seed, images, labels, outdir):
    fn = os.path.join(outdir, f"{name}{'_abs' if CTL_ONLY else ''}_s{seed}.json")
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
    shuf = r2g.random(len(allp)) < 0.5
    cond = {
        "rand": ([E34.rand_canvas(pil[x], pil[y], *p) for (x, y), p in zip(allp, pl)], [E34.rand_box(*p) for p in pl]),
        "fixed": ([V.compose("beside", pil[x], pil[y]) for x, y in allp], [E34.FIXED_BOX] * len(allp)),
    }
    m = V.load(name, dev)
    codes = {"center": V.embed(m, [V.canvas([(pil[i], 112, (56, 56))]) for i in used], dev),
             "full": V.embed(m, [pil[i].resize((224, 224), Image.BICUBIC) for i in used], dev)}
    Ts = {cn: {l: E30.make_T(P, pca) for l, P in cd.items()} for cn, cd in codes.items()}
    sp = {k: [[idx[p[r]] for p in pairs[k]] for r in (0, 1)] for k in ("fit", "val", "fresh")}
    cidx = {"val": [idx[x] for x in splits["val"]], "fresh": [idx[x] for x in splits["fresh"]]}
    for k in ("val", "fresh"):
        sp[k + "_c"] = [np.array([splits[k].index(p[r]) for p in pairs[k]]) for r in (0, 1)]
    ia = [idx[x] for x, _ in allp]; ib = [idx[y] for _, y in allp]
    S = pd.read_csv(E34CSV)
    S = S[(S.model == name) & (S.seed == seed)]
    sel = {(r.cond, r.readout): (r.layer, r.code, r.kappa) for r in S.itertuples()}
    refs = {}

    def get_ref(layer, code, probe):
        key = (layer, code, probe)
        if key not in refs:
            P = codes[code][layer]; T = Ts[code][layer]
            out = {}
            for s_ in ("ref_concat", "ref_sum"):
                X = np.c_[P[ia], P[ib]] if s_ == "ref_concat" else P[ia] + P[ib]
                out[s_] = mean_rec(lin_score(X, n, T, sp, cidx) if probe == "linear" else mlp_score(X, n, T, sp, seed))
            refs[key] = out
        return refs[key]

    rows = []

    def record(tag, cname, rd, layer, code, X, T, probes, refcodes_layer=None, extra=None):
        for probe in probes:
            row = lin_score(X, n, T, sp, cidx) if probe == "linear" else mlp_score(X, n, T, sp, seed)
            rl, rc = refcodes_layer if refcodes_layer else (layer, code)
            rf = get_ref(rl, rc, probe); rfl = get_ref(rl, rc, "linear")
            R = mean_rec(row)
            rec = dict(model=name, seed=seed, cond=cname, readout=rd, layer=layer, layer_tag=tag, code=code, probe=probe, dim=int(X.shape[1]),
                       r2=R, ref_concat=rf["ref_concat"], ref_sum=rf["ref_sum"], kappa=(R - rf["ref_sum"]) / (rf["ref_concat"] - rf["ref_sum"]),
                       kappa_linref=(R - rfl["ref_sum"]) / (rfl["ref_concat"] - rfl["ref_sum"]),
                       role_acc=(row["role_fresh_role1"] + row["role_fresh_role2"]) / 2, mlp_wd=row.get("mlp_wd"))
            if extra:
                rec.update(extra)
            rows.append(rec)
            print(name, seed, cname, rd, tag, layer, probe, "kappa", round(rec["kappa"], 3), "linref", round(rec["kappa_linref"], 3), flush=True)

    feats = {}
    cnds = {} if CTL_ONLY else cond
    for cname, (ims, bx) in cnds.items():
        feats[cname] = extract(m, ims, bx, shuf, dev, size=(CANVAS if cname == "rand" else None))
    Ls = sorted({k for k in codes["center"] if k[0] == "L" and k[1:].isdigit()}, key=lambda s: int(s[1:]))
    mid, fin = Ls[1], Ls[-1]
    for cname in (() if CTL_ONLY else ("rand", "fixed")):
        for rd in ("G2", "OBJr"):
            sl, code, k34 = sel[(cname, rd)]
            done = {}
            for tag, layer in (("sel", sl), ("mid", mid), ("final", fin)):
                probes = ("linear", "mlp") if tag in ("sel", "final") else ("linear",)
                for pr_ in probes:
                    if (layer, pr_) in done:  # same layer already scored: copy under this tag
                        r_ = dict(done[(layer, pr_)]); r_["layer_tag"] = tag; rows.append(r_); continue
                    n0 = len(rows)
                    record(tag, cname, rd, layer, code, feats[cname][(rd, layer)], Ts[code][layer], (pr_,), extra=dict(kappa_e34=k34))
                    done[(layer, pr_)] = rows[n0]
    # b. constructed controls through OBJr, rand placements (slot order shuffled as OBJr), f = final-layer part code
    _, code, _ = sel[("rand", "OBJr")]
    T = Ts[code][fin]
    fa, fb = T[ia], T[ib]
    r = np.random.default_rng(55000 + seed).standard_normal(T.shape[1])
    sw = shuf[:, None]
    def slots(xa, xb, ordered=False):
        if ordered:
            return np.c_[xa, xb]
        return np.where(sw, np.c_[xb, xa], np.c_[xa, xb])
    ctl = {
        "ctl_sorted": slots(fa, fb, True),
        "ctl_noorder": slots(fa, fb),
        "ctl_add": slots(fa + 0 * r, fb + 1 * r),
        "ctl_mul": slots(np.c_[fa, 0 * fa], np.c_[0 * fb, fb]),
    }
    # absolute-position tags: continuous x of each box centre (fraction of canvas), no order information beyond x itself
    pa = np.array(pl, float)
    xa_, xb_ = (pa[:, 0] + PH / 2) / CANVAS, (pa[:, 1] + PH / 2) / CANVAS
    def qx(x):
        ks = np.arange(1, 5)[None]
        return np.c_[(x - .5) * np.sqrt(12), np.cos(2 * np.pi * ks * x[:, None]), np.sin(2 * np.pi * ks * x[:, None])]
    qa, qb = qx(xa_), qx(xb_)
    Wq = np.random.default_rng(56000 + seed).standard_normal((qa.shape[1], T.shape[1])) / np.sqrt(qa.shape[1])
    mulf = lambda f, q: (f[:, :, None] * q[:, None, :]).reshape(len(f), -1)
    new = {"ctl_absx": slots(fa + qa @ Wq, fb + qb @ Wq), "ctl_absx_mul": slots(mulf(fa, qa), mulf(fb, qb))}
    ctl = new if CTL_ONLY else {**ctl, **new}
    for nm, X in ctl.items():
        record("final", "rand", nm, fin, code, X, T, ("linear", "mlp"))
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--seeds", default="0+1+2")
    ap.add_argument("--outdir", default="/coc/flash10/ksingara3/compositionality-theory/experiments/results/e42")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ctl-only", action="store_true")
    a = ap.parse_args()
    SMOKE = a.smoke; E34.SMOKE = a.smoke; CTL_ONLY = a.ctl_only
    E.ALPHAS = np.logspace(-3, 9, 25)
    os.makedirs(a.outdir, exist_ok=True)
    images, labels = V.load_images()
    for seed in map(int, a.seeds.split("+")):
        run(a.system, seed, images, labels, a.outdir)
