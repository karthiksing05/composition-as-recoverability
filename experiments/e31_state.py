"""D2 state tests.
  --part a      text side: ridge readout of the figure noun's part code from the CLIP-L states the SD1.5 U-Net receives
                (token-mean, e4_pretrained.embed/RidgePath/codes), fit on canonical captions of fitting nouns, read on
                canonical (control) and converse captions of held-out nouns.  3 split seeds.
  --part bfeat  image side, step 1: U-Net up-block features at t=300 (fixed noise, empty caption) of the VAE/generation latent of
                every image in the given index files, pooled over the two halves along the relation's axis
                (declared in advance: top/bottom for vertical, left/right for horizontal); singles pooled over the whole map.
  --part bfit   image side, step 2: ridge readout (fit nouns' images, penalty and layer chosen on validation nouns) of the part code of the
                noun drawn in each half (from the detections), scored on held-out-noun D1 images."""
import argparse, json, os, glob
import numpy as np, torch
from e31_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True)
ap.add_argument("--thr", type=float, default=0.35)
ap.add_argument("--t", type=int, default=300)
a = ap.parse_args()
dev = "cuda"


def part_a():
    tok, mod = e4.load("sd15-clip-l", dev)
    sp0 = None
    out = []
    for seed in (0, 1, 2):
        sp = split(seed)
        fit_n, val_n, fr_n, allN = sp["fit"], sp["val"], sp["fresh"], sp["all"]
        idx = {x: i for i, x in enumerate(allN)}
        Pc = e4.embed(tok, mod, [f"a {x}" for x in allN], dev, "sd15-clip-l")  # layer -> (305, D)
        rng = np.random.default_rng(seed + 1000)
        for rel, (canon, conv, axis) in RELS.items():
            def samp(ns, k):
                prs = [(x, y) for x in ns for y in ns if x != y]
                return [prs[i] for i in rng.choice(len(prs), min(k, len(prs)), replace=False)]
            fit_p, val_p, fr_p = samp(fit_n, 3000), samp(val_n, 600), samp(fr_n, 1000)
            H = {}
            for nm, prs, tm in (("fit", fit_p, canon), ("val", val_p, canon), ("new_canon", fr_p, canon), ("new_conv", fr_p, conv)):
                H[nm] = e4.embed(tok, mod, [tm.format(x=x, y=y) for x, y in prs], dev, "sd15-clip-l")
            for layer in Pc:
                P = Pc[layer]
                pidx = [idx[x] for x in fit_n]
                pmu = P[pidx].mean(0)
                _, sv, Vt = np.linalg.svd(P[pidx] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:e4.K_PCA].T
                mu = H["fit"][layer].mean(0)
                tgt = lambda prs, j: T[[idx[p[j]] for p in prs]]
                reg = e4.RidgePath(H["fit"][layer] - mu, tgt(fit_p, 0)).select(H["val"][layer] - mu, tgt(val_p, 0))
                C = T[[idx[x] for x in fr_n]]
                tg = np.array([fr_n.index(p[0]) for p in fr_p])
                row = dict(seed=seed, rel=rel, layer=layer, alpha=float(reg.a), val_r2=e4.r2(reg.predict(H["val"][layer] - mu), tgt(val_p, 0)))
                for nm in ("new_canon", "new_conv"):
                    p = reg.predict(H[nm][layer] - mu)
                    y, yo = tgt(fr_p, 0), tgt(fr_p, 1)  # figure code, ground code
                    row[f"{nm}_r2_figure"] = e4.r2(p, y)
                    row[f"{nm}_share_nearer_figure"] = float((((p - y) ** 2).sum(1) < ((p - yo) ** 2).sum(1)).mean())
                    row[f"{nm}_id_figure"] = float((e4.cosmat(p, C).argmax(1) == tg).mean())
                    tg2 = np.array([fr_n.index(p_[1]) for p_ in fr_p])
                    row[f"{nm}_id_ground"] = float((e4.cosmat(p, C).argmax(1) == tg2).mean())
                out.append(row)
                print(row, flush=True)
    json.dump(out, open(f"{R31}/d2a_text.json", "w", encoding="utf-8"), indent=1)


def load_unet():
    from diffusers import UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    M = "runwayml/stable-diffusion-v1-5"
    unet = UNet2DConditionModel.from_pretrained(M, subfolder="unet", torch_dtype=torch.float16).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(M, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(M, subfolder="text_encoder", torch_dtype=torch.float16).to(dev).eval()
    sch = DDPMScheduler.from_pretrained(M, subfolder="scheduler")
    return unet, tok, te, sch


LAYERS = [0, 1, 2]


@torch.no_grad()
def part_bfeat():
    unet, tok, te, sch = load_unet()
    ids = tok([""], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(dev)
    emb = te(ids)[0]
    store = {}
    hooks = [unet.up_blocks[l].register_forward_hook(lambda m, i, o, l=l: store.__setitem__(l, o)) for l in LAYERS]
    g = torch.Generator("cpu").manual_seed(0)
    noise = torch.randn((1, 4, 64, 64), generator=g).to(dev).half()
    tt = torch.tensor([a.t], device=dev)
    for ixf in sorted(glob.glob(f"{R31}/index/sd15_*.json")):
        tag = os.path.basename(ixf)[:-5]
        outf = f"{R31}/state/feats_{tag}.npz"
        if os.path.exists(outf):
            continue
        os.makedirs(f"{R31}/state", exist_ok=True)
        recs = json.load(open(ixf, encoding="utf-8"))
        recs = [r for r in recs if os.path.exists(f"{R31}/latents/sd15/{r['file']}.npy")]
        F = {l: [] for l in LAYERS}
        for b in range(0, len(recs), 16):
            batch = recs[b:b + 16]
            z = torch.stack([torch.from_numpy(np.load(f"{R31}/latents/sd15/{r['file']}.npy")) for r in batch]).to(dev)
            zt = sch.add_noise(z.float(), noise.float().expand_as(z), tt.expand(len(z))).half()
            unet(zt, tt, encoder_hidden_states=emb.expand(len(z), -1, -1))
            for l in LAYERS:
                f = store[l][0] if isinstance(store[l], tuple) else store[l]
                f = f.float()  # (B,C,H,W)
                for i, r in enumerate(batch):
                    if r["kind"] == "single":
                        F[l].append(f[i].mean((1, 2))[None].repeat(2, 1).cpu().numpy())  # same vector twice (shape-compatible)
                    elif r["axis"] == "v":
                        h = f.shape[2] // 2
                        F[l].append(torch.stack([f[i, :, :h].mean((1, 2)), f[i, :, h:].mean((1, 2))]).cpu().numpy())
                    else:
                        h = f.shape[3] // 2
                        F[l].append(torch.stack([f[i, :, :, :h].mean((1, 2)), f[i, :, :, h:].mean((1, 2))]).cpu().numpy())
        np.savez(outf, files=np.array([r["file"] for r in recs]), **{f"L{l}": np.stack(F[l]) for l in LAYERS})
        print(tag, len(recs), {l: np.stack(F[l]).shape for l in LAYERS}, flush=True)


def part_bfit():
    import sys
    sp = split(0); im = imageable()
    idx = {x: i for i, x in enumerate(sp["all"])}

    def load_feats(tag):
        z = np.load(f"{R31}/state/feats_{tag}.npz")
        return {f: {l: z[f"L{l}"][i] for l in LAYERS} for i, f in enumerate(z["files"])}
    singles = load_feats("sd15_singles")
    Fs = {l: np.stack([singles[f"singles/{x}_s0"][l][0] for x in sp["all"]]) for l in LAYERS}
    # detections
    det = {}
    for fn in glob.glob(f"{R31}/det/det_sd15_*.jsonl"):
        for ln in open(fn, encoding="utf-8"):
            d = json.loads(ln); det[d["file"]] = d["det"]

    def samples(tags):
        """(half feature by layer, drawn noun, other noun, meta) for images with both nouns detected in different halves."""
        out = []
        for tag in tags:
            fe = load_feats(tag)
            for r in json.load(open(f"{R31}/index/{tag}.json", encoding="utf-8")):
                if r["file"] not in fe or r["file"] not in det:
                    continue
                d = det[r["file"]]
                if d[r["x"]]["score"] < a.thr or d[r["y"]]["score"] < a.thr:
                    continue
                k = 1 if r["axis"] == "v" else 0  # box (cx,cy,w,h): cy for vertical, cx for horizontal
                cx, cy = d[r["x"]]["box"][k], d[r["y"]]["box"][k]
                if not ((cx < .5) != (cy < .5)):
                    continue
                first, second = (r["x"], r["y"]) if cx < cy else (r["y"], r["x"])
                out.append(dict(r=r, first=first, second=second, f=fe[r["file"]]))
        return out
    S_fit = samples([t for t in ["sd15_fit_on", "sd15_fit_left"] if os.path.exists(f"{R31}/state/feats_{t}.npz")])
    S_val = samples([t for t in ["sd15_val_on", "sd15_val_left"] if os.path.exists(f"{R31}/state/feats_{t}.npz")])
    S_te = samples([f"sd15_pairs_{r}" for r in RELS])
    print("samples fit/val/test", len(S_fit), len(S_val), len(S_te), flush=True)
    pidx = [idx[x] for x in sp["fit"]]
    res = []
    for layer in LAYERS:
        P = Fs[layer]; pmu = P[pidx].mean(0)
        _, sv, Vt = np.linalg.svd(P[pidx] - pmu, full_matrices=False)
        T = (P - pmu) @ Vt[:e4.K_PCA].T
        def XY(S):
            X = np.concatenate([np.stack([s["f"][layer][0] for s in S]), np.stack([s["f"][layer][1] for s in S])])
            Y = T[[idx[s["first"]] for s in S] + [idx[s["second"]] for s in S]]
            O = T[[idx[s["second"]] for s in S] + [idx[s["first"]] for s in S]]
            return X, Y, O
        Xf, Yf, _ = XY(S_fit); Xv, Yv, _ = XY(S_val); Xt, Yt, Ot = XY(S_te)
        mu = Xf.mean(0)
        reg = e4.RidgePath(Xf - mu, Yf).select(Xv - mu, Yv)
        p = reg.predict(Xt - mu)
        acc = (((p - Yt) ** 2).sum(1) < ((p - Ot) ** 2).sum(1))
        cands = [x for x in im["new"]]
        C = T[[idx[x] for x in cands]]
        tg = np.array([cands.index(n) for n in [s["first"] for s in S_te] + [s["second"] for s in S_te]])
        idacc = (e4.cosmat(p, C).argmax(1) == tg)
        nt = len(S_te)
        rows = {}
        for nm, sel in (("all", np.ones(nt, bool)),) + tuple(
                (f"{rel}_{ph}", np.array([s["r"]["rel"] == rel and s["r"]["phr"] == ph for s in S_te])) for rel in RELS for ph in ("canon", "conv")):
            m = np.concatenate([sel, sel])
            rows[nm] = dict(n_images=int(sel.sum()), acc2=float(acc[m].mean()) if m.any() else None, kappa=float(2 * acc[m].mean() - 1) if m.any() else None,
                            id_acc=float(idacc[m].mean()) if m.any() else None, r2=float(e4.r2(p[m], Yt[m])) if m.sum() > 2 else None)
        res.append(dict(layer=layer, alpha=float(reg.a), val_r2=float(e4.r2(reg.predict(Xv - mu), Yv)), n_fit=len(S_fit), n_val=len(S_val), n_test=nt, thr=a.thr, rows=rows, chance_id=1 / len(cands)))
        print(layer, res[-1]["val_r2"], rows["all"], flush=True)
    json.dump(res, open(f"{R31}/d2b_image.json", "w", encoding="utf-8"), indent=1)


def part_asum():
    R = json.load(open(f"{R31}/d2a_text.json", encoding="utf-8"))
    keys = [k for k in R[0] if k.startswith("new_")] + ["val_r2"]
    out = {}
    for layer in sorted({r["layer"] for r in R}):
        out[layer] = {}
        for rel in list(RELS) + ["ALL"]:
            rows = [r for r in R if r["layer"] == layer and (rel == "ALL" or r["rel"] == rel)]
            per_seed = {s: [r for r in rows if r["seed"] == s] for s in (0, 1, 2)}
            out[layer][rel] = {k: dict(mean=float(np.mean([r[k] for r in rows])),
                                       seed_means=[float(np.mean([r[k] for r in per_seed[s]])) for s in (0, 1, 2)]) for k in keys}
    json.dump(out, open(f"{R31}/d2a_text_summary.json", "w", encoding="utf-8"), indent=1)
    for layer in out:
        o = out[layer]["ALL"]
        print(layer, {k: round(o[k]["mean"], 3) for k in ("new_canon_share_nearer_figure", "new_conv_share_nearer_figure", "new_canon_id_figure", "new_conv_id_figure", "new_conv_id_ground")},
              "seed means conv share", [round(v, 3) for v in o["new_conv_share_nearer_figure"]["seed_means"]], flush=True)


def part_bdiag():
    """Control for D2b: the same probe trained and tested on held-out-noun D1 images split BY NOUN (even/odd sorted index):
    train on pairs with both nouns in group g, test on pairs with both nouns in the other group. Thresholds 0.25, 0.35."""
    sp = split(0); im = imageable()
    idx = {x: i for i, x in enumerate(sp["all"])}
    z = np.load(f"{R31}/state/feats_sd15_singles.npz")
    sg = {f: {l: z[f"L{l}"][i] for l in LAYERS} for i, f in enumerate(z["files"])}
    Fs = {l: np.stack([sg[f"singles/{x}_s0"][l][0] for x in sp["all"]]) for l in LAYERS}
    det = {}
    for fn in glob.glob(f"{R31}/det/det_sd15_pairs_*.jsonl"):
        for ln in open(fn, encoding="utf-8"):
            d = json.loads(ln); det[d["file"]] = d["det"]
    new = sorted(im["new"]); grp = {x: i % 2 for i, x in enumerate(new)}
    out = []
    feats = {f"sd15_pairs_{rel}": np.load(f"{R31}/state/feats_sd15_pairs_{rel}.npz") for rel in RELS}
    for thr in (0.25, 0.35):
        S = []
        for rel in RELS:
            tag = f"sd15_pairs_{rel}"
            fe = {f: i for i, f in enumerate(feats[tag]["files"])}
            for r in json.load(open(f"{R31}/index/{tag}.json", encoding="utf-8")):
                if r["file"] not in fe or r["file"] not in det: continue
                d = det[r["file"]]
                if d[r["x"]]["score"] < thr or d[r["y"]]["score"] < thr: continue
                k = 1 if r["axis"] == "v" else 0
                cx, cy = d[r["x"]]["box"][k], d[r["y"]]["box"][k]
                if not ((cx < .5) != (cy < .5)): continue
                first, second = (r["x"], r["y"]) if cx < cy else (r["y"], r["x"])
                S.append(dict(first=first, second=second, i=fe[r["file"]], tag=tag, g=(grp[first], grp[second])))
        pidx = [idx[x] for x in sp["fit"]]
        for layer in LAYERS:
            P = Fs[layer]; pmu = P[pidx].mean(0)
            _, sv, Vt = np.linalg.svd(P[pidx] - pmu, full_matrices=False)
            T = (P - pmu) @ Vt[:e4.K_PCA].T
            for trg in (0, 1):
                tr = [s for s in S if s["g"] == (trg, trg)]; te = [s for s in S if s["g"] == (1 - trg, 1 - trg)]
                if len(tr) < 10 or len(te) < 10: continue
                def XY(L):
                    X = np.concatenate([np.stack([feats[s["tag"]][f"L{layer}"][s["i"]][0] for s in L]), np.stack([feats[s["tag"]][f"L{layer}"][s["i"]][1] for s in L])])
                    Y = T[[idx[s["first"]] for s in L] + [idx[s["second"]] for s in L]]
                    O = T[[idx[s["second"]] for s in L] + [idx[s["first"]] for s in L]]
                    return X, Y, O
                Xa, Ya, _ = XY(tr); Xt, Yt, Ot = XY(te)
                best = None
                for a_ in e4.ALPHAS[::2]:  # penalty by 4-fold CV on the training group
                    sc = []
                    for f in range(4):
                        m = np.arange(len(Xa)) % 4 == f
                        mu_ = Xa[~m].mean(0)
                        reg = e4.RidgePath(Xa[~m] - mu_, Ya[~m]); reg.a = a_; reg.W = reg.coef(a_)
                        sc.append(e4.r2(reg.predict(Xa[m] - mu_), Ya[m]))
                    if best is None or np.mean(sc) > best[0]: best = (float(np.mean(sc)), float(a_))
                mu = Xa.mean(0)
                reg = e4.RidgePath(Xa - mu, Ya); reg.a = best[1]; reg.W = reg.coef(best[1])
                p = reg.predict(Xt - mu)
                acc = float((((p - Yt) ** 2).sum(1) < ((p - Ot) ** 2).sum(1)).mean())
                out.append(dict(thr=thr, layer=layer, train_group=trg, n_train_images=len(tr), n_test_images=len(te), alpha=best[1], cv_r2=best[0], acc2=acc, kappa=2 * acc - 1,
                                r2=float(e4.r2(p, Yt))))
                print(out[-1], flush=True)
    json.dump(out, open(f"{R31}/d2b_control.json", "w", encoding="utf-8"), indent=1)


{"a": part_a, "bfeat": part_bfeat, "bfit": part_bfit, "asum": part_asum, "bdiag": part_bdiag}[a.part]()
