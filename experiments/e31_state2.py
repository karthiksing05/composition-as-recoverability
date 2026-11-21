"""D2b repaired. U-Net features of the VAE/generation latent noised at t in {100,300,500} (fixed noise, empty caption),
up-blocks 0,1,2, averaged INSIDE each detected noun's box (GroundingDINO, box threshold 0.35, IoU of the two boxes < 0.6).
Probe: ridge from the box feature to the noun's part code (= whole-map feature of a generated image of "a x", seed 0, same block and t;
16 PCA components fitted on the 165 fitting nouns), penalty chosen on validation nouns.  Fit: boxes of fit-noun images
("pairs": fit_on/fit_left images; "pairs+singles": plus single-object seed-1 images of fit nouns).  Test: boxes of all D1 images of held-out
nouns (3 relations, both phrasings).  Positive control: same probe, singles seed 1 (box-pooled) of fit nouns -> singles of held-out nouns.
(t, block, regime) chosen on validation nouns."""
import json, os, glob, sys
import numpy as np, torch
from e31_common import *

THR, IOU = 0.35, 0.6
TS, BL = (100, 300, 500), (0, 1, 2)
dev = "cuda"
sp = split(0); im = imageable()
idx = {x: i for i, x in enumerate(sp["all"])}
new = im["new"]; cand_idx = [idx[x] for x in new]


def iou(b, c):
    def xy(b): return b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    p, q = xy(b), xy(c)
    iw, ih = max(0, min(p[2], q[2]) - max(p[0], q[0])), max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = iw * ih
    return i / (b[2] * b[3] + c[2] * c[3] - i + 1e-9)


det = {}
for fn in glob.glob(f"{R31}/det/det_sd15_*.jsonl"):
    for ln in open(fn, encoding="utf-8"):
        d = json.loads(ln); det[d["file"]] = d["det"]

# ---- sample list: one entry per box -------------------------------------------------------------------------------
boxes = []  # dict(file, box, noun, other, role, slot, set, rel, phr)
def add_multi(tag, setname):
    for r in json.load(open(f"{R31}/index/{tag}.json", encoding="utf-8")):
        d = det.get(r["file"])
        if not d or not os.path.exists(f"{R31}/latents/sd15/{r['file']}.npy"): continue
        dx, dy = d[r["x"]], d[r["y"]]
        if dx["score"] < THR or dy["score"] < THR or iou(dx["box"], dy["box"]) >= IOU: continue
        k = 1 if r["axis"] == "v" else 0
        xfirst = dx["box"][k] < dy["box"][k]
        for nm, o, role, bx, first in ((r["x"], r["y"], "figure", dx["box"], xfirst), (r["y"], r["x"], "ground", dy["box"], not xfirst)):
            boxes.append(dict(file=r["file"], box=bx, noun=nm, other=o, role=role, slot="first" if first else "second", set=setname,
                              rel=r["rel"], phr=r["phr"], layout_ok=bool(xfirst)))
for tag, s in (("sd15_fit_on", "fit"), ("sd15_fit_left", "fit"), ("sd15_val_on", "val"), ("sd15_val_left", "val"),
               ("sd15_pairs_on", "test"), ("sd15_pairs_above", "test"), ("sd15_pairs_left", "test")):
    add_multi(tag, s)
for x in sp["all"]:  # singles seed 1: box-pooled inputs of the positive control
    f = f"singles/{x}_s1"; d = det.get(f)
    if d and d[x]["score"] >= THR:
        s = "fit" if x in sp["fit"] else "val" if x in sp["val"] else "test"
        if s == "test" and x not in new: continue
        boxes.append(dict(file=f, box=d[x]["box"], noun=x, other=None, role="single", slot="single", set=s, rel="single", phr="single", layout_ok=True))
files = sorted({b["file"] for b in boxes} | {f"singles/{x}_s0" for x in sp["all"]})
print("boxes", len(boxes), "images", len(files), {s: sum(b["set"] == s and b["role"] != "single" for b in boxes) for s in ("fit", "val", "test")}, flush=True)

# ---- features ------------------------------------------------------------------------------------------------------
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
M = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(M, subfolder="unet", torch_dtype=torch.float16).to(dev).eval()
tok = CLIPTokenizer.from_pretrained(M, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(M, subfolder="text_encoder", torch_dtype=torch.float16).to(dev).eval()
sch = DDPMScheduler.from_pretrained(M, subfolder="scheduler")
emb = te(tok([""], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(dev))[0]
store = {}
for l in BL:
    unet.up_blocks[l].register_forward_hook(lambda m, i, o, l=l: store.__setitem__(l, o[0] if isinstance(o, tuple) else o))
noise = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(dev)
byfile = {}
for i, b in enumerate(boxes): byfile.setdefault(b["file"], []).append(i)
FB = {(t, l): np.zeros((len(boxes), 0), np.float32) for t in TS for l in BL}
FB = {k: [None] * len(boxes) for k in FB}
CODE = {(t, l): {} for t in TS for l in BL}  # noun -> whole-map pooled feature of singles seed 0
with torch.no_grad():
    for t in TS:
        tt = torch.tensor([t], device=dev)
        for s in range(0, len(files), 16):
            fb = files[s:s + 16]
            z = torch.stack([torch.from_numpy(np.load(f"{R31}/latents/sd15/{f}.npy")) for f in fb]).to(dev)
            zt = sch.add_noise(z.float(), noise.expand_as(z.float()), tt.expand(len(z))).half()
            unet(zt, tt, encoder_hidden_states=emb.expand(len(z), -1, -1))
            for l in BL:
                f_ = store[l].float()
                H, W = f_.shape[2], f_.shape[3]
                for i, fname in enumerate(fb):
                    if fname.endswith("_s0") and fname.startswith("singles/"):
                        CODE[(t, l)][fname[8:-3]] = f_[i].mean((1, 2)).cpu().numpy()
                    for bi in byfile.get(fname, []):
                        cx, cy, w, h = boxes[bi]["box"]
                        x0, x1 = int(np.floor((cx - w / 2) * W)), int(np.ceil((cx + w / 2) * W))
                        y0, y1 = int(np.floor((cy - h / 2) * H)), int(np.ceil((cy + h / 2) * H))
                        x0, y0 = max(0, min(x0, W - 1)), max(0, min(y0, H - 1)); x1, y1 = max(x0 + 1, min(x1, W)), max(y0 + 1, min(y1, H))
                        FB[(t, l)][bi] = f_[i, :, y0:y1, x0:x1].mean((1, 2)).cpu().numpy()
        print("features t", t, flush=True)
FB = {k: np.stack(v) for k, v in FB.items()}
nb = len(boxes)
S = lambda name, cond: np.array([cond(b) for b in boxes])
is_multi = S("m", lambda b: b["role"] != "single")
setarr = np.array([b["set"] for b in boxes])
fit_nouns = sp["fit"]; pidx = [idx[x] for x in fit_nouns]
rng = np.random.default_rng(0)

res = dict(thr=THR, iou=IOU, n_boxes=nb, grid=[], note="see script docstring")
best = None
for (t, l), X in FB.items():
    P = np.stack([CODE[(t, l)][x] for x in sp["all"]])
    pmu = P[pidx].mean(0)
    _, sv, Vt = np.linalg.svd(P[pidx] - pmu, full_matrices=False)
    T = (P - pmu) @ Vt[:e4.K_PCA].T
    tgt = np.stack([T[idx[b["noun"]]] for b in boxes])
    C = T[cand_idx]

    def run(train_mask, val_mask, test_mask):
        mu = X[train_mask].mean(0)
        reg = e4.RidgePath(X[train_mask] - mu, tgt[train_mask]).select(X[val_mask] - mu, tgt[val_mask])
        p = reg.predict(X[test_mask] - mu)
        pv = reg.predict(X[val_mask] - mu)
        return reg, mu, p, e4.r2(pv, tgt[val_mask])

    for regime, trm, vam in (("pairs", (setarr == "fit") & is_multi, (setarr == "val") & is_multi),
                             ("pairs+singles", (setarr == "fit"), (setarr == "val")),
                             ("control_singles", (setarr == "fit") & ~is_multi, (setarr == "val") & ~is_multi)):
        tem = (setarr == "test") & (~is_multi if regime == "control_singles" else is_multi)
        reg, mu, p, vr2 = run(trm, vam, tem)
        ii = np.where(tem)[0]
        nouns_te = [boxes[i]["noun"] for i in ii]
        tg = np.array([new.index(n) for n in nouns_te])
        ident = (e4.cosmat(p, C).argmax(1) == tg)
        if regime == "control_singles":
            oth = [new[j] for j in rng.integers(0, len(new), len(ii))]
        else:
            oth = [boxes[i]["other"] for i in ii]
        own = T[[idx[n] for n in nouns_te]]; ot = T[[idx[n] for n in oth]]
        acc2 = (((p - own) ** 2).sum(1) < ((p - ot) ** 2).sum(1))
        row = dict(t=t, block=l, regime=regime, alpha=float(reg.a), val_r2=float(vr2), n_train=int(trm.sum()), n_val=int(vam.sum()), n_test=int(len(ii)),
                   ident=float(ident.mean()), acc2=float(acc2.mean()), kappa=float(2 * acc2.mean() - 1), r2_test=float(e4.r2(p, tgt[ii])), chance_ident=1 / len(new))
        meta = [boxes[i] for i in ii]
        sub = {}
        for nm, f_ in (("slot_first", lambda m: m["slot"] == "first"), ("slot_second", lambda m: m["slot"] == "second"), ("role_figure", lambda m: m["role"] == "figure"),
                       ("role_ground", lambda m: m["role"] == "ground"), ("layout_ok", lambda m: m["layout_ok"] and m["role"] != "single"),
                       ("layout_wrong", lambda m: (not m["layout_ok"]) and m["role"] != "single"), ("canon", lambda m: m["phr"] == "canon"), ("conv", lambda m: m["phr"] == "conv"),
                       ("on", lambda m: m["rel"] == "on"), ("above", lambda m: m["rel"] == "above"), ("left", lambda m: m["rel"] == "left")):
            m = np.array([f_(x) for x in meta])
            if m.sum() > 0:
                sub[nm] = dict(n=int(m.sum()), ident=float(ident[m].mean()), acc2=float(acc2[m].mean()))
        row["sub"] = sub
        res["grid"].append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "sub"}, flush=True)
# selection on validation R2, per regime
res["selected"] = {}
for regime in ("pairs", "pairs+singles", "control_singles"):
    rows = [r for r in res["grid"] if r["regime"] == regime]
    b = max(rows, key=lambda r: r["val_r2"])
    res["selected"][regime] = b
    print("SELECTED", regime, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items() if k != "sub"}, flush=True)
    print("   sub", {k: (v["n"], round(v["ident"], 3), round(v["acc2"], 3)) for k, v in b["sub"].items()}, flush=True)
json.dump(res, open(f"{R31}/d2b_boxpool.json", "w", encoding="utf-8"), indent=1)
