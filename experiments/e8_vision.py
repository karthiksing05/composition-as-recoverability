"""E8: recoverability in pretrained image encoders.

Parts are natural photographs (Tiny-ImageNet, 64 x 64, 200 classes).  A composite places two
photographs on one 224 x 224 grey canvas under one of three relations:
  beside   x in the left half, y in the right half (112 x 112 each)
  above    x in the top half, y in the bottom half
  overlap  x at 168 x 168 in the top-left, y at 112 x 112 in the bottom-right, covering part of x
A part's own representation is the encoder's state for the photograph alone ("center": at
112 x 112 in the middle of the canvas; "full": filling the canvas).  The classes are split per
seed into 120 fitting, 30 validation and 50 fresh classes, so fresh parts are new photographs of
classes the readout never saw.  The readouts, references (concatenation and sum of the same part
codes), whitened 16-PC targets, penalty selection on validation parts, delta and the 12 x 12
interaction grid are those of E4 (e4_pretrained.evaluate).
"""
import argparse, io, json, os, glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
import e4_pretrained as E

N_CLS = (120, 30, 50)
PER_CLS = (5, 5, 6)
BG = (124, 116, 104)
RELATIONS = ("beside", "above", "overlap")
CODES = ("center", "full")
SPECS = {  # name: (loader, repo, input size)
    "clip-b32-img": ("clip", "openai/clip-vit-base-patch32", 224),
    "clip-h14-img": ("clip", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 224),
    "dinov2-s": ("dino", "facebook/dinov2-small", 224),
    "dinov2-l": ("dino", "facebook/dinov2-large", 224),
    "dinov2-g": ("dino", "facebook/dinov2-giant", 224),
    "siglip-so400m": ("siglip", "google/siglip-so400m-patch14-384", 384),
}


def load_images():
    f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--zh-plus--tiny-imagenet/snapshots/*/data/train-*.parquet"))[0]
    df = pd.read_parquet(f)
    return df["image"].map(lambda d: d["bytes"]).tolist(), df["label"].to_numpy()


def load(name, dev):
    from transformers import AutoImageProcessor, CLIPVisionModel, AutoModel, SiglipVisionModel
    rand = name.startswith("rand-")
    kind, repo, size = SPECS[name[5:] if rand else name]
    proc = AutoImageProcessor.from_pretrained(repo)
    cls = {"clip": CLIPVisionModel, "dino": AutoModel, "siglip": SiglipVisionModel}[kind]
    mod = cls.from_pretrained(repo, torch_dtype=torch.float32)
    if rand:
        torch.manual_seed(0)
        mod = type(mod)(mod.config)
    mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
    return (mod.to(dev).eval(), size, mean, std, kind)


def canvas(parts):
    """parts: list of (PIL image, size, (left, top)); later parts are drawn over earlier ones."""
    c = Image.new("RGB", (224, 224), BG)
    for im, s, xy in parts:
        c.paste(im.resize((s, s), Image.BICUBIC), xy)
    return c


def compose(rel, a, b):
    if rel == "beside":
        return canvas([(a, 112, (0, 56)), (b, 112, (112, 56))])
    if rel == "above":
        return canvas([(a, 112, (56, 0)), (b, 112, (56, 112))])
    return canvas([(a, 168, (0, 0)), (b, 112, (112, 112))])


@torch.no_grad()
def embed(m, ims, dev):
    mod, size, mean, std, kind = m
    out = {}
    for s in range(0, len(ims), 128):
        x = np.stack([(np.asarray(im.resize((size, size), Image.BICUBIC), np.float32) / 255 - mean) / std for im in ims[s:s + 128]])
        o = mod(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2).to(dev), output_hidden_states=True)
        hs = o.hidden_states
        L = len(hs) - 1
        for li in sorted({L // 4, L // 2, (3 * L) // 4, L}):
            out.setdefault(f"L{li}", []).append(hs[li].mean(1).float().cpu().numpy())
        out.setdefault("pooled", []).append(o.pooler_output.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def run(name, seed, dev, images, labels):
    rng = np.random.default_rng(seed)
    cls = rng.permutation(200)
    splits, s0 = {}, 0
    for k, n, per in zip(("fit", "val", "fresh"), N_CLS, PER_CLS):
        ids = []
        for c in cls[s0:s0 + n]:
            ids += list(rng.choice(np.where(labels == c)[0], per, replace=False))
        splits[k] = [int(i) for i in ids]; s0 += n
    used = splits["fit"] + splits["val"] + splits["fresh"]
    pil = {i: Image.open(io.BytesIO(images[i])).convert("RGB") for i in used}
    m = load(name, dev)
    codes = {"center": embed(m, [canvas([(pil[i], 112, (56, 56))]) for i in used], dev),
             "full": embed(m, [pil[i].resize((224, 224), Image.BICUBIC) for i in used], dev)}
    idx = {x: j for j, x in enumerate(used)}
    pca = list(range(len(splits["fit"])))
    res = []
    for rel in RELATIONS:
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
        H = embed(m, [compose(rel, pil[x], pil[y]) for x, y in allp], dev)
        cands = {"val": splits["val"], "fresh": splits["fresh"]}
        for layer, Hall in H.items():
            for cname in CODES:
                P = codes[cname][layer]
                pmu = P[pca].mean(0)
                _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
                T = (P - pmu) @ Vt[:E.K_PCA].T / (sv[:E.K_PCA] / np.sqrt(len(pca)))
                A = P[[idx[x] for x, _ in allp]]; B = P[[idx[y] for _, y in allp]]
                for sname, Hs in {"model": Hall, "ref_concat": np.c_[A, B], "ref_sum": A + B}.items():
                    row = dict(model=name, seed=seed, rel=rel, layer=layer, code=cname, system=sname, dim=int(Hs.shape[1]))
                    row.update(E.evaluate(Hs, T, idx, sets, cands))
                    res.append(row)
                r = res[-3]
                print(name, rel, layer, cname, round((r["rec_fresh_role1"] + r["rec_fresh_role2"]) / 2, 3), flush=True)
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
    images, labels = load_images()
    for name in a.models.split("+"):
        for seed in map(int, a.seeds.split("+")):
            fn = os.path.join(a.outdir, f"{name}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(name, seed, dev, images, labels)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
