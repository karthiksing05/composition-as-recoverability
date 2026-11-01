"""E17: the behavior of image-text encoders on the composites whose states E8 reads.

E8 reads fresh photographs from the image states of CLIP and SigLIP, and quadrant pooling brings
kappa to 0.81-0.94.  Here the encoders' own behavior is scored: image-text matching of a composite
against captions.  For a fresh pair (x, y) under BESIDE (x left, y right) or ABOVE (x top, y bottom):
  roles     the correct caption beats the caption with the two objects' roles swapped, with the
            correct object named first in half the captions and second in the other half, so
            an order-of-mention shortcut scores 0.5 ("a photo of a cat on the left and a dog on
            the right" against "a photo of a dog on the left and a cat on the right")
  identity  the correct caption beats one with the second object replaced by another fresh class
            (the control: the encoder recognizes the photographs)
The classes and photographs are E8's fresh split for each seed; classes without a WordNet lemma in
timm's table are skipped.
"""
import argparse, io, json, os
import numpy as np
import torch
from PIL import Image
import e8_vision as V

MODELS = {"clip-b32": ("clip", "openai/clip-vit-base-patch32"),
          "clip-h14": ("clip", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
          "siglip-so400m": ("siglip", "google/siglip-so400m-patch14-384")}
SLOTS = {"beside": ("on the left", "on the right"), "above": ("at the top", "at the bottom")}
N_PAIRS = 1000


def class_names():
    import glob, timm
    info = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--zh-plus--tiny-imagenet/snapshots/*/dataset_infos.json"))[0]
    wnids = next(iter(json.load(open(info, encoding="utf-8")).values()))["features"]["label"]["names"]
    lem = {}
    with open(os.path.join(os.path.dirname(timm.__file__), "data/_info/imagenet_synset_to_lemma.txt"), encoding="utf-8") as f:
        for line in f:
            w, l = line.rstrip("\n").split("\t")
            lem[w] = l.split(",")[0].strip()
    return [lem.get(w) for w in wnids]


def caption(rel, first, second, first_slot):
    """Names `first` before `second`; first_slot = 0 puts `first` in slot 0 (left / top)."""
    s = SLOTS[rel]
    return f"a photo of a {first} {s[first_slot]} and a {second} {s[1 - first_slot]}"


def run(name, seed, dev, images, labels, names):
    from transformers import AutoProcessor, CLIPModel, SiglipModel
    kind, repo = MODELS[name]
    proc = AutoProcessor.from_pretrained(repo)
    mod = (CLIPModel if kind == "clip" else SiglipModel).from_pretrained(repo, torch_dtype=torch.float32).to(dev).eval()
    rng = np.random.default_rng(seed)  # E8's split: the same draws as e8_vision.run
    cls = rng.permutation(200)
    fresh_cls = [int(c) for c in cls[150:200] if names[c]]
    rng2 = np.random.default_rng(10_000 + seed)
    pool = {c: list(np.where(labels == c)[0]) for c in fresh_cls}

    @torch.no_grad()
    def text_emb(caps):
        kw = dict(padding="max_length", max_length=64) if kind == "siglip" else dict(padding=True)
        out = []
        for s in range(0, len(caps), 256):
            t = proc.tokenizer(caps[s:s + 256], return_tensors="pt", truncation=True, **kw).to(dev)
            e = mod.get_text_features(**t)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def image_emb(ims):
        out = []
        for s in range(0, len(ims), 64):
            p = proc.image_processor(ims[s:s + 64], return_tensors="pt").to(dev)
            e = mod.get_image_features(**p)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)

    res = []
    for rel in SLOTS:
        ca = rng2.choice(fresh_cls, N_PAIRS); cb = np.array([rng2.choice([c for c in fresh_cls if c != a]) for a in ca])
        cz = np.array([rng2.choice([c for c in fresh_cls if c not in (a, b)]) for a, b in zip(ca, cb)])
        ia = [int(rng2.choice(pool[c])) for c in ca]; ib = [int(rng2.choice(pool[c])) for c in cb]
        pil = lambda i: Image.open(io.BytesIO(images[i])).convert("RGB")
        I = image_emb([V.compose(rel, pil(a), pil(b)) for a, b in zip(ia, ib)])
        x = [names[c] for c in ca]; y = [names[c] for c in cb]; z = [names[c] for c in cz]
        order = np.arange(N_PAIRS) % 2  # 0: x named first, 1: y named first
        good = [caption(rel, x[i], y[i], 0) if o == 0 else caption(rel, y[i], x[i], 1) for i, o in enumerate(order)]
        swap = [caption(rel, y[i], x[i], 0) if o == 0 else caption(rel, x[i], y[i], 1) for i, o in enumerate(order)]
        wrong = [caption(rel, x[i], z[i], 0) if o == 0 else caption(rel, z[i], x[i], 1) for i, o in enumerate(order)]
        sg, ss, sw = [(I * text_emb(c)).sum(-1).cpu().numpy() for c in (good, swap, wrong)]
        res.append(dict(model=name, seed=seed, rel=rel, n_classes=len(fresh_cls),
                        roles=float((sg > ss).mean()), roles_first=float((sg > ss)[order == 0].mean()),
                        roles_second=float((sg > ss)[order == 1].mean()), identity=float((sg > sw).mean())))
        print(res[-1], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--outdir", default="results/e17")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    images, labels = V.load_images()
    names = class_names()
    for name in a.models.split(","):
        for seed in map(int, a.seeds.split(",")):
            fn = os.path.join(a.outdir, f"{name}_s{seed}.json")
            if os.path.exists(fn):
                continue
            r = run(name, seed, dev, images, labels, names)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
