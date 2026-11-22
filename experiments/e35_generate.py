"""E35 generation (E31's D1 test on PixArt-Sigma and Lumina-Image-2.0).  Same held-out imageable nouns, pair draws, canonical
and converse captions, and shared initial noise per (pair, seed) as e31_diffusion_generate.py; images go to
results/e31/images/<model>/ with an index in results/e31/index/, so e31_score.py and e31_analyze.py read them unchanged.
Each model runs at 1024 px with its own default sampler: PixArt-Sigma 20 DPM-Solver steps at guidance 4.5,
Lumina-Image-2.0 30 flow steps at guidance 4.0 (its pipeline defaults, with CFG on throughout).
usage: e35_generate.py --model pixart|lumina --rel on|above|left [--npairs 150 --seeds 2]"""
import argparse, json, os
import numpy as np, torch
from PIL import Image
from e31_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--rel", required=True)
ap.add_argument("--npairs", type=int, default=150)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
SZ = 1024
if a.model == "pixart":
    from diffusers import PixArtSigmaPipeline
    pipe = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16, local_files_only=True).to("cuda")
    kw0 = dict(num_inference_steps=20, guidance_scale=4.5)
else:
    from diffusers import Lumina2Pipeline
    pipe = Lumina2Pipeline.from_pretrained("Alpha-VLLM/Lumina-Image-2.0", torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    kw0 = dict(num_inference_steps=30, guidance_scale=4.0, cfg_trunc_ratio=1.0, cfg_normalization=True)
pipe.set_progress_bar_config(disable=True)

im = imageable()
canon, conv, axis = RELS[a.rel]
prs = sample_pairs(im["new"], a.npairs, 100 + list(RELS).index(a.rel))  # E31's draw for mode "pairs"
recs = []
for pi, (x, y) in enumerate(prs):
    for s in range(a.seeds):
        for phr, tmpl in (("canon", canon), ("conv", conv)):
            recs.append(dict(file=f"pairs/{a.rel}/p{pi:03d}_{phr}_s{s}", rel=a.rel, axis=axis, pair=pi, phr=phr, seed=s, x=x, y=y,
                             first=x if phr == "canon" else y, caption=tmpl.format(x=x, y=y), kind="pairs", model=a.model))
if a.limit:
    recs = recs[:a.limit]
tag = f"{a.model}_pairs_{a.rel}"
os.makedirs(f"{R31}/index", exist_ok=True)
with open(f"{R31}/index/{tag}.json", "w", encoding="utf-8") as f:
    json.dump(recs, f)
todo = [r for r in recs if not os.path.exists(f"{R31}/images/{a.model}/{r['file']}.png")]
print(tag, len(recs), "records,", len(todo), "to do", flush=True)
for b in range(0, len(todo), a.bs):
    batch = todo[b:b + a.bs]
    gens = [torch.Generator("cuda").manual_seed(int(r["seed"]) + 12345) for r in batch]  # same noise for both phrasings
    out = pipe(prompt=[r["caption"] for r in batch], generator=gens, height=SZ, width=SZ, **kw0).images
    for r, img in zip(batch, out):
        fn = f"{R31}/images/{a.model}/{r['file']}.png"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        img.resize((384, 384), Image.LANCZOS).save(fn, compress_level=6)
    print(b + len(batch), "/", len(todo), flush=True)
