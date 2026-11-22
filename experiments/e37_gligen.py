"""E37: a layout harness for text-to-image generation (plan, then draw each part in its box).

For every E31 caption (held-out imageable nouns, canonical and converse phrasing, relations on / above / left, 150 pairs,
seeds 0 and 1 with E31's initial noise) GLIGEN (SD 1.5 with box grounding, masterful/gligen-1-4-generation-text-box)
draws BOTH layouts: A puts the figure x in the first slot (top / left) and B puts the ground y there.  A planner then
picks one layout per caption; the harness's behavior is the image of the layout it picked.  Because every planner
chooses between the same two images, planners differ only in their role assignment (analysis: e37_analyze.py).
Boxes (x0, y0, x1, y1, fractions of the canvas), first slot / second slot:
  vertical   [.25 .06 .75 .46] / [.15 .50 .85 .94]       horizontal  [.04 .25 .46 .75] / [.54 .25 .96 .75]
The prompt is the caption itself; the grounding phrases are the two nouns.  50 PNDM steps, guidance 7.5,
gligen_scheduled_sampling_beta = 1 (grounding at every step).  Images go to results/e31/images/gligen/ (384 px), with an
index that e31_score.py reads.  usage: e37_gligen.py --rel on --layout A"""
import argparse, json, os
import numpy as np, torch
from PIL import Image
from e31_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--rel", required=True)
ap.add_argument("--layout", required=True, choices=["A", "B"])
ap.add_argument("--npairs", type=int, default=150)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
BOX = {"v": ([.25, .06, .75, .46], [.15, .50, .85, .94]), "h": ([.04, .25, .46, .75], [.54, .25, .96, .75])}

from diffusers import StableDiffusionGLIGENPipeline
pipe = StableDiffusionGLIGENPipeline.from_pretrained("masterful/gligen-1-4-generation-text-box", torch_dtype=torch.float16,
                                                     safety_checker=None, local_files_only=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
im = imageable()
canon, conv, axis = RELS[a.rel]
prs = sample_pairs(im["new"], a.npairs, 100 + list(RELS).index(a.rel))  # E31's draw
recs = []
for pi, (x, y) in enumerate(prs):
    for s in range(a.seeds):
        for phr, tmpl in (("canon", canon), ("conv", conv)):
            first_slot = x if a.layout == "A" else y
            recs.append(dict(file=f"{a.rel}/p{pi:03d}_{phr}_s{s}_{a.layout}", rel=a.rel, axis=axis, pair=pi, phr=phr, seed=s, x=x, y=y,
                             layout=a.layout, slot1=first_slot, first=x if phr == "canon" else y, caption=tmpl.format(x=x, y=y),
                             kind="gligen", model="gligen"))
if a.limit:
    recs = recs[:a.limit]
os.makedirs(f"{R31}/index", exist_ok=True)
with open(f"{R31}/index/gligen_{a.rel}_{a.layout}.json", "w", encoding="utf-8") as f:
    json.dump(recs, f)
todo = [r for r in recs if not os.path.exists(f"{R31}/images/gligen/{r['file']}.png")]
print(a.rel, a.layout, len(recs), "records,", len(todo), "to do", flush=True)
for k, r in enumerate(todo):
    b1, b2 = BOX[r["axis"]]
    other = r["y"] if r["slot1"] == r["x"] else r["x"]
    g = torch.Generator("cuda").manual_seed(int(r["seed"]) + 12345)
    lat = torch.randn((1, 4, 64, 64), generator=g, device="cuda", dtype=torch.float16)  # E31's SD 1.5 noise
    img = pipe(prompt=r["caption"], gligen_phrases=[r["slot1"], other], gligen_boxes=[b1, b2], gligen_scheduled_sampling_beta=1.0,
               num_inference_steps=50, guidance_scale=7.5, latents=lat, height=512, width=512).images[0]
    fn = f"{R31}/images/gligen/{r['file']}.png"
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    img.resize((384, 384), Image.LANCZOS).save(fn, compress_level=6)
    if k % 50 == 0:
        print(k, "/", len(todo), flush=True)
