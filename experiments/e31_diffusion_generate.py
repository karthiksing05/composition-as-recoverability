"""D1 generation. Modes: singles (one image of "a x" per noun), pairs (held-out imageable nouns, canonical +
converse caption, same init noise per seed), fit / val (canonical captions of fitting / validation nouns, for D2b).
Writes 384px PNGs under results/e31/images/<model>/..., the final latent (fp16 .npy) under results/e31/latents/, and a JSON index.
Resumable: existing images are skipped."""
import argparse, json, os
import numpy as np, torch
from e31_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="sd15")
ap.add_argument("--mode", required=True)  # singles | pairs | fit | val
ap.add_argument("--rel", default="on")
ap.add_argument("--npairs", type=int, default=150)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default="")
ap.add_argument("--sseed", type=int, default=0)
a = ap.parse_args()

MID = {"sd15": "runwayml/stable-diffusion-v1-5", "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}[a.model]
SZ = 512 if a.model == "sd15" else 1024
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
if a.model == "sd15":
    from diffusers import StableDiffusionPipeline as P
    pipe = P.from_pretrained(MID, torch_dtype=torch.float16, safety_checker=None).to("cuda")
else:
    from diffusers import StableDiffusionXLPipeline as P
    pipe = P.from_pretrained(MID, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

sp = split(0)
recs = []
if a.mode == "singles":
    for i, x in enumerate(sp["all"]):
        recs.append(dict(file=f"singles/{x}_s{a.sseed}", x=x, caption=f"a {x}", seed=a.sseed, kind="single"))
else:
    im = imageable()
    if a.mode == "pairs":
        pool = im["new"]
    else:
        pool = im[{"fit": "fit", "val": "val"}[a.mode]]
    rels = [a.rel]
    for ri, rel in enumerate(rels):
        canon, conv, axis = RELS[rel]
        prs = sample_pairs(pool, a.npairs, 100 + list(RELS).index(rel) + (0 if a.mode == "pairs" else 10 if a.mode == "fit" else 20))
        for pi, (x, y) in enumerate(prs):
            for s in range(a.seeds):
                for phr, tmpl in (("canon", canon), ("conv", conv)):
                    if a.mode != "pairs" and phr == "conv":
                        continue
                    recs.append(dict(file=f"{a.mode}/{rel}/p{pi:03d}_{phr}_s{s}", rel=rel, axis=axis, pair=pi, phr=phr, seed=s,
                                     x=x, y=y, first=x if phr == "canon" else y, caption=tmpl.format(x=x, y=y), kind=a.mode))
if a.limit:
    recs = recs[:a.limit]
tag = f"{a.model}_{a.mode}" + (f"_{a.rel}" if a.mode != "singles" else "") + a.tag
os.makedirs(f"{R31}/index", exist_ok=True)
for r in recs:
    r["model"] = a.model
with open(f"{R31}/index/{tag}.json", "w", encoding="utf-8") as f:
    json.dump(recs, f)

todo = [r for r in recs if not os.path.exists(f"{R31}/images/{a.model}/{r['file']}.png")]
print(tag, len(recs), "records,", len(todo), "to do", flush=True)
for b in range(0, len(todo), a.bs):
    batch = todo[b:b + a.bs]
    lat = []
    for r in batch:
        g = torch.Generator("cuda").manual_seed(int(r["seed"]) + 12345)
        lat.append(torch.randn((1, 4, SZ // 8, SZ // 8), generator=g, device="cuda", dtype=torch.float16))
    lat = torch.cat(lat)
    kw = dict(prompt=[r["caption"] for r in batch], latents=lat, num_inference_steps=30, guidance_scale=7.5, height=SZ, width=SZ, output_type="latent")
    z = pipe(**kw).images
    with torch.no_grad():
        vae = pipe.vae
        zz = z.to(vae.dtype) / vae.config.scaling_factor
        if a.model == "sdxl":
            vae.to(torch.float32); zz = zz.float()
        img = vae.decode(zz).sample
        img = ((img.float() / 2 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        if a.model == "sdxl":
            vae.to(torch.float16)
    for r, im_, z_ in zip(batch, img, z):
        fn = f"{R31}/images/{a.model}/{r['file']}.png"
        lf = f"{R31}/latents/{a.model}/{r['file']}.npy"
        os.makedirs(os.path.dirname(fn), exist_ok=True); os.makedirs(os.path.dirname(lf), exist_ok=True)
        np.save(lf, z_.float().cpu().numpy().astype(np.float16))
        Image.fromarray(im_).resize((384, 384), Image.LANCZOS).save(fn, compress_level=6)
    print(b + len(batch), "/", len(todo), flush=True)
