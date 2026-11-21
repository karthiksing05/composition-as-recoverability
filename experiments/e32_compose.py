"""D4 score composition (Liu et al. 2022, AND): eps = eps(x|"") + w*sum_k (eps(x|c_k) - eps(x|"")).
Modes: and  (c = "a x", "a y");  pos (c = "a x on the left", "a y on the right").
Also --swapcheck: same seed, concepts in swapped order, compare the final latents numerically.
Held-out imageable pairs = the same 150 pairs as the D1 'left' relation. Same init noise as D1 (seed+12345)."""
import argparse, json, os
import numpy as np, torch
from PIL import Image
from e31_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="and")
ap.add_argument("--w", type=float, default=7.5)
ap.add_argument("--npairs", type=int, default=150)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--swapcheck", action="store_true")
ap.add_argument("--dtype", default="fp16")
ap.add_argument("--rel", default="left")
a = ap.parse_args()
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
dt = torch.float16 if a.dtype == "fp16" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dt, safety_checker=None).to("cuda")
sch = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
unet, vae = pipe.unet, pipe.vae


@torch.no_grad()
def enc(prompts):
    ids = pipe.tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
    return pipe.text_encoder(ids)[0]


@torch.no_grad()
def sample(concepts, seeds, w):
    """concepts: list (per item) of list of prompts; returns final scaled latents (B,4,64,64)."""
    B, K = len(concepts), len(concepts[0])
    lat = torch.cat([torch.randn((1, 4, 64, 64), generator=torch.Generator("cuda").manual_seed(int(s) + 12345), device="cuda", dtype=torch.float16) for s in seeds]).to(dt)
    e0 = enc([""])
    ec = enc([c for cs in concepts for c in cs]).view(B, K, 77, -1)
    cond = torch.cat([e0.expand(B, -1, -1)[:, None], ec], 1).reshape(B * (K + 1), 77, -1)  # per item: [uncond, c1..cK]
    sch.set_timesteps(30, device="cuda")
    for t in sch.timesteps:
        inp = lat[:, None].expand(-1, K + 1, -1, -1, -1).reshape(B * (K + 1), 4, 64, 64)
        eps = unet(sch.scale_model_input(inp, t), t, encoder_hidden_states=cond).sample.view(B, K + 1, 4, 64, 64)
        e = eps[:, 0] + w * (eps[:, 1:] - eps[:, :1]).sum(1)
        lat = sch.step(e, t, lat).prev_sample
    return lat


def prompts(mode, x, y):
    if mode == "and": return [f"a {x}", f"a {y}"]
    if mode == "posv": return [f"a {x} at the top", f"a {y} at the bottom"]
    return [f"a {x} on the left", f"a {y} on the right"]


@torch.no_grad()
def decode(lat):
    img = vae.decode((lat / vae.config.scaling_factor).to(vae.dtype)).sample
    return ((img.float() / 2 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


im = imageable()
prs = sample_pairs(im["new"], 150, 100 + list(RELS).index(a.rel))[:a.npairs]
if a.swapcheck:
    res = []
    for mode in ("and", "pos"):
        for pi, (x, y) in enumerate(prs[:12]):
            s = pi % 4
            l1 = sample([prompts(mode, x, y)], [s], a.w)
            l2 = sample([prompts(mode, x, y)[::-1]], [s], a.w)  # same two concepts, order swapped
            i1, i2 = decode(l1)[0], decode(l2)[0]
            res.append(dict(mode=mode, x=x, y=y, seed=s, max_abs_latent_diff=float((l1 - l2).abs().max()), latent_absmax=float(l1.abs().max()),
                            max_abs_pixel_diff=int(np.abs(i1.astype(int) - i2.astype(int)).max()), identical_pixels=bool((i1 == i2).all())))
            print(res[-1], flush=True)
    json.dump(res, open(f"{R32}/swapcheck_{a.dtype}.json", "w", encoding="utf-8"), indent=1)
    raise SystemExit
tag = f"{a.mode}_w{a.w:g}" + ("" if a.rel == "left" else f"_{a.rel}") + ("" if a.npairs == 150 else f"_n{a.npairs}") + ("" if a.seeds == 4 else f"_s{a.seeds}")
recs = [dict(file=f"{tag}/p{pi:03d}_canon_s{s}", rel=a.rel, axis=RELS[a.rel][2], pair=pi, phr="canon", seed=s, x=x, y=y, first=x, caption=" + ".join(prompts(a.mode, x, y)),
             kind=a.mode, model="sd15") for pi, (x, y) in enumerate(prs) for s in range(a.seeds)]
os.makedirs(f"{R32}/index", exist_ok=True)
json.dump(recs, open(f"{R32}/index/{tag}.json", "w", encoding="utf-8"))
todo = [r for r in recs if not os.path.exists(f"{R32}/images/{r['file']}.png")]
print(tag, len(recs), len(todo), flush=True)
for b in range(0, len(todo), a.bs):
    bt = todo[b:b + a.bs]
    lat = sample([prompts(a.mode, r["x"], r["y"]) for r in bt], [r["seed"] for r in bt], a.w)
    for r, ig in zip(bt, decode(lat)):
        fn = f"{R32}/images/{r['file']}.png"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        Image.fromarray(ig).resize((384, 384), Image.LANCZOS).save(fn, compress_level=6)
    print(b + len(bt), "/", len(todo), flush=True)
