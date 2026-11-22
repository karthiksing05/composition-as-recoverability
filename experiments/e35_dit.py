"""E35: E30 (fixed-quadrant states, random-weight control, diffusion classifier) and E34 (moved photographs) on
transformer denoisers without a CLIP text encoder: PixArt-Sigma (DiT, cross-attention to T5-XXL) and Lumina-Image-2.0
(single-stream joint attention over caption and image tokens, Gemma-2-2B captions, flow matching).  FLUX.1-schnell was
the planned joint-attention model; its repository is gated for this account, so Lumina-Image-2.0 takes its place.
The protocol is E30's and E34's, imported unchanged: splits, pair draws, part codes, PCA targets, ridge,
kappa, readouts and layer/timestep selection on validation parts.  Only the model interface differs:

  state      image tokens of four transformer blocks at about 1/4, 1/2, 3/4 and the end of the stack, reshaped to their
             (h, w) grid; named mid, up0, up1, up2 so that E30/E34 aggregation code reads them unchanged
             (PixArt: blocks 6, 13, 20, 27 of 28; Lumina: joint layers 6, 13, 19, 25 of 26, image tokens only)
  noise      PixArt: DDPM, x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) eps, predicts eps (first 4 of 8 output channels)
             Lumina: rectified flow, x_t = (1 - s) x0 + s eps with s = t / 1000; the model takes 1 - s and predicts x0 - eps
  caption    empty caption for the states (as E30; Lumina's pipeline adds its system prompt); E17's captions for the classifier
Images at 512 px as E30 (SDXL also ran at 512 there).
usage: e35_dit.py --model pixart|lumina --task trained|random|classifier|moved --seeds 0+1+2 [--smoke]
"""
import argparse, io, json, os
import numpy as np
import torch
from PIL import Image
import e4_pretrained as E
import e8_vision as V
import e17_encoder_behavior as B
import e30_diffusion_photos as E30
import e34_relative_places as E34

REPOS = {"pixart": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "lumina": "Alpha-VLLM/Lumina-Image-2.0"}
NAMES = ("mid", "up0", "up1", "up2")


class DiT:
    def __init__(self, key, res, dev):
        self.key, self.res, self.dev = key, res, dev
        kw = dict(local_files_only=True)
        if key == "pixart":
            from diffusers import PixArtSigmaPipeline
            pipe = PixArtSigmaPipeline.from_pretrained(REPOS[key], torch_dtype=torch.float16, **kw)
            self.dtype = torch.float16
            self.ab = pipe.scheduler.alphas_cumprod.to(dev).float()
            self.blk_idx = [("transformer_blocks", i) for i in (6, 13, 20, 27)]
        else:
            from diffusers import Lumina2Pipeline
            pipe = Lumina2Pipeline.from_pretrained(REPOS[key], torch_dtype=torch.bfloat16, **kw)
            self.dtype = torch.bfloat16
            self.blk_idx = [("layers", i) for i in (6, 13, 19, 25)]
        self.bs = 32
        self.caplen = None  # Lumina: effective caption length per sample of the current forward (image tokens follow it)
        self.pipe = pipe.to(dev)
        self.vae = pipe.vae.to(torch.float32)
        self.sf = self.vae.config.scaling_factor
        self.shift = getattr(self.vae.config, "shift_factor", None) or 0.0
        self.C = self.vae.config.latent_channels
        self.unets = {"trained": pipe.transformer.eval()}
        self.cache = {}

    # ---- interface used by extract / classifier
    def random_unet(self):
        if "random" not in self.unets:
            torch.manual_seed(0)
            t = self.unets["trained"]
            u = type(t).from_config(t.config).eval().to(self.dev, torch.float32)  # fp32: random weights can overflow in half
            self.unets["random"] = u
        return self.unets["random"]

    @torch.no_grad()
    def cond(self, prompts, n):
        key = (tuple(prompts), n)
        if key in self.cache:
            return self.cache[key]
        pe, pm, _, _ = self.pipe.encode_prompt(prompts, do_classifier_free_guidance=False, device=self.dev,
                                               max_sequence_length=300 if self.key == "pixart" else 256)
        c = dict(encoder_hidden_states=pe, encoder_attention_mask=pm)
        if len(prompts) == 1 and n > 1:
            c = {k: v.expand(n, *v.shape[1:]) for k, v in c.items()}
        if len(prompts) == 1:
            self.cache[key] = c
        return c

    @staticmethod
    def slice_cond(c, n):
        return {k: v[:n] for k, v in c.items()}

    @staticmethod
    def cat_cond(cs):
        return {k: torch.cat([c[k] for c in cs]) for k in cs[0]}

    @torch.no_grad()
    def latents(self, ims):
        x = np.stack([np.asarray(im.resize((self.res, self.res), Image.BICUBIC), np.float32) / 127.5 - 1 for im in ims])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.dev)
        z = torch.cat([self.vae.encode(x[i:i + 8]).latent_dist.mean for i in range(0, len(x), 8)])  # chunks of 8: fp32 VAE memory
        return (z - self.shift) * self.sf

    def noisy(self, x0, t, eps):
        if self.key == "pixart":
            a = self.ab[t]
            return a.sqrt() * x0 + (1 - a).sqrt() * eps
        s = t / 1000.0
        return (1 - s) * x0 + s * eps

    def target(self, x0, eps):
        return eps if self.key == "pixart" else x0 - eps

    def forward(self, u, xt, t, c):
        """xt (B, C, h, w) -> prediction (B, C, h, w) in the space of target()."""
        B_, C, h, w = xt.shape
        dt = next(u.parameters()).dtype
        cc = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in c.items()}
        if self.key == "pixart":
            o = u(xt.to(dt), timestep=torch.full((B_,), int(t), device=self.dev), added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                  return_dict=False, **cc)[0]
            return o[:, :C].float()
        self.caplen = cc["encoder_attention_mask"].sum(1).tolist()
        o = u(xt.to(dt), timestep=torch.full((B_,), 1 - t / 1000.0, device=self.dev, dtype=dt), return_dict=False, **cc)[0]
        return o.float()

    def hooks(self, u, cap):
        """Register hooks that store each chosen block's image tokens as a (B, D, gh, gw) map under cap[name]."""
        hs = []
        for name, (attr, i) in zip(NAMES, self.blk_idx):
            def fn(m, inp, o, name=name):
                x = o[0] if isinstance(o, tuple) else o
                if self.key == "lumina":  # joint sequence: caption tokens, then the image tokens, then padding
                    n_img = (self.res // 16) ** 2
                    x = torch.stack([x[b, L:L + n_img] for b, L in enumerate(self.caplen)])
                g = int(round(x.shape[1] ** 0.5))
                cap[name] = x.reshape(x.shape[0], g, g, x.shape[2]).permute(0, 3, 1, 2)
            hs.append(getattr(u, attr)[i].register_forward_hook(fn))
        return hs


# ---------------------------------------------------------------- E30.extract for DiTs (same outputs and keys)
@torch.no_grad()
def extract(D, ims, sources, ts, bs=32):
    out = {"latent": {}}
    for s in sources:
        out[s] = {}
    g = torch.Generator().manual_seed(E30.NOISE_SEED)
    h = D.res // 8
    eps = torch.randn(1, D.C, h, h, generator=g).to(D.dev)
    cap, hooks = {}, []
    for s in sources:
        u = D.unets[s] if s != "random" else D.random_unet()
        hooks += D.hooks(u, cap)
    ec = {s: D.cond([""], bs) for s in sources}
    for s0 in range(0, len(ims), bs):
        x0 = D.latents(ims[s0:s0 + bs])
        B_ = len(x0)
        for nm, v in (("Mlat", x0.mean((2, 3))), ("Glat", E30.quad(x0)), ("Qlat", E30.cells(x0)),
                       ("Rlat", E30.cells(x0, 8)), ("Slat", E30.cells(x0, 16))):
            out["latent"].setdefault(nm, []).append(v.float().cpu().numpy())
        for s in sources:
            u = D.unets[s]
            for t in ts:
                cap.clear()
                D.forward(u, D.noisy(x0, t, eps), t, D.slice_cond(ec[s], B_))
                for name, f in cap.items():
                    f = f.float()
                    out[s].setdefault(f"M{t}_{name}", []).append(f.mean((2, 3)).cpu().numpy())
                    out[s].setdefault(f"G{t}_{name}", []).append(E30.quad(f).cpu().numpy())
    for hk in hooks:
        hk.remove()
    return {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in out.items()}


# ---------------------------------------------------------------- E34.sd_extract for DiTs
@torch.no_grad()
def moved_extract(D, ims, boxes, shuf, ts, bs=24):
    out, cap = {}, {}
    u = D.unets["trained"]
    hooks = D.hooks(u, cap)
    g = torch.Generator().manual_seed(E30.NOISE_SEED)
    h = D.res // 8
    eps = torch.randn(1, D.C, h, h, generator=g).to(D.dev)
    c = D.cond([""], bs)
    for s0 in range(0, len(ims), bs):
        x0 = D.latents(ims[s0:s0 + bs]); B_ = len(x0)
        for t in ts:
            cap.clear()
            D.forward(u, D.noisy(x0, t, eps), t, D.slice_cond(c, B_))
            for name, f in cap.items():
                f = f.float()
                d = E34.readouts(f, boxes[s0:s0 + bs], shuf[s0:s0 + bs], (2, 4))
                d["M"] = f.mean((2, 3))
                for k, v in d.items():
                    out.setdefault((k, f"{t}_{name}"), []).append(v.half().cpu())
    for hk in hooks:
        hk.remove()
    return {k: torch.cat(v) for k, v in out.items()}


# ---------------------------------------------------------------- E30.run_classifier for DiTs (same captions, draws and scores)
@torch.no_grad()
def run_classifier(D, seed, images, labels, outdir, name, n_pairs, n_t, bs=8):
    names = B.class_names()
    rng = np.random.default_rng(seed)
    cls = rng.permutation(200)
    fresh_cls = [int(c) for c in cls[150:200] if names[c]]
    rng2 = np.random.default_rng(10_000 + seed)
    pool = {c: list(np.where(labels == c)[0]) for c in fresh_cls}
    u = D.unets["trained"]
    tgrid = np.linspace(50, 950, n_t).round().astype(int)
    h = D.res // 8
    for rel in B.SLOTS:
        ca = rng2.choice(fresh_cls, B.N_PAIRS); cb = np.array([rng2.choice([c for c in fresh_cls if c != a]) for a in ca])
        cz = np.array([rng2.choice([c for c in fresh_cls if c not in (a, b)]) for a, b in zip(ca, cb)])
        ia = [int(rng2.choice(pool[c])) for c in ca]; ib = [int(rng2.choice(pool[c])) for c in cb]
        n = n_pairs
        fn = os.path.join(outdir, f"{name}_classifier_s{seed}_{rel}.json")
        if os.path.exists(fn):
            continue
        pil = lambda i: Image.open(io.BytesIO(images[i])).convert("RGB")
        x = [names[c] for c in ca[:n]]; y = [names[c] for c in cb[:n]]; z = [names[c] for c in cz[:n]]
        order = np.arange(n) % 2
        good = [B.caption(rel, x[i], y[i], 0) if o == 0 else B.caption(rel, y[i], x[i], 1) for i, o in enumerate(order)]
        swap = [B.caption(rel, y[i], x[i], 0) if o == 0 else B.caption(rel, x[i], y[i], 1) for i, o in enumerate(order)]
        wrong = [B.caption(rel, x[i], z[i], 0) if o == 0 else B.caption(rel, z[i], x[i], 1) for i, o in enumerate(order)]
        err = np.zeros((n, 3, len(tgrid)), np.float32)
        g = torch.Generator(device="cpu").manual_seed(777 + seed)
        for s0 in range(0, n, bs):
            sl = slice(s0, min(n, s0 + bs)); Bn = sl.stop - sl.start
            x0 = D.latents([V.compose(rel, pil(a), pil(b)) for a, b in zip(ia[sl], ib[sl])])
            c = D.cat_cond([D.cond(caps[sl], Bn) for caps in (good, swap, wrong)])
            for ti, t in enumerate(tgrid):
                eps = torch.randn(Bn, D.C, h, h, generator=g).to(D.dev)  # identical for the three captions
                xt = D.noisy(x0, int(t), eps)
                o = D.forward(u, xt.repeat(3, 1, 1, 1), int(t), c)
                tg = D.target(x0, eps).repeat(3, 1, 1, 1)
                e = ((o - tg) ** 2).mean((1, 2, 3)).view(3, Bn).cpu().numpy()
                err[sl, :, ti] = e.T
        m = err.mean(2)
        res = dict(model=name, seed=seed, rel=rel, n_pairs=n, n_t=len(tgrid), res=D.res, n_classes=len(fresh_cls),
                   roles=float((m[:, 0] < m[:, 1]).mean()), roles_first=float((m[order == 0, 0] < m[order == 0, 1]).mean()),
                   roles_second=float((m[order == 1, 0] < m[order == 1, 1]).mean()), identity=float((m[:, 0] < m[:, 2]).mean()),
                   roles_by_t=[float((err[:, 0, k] < err[:, 1, k]).mean()) for k in range(len(tgrid))],
                   identity_by_t=[float((err[:, 0, k] < err[:, 2, k]).mean()) for k in range(len(tgrid))],
                   t_grid=[int(t) for t in tgrid], mean_err=m.mean(0).tolist())
        print(res["model"], rel, {k: round(v, 3) for k, v in res.items() if k in ("roles", "roles_first", "roles_second", "identity")}, flush=True)
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REPOS))
    ap.add_argument("--task", required=True, choices=["trained", "random", "classifier", "moved"])
    ap.add_argument("--seeds", default="0+1+2")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--ts", default="100+300+500")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--n-t", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    E30.SMOKE = E34.SMOKE = a.smoke
    E30.TS = tuple(map(int, a.ts.split("+")))
    E30.extract = extract          # E30.run_states and E34.run_system look these up at call time
    E34.sd_extract = moved_extract
    E.ALPHAS = np.logspace(-3, 9, 25)
    outdir = a.outdir or f"/coc/flash10/ksingara3/compositionality-theory/experiments/results/{'e34' if a.task == 'moved' else 'e30'}"
    os.makedirs(outdir, exist_ok=True)
    images, labels = V.load_images()
    D = DiT(a.model, a.res, "cuda")
    name = a.model + (f"-{a.res}" if a.res != 512 else "")
    for seed in map(int, a.seeds.split("+")):
        if a.task == "classifier":
            run_classifier(D, seed, images, labels, outdir, name, 24 if a.smoke else a.n_pairs, 4 if a.smoke else a.n_t)
        elif a.task == "moved":
            E34.run_system(name, seed, images, labels, outdir, D)
        else:
            E30.run_states(D, a.task, seed, images, labels, outdir, name)
