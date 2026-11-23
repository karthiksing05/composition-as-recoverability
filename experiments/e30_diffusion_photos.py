"""E30: the two-photographs-on-a-canvas test on Stable Diffusion (plan D3 and D6).

Tasks (one per invocation; each writes results/e30/<model>_<task>_s<seed>[_<rel>].json and resumes):
  trained   state of the trained U-Net (mid block and every up block; t in {100,300,500}; empty caption;
            fixed noise) read by E8's readouts, plus the VAE latent itself (D6 origin i).
            Pooling: M = mean over the map, G = four quadrants concatenated (E8 grid); latent also Q = 4 x 4 cells.
  random    same architecture and VAE, randomly initialised U-Net (D6 origin ii).  Read against its own
            part codes ("own", E8's rand- convention) and against the trained U-Net's codes ("trained",
            E4's --codes-from convention).
  classifier  the diffusion classifier (Li et al. 2023) on E17's captions: roles (places swapped) and
            identity (second object replaced), error averaged over timesteps with identical noise.
Splits, pair draws, part codes (state of the photograph alone, PCA-16, whitened), references and readouts are
those of e8_grid (imported: e4_pretrained.evaluate).  Canvases are E8's 224 canvases resized to RES.
"""
import argparse, io, json, os
import numpy as np
import torch
from PIL import Image
import e4_pretrained as E
import e8_vision as V
import e17_encoder_behavior as B

REPOS = {"sd15": "runwayml/stable-diffusion-v1-5", "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}
TS = (100, 300, 500)
NOISE_SEED = 12345
SMOKE = False


# ---------------------------------------------------------------- models
class Diff:
    def __init__(self, key, res, dev):
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
        self.key, self.res, self.dev = key, res, dev
        kw = dict(torch_dtype=torch.float16, local_files_only=True)
        if key == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(REPOS[key], **kw)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(REPOS[key], safety_checker=None, **kw)
        self.pipe = pipe.to(dev)
        self.vae = pipe.vae.to(torch.float32)
        self.sf = self.vae.config.scaling_factor
        self.unets = {"trained": pipe.unet.eval()}
        self.ab = pipe.scheduler.alphas_cumprod.to(dev).float()
        self.cls = UNet2DConditionModel
        self.cond_cache = {}

    def random_unet(self):
        if "random" not in self.unets:
            torch.manual_seed(0)
            u = self.cls.from_config(self.unets["trained"].config).eval().to(self.dev)
            self.unets["random"] = u.half()
            # probe: fp16 with random weights can overflow; fall back to fp32 if so
            lat = torch.randn(1, 4, self.res // 8, self.res // 8, device=self.dev, generator=None).half()
            c = self.cond(["" ], 1)
            with torch.no_grad():
                o = u.half()(lat, torch.tensor([300], device=self.dev), encoder_hidden_states=c[0], added_cond_kwargs=c[1]).sample
            if not torch.isfinite(o).all():
                print("random U-Net not finite in fp16; using fp32", flush=True)
                self.unets["random"] = u.float()
        return self.unets["random"]

    @torch.no_grad()
    def cond(self, prompts, n):
        """(encoder_hidden_states, added_cond_kwargs) for a list of prompts (len 1 expanded to n)."""
        if self.key == "sdxl":
            pe, _, pooled, _ = self.pipe.encode_prompt(prompt=prompts, prompt_2=None, device=self.dev, num_images_per_prompt=1,
                                                       do_classifier_free_guidance=False)
            tid = torch.tensor([[self.res, self.res, 0, 0, self.res, self.res]], device=self.dev, dtype=pe.dtype).expand(len(prompts), -1)
            added = {"text_embeds": pooled, "time_ids": tid}
        else:
            pe = self.pipe.encode_prompt(prompt=prompts, device=self.dev, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
            added = None
        if len(prompts) == 1 and n > 1:
            pe = pe.expand(n, -1, -1)
            if added:
                added = {k: v.expand(n, -1) for k, v in added.items()}
        return pe, added

    @torch.no_grad()
    def latents(self, ims):
        x = np.stack([np.asarray(im.resize((self.res, self.res), Image.BICUBIC), np.float32) / 127.5 - 1 for im in ims])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.dev)
        return self.vae.encode(x).latent_dist.mean * self.sf

    def noisy(self, x0, t, eps):
        a = self.ab[t]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps


def quad(x):  # (B, C, H, W) -> (B, 4C): mean in each quadrant, as e8_grid
    H, W = x.shape[-2:]
    out = []
    for r in np.array_split(np.arange(H), 2):
        for c in np.array_split(np.arange(W), 2):
            out.append(x[:, :, r[0]:r[-1] + 1, c[0]:c[-1] + 1].mean((2, 3)))
    return torch.cat(out, 1)


def cells(x, g=4):
    H, W = x.shape[-2:]
    out = []
    for r in np.array_split(np.arange(H), g):
        for c in np.array_split(np.arange(W), g):
            out.append(x[:, :, r[0]:r[-1] + 1, c[0]:c[-1] + 1].mean((2, 3)))
    return torch.cat(out, 1)


@torch.no_grad()
def extract(D, ims, sources, ts, bs=None):
    """dict: 'lat' -> {Mlat,Glat,Qlat}; source -> {M<t>_<blk>, G<t>_<blk>}.  float32 numpy, one row per image."""
    bs = bs or (8 if D.res > 512 else 32)  # 1,024 px runs out of A40 memory at 32
    out = {"latent": {}}
    for s in sources:
        out[s] = {}
    g = torch.Generator().manual_seed(NOISE_SEED)
    h = D.res // 8
    eps = torch.randn(1, 4, h, h, generator=g).to(D.dev)
    cap = {}
    for s in sources:
        u = D.unets[s] if s != "random" else D.random_unet()
        hooks = []
        blocks = [("mid", u.mid_block)] + [(f"up{i}", b) for i, b in enumerate(u.up_blocks)]
        for name, b in blocks:
            hooks.append(b.register_forward_hook(lambda m, i, o, name=name: cap.__setitem__(name, o[0] if isinstance(o, tuple) else o)))
        D.unets.setdefault("_hooks", []).append(hooks)
    ec = {s: D.cond([""], bs) for s in sources}
    for s0 in range(0, len(ims), bs):
        x0 = D.latents(ims[s0:s0 + bs])
        B_ = len(x0)
        for nm, v in (("Mlat", x0.mean((2, 3))), ("Glat", quad(x0)), ("Qlat", cells(x0)),
                       ("Rlat", cells(x0, 8)), ("Slat", cells(x0, 16))):  # 4x4, 8x8, 16x16 cells
            out["latent"].setdefault(nm, []).append(v.float().cpu().numpy())
        for s in sources:
            u = D.unets[s]
            pe, added = ec[s]
            pe = pe[:B_]; added = {k: v[:B_] for k, v in added.items()} if added else None
            for t in ts:
                xt = D.noisy(x0, t, eps).to(u.dtype)
                u(xt, torch.full((B_,), t, device=D.dev), encoder_hidden_states=pe.to(u.dtype),
                  added_cond_kwargs={k: v.to(u.dtype) for k, v in added.items()} if added else None)
                for name, f in cap.items():
                    f = f.float()
                    out[s].setdefault(f"M{t}_{name}", []).append(f.mean((2, 3)).cpu().numpy())
                    out[s].setdefault(f"G{t}_{name}", []).append(quad(f).cpu().numpy())
    for hs in D.unets.pop("_hooks", []):
        for hk in hs:
            hk.remove()
    return {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in out.items()}


# ---------------------------------------------------------------- readout
def make_T(P, pca):
    pmu = P[pca].mean(0)
    _, sv, Vt = np.linalg.svd(P[pca] - pmu, full_matrices=False)
    k = min(E.K_PCA, len(sv))
    return (P - pmu) @ Vt[:k].T / (sv[:k] / np.sqrt(len(pca)))


def draw_sets(rng, splits):
    """The pair draws of e8_grid.run for the three relations, in the same rng order."""
    sc = 0.1 if SMOKE else 1.0
    out = {}
    for rel in V.RELATIONS:
        def sample(ns, k):
            prs = [(x, y) for x in ns for y in ns if x != y]
            return [prs[i] for i in rng.choice(len(prs), min(int(k), len(prs)), replace=False)]
        allfit = sample(splits["fit"], 3800 * sc)
        n_tr = int(len(allfit) * 3000 / 3800) if SMOKE else 3000
        fit_pairs, pairs_pairs = allfit[:n_tr], allfit[n_tr:]
        val_pairs, fresh_pairs = sample(splits["val"], 600 * sc), sample(splits["fresh"], 1000 * sc)
        g12 = splits["fit"][:12]
        grid = [(x, y) for x in g12 for y in g12]
        rev_pairs = [(y, x) for x, y in fresh_pairs]
        sets, allp, s0 = {}, [], 0
        for k, prs in (("fit", fit_pairs), ("pairs", pairs_pairs), ("val", val_pairs), ("fresh", fresh_pairs), ("grid", grid), ("rev", rev_pairs)):
            sets[k] = slice(s0, s0 + len(prs)); s0 += len(prs); allp += prs
            sets[f"{k}_pairs"] = prs
        out[rel] = (sets, allp)
    return out


def get_split(seed, labels):
    rng = np.random.default_rng(seed)
    cls = rng.permutation(200)
    splits, s0 = {}, 0
    for k, n, per in zip(("fit", "val", "fresh"), V.N_CLS, V.PER_CLS):
        ids = []
        for c in cls[s0:s0 + n]:
            ids += list(rng.choice(np.where(labels == c)[0], per, replace=False))
        splits[k] = [int(i) for i in ids]; s0 += n
    return rng, splits


def rows_for(H, codes, own_codes_for, sets, allp, idx, pca, cands, meta, with_ref):
    """H: layer -> states of the composites.  codes: part-code dict {cname: {Mlayer: P}} to read against.
    own_codes_for(layer) -> code layer key."""
    res = []
    for layer, Hall in H.items():
        ck = own_codes_for(layer)
        for cname in V.CODES:
            P = codes[cname][ck]
            T = make_T(P, pca)
            A = P[[idx[x] for x, _ in allp]]; Bm = P[[idx[y] for _, y in allp]]
            systems = {"model": Hall}
            if with_ref and layer.startswith("M"):
                systems.update({"ref_concat": np.c_[A, Bm], "ref_sum": A + Bm})
            for sname, Hs in systems.items():
                row = dict(meta, layer=layer, code=cname, code_layer=ck, system=sname, dim=int(Hs.shape[1]))
                row.update(E.evaluate(Hs, T, idx, sets, cands))
                res.append(row)
            r = [x for x in res if x["system"] == "model"][-1]
            print(meta["model"], meta["rel"], layer, cname, "code", ck, round((r["rec_fresh_role1"] + r["rec_fresh_role2"]) / 2, 3), flush=True)
    return res


def run_states(D, task, seed, images, labels, outdir, name):
    ts = (300,) if SMOKE else TS
    rng, splits = get_split(seed, labels)
    used = splits["fit"] + splits["val"] + splits["fresh"]
    pil = {i: Image.open(io.BytesIO(images[i])).convert("RGB") for i in used}
    idx = {x: j for j, x in enumerate(used)}
    pca = list(range(len(splits["fit"])))
    draws = draw_sets(rng, splits)
    cands = {"val": splits["val"], "fresh": splits["fresh"]}
    alone = {"center": [V.canvas([(pil[i], 112, (56, 56))]) for i in used],
             "full": [pil[i].resize((224, 224), Image.BICUBIC) for i in used]}
    srcs = ["trained"] if task == "trained" else ["trained", "random"]
    codes = {s: {} for s in srcs + ["latent"]}
    for cn, ims in alone.items():
        e = extract(D, ims, srcs, ts)
        for s in srcs + ["latent"]:
            codes[s][cn] = {k: v for k, v in e[s].items() if k.startswith("M")}
    for rel in V.RELATIONS:
        fn = os.path.join(outdir, f"{name}_{task}_s{seed}_{rel}.json")
        if os.path.exists(fn):
            continue
        sets, allp = draws[rel]
        e = extract(D, [V.compose(rel, pil[x], pil[y]) for x, y in allp], srcs[-1:] if task == "random" else ["trained"], ts) \
            if task == "trained" else extract(D, [V.compose(rel, pil[x], pil[y]) for x, y in allp], ["random"], ts)
        res = []
        meta = lambda src, cs: dict(model=f"{name}|{src}|{cs}", seed=seed, rel=rel, source=src, codesrc=cs)
        common = (sets, allp, idx, pca, cands)
        if task == "trained":
            res += rows_for(e["trained"], codes["trained"], lambda l: "M" + l[1:], *common, meta("trained", "own"), True)
            # D6 origin (i): the VAE latent, read against the trained U-Net's codes at every layer and against its own mean code
            for ck in codes["trained"]["center"]:
                cd = {c: {ck: codes["trained"][c][ck]} for c in V.CODES}
                res += rows_for(e["latent"], cd, lambda l, ck=ck: ck, *common, dict(meta("latent", "trained"), code_layer_tag=ck), False)
            res += rows_for(e["latent"], codes["latent"], lambda l: "Mlat", *common, meta("latent", "own"), False)
        else:
            res += rows_for(e["random"], codes["random"], lambda l: "M" + l[1:], *common, meta("random", "own"), True)
            res += rows_for(e["random"], codes["trained"], lambda l: "M" + l[1:], *common, meta("random", "trained"), False)
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


# ---------------------------------------------------------------- diffusion classifier
@torch.no_grad()
def run_classifier(D, seed, images, labels, outdir, name, n_pairs, n_t, bs=8):
    names = B.class_names()
    rng = np.random.default_rng(seed)  # E8 split, as e17
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
            pes, adds = [], []
            for caps in (good, swap, wrong):
                pe, ad = D.cond(caps[sl], Bn)
                pes.append(pe); adds.append(ad)
            pe = torch.cat(pes)
            ad = {k: torch.cat([a[k] for a in adds]) for k in adds[0]} if adds[0] else None
            for ti, t in enumerate(tgrid):
                eps = torch.randn(Bn, 4, h, h, generator=g).to(D.dev)  # identical for the three captions
                xt = D.noisy(x0, int(t), eps).half()
                o = u(xt.repeat(3, 1, 1, 1), torch.full((3 * Bn,), int(t), device=D.dev), encoder_hidden_states=pe,
                      added_cond_kwargs=ad).sample.float()
                e = ((o - eps.repeat(3, 1, 1, 1)) ** 2).mean((1, 2, 3)).view(3, Bn).cpu().numpy()
                err[sl, :, ti] = e.T
        m = err.mean(2)
        thirds = np.array_split(np.arange(len(tgrid)), 3)
        acc = lambda a, b, sel=slice(None), idxs=slice(None): float((err[idxs][:, 0, sel] < err[idxs][:, b, sel]).mean())
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
    ap.add_argument("--model", default="sd15", choices=list(REPOS))
    ap.add_argument("--task", required=True, choices=["trained", "random", "classifier"])
    ap.add_argument("--seeds", default="0+1+2")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--outdir", default="results/e30")
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--n-t", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    SMOKE = a.smoke
    E.ALPHAS = np.logspace(-3, 9, 25)
    dev = "cuda"
    os.makedirs(a.outdir, exist_ok=True)
    images, labels = V.load_images()
    D = Diff(a.model, a.res, dev)
    name = a.model + (f"-{a.res}" if a.res != 512 else "")
    for seed in map(int, a.seeds.split("+")):
        if a.task == "classifier":
            run_classifier(D, seed, images, labels, a.outdir, name, 24 if a.smoke else a.n_pairs, 4 if a.smoke else a.n_t)
        else:
            run_states(D, a.task, seed, images, labels, a.outdir, name)
