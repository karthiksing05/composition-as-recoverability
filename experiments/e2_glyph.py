"""E2: do trained networks represent composites recoverably?

World.  A part is a latent z in R^q (q = 8) rendered as an 8x8 glyph by a fixed
random nonlinear generator G.  A composite of parts (a, b) in roles (1, 2) is
rendered on an 8x16 canvas under one of two relations:
  BESIDE(a, b) = [G(a) | G(b)]                      (disjoint regions)
  OVER(a, b)   = G(a) + fliplr(G(b)) in the middle  (same region, roles by transform)
Both renderings are injective for continuous parts, so every composite is
recoverable in principle; OVER requires unmixing.

Task.  An encoder sees only the composite image and writes a bottleneck h in R^m.
A decoder receives h and a flag for the output relation and must render the same
parts, in the same roles, under the requested relation.  h is therefore the
system's representation of the composite, and the task requires both parts.

Splits.  A quarter of the vocabulary ("H") never occupies role 2 in training.
  train    : vocabulary pairs, role 2 not in H, minus 20% withheld at random
  heldout  : the withheld pairs (difficulty-matched baseline)
  strong   : role 2 in H (a part in a role it never occupied; Hadley's strong systematicity)
  fresh    : both parts drawn fresh from the part distribution (never seen by anything)
Measures.  Behaviour: output R^2 and role errors.  Representation: linear and MLP
readouts of each role's latent from h (fit on train composites), additivity of h
over a grid of parts (two-way ANOVA interaction fraction), and edit leakage
(change in role-2 output region when only the role-1 part changes).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

Q, GH, GW = 8, 8, 8
CANVAS = (8, 16)
SECOND = {"rel": "over"}  # second relation: "over" (additive) or "occ" (occlusion, non-additive)


def pool2(g):  # 8x8 -> 4x4 average pooling
    return g.reshape(*g.shape[:-2], 4, 2, 4, 2).mean((-3, -1))


class World:
    def __init__(self, seed, q=Q):
        g = np.random.default_rng(1000 + seed)
        self.q = q
        self.W1 = g.normal(size=(q, 48)) / np.sqrt(q)
        self.b1 = g.normal(size=48) * 0.3
        self.W2 = g.normal(size=(48, GH * GW)) / np.sqrt(48)
        # spatially smooth the output weights so glyphs look like strokes, not noise
        k = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2); k /= k.sum()
        W2 = self.W2.reshape(48, GH, GW)
        W2 = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, W2)
        W2 = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 2, W2)
        self.W2 = W2.reshape(48, -1)
        self.W2 /= self.W2.std() * np.sqrt(48)

    def glyph(self, z):
        h = np.tanh(z @ self.W1 + self.b1)
        return 1 / (1 + np.exp(-3 * (h @ self.W2))).reshape(*z.shape[:-1], GH, GW)

    def render(self, za, zb, rel):
        ga, gb = self.glyph(za), self.glyph(zb)
        out = np.zeros((*za.shape[:-1], *CANVAS))
        if rel == 0:  # BESIDE
            out[..., :, :8], out[..., :, 8:] = ga, gb
        elif SECOND["rel"] == "over":  # OVER
            out[..., :, 4:12] = ga + gb[..., ::-1]
        else:  # OCC: part b, at half scale, occludes the host's brightest quadrant
            quad = ga.reshape(*ga.shape[:-2], 2, 4, 2, 4).mean((-3, -1)).reshape(*ga.shape[:-2], 4).argmax(-1)
            sub = pool2(gb)
            host = ga.copy()
            for k in range(4):
                r0, c0 = 4 * (k // 2), 4 * (k % 2)
                sel = quad == k
                host[sel, r0:r0 + 4, c0:c0 + 4] = sub[sel]
            out[..., :, 4:12] = host
        return out


class MLPEnc(nn.Module):
    def __init__(self, m, w=256):
        super().__init__()
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(128, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, m))

    def forward(self, x):
        return self.f(x)


class CNNEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 4 * 8, m))

    def forward(self, x):
        return self.f(x[:, None])


class TfEnc(nn.Module):
    def __init__(self, m, dm=64):
        super().__init__()
        self.inp = nn.Linear(8, dm)
        self.pos = nn.Parameter(torch.randn(17, dm) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        layer = nn.TransformerEncoderLayer(dm, 4, 128, dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(dm, m)

    def forward(self, x):
        tok = self.inp(x.transpose(1, 2))  # 16 column tokens of 8 pixels
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1) + self.pos
        return self.out(self.tf(tok)[:, 0])


class MLPDec(nn.Module):
    def __init__(self, m, w=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(m + 2, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 128))

    def forward(self, h, rel):
        return self.f(torch.cat([h, F.one_hot(rel, 2).float()], 1)).view(-1, *CANVAS)


class SlotModel(nn.Module):
    """Role-filler reference: two role slots, one shared glyph decoder, and the
    relation applied by a fixed renderer, so the decoding procedure is shared
    across fillers and roles."""

    def __init__(self, m, w=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(128, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, m))
        self.gdec = nn.Sequential(nn.Linear(m // 2, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 64))
        self.m = m

    def encode(self, x):
        return self.enc(x)

    def decode(self, h, rel):
        s1, s2 = h[:, :self.m // 2], h[:, self.m // 2:]
        ga, gb = self.gdec(s1).view(-1, 8, 8), self.gdec(s2).view(-1, 8, 8)
        out = torch.zeros(len(h), *CANVAS)
        beside = torch.cat([ga, gb], 2)
        over = torch.zeros(len(h), *CANVAS)
        if SECOND["rel"] == "over":
            over[:, :, 4:12] = ga + gb.flip(2)
        else:
            quad = ga.reshape(-1, 2, 4, 2, 4).mean((2, 4)).reshape(-1, 4).argmax(-1)
            sub = F.avg_pool2d(gb[:, None], 2)[:, 0]
            host = ga.clone()
            for k in range(4):
                r0, c0 = 4 * (k // 2), 4 * (k % 2)
                sel = (quad == k).view(-1, 1, 1).float()
                host[:, r0:r0 + 4, c0:c0 + 4] = sel * sub + (1 - sel) * host[:, r0:r0 + 4, c0:c0 + 4]
            over[:, :, 4:12] = host
        r = rel.view(-1, 1, 1).float()
        return (1 - r) * beside + r * over


class Generic(nn.Module):
    def __init__(self, arch, m):
        super().__init__()
        self.enc = {"mlp": MLPEnc, "cnn": CNNEnc, "tf": TfEnc}[arch](m)
        self.dec = MLPDec(m)

    def encode(self, x):
        return self.enc(x)

    def decode(self, h, rel):
        return self.dec(h, rel)


def make_model(arch, m):
    return SlotModel(m) if arch == "slot" else Generic(arch, m)


def splits(V, rng):
    H = set(range(V - max(1, V // 4), V))
    pairs = [(i, j) for i in range(V) for j in range(V) if i != j]
    base = [p for p in pairs if p[1] not in H]
    strong = np.array([p for p in pairs if p[1] in H])
    base = np.array(base)
    rng.shuffle(base)
    n_ho = len(base) // 5
    return base[n_ho:], base[:n_ho], strong


def r2(pred, true):
    pred, true = pred.reshape(len(pred), -1), true.reshape(len(true), -1)
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean(0)) ** 2).sum())


def anova_interaction(Hgrid):
    # Hgrid: n x n x m composite states; additive model mu + alpha_i + beta_j
    mu = Hgrid.mean((0, 1), keepdims=True)
    a = Hgrid.mean(1, keepdims=True) - mu
    b = Hgrid.mean(0, keepdims=True) - mu
    resid = Hgrid - mu - a - b
    return float((resid ** 2).sum() / ((Hgrid - mu) ** 2).sum())


def run(arch, m, V, seed, steps, coverage=1.0, q=Q):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    world = World(seed, q)
    vocab = rng.normal(size=(V, q))
    fresh = rng.normal(size=(64, q))
    train, heldout, strong = splits(V, rng)
    if coverage < 1.0:
        train = train[: max(2, int(round(coverage * len(train))))]
    model = make_model(arch, m)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    # precompute all training composites
    cache = {}
    for rin in (0, 1):
        cache[rin] = torch.tensor(world.render(vocab[train[:, 0]], vocab[train[:, 1]], rin), dtype=torch.float32)
    t0 = time.time()
    bs = 256
    for step in range(steps):
        idx = torch.randint(0, len(train), (bs,))
        rin = torch.randint(0, 2, (bs,))
        rout = torch.randint(0, 2, (bs,))
        x = torch.where(rin.view(-1, 1, 1) == 0, cache[0][idx], cache[1][idx])
        y = torch.where(rout.view(-1, 1, 1) == 0, cache[0][idx], cache[1][idx])
        loss = F.mse_loss(model.decode(model.encode(x), rout), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    train_secs = time.time() - t0
    model.eval()

    def enc(za, zb, rin):
        with torch.no_grad():
            return model.encode(torch.tensor(world.render(za, zb, rin), dtype=torch.float32)).numpy()

    def dec(h, rout):
        with torch.no_grad():
            return model.decode(torch.tensor(h, dtype=torch.float32), torch.full((len(h),), rout, dtype=torch.long)).numpy()

    fr_pairs = np.stack([np.arange(0, 64, 2), np.arange(1, 64, 2)], 1)
    sets = {"train": (vocab, train[:512]), "heldout": (vocab, heldout), "strong": (vocab, strong), "fresh": (fresh, fr_pairs)}
    res = dict(arch=arch, m=m, V=V, seed=seed, steps=steps, coverage=coverage, q=q, n_train=int(len(train)),
               final_loss=float(loss), train_secs=train_secs)
    feats = {}
    for name, (P, pr) in sets.items():
        za, zb = P[pr[:, 0]], P[pr[:, 1]]
        for rin in (0, 1):
            h = enc(za, zb, rin)
            feats[(name, rin)] = (h, za, zb)
            for rout in (0, 1):
                y = world.render(za, zb, rout)
                yrev = world.render(zb, za, rout)
                out = dec(h, rout)
                res[f"beh_{name}_r{rin}{rout}"] = r2(out, y)
                e_ok = ((out - y) ** 2).reshape(len(y), -1).sum(1)
                e_rev = ((out - yrev) ** 2).reshape(len(y), -1).sum(1)
                res[f"roleerr_{name}_r{rin}{rout}"] = float((e_rev < e_ok).mean())
            # edit leakage: change role-1 part only, measure change in role-2 region of BESIDE output
            za2 = P[np.roll(pr[:, 0], 1)]
            h2 = enc(za2, zb, rin)
            o1, o2 = dec(h, 0), dec(h2, 0)
            dl = np.abs(o2[:, :, :8] - o1[:, :, :8]).mean()
            dr = np.abs(o2[:, :, 8:] - o1[:, :, 8:]).mean()
            res[f"leak_{name}_r{rin}"] = float(dr / (dl + 1e-9))
    # readouts of each role's latent from h, fit on train composites (per input relation)
    for rin in (0, 1):
        htr, zatr, zbtr = feats[("train", rin)]
        # use the full train set for fitting
        za_full, zb_full = vocab[train[:, 0]], vocab[train[:, 1]]
        h_full = enc(za_full, zb_full, rin)
        mu, sd = h_full.mean(0), h_full.std(0) + 1e-6
        for role, ztr in (("1", za_full), ("2", zb_full)):
            lin = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit((h_full - mu) / sd, ztr)
            mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=300, early_stopping=len(ztr) > 50,
                               random_state=seed).fit((h_full - mu) / sd, ztr)
            for name in ("heldout", "strong", "fresh"):
                h, za, zb = feats[(name, rin)]
                zt = za if role == "1" else zb
                res[f"probe_lin_{name}_r{rin}_role{role}"] = r2(lin.predict((h - mu) / sd), zt)
                res[f"probe_mlp_{name}_r{rin}_role{role}"] = r2(mlp.predict((h - mu) / sd), zt)
            # the same readout targeting the part's glyph pixels
            gtr = world.glyph(ztr).reshape(len(ztr), -1)
            linp = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit((h_full - mu) / sd, gtr)
            for name in ("heldout", "strong", "fresh"):
                h, za, zb = feats[(name, rin)]
                gt = world.glyph(za if role == "1" else zb).reshape(len(za), -1)
                res[f"probe_pix_{name}_r{rin}_role{role}"] = r2(linp.predict((h - mu) / sd), gt)
        # additivity over a grid of parts
        n = min(10, V - max(1, V // 4))
        for gname, P in (("fam", vocab[:n]), ("fresh", fresh[:10])):
            k = len(P)
            ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
            hg = enc(P[ii.ravel()], P[jj.ravel()], rin).reshape(k, k, -1)
            res[f"inter_{gname}_r{rin}"] = anova_interaction(hg)
            # role factorization: is a part's role-2 effect a fixed linear map of its
            # role-1 effect (phi_2 = B o phi_1)?  Fit B on half the parts, test on the rest.
            mu_g = hg.mean((0, 1))
            alpha, beta = hg.mean(1) - mu_g, hg.mean(0) - mu_g
            half = k // 2
            Bfit = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit(alpha[:half], beta[:half])
            res[f"rolefac_{gname}_r{rin}"] = r2(Bfit.predict(alpha[half:]), beta[half:])
            xg = world.render(P[ii.ravel()], P[jj.ravel()], rin).reshape(k, k, -1)
            res[f"inter_input_{gname}_r{rin}"] = anova_interaction(xg)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="mlp,cnn,tf,slot")
    ap.add_argument("--ms", default="8,16,32,64")
    ap.add_argument("--Vs", default="16")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--coverages", default="1.0")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--second", default="over")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    SECOND["rel"] = a.second
    os.makedirs(a.outdir, exist_ok=True)
    sep = lambda s: s.replace("+", ",").split(",")
    for seed in map(int, sep(a.seeds)):
        for V in map(int, sep(a.Vs)):
            for cov in map(float, sep(a.coverages)):
                for arch in sep(a.archs):
                    for m in map(int, sep(a.ms)):
                        fn = os.path.join(a.outdir, f"{arch}_m{m}_V{V}_c{cov:g}_s{seed}.json")
                        if os.path.exists(fn):
                            continue
                        r = run(arch, m, V, seed, a.steps, cov)
                        r["second"] = a.second
                        with open(fn, "w", encoding="utf-8") as f:
                            json.dump(r, f, indent=1)
                        print(fn, f"loss={r['final_loss']:.4f} t={r['train_secs']:.0f}s "
                              f"beh_ho={r['beh_heldout_r11']:.3f} beh_fr={r['beh_fresh_r11']:.3f} "
                              f"plin_fr={r['probe_lin_fresh_r1_role1']:.3f}", flush=True)
