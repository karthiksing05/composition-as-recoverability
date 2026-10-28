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
import argparse, copy, json, os, time
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


def train_model(arch, m, V, seed, steps, coverage=1.0, q=Q, npairs=None, aux=0.0):
    """Train one system.  npairs fixes the number of distinct training pairs (to vary
    part diversity at a fixed number of combinations); aux > 0 adds a jointly trained
    linear readout from h to both glyphs, an auxiliary pressure toward recoverability."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    world = World(seed, q)
    vocab = rng.normal(size=(V, q))
    fresh = rng.normal(size=(64, q))
    train, heldout, strong = splits(V, rng)
    if coverage < 1.0:
        train = train[: max(2, int(round(coverage * len(train))))]
    if npairs is not None:
        train = train[:npairs]
    model = make_model(arch, m)
    init_model = copy.deepcopy(model)  # untrained copy for the selectivity control (consumes no RNG)
    # created only when used, so that default runs keep the original random stream
    head = nn.Linear(m, 128) if aux > 0 else None
    params = list(model.parameters()) + (list(head.parameters()) if aux > 0 else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    # precompute all training composites
    cache = {}
    for rin in (0, 1):
        cache[rin] = torch.tensor(world.render(vocab[train[:, 0]], vocab[train[:, 1]], rin), dtype=torch.float32)
    gcache = torch.tensor(np.concatenate([world.glyph(vocab[train[:, 0]]).reshape(len(train), -1),
                                          world.glyph(vocab[train[:, 1]]).reshape(len(train), -1)], 1), dtype=torch.float32)
    t0 = time.time()
    bs = 256
    for step in range(steps):
        idx = torch.randint(0, len(train), (bs,))
        rin = torch.randint(0, 2, (bs,))
        rout = torch.randint(0, 2, (bs,))
        x = torch.where(rin.view(-1, 1, 1) == 0, cache[0][idx], cache[1][idx])
        y = torch.where(rout.view(-1, 1, 1) == 0, cache[0][idx], cache[1][idx])
        h = model.encode(x)
        loss = F.mse_loss(model.decode(h, rout), y)
        total = loss + aux * F.mse_loss(head(h), gcache[idx]) if aux > 0 else loss
        opt.zero_grad(); total.backward(); opt.step(); sched.step()
    model.eval()
    init_model.eval()
    return dict(model=model, init_model=init_model, world=world, vocab=vocab, fresh=fresh, train=train, heldout=heldout, strong=strong,
                loss=float(loss), train_secs=time.time() - t0, rng=rng)


def ey_ceiling(world, m, rng, n=4000):
    """Eckart-Young ceiling: the largest R^2 any affine readout from an m-dimensional
    state can reach for the stacked pixels of two independent glyphs."""
    g = world.glyph(rng.normal(size=(n, world.q))).reshape(n, -1)
    ev = np.sort(np.linalg.eigvalsh(np.cov(g.T)))[::-1]
    ev2 = np.sort(np.concatenate([ev, ev]))[::-1]
    return float(ev2[:m].sum() / ev2.sum())


def run(arch, m, V, seed, steps, coverage=1.0, q=Q, npairs=None, aux=0.0):
    ctx = train_model(arch, m, V, seed, steps, coverage, q, npairs, aux)
    model, world, vocab = ctx["model"], ctx["world"], ctx["vocab"]
    train, heldout, strong, loss, train_secs = ctx["train"], ctx["heldout"], ctx["strong"], ctx["loss"], ctx["train_secs"]
    # parts never seen in training: a probe pool (to fit population readouts) and two
    # disjoint evaluation sets (one for behaviour, one for readouts)
    prng = np.random.default_rng(20_000 + seed)
    pool = prng.normal(size=(400, q))
    pool_pairs = prng.integers(0, 400, (2000, 2))
    fresh_b, fresh_p = prng.normal(size=(256, q)), prng.normal(size=(256, q))
    fpairs = np.stack([np.arange(0, 256, 2), np.arange(1, 256, 2)], 1)
    fpairs = np.concatenate([fpairs, fpairs[:, ::-1]])

    def enc(za, zb, rin, mdl=model):
        with torch.no_grad():
            return mdl.encode(torch.tensor(world.render(za, zb, rin), dtype=torch.float32)).numpy()

    def dec(h, rout):
        with torch.no_grad():
            return model.decode(torch.tensor(h, dtype=torch.float32), torch.full((len(h),), rout, dtype=torch.long)).numpy()

    res = dict(arch=arch, m=m, V=V, seed=seed, steps=steps, coverage=coverage, q=q, n_train=int(len(train)),
               npairs=npairs, aux=aux, final_loss=float(loss), train_secs=train_secs, eval_version=2,
               ey_ceiling=ey_ceiling(world, m, prng))
    beh_sets = {"train": (vocab, train[:512]), "heldout": (vocab, heldout), "strong": (vocab, strong), "fresh": (fresh_b, fpairs)}
    for name, (P, pr) in beh_sets.items():
        za, zb = P[pr[:, 0]], P[pr[:, 1]]
        for rin in (0, 1):
            h = enc(za, zb, rin)
            for rout in (0, 1):
                y, yrev = world.render(za, zb, rout), world.render(zb, za, rout)
                out = dec(h, rout)
                res[f"beh_{name}_r{rin}{rout}"] = r2(out, y)
                e_ok = ((out - y) ** 2).reshape(len(y), -1).sum(1)
                e_rev = ((out - yrev) ** 2).reshape(len(y), -1).sum(1)
                res[f"roleerr_{name}_r{rin}{rout}"] = float((e_rev < e_ok).mean())
                if name == "fresh":  # per-composite errors, for within-condition analyses
                    res[f"beh_fresh_items_r{rin}{rout}"] = (1 - e_ok / ((y - y.mean(0)) ** 2).reshape(len(y), -1).sum(1).mean()).round(4).tolist()
            # edit leakage: change role-1 part only, measure change in role-2 region of BESIDE output
            h2 = enc(P[np.roll(pr[:, 0], 1)], zb, rin)
            o1, o2 = dec(h, 0), dec(h2, 0)
            res[f"leak_{name}_r{rin}"] = float(np.abs(o2[:, :, 8:] - o1[:, :, 8:]).mean() / (np.abs(o2[:, :, :8] - o1[:, :, :8]).mean() + 1e-9))
    probe_eval = {"heldout": (vocab, heldout), "strong": (vocab, strong), "fresh": (fresh_p, fpairs)}
    for rin in (0, 1):
        fits = {}
        # population readouts: fit on composites of pool parts (never trained on)
        for tag, mdl in (("pool", model), ("init", ctx["init_model"])):
            za, zb = pool[pool_pairs[:, 0]], pool[pool_pairs[:, 1]]
            fits[tag] = (mdl, enc(za, zb, rin, mdl), world.glyph(za).reshape(len(za), -1), world.glyph(zb).reshape(len(zb), -1))
        # vocabulary readouts: fit on the system's own training composites (the biased protocol)
        za, zb = vocab[train[:, 0]], vocab[train[:, 1]]
        fits["vocab"] = (model, enc(za, zb, rin), world.glyph(za).reshape(len(za), -1), world.glyph(zb).reshape(len(zb), -1))
        for tag, (mdl, Hfit, g1, g2) in fits.items():
            mu, sd = Hfit.mean(0), Hfit.std(0) + 1e-6
            for role, gt in (("1", g1), ("2", g2)):
                lin = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit((Hfit - mu) / sd, gt)
                for name, (P, pr) in probe_eval.items():
                    za, zb = P[pr[:, 0]], P[pr[:, 1]]
                    h = enc(za, zb, rin, mdl)
                    tgt = world.glyph(za if role == "1" else zb).reshape(len(za), -1)
                    res[f"probe_{tag}_{name}_r{rin}_role{role}"] = r2(lin.predict((h - mu) / sd), tgt)
        # additivity over a grid of fresh parts
        P = fresh_p[:10]
        ii, jj = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
        res[f"inter_fresh_r{rin}"] = anova_interaction(enc(P[ii.ravel()], P[jj.ravel()], rin).reshape(10, 10, -1))
        res[f"inter_input_fresh_r{rin}"] = anova_interaction(world.render(P[ii.ravel()], P[jj.ravel()], rin).reshape(10, 10, -1))
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
    ap.add_argument("--q", type=int, default=Q)
    ap.add_argument("--npairs", type=int, default=None)
    ap.add_argument("--auxs", default="0")
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
                      for aux in map(float, sep(a.auxs)):
                        tag = (f"_q{a.q}" if a.q != Q else "") + (f"_n{a.npairs}" if a.npairs else "") + (f"_aux{aux:g}" if aux else "")
                        fn = os.path.join(a.outdir, f"{arch}_m{m}_V{V}_c{cov:g}{tag}_s{seed}.json")
                        if os.path.exists(fn):
                            continue
                        r = run(arch, m, V, seed, a.steps, cov, q=a.q, npairs=a.npairs, aux=aux)
                        r["second"] = a.second
                        with open(fn, "w", encoding="utf-8") as f:
                            json.dump(r, f, indent=1)
                        print(fn, f"loss={r['final_loss']:.4f} t={r['train_secs']:.0f}s "
                              f"beh_ho={r['beh_heldout_r11']:.3f} beh_fr={r['beh_fresh_r11']:.3f} "
                              f"pool_fr={r['probe_pool_fresh_r1_role1']:.3f} vocab_fr={r['probe_vocab_fresh_r1_role1']:.3f} init_fr={r['probe_init_fresh_r1_role1']:.3f}", flush=True)
