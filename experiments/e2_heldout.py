"""E2 addendum: how the ridge penalty of a readout is selected (round-3 review).

Retrains the e2_glyph networks (same seeds, so the same systems) and refits the vocabulary,
population, untrained-population, and raw-input vocabulary readouts under three rules:
  cv     RidgeCV (leave-one-composite-out) on standardized states: the rule of e2_glyph.run
  glyph  leave-glyphs-out cross-validation within the training vocabulary (folds group the
         composites by the target role's glyph), on centred, unstandardized states
  fresh  penalty chosen on composites of a separate set of fresh validation glyphs
The "glyph" rule uses only the vocabulary; the "fresh" rule is an oracle for the penalty.
All readouts are evaluated on the same fresh glyphs as e2_glyph.run (role 1 and 2, rin = 1)."""
import argparse, json, os
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
import e2_glyph as E

ALPHAS = np.logspace(-4, 5, 19)


class Path:
    """Ridge solutions for every penalty in ALPHAS from one SVD of centred X."""

    def __init__(self, X, Y):
        self.mx, self.my = X.mean(0), Y.mean(0)
        U, self.s, self.Vt = np.linalg.svd(X - self.mx, full_matrices=False)
        self.UtY = U.T @ (Y - self.my)

    def W(self, a):
        return self.Vt.T @ ((self.s / (self.s ** 2 + a))[:, None] * self.UtY)

    def predict(self, X, a):
        return (X - self.mx) @ self.W(a) + self.my


def pick_glyph_cv(X, Y, groups, k=5):
    g = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(g)
    folds = np.array_split(g, min(k, len(g)))
    sc = np.zeros(len(ALPHAS))
    for f in folds:
        te = np.isin(groups, f)
        p = Path(X[~te], Y[~te])
        # squared error, not R^2: a fold may hold a single glyph, whose targets are constant
        sc -= [((p.predict(X[te], a) - Y[te]) ** 2).sum() for a in ALPHAS]
    return ALPHAS[int(np.argmax(sc))]


def readouts(Xfit, Yfit, groups, Xval, Yval, Xev, Yev):
    out = {}
    mu, sd = Xfit.mean(0), Xfit.std(0) + 1e-6
    lin = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit((Xfit - mu) / sd, Yfit)
    out["cv"] = E.r2(lin.predict((Xev - mu) / sd), Yev)
    p = Path(Xfit, Yfit)
    a = pick_glyph_cv(Xfit, Yfit, groups)
    out["glyph"], out["glyph_alpha"] = E.r2(p.predict(Xev, a), Yev), float(a)
    a = ALPHAS[int(np.argmax([E.r2(p.predict(Xval, b), Yval) for b in ALPHAS]))]
    out["fresh"], out["fresh_alpha"] = E.r2(p.predict(Xev, a), Yev), float(a)
    return out


def run(arch, m, V, seed, steps):
    ctx = E.train_model(arch, m, V, seed, steps)
    model, world, vocab, train = ctx["model"], ctx["world"], ctx["vocab"], ctx["train"]
    prng = np.random.default_rng(20_000 + seed)  # the same streams as e2_glyph.run
    pool = prng.normal(size=(400, E.Q))
    pool_pairs = prng.integers(0, 400, (2000, 2))
    fresh_b, fresh_p = prng.normal(size=(256, E.Q)), prng.normal(size=(256, E.Q))
    fpairs = np.stack([np.arange(0, 256, 2), np.arange(1, 256, 2)], 1)
    fpairs = np.concatenate([fpairs, fpairs[:, ::-1]])
    vrng = np.random.default_rng(40_000 + seed)
    val = vrng.normal(size=(128, E.Q))
    vpairs = vrng.integers(0, 128, (512, 2))
    G = lambda z: world.glyph(z).reshape(len(z), -1)
    rin = 1

    def enc(za, zb, mdl):
        with torch.no_grad():
            return mdl.encode(torch.tensor(world.render(za, zb, rin), dtype=torch.float32)).numpy()

    def dec(h):
        with torch.no_grad():
            return model.decode(torch.tensor(h, dtype=torch.float32), torch.full((len(h),), 1, dtype=torch.long)).numpy()

    res = dict(arch=arch, m=m, V=V, seed=seed, steps=steps)
    fa, fb = fresh_p[fpairs[:, 0]], fresh_p[fpairs[:, 1]]
    va, vb = val[vpairs[:, 0]], val[vpairs[:, 1]]
    ta, tb = vocab[train[:, 0]], vocab[train[:, 1]]
    pa, pb = pool[pool_pairs[:, 0]], pool[pool_pairs[:, 1]]
    # behaviour on fresh glyphs, recomputed to check that the retrained system matches e2v2
    ba, bb = fresh_b[fpairs[:, 0]], fresh_b[fpairs[:, 1]]
    res["beh_fresh_r11"] = E.r2(dec(enc(ba, bb, model)), world.render(ba, bb, 1))
    raw = lambda za, zb, mdl=None: world.render(za, zb, rin).reshape(len(za), -1)
    systems = {"vocab": (lambda a, b: enc(a, b, model), ta, tb, train),
               "pool": (lambda a, b: enc(a, b, model), pa, pb, pool_pairs),
               "init": (lambda a, b: enc(a, b, ctx["init_model"]), pa, pb, pool_pairs),
               "inputvocab": (raw, ta, tb, train)}
    for tag, (f, za, zb, pr) in systems.items():
        Xfit, Xval, Xev = f(za, zb), f(va, vb), f(fa, fb)
        for role, (gfit, gval, gev, grp) in (("1", (G(za), G(va), G(fa), pr[:, 0])), ("2", (G(zb), G(vb), G(fb), pr[:, 1]))):
            for k, v in readouts(Xfit, gfit, grp, Xval, gval, Xev, gev).items():
                res[f"{tag}_{k}_role{role}"] = v
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="slot+mlp+cnn+tf")
    ap.add_argument("--ms", default="8+16+32+64")
    ap.add_argument("--V", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    E.SECOND["rel"] = "over"
    os.makedirs(a.outdir, exist_ok=True)
    for arch in a.archs.split("+"):
        for m in map(int, a.ms.split("+")):
            fn = os.path.join(a.outdir, f"{arch}_m{m}_V{a.V}_s{a.seed}.json")
            if os.path.exists(fn):
                continue
            r = run(arch, m, a.V, a.seed, a.steps)
            with open(fn, "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
            print(fn, {k: round(v, 3) for k, v in r.items() if k.startswith(("beh", "vocab_", "inputvocab_")) and "alpha" not in k}, flush=True)
