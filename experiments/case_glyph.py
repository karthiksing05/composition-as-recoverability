"""Case studies for the glyph networks (MLP, m = 64, OVER input, seed 0).

decode   for fresh pairs: the input composite, the target BESIDE rendering, the network's own
         rendering, and the glyphs a population readout reads from the state (V = 8 and 64)
swap     an interchange intervention on role 1 through the fitted role maps (V = 64), against
         the natural direction of the same norm
Writes results/case/glyph.npz."""
import os
import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeCV
import e2_glyph as E

E.SECOND["rel"] = "over"
K, M, SEED, NEX = 16, 64, 0, 5
out = {}
for V in (8, 64):
    ctx = E.train_model("mlp", M, V, SEED, 4000)
    model, world = ctx["model"], ctx["world"]

    def enc(za, zb, rin=1):
        with torch.no_grad():
            return model.encode(torch.tensor(world.render(za, zb, rin), dtype=torch.float32)).numpy()

    def dec(h, rout=0):
        with torch.no_grad():
            return model.decode(torch.tensor(h, dtype=torch.float32), torch.full((len(h),), rout, dtype=torch.long)).numpy()

    prng = np.random.default_rng(20_000 + SEED)
    pool = prng.normal(size=(400, E.Q)); pp = prng.integers(0, 400, (2000, 2))
    fresh = np.random.default_rng(777).normal(size=(2 * NEX + 2 * 3, E.Q))
    a, b = fresh[:NEX], fresh[NEX:2 * NEX]
    # population readout of both glyphs' pixels from the state, fitted on pool composites
    za, zb = pool[pp[:, 0]], pool[pp[:, 1]]
    H = enc(za, zb)
    Y = np.concatenate([world.glyph(za).reshape(len(za), -1), world.glyph(zb).reshape(len(zb), -1)], 1)
    mu, sd = H.mean(0), H.std(0) + 1e-6
    lin = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit((H - mu) / sd, Y)
    h = enc(a, b)
    rd = np.clip(lin.predict((h - mu) / sd), 0, 1).reshape(NEX, 2, 8, 8)
    out[f"V{V}_input"] = world.render(a, b, 1)
    out[f"V{V}_target"] = world.render(a, b, 0)
    out[f"V{V}_net"] = np.clip(dec(h), 0, 1)
    out[f"V{V}_readout"] = np.concatenate([rd[:, 0], rd[:, 1]], 2)
    if V == 64:
        rng = np.random.default_rng(10_000 + SEED)
        G = world.glyph(rng.normal(size=(4000, E.Q))).reshape(4000, -1)
        gmu = G.mean(0); _, _, Vt = np.linalg.svd(G - gmu, full_matrices=False)
        code = lambda z: (world.glyph(z).reshape(len(z), -1) - gmu) @ Vt[:K].T
        X = np.concatenate([code(za), code(zb)], 1)
        fit = Ridge(alpha=1e-3).fit(X, H); B1 = fit.coef_[:, :K]
        U = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(H, X)
        s_a, s_b = fresh[2 * NEX:2 * NEX + 3], fresh[2 * NEX + 3:]
        d_a = np.random.default_rng(778).normal(size=(3, E.Q)); d_b = np.random.default_rng(779).normal(size=(3, E.Q))
        h0, hd = enc(s_a, s_b), enc(d_a, d_b)
        dlt = (U.predict(hd)[:, :K] - U.predict(h0)[:, :K]) @ B1.T
        nat = hd - h0
        nat = nat / np.linalg.norm(nat, axis=1, keepdims=True) * np.linalg.norm(dlt, axis=1, keepdims=True)
        out["swap_orig"] = np.clip(dec(h0), 0, 1)
        out["swap_edit"] = np.clip(dec(h0 + dlt), 0, 1)
        out["swap_nat"] = np.clip(dec(h0 + nat), 0, 1)
        out["swap_target"] = world.render(d_a, s_b, 0)
        out["swap_donor"] = world.glyph(d_a)
os.makedirs("results/case", exist_ok=True)
np.savez("results/case/glyph.npz", **out)
print({k: v.shape for k, v in out.items()})
