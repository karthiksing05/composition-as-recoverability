"""E2 references, with the same population readout as e2_glyph.run (no network is trained):
  input      the rendered composite itself (identity encoder, 128 pixels)
  concat_m   role-filler reference: the two glyphs' pixels concatenated and randomly projected to m
  sum_m      blend reference: the sum of the two glyphs' pixels randomly projected to m (m <= 64)
and a vocabulary readout fitted on the raw input of each run's training composites (same
vocabulary and pairs as e2_glyph.train_model for that seed), to test whether the vocabulary
readout tracks behavior because of the vocabulary's geometry rather than the trained state."""
import argparse, json, os
import numpy as np
from sklearn.linear_model import RidgeCV
import e2_glyph as E


def readout_r2(Hfit, g1, g2, Hev, t1, t2):
    mu, sd = Hfit.mean(0), Hfit.std(0) + 1e-6
    out = []
    for gt, tt in ((g1, t1), (g2, t2)):
        lin = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit((Hfit - mu) / sd, gt)
        out.append(E.r2(lin.predict((Hev - mu) / sd), tt))
    return float(np.mean(out))


def run(V, seed, q=E.Q):
    rng = np.random.default_rng(seed)
    world = E.World(seed, q)
    vocab = rng.normal(size=(V, q))
    rng.normal(size=(64, q))
    train, heldout, strong = E.splits(V, rng)
    prng = np.random.default_rng(20_000 + seed)
    pool = prng.normal(size=(400, q))
    pool_pairs = prng.integers(0, 400, (2000, 2))
    prng.normal(size=(256, q)); fresh_p = prng.normal(size=(256, q))
    fpairs = np.stack([np.arange(0, 256, 2), np.arange(1, 256, 2)], 1)
    fpairs = np.concatenate([fpairs, fpairs[:, ::-1]])
    G = lambda z: world.glyph(z).reshape(len(z), -1)
    za, zb = pool[pool_pairs[:, 0]], pool[pool_pairs[:, 1]]
    fa, fb = fresh_p[fpairs[:, 0]], fresh_p[fpairs[:, 1]]
    va, vb = vocab[train[:, 0]], vocab[train[:, 1]]
    res = dict(V=V, seed=seed)
    proj = np.random.default_rng(30_000 + seed)
    Pc = {m: proj.normal(size=(128, m)) / np.sqrt(128) for m in (8, 16, 32, 64)}
    Ps = {m: proj.normal(size=(64, m)) / np.sqrt(64) for m in (8, 16, 32, 64)}
    for rel in ("over", "occ"):
        E.SECOND["rel"] = rel
        for rin in (0, 1):
            tag = f"{rel}_r{rin}"
            Xp = world.render(za, zb, rin).reshape(len(za), -1)
            Xf = world.render(fa, fb, rin).reshape(len(fa), -1)
            res[f"input_pool_{tag}"] = readout_r2(Xp, G(za), G(zb), Xf, G(fa), G(fb))
            Xv = world.render(va, vb, rin).reshape(len(va), -1)
            res[f"input_vocab_{tag}"] = readout_r2(Xv, G(va), G(vb), Xf, G(fa), G(fb))
    for m in (8, 16, 32, 64):
        Cp, Cf = np.concatenate([G(za), G(zb)], 1) @ Pc[m], np.concatenate([G(fa), G(fb)], 1) @ Pc[m]
        res[f"concat_m{m}"] = readout_r2(Cp, G(za), G(zb), Cf, G(fa), G(fb))
        Sp, Sf = (G(za) + G(zb)) @ Ps[m], (G(fa) + G(fb)) @ Ps[m]
        res[f"sum_m{m}"] = readout_r2(Sp, G(za), G(zb), Sf, G(fa), G(fb))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0+1+2+3+4+5+6+7+8+9")
    ap.add_argument("--Vs", default="8+16+32+64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [run(V, s) for s in map(int, a.seeds.split("+")) for V in map(int, a.Vs.split("+"))]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("wrote", a.out, len(rows))
