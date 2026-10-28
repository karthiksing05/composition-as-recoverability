"""Informativeness of the glyph part codes (Definition 2): affine R^2 from the code e(x)
to the system-independent description z (the glyph latent), fitted on a pool of glyphs and
evaluated on fresh glyphs.  Codes: the 64 glyph pixels (Section 4.2) and their top 16
principal components (Section 4.3)."""
import json
import numpy as np
from sklearn.linear_model import RidgeCV
import e2_glyph as E

out = []
for seed in range(10):
    world = E.World(seed)
    rng = np.random.default_rng(90_000 + seed)
    zf, zt = rng.normal(size=(400, E.Q)), rng.normal(size=(256, E.Q))
    Gf, Gt = world.glyph(zf).reshape(400, -1), world.glyph(zt).reshape(256, -1)
    mu = Gf.mean(0)
    _, _, Vt = np.linalg.svd(Gf - mu, full_matrices=False)
    for tag, f in (("pixels", lambda G: G), ("pc16", lambda G: (G - mu) @ Vt[:16].T)):
        reg = RidgeCV(alphas=np.logspace(-6, 3, 19)).fit(f(Gf), zf)
        out.append(dict(seed=seed, code=tag, r2_fresh=E.r2(reg.predict(f(Gt)), zt), r2_fit=E.r2(reg.predict(f(Gf)), zf)))
json.dump(out, open("results/e2_informative.json", "w", encoding="utf-8"), indent=1)
import pandas as pd
d = pd.DataFrame(out)
print(d.groupby("code")[["r2_fit", "r2_fresh"]].agg(["mean", "std", "min"]).round(3).to_string())
