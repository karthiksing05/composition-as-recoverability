"""E2h: does the trained-state vocabulary readout predict behaviour beyond the raw-input
vocabulary readout, under each penalty rule? Behaviour = rin 1 -> rout 1 on fresh glyphs,
measured on the retrained systems themselves (e2h)."""
import json, glob
import numpy as np, pandas as pd
from agg_r2 import within_corr
h = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e2h/*.json")])
for s in ("vocab", "pool", "init", "inputvocab"):
    for r in ("cv", "glyph", "fresh"):
        h[f"{s}_{r}"] = h[[f"{s}_{r}_role1", f"{s}_{r}_role2"]].mean(1)
K = ["arch", "V", "m"]
print("runs", len(h))
for r in ("cv", "glyph", "fresh"):
    out = {x: round(within_corr(h, f"{x}_{r}", "beh_fresh_r11", K)[0], 3) for x in ("vocab", "pool", "inputvocab")}
    d = h.copy()
    for c in (f"vocab_{r}", f"inputvocab_{r}", "beh_fresh_r11"):
        d[c + "_w"] = d[c] - d.groupby(K)[c].transform("mean")
        d[c + "_w"] /= d[c + "_w"].std()
    X = np.c_[d[f"vocab_{r}_w"], d[f"inputvocab_{r}_w"]]
    beta = np.linalg.lstsq(X, d["beh_fresh_r11_w"], rcond=None)[0]
    rng = np.random.default_rng(0); bs = []; df_ = []
    for _ in range(1000):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        bs.append(np.linalg.lstsq(np.c_[s[f"vocab_{r}_w"], s[f"inputvocab_{r}_w"]], s["beh_fresh_r11_w"], rcond=None)[0])
        df_.append(within_corr(s, f"vocab_{r}", "beh_fresh_r11", K)[0] - within_corr(s, f"pool_{r}", "beh_fresh_r11", K)[0])
    bs = np.array(bs)
    print(r, "within corr", out, "| std. betas (trained, input)", beta.round(3), "CI trained", np.percentile(bs[:, 0], [2.5, 97.5]).round(3),
          "CI input", np.percentile(bs[:, 1], [2.5, 97.5]).round(3), "| vocab-pool within diff CI", np.percentile(df_, [2.5, 97.5]).round(3))
    print("   pooled r:", {x: round(within_corr(h, f"{x}_{r}", "beh_fresh_r11", K)[1], 3) for x in ("vocab", "pool", "inputvocab")})
print(h[h.m == 64].groupby(["V", "arch"])[["beh_fresh_r11", "vocab_cv", "vocab_glyph", "pool_glyph", "init_glyph", "inputvocab_glyph"]].mean().round(3).to_string())
