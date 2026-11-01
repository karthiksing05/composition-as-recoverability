"""Aggregate E12: fresh-glyph and held-out-pair behavior by condition and architecture."""
import json, glob
import numpy as np, pandas as pd
rows = []
for f in glob.glob("results/e12/*.json"):
    r = json.load(open(f, encoding="utf-8"))
    rows.append(dict(arch=r["arch"], cond=r["cond"], seed=r["seed"],
                     fresh=np.mean([r["fresh"]["r10"], r["fresh"]["r11"]]), fresh_all=np.mean(list(r["fresh"].values())),
                     heldout=np.mean([r["heldout"]["r10"], r["heldout"]["r11"]])))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for k in ("fresh", "fresh_all", "heldout"):
    t = d.pivot_table(index="arch", columns="cond", values=k, aggfunc="mean")[["base", "single8", "single64", "single256"]]
    t.loc["mean3 (mlp,cnn,slot)"] = t.loc[["mlp", "cnn", "slot"]].mean()
    print(f"\n== {k} (over_in = mean r10,r11, as in Fig 5b)"); print(t.round(3).to_string())
s = d.pivot_table(index="arch", columns="cond", values="fresh", aggfunc="sem")
print("\n== sem fresh"); print(s.round(3).to_string())
