"""Aggregate E12 diagnostics: where the singleton gain stops."""
import json, glob
import numpy as np, pandas as pd, sys
DIR = sys.argv[1] if len(sys.argv) > 1 else "e12b"
rows = []
for f in glob.glob("results/" + DIR + "/*.json"):
    r = json.load(open(f, encoding="utf-8"))
    mo = lambda d: np.mean([d["r10"], d["r11"]])
    rows.append(dict(arch=r["arch"], cond=r["cond"], V=r["V"], share=r["share"], seed=r["seed"], fresh=mo(r["fresh"]), heldout=mo(r["heldout"]),
                     single_r1=r["single_fresh_role1"], single_r2=r["single_fresh_role2"],
                     half_r1=mo(r["half_fresh_role1"]), half_r2=mo(r["half_fresh_role2"])))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
t = d.groupby(["V", "share", "cond", "arch"])[["fresh", "half_r1", "half_r2", "single_r1", "single_r2", "heldout"]].mean()
print(t.round(3).to_string())
print(d.groupby(["V", "share", "cond"])[["fresh", "half_r1", "half_r2", "single_r1", "single_r2", "heldout"]].mean().round(3).to_string())
