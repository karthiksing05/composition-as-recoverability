"""Aggregate E4 (v2 protocol): per model, choose layer and part code on validation nouns
(mean rec_val over roles and the two asymmetric relations 'on' and 'left'), then report
fresh-noun recovery for the model and the two references under that choice."""
import json, glob
import numpy as np, pandas as pd

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
rows = []
for f in glob.glob("results/e4b/*.json"):
    with open(f, encoding="utf-8") as fh:
        rows += json.load(fh)
df = pd.DataFrame(rows)
for s in ("pairs", "val", "fresh"):
    df[f"rec_{s}"] = df[[f"rec_{s}_role1", f"rec_{s}_role2"]].mean(1)
for s in ("val", "fresh"):
    df[f"id_{s}"] = df[[f"id_{s}_role1", f"id_{s}_role2"]].mean(1)
    df[f"role_{s}"] = df[[f"role_{s}_role1", f"role_{s}_role2"]].mean(1)
asym = ["on", "left", "under", "inside", "behind"]
mdl = df[(df.system == "model") & df.rel.isin(asym)]
sel = mdl.groupby(["model", "layer", "code"]).rec_val.mean().reset_index()
best = sel.loc[sel.groupby("model").rec_val.idxmax()].set_index("model")
print("selected (layer, code) on validation nouns:\n", best.to_string())
out = []
for m, b in best.iterrows():
    d = df[(df.model == m) & (df.layer == b.layer) & (df.code == b.code)]
    for (sysn, rel), g in d.groupby(["system", "rel"]):
        out.append(dict(model=m, layer=b.layer, code=b.code, system=sysn, rel=rel, n=len(g),
                        **{k: g[k].mean() for k in ("rec_pairs", "rec_val", "rec_fresh", "rec_fresh_role1", "rec_fresh_role2",
                                                    "id_fresh", "role_fresh", "rev_sim", "inter")},
                        rec_fresh_sem=g.rec_fresh.std(ddof=1) / np.sqrt(len(g))))
o = pd.DataFrame(out)
o.to_csv("results/e4b_summary.csv", index=False)
piv = o.pivot_table(index=["model", "rel"], columns="system", values="rec_fresh").round(3)
print("\nrec_fresh by system:\n", piv.to_string())
a = o[o.rel.isin(asym)].groupby(["model", "system"])[["rec_pairs", "rec_fresh", "rec_fresh_role1", "rec_fresh_role2", "id_fresh", "role_fresh", "rev_sim", "inter"]].mean().round(3)
print("\nmean over asymmetric relations:\n", a.to_string())
print("\n'and' vs asymmetric (model):\n", o[o.system == "model"].assign(sym=lambda x: x.rel == "and").groupby(["model", "sym"])[["rec_fresh", "role_fresh", "rev_sim"]].mean().round(3).unstack().to_string())
# sensitivity: best layer per model chosen on fresh (oracle) vs val, to show selection is not driving results
orc = mdl.groupby(["model", "layer", "code"]).rec_fresh.mean().groupby("model").max()
print("\noracle-best rec_fresh (upper bound on selection):\n", orc.round(3).to_string())
