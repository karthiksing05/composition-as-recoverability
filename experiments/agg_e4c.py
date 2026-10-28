"""Aggregate E4 (round 3): penalty chosen on validation nouns, PCA on fitting nouns, a
curve over the number of fitting nouns, and randomly initialized models. Layer and code are
chosen per model on validation nouns at n_fit = 165 (mean over the five asymmetric relations)."""
import json, glob
import numpy as np, pandas as pd

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
rows = []
for f in glob.glob("results/e4c/*.json"):
    rows += json.load(open(f, encoding="utf-8"))
df = pd.DataFrame(rows)
for s in ("pairs", "val", "fresh"):
    df[f"rec_{s}"] = df[[f"rec_{s}_role1", f"rec_{s}_role2"]].mean(1)
for s in ("val", "fresh"):
    df[f"id_{s}"] = df[[f"id_{s}_role1", f"id_{s}_role2"]].mean(1)
    df[f"role_{s}"] = df[[f"role_{s}_role1", f"role_{s}_role2"]].mean(1)
asym = ["on", "left", "under", "inside", "behind"]
A = df[df.rel.isin(asym)]
sel = A[(A.system == "model") & (A.n_fit == 165)].groupby(["model", "layer", "code"]).rec_val.mean().reset_index()
best = sel.loc[sel.groupby("model").rec_val.idxmax()].set_index("model")
print("selected on validation nouns (n_fit=165):\n", best.to_string())
B = A.merge(best[["layer", "code"]].reset_index(), on=["model", "layer", "code"])
K = ["rec_pairs", "rec_fresh", "rec_fresh_role1", "rec_fresh_role2", "id_fresh", "role_fresh", "rev_sim", "inter"]
t = B[B.n_fit == 165].groupby(["model", "system"])[K].mean().round(3)
print("\n=== n_fit = 165, mean over asymmetric relations and seeds\n", t.to_string())
B[B.n_fit == 165].groupby(["model", "system"])[K].agg(["mean", "sem"]).to_csv("results/e4c_summary.csv")
print("\n=== curve: rec_fresh by n_fit\n", B.pivot_table(index=["model", "system"], columns="n_fit", values="rec_fresh").round(3).to_string())
print("\n=== curve: id_fresh by n_fit (model)\n", B[B.system == "model"].pivot_table(index="model", columns="n_fit", values="id_fresh").round(3).to_string())
# kappa: position between the sum (blend) and concatenation (role-filler) references, on R^2
p = t["rec_fresh"].unstack("system")
p["kappa"] = (p["model"] - p["ref_sum"]) / (p["ref_concat"] - p["ref_sum"])
print("\n=== kappa (n_fit=165)\n", p.round(3).to_string())
# rankings by metric
m = t.xs("model", level="system")
print("\nrank by rec_fresh:", list(m.rec_fresh.sort_values(ascending=False).index))
print("rank by id_fresh:", list(m.id_fresh.sort_values(ascending=False).index))
# per-model sum-reference identification
print("\nid_fresh model vs own sum reference:\n", t["id_fresh"].unstack("system").round(3).to_string())
# 'and' control at the selected layer/code
C = df[(df.rel == "and") & (df.n_fit == 165)].merge(best[["layer", "code"]].reset_index(), on=["model", "layer", "code"])
print("\n'and' control (model):\n", C[C.system == "model"].groupby("model")[["rec_fresh", "role_fresh", "rev_sim"]].mean().round(3).to_string())
# code robustness at the selected layer: rec_fresh of the model for each code
L = A[(A.system == "model") & (A.n_fit == 165)].merge(best[["layer"]].reset_index(), on=["model", "layer"])
print("\nrec_fresh by part code at the selected layer:\n", L.pivot_table(index="model", columns="code", values="rec_fresh").round(3).to_string())
print("\nselected alphas (model, role1):\n", B[(B.system == "model") & (B.n_fit == 165)].groupby("model").alpha_role1.median().to_string())
