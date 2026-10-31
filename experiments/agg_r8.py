"""Aggregate round 8: behavioral role answers in LMs (e9), Pythia-410m seeds and dense Pythia-1.4b
checkpoints (e10, with e6), and quadrant-pooled image states (e8grid)."""
import numpy as np, pandas as pd
from agg_r7 import load, e4style, ASYM
from audit_roles import short, per_role

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)

# e9: the model's own answers
b = short(load("e9"))
b["order"] = np.where(b.answer_first, "first", "second")
t = b.groupby(["model", "seed"]).apply(lambda d: pd.Series({
    "choice": d.choice.mean(),
    "first": d[d.answer_first].choice.mean(), "second": d[~d.answer_first].choice.mean(),
    "canon": d[d.phrasing == "canonical"].choice.mean(), "conv": d[d.phrasing == "converse"].choice.mean(),
    "ident": d.ident.mean()}))
e9 = t.groupby("model").mean().join(t.groupby("model").choice.sem().rename("choice_sem"))
print("== e9 behavior (choice = correct noun preferred; 0.5 = word order or chance)")
print(e9.round(3).to_string())
print(b.groupby(["model", "rel"]).choice.mean().unstack().round(2).to_string())
e9.to_csv("results/agg_r8_e9.csv")

# e10 + e6: representational kappa along training
r = pd.concat([short(load("e6")), short(load("e10"))])
r = r[r.model.str.contains("@") | r.model.str.startswith("pythia-410m") | (r.model == "pythia-1.4b")]
k, _ = e4style(r.copy(), ASYM, "r8")
pr = per_role(r.copy(), ASYM)
k = k.join(pr[["kappa1", "kappa2"]])
print("\n== kappa along training (e6 + e10)")
print(k[["layer", "code", "n_seeds", "rec_fresh", "kappa", "kappa_sem", "kappa1", "kappa2", "inter"]].round(3).to_string())
k.to_csv("results/agg_r8_e10.csv")

# e8grid: quadrant-pooled composite states against the mean-pooled references
g = load("e8grid")
for s in ("val", "fresh"):
    g[f"rec_{s}"] = g[[f"rec_{s}_role1", f"rec_{s}_role2"]].mean(1)
    if s == "fresh":
        g["role_fresh"] = g[["role_fresh_role1", "role_fresh_role2"]].mean(1)
        g["id_fresh"] = g[["id_fresh_role1", "id_fresh_role2"]].mean(1)
g["li"] = g.layer.str[1:]
g["pool"] = np.where(g.layer.str.startswith("G"), "grid", "mean")
refs = g[g.system != "model"].pivot_table(index=["model", "seed", "rel", "li", "code"], columns="system", values="rec_fresh")
M = g[g.system == "model"].set_index(["model", "seed", "rel", "li", "code"]).join(refs).reset_index()
M["kappa"] = (M.rec_fresh - M.ref_sum) / (M.ref_concat - M.ref_sum)
rows = []
for (m, pool), d in M.groupby(["model", "pool"]):
    sel = d.groupby(["li", "code"]).rec_val.mean().idxmax()
    dd = d.set_index(["li", "code"]).loc[sel]
    ks = dd.groupby("seed").kappa.mean()
    rows.append(dict(model=m, pool=pool, layer=sel[0], code=sel[1], rec_fresh=dd.rec_fresh.mean(), kappa=ks.mean(),
                     kappa_sem=ks.sem(), role=dd.role_fresh.mean(), ident=dd.id_fresh.mean(), rev_sim=dd.rev_sim.mean(),
                     **{f"k_{rel}": dd[dd.rel == rel].kappa.mean() for rel in ("beside", "above", "overlap")}))
G = pd.DataFrame(rows).set_index(["model", "pool"])
print("\n== e8grid (mean pooling vs quadrant pooling of the composite state)")
print(G.round(3).to_string())
G.to_csv("results/agg_r8_e8grid.csv")
