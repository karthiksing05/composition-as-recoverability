"""Aggregate E42 -> results/e42/summary.json and a printed table.  usage: e42_agg.py [dir]"""
import glob, json, sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
d = sys.argv[1] if len(sys.argv) > 1 else "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e42"
rows = []
for f in sorted(glob.glob(f"{d}/*_s[0-9].json")):
    rows += json.load(open(f, encoding="utf-8"))
df = pd.DataFrame(rows)
g = df.groupby(["model", "cond", "readout", "layer_tag", "probe"])
T = g.agg(kappa=("kappa", "mean"), kappa_sd=("kappa", "std"), kappa_linref=("kappa_linref", "mean"), r2=("r2", "mean"),
          ref_concat=("ref_concat", "mean"), ref_sum=("ref_sum", "mean"), role_acc=("role_acc", "mean"), n=("seed", "nunique"),
          layer=("layer", lambda x: ",".join(sorted(set(x))))).reset_index()
summ = {"table": json.loads(T.to_json(orient="records")), "per_seed": json.loads(df.drop(columns=["mlp_wd"], errors="ignore").to_json(orient="records"))}
with open(f"{d}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=1)
T.to_csv(f"{d}/summary_table.csv", index=False)
cell = lambda r: f"{r.kappa:.2f}({r.kappa_sd:.2f})"
print("=== a/c: kappa mean(sd) over seeds, own-probe refs | kappa_linref")
A = T[T.readout.isin(["G2", "OBJr"])].copy()
A["v"] = A.apply(lambda r: f"{r.kappa:.2f}({r.kappa_sd:.2f})|{r.kappa_linref:.2f} {r.layer}", axis=1)
print(A.pivot_table(index=["model", "cond", "readout", "layer_tag"], columns="probe", values="v", aggfunc="first").to_string())
print("=== b: constructed controls (rand, OBJr slot order shuffled)")
B = T[T.readout.str.startswith("ctl_")].copy()
B["v"] = B.apply(lambda r: f"{r.kappa:.2f}({r.kappa_sd:.2f})|{r.kappa_linref:.2f} acc{r.role_acc:.2f}", axis=1)
print(B.pivot_table(index=["model", "readout"], columns="probe", values="v", aggfunc="first").to_string())
# ---- follow-up keys: random-weight encoders, absolute-position controls
rw = T[T.model.str.startswith("rand-") & T.readout.isin(["G2", "OBJr"]) & T.layer_tag.isin(["sel", "mid", "final"])]
ab = T[T.readout.str.startswith("ctl_")]
summ["random_weight"] = json.loads(rw.to_json(orient="records"))
summ["absx_controls"] = json.loads(ab[ab.readout.str.startswith("ctl_absx")].to_json(orient="records"))
summ["all_controls"] = json.loads(ab.to_json(orient="records"))
with open(f"{d}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=1)
print("=== random-weight encoders (G2/OBJr, sel layer, kappa|linref)")
R = rw[rw.layer_tag == "sel"].copy(); R["v"] = R.apply(lambda r: f"{r.kappa:.2f}({r.kappa_sd:.2f})|{r.kappa_linref:.2f} {r.layer}", axis=1)
print(R.pivot_table(index=["model", "cond", "readout"], columns="probe", values="v", aggfunc="first").to_string())
print("=== all constructed controls incl. absx (kappa(sd)|linref acc)")
B = ab.copy(); B["v"] = B.apply(lambda r: f"{r.kappa:.2f}({r.kappa_sd:.2f})|{r.kappa_linref:.2f} acc{r.role_acc:.2f}", axis=1)
print(B.pivot_table(index=["model", "readout"], columns="probe", values="v", aggfunc="first").to_string())
