"""Aggregate E19: ridge, kernel and MLP readouts of the figure, fitted on canonical sentences, scored on converse ones."""
import json, glob
import pandas as pd
d = pd.DataFrame([r for f in glob.glob("results/e19/*.json") for r in json.load(open(f, encoding="utf-8"))])
d["model"] = d.model.str.split("/").str[-1]
cols = [f"{c}_{k}" for c in ("ridge", "kernel", "mlp") for k in ("can", "conv")]
pd.set_option("display.width", 200)
print(d.groupby(["model", "state"])[cols].mean().round(3).to_string())
print(d.groupby(["state"])[cols].agg(["min", "max"]).round(3).to_string())
print("kernel hp", d.kernel_hp.apply(lambda h: (h["bw"], h["alpha"])).value_counts().to_string())
print("mlp hp", d.mlp_hp.apply(lambda h: h["wd"]).value_counts().to_string())
oc = [f"{c}_oracle_{k}" for c in ("kernel", "mlp") for k in ("can", "conv")]
print(d.groupby(["model", "state"])[oc].mean().round(3).to_string())
print(d.groupby("state")[oc].max().round(3).to_string())
print("seeds", d.groupby("model").seed.nunique().to_dict())
