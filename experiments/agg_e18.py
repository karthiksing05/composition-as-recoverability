"""Aggregate E18: paraphrase matching by text encoders (roles, order, identity), beside E7's transfer readout."""
import json, glob
import pandas as pd
d = pd.DataFrame([r for f in glob.glob("results/e18/*.json") for r in json.load(open(f, encoding="utf-8"))])
print(d.groupby(["model", "rel"])[["roles", "order", "identity"]].mean().round(3).to_string())
print(d.groupby("model")[["roles", "order", "identity"]].agg(["mean", "sem"]).round(3).to_string())
print("seeds", d.groupby("model").seed.nunique().to_dict())
e7 = pd.DataFrame([r for f in glob.glob("results/e7/*.json") for r in json.load(open(f, encoding="utf-8"))])
e7 = e7[e7.model.isin(d.model.unique())]
print(e7.columns.tolist())
