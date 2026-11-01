"""Aggregate E17: image-text matching of composites (roles against identity)."""
import json, glob
import pandas as pd
d = pd.DataFrame([r for f in glob.glob("results/e17/*.json") for r in json.load(open(f, encoding="utf-8"))])
print(d.groupby(["model", "rel"])[["roles", "roles_first", "roles_second", "identity"]].mean().round(3).to_string())
print(d.groupby("model")[["roles", "identity"]].agg(["mean", "sem"]).round(3).to_string())
print("seeds", d.groupby("model").seed.nunique().to_dict())
