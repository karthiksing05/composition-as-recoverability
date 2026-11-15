import json, glob, pandas as pd, numpy as np, sys
sys.path.insert(0,".")
pd.set_option("display.width",220); pd.set_option("display.max_columns",40)
print(pd.read_csv("results/agg_r8_e9.csv",index_col=0).round(3).to_string())
d=pd.DataFrame([r for f in glob.glob("results/e14/*.json") for r in json.load(open(f,encoding="utf-8"))])
print(d.columns.tolist(), d.emb.unique(), d.model.unique())
import subprocess
print(open("agg_r2.py").read()[:1800])
from agg_r2 import load,e2cols
df=e2cols(load("e2v2")); print([c for c in df.columns if "beh" in c][:30]); print(df.arch.unique(), sorted(df.V.unique()), sorted(df.m.unique()))
e11=pd.DataFrame([r for f in glob.glob("results/e11/*.json") for r in json.load(open(f,encoding="utf-8"))]); print(e11.model.unique())
e=pd.DataFrame([json.load(open(f,encoding="utf-8")) for f in glob.glob("results/e5d/*.json")]); print([c for c in e.columns if "newpart" in c])
print(pd.DataFrame([r for f in glob.glob("results/e17/*.json") for r in json.load(open(f))]).columns.tolist())
