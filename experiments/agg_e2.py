import json, glob, sys, pandas as pd, numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "results/e2"
df = pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{d}/*.json")])
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
for r in ("0", "1"):
    df[f"beh_ho_{r}"] = df[[f"beh_heldout_r{r}{o}" for o in "01"]].mean(1)
    df[f"beh_st_{r}"] = df[[f"beh_strong_r{r}{o}" for o in "01"]].mean(1)
    df[f"beh_fr_{r}"] = df[[f"beh_fresh_r{r}{o}" for o in "01"]].mean(1)
    df[f"beh_tr_{r}"] = df[[f"beh_train_r{r}{o}" for o in "01"]].mean(1)
    df[f"pix_ho_{r}"] = df[[f"probe_pix_heldout_r{r}_role{k}" for k in "12"]].mean(1)
    df[f"pix_st_{r}"] = df[f"probe_pix_strong_r{r}_role2"]
    df[f"pix_fr_{r}"] = df[[f"probe_pix_fresh_r{r}_role{k}" for k in "12"]].mean(1)
cols = ["beh_tr_1","beh_ho_1","beh_st_1","beh_fr_1","pix_ho_1","pix_st_1","pix_fr_1","inter_fam_r1","inter_fresh_r1","rolefac_fam_r1","leak_heldout_r1","leak_fresh_r1",
        "beh_ho_0","beh_st_0","beh_fr_0","pix_fr_0","inter_fam_r0"]
g = df.groupby(["arch","m","V"])[cols]
print(g.mean().round(2).to_string())
print("n per cell", g.size().min(), g.size().max(), "total", len(df))
print(df.groupby(["arch","V"])[["beh_st_1","beh_fr_1"]].std().round(3).to_string())
