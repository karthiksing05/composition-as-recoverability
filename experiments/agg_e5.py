"""Aggregate E5b (structural consequences, regularized unbinding) into summary tables."""
import json, glob
import numpy as np, pandas as pd
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60)
df = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e5b/*.json")])
r = "1"
for k in ("pred", "edit", "true", "noswap"):
    df[f"rev_out_{k}"] = df[[f"rev_out_{k}_r{r}{o}" for o in "01"]].mean(1)
cols = ["code_var", f"delta_fresh_r{r}", f"readout_fresh_r{r}", f"rev_state_r{r}", f"rev_state_noswap_r{r}", "rev_out_pred", "rev_out_true", "rev_out_noswap",
        f"newrole_state_r{r}", f"newrole_out_pred_r{r}", f"newrole_out_net_r{r}",
        f"swap_left_r{r}", f"swap_left_base_r{r}", f"swap_rand_left_r{r}", f"swap_right_r{r}", f"swap_right_base_r{r}", f"swap_right_change_r{r}",
        "newpart_role1_before", "newpart_role1_replay0", "newpart_role1_replay1", "newpart_role2_before", "newpart_role2_replay0", "newpart_role2_replay1",
        "newpart_heldout_before", "newpart_heldout_replay0", "newpart_heldout_replay1"]
for sec in ("over", "occ"):
    d = df[df.second == sec]
    g = d.groupby(["arch", "V", "m"])[cols]
    print(f"=== {sec} n={g.size().min()}-{g.size().max()}")
    print(g.mean().round(3).T.to_string())
    print(d.groupby("V")[cols].mean().round(3).T.to_string())
