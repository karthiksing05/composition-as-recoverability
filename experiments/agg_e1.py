import json, glob, pandas as pd
rows = []
for f in glob.glob("results/e1/*.json"):
    rows += json.load(open(f))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
lin = df[df.readout != "nn_table"].groupby(["op", "readout", "V"])[["fam_r2", "fresh_r2", "fam_id", "fresh_id", "fam_role", "fresh_role"]].mean().round(2)
print(lin.reset_index().query("V in [8,32,128]").to_string())
nn = df[df.readout == "nn_table"].groupby(["op", "V"])[["fam_id", "fresh_id"]].mean().round(2)
print(nn.reset_index().query("V in [8,32,128]").to_string())
