"""E16 per layer: transfer to converse sentences and validation recovery, best code per layer."""
import json, glob
import pandas as pd
d = pd.DataFrame([r for f in glob.glob("results/e16/*.json") for r in json.load(open(f, encoding="utf-8"))])
for pre in ("EleutherAI/", "allenai/", "microsoft/", "Qwen/"):
    d["model"] = d.model.str.replace(pre, "")
d["li"] = d.layer.str[1:].astype(int)
t = d.groupby(["model", "variant", "li", "code"])[["xfer_rec_val", "xfer_role_can", "xfer_role_conv"]].mean().round(3)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
print(t.to_string())
