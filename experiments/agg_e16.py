"""Aggregate E16: answer accuracy and role-question transfer under other prompts and the ground task."""
import json, glob
import pandas as pd

rows = []
for f in glob.glob("results/e16/*.json"):
    rows += json.load(open(f, encoding="utf-8"))
d = pd.DataFrame(rows)
for pre in ("EleutherAI/", "allenai/", "microsoft/", "Qwen/"):
    d["model"] = d.model.str.replace(pre, "")
# The layer and code of E11's query state for each model (selected there on validation nouns with
# both phrasings).  E16 fits on canonical sentences only, and selecting on canonical validation
# favours early layers that bind the figure to word order, so the E11 choice is held fixed.
E11 = {"phi-4": ("L30", "phrase"), "Qwen3-8B": ("L27", "carrier"), "OLMo-2-1124-13B": ("L30", "phrase"),
       "OLMo-2-1124-7B": ("L16", "carrier"), "gpt2-large": ("L18", "carrier"), "pythia-1.4b": ("L12", "carrier"),
       "pythia-12b": ("L18", "carrier")}
out = []
for (m, v), g in d.groupby(["model", "variant"]):
    lc = E11[m]
    gg = g[(g.layer == lc[0]) & (g.code == lc[1])]
    out.append(dict(model=m, variant=v, layer=lc[0], code=lc[1], seeds=gg.seed.nunique(),
                    answer=gg.choice.mean(), answer_can=gg.choice_can.mean(), answer_conv=gg.choice_conv.mean(),
                    xfer_can=gg.xfer_role_can.mean(), xfer_conv=gg.xfer_role_conv.mean()))
pd.set_option("display.width", 200)
print(pd.DataFrame(out).set_index(["model", "variant"]).round(3).to_string())
