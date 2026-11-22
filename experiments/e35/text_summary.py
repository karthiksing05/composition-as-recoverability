import json, sys, pandas as pd
pd.set_option("display.width", 250)
R = "/coc/flash10/ksingara3/compositionality-theory/results/e31"
for enc in ("pixart-t5", "lumina-gemma"):
    d = pd.DataFrame(json.load(open(f"{R}/d2a_text_{enc}.json", encoding="utf-8")))
    g = d.groupby(["layer", "rel"])[["val_r2", "new_canon_share_nearer_figure", "new_conv_share_nearer_figure", "new_canon_id_figure", "new_conv_id_figure", "new_conv_id_ground"]].mean()
    print("===", enc); print(g.round(3).to_string())
    print(d.groupby("layer")[["new_canon_share_nearer_figure", "new_conv_share_nearer_figure"]].agg(["mean", "min", "max"]).round(3).to_string())
