import json
s = json.load(open("results/e42/summary.json", encoding="utf-8"))
for k in ["table", "random_weight", "absx_controls"]:
    for r in s[k]:
        if r["layer_tag"] in ("sel",) or k == "absx_controls":
            print(k[:6], r["model"][:14], r["cond"], r["readout"], r["layer_tag"], r["probe"], round(r["kappa"], 2), round(r["kappa_linref"], 2), round(r["role_acc"], 2))
e = json.load(open("results/e41/summary.json", encoding="utf-8"))
for m, d in e.items():
    for p in ("canonical", "converse"):
        x = d[p]; print("E41", m[-10:], p, x["n_wrong"], round(x["wrong_probe_follows_answer"], 3), round(x["wrong_probe_returns_figure"], 3), round(x["right_probe_returns_figure"], 3))
