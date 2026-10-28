"""Figure: depth (E3) and pretrained text encoders (E4)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *

rows = []
for f in glob.glob("results/e3/*.json"):
    rows += json.load(open(f))
e3 = pd.DataFrame(rows)
raam = e3[e3.model == "raam"]
hrr = pd.DataFrame(json.load(open("results/e3hrr/hrr_unitary.json")))
e4 = pd.DataFrame([r for f in glob.glob("results/e4/*.json") for r in json.load(open(f))])
for k in ("rec_train", "rec_fresh", "id_fresh", "role_fresh"):
    e4[k] = e4[[f"{k}_role1", f"{k}_role2"]].mean(1)

fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.6), gridspec_kw=dict(wspace=0.32, hspace=0.45)); axs = axs.ravel()

# a: fresh-leaf recovery by depth for the learned recursive composer
ax = axs[0]
ax.axvspan(0.6, 2.4, color=LGREY, alpha=0.35, lw=0)
for w, c in ((32, ORANGE), (64, GREY), (256, BLUE)):
    t = raam[(raam.V == 16) & (raam.w == w)].groupby("depth")["fresh_r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"].clip(lower=-0.5), t["sem"], c, marker="o", ms=3, label=f"$w$={w}")
    b = raam[(raam.V == 16) & (raam.w == w)].groupby("depth")["bound"].mean()
    ax.plot(b.index, b.values, color=c, ls=":", lw=0.8)
ax.set_ylim(-0.5, 1.05); ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("depth $n$"); ax.set_ylabel("recovery of fresh leaves ($R^2$)")
ax.text(1.5, -0.42, "trained", ha="center", fontsize=6.3, color=DGREY)
ax.legend(loc="lower left", fontsize=6.0, handlelength=1.2, bbox_to_anchor=(0, 0.08))
panel(ax, "a", "Learned recursion")

# b: identification of familiar leaves by depth
ax = axs[1]
ax.axvspan(0.6, 2.4, color=LGREY, alpha=0.35, lw=0)
t = raam[(raam.V == 16) & (raam.w == 256)].groupby("depth")["fam_id"].agg(["mean", "sem"])
band(ax, t.index, t["mean"], t["sem"], BLUE, marker="o", ms=3, label="learned, $w$=256")
for w, ls in ((256, "-"), (1024, "--")):
    t = hrr[(hrr.V == 16) & (hrr.w == w) & (hrr.split == "fam")].groupby("depth")["id"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"], t["sem"], GREY, ls=ls, marker="s", ms=3, label=f"HRR, $w$={w}")
ax.axhline(1 / 16, color=LGREY, lw=0.6, ls=":")
ax.set_ylim(0, 1.03); ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("depth $n$"); ax.set_ylabel("identification (chance 1/16)")
ax.legend(loc="lower left", fontsize=6.0, handlelength=1.2, bbox_to_anchor=(0, 0.08))
panel(ax, "b", "Familiar leaves")

# c: pretrained encoders, best layer per model, against references built from their own part codes
ax = axs[2]
NAMES = {"clip-b32": "CLIP-B", "sd15-clip-l": "CLIP-L", "t5-small": "T5-s", "t5-base": "T5-b",
         "bert-base-uncased": "BERT", "gpt2": "GPT-2"}
models = [m for m in NAMES if m in set(e4.model)]
mod = e4[e4.system == "model"]
best = mod.groupby(["model", "layer"])["rec_fresh"].mean().reset_index().sort_values("rec_fresh").groupby("model").tail(1)
best = dict(zip(best.model, best.layer))
x = np.arange(len(models))
summ = {}
for i, m in enumerate(models):
    s = e4[(e4.model == m) & (e4.layer == best[m])]
    g = s.groupby("system")[["rec_train", "rec_fresh", "id_fresh", "role_fresh", "rev_sim", "inter"]].mean()
    summ[m] = g
    ax.bar(i - 0.18, g.loc["model", "rec_train"], 0.34, color=LGREY)
    ax.bar(i + 0.18, max(g.loc["model", "rec_fresh"], -1), 0.34, color=ORANGE)
    ax.scatter([i + 0.18], [g.loc["ref_concat", "rec_fresh"]], marker="_", s=70, color=BLUE, lw=1.6, zorder=3)
    ax.scatter([i + 0.18], [g.loc["ref_sum", "rec_fresh"]], marker="_", s=70, color=DGREY, lw=1.6, zorder=3)
ax.axhline(0, color=DGREY, lw=0.4)
ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in models], rotation=0, fontsize=6.0)
ax.set_ylim(-1.05, 1.05); ax.set_ylabel("recovery of nouns ($R^2$)")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend([Patch(color=LGREY), Patch(color=ORANGE), Line2D([], [], color=BLUE, lw=1.6), Line2D([], [], color=DGREY, lw=1.6)],
          ["training nouns", "fresh nouns", "concat. reference", "sum reference"], fontsize=5.8, loc="upper center",
          ncol=2, handlelength=1, columnspacing=0.6)
panel(ax, "c", "Text encoders")

# d: roles survive, identities partly
ax = axs[3]
for i, m in enumerate(models):
    g = summ[m]
    ax.scatter(g.loc["model", "role_fresh"], g.loc["model", "id_fresh"], color=ORANGE, s=14, zorder=3)
    off = {"gpt2": (-22, 4), "t5-small": (4, 4), "t5-base": (-22, -2), "bert-base-uncased": (4, 2),
           "clip-b32": (4, -6), "sd15-clip-l": (4, 0)}.get(m, (3, 2))
    ax.annotate(NAMES[m], (g.loc["model", "role_fresh"], g.loc["model", "id_fresh"]), off,
                textcoords="offset points", fontsize=6.2, color=DGREY)
gc = pd.concat(summ.values())
for sysname, c, lab in (("ref_concat", BLUE, "concat."), ("ref_sum", DGREY, "sum")):
    r = gc.loc[sysname]
    ax.scatter(r["role_fresh"].mean(), r["id_fresh"].mean(), color=c, marker="s", s=14, zorder=3)
    ax.annotate(lab, (r["role_fresh"].mean(), r["id_fresh"].mean()), (3, -7), textcoords="offset points", fontsize=5.8, color=c)
ax.set_xlim(0.45, 1.03); ax.set_ylim(0, 1.03)
ax.set_xlabel("role assigned correctly"); ax.set_ylabel("noun identified (chance 1/80)")
panel(ax, "d", "Fresh nouns")

fig.savefig(f"{FIGDIR}/fig_depth_pretrained.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_depth_pretrained.png", dpi=200)
pd.set_option("display.width", 250)
print("best layers", best)
for m in models:
    print(m, summ[m].round(3).to_string(), sep="\n")
