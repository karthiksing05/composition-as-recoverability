"""Figure: depth (E3) and pretrained text encoders (E4)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *

rows = []
for f in glob.glob("results/e3v2/*.json"):
    rows += json.load(open(f))
e3 = pd.DataFrame(rows)
raam = e3[e3.model == "raam"]
hrr = pd.DataFrame(json.load(open("results/e3hrr3/hrr_unitary.json")))  # one pair of unitary roles per height
hrr = hrr[hrr.split == "pool"]
# whitened part codes (each of the 16 components counts equally); see agg_r4.py e4w
e4all = pd.DataFrame([r for f in glob.glob("results/e4w/*.json") for r in json.load(open(f))])
e4all = e4all[(e4all.n_fit == 165) & (e4all.rel != "and")]  # the symmetric control is reported in the text
for k in ("rec_pairs", "rec_val", "rec_fresh", "id_fresh", "role_fresh"):
    e4all[k] = e4all[[f"{k}_role1", f"{k}_role2"]].mean(1)
e4 = e4all[~e4all.model.str.startswith("rand-")]


def rand_select(rnd, k="rec_fresh"):  # layer and code of a random-weight control, chosen on its validation nouns
    rsel = rnd.groupby(["model", "layer", "code"])["rec_val"].mean().reset_index()
    rsel = rsel.loc[rsel.groupby("model").rec_val.idxmax()]
    return {r.model[5:]: rnd[(rnd.model == r.model) & (rnd.layer == r.layer) & (rnd.code == r.code)][k].mean()
            for r in rsel.itertuples()}


rnd = e4all[e4all.model.str.startswith("rand-") & (e4all.system == "model")]
rand_fresh = rand_select(rnd[rnd.codes_from == rnd.model])  # scored on its own part codes
rand_trained = rand_select(rnd[rnd.codes_from != rnd.model])  # scored on the trained model's part codes
rand_role = rand_select(rnd[rnd.codes_from == rnd.model], "role_fresh")
rand_id = rand_select(rnd[rnd.codes_from == rnd.model], "id_fresh")

fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.9), gridspec_kw=dict(wspace=0.32, hspace=0.62)); axs = axs.ravel()

# a: fresh-leaf recovery by depth for the learned recursive composer
ax = axs[0]
ax.axvspan(0.6, 2.4, color=LGREY, alpha=0.35, lw=0)
for w, c in ((32, ORANGE), (64, GREY), (256, BLUE)):
    s = raam[(raam.V == 16) & (raam.w == w)]
    t = s.groupby("depth")["fresh_r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"].clip(lower=-0.5), t["sem"], c, marker="o", ms=3, label=f"$w$={w}")
    t = s.groupby("depth")["pop_r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"], t["sem"], c, ls="--")
    b = s.groupby("depth")["bound"].mean()
    ax.plot(b.index, b.values, color=c, ls=":", lw=0.8)
ax.set_ylim(-0.5, 1.05); ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("depth $n$"); ax.set_ylabel("recovery of fresh leaves ($R^2$)")
ax.text(1.5, -0.42, "trained", ha="center", fontsize=6.3, color=DGREY)
ax.legend(loc="lower left", fontsize=6.0, handlelength=1.2, bbox_to_anchor=(0, 0.08))
panel(ax, "a", "Learned recursion")

# b: HRR: correlation unbinding (solid) vs a population readout of the same state (dashed)
ax = axs[1]
for w, c in ((64, GREY), (256, BLUE), (1024, GREEN)):
    s = hrr[hrr.w == w]
    t = s.groupby("depth")["r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"].clip(lower=-0.5), t["sem"], c, marker="s", ms=3, label=f"$w$={w}")
    t = s.groupby("depth")["pop_r2"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"], t["sem"], c, ls="--")
    t = s.groupby("depth")["crosstalk_pred"].mean()
    ax.plot(t.index, t.clip(lower=-0.5).values, color=c, ls=":", lw=0.8)
ax.set_ylim(-0.5, 1.05); ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("depth $n$"); ax.set_ylabel("recovery of fresh leaves ($R^2$)")
ax.legend(loc="lower left", fontsize=6.0, handlelength=1.2, bbox_to_anchor=(0, 0.08))
panel(ax, "b", "HRR")

# c: pretrained encoders, best layer per model, against references built from their own part codes
ax = axs[2]
NAMES = {"clip-b32": "CLIP-B", "sd15-clip-l": "CLIP-L", "clip-h14": "CLIP-H", "t5-small": "T5-s", "t5-base": "T5-b",
         "t5-large": "T5-l", "bert-base-uncased": "BERT-b", "bert-large-uncased": "BERT-l", "gpt2": "GPT2-s",
         "gpt2-medium": "GPT2-m", "gpt2-large": "GPT2-l"}
models = [m for m in NAMES if m in set(e4.model)]
mod = e4[e4.system == "model"]
# layer and part code chosen on validation nouns, never on the fresh nouns that are reported
sel = mod.groupby(["model", "layer", "code"])["rec_val"].mean().reset_index()
sel = sel.loc[sel.groupby("model").rec_val.idxmax()]
best = {r.model: (r.layer, r.code) for r in sel.itertuples()}
x = np.arange(len(models))
summ = {}
for i, m in enumerate(models):
    s = e4[(e4.model == m) & (e4.layer == best[m][0]) & (e4.code == best[m][1])]
    g = s.groupby("system")[["rec_pairs", "rec_fresh", "id_fresh", "role_fresh", "rev_sim", "inter"]].mean()
    summ[m] = g
    ax.bar(i - 0.18, g.loc["model", "rec_pairs"], 0.34, color=LGREY)
    ax.bar(i + 0.18, max(g.loc["model", "rec_fresh"], -1), 0.34, color=ORANGE)
    ax.scatter([i + 0.18], [g.loc["ref_concat", "rec_fresh"]], marker="_", s=70, color=BLUE, lw=1.6, zorder=3)
    ax.scatter([i + 0.18], [g.loc["ref_sum", "rec_fresh"]], marker="_", s=70, color=DGREY, lw=1.6, zorder=3)
    if m in rand_fresh:
        ax.scatter([i + 0.18], [rand_fresh[m]], marker="x", s=14, color="k", lw=0.9, zorder=4)
    ax.scatter([i + 0.18], [max(rand_trained[m], 0.0)], marker="o", s=6, color="k", lw=0, zorder=4)
ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in models], rotation=60, fontsize=5.6)
ax.set_ylim(0, 1.5); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("recovery of nouns ($R^2$)")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend([Patch(color=LGREY), Patch(color=ORANGE), Line2D([], [], color=BLUE, lw=1.6), Line2D([], [], color=DGREY, lw=1.6),
           Line2D([], [], color="k", marker="x", lw=0, ms=4), Line2D([], [], color="k", marker="o", lw=0, ms=2.5)],
          ["new pairs", "fresh nouns", "concat. ref.", "sum ref.", "random, own codes",
           "random, trained codes"],
          fontsize=5.4, loc="upper center", ncol=3, handlelength=1, columnspacing=0.6)
panel(ax, "c", "Text encoders")

# d: roles survive, identities partly
ax = axs[3]
for i, m in enumerate(models):
    g = summ[m]
    ax.bar(i - 0.18, g.loc["model", "role_fresh"], 0.34, color=GREEN)
    ax.bar(i + 0.18, g.loc["model", "id_fresh"], 0.34, color=ORANGE)
    ax.scatter([i + 0.18], [g.loc["ref_sum", "id_fresh"]], marker="_", s=70, color=DGREY, lw=1.6, zorder=3)
    ax.scatter([i - 0.18, i + 0.18], [rand_role[m], rand_id[m]], marker="x", s=14, color="k", lw=0.9, zorder=4)
ax.axhline(0.5, color=GREEN, lw=0.6, ls=":")
ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in models], rotation=60, fontsize=5.6)
ax.set_ylim(0, 1.35); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("fraction correct, fresh nouns")
ax.legend([Patch(color=GREEN), Patch(color=ORANGE), Line2D([], [], color=DGREY, lw=1.6), Line2D([], [], color="k", marker="x", lw=0, ms=4)],
          ["role assigned", "identified (chance 0.01)", "sum ref., identified", "random, own codes"], fontsize=5.6,
          loc="upper center", ncol=2, handlelength=1, columnspacing=0.6)
panel(ax, "d", "Fresh nouns")

fig.savefig(f"{FIGDIR}/fig_depth_pretrained.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_depth_pretrained.png", dpi=200)
pd.set_option("display.width", 250)
print("best layers", best)
print("random, own codes", {k: round(v, 3) for k, v in rand_fresh.items()})
print("random, trained codes", {k: round(v, 3) for k, v in rand_trained.items()})
for m in models:
    print(m, summ[m].round(3).to_string(), sep="\n")
