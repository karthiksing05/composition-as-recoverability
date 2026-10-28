"""Figure: trained networks on glyph composites (E2, v2 evaluation; E6 fixed-pairs control)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *
import e2_glyph as E
from agg_r2 import load, e2cols, within_corr

df = e2cols(load("e2v2"))
occ = e2cols(load("e2v2occ"))
fp = e2cols(load("e6pairs"))
# readouts with the penalty chosen by leaving training glyphs out (e2h, the same systems retrained)
eh = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e2h/*.json")])
for s_ in ("vocab", "pool"):
    eh[f"{s_}_g"] = eh[[f"{s_}_glyph_role1", f"{s_}_glyph_role2"]].mean(1)
ARCH = {"slot": ("role-filler decoder", BLUE, "o"), "mlp": ("MLP", ORANGE, "s"),
        "cnn": ("CNN", GREEN, "^"), "tf": ("transformer", PURPLE, "D")}

fig = plt.figure(figsize=(6.5, 4.5))
gs = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.72)
axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

# a: the domain
ax = axs[0]
w = E.World(0)
rng = np.random.default_rng(3)
za, zb = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
imgs = [w.render(za, zb, 0)[0]]
E.SECOND["rel"] = "over"; imgs.append(w.render(za, zb, 1)[0])
E.SECOND["rel"] = "occ"; imgs.append(w.render(za, zb, 1)[0])
canvas = np.full((3 * 8 + 2 * 2, 16), np.nan)
for i, im in enumerate(imgs):
    canvas[i * 10:i * 10 + 8] = im / im.max()
ax.imshow(canvas, cmap="Greys", interpolation="nearest", aspect="equal")
for i, t in enumerate(["BESIDE", "OVER", "OCCLUDE"]):
    ax.text(-0.8, i * 10 + 3.5, t, ha="right", va="center", fontsize=6.5, color=DGREY)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
panel(ax, "a", "Composites")


def curve(ax, d, key, xkey, ls="-", marker=True, archs=ARCH):
    for a in archs:
        name, c, mk = ARCH[a]
        t = d[d.arch == a].groupby(xkey)[key].agg(["mean", "sem"])
        if len(t):
            band(ax, t.index, t["mean"].clip(lower=-0.1), t["sem"].fillna(0), c, ls=ls, marker=mk if marker else None, ms=3)


def log2axis(ax, ticks):
    ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


# b: behaviour vs number of training parts (m = 64)
ax = axs[1]
d = df[df.m == 64]
curve(ax, d, "beh_heldout_1", "V", ls="--", marker=False)
curve(ax, d, "beh_fresh_1", "V")
log2axis(ax, [8, 16, 32, 64])
ax.set_ylim(-0.02, 1.03); ax.set_xlabel("training parts $V$"); ax.set_ylabel("output $R^2$")
panel(ax, "b", "Behavior")

# c: what the state holds (m = 64): population readout, untrained network, vocabulary readout
ax = axs[2]
curve(ax, d, "pool_fresh_1", "V")
curve(ax, eh[eh.m == 64], "vocab_g", "V", ls="--", marker=False)
for a in ARCH:  # untrained networks, one dotted line per architecture
    t = d[d.arch == a].groupby("V")["init_fresh_1"].mean()
    ax.plot(t.index, t.values, color=ARCH[a][1], ls=":", lw=0.9)
ax.text(64, 0.47, "untrained (dotted)", fontsize=6.0, color=DGREY, ha="right")
refs = pd.DataFrame(json.load(open("results/e2refs/refs.json", encoding="utf-8")))
ax.axhline(refs.input_pool_over_r1.mean(), color="k", lw=0.8, ls=(0, (4, 2)))
ax.text(8.2, refs.input_pool_over_r1.mean() + 0.025, "input", fontsize=6.0, color="k")
log2axis(ax, [8, 16, 32, 64])
ax.set_ylim(-0.02, 1.03); ax.set_xlabel("training parts $V$"); ax.set_ylabel("readout $R^2$, fresh parts")
panel(ax, "c", "State")

# d: the network's own decoder behaves like a readout fitted on its vocabulary
ax = axs[3]
cl = lambda x: x.clip(lower=-0.2)
for a, (name, c, mk) in ARCH.items():
    s = eh[eh.arch == a]
    ax.scatter(cl(s.pool_g), cl(s.beh_fresh_r11), s=4, color=LGREY, marker=mk, lw=0, alpha=0.6, zorder=0)
    ax.scatter(cl(s.vocab_g), cl(s.beh_fresh_r11), s=5, color=c, marker=mk, lw=0, alpha=0.7, label=name)
ax.plot([-0.2, 1], [-0.2, 1], color=LGREY, lw=0.6, ls=":")
# correlations on unclipped values, as in the text
wv, r_v, _ = within_corr(eh, "vocab_g", "beh_fresh_r11", ["arch", "V", "m"])
wp, r_p, _ = within_corr(eh, "pool_g", "beh_fresh_r11", ["arch", "V", "m"])
ax.text(0.03, 0.97, f"vocabulary: $r$={r_v:.2f} ({wv:.2f} within)\npopulation (grey): $r$={r_p:.2f} ({wp:.2f} within)",
        transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=DGREY,
        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.set_xlim(-0.2, 1); ax.set_ylim(-0.2, 1.03)
ax.set_xlabel("readout $R^2$, fresh parts"); ax.set_ylabel("output $R^2$, fresh parts")
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), markerscale=2.2, fontsize=7)
panel(ax, "d", "Decoder vs. readouts")

# e: parts, not pairs: 140 training pairs drawn from V parts, against the full design
ax = axs[4]
for a in ARCH:
    name, c, mk = ARCH[a]
    t = fp[fp.arch == a].groupby("V")["beh_fresh_1"].agg(["mean", "sem"])
    band(ax, t.index, t["mean"], t["sem"].fillna(0), c, marker=mk, ms=3)
    t = fp[fp.arch == a].groupby("V")["beh_heldout_1"].mean()
    ax.plot(t.index, t.values, color=c, ls="--", lw=0.9)
log2axis(ax, [16, 32, 64])
ax.set_ylim(-0.02, 1.03); ax.set_xlabel("training parts $V$ (140 pairs)"); ax.set_ylabel("output $R^2$")
ax.minorticks_off()
panel(ax, "e", "Parts, not pairs")

# f: width and interaction: population readout vs bottleneck, with the Eckart-Young ceiling (V = 64)
ax = axs[5]
d = df[df.V == 64]; o = occ[occ.V == 64]
curve(ax, d, "pool_fresh_1", "m")
curve(ax, o, "pool_fresh_1", "m", ls="--", marker=False)
t = d.groupby("m")["ey_ceiling"].mean()
ax.plot(t.index, t.values, color="k", ls=":", lw=1.0)
ax.text(64, t.values[-1] + 0.02, "ceiling", fontsize=6.0, color="k", ha="right", va="bottom")
log2axis(ax, [8, 16, 32, 64])
ax.set_ylim(-0.02, 1.03); ax.set_xlabel("bottleneck width $m$"); ax.set_ylabel("readout $R^2$, fresh parts")
panel(ax, "f", "Width and occlusion")

fig.savefig(f"{FIGDIR}/fig_networks.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_networks.png", dpi=200)
print("n runs", len(df), len(occ), len(fp), len(eh), "r_vocab", round(r_v, 3), round(wv, 3), "r_pool", round(r_p, 3), round(wp, 3))
