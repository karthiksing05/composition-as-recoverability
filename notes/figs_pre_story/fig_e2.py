"""Figure: trained networks on glyph composites (E2, v2 evaluation; fixed-pairs control; case study)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *
from agg_r2 import load, e2cols, within_corr

df = e2cols(load("e2v2"))
occ = e2cols(load("e2v2occ"))
fp = e2cols(load("e6pairs"))
eh = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e2h/*.json")])
for s_ in ("vocab", "pool"):
    eh[f"{s_}_g"] = eh[[f"{s_}_glyph_role1", f"{s_}_glyph_role2"]].mean(1)
case = np.load("results/case/glyph.npz")
THREE = ["slot", "mlp", "cnn"]  # the transformer is reported in the text
DEC, POP, VOC = INK, BLUE, GREEN

fig = plt.figure(figsize=(6.5, 3.75))
gs = fig.add_gridspec(2, 4, height_ratios=[0.8, 1.1], hspace=0.4, wspace=0.62)

# a: case study
ax = fig.add_subplot(gs[0, :])
ax.axis("off")
panel(ax, "a", "Case study: the state holds fresh glyphs that a decoder trained on 8 glyphs cannot render (MLP, $m = 64$)")
ROWS = [("target", "target"), ("net", "network's output"), ("readout", "read from its state")]
NK, TW, TH = 4, 0.092, 0.29
for b, V in enumerate((8, 64)):
    x0 = 0.15 + b * 0.45
    tgt = case[f"V{V}_target"][:NK]
    var = ((tgt - tgt.mean(0)) ** 2).sum((1, 2)).mean()
    for r, (key, lab) in enumerate(ROWS):
        im = case[f"V{V}_{key}"]
        yb = 0.58 - r * 0.32
        for k in range(NK):
            iax = ax.inset_axes([x0 + k * (TW + 0.006), yb, TW, TH])
            iax.imshow(im[k], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            iax.set_xticks([]); iax.set_yticks([])
            col = {"net": DEC, "readout": POP}.get(key, LGREY)
            for sp in iax.spines.values():
                sp.set_color(col); sp.set_linewidth(1.2 if key != "target" else 0.6)
            if key != "target":
                r2 = 1 - ((im[k] - tgt[k]) ** 2).sum() / var
                iax.text(0.97, 0.05, f"{r2:.2f}", transform=iax.transAxes, ha="right", va="bottom", fontsize=5.4, color=col,
                         bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
        if b == 0:
            ax.text(x0 - 0.012, yb + TH / 2, lab, ha="right", va="center", fontsize=6.6,
                    color={"net": DEC, "readout": POP}.get(key, DGREY), transform=ax.transAxes)
    ax.text(x0 + 2 * (TW + 0.006), 0.9, f"trained on $V = {V}$ glyphs", ha="center", va="bottom", fontsize=7, transform=ax.transAxes)

def mean3(d, key, x):
    t = d[d.arch.isin(THREE)].groupby(["arch", x])[key].mean().unstack(0)
    return t.index.values, t.mean(1).values, t.std(1).values / np.sqrt(3)

def log2axis(ax, ticks):
    ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()

def lab(ax, x, y, s, c, dy=0.0):
    ax.text(x * 1.08, y + dy, s, fontsize=6.0, color=c, ha="left", va="center")

# b: decoder against state
ax = fig.add_subplot(gs[1, 0]); ygrid(ax)
d = df[df.m == 64]
for key, c, ls, s in (("pool_fresh_1", POP, "-", "population"), ("init_fresh_1", POP, ":", "untrained"),
                      ("beh_fresh_1", DEC, "-", "network")):
    x, m_, e_ = mean3(d, key, "V")
    band(ax, x, m_, e_, c, ls=ls, marker="o" if ls == "-" else None, ms=2.6, label=s)
x, m_, e_ = mean3(eh[eh.m == 64], "vocab_g", "V")
band(ax, x, m_, e_, VOC, ls="--", label="vocabulary")
# the encoder trained on 8 glyphs, frozen, with a new decoder trained on 64 other glyphs
rd = pd.DataFrame([dict(arch=r["arch"], **{c: np.mean([r[c]["r10"], r[c]["r11"]]) for c in ("orig", "redec_trained", "redec_init")})
                   for r in (json.load(open(f, encoding="utf-8")) for f in glob.glob("results/e2redec/*.json"))])
rd = rd[rd.arch.isin(THREE)].groupby("arch").mean().mean()
ax.annotate("", xy=(8, rd.redec_trained), xytext=(8, rd.orig + 0.04), arrowprops=dict(arrowstyle="-|>", color=DEC, lw=0.7, mutation_scale=6))
ax.plot([8], [rd.redec_trained], "s", color=DEC, mfc="white", ms=3.2, zorder=5)
ax.text(9.3, rd.redec_trained - 0.035, "new decoder", fontsize=5.4, color=DEC, ha="left", va="center")
ax.legend(loc="lower right", fontsize=5.4, handlelength=1.4, borderaxespad=0.1, labelspacing=0.25)
log2axis(ax, [8, 16, 32, 64]); ax.set_xlim(7, 64)
ax.set_ylim(0, 1.0); ax.set_xlabel("training glyphs $V$"); ax.set_ylabel("$R^2$ on fresh glyphs")
panel(ax, "b", "State versus decoder")

# c: the decoder behaves like a vocabulary readout
ax = fig.add_subplot(gs[1, 1]); ygrid(ax)
cl = lambda v: v.clip(lower=-0.2)
s = eh
ax.scatter(cl(s.pool_g), cl(s.beh_fresh_r11), s=3, color=POP, lw=0, alpha=0.35, zorder=1)
ax.scatter(cl(s.vocab_g), cl(s.beh_fresh_r11), s=3, color=VOC, lw=0, alpha=0.55, zorder=2)
ax.plot([-0.2, 1], [-0.2, 1], color=LGREY, lw=0.6, ls=":")
wv, r_v, _ = within_corr(eh, "vocab_g", "beh_fresh_r11", ["arch", "V", "m"])
wp, r_p, _ = within_corr(eh, "pool_g", "beh_fresh_r11", ["arch", "V", "m"])
ax.text(0.03, 0.99, f"vocab. $r$ = {wv:.2f}", transform=ax.transAxes, fontsize=5.6, color=VOC, va="top")
ax.text(0.03, 0.905, f"popul. $r$ = {wp:.2f}", transform=ax.transAxes, fontsize=5.6, color=POP, va="top")
ax.text(0.03, 0.82, "within conditions", transform=ax.transAxes, fontsize=5.6, color=DGREY, va="top")
ax.set_xlim(-0.2, 1); ax.set_ylim(-0.2, 1.22); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("readout $R^2$"); ax.set_ylabel("network's output $R^2$")
panel(ax, "c", "Decoder tracks probe")

# d: parts, not pairs
ax = fig.add_subplot(gs[1, 2]); ygrid(ax)
for key, ls, s in (("beh_heldout_1", "--", "held-out pairs"), ("beh_fresh_1", "-", "fresh glyphs")):
    x, m_, e_ = mean3(fp, key, "V")
    band(ax, x, m_, e_, DEC, ls=ls, marker="o" if ls == "-" else None, ms=2.6, label=s)
ax.legend(loc="lower right", fontsize=5.5, handlelength=1.5, borderaxespad=0.1)
log2axis(ax, [16, 32, 64]); ax.set_xlim(14, 64)
ax.set_ylim(0, 1.03); ax.set_xlabel("distinct glyphs (140 pairs)"); ax.set_ylabel("network's output $R^2$")
panel(ax, "d", "Parts, not pairs")

# e: width and occlusion (V = 64)
ax = fig.add_subplot(gs[1, 3]); ygrid(ax)
t = df[df.V == 64].groupby("m")["ey_ceiling"].mean()
ax.plot(t.index, t.values, color=INK, ls=":", lw=1.0, label="ceiling (Prop. 4)")
for dd_, ls, s in ((df[df.V == 64], "-", "over"), (occ[occ.V == 64], "--", "occlude")):
    x, m_, e_ = mean3(dd_, "pool_fresh_1", "m")
    band(ax, x, m_, e_, POP, ls=ls, marker="o" if ls == "-" else None, ms=2.6, label=s)
ax.legend(loc="lower right", fontsize=5.5, handlelength=1.5, borderaxespad=0.1)
log2axis(ax, [8, 16, 32, 64]); ax.set_xlim(7, 64)
ax.set_ylim(0.3, 1.05); ax.set_xlabel("state width $m$"); ax.set_ylabel("state $R^2$, fresh glyphs")
panel(ax, "e", "Width, occlusion")

fig.savefig(f"{FIGDIR}/fig_networks.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_networks.png", dpi=220)
print("r_vocab", round(r_v, 3), round(wv, 3), "r_pool", round(r_p, 3), round(wp, 3))
print("redecode", rd.round(3).to_dict())
for V in (8, 64):
    t, n_, r_ = case[f"V{V}_target"], case[f"V{V}_net"], case[f"V{V}_readout"]
    r2 = lambda p: 1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()
    print(V, "net", round(float(r2(n_)), 3), "readout", round(float(r2(r_)), 3))
