"""Appendix glyph cases from results/case/glyph.npz (written by case_glyph.py): one MLP (m = 64, OVER input, seed 0)
trained on V = 8 and on V = 64 glyphs, rendering wholes of fresh glyphs (never in either training vocabulary).

Selection rule: every saved example, in file order (the file holds 5 wholes per V and 3 swap cases; nothing was
filtered). R^2 under an image = 1 - SSE(image, target) / var, with var the mean squared deviation of the targets of
that block from their per-pixel mean over the shown examples (as in fig_strong_only, but over 5 wholes, not 4, so
the first four values can differ slightly from that figure). The new-decoder-on-frozen-state condition was not saved,
so it is not shown. Panel b is the code swap at V = 64: the role-1 vector of the source state is replaced through
the fitted role maps by the donor's, against a same-norm edit in the natural direction (hd - h0)."""
import os
import numpy as np
from figstyle import *

ROOT = "/coc/flash10/ksingara3/compositionality-theory"
case = np.load(f"{ROOT}/experiments/results/case/glyph.npz")
W = 6.75


def r2s(im, tgt):
    var = ((tgt - tgt.mean(0)) ** 2).sum((1, 2)).mean()
    return 1 - ((im - tgt) ** 2).sum((1, 2)) / var


H = 4.97
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def img(x, y, w, h, a, col, lw=1.1):
    ia = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ia.imshow(a, cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values():
        sp.set_visible(True); sp.set_color(col); sp.set_linewidth(lw)


def head(y, letter, title):
    ax.text(0.04, y, letter, fontsize=9, fontweight="bold", va="center")
    ax.text(0.2, y, title, fontsize=7.6, va="center")


head(0.14, "a", "One MLP trained on 8 glyphs and one trained on 64: wholes of five new glyphs, rendered by the network and read from its state")
IW, IH, G = 0.56, 0.28, 0.025
ROWS = [("target", "target", LGREY), ("net", "own decoder", BEH), ("readout", "probe of\nthe state", REP)]
out = {}
for b, V in enumerate((8, 64)):
    x0 = 0.78 + b * 3.0
    ax.text(x0, 0.4, f"trained on $V$ = {V} glyphs", fontsize=7.2, fontweight="bold", va="center")
    tgt = case[f"V{V}_target"]
    for r, (key, lab, col) in enumerate(ROWS):
        y = 0.55 + r * 0.5
        if b == 0:
            ax.text(0.04, y + IH / 2, lab, fontsize=6.6, color=col if key != "target" else DGREY, va="center", linespacing=0.95)
        im = case[f"V{V}_{key}"]
        sc = r2s(im, tgt) if key != "target" else None
        for k in range(im.shape[0]):
            x = x0 + k * (IW + G)
            img(x, y, IW, IH, im[k], col, 0.6 if key == "target" else 1.2)
            if sc is not None:
                ax.text(x + IW / 2, y + IH + 0.075, f"{sc[k]:.2f}", fontsize=6.2, color=col, ha="center", va="center")
        if sc is not None:
            out[(V, key)] = np.round(sc, 2).tolist()
ax.text(0.04, 2.1, "$R^2$ under an image: against the target of that column", fontsize=5.8, color=DGREY, va="center", ha="left")

head(2.3, "b", "Code swap in the $V$ = 64 network: replacing the role-1 vector of the state with the donor's changes one glyph")
COLS = [("swap_donor", "donor glyph", LGREY, 0.5), ("swap_orig", "own decoder on\nthe source state", BEH, 1.0), ("swap_target", "target: donor\nin role 1, role 2 kept", LGREY, 1.0),
        ("swap_edit", "code swap", FIX, 1.0), ("swap_nat", "same-norm edit,\nnatural direction", GREY, 1.0)]
CW, CH, CG = 1.0, 0.46, 0.2
x0 = 0.35
tg = case["swap_target"]
sw = {k: r2s(case[k], tg) for k in ("swap_orig", "swap_edit", "swap_nat")}
for j, (key, lab, col, rel) in enumerate(COLS):
    x = x0 + j * (CW + CG)
    ax.text(x + CW / 2, 2.68, lab, fontsize=6.4, color=col if col != LGREY else DGREY, ha="center", va="center", linespacing=0.95)
    for i in range(3):
        y = 2.9 + i * 0.62
        im = case[key][i]
        if key == "swap_donor":
            img(x + CW / 4, y, CW / 2, CH, im, col, 0.6)
        else:
            img(x, y, CW, CH, im, col, 0.6 if key == "swap_target" else 1.2)
            if key in sw:
                ax.text(x + CW / 2, y + CH + 0.075, f"{sw[key][i]:.2f}", fontsize=6.4, color=col if col != GREY else DGREY, ha="center", va="center", fontweight="bold")
ax.text(0.04, 4.9, "$R^2$ under an image: against the target of that row", fontsize=5.8, color=DGREY, va="center", ha="left")
os.makedirs(f"{ROOT}/notes/audit/figs", exist_ok=True)
fig.savefig(f"{FIGDIR}/fig_case_glyph.pdf", bbox_inches=None, pad_inches=0)
fig.savefig(f"{ROOT}/notes/audit/figs/fig_case_glyph.png", dpi=170, bbox_inches=None, pad_inches=0)
for k, v in out.items(): print(k, v)
for k, v in sw.items(): print(k, np.round(v, 2).tolist())
