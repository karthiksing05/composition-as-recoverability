"""Figure 2: the spectrum of demands on decomposition, with frameworks and this paper's systems in each set.
Colors follow the paper's rule: red = blend, orange = weak composition (own procedure), blues = strong composition."""
from figstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NOTES = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs"
NAVY = "#1b2a49"
RUNGS = [
    ("Blend", LOST, ["No procedure", "returns the parts"],
     ["Conceptual blending", "Composable diffusion"],
     ["x+y: R² 0.50", "HRR pairs: R² 0.44"]),
    ("Weak", BEH, ["The system acts on", "new wholes"],
     ["Frege, Montague", "SCAN, COGS"],
     ["Glyph decoders, V=8:", "new pairs 0.85–1.00", "LM answers: 0.92–0.99"]),
    ("Strong", REP, ["A probe fixed in", "advance reads new parts"],
     ["Reverse compositionality", "Context-dependent"],
     ["Glyph states: R² .83–.86", "Fixed places: κ ≥ 0.81", "Tree roots: R² 0.99"]),
    ("Shared role codes", REP2, ["One vector per part,", "changed only by its role"],
     ["Tensor products", "Additive decoders"],
     ["Glyph states, OVER:", "δ 0.02–0.04"]),
    ("Explicit copies", NAVY, ["The part itself is", "copied into the whole"],
     ["Classical architecture"],
     ["Concatenation", "reference: κ = 1"]),
]


def tint(hexcol, a):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(255 - a * (255 - c)) for c in (r, g, b))


H = 1.58
fig = plt.figure(figsize=(5.4, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 5.4); ax.set_ylim(0, H); ax.axis("off")
W, G, X0 = 1.04, 0.045, 0.01
TOP, BOT = 1.38, 0.27
FS = 6.0
xs = [X0 + i * (W + G) for i in range(5)]
rend = fig.canvas.get_renderer()
texts = []


def T(x, y, s, **kw):
    t = ax.text(x, y, s, va="center", **kw)
    texts.append((t, x))
    return t


for x, (name, col, demand, fw, ours) in zip(xs, RUNGS):
    ax.add_patch(FancyBboxPatch((x, BOT), W, TOP - BOT, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=tint(col, 0.09), ec=col, lw=0.9))
    tx = x + 0.05
    T(tx, TOP - 0.09, name, fontsize=7.0, fontweight="bold", color=col)
    for j, d in enumerate(demand):
        T(tx, TOP - 0.20 - 0.095 * j, d, fontsize=FS, color=INK)
    T(tx, TOP - 0.42, "Frameworks", fontsize=FS, fontweight="bold", color=DGREY)
    for j, f in enumerate(fw):
        T(tx, TOP - 0.515 - 0.095 * j, f, fontsize=FS, color=INK)
    T(tx, TOP - 0.75, "In this paper", fontsize=FS, fontweight="bold", color=DGREY)
    for j, s in enumerate(ours):
        T(tx, TOP - 0.845 - 0.095 * j, s, fontsize=FS, color=col if col != NAVY else INK)


def bracket(x0, x1, y, label, col):
    ax.plot([x0, x0, x1, x1], [y - 0.05, y, y, y - 0.05], color=col, lw=0.8)
    T((x0 + x1) / 2, y + 0.06, label, fontsize=FS, color=col, ha="center")


bracket(xs[0] + 0.03, xs[0] + W - 0.03, TOP + 0.08, "composition that loses parts", LOST)
bracket(xs[1] + 0.03, xs[1] + W - 0.03, TOP + 0.08, "composition", BEH)
bracket(xs[2] + 0.03, xs[4] + W - 0.03, TOP + 0.08, "composition and decomposition", REP)
ax.add_patch(FancyArrowPatch((xs[1] + 0.03, 0.17), (xs[4] + W - 0.03, 0.17), arrowstyle="-|>",
                             mutation_scale=7, color=DGREY, lw=0.8))
T((xs[1] + xs[4] + W) / 2, 0.06, "each set fixes more of the decomposition in advance",
  fontsize=FS, color=DGREY, ha="center")
fig.canvas.draw()
for t, x in texts:
    bb = t.get_window_extent(rend).transformed(ax.transData.inverted())
    if t.get_ha() == "left" and bb.x1 > x - 0.05 + W - 0.02:
        print("OVERFLOW", t.get_text(), round(bb.x1 - (x - 0.05 + W), 3))
fig.savefig(f"{FIGDIR}/fig_spectrum.pdf", bbox_inches=None, pad_inches=0)
fig.savefig(f"{NOTES}/fig_spectrum.png", dpi=170, bbox_inches=None, pad_inches=0)
print("saved")
