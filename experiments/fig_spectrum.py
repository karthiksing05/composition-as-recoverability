"""Figure 2: the spectrum of demands on decomposition, with frameworks and this paper's systems in each set.
Colors follow the paper's rule: red = blend, orange = weak composition (own procedure), blues = strong composition."""
from figstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NOTES = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs"
NAVY = "#1b2a49"
RUNGS = [
    ("Blend", LOST, "No procedure returns\nthe parts",
     ["Conceptual blending", "Pet fish, red hair"],
     ["$x+y$: $R^2$ 0.50", "HRR pairs: $R^2$ 0.44", "Sentence states:", "  transfer $\\leq$ 0.05"]),
    ("Weak", BEH, "The system acts on new\ncombinations of known parts",
     ["Frege, Montague", "SCAN, COGS, Hupkes", "Text-to-image, diffusion", "RAAM, VSA clean-up"],
     ["Glyph decoders, V = 8:", "  new pairs 0.85–1.00,", "  new glyphs 0.20–0.34"]),
    ("Strong", REP, "A linear probe fixed in\nadvance returns new parts",
     ["Functional compositionality", "Reverse compositionality", "Context-dependent", "  constituents"],
     ["Glyph states: $R^2$ 0.83–0.86", "Image quadrants: κ ≥ 0.81", "phi-4 question state:", "  transfer 0.98"]),
    ("Shared role codes", REP2, "One vector per part,\nchanged only by its role",
     ["Tensor products", "Disentanglement", "Additive decoders", "TRE, Kolmogorov scores"],
     ["Glyph states, OVER:", "  δ 0.02–0.04"]),
    ("Explicit copies", NAVY, "The part itself is\ncopied into the whole",
     ["Classical architecture", "  (Fodor and Pylyshyn)"],
     ["Concatenation", "  reference: κ = 1"]),
]


def tint(hexcol, a):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(255 - a * (255 - c)) for c in (r, g, b))


fig = plt.figure(figsize=(7.0, 3.35))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 7.0); ax.set_ylim(0, 3.35); ax.axis("off")
W, G, X0 = 1.3, 0.1, 0.05
TOP, BOT = 2.82, 0.62
xs = [X0 + i * (W + G) for i in range(5)]
for x, (name, col, demand, fw, ours) in zip(xs, RUNGS):
    ax.add_patch(FancyBboxPatch((x, BOT), W, TOP - BOT, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=tint(col, 0.09), ec=col, lw=0.9))
    ax.text(x + 0.08, TOP - 0.13, name, fontsize=7.8, fontweight="bold", color=col, va="center")
    ax.text(x + 0.08, TOP - 0.40, demand, fontsize=6.3, color=INK, va="center", linespacing=1.15)
    ax.text(x + 0.08, TOP - 0.70, "Frameworks", fontsize=6.0, fontweight="bold", color=DGREY, va="center")
    for j, f in enumerate(fw):
        ax.text(x + 0.08, TOP - 0.86 - 0.135 * j, f, fontsize=6.2, color=INK, va="center")
    y_ours = TOP - 1.50
    ax.text(x + 0.08, y_ours, "In this paper", fontsize=6.0, fontweight="bold", color=DGREY, va="center")
    for j, s in enumerate(ours):
        ax.text(x + 0.08, y_ours - 0.16 - 0.135 * j, s, fontsize=6.2, color=col if col != NAVY else INK, va="center")


def bracket(x0, x1, y, label, col):
    ax.plot([x0, x0, x1, x1], [y - 0.06, y, y, y - 0.06], color=col, lw=0.8)
    ax.text((x0 + x1) / 2, y + 0.11, label, fontsize=6.8, color=col, ha="center", va="center")


bracket(xs[0] + 0.05, xs[0] + W - 0.05, TOP + 0.13, "composition that loses parts", LOST)
bracket(xs[1] + 0.05, xs[1] + W - 0.05, TOP + 0.13, "composition", BEH)
bracket(xs[2] + 0.05, xs[4] + W - 0.05, TOP + 0.13, "composition and decomposition", REP)
ax.add_patch(FancyArrowPatch((xs[1] + 0.05, BOT - 0.17), (xs[4] + W - 0.05, BOT - 0.17), arrowstyle="-|>",
                             mutation_scale=8, color=DGREY, lw=0.8))
ax.text((xs[1] + xs[4] + W) / 2, BOT - 0.31, "each set fixes more of the decomposition in advance",
        fontsize=6.6, color=DGREY, ha="center", va="center")
ax.text(xs[0] + 0.02, 0.15, "Outside the spectrum: unrestricted probes, recodings of the parts and probes scored on training parts,\nwhich every injective system passes.", linespacing=1.2, fontsize=6.4, color=GREY, va="center")
fig.savefig(f"{FIGDIR}/fig_spectrum.pdf")
fig.savefig(f"{NOTES}/fig_spectrum.png", dpi=170)
print("saved")
