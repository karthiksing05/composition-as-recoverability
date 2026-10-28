"""Conceptual figures: taxonomy of modes of combination, and fibers of a blend."""
import numpy as np
from matplotlib.patches import FancyBboxPatch
from figstyle import *

TINT = {BLUE: "#e4ebf5", GREY: "#efefef", ORANGE: "#f9eadf"}


def box(ax, x, y, w, h, title, body, color, dashed=False, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=TINT.get(color, "white"), ec=color, lw=1.4, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=fs, fontweight="bold", color="#222222")
    ax.text(x + w / 2, y + h * 0.3, body, ha="center", va="center", fontsize=6.2, color=DGREY, linespacing=1.15)


def line(ax, pts):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=GREY, lw=0.9, solid_capstyle="round")


# ---------------- taxonomy ----------------
fig, ax = plt.subplots(figsize=(6.5, 2.75))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.25, 0.83, 0.48, 0.15, "Combination of parts", "the whole is a function of its parts", "#222222")
box(ax, 0.77, 0.83, 0.22, 0.15, "Chunking", "orthogonal: stores a\ncomposite as a unit", GREY, dashed=True)
line(ax, [(0.5, 0.83), (0.5, 0.76)]); line(ax, [(0.15, 0.76), (0.85, 0.76)])
for x in (0.15, 0.5, 0.85):
    line(ax, [(x, 0.76), (x, 0.70)])
box(ax, 0.01, 0.50, 0.28, 0.20, "Blending", "parts not recoverable, even\nfrom a vocabulary, or roles collide", ORANGE)
box(ax, 0.335, 0.50, 0.33, 0.20, "Fixed-width superposition", "role-distinct; recoverable only by\nclean-up over a known vocabulary", GREY)
box(ax, 0.71, 0.50, 0.28, 0.20, "Role–filler composition", "parts recoverable, including\nparts never seen before", BLUE)
ax.text(0.15, 0.39, "symmetric: $x+y$, $x\\odot y$;\nhead\u2013modifier, prototypes;\nconceptual blending", ha="center", va="top", fontsize=6.3, color=DGREY, style="italic")
ax.text(0.5, 0.39, "HRR, MAP, $x+0.5y$,\nprojections with $m<kd$\n(fresh-part ceiling $\\mathrm{rank}/kd$)",
        ha="center", va="top", fontsize=6.3, color=DGREY, style="italic")
ax.text(0.85, 0.39, "tensor products, concatenation,\nsuperposition in disjoint regions,\nslots and templates",
        ha="center", va="top", fontsize=6.3, color=DGREY, style="italic")
ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.655, 0.13, boxstyle="round,pad=0.006,rounding_size=0.012", fc="#fbfbfb", ec=LGREY, lw=0.8))
ax.text(0.3375, 0.085, "Familiar parts: decodable whenever the relation is injective on the vocabulary\n"
        "Systematicity and productivity are contingent on the data", ha="center", va="center", fontsize=6.6, color=DGREY)
ax.add_patch(FancyBboxPatch((0.71, 0.02), 0.28, 0.13, boxstyle="round,pad=0.006,rounding_size=0.012", fc=TINT[BLUE], ec=BLUE, lw=0.8))
ax.text(0.85, 0.085, "Fresh parts: recoverable\nSystematicity is necessary", ha="center", va="center", fontsize=6.6, color=BLUE)
fig.savefig(f"{FIGDIR}/fig_taxonomy.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_taxonomy.png", dpi=200)

# ---------------- fibers ----------------
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw=dict(wspace=0.35))
rng = np.random.default_rng(1)
voc = np.sort(rng.uniform(-1, 1, 7))
X, Y = np.meshgrid(voc, voc)
z = 0.3
ax = axs[0]
xs = np.linspace(-1.2, 1.2, 50)
ax.plot(xs, z - xs, color=ORANGE, lw=1.4)
ax.scatter(X, Y, s=12, color=LGREY, zorder=2, label="vocabulary pairs")
near = np.abs(X + Y - z) < 0.06
ax.scatter(X[near], Y[near], s=16, color=DGREY, zorder=3)
fr = rng.uniform(-1, 1, 6)
ax.scatter(fr, z - fr, s=16, marker="D", color=ORANGE, zorder=4, ec="white", lw=0.4, label="fresh pairs, same composite")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
ax.set_xlabel("part in role 1, $x$"); ax.set_ylabel("part in role 2, $y$")
ax.text(1.1, -0.95, "$x+y=z$", color=ORANGE, fontsize=7, ha="right")
ax.legend(loc="upper left", fontsize=5.9, handletextpad=0.2, borderaxespad=0.1)
panel(ax, "a", "Blend: many decompositions")
ax = axs[1]
ax.scatter(X, Y, s=12, color=LGREY, zorder=2)
ax.scatter([0.45], [-0.15], s=26, color=BLUE, zorder=4, ec="white", lw=0.5)
ax.axvline(0.45, color=BLUE, lw=0.6, ls=":"); ax.axhline(-0.15, color=BLUE, lw=0.6, ls=":")
ax.text(-1.1, 1.0, "$(x, y)$ read back from $x\\otimes\\rho_1 + y\\otimes\\rho_2$", color=BLUE, fontsize=6.6)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
ax.set_xlabel("part in role 1, $x$"); ax.set_ylabel("part in role 2, $y$")
panel(ax, "b", "Role\u2013filler: one decomposition")
fig.savefig(f"{FIGDIR}/fig_fibers.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_fibers.png", dpi=200)
print("ok")
