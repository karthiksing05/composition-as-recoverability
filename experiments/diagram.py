"""Drawing helpers shared by the design figures: a full-figure canvas in figure fractions,
soft rounded cards, arrows, text, thumbnails and word chips."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import *

TINT = {BLUE: "#e8eef7", ORANGE: "#fbeee4", GREEN: "#e6f2e9", GREY: "#f2f2f2", PURPLE: "#eeebf6", RED: "#f8e6e6"}


class Canvas:
    def __init__(self, w_in, h_in):
        self.W, self.H = w_in, h_in
        self.fig = plt.figure(figsize=(w_in, h_in))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.ax.axis("off")

    def card(self, x, y, w, h, color=GREY, fill=True, lw=0.8, ls="-", ec=None):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012",
                                         fc=TINT.get(color, "white") if fill else "white", ec=ec or color, lw=lw, ls=ls))

    def arrow(self, p, q, color=DGREY, lw=0.8, rad=0.0, style="-|>"):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                          connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))

    def text(self, x, y, s, fs=6.6, c=INK, **kw):
        self.ax.text(x, y, s, fontsize=fs, color=c, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), **kw)

    def title(self, x, y, letter, s):
        self.ax.text(x, y, f"$\\bf{{{letter}}}$  {s}", fontsize=8.2, ha="left", va="center", color=INK)

    def img(self, x, y, a, h_in, edge=None, lw=0.9, gray=True):
        """Thumbnail with its lower-left corner at (x, y), h_in inches tall; returns its width."""
        w_in = h_in * a.shape[1] / a.shape[0]
        iax = self.fig.add_axes([x, y, w_in / self.W, h_in / self.H])
        iax.imshow(a, cmap="Greys" if gray else None, vmin=0, vmax=1, interpolation="nearest" if gray else "bilinear")
        iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_visible(edge is not None); s.set_color(edge or "white"); s.set_linewidth(lw)
        return w_in / self.W

    def chip(self, x, y, s, color=GREY, fs=6.3, ha="left"):
        self.ax.text(x, y, s, fontsize=fs, ha=ha, va="center", color=INK,
                     bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.25", fc="white", ec=color, lw=0.8))

    def state(self, x, y, n=8, w=0.016, h=0.011, color=BLUE):
        for j in range(n):
            v = 0.25 + 0.6 * abs(np.sin(3.1 * j + 1))
            self.ax.add_patch(plt.Rectangle((x, y + j * h * 1.1), w, h, fc=color, alpha=v, ec="none"))
        return y + n * h * 1.1

    def save(self, name):
        self.fig.savefig(f"{FIGDIR}/{name}.pdf")
        self.fig.savefig(f"/coc/flash10/ksingara3/compositionality-theory/notes/{name}.png", dpi=220)


class Flow(Canvas):
    """A canvas in inches, for flow charts laid out on a grid."""

    def __init__(self, w_in, h_in):
        super().__init__(w_in, h_in)
        self.ax.set_xlim(0, w_in); self.ax.set_ylim(0, h_in)

    def card(self, x, y, w, h, color=GREY, fill=True, lw=0.7, ls="-", ec=None, r=0.05):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=TINT.get(color, "white") if fill else "white", ec=ec or color, lw=lw, ls=ls))

    def box(self, x, y, w, h, head, sub=None, color=GREY, fill=True, ls="-", fs=6.6):
        """A card with a bold head line and an optional grey second line, both centred."""
        self.card(x, y, w, h, color, fill=fill, ls=ls)
        dy = 0.055 if sub else 0
        self.text(x + w / 2, y + h / 2 + dy, head, fs=fs, fontweight="bold")
        if sub:
            self.text(x + w / 2, y + h / 2 - 0.065, sub, fs=fs - 0.9, c=DGREY)

    def arrow(self, p, q, color=DGREY, lw=0.7, rad=0.0, style="-|>", elbow=False):
        cs = "angle,angleA=0,angleB=90,rad=0" if elbow else f"arc3,rad={rad}"
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6.5, color=color, lw=lw,
                                          connectionstyle=cs, shrinkA=1.0, shrinkB=1.0))

    def head(self, x, y, letter, s):
        self.ax.text(x, y, f"$\\bf{{{letter}}}$  {s}", fontsize=7.8, ha="left", va="baseline", color=INK)

    def step(self, x, y, n, color=INK):
        self.ax.add_patch(plt.Circle((x, y), 0.055, fc=color, ec="none"))
        self.text(x, y - 0.003, str(n), fs=5.6, c="white", fontweight="bold")

    def img(self, x, y, a, h_in, edge=None, lw=0.8, gray=True):
        w_in = h_in * a.shape[1] / a.shape[0]
        iax = self.fig.add_axes([x / self.W, y / self.H, w_in / self.W, h_in / self.H])
        iax.imshow(a, cmap="Greys" if gray else None, vmin=0, vmax=1, interpolation="nearest" if gray else "bilinear", aspect="auto")
        iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_visible(edge is not None); s.set_color(edge or "white"); s.set_linewidth(lw)
        return w_in

    def state(self, x, y, n=7, w=0.07, h=0.042, color=BLUE):
        for j in range(n):
            v = 0.25 + 0.6 * abs(np.sin(3.1 * j + 1))
            self.ax.add_patch(plt.Rectangle((x, y + j * h), w, h * 0.82, fc=color, alpha=v, ec="none"))
        return y + n * h

    def chip(self, x, y, s, color=GREY, fs=6.2, ha="left"):
        self.ax.text(x, y, s, fontsize=fs, ha=ha, va="center", color=INK,
                     bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.2", fc="white", ec=color, lw=0.7))
