"""Shared figure style, matched to the reference preprint (seaborn-deep palette,
thin axes, sans labels, STIX math, bold lowercase panel letters)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, RED, PURPLE = "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"
GREY, LGREY, DGREY = "#929292", "#d5d5d5", "#5e5e5e"
FIGDIR = "/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def panel(ax, letter, title):
    ax.set_title(f"$\\bf{{{letter}.}}$ {title}", loc="left", fontweight="normal")


def band(ax, x, mean, sem, color, label=None, **kw):
    ax.plot(x, mean, color=color, label=label, **kw)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
