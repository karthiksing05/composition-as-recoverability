"""Shared figure style: Fira Sans for text and math, seaborn-deep palette, thin grey axes,
bold lowercase panel letters."""
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "*.ttf")):
    fm.fontManager.addfont(_f)

BLUE, ORANGE, GREEN, RED, PURPLE = "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"
GREY, LGREY, DGREY = "#929292", "#d5d5d5", "#5e5e5e"
INK = "#262626"
FIGDIR = "/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Fira Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Fira Sans", "mathtext.it": "Fira Sans:italic",
    "mathtext.bf": "Fira Sans:bold", "mathtext.sf": "Fira Sans", "mathtext.fallback": "stixsans",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": DGREY, "ytick.color": DGREY,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.edgecolor": DGREY,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def panel(ax, letter, title):
    ax.set_title(f"$\\bf{{{letter}}}$  {title}", loc="left", fontweight="normal", pad=6)


def band(ax, x, mean, sem, color, label=None, **kw):
    ax.plot(x, mean, color=color, label=label, **kw)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)


def ygrid(ax):
    ax.yaxis.grid(True, color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
