"""Figure 1: composition in the wild. Each column composes two parts into a whole, and the mark under it says
whether the parts can be taken back out. Blue = parts return (decomposition), red = blend (parts lost)."""
import glob, io, os
import numpy as np
import pandas as pd
from PIL import Image
from figstyle import *
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NOTES = "/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs"
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--zh-plus--tiny-imagenet/snapshots/*/data/train-*.parquet"))[0]
df = pd.read_parquet(f)
photo = lambda i: Image.open(io.BytesIO(df["image"][i]["bytes"])).convert("RGB")
dog, swan = photo(32503), photo(85003)
BG = (124, 116, 104)


def canvas_beside(a, b):
    c = Image.new("RGB", (224, 224), BG)
    c.paste(a.resize((112, 112), Image.BICUBIC), (0, 56)); c.paste(b.resize((112, 112), Image.BICUBIC), (112, 56))
    return c


def blend(a, b):
    x = (np.asarray(a.resize((224, 224), Image.BICUBIC), float) + np.asarray(b.resize((224, 224), Image.BICUBIC), float)) / 2
    return Image.fromarray(x.astype(np.uint8))


av, ch = Image.open(f"{A}/im13.png"), Image.open(f"{A}/im14.png")
COLS = [
    ("An avocado on an armchair", (av, ch), Image.open(f"{A}/im15.png"), True, "each part keeps its role"),
    ("An avocado armchair", (av, ch), Image.open(f"{A}/im16.png"), False, "the parts merge into one shape"),
    ("A lighthouse beside a swan", (dog, swan), canvas_beside(dog, swan), True, "each photograph keeps its place"),
    ("A lighthouse averaged\nwith a swan", (dog, swan), blend(dog, swan), False, "every pixel mixes both"),
    ("A cake on a plate", ("cake", "plate"), None, True, "the cake is on top in\n“a plate under a cake” too"),
]

fig = plt.figure(figsize=(7.0, 2.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 7.0); ax.set_ylim(0, 2.75)
W, G = 1.3, 0.125
for k, (title, parts, whole, ok, why) in enumerate(COLS):
    x = 0.005 + k * (W + G)
    ax.add_patch(FancyBboxPatch((x, 0.05), W, 2.62, boxstyle="round,pad=0,rounding_size=0.05", fc="#fafafa", ec=LGREY, lw=0.7))
    ax.text(x + W / 2, 2.55, title, fontsize=6.6, fontweight="bold", color=INK, ha="center", va="center")
    for j, p in enumerate(parts):
        px = x + 0.17 + j * 0.55
        if isinstance(p, str):
            ax.text(px + 0.21, 2.2, p, fontsize=8, color=INK, ha="center", va="center", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LGREY, lw=0.6))
        else:
            ia = fig.add_axes([px / 7.0, 1.99 / 2.75, 0.42 / 7.0, 0.42 / 2.75]); ia.imshow(p); ia.axis("off")
    ax.add_patch(FancyArrowPatch((x + W / 2 - 0.12, 1.92), (x + W / 2 - 0.12, 1.62), arrowstyle="-|>", mutation_scale=7, color=DGREY, lw=0.8))
    ax.text(x + W / 2 - 0.05, 1.77, "compose", fontsize=6.0, color=DGREY, va="center")
    if whole is None:
        ax.text(x + W / 2, 1.27, "a cake on a plate", fontsize=7.6, color=INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LGREY, lw=0.6))
        ax.text(x + W / 2, 0.95, "a plate under a cake", fontsize=7.6, color=INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LGREY, lw=0.6))
    else:
        ia = fig.add_axes([(x + W / 2 - 0.42) / 7.0, 0.78 / 2.75, 0.84 / 7.0, 0.84 / 2.75]); ia.imshow(whole); ia.axis("off")
    col = REP if ok else LOST
    ax.text(x + W / 2, 0.58, "parts return" if ok else "parts lost", fontsize=7.4, fontweight="bold", color=col, ha="center", va="center")
    ax.text(x + W / 2, 0.30, why, fontsize=6.1, color=DGREY, ha="center", va="center", linespacing=1.1)
fig.savefig(f"{FIGDIR}/fig_intro.pdf"); fig.savefig(f"{NOTES}/fig_intro.png", dpi=170)
print("saved")
