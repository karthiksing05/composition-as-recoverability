"""Shared helpers for the fig_case_* figures: an inch-coordinate canvas (y grows downward), detector records of the e31/e32
diffusion tests, the box-threshold rule of e31_analyze.py, and the Tiny-ImageNet photographs of the image-encoder cases.
Everything shown comes from saved outputs; nothing here runs a model."""
import io, json, os, textwrap
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from figstyle import *

ROOT = "/coc/flash10/ksingara3/compositionality-theory"
R31, R32 = f"{ROOT}/results/e31", f"{ROOT}/results/e32"
RES = f"{ROOT}/experiments/results"
PNGDIR = f"{ROOT}/notes/audit/figs"
THR, IOU = 0.35, 0.6  # box threshold and overlap limit of e31_analyze.py (the paper's detector setting)
FIGC, GRDC = BLUE, GREY  # figure noun box, ground noun box
OKC, BADC = "#2e7d46", RED
plt.rcParams["image.composite_image"] = False  # keep each photograph its own PDF image (otherwise matplotlib merges them into one page-sized image)


class Page:
    """One axes spanning the figure; coordinates in inches from the top left."""
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(h, 0); self.ax.axis("off")

    def img(self, x, y, a, s, edge=None, lw=0.6, h=None):
        h = s if h is None else h
        self.ax.imshow(a, extent=(x, x + s, y + h, y), interpolation="lanczos", zorder=1)
        self.ax.set_xlim(0, self.W); self.ax.set_ylim(self.H, 0)
        if edge:
            self.ax.add_patch(Rectangle((x, y), s, h, fill=False, ec=edge, lw=lw, zorder=4))

    def box(self, x, y, s, b, color, lw=1.1, ls="-"):
        """b = (cx, cy, w, h) normalised to an image of side s drawn at (x, y)."""
        cx, cy, w, h = b
        x0, y0, x1, y1 = (max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2))  # clip to the image
        self.ax.add_patch(Rectangle((x + x0 * s, y + y0 * s), (x1 - x0) * s, (y1 - y0) * s, fill=False, ec=color, lw=lw, ls=ls, zorder=3))

    def text(self, x, y, s, fs=6.2, c=INK, ha="left", va="top", **kw):
        return self.ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=5, **kw)

    def head(self, x, y, letter, s, fs=7.2):
        self.ax.text(x, y, f"$\\bf{{{letter}}}$  {s}", fontsize=fs, color=INK, ha="left", va="top")

    def hline(self, x0, x1, y, c=GREY, lw=0.8, ls="-"):
        self.ax.plot([x0, x1], [y, y], color=c, lw=lw, ls=ls, zorder=2)

    def arrow(self, p, q, c=DGREY, lw=0.7):
        self.ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=6))

    def save(self, name):
        os.makedirs(PNGDIR, exist_ok=True)
        self.fig.savefig(f"{FIGDIR}/{name}.pdf")
        self.fig.savefig(f"{PNGDIR}/{name}.png", dpi=170)
        plt.close(self.fig)
        print("saved", name)


def wrap(s, n=22, lines=2):
    w = textwrap.wrap(s, n)
    return "\n".join(w[:lines]) if len(w) <= lines else "\n".join(textwrap.wrap(s, n + 6))


def jl(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l); d[r["file"]] = r["det"]
    return d


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iou(b, c):
    def xy(b): return b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    p, q = xy(b), xy(c)
    iw, ih = max(0, min(p[2], q[2]) - max(p[0], q[0])), max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = iw * ih
    return i / (b[2] * b[3] + c[2] * c[3] - i + 1e-9)


def info(rec, det):
    """Both objects detected as separate boxes (e31_analyze rule) and whether the figure x sits first (left / top)."""
    d = det.get(rec["file"])
    if d is None:
        return None
    dx, dy = d[rec["x"]], d[rec["y"]]
    both = dx["score"] >= THR and dy["score"] >= THR and iou(dx["box"], dy["box"]) < IOU
    k = 1 if rec["axis"] == "v" else 0
    return dict(both=both, xfirst=dx["box"][k] < dy["box"][k], bx=dx["box"], by=dy["box"])


def verdict(rec, i):
    """Text and colour of the mark: did the figure x land in its place (left of / above / on top of the ground)?"""
    ok = i["xfirst"]
    side = {"h": ("left", "right"), "v": ("above", "below")}[rec["axis"]][0 if ok else 1]
    return f"{rec['x']} {side}", OKC if ok else BADC


def photo_loader():
    import e8_vision as V
    ims, labels = V.load_images()
    return V, ims, labels, (lambda i: Image.open(io.BytesIO(ims[i])).convert("RGB"))


def class_names():
    import glob, timm
    info_ = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--zh-plus--tiny-imagenet/snapshots/*/dataset_infos.json"))[0]
    wnids = next(iter(json.load(open(info_, encoding="utf-8")).values()))["features"]["label"]["names"]
    lem = {}
    with open(os.path.join(os.path.dirname(timm.__file__), "data/_info/imagenet_synset_to_lemma.txt"), encoding="utf-8") as f:
        for line in f:
            w, l = line.rstrip("\n").split("\t")
            lem[w] = l.split(",")[0].strip()
    return [lem.get(w) for w in wnids]


def pick_pairs(model, rel, k, seed=0):
    """Selection rule (fixed, no hand picking): walk the pairs in index order; keep a pair when, at `seed`, both phrasings
    (canonical and converse) have both objects detected as separate boxes (score >= 0.35, IoU < 0.6); take the first k.
    Returns a list of (canon record, canon info, conv record, conv info)."""
    recs = jload(f"{R31}/index/{model}_pairs_{rel}.json")
    det = jl(f"{R31}/det/det_{model}_pairs_{rel}.jsonl")
    by = {}
    for r in recs:
        if r["seed"] == seed:
            by.setdefault(r["pair"], {})[r["phr"]] = r
    out = []
    for p in sorted(by):
        c, v = by[p].get("canon"), by[p].get("conv")
        if not c or not v:
            continue
        ic, iv = info(c, det), info(v, det)
        if ic and iv and ic["both"] and iv["both"]:
            out.append((c, ic, v, iv))
        if len(out) == k:
            break
    return out


def load_png(model, rec):
    return np.asarray(Image.open(f"{R31}/images/{model}/{rec['file']}.png").convert("RGB"))


def mark(P, x, y, ok, s=0.075, lw=1.3):
    """Vector check / cross with its top-left corner at (x, y) (Fira Sans has no check-mark glyphs)."""
    c = OKC if ok else BADC
    if ok:
        P.ax.plot([x, x + s * 0.4, x + s * 1.1], [y + s * 0.55, y + s * 0.95, y + s * 0.05], color=c, lw=lw, solid_capstyle="round", zorder=5)
    else:
        P.ax.plot([x + s * .1, x + s], [y + s * .05, y + s * .95], color=c, lw=lw, solid_capstyle="round", zorder=5)
        P.ax.plot([x + s * .1, x + s], [y + s * .95, y + s * .05], color=c, lw=lw, solid_capstyle="round", zorder=5)
