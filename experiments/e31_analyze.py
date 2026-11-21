"""Metrics from detections at a declared box threshold (D1 and D4 position/AND variants)."""
import argparse, json, glob, collections
import numpy as np
from e31_common import *
ap = argparse.ArgumentParser()
ap.add_argument("--index", nargs="+", required=True)
ap.add_argument("--det", nargs="+", required=True)
ap.add_argument("--thr", type=float, default=0.35)
ap.add_argument("--out", default=None)
ap.add_argument("--iou", type=float, default=0.6)
a = ap.parse_args()
recs = [r for f in a.index for r in json.load(open(f, encoding="utf-8"))]
det = {}
for f in a.det:
    for l in open(f, encoding="utf-8"):
        d = json.loads(l); det[d["file"]] = d["det"]


def iou(b, c):
    def xy(b): return b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    p, q = xy(b), xy(c)
    iw, ih = max(0, min(p[2], q[2]) - max(p[0], q[0])), max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = iw * ih
    return i / (b[2] * b[3] + c[2] * c[3] - i + 1e-9)


def info(r):
    d = det.get(r["file"])
    if d is None: return None
    dx, dy = d[r["x"]], d[r["y"]]
    both = dx["score"] >= a.thr and dy["score"] >= a.thr and iou(dx["box"], dy["box"]) < a.iou
    k = 1 if r["axis"] == "v" else 0
    xfirst = dx["box"][k] < dy["box"][k]  # figure x in first slot (top / left)
    return dict(both=both, xfirst=xfirst)


groups = collections.defaultdict(list)
for r in recs:
    i = info(r)
    if i: groups[(r["kind"], r["rel"], r["phr"])].append((r, i))
rows = []
for (kind, rel, phr), L in sorted(groups.items()):
    n = len(L); B = [(r, i) for r, i in L if i["both"]]
    corr = np.mean([i["xfirst"] for r, i in B]) if B else float("nan")  # meaning-correct: figure x in first slot
    wob = np.mean([(i["xfirst"] if phr == "canon" else not i["xfirst"]) for r, i in B]) if B else float("nan")  # first-mentioned noun in first slot
    # converse agreement over (pair, seed)
    row = dict(kind=kind, rel=rel, phr=phr, n=n, both=len(B) / n, n_both=len(B), rel_acc=float(corr), word_order=float(wob))
    rows.append(row)
# converse agreement
by = collections.defaultdict(dict)
for r in recs:
    i = info(r)
    if i and i["both"]: by[(r["kind"], r["rel"], r["pair"], r["seed"])][r["phr"]] = i
agree = collections.defaultdict(list)
for (kind, rel, pi, s), d in by.items():
    if "canon" in d and "conv" in d:
        agree[(kind, rel)].append((d["canon"]["xfirst"] == d["conv"]["xfirst"], d["canon"]["xfirst"] and d["conv"]["xfirst"]))
for row in rows:
    g = agree.get((row["kind"], row["rel"]))
    row["conv_agree_same_layout"] = float(np.mean([u for u, v in g])) if g else None
    row["conv_both_correct"] = float(np.mean([v for u, v in g])) if g else None
    row["n_conv_pairs"] = len(g) if g else 0
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
if a.out:
    json.dump(dict(thr=a.thr, rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
