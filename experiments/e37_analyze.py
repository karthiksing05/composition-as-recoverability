"""E37 analysis.  For every caption and seed GLIGEN drew layout A (figure in the first slot) and layout B (ground there).
Detections (e31_score.py) at E31's threshold (0.35, IoU < 0.6) say which noun sits in the first slot (top / left).
  compliance   share of images whose first-slot noun is the one the layout put there (both nouns detected)
  planner P    picks A when P names the figure, else B; placement = the figure sits in the first slot of the chosen image,
               over captions whose chosen image holds both nouns.  Planners:
                 position  the first-mentioned noun takes the first slot (roles by position)
                 lexicon   code parses the caption with a lexicon fixed in advance (on/under, above/below, to the left of /
                           to the right of -> which mentioned noun is the figure)
                 oracle    the figure (upper bound set by GLIGEN's compliance)
                 <LM>      results/e37/plan_<LM>.json
Also the unplanned baseline: SD 1.5 on the same captions and seeds (E31, results/e31/det).
usage: e37_analyze.py"""
import glob, json, os, collections
import numpy as np

R31 = "/coc/flash10/ksingara3/compositionality-theory/results/e31"
OUT = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e37"
THR, IOU = 0.35, 0.6
LEX = {"on": 0, "under": 1, "above": 0, "below": 1, "to the left of": 0, "to the right of": 1}


def iou(b, c):
    def xy(b): return b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    p, q = xy(b), xy(c)
    iw, ih = max(0, min(p[2], q[2]) - max(p[0], q[0])), max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = iw * ih
    return i / (b[2] * b[3] + c[2] * c[3] - i + 1e-9)


def load_det(files):
    det = {}
    for f in files:
        for l in open(f, encoding="utf-8"):
            d = json.loads(l); det[d["file"]] = d["det"]
    return det


def xfirst(r, d):
    """None if the two nouns are not both detected apart; else whether the figure x is in the first slot."""
    if d is None: return None
    dx, dy = d[r["x"]], d[r["y"]]
    if not (dx["score"] >= THR and dy["score"] >= THR and iou(dx["box"], dy["box"]) < IOU): return None
    k = 1 if r["axis"] == "v" else 0
    return dx["box"][k] < dy["box"][k]


recs = [r for f in sorted(glob.glob(f"{R31}/index/gligen_*_[AB].json")) for r in json.load(open(f, encoding="utf-8"))]
det = load_det(glob.glob(f"{R31}/det/det_gligen_*.jsonl"))
img = {}
for r in recs:
    img[(r["rel"], r["pair"], r["phr"], r["seed"], r["layout"])] = (r, xfirst(r, det.get(r["file"])))
# compliance
comp = collections.defaultdict(list)
for (rel, pi, phr, s, L), (r, xf) in img.items():
    if xf is not None:
        comp[(rel, L)].append(xf if L == "A" else not xf)
rows = {"compliance": {f"{rel}|{L}": dict(rate=float(np.mean(v)), n=len(v)) for (rel, L), v in sorted(comp.items())}}
# planners
plans = {}
for f in sorted(glob.glob(f"{OUT}/plan_*.json")):
    P = json.load(open(f, encoding="utf-8")); plans[P["model"]] = P["choices"]


def pick(planner, r):
    if planner == "oracle": return r["x"]
    if planner == "position": return r["first"]
    if planner == "lexicon":
        words = r["caption"].split(" ")
        for ph, fig in LEX.items():
            if f" {ph} " in f" {r['caption']} ":
                first, second = (r["x"], r["y"]) if r["phr"] == "canon" else (r["y"], r["x"])
                return (first, second)[fig]
        return r["first"]
    return plans[planner][f"{r['rel']}|{r['pair']}|{r['phr']}"]


res = {}
for planner in ["oracle", "lexicon", "position"] + list(plans):
    acc = collections.defaultdict(list); fig_ok = collections.defaultdict(list)
    for (rel, pi, phr, s, L), (r, xf) in img.items():
        if L != "A": continue
        f = pick(planner, r)
        chosen = "A" if f == r["x"] else "B"
        fig_ok[(rel, phr)].append(f == r["x"])
        rr, xc = img.get((rel, pi, phr, s, chosen), (None, None))
        if xc is not None:
            acc[(rel, phr)].append(xc)
    res[planner] = {f"{rel}|{phr}": dict(placement=float(np.mean(v)), n=len(v), planner_role_acc=float(np.mean(fig_ok[(rel, phr)])))
                    for (rel, phr), v in sorted(acc.items())}
    allv = [x for v in acc.values() for x in v]
    res[planner]["ALL"] = dict(placement=float(np.mean(allv)) if allv else None, n=len(allv),
                               planner_role_acc=float(np.mean([x for v in fig_ok.values() for x in v])))
rows["planners"] = res
# unplanned SD 1.5 baseline on the same captions, seeds 0-1
base = collections.defaultdict(list)
sdrec = [r for rel in ("on", "above", "left") for r in json.load(open(f"{R31}/index/sd15_pairs_{rel}.json", encoding="utf-8")) if r["seed"] < 2]
sdd = load_det([f"{R31}/det/det_sd15_pairs_{rel}.jsonl" for rel in ("on", "above", "left")])
for r in sdrec:
    xf = xfirst(r, sdd.get(r["file"]))
    if xf is not None:
        base[(r["rel"], r["phr"])].append(xf)
rows["sd15_unplanned"] = {f"{k[0]}|{k[1]}": dict(placement=float(np.mean(v)), n=len(v)) for k, v in sorted(base.items())}
rows["sd15_unplanned"]["ALL"] = dict(placement=float(np.mean([x for v in base.values() for x in v])))
os.makedirs(OUT, exist_ok=True)
json.dump(rows, open(f"{OUT}/summary.json", "w", encoding="utf-8"), indent=1)
for k, v in rows["compliance"].items(): print("compliance", k, round(v["rate"], 3), v["n"])
for p, d in res.items():
    print(f"{p:32s}", " ".join(f"{k}:{v['placement']:.2f}({v['planner_role_acc']:.2f})" for k, v in d.items() if v["placement"] is not None))
print("sd15 unplanned", {k: round(v["placement"], 3) for k, v in rows["sd15_unplanned"].items()})
