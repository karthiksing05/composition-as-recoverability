"""Imageable nouns: a noun is imageable if GroundingDINO finds it in the generated single image "a x" with score >= thr.
Writes results/e31/imageable.json with keys fit (capped), val, new (= the 100 held-out nouns of the e4 split), plus all-passing lists."""
import argparse, json
from e31_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--thr", type=float, default=0.35); ap.add_argument("--fitcap", type=int, default=80)
a = ap.parse_args()
sp = split(0); sc = {}
for l in open(f"{R31}/det/det_sd15_singles.jsonl", encoding="utf-8"):
    d = json.loads(l); sc[d["file"].split("/")[1][:-3]] = list(d["det"].values())[0]["score"]
ok = lambda x: sc.get(x, 0) >= a.thr
out = dict(thr=a.thr, fit=[x for x in sp["fit"] if ok(x)][:a.fitcap], fit_all=[x for x in sp["fit"] if ok(x)], val=[x for x in sp["val"] if ok(x)],
           new=[x for x in sp["fresh"] if ok(x)], scores=sc)
json.dump(out, open(f"{R31}/imageable.json", "w", encoding="utf-8"), indent=1)
print({k: len(v) for k, v in out.items() if k != "thr" and k != "scores"}, "of", {k: len(sp[k]) for k in ("fit", "val", "fresh")})
print("held-out dropped:", [x for x in sp["fresh"] if not ok(x)])
