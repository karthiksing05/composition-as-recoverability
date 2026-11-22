"""Aggregate E36: mean over seeds and items by model x kind (train/new) x phrasing, for direct / verify / parse."""
import glob, json, collections, os
import numpy as np
D = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e36"
rows = [r for f in sorted(glob.glob(f"{D}/*_s*.json")) for r in json.load(open(f, encoding="utf-8"))]
g = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for key in [(r["model"], r["kind"], r["phrasing"]), (r["model"], r["kind"], "all")]:
        for m in ("direct", "verify", "parse", "parse_fallback", "copy_exact", "rel_exact"):
            g[key][m].append(r[m])
    g[(r["model"], r["kind"], "rel:" + r["rel"])]["parse"].append(r["parse"])
    g[(r["model"], r["kind"], "rel:" + r["rel"])]["direct"].append(r["direct"])
    g[(r["model"], r["kind"], "rel:" + r["rel"])]["rel_exact"].append(r["rel_exact"])
out = {"|".join(k): {m: float(np.mean(v)) for m, v in d.items()} for k, d in sorted(g.items())}
json.dump(out, open(f"{D}/summary.json", "w", encoding="utf-8"), indent=1)
for k, d in out.items():
    print(f"{k:55s}", " ".join(f"{m}={v:.3f}" for m, v in d.items()))
