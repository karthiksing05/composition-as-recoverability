import glob, json
import numpy as np
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/"
for a in ("mlp", "cnn", "slot", "tf"):
    rr = [json.load(open(f, encoding="utf-8")) for f in glob.glob(R + f"e2redec/{a}_m64_V8_s*.json")]
    print("e2redec", a, len(rr), "orig", {k: round(float(np.mean([r["orig"][k] for r in rr])), 3) for k in rr[0]["orig"]},
          "redec_trained", rr[0]["redec_trained"] if isinstance(rr[0]["redec_trained"], float) else {k: round(float(np.mean([r["redec_trained"][k] for r in rr])), 3) for k in rr[0]["redec_trained"]})
    vs = [json.load(open(f, encoding="utf-8")) for f in glob.glob(R + f"e2v2/{a}_m64_V8_c*_s*.json")]
    vs = [v[0] if isinstance(v, list) else v for v in vs]
    print("  e2v2", len(vs), "coverage", sorted({v.get("coverage") for v in vs}), {c: round(float(np.mean([v[c] for v in vs])), 3) for c in ("beh_fresh_r00", "beh_fresh_r01", "beh_fresh_r10", "beh_fresh_r11")})
print("e20cal gain0", [(r["V"], r["seed"], round(r["own_heldout"], 3)) for r in (json.load(open(f, encoding="utf-8")) for f in glob.glob(R + "e20cal/*g0*.json"))])
# E26 and the lower-left cell
rows = [r for f in glob.glob(R + "e19/*OLMo-2-1124-7B*.json") for r in json.load(open(f, encoding="utf-8"))]
print("e19 7B rows", len(rows), sorted(rows[0].keys())[:24] if rows else None)
