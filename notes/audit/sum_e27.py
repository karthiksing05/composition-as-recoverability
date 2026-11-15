"""E27 summary: input floor on the Random Hierarchy Model, held-out vs training symbols, kappa."""
import glob, json, collections
import numpy as np
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e27/"
rows = [r for f in glob.glob(R + "*.json") for r in json.load(open(f, encoding="utf-8"))]
print("n files", len(glob.glob(R + "*.json")), "rows", len(rows))
g = collections.defaultdict(list)
for r in rows:
    g[(r["s"], r["L"], r["v"], r["m"], r["level"], r["code"], r["system"])].append(r)
keys = sorted({k[:6] for k in g})
print("s L v m lvl code | input R2 held-out / training-sym | sum ref held-out | concat | kappa(input)")
allh, allt = [], []
for k in keys:
    def m(sysname, f):
        return float(np.mean([f(x) for x in g[k + (sysname,)]])) if g[k + (sysname,)] else np.nan
    ih, it = m("input", lambda x: 1 - x["eps_heldout"]), m("input", lambda x: 1 - x["eps_train_sym"])
    sh, ch = m("ref_sum", lambda x: 1 - x["eps_heldout"]), m("ref_concat", lambda x: 1 - x["eps_heldout"])
    eb, er, ei = 1 - sh, 1 - ch, 1 - ih
    kap = (eb - ei) / (eb - er) if abs(eb - er) > 1e-6 else np.nan
    allh.append(ih); allt.append(it)
    print(*k, "| %.2f / %.2f | %.2f | %.2f | %.2f" % (ih, it, sh, ch, kap), "seeds", len(g[k + ("input",)]))
print("input R2 on held-out symbols: min %.2f max %.2f; on training symbols: min %.2f max %.2f" % (min(allh), max(allh), min(allt), max(allt)))
print("settings with held-out input R2 <= 0.5:", sum(x <= 0.5 for x in allh), "of", len(allh))
