"""E26 at later layers, E24b pixels vs quadrant states, E23 and E24 final, E20 gain 0 completeness."""
import glob, json, collections
import numpy as np
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/"


def load(d):
    out = []
    for f in glob.glob(R + d + "/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        out += r if isinstance(r, list) else [r]
    return out


print("== E26")
for d in ("e19", "e26L24", "e26L32"):
    rows = [r for r in load(d) if "OLMo-2-1124-7B" in r["model"] and r["state"] == "query"]
    if rows:
        print(d, "seeds", sorted({r["seed"] for r in rows}), "layer", {r["layer"] for r in rows}, {a: round(float(np.mean([x[a] for x in rows])), 3) for a in ("ridge_conv", "kernel_conv", "mlp_conv", "kernel_oracle_conv", "mlp_oracle_conv", "ridge_can")})
print("== E24b pixels (kappa at each encoder's validation-best layer and code)")
rec = lambda x: (x["rec_fresh_role1"] + x["rec_fresh_role2"]) / 2
rows = load("e24b")
for m in sorted({r["model"] for r in rows}):
    c = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["model"] == m:
            c[(r["layer"], r["code"])][r["system"]].append(r)
    best = {}
    for k, v in c.items():
        rs, rc, p = [np.mean([rec(x) for x in v[s]]) for s in ("ref_sum", "ref_concat", "pixels")]
        best[k] = (p - rs) / (rc - rs)
    print(m, "seeds", sorted({r["seed"] for r in rows if r["model"] == m}), {k: round(float(v), 3) for k, v in sorted(best.items())})
print("== E8grid quadrant and mean, same units")
g = load("e8grid")
for m in sorted({r["model"] for r in rows}):
    gg = [r for r in g if r["model"] == m]
    out = {}
    for lay in sorted({r["layer"] for r in gg}):
        for code in sorted({r["code"] for r in gg}):
            lr = "L" + lay[1:]
            refs = [r for r in gg if r["layer"] == lr and r["code"] == code]
            a = [rec(r) for r in gg if r["layer"] == lay and r["code"] == code and r["system"] == "model"]
            rs = np.mean([rec(r) for r in refs if r["system"] == "ref_sum"]); rc = np.mean([rec(r) for r in refs if r["system"] == "ref_concat"])
            out[(lay, code)] = round(float((np.mean(a) - rs) / (rc - rs)), 3)
    print(m, out)
print("== E23 counts")
print(collections.Counter(r["model"].split("/")[-1] for r in load("e23") if r["picked"] and r["test"] == "nonce"))
print("== E24 counts", collections.Counter(r["model"].split("/")[-1] for r in load("e24") if r["system"] == "model" and r["code"] == "phrase" and r["layer"] == sorted({x["layer"] for x in load("e24") if x["model"] == r["model"]})[0]))
print("== E20 gain 0 counts", collections.Counter((r["arch"], r["arm"]) for r in load("e20") if r["gain"] == 0))
print("== E21b counts", collections.Counter(r["model"].split("/")[-1] + f"_s{r['seed']}" for r in load("e21b") if r["kind"] == "whole" and r["rel"] == "on" or False))
