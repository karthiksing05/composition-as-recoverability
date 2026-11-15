import glob, json, collections
import numpy as np
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/"


def load(d):
    out = []
    for f in glob.glob(R + d + "/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        out += r if isinstance(r, list) else [r]
    return out


print("== E20 base, m=64: mean (n) over seeds")
r20 = load("e20")
print(collections.Counter((r["gain"], r["arch"], r["arm"]) for r in r20))
F = ["own_heldout", "own_newpairs", "read_trained", "read_untrained", "read_input", "redec_trained", "redec_untrained"]
g = collections.defaultdict(list)
for r in r20:
    g[(r["gain"], r["arch"], r["V"], r["arm"])].append(r)
print("gain arch V arm n | " + " ".join(c[:10] for c in F) + " through")
for k in sorted(g):
    print(*k, len(g[k]), "| " + " ".join(f"{np.mean([x[c] for x in g[k]]):.3f}" for c in F), round(float(np.mean([x.get("through", np.nan) for x in g[k]])), 3))
print("== E20 V=8 base: paired trained-input and trained-untrained, 2 SE")
for gn in (0.0, 4.0):
    for a in ("mlp", "cnn", "slot", "tf"):
        x = sorted(g.get((gn, a, 8, "base"), []), key=lambda r: r["seed"])
        if len(x) > 1:
            di = np.array([r["read_trained"] - r["read_input"] for r in x]); du = np.array([r["read_trained"] - r["read_untrained"] for r in x])
            print(gn, a, f"tr-in {di.mean():.3f}+-{2 * di.std(ddof=1) / np.sqrt(len(x)):.3f} tr-un {du.mean():.3f}+-{2 * du.std(ddof=1) / np.sqrt(len(x)):.3f}")
print("== E25: state readout vs new decoder, r across conditions (arch, V), base")
for gn in (0.0, 4.0):
    X = [(np.mean([r["read_trained"] for r in v]), np.mean([r["redec_trained"] for r in v])) for k, v in g.items() if k[0] == gn and k[3] == "base"]
    if len(X) > 2:
        print(gn, len(X), "r", round(float(np.corrcoef(np.array(X).T)[0, 1]), 3))
print("== E21b mean flip by model and layer; patch check at readout_16")
r = load("e21b")
print(collections.Counter(x["model"].split("/")[-1] + f"_s{x['seed']}" for x in r if x["kind"] == "whole"))
K = ["whole", "readout_16", "readout_64", "readout_128", "diff_16", "diff_64", "diff_256", "pc_64", "pc_256", "random_64", "random_256", "allbut_readout_16", "allbut_diff_64", "samefig_diff_64"]
c = collections.defaultdict(lambda: collections.defaultdict(list))
for x in r:
    c[(x["model"].split("/")[-1], x["layer"])][x["kind"]].append(x)
for k in sorted(c):
    print(k, "src_flip", round(float(np.mean([x["src_flip"] for x in c[k]["whole"]])), 2), {q: round(float(np.mean([x["flip"] for x in c[k][q]])), 2) for q in K})
    ro = c[k]["readout_16"]
    print("    readout_16 patch reads: own figure", round(float(np.mean([x["patch_rec"] for x in ro])), 3), "source figure", round(float(np.mean([x["patch_rec_src"] for x in ro])), 3), "shift/norm", round(float(np.mean([x["shift"] / x["norm"] for x in ro])), 3), "diff_64", round(float(np.mean([x["shift"] / x["norm"] for x in c[k]["diff_64"]])), 3))
