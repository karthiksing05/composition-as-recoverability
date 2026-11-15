import glob, json, collections
import numpy as np
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/"
rows = []
for f in glob.glob(R + "e20/*.json"):
    r = json.load(open(f, encoding="utf-8"))
    rows += r if isinstance(r, list) else [r]
g = {}
for r in rows:
    g[(r["gain"], r["arch"], r["V"], r["arm"], r["seed"])] = r
print("n", len(rows))
A = ("mlp", "cnn", "slot", "tf")


def stat(gain, arch, V, f):
    v = [f(g[(gain, arch, V, k[3], k[4])], k[4]) for k in g if k[0] == gain and k[1] == arch and k[2] == V and k[3] == "base"]
    v = np.array([x for x in v if x is not None])
    return f"{v.mean():+.3f}+-{2 * v.std(ddof=1) / np.sqrt(len(v)):.3f}(n{len(v)})"


def pair(gain, arch, V, arm, c1, c2=None):
    out = []
    for s in range(5):
        a, b = g.get((gain, arch, V, arm, s)), g.get((gain, arch, V, "base", s))
        if a and b:
            out.append(a[c1] - b[c2 or c1])
    v = np.array(out)
    return f"{v.mean():+.3f}+-{2 * v.std(ddof=1) / np.sqrt(len(v)):.3f}"


for gain in (0.0, 4.0):
    print(f"=== gain {gain}")
    for a in A:
        m = lambda V, arm, c: np.mean([r[c] for k, r in g.items() if k[:4] == (gain, a, V, arm)])
        print(a, "V8 own %.3f read %.3f untr %.3f input %.3f redec %.3f redecU %.3f | V64 own %.3f read %.3f redec %.3f" % (m(8, "base", "own_heldout"), m(8, "base", "read_trained"), m(8, "base", "read_untrained"), m(8, "base", "read_input"), m(8, "base", "redec_trained"), m(8, "base", "redec_untrained"), m(64, "base", "own_heldout"), m(64, "base", "read_trained"), m(64, "base", "redec_trained")))
        print("   tr-in", stat(gain, a, 8, lambda r, s: r["read_trained"] - r["read_input"]), "tr-un", stat(gain, a, 8, lambda r, s: r["read_trained"] - r["read_untrained"]))
        for arm in ("auxvocab", "auxpool"):
            print("  ", arm, "own %.3f read %.3f through %.3f" % (m(8, arm, "own_heldout"), m(8, arm, "read_trained"), m(8, arm, "through")), "| dread", pair(gain, a, 8, arm, "read_trained"), "down", pair(gain, a, 8, arm, "own_heldout"), "through-own", pair(gain, a, 8, arm, "through", "own_heldout"), "through-redec", pair(gain, a, 8, arm, "through", "redec_trained"))
    X = [(np.mean([r["read_trained"] for k, r in g.items() if k[:4] == (gain, a, V, "base")]), np.mean([r["redec_trained"] for k, r in g.items() if k[:4] == (gain, a, V, "base")])) for a in A for V in (8, 16, 32, 64)]
    print("E25 r", round(float(np.corrcoef(np.array(X).T)[0, 1]), 3), "n", len(X))
    B = [(r["read_trained"], r["redec_trained"]) for k, r in g.items() if k[0] == gain and k[3] == "base"]
    print("E25 r runs", round(float(np.corrcoef(np.array(B).T)[0, 1]), 3), len(B))
    for V in (8, 16, 32, 64):
        C = [(r["read_trained"], r["redec_trained"]) for k, r in g.items() if k[0] == gain and k[3] == "base" and k[2] == V]
        print("  within V", V, round(float(np.corrcoef(np.array(C).T)[0, 1]), 3), len(C))
print("=== E21b at picked layers")
rr = []
for f in glob.glob(R + "e21b/*.json"):
    rr += json.load(open(f, encoding="utf-8"))
K = ["whole", "readout_16", "readout_64", "readout_128", "diff_16", "diff_64", "diff_256", "pc_16", "pc_64", "pc_256", "random_16", "random_64", "random_256", "allbut_readout_16", "allbut_diff_64", "samefig_diff_64"]
for m, L in (("phi-4", 30), ("Qwen3-8B", 27), ("OLMo-2-1124-13B", 30), ("OLMo-2-1124-7B", 24), ("OLMo-2-1124-7B", 32)):
    x = [r for r in rr if r["model"].endswith(m) and r["layer"] == L]
    print(m, L, "seeds", sorted({r["seed"] for r in x}), "orig", round(float(np.mean([r["orig_acc"] for r in x])), 3), {k: round(float(np.mean([r["flip"] for r in x if r["kind"] == k])), 2) for k in K})
