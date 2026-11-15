"""Uncertainty of the answer/transfer correlation: resampling models, leaving one out, and within prompt."""
import sys, json
import numpy as np
sys.path.insert(0, ".")
sys.argv = ["fig_flow.py", "none"]
src = open("fig_flow.py", encoding="utf-8").read().split('mode = sys.argv[1]')[0]
exec(src)
pts = {k: v for k, v in prompt.items()}
ms = sorted({m for m, _ in pts})
A = lambda keep: np.array([pts[k] for k in pts if k[0] in keep])
print("points", len(pts), "r", round(np.corrcoef(A(ms).T)[0, 1], 3))
g = np.random.default_rng(0)
B = []
for _ in range(10000):
    s = g.choice(ms, len(ms))
    X = np.concatenate([A([m]) for m in s])
    if X[:, 0].std() > 1e-6 and X[:, 1].std() > 1e-6:
        B.append(np.corrcoef(X.T)[0, 1])
print("model bootstrap", np.round(np.percentile(B, [2.5, 50, 97.5]), 3))
for m in ms:
    print(" without", m, round(np.corrcoef(A([x for x in ms if x != m]).T)[0, 1], 3))
print("meaning models only", round(np.corrcoef(A(["phi-4", "Qwen3-8B", "OLMo-2-1124-13B", "OLMo-2-1124-7B"]).T)[0, 1], 3))
print("word-order models only", round(np.corrcoef(A(["gpt2-large", "pythia-1.4b", "pythia-12b"]).T)[0, 1], 3))
# within model: does a prompt that helps answers also help transfer
d = []
for m in ms:
    X = A([m]); d.append(X - X.mean(0))
d = np.concatenate(d)
print("within model", round(np.corrcoef(d.T)[0, 1], 3))
for k, v in sorted(pts.items()):
    print(" ", k, np.round(v, 3))
