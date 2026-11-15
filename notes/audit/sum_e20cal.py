import glob, json, collections
import numpy as np
rows = [json.load(open(f, encoding="utf-8")) for f in glob.glob("experiments/results/e20cal/*.json")]
rows = [r[0] if isinstance(r, list) else r for r in rows]
print(len(rows), "files; keys", sorted(rows[0].keys()))
g = collections.defaultdict(list)
for r in rows:
    g[(r["gain"], r["layers"], r["V"])].append(r)
F = ["own_heldout", "own_newpairs", "read_trained", "read_untrained", "read_input", "read_vocab", "redec_trained", "redec_untrained", "final_loss"]
print("gain lay V n | " + " ".join(f"{k[:11]:>11}" for k in F))
for k in sorted(g):
    print(*k, len(g[k]), "| " + " ".join(f"{np.mean([x.get(c, np.nan) if x.get(c) is not None else np.nan for x in g[k]]):11.3f}" for c in F))
