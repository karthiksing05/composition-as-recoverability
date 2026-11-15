import json, collections
import numpy as np
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/"
r21 = json.load(open(R + "smoke21/gpt2-large_s90.json", encoding="utf-8"))
print("E21 rows", len(r21), "orig_acc", np.mean([r["orig_acc"] for r in r21]), "noop_err", max(r["noop_err"] for r in r21 if r["kind"] == "noop"))
g = collections.defaultdict(list)
for r in r21:
    if r["picked"]:
        g[(r["source"], r["kind"])].append((r["to_answer"], r.get("src_acc") or np.nan))
for k in sorted(g):
    print(" ", k, np.round(np.mean(g[k], 0), 3))
r23 = json.load(open(R + "smoke23/gpt2-large_s90.json", encoding="utf-8"))
print("E23 rows", len(r23), "tokens", r23[0]["tokens_real"], r23[0]["tokens_nonce"])
g = collections.defaultdict(list)
for r in r23:
    if r["picked"]:
        g[r["test"]].append((r["answers"], r["answers_can"], r["answers_conv"], r["xfer_conv"], r["model_rec"], r["ref_concat_rec"], r["ref_sum_rec"]))
for k in g:
    print(" ", k, "answers/can/conv/xfer/rec/concat/sum", np.round(np.mean(g[k], 0), 3))
r24 = json.load(open(R + "smoke24/gpt2_s90.json", encoding="utf-8"))
print("E24 rows", len(r24), sorted(r24[0].keys())[:14])
g = collections.defaultdict(list)
for r in r24:
    g[(r["layer"], r["code"], r["system"])].append((r["rec_fresh_role1"] + r["rec_fresh_role2"]) / 2)
for k in sorted(g, key=str):
    if k[0] in ("L6", "L12") :
        print(" ", k, round(float(np.mean(g[k])), 3))
