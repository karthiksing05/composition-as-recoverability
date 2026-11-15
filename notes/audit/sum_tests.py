"""Score E20 calibration, E21, E23, E24, E26 against notes/audit/PREDICTIONS_E20.md."""
import glob, json, collections, os
import numpy as np
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/"


def load(d):
    out = []
    for f in glob.glob(R + d + "/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        out += r if isinstance(r, list) else [r]
    return out


def short(m):
    return m.split("/")[-1]


print("== E20cal2: V=8 and V=64, gain/layers/m/steps")
g = collections.defaultdict(dict)
for r in load("e20cal2"):
    g[(r["gain"], r["layers"], r["m"], r["steps"])][r["V"]] = r
for k in sorted(g):
    a, b = g[k].get(8), g[k].get(64)
    if a and b:
        print(k, f"input {a['read_input']:.3f} | V8 own {a['own_heldout']:.3f} read {a['read_trained']:.3f} untr {a['read_untrained']:.3f} redec {a['redec_trained']:.3f}/{a['redec_untrained']:.3f} | V64 own {b['own_heldout']:.3f} read {b['read_trained']:.3f} newpairs {b['own_newpairs']:.3f}")

print("== E21 picked layer: mean over rel and seed of to_answer")
r21 = [r for r in load("e21") if r["picked"]]
print(" seeds per model", collections.Counter(short(r["model"]) for r in r21 if r["kind"] == "noop"))
ms = sorted({r["model"] for r in r21})
for m in ms:
    rr = [r for r in r21 if r["model"] == m]
    c = collections.defaultdict(list)
    for r in rr:
        c[(r["source"], r["kind"])].append(r["to_answer"])
    kept = collections.defaultdict(list)
    for r in rr:
        if r["source"] == "none":
            kept[r["kind"]].append(r["kept_correct"])
    print(short(m), "orig", round(float(np.mean([r["orig_acc"] for r in rr])), 3), "noop_err", round(max(r["noop_err"] or 0 for r in rr), 3))
    for s in ("reversed", "swapped", "samefig"):
        print("   ", s, {k: round(float(np.mean(c[(s, k)])), 3) for k in ("whole", "readout", "pc", "random")})
    print("    ablate kept", {k: round(float(np.mean(kept[k])), 3) for k in ("noop", "ablate_readout", "ablate_random")}, "write to_other", round(float(np.mean(c[("none", "write")])), 3))
print("== E21 whole-state flip by layer, reversed source")
a21 = load("e21")
for m in ms:
    c = collections.defaultdict(list)
    for r in a21:
        if r["model"] == m and r["source"] == "reversed" and r["kind"] == "whole":
            c[r["layer"]].append(r["to_answer"])
    print(short(m), {k: round(float(np.mean(v)), 2) for k, v in sorted(c.items())})

print("== E23 picked: real vs nonce")
r23 = [r for r in load("e23") if r["picked"]]
for m in sorted({r["model"] for r in r23}):
    for t in ("real", "nonce"):
        rr = [r for r in r23 if r["model"] == m and r["test"] == t]
        kap = np.mean([(r["model_rec"] - r["ref_sum_rec"]) / max(r["ref_concat_rec"] - r["ref_sum_rec"], 1e-6) for r in rr])
        print(short(m), t, "answers", round(float(np.mean([r["answers"] for r in rr])), 3), "xfer", round(float(np.mean([r["xfer_conv"] for r in rr])), 3), "kappa", round(float(kap), 3), "tokens", round(rr[0]["tokens_real"] if t == "real" else rr[0]["tokens_nonce"], 2), "n", len(rr))

print("== E24 at the layer and code with best validation for the model state (kappa)")
r24 = load("e24")
for m in sorted({r["model"] for r in r24}):
    c = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in r24:
        if r["model"] == m:
            c[(r["layer"], r["code"])][r["system"]].append(r)
    best, bv = None, -9
    for k, v in c.items():
        val = np.mean([(x["rec_val_role1"] + x["rec_val_role2"]) / 2 for x in v["model"]]) if "rec_val_role1" in v["model"][0] else np.mean([(x["id_val_role1"] + x["id_val_role2"]) / 2 for x in v["model"]])
        if val > bv:
            best, bv = k, val
    rec = lambda s: float(np.mean([(x["rec_fresh_role1"] + x["rec_fresh_role2"]) / 2 for x in c[best][s]]))
    rc, rs = rec("ref_concat"), rec("ref_sum")
    print(short(m), best, {s: round((rec(s) - rs) / (rc - rs), 3) for s in ("model", "input_concat", "input_avg")}, "last layer", end=" ")
    L = sorted({k[0] for k in c}, key=lambda s: int(s[1:]))[-1]
    k2 = (L, best[1])
    rc2 = float(np.mean([(x["rec_fresh_role1"] + x["rec_fresh_role2"]) / 2 for x in c[k2]["ref_concat"]])); rs2 = float(np.mean([(x["rec_fresh_role1"] + x["rec_fresh_role2"]) / 2 for x in c[k2]["ref_sum"]]))
    print({s: round((float(np.mean([(x["rec_fresh_role1"] + x["rec_fresh_role2"]) / 2 for x in c[k2][s]])) - rs2) / (rc2 - rs2), 3) for s in ("model", "input_concat", "input_avg")})

print("== E26: OLMo-7B readouts on converse (E19)")
r19 = [r for r in load("e19") if "OLMo-2-1124-7B" in r["model"]]
c = collections.defaultdict(list)
for r in r19:
    c[(r["state"], r["layer"], r["code"])].append(r)
for k in sorted(c, key=str):
    if k[0] != "mean":
        print(k, "n", len(c[k]), {a: round(float(np.mean([x[a] for x in c[k]])), 3) for a in ("ridge_conv", "kernel_conv", "mlp_conv", "kernel_oracle_conv", "mlp_oracle_conv")})
