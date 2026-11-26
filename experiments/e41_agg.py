"""Aggregate E41: per model and phrasing, contingency of model answer x probe nearest noun."""
import glob, json, os, sys
import numpy as np
D = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/e41"
if len(sys.argv) > 1:
    D = sys.argv[1]
by = {}
for fn in sorted(glob.glob(os.path.join(D, "*_s?.json"))):
    r = json.load(open(fn, encoding="utf-8"))
    by.setdefault(r["model"], []).extend(r["items"])
out = {}
def stats(its):
    n = len(its)
    c = {(a, p): sum(1 for t in its if t["ans"] == a and t["probe"] == p) for a in ("fig", "gnd") for p in ("fig", "gnd")}
    right, wrong = c["fig", "fig"] + c["fig", "gnd"], c["gnd", "fig"] + c["gnd", "gnd"]
    f = lambda a, b: a / b if b else None
    return dict(n=n, acc=f(right, n), transfer=f(c["fig", "fig"] + c["gnd", "fig"], n), n_wrong=wrong, n_right=right,
                wrong_probe_follows_answer=f(c["gnd", "gnd"], wrong), wrong_probe_returns_figure=f(c["gnd", "fig"], wrong),
                right_probe_returns_figure=f(c["fig", "fig"], right), n_greedy_neither=sum(t["greedy"] == "neither" for t in its),
                contingency={"ans_fig_probe_fig": c["fig", "fig"], "ans_fig_probe_gnd": c["fig", "gnd"],
                             "ans_gnd_probe_fig": c["gnd", "fig"], "ans_gnd_probe_gnd": c["gnd", "gnd"]})
for m, its in by.items():
    out[m] = {"seeds": sorted({t["seed"] for t in its}), "all": stats(its)}
    for ph in ("canonical", "converse"):
        out[m][ph] = stats([t for t in its if t["phrasing"] == ph])
    out[m]["per_seed_all"] = {s: stats([t for t in its if t["seed"] == s]) for s in out[m]["seeds"]}
    out[m]["per_rel_all"] = {r: stats([t for t in its if t["rel"] == r]) for r in sorted({t["rel"] for t in its})}
json.dump(out, open(os.path.join(D, "summary.json"), "w", encoding="utf-8"), indent=1)
fm = lambda v: "  -  " if v is None else f"{v:.3f}"
print(f"{'model':26s}{'phr':10s}{'n':>6s}{'acc':>7s}{'xfer':>7s}{'nWr':>6s}{'wr:ans':>8s}{'wr:fig':>8s}{'nRt':>6s}{'rt:fig':>8s}{'neith':>6s}  [fig/fig fig/gnd gnd/fig gnd/gnd]")
for m, o in out.items():
    for ph in ("canonical", "converse", "all"):
        s = o[ph]; c = s["contingency"]
        print(f"{m:26s}{ph:10s}{s['n']:6d}{fm(s['acc']):>7s}{fm(s['transfer']):>7s}{s['n_wrong']:6d}{fm(s['wrong_probe_follows_answer']):>8s}"
              f"{fm(s['wrong_probe_returns_figure']):>8s}{s['n_right']:6d}{fm(s['right_probe_returns_figure']):>8s}{s['n_greedy_neither']:6d}  "
              f"{c['ans_fig_probe_fig']} {c['ans_fig_probe_gnd']} {c['ans_gnd_probe_fig']} {c['ans_gnd_probe_gnd']}")
