"""Figure: trained networks on glyph composites (E2)."""
import json, glob
import numpy as np, pandas as pd
from figstyle import *
import e2_glyph as E


def load(d):
    df = pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{d}/*.json")])
    for s in ("train", "heldout", "strong", "fresh"):
        df[f"b_{s}"] = df[[f"beh_{s}_r{i}{o}" for i in "01" for o in "01"]].mean(1)
        df[f"b_{s}_in1"] = df[[f"beh_{s}_r1{o}" for o in "01"]].mean(1)
        df[f"b_{s}_in0"] = df[[f"beh_{s}_r0{o}" for o in "01"]].mean(1)
    df["p_fresh"] = df[[f"probe_pix_fresh_r{i}_role{k}" for i in "01" for k in "12"]].mean(1)
    df["p_held"] = df[[f"probe_pix_heldout_r{i}_role{k}" for i in "01" for k in "12"]].mean(1)
    df["p_strong"] = df[[f"probe_pix_strong_r{i}_role2" for i in "01"]].mean(1)
    return df


df = load("results/e2")
occ = load("results/e2occ")
ARCH = {"slot": ("role-filler decoder", BLUE, "o"), "mlp": ("MLP", ORANGE, "s"),
        "cnn": ("CNN", GREEN, "^"), "tf": ("transformer", PURPLE, "D")}
LOG2 = dict(base=2)

fig = plt.figure(figsize=(6.5, 4.5))
gs = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.72)
axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

# a: the domain
ax = axs[0]
w = E.World(0)
rng = np.random.default_rng(3)
za, zb = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
imgs = [w.render(za, zb, 0)[0]]
E.SECOND["rel"] = "over"; imgs.append(w.render(za, zb, 1)[0])
E.SECOND["rel"] = "occ"; imgs.append(w.render(za, zb, 1)[0])
canvas = np.full((3 * 8 + 2 * 2, 16), np.nan)
for i, im in enumerate(imgs):
    canvas[i * 10:i * 10 + 8] = im / im.max()
ax.imshow(canvas, cmap="Greys", interpolation="nearest", aspect="equal")
for i, t in enumerate(["BESIDE", "OVER", "OCCLUDE"]):
    ax.text(-0.8, i * 10 + 3.5, t, ha="right", va="center", fontsize=6.5, color=DGREY)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
panel(ax, "a", "Composites")


def curve(ax, d, key, xkey, ls="-", marker=True):
    for a, (name, c, mk) in ARCH.items():
        t = d[d.arch == a].groupby(xkey)[key].agg(["mean", "sem"])
        band(ax, t.index, t["mean"].clip(lower=-0.1), t["sem"], c, ls=ls, marker=mk if marker else None, ms=3)


def log2axis(ax, ticks):
    ax.set_xscale("log", **LOG2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


# b: behaviour vs bottleneck width (V = 64)
ax = axs[1]
d = df[df.V == 64]
curve(ax, d, "b_heldout", "m", ls="--", marker=False)
curve(ax, d, "b_fresh", "m")
ax.axvline(16, color=LGREY, lw=0.6, ls=":"); ax.text(17, 0.02, "$m=kq$", fontsize=6.3, color=DGREY)
log2axis(ax, [8, 16, 32, 64])
ax.set_ylim(-0.02, 1.03); ax.set_xlabel("bottleneck width $m$"); ax.set_ylabel("output $R^2$")
panel(ax, "b", "Bottleneck width")

# c: behaviour vs vocabulary (m = 64)
ax = axs[2]
d = df[df.m == 64]
curve(ax, d, "b_heldout", "V", ls="--", marker=False)
curve(ax, d, "b_fresh", "V")
log2axis(ax, [8, 16, 32, 64])
ax.set_ylim(-0.02, 1.03); ax.set_xlabel("training parts $V$"); ax.set_ylabel("output $R^2$")
panel(ax, "c", "Part diversity")

# d: representation predicts behaviour on fresh parts
ax = axs[3]
c_ = lambda x: x.clip(lower=-0.2)
for a, (name, c, mk) in ARCH.items():
    s = df[df.arch == a]
    ax.scatter(c_(s.p_fresh), c_(s.b_heldout), s=4, color=LGREY, marker=mk, lw=0, alpha=0.6, zorder=0)
    ax.scatter(c_(s.p_fresh), c_(s.b_fresh), s=5, color=c, marker=mk, lw=0, alpha=0.7, label=name)
r_f = np.corrcoef(c_(df.p_fresh), c_(df.b_fresh))[0, 1]
r_h = np.corrcoef(c_(df.p_fresh), c_(df.b_heldout))[0, 1]
ax.text(0.97, 0.03, f"fresh parts: $r$={r_f:.2f}\nheld-out pairs (grey): $r$={r_h:.2f}", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=6.3, color=DGREY)
ax.set_xlim(-0.2, 1); ax.set_ylim(-0.2, 1.03)
ax.set_xlabel("recovery of fresh parts from $h$ ($R^2$)"); ax.set_ylabel("output $R^2$")
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), markerscale=2.2, fontsize=7)
panel(ax, "d", "Recovery predicts transfer")

# e: relations (m = 64, V = 64), by input relation
ax = axs[4]
rows = []
for a in ARCH:
    s = df[(df.arch == a) & (df.m == 64) & (df.V == 64)]
    o = occ[(occ.arch == a) & (occ.m == 64) & (occ.V == 64)]
    for lab, x in (("BESIDE", s.b_fresh_in0), ("OVER", s.b_fresh_in1), ("OCCLUDE", o.b_fresh_in1)):
        rows.append((a, lab, x.mean(), x.sem()))
t = pd.DataFrame(rows, columns=["arch", "rel", "mean", "sem"])
for j, a in enumerate(ARCH):
    s = t[t.arch == a]
    ax.bar(np.arange(3) + (j - 1.5) * 0.2, s["mean"], 0.19, yerr=s["sem"], color=ARCH[a][1], error_kw=dict(lw=0.5))
ax.set_xticks(range(3)); ax.set_xticklabels(["BESIDE", "OVER", "OCCLUDE"], fontsize=6.5)
ax.set_ylim(0, 1.03); ax.set_ylabel("output $R^2$ on fresh parts")
panel(ax, "e", "Relations")

# f: contingency: spread across seeds of new-role (strong) behaviour
ax = axs[5]
for a, (name, c, mk) in ARCH.items():
    t2 = df[(df.arch == a) & (df.m >= 32)].groupby("V")["b_strong"].std()
    ax.plot(t2.index, t2.values, color=c, marker=mk, ms=3)
log2axis(ax, [8, 16, 32, 64])
ax.set_xlabel("training parts $V$"); ax.set_ylabel("s.d. over seeds")
panel(ax, "f", "Seed spread, new roles")

fig.savefig(f"{FIGDIR}/fig_networks.pdf")
fig.savefig("/coc/flash10/ksingara3/compositionality-theory/notes/fig_networks.png", dpi=200)
print("n runs", len(df), len(occ), "r_fresh", round(r_f, 3), "r_held", round(r_h, 3))
pd.set_option("display.width", 250)
print(df.groupby(["arch", "m", "V"])[["b_heldout", "b_strong", "b_fresh", "p_held", "p_strong", "p_fresh"]].agg("mean").round(2).to_string())
print(t.round(2).to_string())
print(occ.groupby(["arch", "m", "V"])[["b_heldout_in1", "b_strong_in1", "b_fresh_in1", "inter_fresh_r1", "inter_input_fresh_r1"]].mean().round(3).to_string())
print(df.groupby(["arch"])[["inter_fresh_r1", "rolefac_fresh_r1", "rolefac_fam_r1", "leak_fresh_r1", "leak_heldout_r1"]].mean().round(3).to_string())
