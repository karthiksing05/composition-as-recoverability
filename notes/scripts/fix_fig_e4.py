p = "/coc/flash10/ksingara3/compositionality-theory/experiments/fig_e3e4.py"
s = open(p, encoding="utf-8").read()
R = [
 ('''e4all = pd.DataFrame([r for f in glob.glob("results/e4c/*.json") for r in json.load(open(f))])''',
  '''# whitened part codes (each of the 16 components counts equally); see agg_r4.py e4w
e4all = pd.DataFrame([r for f in glob.glob("results/e4w/*.json") for r in json.load(open(f))])'''),
 ('''e4 = e4all[~e4all.model.str.startswith("rand-")]
rnd = e4all[e4all.model.str.startswith("rand-") & (e4all.system == "model")]
rsel = rnd.groupby(["model", "layer", "code"])["rec_val"].mean().reset_index()
rsel = rsel.loc[rsel.groupby("model").rec_val.idxmax()]
rand_fresh = {r.model[5:]: rnd[(rnd.model == r.model) & (rnd.layer == r.layer) & (rnd.code == r.code)].rec_fresh.mean()
              for r in rsel.itertuples()}''',
  '''e4 = e4all[~e4all.model.str.startswith("rand-")]


def rand_select(rnd):  # layer and code of a random-weight control, chosen on its validation nouns
    rsel = rnd.groupby(["model", "layer", "code"])["rec_val"].mean().reset_index()
    rsel = rsel.loc[rsel.groupby("model").rec_val.idxmax()]
    return {r.model[5:]: rnd[(rnd.model == r.model) & (rnd.layer == r.layer) & (rnd.code == r.code)].rec_fresh.mean()
            for r in rsel.itertuples()}


rnd = e4all[e4all.model.str.startswith("rand-") & (e4all.system == "model")]
rand_fresh = rand_select(rnd[rnd.codes_from == rnd.model])  # scored on its own part codes
rand_trained = rand_select(rnd[rnd.codes_from != rnd.model])  # scored on the trained model's part codes'''),
 ('''    ax.scatter([i + 0.18], [rand_fresh[m]], marker="x", s=14, color="k", lw=0.9, zorder=4)''',
  '''    if m in rand_fresh:
        ax.scatter([i + 0.18], [rand_fresh[m]], marker="x", s=14, color="k", lw=0.9, zorder=4)
    ax.scatter([i + 0.18], [max(rand_trained[m], 0.0)], marker="o", s=6, color="k", lw=0, zorder=4)'''),
 ('''           Line2D([], [], color="k", marker="x", lw=0, ms=4)],
          ["new pairs, fitting nouns", "fresh nouns", "concat. reference", "sum reference", "random weights, fresh"],''',
  '''           Line2D([], [], color="k", marker="x", lw=0, ms=4), Line2D([], [], color="k", marker="o", lw=0, ms=2.5)],
          ["new pairs, fitting nouns", "fresh nouns", "concat. reference", "sum reference", "random weights, own codes",
           "random weights, trained codes"],'''),
 ('''ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("recovery of nouns ($R^2$)")''',
  '''ax.set_ylim(0, 1.5); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("recovery of nouns ($R^2$)")'''),
 ('''print("best layers", best)''', '''print("best layers", best)
print("random, own codes", {k: round(v, 3) for k, v in rand_fresh.items()})
print("random, trained codes", {k: round(v, 3) for k, v in rand_trained.items()})'''),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
