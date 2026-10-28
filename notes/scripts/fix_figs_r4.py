d = "/coc/flash10/ksingara3/compositionality-theory/experiments/"
s = open(d + "fig_concepts.py", encoding="utf-8").read()
R = [
 ('''def box(ax, x, y, w, h, title, body, color, dashed=False, fs=7.5):''', '''def box(ax, x, y, w, h, title, body, color, dashed=False, fs=7.5, ty=0.68, by=0.3):'''),
 ('''    ax.text(x + w / 2, y + h * 0.68, title,''', '''    ax.text(x + w / 2, y + h * ty, title,'''),
 ('''    ax.text(x + w / 2, y + h * 0.3, body,''', '''    ax.text(x + w / 2, y + h * by, body,'''),
 ('''"orthogonal: stores a\\ncomposite as a unit", GREY, dashed=True)''', '''"orthogonal: stores a\\ncomposite as a unit", GREY, dashed=True, ty=0.78, by=0.3)'''),
 ('''"parts not recoverable, even\\nfrom a vocabulary, or roles collide"''', '''"fresh parts not recoverable;\\nroles collide when symmetric"'''),
 ('''"parts recoverable, including\\nparts never seen before"''', '''"parts recoverable, including parts\\nthe readout was not fitted on"'''),
 ('''"symmetric: $x+y$, $x\\\\odot y$;\\nhead\\u2013modifier, prototypes;\\nconceptual blending"''', '''"symmetric: $x+y$, $x\\\\odot y$;\\nprototypes;\\nconceptual blending"'''),
 ('''"HRR, MAP, $x+0.5y$,\\nprojections with $m<kd$\\n(fresh-part ceiling $\\\\mathrm{rank}/kd$)"''',
  '''"HRR, MAP, head\\u2013modifier $x+\\\\alpha y$,\\nprojections with $m<kd$\\n(fresh-part ceiling $\\\\mathrm{rank}/kd$)"'''),
 ('''"tensor products, concatenation,\\nsuperposition in disjoint regions,\\nslots and templates"''', '''"tensor products, concatenation,\\nsuperposition in disjoint regions"'''),
 ('''"Familiar parts: decodable whenever the relation is injective on the vocabulary\\n"
        "Systematicity and productivity are contingent on the data", ha="center", va="center", fontsize=6.6''',
  '''"Familiar parts: decodable when the relation is injective on the vocabulary\\n"
        "Systematicity and productivity are contingent on the data", ha="center", va="center", fontsize=6.2'''),
 ('''"Fresh parts: recoverable\\nSystematicity is necessary"''', '''"Fresh parts: recoverable\\nReversals fixed by the state"'''),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(d + "fig_concepts.py", "w", encoding="utf-8").write(s)
s = open(d + "fig_e3e4.py", encoding="utf-8").read()
R = [
 ('''def rand_select(rnd):  # layer and code of a random-weight control, chosen on its validation nouns
    rsel = rnd.groupby(["model", "layer", "code"])["rec_val"].mean().reset_index()
    rsel = rsel.loc[rsel.groupby("model").rec_val.idxmax()]
    return {r.model[5:]: rnd[(rnd.model == r.model) & (rnd.layer == r.layer) & (rnd.code == r.code)].rec_fresh.mean()
            for r in rsel.itertuples()}''',
  '''def rand_select(rnd, k="rec_fresh"):  # layer and code of a random-weight control, chosen on its validation nouns
    rsel = rnd.groupby(["model", "layer", "code"])["rec_val"].mean().reset_index()
    rsel = rsel.loc[rsel.groupby("model").rec_val.idxmax()]
    return {r.model[5:]: rnd[(rnd.model == r.model) & (rnd.layer == r.layer) & (rnd.code == r.code)][k].mean()
            for r in rsel.itertuples()}'''),
 ('''rand_trained = rand_select(rnd[rnd.codes_from != rnd.model])  # scored on the trained model's part codes''',
  '''rand_trained = rand_select(rnd[rnd.codes_from != rnd.model])  # scored on the trained model's part codes
rand_role = rand_select(rnd[rnd.codes_from == rnd.model], "role_fresh")
rand_id = rand_select(rnd[rnd.codes_from == rnd.model], "id_fresh")'''),
 ('''    ax.scatter([i + 0.18], [g.loc["ref_sum", "id_fresh"]], marker="_", s=70, color=DGREY, lw=1.6, zorder=3)''',
  '''    ax.scatter([i + 0.18], [g.loc["ref_sum", "id_fresh"]], marker="_", s=70, color=DGREY, lw=1.6, zorder=3)
    ax.scatter([i - 0.18, i + 0.18], [rand_role[m], rand_id[m]], marker="x", s=14, color="k", lw=0.9, zorder=4)'''),
 ('''ax.legend([Patch(color=GREEN), Patch(color=ORANGE), Line2D([], [], color=DGREY, lw=1.6)],
          ["role assigned", "identified (chance 0.01)", "sum ref., identified"]''',
  '''ax.legend([Patch(color=GREEN), Patch(color=ORANGE), Line2D([], [], color=DGREY, lw=1.6), Line2D([], [], color="k", marker="x", lw=0, ms=4)],
          ["role assigned", "identified (chance 0.01)", "sum ref., identified", "random, own codes"]'''),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(d + "fig_e3e4.py", "w", encoding="utf-8").write(s)
print("figs ok")
