import pymupdf
for n in ("fig_beh_toy", "fig_beh_real", "fig_rep_toy", "fig_rep_real", "fig_rep_codes", "fig_rep_roles", "fig_parts", "fig_protocol"):
    pymupdf.open(f"/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/figures/{n}.pdf")[0].get_pixmap(dpi=130).save(f"/coc/flash10/ksingara3/compositionality-theory/notes/audit/figs/{n}.png")
print("ok")
