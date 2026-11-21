"""Page check for the ICML build: page count, where the body ends, undefined refs, and PNG renders.
usage: python icml_check.py PDF OUTDIR [first_n_pages]"""
import sys, os, re, fitz

pdf, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
os.makedirs(out, exist_ok=True)
doc = fitz.open(pdf)
print("pages", doc.page_count)
for i, p in enumerate(doc):
    t = p.get_text()
    for key in ("Impact Statement", "AI Use Statement", "References", "Conclusion and Limitations"):
        if key in t:
            print(f"{key!r} on page {i + 1}")
    if "??" in t:
        print(f"'??' on page {i + 1}:", [m.start() for m in re.finditer(r"\?\?", t)][:5])
    if i < n:
        p.get_pixmap(dpi=100).save(f"{out}/p{i + 1:02d}.png")
fonts = set()
for p in doc:
    for f in p.get_fonts():
        fonts.add((f[1], f[2]))
print("font types", sorted({t for t, _ in fonts}))
print("type3 fonts", sorted({nm for t, nm in fonts if t == "Type3"})[:10])
