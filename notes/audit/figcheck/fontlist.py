import sys, fitz
doc = fitz.open(sys.argv[1]); seen = {}
for i, p in enumerate(doc):
    for f in p.get_fonts():
        seen.setdefault((f[3], f[1]), i + 1)
for (name, typ), pg in sorted(seen.items(), key=lambda x: x[1]):
    print(pg, typ, name)
