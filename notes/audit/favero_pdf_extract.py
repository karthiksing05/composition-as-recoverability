"""Extract per-page text from PDFs to .txt beside each PDF. Usage: python -I this.py a.pdf [b.pdf ...] [out_dir]
Each page is preceded by a marker line: ===== PDF PAGE n =====
If the last argument is a directory, the txt goes there (used for read-only source PDFs)."""
import sys, os, importlib

libs = {}
for name in ("fitz", "pypdf", "pdfminer"):
    try:
        libs[name] = importlib.import_module(name)
    except Exception as e:
        print(name, "missing:", e)
print("available:", list(libs))

args = sys.argv[1:]
out_dir = None
if args and os.path.isdir(args[-1]):
    out_dir = args.pop()

for path in args:
    base = os.path.basename(path)[:-4] + ".txt"
    out = os.path.join(out_dir or os.path.dirname(path), base)
    pages = []
    if "fitz" in libs:
        doc = libs["fitz"].open(path)
        pages = [p.get_text() for p in doc]
        used = "fitz"
    elif "pypdf" in libs:
        r = libs["pypdf"].PdfReader(path)
        pages = [p.extract_text() or "" for p in r.pages]
        used = "pypdf"
    else:
        from pdfminer.high_level import extract_text
        pages = extract_text(path).split("\f")
        used = "pdfminer"
    with open(out, "w", encoding="utf-8") as f:
        for i, t in enumerate(pages, 1):
            f.write(f"\n===== PDF PAGE {i} =====\n{t}")
    print(base, used, "pages", len(pages), "chars", sum(map(len, pages)), "->", out)
