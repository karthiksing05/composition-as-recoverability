"""Render selected PDF pages to PNG. Usage: python -I this.py file.pdf out_dir dpi page[,page...] (1-indexed)"""
import sys, os
import pymupdf

path, out, dpi, pages = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
doc = pymupdf.open(path)
for p in pages.split(","):
    n = int(p)
    f = os.path.join(out, f"p{n}.png")
    doc[n - 1].get_pixmap(dpi=int(dpi)).save(f)
    print(f)
