"""Report word runs that the paper shares with reference texts (plagiarism check).

Usage: overlap.py PAPER.tex N REF1 [REF2 ...]   (refs: .txt or .tex)
Prints every maximal shared run of at least N words, per reference."""
import re, sys


def words(path):
    s = open(path, encoding="utf-8", errors="ignore").read()
    if path.endswith(".tex"):
        s = re.sub(r"(?<!\\)%.*", " ", s)
        s = re.sub(r"\\(cite[tp]?|citealt|ref|label|eqref|includegraphics|bibliography\w*|usepackage|begin|end)\*?(\[[^\]]*\])*\{[^}]*\}", " ", s)
        s = re.sub(r"\$[^$]*\$", " MATH ", s)
        s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"-\n", "", s)
    return re.findall(r"[a-z0-9]+", s.lower())


paper, n, refs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
P = words(paper)
for ref in refs:
    R = words(ref)
    grams = {tuple(R[i:i + n]) for i in range(len(R) - n + 1)}
    hits = [i for i in range(len(P) - n + 1) if tuple(P[i:i + n]) in grams]
    runs, start, last = [], None, None
    for i in hits:
        if start is None or i > last + 1:
            if start is not None:
                runs.append((start, last + n))
            start = i
        last = i
    if start is not None:
        runs.append((start, last + n))
    runs = [r for r in runs if "math" not in P[r[0]:r[1]] or r[1] - r[0] > n + 3]
    print(f"== {ref}: {len(runs)} shared runs of >= {n} words")
    for a, b in sorted(runs, key=lambda r: r[0] - r[1]):
        print(f"  [{b - a}] " + " ".join(P[a:b]))
