#!/bin/bash
cd /coc/flash10/ksingara3/compositionality-theory
PY=~/flash/miniconda3/envs/chunks/bin/python
bash slurm/build.sh 2>&1 | grep -v fitz
$PY notes/audit/pages.py 2>&1 | grep -E "^pages"
$PY -W ignore - <<'P'
import fitz,re
d=fitz.open("recoverability-theory/main.pdf")
T=[p.get_text() for p in d]; t=" ".join(T)
print("?? count", t.count("??"))
for n in ("Smolensky, 1987","Smolensky, 1988","Smolensky (1987)","Smolensky (1988","Smolensky and Legendre","(Marcus, 1998","Marcus (1998"): print(n, t.count(n))
i=next(k for k,x in enumerate(T) if "\nReferences\n" in x or x.lstrip().startswith("References"))
for k,x in enumerate(T):
    for m in re.finditer(r"Smolensky, P\.[^\n]*\n[^\n]*\n[^\n]*",x):
        if k>=i: print("REF p",k+1,re.sub(r"\s+"," ",m.group(0))[:170])
print("refs start",i+1)
P
