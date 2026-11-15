#!/bin/bash
cd /coc/flash10/ksingara3/compositionality-theory/experiments
PY=~/flash/miniconda3/envs/chunks/bin/python
$PY fig_protocol.py 2>&1 | grep -v findfont | tail -1
cd .. && bash slurm/build.sh 2>&1 | grep -v fitz
$PY notes/audit/mont2.py 2>&1 | tail -1
$PY - <<'P' 2>/dev/null
import fitz
d=fitz.open("recoverability-theory/main.pdf")
t=" ".join(p.get_text() for p in d)
print("?? count", t.count("??"), "fresh", t.lower().count("fresh"), "familiar", t.lower().count("familiar"))
P
