#!/bin/bash
cd /coc/flash10/ksingara3/compositionality-theory/experiments
PY=~/flash/miniconda3/envs/chunks/bin/python
for s in fig_spectrum.py fig_intro.py fig_ladder.py; do $PY $s 2>&1 | grep -v findfont | tail -1; done
$PY fig_story.py strong_only 2>&1 | grep -v findfont | tail -1
cd .. && bash slurm/build.sh 2>&1 | grep -v fitz | tail -8
$PY experiments/check_figure_overlaps.py 2>&1 | tail -14
$PY - <<'P' 2>/dev/null
import pymupdf
d=pymupdf.open("recoverability-theory/main.pdf")
t=" ".join(p.get_text() for p in d)
print("pages", len(d), "?? count", t.count("??"), "fresh", t.lower().count("fresh"), "familiar", t.lower().count("familiar"))
for i,p in enumerate(d[:16]): p.get_pixmap(dpi=70).save(f"notes/audit/ws/r{i+1}.png")
P
