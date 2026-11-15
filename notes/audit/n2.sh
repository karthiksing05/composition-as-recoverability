#!/bin/bash
cd /coc/flash10/ksingara3/compositionality-theory/experiments
PY=~/flash/miniconda3/envs/chunks/bin/python
$PY fig_protocol.py 2>&1 | grep -E "Traceback|Error|^ok"
$PY fig_concepts.py 2>&1 | grep -E "Traceback|Error|^ok"
$PY check_figure_overlaps.py 2>&1 | grep -v fitz | tail -15
cd .. && bash slurm/build.sh 2>&1 | grep -v fitz
$PY notes/audit/pages.py 2>&1 | grep -E "^pages"
$PY - <<'P' 2>/dev/null
import fitz
d=fitz.open("recoverability-theory/main.pdf")
t=" ".join(p.get_text() for p in d)
print("?? count", t.count("??"), "fresh", t.lower().count("fresh"), "familiar", t.lower().count("familiar"))
for n in ("fig_protocol","fig_parts"):
    fitz.open(f"recoverability-theory/figures/{n}.pdf")[0].get_pixmap(dpi=170).save(f"notes/audit/figs/{n}.png")
P
