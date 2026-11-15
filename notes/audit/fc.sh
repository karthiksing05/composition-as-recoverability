#!/bin/bash
cd /coc/flash10/ksingara3/compositionality-theory/experiments
PY=~/flash/miniconda3/envs/chunks/bin/python
$PY fig_flow.py all 2>&1 | grep -E "Traceback|Error|^done" | tail -3
$PY fig_protocol.py 2>&1 | grep -E "Traceback|Error" ; $PY fig_concepts.py 2>&1 | grep -E "Traceback|Error"
$PY check_figure_overlaps.py 2>&1 | grep -v "fitz"
$PY - <<'P' 2>/dev/null
import fitz
for n in ("fig_protocol","fig_fibers"):
    fitz.open(f"../recoverability-theory/figures/{n}.pdf")[0].get_pixmap(dpi=170).save(f"../notes/audit/figs/{n}.png")
P
cd .. && bash slurm/build.sh 2>&1 | grep -v fitz
$PY notes/audit/mont2.py 2>&1 | tail -1
