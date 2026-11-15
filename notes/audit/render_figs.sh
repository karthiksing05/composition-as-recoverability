#!/bin/bash
cd /coc/flash10/ksingara3/compositionality-theory/recoverability-theory/figures
~/flash/miniconda3/envs/chunks/bin/python - <<'P'
import fitz, glob, os
for f in sorted(glob.glob("*.pdf")):
    d = fitz.open(f); p = d[0]
    p.get_pixmap(dpi=170).save("../../notes/audit/figs/" + os.path.basename(f)[:-4] + ".png")
    print(f, round(p.rect.width), round(p.rect.height))
P
