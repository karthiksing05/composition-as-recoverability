"""Download the round-8 checkpoints once, before the GPU arrays start: denser Pythia-1.4b
checkpoints and Pythia-410m with five PolyPythia seeds (van der Wal et al. 2025)."""
import shutil
from huggingface_hub import snapshot_download
from prefetch_r7 import IGN
STEPS_410 = (0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 143000)
jobs = [("EleutherAI/pythia-1.4b", f"step{s}") for s in (1000, 4000, 16000, 64000)]
jobs += [(r, f"step{s}") for r in ["EleutherAI/pythia-410m"] + [f"EleutherAI/pythia-410m-seed{i}" for i in range(1, 6)] for s in STEPS_410]
for r, rev in jobs:
    for k in range(3):
        try:
            print(r, rev, snapshot_download(r, revision=rev, ignore_patterns=IGN), flush=True)
            break
        except Exception as e:
            print(r, rev, "RETRY", type(e).__name__, e, flush=True)
print(shutil.disk_usage("/coc/flash10/ksingara3"))
