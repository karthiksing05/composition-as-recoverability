from huggingface_hub import snapshot_download
import sys
for repo, allow, ign in (
    ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", None, ["*.bin", "*.pth", "*.ckpt"]),
    ("black-forest-labs/FLUX.1-schnell", None, ["flux1-schnell.safetensors", "ae.safetensors", "*.bin"]),
):
    try:
        p = snapshot_download(repo, ignore_patterns=ign, max_workers=8)
        print("OK", repo, p, flush=True)
    except Exception as e:
        print("FAIL", repo, repr(e)[:400], flush=True)
