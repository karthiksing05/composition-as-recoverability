from huggingface_hub import snapshot_download
for repo in ("Alpha-VLLM/Lumina-Image-2.0",):
    try:
        p = snapshot_download(repo, ignore_patterns=["*.bin", "*.pth", "*.ckpt", "*.pt"], max_workers=8)
        print("OK", repo, p, flush=True)
    except Exception as e:
        print("FAIL", repo, repr(e)[:300], flush=True)
