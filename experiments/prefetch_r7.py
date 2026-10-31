"""Download the round-7 models once, before the GPU arrays start."""
from huggingface_hub import snapshot_download
IGN = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "flax_model*", "tf_model*", "*.gguf", "original/*"]
repos = ["google/gemma-2-9b", "Qwen/Qwen3-8B", "allenai/OLMo-2-1124-7B", "microsoft/phi-4"] + \
        [f"EleutherAI/pythia-{s}" for s in ("160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")] + \
        ["facebook/dinov2-small", "facebook/dinov2-large", "facebook/dinov2-giant", "google/siglip-so400m-patch14-384",
         "openai/clip-vit-base-patch32", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"]
def get(r, **kw):
    try:
        print(r, kw, snapshot_download(r, ignore_patterns=IGN, **kw), flush=True)
    except Exception as e:  # gated repos already in the cache load offline
        try:
            print(r, kw, "CACHED", snapshot_download(r, ignore_patterns=IGN, local_files_only=True, **kw), flush=True)
        except Exception as e2:
            print(r, kw, "FAILED", type(e).__name__, type(e2).__name__, flush=True)


for r in repos:
    get(r)
for step in (0, 512, 2000, 8000, 32000):
    get("EleutherAI/pythia-1.4b", revision=f"step{step}")
