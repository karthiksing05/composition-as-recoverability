"""List PolyPythia repos and their checkpoint branches."""
from huggingface_hub import HfApi
api = HfApi()
for m in api.list_models(author="EleutherAI", search="pythia-410m-seed", limit=50):
    print(m.id)
for r in ("EleutherAI/pythia-410m-seed1", "EleutherAI/pythia-160m-seed1"):
    try:
        b = sorted(x.name for x in api.list_repo_refs(r).branches)
        print(r, len(b), b[:12], b[-5:])
    except Exception as e:
        print(r, "ERR", type(e).__name__, e)
