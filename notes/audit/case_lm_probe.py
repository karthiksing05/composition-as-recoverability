import json, numpy as np, sys
sys.path.insert(0, "/coc/flash10/ksingara3/compositionality-theory/experiments")
import e4_pretrained as E
rng = np.random.default_rng(0)
nouns = list(dict.fromkeys(E.NOUNS)); rng.shuffle(nouns)
fr = nouns[E.N_FIT + E.N_VAL:E.N_FIT + E.N_VAL + E.N_FRESH]
print("HELDOUT", fr)
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/results/"
for m in ["microsoft_phi-4", "Qwen_Qwen3-8B", "allenai_OLMo-2-1124-7B", "EleutherAI_pythia-1.4b", "gpt2-large"]:
    try:
        d = json.load(open(R + f"e9/{m}_s0.json", encoding="utf-8"))
        on = [r["choice"] for r in d if r["rel"] == "on"]; al = [r["choice"] for r in d]
        print("E9", m, "on", np.mean(on), "all", np.mean(al))
    except Exception as e: print("E9", m, e)
    try:
        d = json.load(open(R + f"e19/{m}_s0.json", encoding="utf-8"))
        for r in d:
            if r["rel"] == "on": print("E19", m, r["state"], r["layer"], r["code"], "ridge_conv", r["ridge_conv"], "ridge_can", r["ridge_can"])
    except Exception as e: print("E19", m, e)
    pick = {"gpt2-large": ("L18", "carrier"), "EleutherAI_pythia-1.4b": ("L12", "carrier"), "microsoft_phi-4": ("L30", "phrase"),
            "Qwen_Qwen3-8B": ("L27", "carrier"), "allenai_OLMo-2-1124-7B": ("L16", "carrier")}[m]
    try:
        d = json.load(open(R + f"e11/{m}_s0.json", encoding="utf-8"))
        for r in d:
            if r["system"] == "model" and r["state"] in ("query", "mean") and (r["layer"], r["code"]) == pick:
                print("E11", m, r["rel"], r["state"], pick, "xfer_role_conv", round(r["xfer_role_conv"], 3))
    except Exception as e: print("E11", m, e)
