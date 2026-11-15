"""Clean the nonce list; write task files for E21, E23, E24 and the second E20 calibration."""
R = "/coc/flash10/ksingara3/compositionality-theory/experiments/"
p = R + "e23_nonce.py"
s = open(p, encoding="utf-8").read()
a = 'non = [w for w in dict.fromkeys(NONCE) if w not in set(nouns)]'
b = ('REAL = {"marvin", "unkle", "pelvin", "nurdle", "clumber", "hoskin", "rundle", "frindle", "wintle", "dretch", "moskin", "sprock"}\n'
     '    non = [w for w in dict.fromkeys(NONCE) if w not in set(nouns) and w not in REAL]')
if a in s:
    s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
M = ["microsoft/phi-4", "Qwen/Qwen3-8B", "allenai/OLMo-2-1124-13B", "allenai/OLMo-2-1124-7B", "gpt2-large", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-12b"]
L = []
for script, out in (("e21_intervene.py", "e21"), ("e23_nonce.py", "e23"), ("e24_input.py", "e24")):
    for m in M:
        for sd in range(3):
            L.append(f"{script} --models {m} --seeds {sd} --outdir results/{out}")
open(R + "tasks_e21_24.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
C = []
for g in (8, 16, 32):
    for l in (3, 4, 6):
        C.append(f"e20_entangled.py --archs mlp --ms 64+256 --Vs 8+64 --seed 0 --gain {g} --layers {l} --outdir results/e20cal2")
        C.append(f"e20_entangled.py --archs mlp --ms 256 --Vs 8+64 --seed 0 --gain {g} --layers {l} --steps 12000 --outdir results/e20cal2")
open(R + "tasks_e20cal2.txt", "w", encoding="utf-8").write("\n".join(C) + "\n")
print(len(L), len(C))
