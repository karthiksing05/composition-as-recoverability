"""Diagnostics for E4 part codes: how much of each model's 16-PC part code is explained by
the number of subword tokens of the noun (a non-lexical feature), and the share of the
code's variance in its first component."""
import numpy as np, sys
import e4_pretrained as E
nouns = list(dict.fromkeys(E.NOUNS))
for name in sys.argv[1:]:
    tok, mod = E.load(name, "cpu")
    nt = np.array([len(tok(x, add_special_tokens=False)["input_ids"]) for x in nouns], float)
    cd = {"phrase": E.embed(tok, mod, [f"a {x}" for x in nouns], "cpu", name),
          "carrier": E.embed(tok, mod, [f"there is a {x} here" for x in nouns], "cpu", name)}
    for cname, layers in cd.items():
        for layer, P in layers.items():
            P = P - P.mean(0)
            U, s, Vt = np.linalg.svd(P, full_matrices=False)
            T = U[:, :16] * s[:16]
            X = np.c_[np.ones(len(nt)), nt, nt ** 2]
            beta, *_ = np.linalg.lstsq(X, T, rcond=None)
            r2 = 1 - ((T - X @ beta) ** 2).sum() / (T ** 2).sum()
            print(f"{name:20s} {cname:8s} {layer:7s} share_pc1={s[0]**2/(s[:16]**2).sum():.2f} R2(code ~ ntokens)={r2:.2f} multi-token nouns={int((nt>1).sum())}", flush=True)
