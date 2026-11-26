D = "/coc/flash10/ksingara3/compositionality-theory/icml2027/main.tex"
t = open(D, encoding="utf-8").read()
R = [
 (" The states of each set support every capability of the sets before it, although a system exhibits those capabilities only when its own procedures use its states.", ""),
 ("    Lexicon harness & Weak: lookup$^\\dagger$ & Weak: lookup$^\\dagger$ \\\\\n", ""),
 (" $^\\dagger$A lookup over relation words assigns the roles.", ""),
]
for a, b in R:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open(D, "w", encoding="utf-8").write(t); print("ok")
