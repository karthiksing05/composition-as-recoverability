import json, glob
s = json.load(open("results/e42/summary.json", encoding="utf-8"))
print("E42 keys", list(s.keys()))
def show(x, depth=0, maxn=60):
    out = json.dumps(x)[:maxn*40]
    print(out)
for k in ["table", "random_weight", "absx_controls"]:
    print("==", k); show(s.get(k))
e = json.load(open("results/e41/summary.json", encoding="utf-8"))
print("E41 keys", list(e.keys())[:20])
show({k: e[k] for k in list(e.keys())[:3]})
