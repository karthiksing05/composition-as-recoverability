import re
tex=open("main.tex",encoding="utf-8").read(); bib=open("references.bib",encoding="utf-8").read()
cited=set()
for m in re.findall(r"\\[cC]ite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}",tex):
    cited|={k.strip() for k in m.split(",")}
ents={}
for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}",bib,re.S):
    b=m.group(3)
    g=lambda f:(re.search(f+r"\s*=\s*\{(.+?)\}\s*,?\s*(?:\n|$)",b,re.S) or [None,""])[1]
    ents[m.group(2).strip()]=(g("author")[:40].replace("\n"," "),g("title")[:75].replace("\n"," "),(re.search(r"year\s*=\s*\{?(\d+)",b) or [0,"?"])[1])
print("CITED",len(cited),"BIB",len(ents),"missing",cited-set(ents))
print("--- cited")
for k in sorted(cited&set(ents)): print(k,"|",*ents[k][:1],"|",ents[k][2])
print("--- in bib, NOT cited")
for k in sorted(set(ents)-cited): print(k,"|",ents[k][0],"|",ents[k][1],"|",ents[k][2])
