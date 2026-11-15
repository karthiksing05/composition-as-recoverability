import fitz, re, unicodedata, shutil
D="notes/audit/web_sources/smol/"
print("tesseract",shutil.which("tesseract"))
try:
    import pytesseract; print("pytesseract ok")
except Exception as e: print("no pytesseract")
for k,kw in (("s1990",r"saturat|graceful|linearly independent|Fodor|cross-?talk|compositional"),("s1988b",r"coffee|approximat\w+ .{0,60}subsymbolic|subsymbolic hypothesis|Fodor|constituent")):
    d=fitz.open(D+k+".pdf"); T=[unicodedata.normalize("NFKC",p.get_text()) for p in d]
    print("=====",k,"pages",len(d),"chars per page",[len(t) for t in T][:12])
    open(D+k+".txt","w",encoding="utf-8").write("\n\f".join(T))
    n=0
    for i,t in enumerate(T):
        t=re.sub(r"-\s*\n","",t); t=re.sub(r"\s+"," ",t)
        last=-1000
        for m in re.finditer(kw,t,re.I):
            if m.start()-last<400: continue
            last=m.start(); n+=1
            if n<=40: print(f"[pdf p{i+1}] {t[max(0,m.start()-170):m.end()+200]}")
    print("hits",n)
