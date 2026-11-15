import fitz, sys
d = fitz.open("recoverability-theory/main.pdf")
print("pages", len(d))
n = len(d)
W, H = 612, 792
for s in range(0, min(n, 16), 8):
    M = fitz.open(); page = M.new_page(width=4*W*0.5, height=2*H*0.5)
    for i in range(s, min(s+8, n)):
        k = i - s
        r = fitz.Rect((k%4)*W*0.5, (k//4)*H*0.5, (k%4+1)*W*0.5, (k//4+1)*H*0.5)
        page.show_pdf_page(r, d, i)
    page.get_pixmap(dpi=110).save(f"notes/audit/figs/pages_{s:02d}.png")
