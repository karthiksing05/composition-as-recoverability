import fitz
d=fitz.open("/coc/flash10/ksingara3/compositionality-theory/recoverability-theory/main.pdf")
print("pages",len(d))
for i,p in enumerate(d):
    H=p.rect.height
    # occupied vertical intervals from text blocks, images, drawings
    iv=[(b[1],b[3]) for b in p.get_text("blocks")]
    iv+=[(r.y0,r.y1) for r in [p.get_image_bbox(im) for im in p.get_images(full=True)] if r.is_valid]
    iv+=[(dr["rect"].y0,dr["rect"].y1) for dr in p.get_drawings() if dr["rect"].height<H*0.9]
    iv=sorted((a,b) for a,b in iv if b>a and a>40 and b<H-40)
    gaps=[];cur=iv[0][1] if iv else 0
    for a,b in iv[1:]:
        if a-cur>30: gaps.append((round(cur),round(a),round(a-cur)))
        cur=max(cur,b)
    first=p.get_text()[:60].replace("\n"," | ")
    print(f"p{i+1:02d} bottom={round(cur)}/{round(H)} gaps>30pt={gaps} :: {first}")
    p.get_pixmap(dpi=80).save(f"/coc/flash10/ksingara3/compositionality-theory/notes/audit/ws/p{i+1:02d}.png")
