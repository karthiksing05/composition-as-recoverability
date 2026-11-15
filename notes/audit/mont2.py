import fitz, numpy as np
from PIL import Image
d=fitz.open("recoverability-theory/main.pdf")
ims=[]
for i in range(len(d)):
    p=d[i].get_pixmap(dpi=75); ims.append(Image.frombytes("RGB",(p.w,p.h),p.samples))
w,h=ims[0].size
for k in range(0,len(ims),3):
    M=Image.new("RGB",(3*w,h),"white")
    for j,im in enumerate(ims[k:k+3]): M.paste(im,(j*w,0))
    M.save(f"notes/audit/figs/mont_{k//3}.png")
print(len(ims))
