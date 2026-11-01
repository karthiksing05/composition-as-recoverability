from PIL import Image
import glob
fs=sorted(glob.glob("/coc/flash10/ksingara3/compositionality-theory/notes/audit/ws/p??.png"))
for k in range(0,len(fs),4):
    ims=[Image.open(f) for f in fs[k:k+4]]
    w,h=ims[0].size
    m=Image.new("RGB",(w*4+30,h),"gray")
    for j,im in enumerate(ims): m.paste(im,(j*(w+10),0))
    m.save(f"/coc/flash10/ksingara3/compositionality-theory/notes/audit/ws/m{k//4:02d}.png")
