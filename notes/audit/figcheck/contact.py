"""Contact sheet of PDF pages: python contact.py PDF OUT.png first last cols"""
import sys, fitz
from PIL import Image
pdf, out, a, b, cols = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
doc = fitz.open(pdf); ims = []
for i in range(a - 1, min(b, doc.page_count)):
    pm = doc[i].get_pixmap(dpi=45)
    ims.append(Image.frombytes("RGB", (pm.width, pm.height), pm.samples))
w, h = ims[0].size; rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * h), "white")
for k, im in enumerate(ims):
    sheet.paste(im, ((k % cols) * w, (k // cols) * h))
sheet.save(out); print(sheet.size)
