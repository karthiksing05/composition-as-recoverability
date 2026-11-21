"""Check every figure the paper includes for text that touches other text, a line, a filled shape or an image.
Works on the final PDFs, so it covers figures from any script.  Text boxes are the glyph extents, trimmed vertically
to the ink.  A filled shape may contain a text entirely when it is a large container (a card); a stroke crossing the
text box or a fill that only partly covers it is a hit.  Exits 1 on any hit."""
import re, sys, glob, os
import fitz

ROOT = "/coc/flash10/ksingara3/compositionality-theory/icml2027"
import glob
tex = "".join(open(p, encoding="utf-8").read() for p in [f"{ROOT}/main.tex"] + sorted(glob.glob(f"{ROOT}/parts/*.tex")))
names = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+\.pdf)\}", tex)))
PADX, TRIM = 0.2, 0.22


def seg_rect(p, q, r):
    """Liang-Barsky: does segment pq cross rectangle r?"""
    x0, y0, dx, dy = p[0], p[1], q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for a, b in ((-dx, x0 - r.x0), (dx, r.x1 - x0), (-dy, y0 - r.y0), (dy, r.y1 - y0)):
        if a == 0:
            if b < 0:
                return False
        else:
            t = b / a
            if a < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def inside_poly(pt, segs):
    """Even-odd rule over the flattened outline."""
    x, y = pt.x, pt.y
    n = 0
    for (x0, y0), (x1, y1) in segs:
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            n += 1
    return n % 2 == 1


def segments(d):
    out = []
    for it in d["items"]:
        k = it[0]
        if k == "l":
            out.append((tuple(it[1]), tuple(it[2])))
        elif k == "re":
            r = fitz.Rect(it[1]); c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
            out += [(c[i], c[(i + 1) % 4]) for i in range(4)]
        elif k == "c":
            P = [tuple(x) for x in it[1:5]]
            pts = [tuple(sum(w * P[j][a] for j, w in enumerate(((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3))) for a in (0, 1)) for t in [i / 8 for i in range(9)]]
            out += list(zip(pts[:-1], pts[1:]))
        elif k == "qu":
            q = it[1]; c = [tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll)]
            out += [(c[i], c[(i + 1) % 4]) for i in range(4)]
    return out


bad = 0
for n in names:
    page = fitz.open(f"{ROOT}/figures/{n}")[0]
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    r = fitz.Rect(s["bbox"]); h = r.height
                    spans.append((s["text"].strip(), fitz.Rect(r.x0 + PADX, r.y0 + TRIM * h, r.x1 - PADX, r.y1 - TRIM * h)))
    hits = []
    for i, (a, ra) in enumerate(spans):
        for b, rb in spans[i + 1:]:
            # pieces of one mathtext label sit side by side and overlap by less than a point
            if ra.intersects(rb) and min(ra.x1, rb.x1) - max(ra.x0, rb.x0) > 0.8:
                hits.append(f"text/text: '{a}' & '{b}'")
    for im in page.get_image_info():
        ir = fitz.Rect(im["bbox"])
        for a, ra in spans:
            if ra.intersects(ir):
                hits.append(f"text/image: '{a}'")
    clips = {}
    for d in page.get_drawings(extended=True):
        if d["type"] == "clip":
            clips[d["level"]] = fitz.Rect(d["scissor"])
            continue
        if d["type"] == "group":
            continue
        clip = fitz.Rect(page.rect)
        for lv, c in clips.items():
            if lv < d["level"]:
                clip &= c
        segs = segments(d)
        if not segs or clip.is_empty:
            continue
        stroked = d["type"] in ("s", "fs") and (d.get("color") is not None)
        filled = d["type"] in ("f", "fs") and d.get("fill") not in (None, (1.0, 1.0, 1.0))
        R = fitz.Rect(d["rect"])
        V = fitz.Rect(R) & clip if clips else fitz.Rect(R)
        if stroked and R.width > 20 and (R.x1 > page.rect.x1 + 0.2 or R.x0 < -0.2 or R.y1 > page.rect.y1 + 0.2 or R.y0 < -0.2) and not any(c.contains(page.rect) is False and lv < d["level"] for lv, c in clips.items()):
            hits.append(f"shape/edge: {R.width:.0f}x{R.height:.0f} at {R.x0:.0f},{R.y0:.0f}")
        for a, ra in spans:
            rc = fitz.Rect(ra) & clip
            if rc.is_empty:
                continue
            cross = any(seg_rect(p, q, rc) for p, q in segs)
            if stroked and cross:
                hits.append(f"text/line: '{a}' ({d['type']}, {R.width:.0f}x{R.height:.0f} at {R.x0:.0f},{R.y0:.0f})")
            elif filled and cross:
                hits.append(f"text/fill edge: '{a}' ({R.width:.0f}x{R.height:.0f} at {R.x0:.0f},{R.y0:.0f})")
            elif filled and R.width > 0.5 and R.height > 0.5 and inside_poly(rc.tl + (rc.br - rc.tl) / 2, segs):
                if R.get_area() < 6 * ra.get_area():
                    hits.append(f"text/fill: '{a}' ({R.width:.0f}x{R.height:.0f} at {R.x0:.0f},{R.y0:.0f})")
    for a, ra in spans:
        if not fitz.Rect(page.rect).contains(ra):
            hits.append(f"text/edge: '{a}'")
    hits = sorted(set(hits))
    print(f"{n}: {len(hits)} hits ({len(spans)} text spans)")
    for h in hits:
        print("   ", h)
    bad += len(hits)
print("TOTAL", bad)
sys.exit(1 if bad else 0)
