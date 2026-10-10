"""One-off: render logo PNGs from the SVG design (PIL)."""
from PIL import Image, ImageDraw, ImageFont

ARIAL_B = "C:/Windows/Fonts/arialbd.ttf"

BG = (34, 20, 51)
LIP = (238, 66, 102)
LIP_LT = (255, 157, 173)
LIP_DARK = (142, 21, 56)
SEAM = (126, 18, 48)
MOUTH = (66, 6, 15)
PINK = (255, 151, 169)
WHITE = (255, 255, 255)

# outer lip contour as chained cubic segments (512 space)
OUTER = [
    ((100, 258), (130, 215), (150, 192), (185, 190)),
    ((185, 190), (210, 190), (230, 210), (256, 224)),
    ((256, 224), (282, 210), (302, 190), (327, 190)),
    ((327, 190), (362, 192), (382, 215), (412, 258)),
    ((412, 258), (390, 315), (330, 350), (256, 356)),
    ((256, 356), (182, 350), (122, 315), (100, 258)),
]
WRINKLES = [((196, 196), (212, 220)), ((316, 196), (300, 220)),
            ((158, 232), (186, 244)), ((354, 232), (326, 244)),
            ((150, 300), (180, 292)), ((362, 300), (332, 292)),
            ((196, 332), (214, 312)), ((316, 332), (298, 312))]


def _cubic(d, seg, ox, oy, sc, **kw):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = seg
    pts = []
    for i in range(41):
        t = i / 40
        u = 1 - t
        pts.append((ox + (u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3) * sc,
                    oy + (u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3) * sc))
    d.line(pts, **kw)


def draw_lips(d, s, ox, oy):
    sc = s / 512.0

    def E(box, **kw):
        d.ellipse([ox + box[0] * sc, oy + box[1] * sc,
                   ox + box[2] * sc, oy + box[3] * sc], **kw)

    def L(a, b, w=6, fill=LIP_DARK):
        d.line([ox + a[0] * sc, oy + a[1] * sc, ox + b[0] * sc, oy + b[1] * sc],
               fill=fill, width=max(1, int(w * sc)))

    pts = []
    for seg in OUTER:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = seg
        for i in range(41):
            t = i / 40
            u = 1 - t
            pts.append((ox + (u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3) * sc,
                        oy + (u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3) * sc))
    d.polygon(pts, fill=LIP)
    E((167, 203, 227, 235), fill=LIP_LT)   # upper-lobe lights
    E((285, 203, 345, 235), fill=LIP_LT)
    L((256, 192), (256, 220))              # philtrum pinch
    _cubic(d, ((100, 258), (150, 252), (190, 256), (222, 260)), ox, oy, sc,
           fill=SEAM, width=max(1, int(6 * sc)))
    _cubic(d, ((412, 258), (362, 252), (322, 256), (290, 260)), ox, oy, sc,
           fill=SEAM, width=max(1, int(6 * sc)))
    for a, b in WRINKLES:
        L(a, b)
    E((226, 226, 286, 302), fill=MOUTH, outline=PINK, width=max(2, int(7 * sc)))
    E((243, 245, 251, 259), fill=WHITE)   # mouth glint
    E((198, 314, 314, 338), fill=WHITE)   # lower-lip shine


def icon(path, size=1024):
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    m = int(size * 16 / 512)
    d.rounded_rectangle([m, m, size - m, size - m], radius=int(size * 110 / 512),
                        outline=(62, 36, 84), width=max(2, size // 256))
    draw_lips(d, size - 2 * m, m, m)
    img.save(path)
    print("saved", path)


def horizontal(path, h=512):
    w = int(h * 1400 / 512)
    img = Image.new("RGB", (w, h), (23, 18, 43))
    d = ImageDraw.Draw(img)
    draw_lips(d, h - 32, 16, 16)
    try:
        font = ImageFont.truetype(ARIAL_B, int(h * 104 / 512))
    except Exception:
        font = ImageFont.load_default()
    x = int(548 * h / 512)
    y = int(180 * h / 512)
    d.text((x, y), "simple-", font=font, fill=(244, 241, 250))
    d.text((x + d.textlength("simple-", font=font), y), "ehm",
           font=font, fill=(255, 107, 129))
    img.save(path)
    print("saved", path)


if __name__ == "__main__":
    icon("assets/logo-icon.png", 1024)
    horizontal("assets/logo-horizontal.png", 512)
