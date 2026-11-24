"""Hand-drawn ink primitives for FUNCTIONING ADULTS.

Everything is drawn in WORLD coordinates through a Camera. Wobble ("line boil")
is applied in SCREEN space so it reads the same at every zoom. Each stroke is
keyed by a string so its wobble is stable within a drawing and re-rolls
only when the boil frame changes.
"""
import math, zlib, os
import numpy as np
import skia

W, H = 1080, 1920
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

# ---------------------------------------------------------------- palette
def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)

PAPER  = hexc("F2EADC")
SKIN   = hexc("F8F2E7")          # faces/hands: a hair lighter than paper
INK    = hexc("1E1A17")
INK_SOFT = hexc("1E1A17", 0.55)
GREY_L = hexc("DCD3C5")
GREY_M = hexc("BDB2A3")
GREY_D = hexc("8C8277")
CORAL  = hexc("E58F78")          # Greg's shirt / accent
CREAM  = hexc("FBEBDD")
KRAFT  = hexc("C9A47C")
MASK   = hexc("2E2926")
MASK_RIB = hexc("47403B")
MASK_LINE = hexc("A49A90")
HOODIE = hexc("564C45")
HOODIE_D = hexc("443B35")
NIGHT  = hexc("3D4653")
NIGHT_D = hexc("323A45")
SIGN_RED = hexc("C46A5A")
SAGE   = hexc("A9B39A")
MUSTARD = hexc("D8BE84")
DUSTY  = hexc("9FAFBE")
SCREEN = hexc("EEF0EA")
CASH   = hexc("A7B79A")


def _paint(color, stroke=None):
    p = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*color))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap(skia.Paint.kRound_Cap)
        p.setStrokeJoin(skia.Paint.kRound_Join)
    return p


def seed_of(key, boil):
    return (zlib.crc32(key.encode()) * 2654435761 + boil * 97531) & 0xFFFFFFFF


# ---------------------------------------------------------------- curves
def catmull(pts, closed=False, density=0.35):
    """Uniform Catmull-Rom through pts -> dense polyline (np array)."""
    P = np.asarray(pts, dtype=float)
    if len(P) < 2:
        return P
    if closed:
        P = np.vstack([P[-1:], P, P[:2]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        seg = np.linalg.norm(p2 - p1)
        n = max(2, int(seg * density))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        t2, t3 = t * t, t * t * t
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2][None, :])
    return np.vstack(out)


def _arclen(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(d)])


def _normals(P):
    g = np.gradient(P, axis=0)
    n = np.stack([-g[:, 1], g[:, 0]], axis=1)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    l[l == 0] = 1
    return n / l


def _wobble(P, rng, amp, closed=False):
    if len(P) < 3 or amp <= 0:
        return P
    s = _arclen(P)
    L = max(s[-1], 1.0)
    d = np.zeros(len(P))
    for lam, w in ((max(L * 1.3, 60), 0.55), (150.0, 0.35), (48.0, 0.18)):
        ph = rng.uniform(0, 2 * math.pi)
        d += w * np.sin(2 * math.pi * s / lam + ph)
    out = P + _normals(P) * (d * amp)[:, None]
    out += rng.normal(0, amp * 0.35, size=2)  # tiny whole-stroke drift
    return out


# ---------------------------------------------------------------- camera
class Camera:
    def __init__(self, cx=W / 2, cy=H / 2, z=1.0):
        self.cx, self.cy, self.z = cx, cy, z

    def pt(self, x, y):
        return ((x - self.cx) * self.z + W / 2, (y - self.cy) * self.z + H / 2)

    def pts(self, P):
        P = np.asarray(P, dtype=float)
        return np.stack([(P[:, 0] - self.cx) * self.z + W / 2, (P[:, 1] - self.cy) * self.z + H / 2], axis=1)

    @property
    def lw(self):  # line-width multiplier: thicker in close-ups, but not linearly
        return self.z ** 0.62


# ---------------------------------------------------------------- context
class Ink:
    def __init__(self, canvas, cam, boil, wobble=1.25):
        self.c = canvas
        self.cam = cam
        self.boil = boil
        self.amp = wobble

    def rng(self, key):
        return np.random.default_rng(seed_of(key, self.boil))

    # -- stroke: tapered, wobbly, variable-width ink line
    def line(self, pts, w=4.2, key="", color=INK, closed=False, smooth=True, taper=0.45, wob=1.0, overshoot=0.0):
        P = np.asarray(pts, dtype=float)
        if closed and overshoot:
            P = np.vstack([P, P[: max(1, int(len(P) * overshoot) + 1)]])
            closed = False
        S = self.cam.pts(P)
        D = catmull(S, closed=closed) if (smooth and len(S) > 2) else _densify(S, closed)
        rng = self.rng(key)
        D = _wobble(D, rng, self.amp * wob, closed)
        width = w * self.cam.lw
        s = _arclen(D)
        L = max(s[-1], 1e-3)
        t = s / L
        if closed:
            prof = np.ones_like(t)
        else:
            e = 0.10 if L > 80 else 0.25
            prof = taper + (1 - taper) * np.clip(np.minimum(t, 1 - t) / e, 0, 1) ** 0.6
        prof *= 1 + 0.12 * np.sin(2 * math.pi * s / 70 + rng.uniform(0, 6.28))
        r = width * prof / 2
        N = _normals(D)
        left = D + N * r[:, None]
        right = D - N * r[:, None]
        path = skia.Path()
        path.moveTo(*left[0])
        for p in left[1:]:
            path.lineTo(*p)
        for p in right[::-1]:
            path.lineTo(*p)
        path.close()
        if not closed:
            path.addCircle(*D[0], r[0])
            path.addCircle(*D[-1], r[-1])
        self.c.drawPath(path, _paint(color))

    # -- filled shape with wobble + slight misregistration (hand-coloured look)
    def fill(self, pts, color, key="", shift=(2.5, 2.0), wob=0.8, smooth=True, shrink=0.0):
        S = self.cam.pts(pts)
        D = catmull(S, closed=True) if smooth and len(S) > 2 else _densify(S, True)
        rng = self.rng(key + "#f")
        D = _wobble(D, rng, self.amp * wob, True)
        if shrink:
            c = D.mean(axis=0)
            D = c + (D - c) * (1 - shrink)
        D = D + np.array(shift) * self.cam.lw * 0.7
        path = skia.Path()
        path.moveTo(*D[0])
        for p in D[1:]:
            path.lineTo(*p)
        path.close()
        self.c.drawPath(path, _paint(color))

    def shape(self, pts, fill=None, w=4.2, key="", smooth=True, line=True, shift=(2.5, 2.0), lw_color=INK):
        if fill is not None:
            self.fill(pts, fill, key, smooth=smooth, shift=shift)
        if line:
            self.line(pts, w, key, closed=True, smooth=smooth, overshoot=0.06, color=lw_color)

    def ellipse_pts(self, cx, cy, rx, ry, n=28, a0=0.0, rot=0.0, jitter_key=None):
        a = np.linspace(0, 2 * math.pi, n, endpoint=False) + a0
        x, y = rx * np.cos(a), ry * np.sin(a)
        cr, sr = math.cos(rot), math.sin(rot)
        return np.stack([cx + x * cr - y * sr, cy + x * sr + y * cr], axis=1)

    def ellipse(self, cx, cy, rx, ry, fill=None, w=4.2, key="", line=True, rot=0.0, shift=(2.0, 1.5)):
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        a0 = rng.uniform(0, 6.28)
        P = self.ellipse_pts(cx, cy, rx, ry, a0=a0, rot=rot)
        if fill is not None:
            self.fill(P, fill, key, shift=shift)
        if line:
            self.line(P, w, key, closed=True, overshoot=0.09)

    def dot(self, x, y, r, color=INK):
        sx, sy = self.cam.pt(x, y)
        self.c.drawCircle(sx, sy, r * self.cam.z, _paint(color))

    def text(self, s, x, y, size, font="PatrickHand-Regular.ttf", color=INK, align="center", rot=0.0, key=""):
        sx, sy = self.cam.pt(x, y)
        f = font_obj(font, size * self.cam.z)
        wdt = f.measureText(s)
        rng = self.rng(key + "#t")
        jx, jy = rng.normal(0, 0.6, 2)
        self.c.save()
        self.c.translate(sx + jx, sy + jy)
        if rot:
            self.c.rotate(rot)
        ox = {"center": -wdt / 2, "left": 0, "right": -wdt}[align]
        self.c.drawString(s, ox, 0, f, _paint(color))
        self.c.restore()

    def clip(self, pts):
        S = self.cam.pts(pts)
        D = catmull(S, closed=True)
        path = skia.Path()
        path.moveTo(*D[0])
        for p in D[1:]:
            path.lineTo(*p)
        path.close()
        self.c.save()
        self.c.clipPath(path, skia.ClipOp.kIntersect, True)

    def unclip(self):
        self.c.restore()


def _densify(S, closed):
    S = np.asarray(S, float)
    if closed:
        S = np.vstack([S, S[:1]])
    out = []
    for a, b in zip(S[:-1], S[1:]):
        n = max(2, int(np.linalg.norm(b - a) * 0.3))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(a + (b - a) * t)
    out.append(S[-1:])
    return np.vstack(out)


_fonts = {}
def font_obj(name, size):
    k = (name, round(size, 1))
    if k not in _fonts:
        tf = skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, name))
        f = skia.Font(tf, size)
        f.setEdging(skia.Font.Edging.kAntiAlias)
        _fonts[k] = f
    return _fonts[k]


# ---------------------------------------------------------------- paper
_paper = None
def paper_image():
    global _paper
    if _paper is None:
        rng = np.random.default_rng(7)
        base = np.array(PAPER[:3]) * 255
        low = rng.normal(0, 1, (H // 24 + 2, W // 24 + 2))
        from numpy import kron
        low = kron(low, np.ones((24, 24)))[:H, :W]
        # blur low-frequency mottling
        k = np.ones(61) / 61
        low = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, low)
        low = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0, low)
        fine = rng.normal(0, 1, (H, W))
        img = base[None, None, :] + (low * 2.2 + fine * 2.4)[:, :, None] * np.array([1, 0.97, 0.9])[None, None, :]
        # faint fibres
        for _ in range(260):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            a = rng.uniform(0, math.pi); L = rng.uniform(6, 22)
            for t in np.linspace(0, L, int(L)):
                xi, yi = int(x + t * math.cos(a)), int(y + t * math.sin(a))
                if 0 <= xi < W and 0 <= yi < H:
                    img[yi, xi] -= 7
        # warm vignette
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2)
        img *= (1 - 0.10 * np.clip(r - 0.35, 0, 1) ** 1.5)[:, :, None]
        arr = np.clip(img, 0, 255).astype(np.uint8)
        rgba = np.dstack([arr, np.full((H, W), 255, np.uint8)])
        _paper = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType)
    return _paper


_grain = None
def grain_image():
    """Multiply-ish paper tooth laid over the finished frame so fills sit IN the paper."""
    global _grain
    if _grain is None:
        rng = np.random.default_rng(11)
        g = rng.normal(0, 1, (H, W))
        a = np.clip(np.abs(g) * 10, 0, 28).astype(np.uint8)
        rgba = np.dstack([np.full((H, W), 90, np.uint8), np.full((H, W), 75, np.uint8), np.full((H, W), 60, np.uint8), a])
        _grain = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType)
    return _grain


def new_frame():
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.drawImage(paper_image(), 0, 0)
    return surf, c


def finish(surf, c, grain=True):
    if grain:
        p = skia.Paint(BlendMode=skia.BlendMode.kMultiply)
        p.setAlphaf(0.35)
        c.drawImage(grain_image(), 0, 0, skia.SamplingOptions(), p)
    img = surf.makeImageSnapshot()
    return img.toarray(colorType=skia.kRGBA_8888_ColorType)
