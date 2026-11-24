"""Series cast, drawn to match the model sheets in the repo root.

Every character is a function draw_<name>(ink, pose) with a pose dict.
Head-centre coordinates are in world space. Keep these designs stable across
episodes: change poses, never construction.
"""
import math
import numpy as np
from ink import (INK, SKIN, CORAL, CREAM, MASK, MASK_RIB, MASK_LINE, HOODIE, HOODIE_D,
                 KRAFT, GREY_D, hexc)

WHITE = hexc("FFFBF4")
GUN = hexc("3A3531")
MOUTH_D = hexc("4A2C27")
TONGUE = hexc("D98A7A")


def V(x, y):
    return np.array([x, y], dtype=float)


def rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def ik(shoulder, target, l1, l2, bend=1):
    S, T = np.asarray(shoulder, float), np.asarray(target, float)
    d = T - S
    dist = np.linalg.norm(d)
    dist = np.clip(dist, abs(l1 - l2) + 1, l1 + l2 - 0.5)
    a = math.atan2(d[1], d[0])
    cb = (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)
    b = math.acos(np.clip(cb, -1, 1))
    E = S + l1 * np.array([math.cos(a + bend * b), math.sin(a + bend * b)])
    Hh = S + dist * np.array([math.cos(a), math.sin(a)])
    return E, Hh


# ------------------------------------------------------------------ hands
_HANDS = {
    "relaxed": [(0, -.30), (.35, -.42), (.75, -.40), (1.0, -.22), (1.06, 0), (.96, .22), (.70, .34),
                (.48, .36), (.45, .60), (.30, .62), (.18, .34), (0, .30)],
    "point": [(0, -.30), (.40, -.36), (.62, -.30), (1.28, -.22), (1.34, -.12), (1.28, -.03), (.70, 0),
              (.68, .26), (.45, .36), (.42, .55), (.26, .56), (.15, .32), (0, .30)],
    "grip": [(0, -.30), (.32, -.42), (.62, -.36), (.74, -.10), (.70, .20), (.48, .38), (.20, .38), (0, .30)],
    "flat": [(0, -.22), (.45, -.26), (.95, -.22), (1.18, -.10), (1.20, .06), (.95, .14), (.45, .16), (0, .20)],
}
_FINGER_LINES = {
    "relaxed": [[(.98, -.07), (.74, -.06)], [(.94, .11), (.71, .10)]],
    "grip": [[(.66, -.18), (.48, -.18)], [(.66, .04), (.48, .04)], [(.62, .22), (.46, .20)]],
    "flat": [[(1.15, -.03), (.85, -.03)], [(1.12, .08), (.85, .07)]],
    "point": [[(.66, .10), (.48, .08)]],
}


def hand(ink, wrist, angle, size, kind="relaxed", key="", flip=False, fill=SKIN):
    pts = np.array(_HANDS[kind], float)
    if flip:
        pts[:, 1] *= -1
    pts *= size
    P = np.array([wrist + rot(p, angle) for p in pts])
    ink.shape(P, fill=fill, w=3.6, key=key + "h", shift=(1, 1))
    for i, ln in enumerate(_FINGER_LINES.get(kind, [])):
        q = np.array(ln, float)
        if flip:
            q[:, 1] *= -1
        q *= size
        ink.line([wrist + rot(p, angle) for p in q], 2.6, key + f"f{i}", smooth=False)


def gun(ink, wrist, angle, size, key="gun", flip=False):
    loc = [(.15, -.42), (1.75, -.42), (1.78, -.16), (.70, -.14), (.66, .02), (.46, .02), (.50, .60),
           (.12, .62), (.20, -.12)]
    pts = np.array(loc, float)
    if flip:
        pts[:, 1] *= -1
    pts *= size
    P = np.array([wrist + rot(p, angle) for p in pts])
    ink.shape(P, fill=GUN, w=3.6, key=key, smooth=False, shift=(1, 1))


def paper_bag(ink, top_center, w=96, h=118, key="bag", open_top=False, tilt=0.0):
    x, y = top_center
    c = V(x, y + h / 2)
    def tp(px, py):
        return c + rot(V(px, py), tilt)
    body = [tp(-w / 2, -h / 2), tp(w / 2, -h / 2 + 4), tp(w / 2 + 3, h / 2), tp(-w / 2 - 2, h / 2)]
    ink.shape(body, fill=KRAFT, w=3.8, key=key, smooth=False, shift=(2, 1.5))
    # crumpled / folded top edge
    zig = [tp(-w / 2 + i * w / 6, -h / 2 + (6 if i % 2 else -3) + (i * 0.7)) for i in range(7)]
    ink.line(zig, 3.0, key + "z", smooth=False)
    if open_top:
        ink.line([tp(-w / 2 + 6, -h / 2 + 14), tp(w / 2 - 6, -h / 2 + 16)], 2.4, key + "o", color=GREY_D)
    ink.line([tp(-w / 2 + 14, -h / 2 + 30), tp(-w / 2 + 18, h / 2 - 12)], 2.0, key + "c1", color=GREY_D)
    ink.line([tp(w / 2 - 22, -h / 2 + 50), tp(w / 2 - 30, -h / 2 + 64)], 2.0, key + "c2", color=GREY_D)


# ================================================================== GREG
GREG_FLOWERS = [(-48, 40), (8, 26), (52, 64), (-30, 112), (36, 150), (-58, 176), (12, 206), (58, 236),
                (-34, 262), (40, 300), (-6, 330), (-62, 316)]


def _flower(ink, c, r, key):
    for i in range(5):
        a = i * 2 * math.pi / 5 + 0.3
        ink.fill(ink.ellipse_pts(c[0] + math.cos(a) * r * 0.62, c[1] + math.sin(a) * r * 0.62, r * 0.52, r * 0.46, n=10),
                 CREAM, key + f"p{i}", shift=(0, 0), wob=0.4)
    ink.dot(c[0], c[1], r * 0.22, hexc("D9796A"))


def greg_head(ink, p):
    x, y, R = p["x"], p["y"], p.get("R", 88)
    tn = p.get("turn", 0.5)
    face = 1 if tn >= 0 else -1
    nod = p.get("nod", 0.0)
    k = p.get("key", "greg")
    lx, ly = p.get("look", (0.4, 0))
    # head
    ink.ellipse(x, y, R, R * 1.02, fill=SKIN, w=4.4, key=k + "head", shift=(0, 0))
    # hair: messy strands from the back-top of the skull, swept back
    hair = [(-98, 70, 0.5, 18), (-112, 86, 0.8, -14), (-122, 62, 0.3, 10), (-136, 80, 0.9, -20),
            (-150, 58, 0.6, 16), (-166, 50, 1.0, -8), (-104, 48, -0.2, 6)]
    for i, (ang, L, sweep, curl) in enumerate(hair):
        a = math.radians(ang) if face > 0 else math.radians(-180 - ang)
        rdir = V(math.cos(a), math.sin(a))
        back = V(-face, 0.0)
        nrm = V(-rdir[1], rdir[0])
        r0 = V(x, y) + rdir * R * 0.86
        m = r0 + rdir * L * 0.5 + back * L * sweep * 0.2 + nrm * curl * 0.6
        tip = r0 + rdir * L + back * L * sweep * 0.45 - nrm * curl * 0.4
        ink.line([r0, m, tip], 3.8, k + f"hair{i}", taper=0.2)
    # two short front tufts
    for i, ang in enumerate([-82, -70]):
        a = math.radians(ang) if face > 0 else math.radians(-180 - ang)
        rdir = V(math.cos(a), math.sin(a))
        r0 = V(x, y) + rdir * R * 0.9
        ink.line([r0, r0 + rdir * 26 + V(-face * 8, -10)], 3.2, k + f"tuft{i}", taper=0.25)
    # eyes
    fy = y - 0.16 * R + nod * 0.14 * R
    cxf = x + tn * 0.30 * R
    sep = 0.27 * R * (1 - 0.22 * abs(tn))
    blink = p.get("blink", 0)
    for j, sgn in enumerate((-1, 1)):
        near = (sgn == face)
        er = R * (0.255 if near else 0.225)
        ex, ey = cxf + sgn * sep, fy + (0 if near else 2)
        if blink >= 0.75:
            ink.line([(ex - er * 0.9, ey + er * 0.1), (ex, ey + er * 0.32), (ex + er * 0.9, ey + er * 0.1)], 3.8, k + f"cl{j}")
            continue
        ink.ellipse(ex, ey, er, er * 1.08, fill=WHITE, w=3.8, key=k + f"eye{j}", shift=(0, 0))
        ink.dot(ex + lx * er * 0.42, ey + ly * er * 0.38 + 1, er * 0.36)
        if blink > 0.2:
            lid_y = ey - er * 1.08 + 2 * er * 1.08 * blink * 0.55
            lid = [(ex - er * 1.05, lid_y), (ex, lid_y - er * 0.15), (ex + er * 1.05, lid_y)]
            top = [(ex - er * 1.1, ey - er * 1.25)] + lid + [(ex + er * 1.1, ey - er * 1.25)]
            ink.fill(top, SKIN, k + f"lid{j}", shift=(0, 0), smooth=False)
            ink.line(lid, 3.4, k + f"lidl{j}")
        # eyebrow, friendly-high
        b = p.get("brow", 0.0)
        by = ey - er * 1.45 - b * 6
        ink.line([(ex - er * 0.65, by + 5), (ex, by - 3), (ex + er * 0.65, by + 4)], 3.2, k + f"brow{j}", taper=0.3)
    # mouth: the big helpful grin
    mx = x + tn * 0.27 * R
    my = y + 0.30 * R + nod * 0.10 * R
    mw = 0.56 * R * (1 - 0.12 * abs(tn))
    o = p.get("mouth", 0)  # 0 grin, 1 talk, 2 open
    depth = 0.40 * R + o * 0.08 * R
    top = [(mx - mw, my - 0.06 * R), (mx - mw * 0.5, my - 0.015 * R), (mx, my - 0.005 * R),
           (mx + mw * 0.5, my - 0.015 * R), (mx + mw, my - 0.06 * R)]
    bot = [(mx + mw * 0.78, my + depth * 0.62), (mx, my + depth), (mx - mw * 0.78, my + depth * 0.62)]
    outline = top + bot
    if o == 0:
        ink.fill(outline, WHITE, k + "teeth", shift=(0, 0))
        ink.line([(mx - mw * 0.92, my + 0.14 * R), (mx, my + 0.17 * R), (mx + mw * 0.92, my + 0.14 * R)], 3.0, k + "tl")
        for i in range(-2, 3):
            tx = mx + i * mw * 0.30
            ink.line([(tx, my - 0.01 * R), (tx, my + 0.15 * R)], 2.8, k + f"tu{i}", smooth=False)
        for i in range(-1, 2):
            tx = mx + (i + 0.0) * mw * 0.34
            ink.line([(tx, my + 0.18 * R), (tx, my + 0.30 * R - abs(i) * 4)], 2.8, k + f"tb{i}", smooth=False)
    else:
        ink.fill(outline, MOUTH_D, k + "mo", shift=(0, 0))
        ink.clip(outline)
        ink.fill([(mx - mw * 0.7, my + depth * 0.95), (mx, my + depth * 0.7), (mx + mw * 0.7, my + depth * 0.95),
                  (mx, my + depth * 1.2)], TONGUE, k + "tg", shift=(0, 0))
        ink.fill([(mx - mw * 1.1, my - 0.1 * R), (mx + mw * 1.1, my - 0.1 * R), (mx + mw * 1.1, my + 0.1 * R),
                  (mx - mw * 1.1, my + 0.1 * R)], WHITE, k + "tt", shift=(0, 0), smooth=False)
        ink.unclip()
        ink.line([(mx - mw * 0.92, my + 0.09 * R), (mx, my + 0.105 * R), (mx + mw * 0.92, my + 0.09 * R)], 2.8, k + "tl")
        for i in range(-2, 3):
            tx = mx + i * mw * 0.30
            ink.line([(tx, my - 0.01 * R), (tx, my + 0.095 * R)], 2.6, k + f"tu{i}", smooth=False)
    ink.line(outline, 3.8, k + "mouth", closed=True, overshoot=0.05)
    # cheek creases
    for sgn in (-1, 1):
        cx_ = mx + sgn * mw * 1.02
        ink.line([(cx_ - sgn * 2, my - 0.14 * R), (cx_ + sgn * 6, my - 0.05 * R), (cx_ + sgn * 1, my + 0.03 * R)], 2.8, k + f"ck{sgn}")


def greg_torso(ink, p):
    """Shirt + neck. Clipped by caller where the counter hides him."""
    x, y, R = p["x"], p["y"], p.get("R", 88)
    tn = p.get("turn", 0.5)
    k = p.get("key", "greg")
    nb = V(x - tn * 6, y + R * 1.02 + 30)  # neck base
    ink.line([(x - tn * 4, y + R * 0.98), nb + V(0, 4)], 4.0, k + "neck", smooth=False)
    off = tn * 16
    shirt = [nb + V(-24, -2), nb + V(-76, 16), nb + V(-84, 60), nb + V(-82, 160), nb + V(-88, 400),
             nb + V(88, 400), nb + V(82, 160), nb + V(84, 60), nb + V(76, 16), nb + V(24, -2)]
    ink.fill(shirt, CORAL, k + "shirt", smooth=False, shift=(3, 2))
    ink.clip(shirt)
    for i, (fx, fy) in enumerate(GREG_FLOWERS):
        _flower(ink, nb + V(fx + off * 0.4, fy + 20), 15, k + f"fl{i}")
    ink.unclip()
    ink.line(shirt, 4.2, k + "shirtl", closed=False, smooth=False)
    # open collar + placket
    pk = nb + V(off, 0)
    ink.shape([nb + V(-24, -2), pk + V(-6, 46), pk + V(-30, 34)], fill=CORAL, w=3.4, key=k + "colL", smooth=False)
    ink.shape([nb + V(24, -2), pk + V(6, 46), pk + V(30, 34)], fill=CORAL, w=3.4, key=k + "colR", smooth=False)
    ink.fill([nb + V(-20, -1), pk + V(0, 44), nb + V(20, -1)], SKIN, k + "vneck", smooth=False, shift=(0, 0))
    ink.line([pk + V(2, 46), pk + V(4, 390)], 3.0, k + "plk")
    for i in range(4):
        ink.dot(pk[0] + 12, pk[1] + 90 + i * 70, 3.2)
    # sunglasses hooked on the placket
    gx, gy = pk[0] + 4, pk[1] + 66
    ink.line([(gx - 6, gy - 16), (gx + 2, gy - 22), (gx + 8, gy - 14)], 2.8, k + "gl_arm")
    ink.ellipse(gx, gy, 13, 15, fill=INK, w=3.0, key=k + "gl1", shift=(0, 0))
    ink.ellipse(gx + 2, gy + 33, 13, 15, fill=INK, w=3.0, key=k + "gl2", shift=(0, 0))
    ink.line([(gx + 1, gy + 14), (gx + 1, gy + 19)], 3.0, k + "gl_br", smooth=False)
    return nb


def greg_arm(ink, p, side):
    """side: 'near' (screen-left when facing right) or 'far'."""
    x, y, R = p["x"], p["y"], p.get("R", 88)
    tn = p.get("turn", 0.5)
    k = p.get("key", "greg") + side
    nb = V(x - tn * 6, y + R * 1.02 + 30)
    sgn = -1 if side == "near" else 1
    sh = nb + V(sgn * 76, 26)
    a = p["arms"][side]
    tgt = V(*a["hand"])
    bend = a.get("bend", -1 if side == "near" else 1)
    E, Hh = ik(sh, tgt, 104, 100, bend)
    u = (E - sh) / max(np.linalg.norm(E - sh), 1)
    nrm = V(-u[1], u[0])
    s_end = sh + u * 62
    a1, a2 = s_end + nrm * 26, s_end - nrm * 26
    # keep the sleeve opening edge order consistent (outer first)
    if (a1[0] - sh[0]) * sgn < (a2[0] - sh[0]) * sgn:
        a1, a2 = a2, a1
    sleeve = [nb + V(sgn * 60, 8), nb + V(sgn * 84, 18), a1, a2, nb + V(sgn * 70, 78)]
    ink.line([s_end - u * 10, E, Hh], 4.0, k + "arm", taper=0.7)
    ink.shape(sleeve, fill=CORAL, w=3.8, key=k + "slv", smooth=False)
    ink.clip(sleeve)
    _flower(ink, sh + u * 26 + nrm * 8, 12, k + "sf")
    ink.unclip()
    fd = Hh - E
    ang = a.get("angle", math.atan2(fd[1], fd[0]))
    hand(ink, Hh, ang, a.get("size", 40), a.get("kind", "relaxed"), k, flip=a.get("flip", side == "near"))


def draw_greg_upper(ink, p, clip_y=None):
    """Torso + head. Arms separately so props can sit between."""
    if clip_y is not None:
        cam = ink.cam
        ink.c.save()
        import skia
        sy = cam.pt(0, clip_y)[1]
        ink.c.clipRect(skia.Rect(-10, -10, 2000, sy))
    greg_torso(ink, p)
    greg_head(ink, p)
    if clip_y is not None:
        ink.c.restore()


# ================================================================== ROBBER
def robber_head(ink, p):
    x, y, R = p["x"], p["y"], p.get("R", 82)
    tn = p.get("turn", -0.5)
    face = 1 if tn >= 0 else -1
    nod = p.get("nod", 0.0)
    k = p.get("key", "rob")
    lx, ly = p.get("look", (-0.5, 0))
    head = ink.ellipse_pts(x, y, R, R * 1.04, n=30)
    ink.fill(head, MASK, k + "mask", shift=(0, 0))
    ink.clip(head)
    # knit ribs follow the skull
    for i in range(-6, 7):
        off = i / 6.5 + tn * 0.18
        if abs(off) >= 1:
            continue
        xs = x + off * R
        bulge = (1 - off * off) ** 0.5
        ink.line([(xs, y - R * 1.1), (xs + off * R * 0.08 * bulge, y), (xs, y + R * 1.1)], 2.6, k + f"rib{i}", color=MASK_RIB, taper=0.8)
    ink.unclip()
    ink.line(head, 4.4, k + "headl", closed=True, overshoot=0.08)
    # eyes: white ovals, heavy tired lids
    fy = y - 0.08 * R + nod * 0.16 * R
    cxf = x + tn * 0.30 * R
    sep = 0.33 * R * (1 - 0.2 * abs(tn))
    lid = max(p.get("lid", 0.45), p.get("blink", 0))
    for j, sgn in enumerate((-1, 1)):
        near = (sgn == face)
        rx, ry = R * (0.30 if near else 0.27), R * (0.235 if near else 0.215)
        ex, ey = cxf + sgn * sep, fy
        if lid >= 0.92:
            ink.line([(ex - rx, ey + 2), (ex, ey + ry * 0.35), (ex + rx, ey + 2)], 3.6, k + f"cl{j}", color=INK)
        else:
            E = ink.ellipse_pts(ex, ey, rx, ry, n=20)
            ink.fill(E, WHITE, k + f"ew{j}", shift=(0, 0))
            ink.clip(E)
            ink.dot(ex + lx * rx * 0.5, ey + ly * ry * 0.45 + ry * 0.2 * lid, rx * 0.30)
            ly_ = ey - ry * 1.05 + 2.1 * ry * lid
            ink.fill([(ex - rx * 1.3, ey - ry * 1.6), (ex + rx * 1.3, ey - ry * 1.6), (ex + rx * 1.3, ly_), (ex, ly_ + ry * 0.12), (ex - rx * 1.3, ly_)],
                     SKIN, k + f"lidf{j}", shift=(0, 0), smooth=False)
            ink.unclip()
            ink.line(E, 3.4, k + f"eo{j}", closed=True, overshoot=0.08)
            if lid > 0.08:
                ink.line([(ex - rx * 1.05, ly_), (ex, ly_ + ry * 0.12), (ex + rx * 1.05, ly_)], 3.8, k + f"lid{j}")
        # brows (light thread on the dark knit)
        ba = p.get("brow_ang", 0.0)   # + = angry (inner ends down), - = worried
        br = p.get("brow_raise", 0.0)
        by = ey - ry - R * 0.13 - br * R * 0.10
        inner = V(ex - sgn * rx * 0.8, by + ba * R * 0.12)
        outer = V(ex + sgn * rx * 0.9, by - ba * R * 0.05)
        ink.line([outer, (outer + inner) / 2 + V(0, -2), inner], 3.4, k + f"brow{j}", color=MASK_LINE, taper=0.4)
    # mouth through the knit
    mx = x + tn * 0.26 * R
    my = y + 0.50 * R + nod * 0.10 * R
    o = p.get("mouth", 0)
    mw = 0.17 * R
    if o <= 0:
        frown = p.get("frown", 0.5)
        ink.line([(mx - mw, my + frown * 5), (mx, my - frown * 3), (mx + mw, my + frown * 5)], 3.4, k + "mouth", color=MASK_LINE, taper=0.5)
    else:
        oh = 0.05 * R + 0.06 * R * o
        ink.ellipse(mx, my, mw * (0.85 - 0.1 * o), oh, fill=hexc("1A1614"), w=3.0, key=k + "mouthO", shift=(0, 0))
        ink.line(ink.ellipse_pts(mx, my, mw * (0.85 - 0.1 * o) + 2, oh + 2, n=16), 2.6, k + "mouthL", closed=True, color=MASK_LINE, overshoot=0.05)


def robber_body(ink, p):
    x, y, R = p["x"], p["y"], p.get("R", 82)
    tn = p.get("turn", -0.5)
    face = 1 if tn >= 0 else -1
    k = p.get("key", "rob")
    hunch = p.get("hunch", 1.0)
    nb = V(x - face * 16 * hunch, y + R * 0.98 + 18)   # neck base sits behind the head (hunched forward)
    hip = nb + V(-face * 6, 330)
    # legs
    floor = p.get("floor", 1380)
    step = p.get("step", 0)  # -1..1 walk phase
    for j, sgn in enumerate((-1, 1)):
        hp = hip + V(sgn * 24, 0)
        fx = hp[0] + sgn * 10 + step * sgn * 34 * face
        knee = V((hp[0] + fx) / 2 + face * 6 + abs(step) * face * 8, (hp[1] + floor) / 2 - abs(step) * 4)
        foot = V(fx, floor - (8 if step * sgn * face > 0.5 else 0))
        ink.line([hp, knee, foot + V(0, -8)], 4.2, k + f"leg{j}", taper=0.8)
        sh = [foot + V(-30 * face, -16), foot + V(18 * face, -18), foot + V(40 * face, -2), foot + V(34 * face, 6), foot + V(-32 * face, 6)]
        ink.shape(sh, fill=WHITE, w=3.6, key=k + f"shoe{j}", shift=(1, 1))
        ink.line([foot + V(-10, -6), foot + V(8 * face, -6)], 2.0, k + f"lace{j}", color=GREY_D, smooth=False)
    # baggy hoodie
    lean = face * 14 * hunch
    hood = [nb + V(-66 + lean, 22), nb + V(-84 + lean * 0.6, 120), nb + V(-88, 240), nb + V(-80, 350),
            nb + V(80, 352), nb + V(88, 240), nb + V(84 + lean * 0.6, 120), nb + V(66 + lean, 22)]
    ink.shape(hood, fill=HOODIE, w=4.4, key=k + "hood", shift=(3, 2))
    ink.line([nb + V(-78, 330), nb + V(78, 332)], 3.0, k + "hem", color=HOODIE_D)
    pocket = [nb + V(-50, 236), nb + V(52, 234), nb + V(64, 312), nb + V(-62, 314)]
    ink.shape(pocket, fill=HOODIE_D, w=3.0, key=k + "pkt", smooth=False, shift=(1, 1))
    # bunched hood around the neck
    collar = [nb + V(-58 + lean, 6), nb + V(-30 + lean, -14), nb + V(30 + lean, -14), nb + V(64 + lean, 8),
              nb + V(48 + lean, 44), nb + V(0 + lean, 54), nb + V(-46 + lean, 42)]
    ink.shape(collar, fill=HOODIE_D, w=4.0, key=k + "coll", shift=(2, 1))
    ink.line([nb + V(-34 + lean, 18), nb + V(0 + lean, 36), nb + V(34 + lean, 16)], 2.6, k + "coll2", color=INK)
    for j, sx in enumerate((-12, 14)):
        a = nb + V(sx + lean, 46)
        ink.line([a, a + V(-2, 50)], 2.4, k + f"str{j}", smooth=False)
        ink.ellipse(a[0] - 2, a[1] + 56, 3.5, 5, fill=WHITE, w=2.2, key=k + f"aglet{j}", shift=(0, 0))
    return nb


def robber_arm(ink, p, side):
    """side 'far' (screen-left when facing left: gun hand) / 'near' (screen-right)."""
    x, y, R = p["x"], p["y"], p.get("R", 82)
    tn = p.get("turn", -0.5)
    face = 1 if tn >= 0 else -1
    hunch = p.get("hunch", 1.0)
    k = p.get("key", "rob") + side
    nb = V(x - face * 16 * hunch, y + R * 0.98 + 18)
    lean = face * 14 * hunch
    sh = nb + (V(face * 58 + lean, 34) if side == "far" else V(-face * 62 + lean, 36))
    a = p["arms"][side]
    tgt = V(*a["hand"])
    bend = a.get("bend", 1 if side == "far" else -1)
    E, Hh = ik(sh, tgt, 118, 112, bend)
    d = Hh - E
    u2 = d / max(np.linalg.norm(d), 1)
    cuff = Hh - u2 * 6
    # sleeve tube: ink under-stroke, hoodie over-stroke with the same wobble key
    pts = [sh, E, cuff]
    ink.line(pts, 40, k + "tube", taper=0.92, wob=0.6)
    ink.line(pts, 31, k + "tube", taper=0.92, wob=0.6, color=HOODIE)
    ink.line([cuff + V(-u2[1], u2[0]) * 17, cuff - V(-u2[1], u2[0]) * 17], 3.4, k + "cuff", smooth=False)
    ang = a.get("angle", math.atan2(d[1], d[0]))
    kind = a.get("kind", "relaxed")
    flip = a.get("flip", False)
    if a.get("gun"):
        gun(ink, Hh, ang, 54, key=k + "gun", flip=flip)
        kind = "grip"
    if a.get("bag"):
        paper_bag(ink, Hh + V(0, 22), key=k + "bagh", tilt=a.get("bag_tilt", 0.0))
    hand(ink, Hh, ang, a.get("size", 40), kind, k, flip=flip)
