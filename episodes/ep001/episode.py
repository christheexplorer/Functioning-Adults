"""FUNCTIONING ADULTS — Episode 001: The Robbery.

Run:  python3 episodes/ep001/episode.py            (full render -> build/ep001/)
      python3 episodes/ep001/episode.py --stills   (contact sheet of key frames)
"""
import os, sys, json, math, subprocess, argparse
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ink import *                      # noqa
from cast import *                     # noqa
import sfx
from script import LINES

FPS = 12
BUILD = os.path.join(ROOT, "build", "ep001")
# Final dialogue takes (ElevenLabs via vidIQ; see voices/CAST.md). One trimmed wav per line id.
VOICES = os.path.join(ROOT, "voices", "ep001", "wav")

# ------------------------------------------------------------------ timing
# Line start times (seconds). Pauses are the comedy; tune here.
START = {
    "r1": 0.70, "g1": 3.65, "r2": 5.80, "g2": 6.55, "r3": 8.85, "g3": 11.05,
    "g4": 12.80, "r4": 17.85, "g5": 19.60, "g6": 26.95, "g7": 27.85,
}
SHOTS = [  # (start, name)
    (0.00, "WS"),
    (3.50, "MCU_G"),
    (5.25, "MCU_R"),
    (6.45, "MCU_G"),
    (8.05, "MCU_R"),
    (10.10, "MCU_G"),
    (12.05, "WS"),
    (15.30, "INSERT"),
    (16.85, "MCU_R"),
    (19.45, "MCU_G"),
    (21.85, "WS"),
    (24.05, "INSERT"),
    (25.45, "WS"),
    (26.80, "MCU_G"),
    (30.35, "ECU_R"),
    (31.30, "BLACK"),
]
END = 33.45
SFX_CAPTION = (32.30, 33.45, "[card declined]")

# Every beat above is written against these "layout" line lengths. When real takes
# run longer or shorter, the timeline is warped: spoken spans stretch to fit the
# take, every pause between lines keeps its exact length.
LAYOUT_DUR = {"r1": 1.80, "g1": 1.11, "r2": 0.54, "g2": 1.08, "r3": 0.97, "g3": 0.83,
              "g4": 1.55, "r4": 1.31, "g5": 1.18, "g6": 0.58, "g7": 2.38}

DUR = {}
ENV = {}
WARP_OLD, WARP_NEW = [0.0], [0.0]


def to_new(t):
    return float(np.interp(t, WARP_OLD, WARP_NEW))


def to_old(T):
    return float(np.interp(T, WARP_NEW, WARP_OLD))


def build_warp():
    shift = 0.0
    for lid in sorted(START, key=START.get):
        s, od, nd = START[lid], LAYOUT_DUR[lid], DUR[lid]
        WARP_OLD.append(s); WARP_NEW.append(s + shift)
        shift += nd - od
        WARP_OLD.append(s + od); WARP_NEW.append(s + od + shift)
    WARP_OLD.append(END + 1); WARP_NEW.append(END + 1 + shift)


def load_audio():
    for lid, _, _ in LINES:
        x, sr = sf.read(os.path.join(VOICES, f"{lid}.wav"), dtype="float32")
        DUR[lid] = len(x) / sr
        hop = sr // FPS
        n = int(math.ceil(len(x) / hop))
        rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-9) for i in range(n)])
        ENV[lid] = rms / (rms.max() + 1e-9)
    build_warp()


def speaking(t, who):
    T = to_new(t)
    for lid, spk, _ in LINES:
        if spk != who:
            continue
        s = to_new(START[lid])
        if s <= T < s + DUR[lid]:
            i = int((T - s) * FPS)
            e = ENV[lid][min(i, len(ENV[lid]) - 1)]
            return 0 if e < 0.18 else (1 if e < 0.55 else 2)
    return 0


def subtitle(t):
    T = to_new(t)
    for lid, spk, text in LINES:
        s = to_new(START[lid])
        if s - 0.04 <= T < s + DUR[lid] + 0.22:
            return text
    a, b, txt = SFX_CAPTION
    if a <= t < b:
        return txt
    return None


def shot_at(t):
    cur = SHOTS[0]
    for s in SHOTS:
        if t >= s[0]:
            cur = s
    return cur


# ------------------------------------------------------------------ tracks
def _ease(u):
    return u * u * (3 - 2 * u)


def kf(t, keys, ease=True):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        if t0 <= t < t1:
            if v0 is None or isinstance(v0, str):
                return v0
            u = (t - t0) / (t1 - t0)
            u = _ease(u) if ease else u
            if isinstance(v0, tuple):
                return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def step(t, keys):
    v = keys[0][1]
    for tk, vk in keys:
        if t >= tk:
            v = vk
    return v


def blink(t, times):
    for b in times:
        d = t - b
        if 0 <= d < 1 / FPS:
            return 0.5
        if 1 / FPS <= d < 3 / FPS:
            return 1.0
        if 3 / FPS <= d < 4 / FPS:
            return 0.4
    return 0.0


# ------------------------------------------------------------------ set
COUNTER_TOP = 985
COUNTER_END = 700
FLOOR = 1380
MON_X, MON_Y = 525, 838
BAG_POS = (640, COUNTER_TOP - 118)


def draw_store(ink):
    # wall/ceiling seam + fluorescent fixture
    ink.line([(-40, 300), (1120, 296)], 3.0, "seam", color=INK_SOFT)
    for x in (360, 720):
        ink.line([(x, -40), (x + 2, 196)], 2.4, f"wire{x}", smooth=False)
    ink.shape([(300, 196), (780, 194), (790, 232), (292, 234)], fill=hexc("FBF8EF"), w=3.6, key="tube", smooth=False)
    ink.line([(320, 222), (770, 221)], 2.0, "tube2", color=GREY_M)
    # night window behind the register
    win = [(40, 360), (505, 356), (508, 800), (44, 804)]
    ink.shape(win, fill=NIGHT, w=4.4, key="win", smooth=False, shift=(2, 2))
    ink.line([(274, 360), (276, 802)], 3.6, "mullion", smooth=False)
    # street lamp outside
    ink.line([(150, 800), (148, 520), (176, 500), (210, 504)], 3.0, "lamp", color=NIGHT_D)
    ink.ellipse(214, 512, 14, 7, fill=hexc("D8D2B8"), w=0, key="lampglow", line=False)
    # OPEN sign
    ink.shape([(330, 392), (480, 392), (480, 452), (330, 452)], fill=NIGHT_D, w=3.0, key="opensign", smooth=False, shift=(1, 1))
    ink.text("OPEN", 405, 438, 46, color=SIGN_RED, key="open")
    ink.line([(345, 392), (405, 362), (465, 392)], 2.0, "signwire", smooth=False, color=INK_SOFT)
    # shelving behind the customer
    ink.line([(708, 372), (710, FLOOR)], 3.6, "shelfside", smooth=False)
    ink.line([(1076, 370), (1078, FLOOR)], 3.0, "shelfside2", smooth=False, color=INK_SOFT)
    for i, sy in enumerate((540, 712, 884, 1060)):
        ink.line([(708, sy), (1100, sy - 2)], 4.0, f"shelf{i}", smooth=False)
    rng = np.random.default_rng(3)
    prod_cols = [SAGE, MUSTARD, DUSTY, GREY_L, CREAM]
    for row, sy in enumerate((540, 712, 884, 1060)):
        x = 724
        j = 0
        while x < 1060:
            wdt = rng.uniform(42, 62)
            hgt = rng.uniform(70, 120)
            kind = (row + j) % 3
            col = prod_cols[(row * 3 + j) % len(prod_cols)]
            if kind == 0:   # chip bag
                pts = [(x, sy - hgt + 8), (x + wdt, sy - hgt), (x + wdt - 3, sy - 2), (x + 3, sy - 2)]
                ink.shape(pts, fill=col if j % 2 == 0 else None, w=3.0, key=f"p{row}_{j}", smooth=False)
            elif kind == 1:  # bottle
                bw = wdt * 0.55
                pts = [(x + 4, sy - 2), (x + 4, sy - hgt * 0.6), (x + bw / 2 - 2, sy - hgt * 0.75), (x + bw / 2 - 2, sy - hgt),
                       (x + bw / 2 + 8, sy - hgt), (x + bw / 2 + 8, sy - hgt * 0.75), (x + bw + 4, sy - hgt * 0.6), (x + bw + 4, sy - 2)]
                ink.shape(pts, fill=col if j % 3 == 1 else None, w=3.0, key=f"p{row}_{j}", smooth=False)
                wdt = bw + 10
            else:  # box
                pts = [(x, sy - hgt * 0.8), (x + wdt, sy - hgt * 0.8), (x + wdt, sy - 2), (x, sy - 2)]
                ink.shape(pts, fill=col if j % 2 == 1 else None, w=3.0, key=f"p{row}_{j}", smooth=False)
                ink.line([(x + 8, sy - hgt * 0.55), (x + wdt - 10, sy - hgt * 0.55)], 2.0, f"pl{row}_{j}", color=INK_SOFT, smooth=False)
            x += wdt + rng.uniform(8, 18)
            j += 1
    # floor
    ink.line([(-40, FLOOR + 2), (1120, FLOOR - 2)], 3.6, "floor")
    for i, fx in enumerate((150, 430, 760, 1010)):
        ink.line([(fx, FLOOR + 6), (fx - 120 + i * 60, 1960)], 2.0, f"tile{i}", color=hexc("1E1A17", 0.25), smooth=False)
    ink.line([(-40, 1640), (1120, 1636)], 2.0, "tileh", color=hexc("1E1A17", 0.22), smooth=False)


def draw_counter(ink):
    front = [(-40, COUNTER_TOP + 24), (COUNTER_END - 6, COUNTER_TOP + 24), (COUNTER_END - 4, FLOOR), (-40, FLOOR)]
    ink.shape(front, fill=hexc("E3D9CA"), w=4.2, key="cfront", smooth=False)
    top = [(-40, COUNTER_TOP), (COUNTER_END + 6, COUNTER_TOP - 2), (COUNTER_END + 2, COUNTER_TOP + 24), (-40, COUNTER_TOP + 24)]
    ink.shape(top, fill=GREY_L, w=4.2, key="ctop", smooth=False)
    ink.line([(-40, FLOOR - 40), (COUNTER_END - 6, FLOOR - 40)], 2.4, "kick", color=INK_SOFT, smooth=False)
    # candy rack on the counter front
    ink.shape([(70, 1080), (560, 1078), (562, 1250), (68, 1252)], fill=None, w=3.2, key="rack", smooth=False)
    for r, ry in enumerate((1080, 1165)):
        ink.line([(70, ry + 85), (560, ry + 83)], 2.6, f"rackl{r}", smooth=False)
        for i in range(6):
            bx = 82 + i * 80
            col = [CORAL, MUSTARD, SAGE, DUSTY, CREAM, GREY_L][(i + r * 2) % 6]
            ink.shape([(bx, ry + 22), (bx + 66, ry + 20), (bx + 64, ry + 82), (bx + 2, ry + 82)],
                      fill=col if (i + r) % 3 == 0 else None, w=2.6, key=f"candy{r}{i}", smooth=False)
    # lollipop cup
    ink.shape([(92, COUNTER_TOP - 58), (146, COUNTER_TOP - 58), (140, COUNTER_TOP + 2), (98, COUNTER_TOP + 2)],
              fill=CREAM, w=3.2, key="cup", smooth=False)
    for i, (dx, col) in enumerate(((-14, CORAL), (2, MUSTARD), (16, SAGE))):
        ink.line([(119 + dx * 0.5, COUNTER_TOP - 50), (119 + dx, COUNTER_TOP - 108)], 2.4, f"stick{i}", smooth=False)
        ink.ellipse(119 + dx, COUNTER_TOP - 116, 11, 11, fill=col, w=2.8, key=f"pop{i}")


def draw_register(ink, swivel=0.0, drawer=0.0, screen="tip", ws_text=True):
    # cash drawer on the counter, opens toward the clerk (left)
    x0, x1, y0, y1 = 445, 610, COUNTER_TOP - 52, COUNTER_TOP
    if drawer > 0:
        d = 80 * drawer
        tray = [(x0 - d, y0 + 8), (x0 + 10, y0 + 8), (x0 + 10, y1 - 4), (x0 - d, y1 - 4)]
        ink.shape(tray, fill=GREY_M, w=3.4, key="tray", smooth=False)
        for i in range(3):
            bx = x0 - d + 8 + i * (d / 3)
            ink.shape([(bx, y0 + 12), (bx + d / 3 - 10, y0 + 12), (bx + d / 3 - 10, y0 + 22), (bx, y0 + 22)],
                      fill=CASH if i < 2 else None, w=2.0, key=f"bill{i}", smooth=False)
    ink.shape([(x0, y0), (x1, y0 + 2), (x1, y1), (x0, y1)], fill=hexc("CFC5B6"), w=4.0, key="drawer", smooth=False)
    ink.line([(x0 + 14, y0 + 26), (x0 + 60, y0 + 26)], 2.4, "keyslot", smooth=False)
    # pole
    ink.line([(MON_X, y0), (MON_X + 2, MON_Y + 50)], 6, "pole", smooth=False)
    # monitor: 0 = faces Greg (we see its back), 1 = faces customer
    stage = min(3, int(swivel * 3.999))
    widths = [126, 46, 120, 180]
    w = widths[stage]
    h = 128
    cx, cy = MON_X, MON_Y
    if stage <= 1:
        back = [(cx - w / 2, cy - h / 2 + 6), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2 - 6)]
        ink.shape(back, fill=GREY_M, w=4.2, key=f"monb{stage}", smooth=False)
        if stage == 0:
            for i in range(3):
                ink.line([(cx - 28, cy - 18 + i * 14), (cx + 28, cy - 19 + i * 14)], 2.0, f"vent{i}", color=INK_SOFT, smooth=False)
        # glowing face edge on the clerk side
        ink.shape([(cx - w / 2 - 10, cy - h / 2 + 10), (cx - w / 2, cy - h / 2 + 6), (cx - w / 2, cy + h / 2 - 6), (cx - w / 2 - 10, cy + h / 2 - 10)],
                  fill=SCREEN, w=3.0, key=f"mone{stage}", smooth=False)
    else:
        bez = [(cx - w / 2, cy - h / 2 + 4), (cx + w / 2, cy - h / 2 - 2), (cx + w / 2, cy + h / 2 + 2), (cx - w / 2, cy + h / 2 - 4)]
        ink.shape(bez, fill=hexc("57504A"), w=4.2, key=f"monf{stage}", smooth=False)
        inset = 9
        scr = [(cx - w / 2 + inset, cy - h / 2 + inset + 3), (cx + w / 2 - inset, cy - h / 2 + inset - 2),
               (cx + w / 2 - inset, cy + h / 2 - inset + 2), (cx - w / 2 + inset, cy + h / 2 - inset - 3)]
        ink.fill(scr, SCREEN, f"scr{stage}", shift=(0, 0), smooth=False)
        if stage == 3 and ws_text:
            if screen == "tip":
                ink.text("Add a tip?", cx, cy - 18, 24, key="wst")
                for i, lab in enumerate(("15%", "20%", "25%")):
                    bx = cx - 50 + i * 50
                    ink.shape([(bx - 21, cy - 4), (bx + 21, cy - 4), (bx + 21, cy + 22), (bx - 21, cy + 22)], fill=None, w=2.2,
                              key=f"wsb{i}", smooth=False)
                    ink.text(lab, bx, cy + 16, 16, key=f"wsbt{i}")
            elif screen == "thanks":
                ink.text("Thank you!", cx, cy + 8, 28, key="wsty")


# ------------------------------------------------------------------ insert
def draw_insert(ink, t, t0):
    """Close-up of the tip screen, from the customer's side."""
    selected = t >= 24.90 and t0 > 20
    thanks = t >= 25.05 and t0 > 20
    # counter surface + pole
    ink.shape([(-40, 1500), (1120, 1490), (1120, 1960), (-40, 1960)], fill=GREY_L, w=4.4, key="i_ctop", smooth=False)
    ink.shape([(470, 1240), (610, 1240), (620, 1500), (460, 1500)], fill=GREY_M, w=4.4, key="i_pole", smooth=False)
    bez = [(70, 500), (1010, 488), (1016, 1262), (64, 1270)]
    ink.shape(bez, fill=hexc("57504A"), w=5.5, key="i_bez", smooth=False)
    scr = [(120, 548), (962, 538), (966, 1214), (114, 1222)]
    ink.fill(scr, SCREEN, "i_scr", shift=(0, 0), smooth=False)
    ink.line(scr, 3.0, "i_scrl", closed=True, smooth=False, color=INK_SOFT)
    if thanks:
        ink.text("Thank you!", 540, 900, 120, key="i_ty")
        ink.text("Please wait for your receipt", 540, 1000, 44, color=GREY_D, key="i_ty2")
        return
    # smiling tip jar icon
    jx, jy = 540, 650
    ink.shape([(jx - 44, jy - 40), (jx + 44, jy - 42), (jx + 50, jy + 46), (jx - 48, jy + 48)], fill=CREAM, w=3.6, key="jar")
    ink.shape([(jx - 34, jy - 58), (jx + 34, jy - 60), (jx + 36, jy - 40), (jx - 36, jy - 38)], fill=GREY_L, w=3.4, key="jarlid", smooth=False)
    ink.dot(jx - 16, jy - 4, 5); ink.dot(jx + 16, jy - 4, 5)
    ink.line([(jx - 18, jy + 16), (jx, jy + 26), (jx + 18, jy + 16)], 3.2, "jarsmile")
    ink.text("Add a tip?", 540, 820, 104, key="i_q")
    for i, lab in enumerate(("15%", "20%", "25%")):
        bx = 260 + i * 280
        on = selected and lab == "20%"
        ink.shape([(bx - 118, 870), (bx + 118, 866), (bx + 120, 1010), (bx - 116, 1014)],
                  fill=CORAL if on else hexc("FFFDF7"), w=4.4, key=f"ib{i}", smooth=False)
        ink.text(lab, bx, 970, 84, key=f"ibt{i}")
    ink.shape([(330, 1046), (750, 1042), (752, 1118), (328, 1122)], fill=hexc("FFFDF7"), w=3.6, key="ibc", smooth=False)
    ink.text("Custom amount", 540, 1098, 50, key="ibct")
    ink.text("no tip", 540, 1180, 24, color=GREY_D, key="inotip")


def draw_finger(ink, t):
    """The robber's hand entering the insert to choose 20%."""
    keys = [(24.05, (1240.0, 1620.0)), (24.40, (720.0, 1010.0)), (24.62, (740.0, 1000.0)), (24.80, (530.0, 975.0)),
            (24.90, (526.0, 962.0)), (25.00, (530.0, 978.0))]
    tip = np.array(kf(t, keys))
    ang = -2.25
    size = 160
    loc = np.array([1.31 * size, -0.12 * size])
    wrist = tip - rot(loc, ang)
    sleeve_dir = np.array([math.cos(ang), math.sin(ang)])
    a = wrist - sleeve_dir * 30
    b = wrist - sleeve_dir * 900
    n = np.array([-sleeve_dir[1], sleeve_dir[0]])
    ink.shape([a + n * 80, a - n * 80, b - n * 95, b + n * 95], fill=HOODIE, w=5.5, key="i_slv", smooth=False)
    ink.line([a + n * 40 - sleeve_dir * 120, a + n * 30 - sleeve_dir * 420], 3.0, "i_fold", color=HOODIE_D)
    ink.line([a + n * 78, a - n * 78], 5, "i_cuff", smooth=False)
    hand(ink, wrist, ang, size, "point", "i_hand", flip=True)


# ------------------------------------------------------------------ poses
G_BASE = dict(x=300, y=560, R=90, turn=0.5, key="greg")
R_BASE = dict(x=800, y=640, R=84, turn=-0.5, key="rob", lid=0.45, brow_ang=0.25, hunch=1.0)

G_ARMS = {
    "rest": dict(near=dict(hand=(196, 972), kind="flat", angle=0.1, flip=False, bend=1),
                 far=dict(hand=(392, 975), kind="flat", angle=0.05, flip=False, bend=-1)),
    "tap": dict(near=dict(hand=(196, 972), kind="flat", angle=0.1, flip=False, bend=1),
                far=dict(hand=(410, 838), kind="point", angle=-0.15, flip=False, bend=1)),
    "tap_in": dict(near=dict(hand=(196, 972), kind="flat", angle=0.1, flip=False, bend=1),
                   far=dict(hand=(398, 840), kind="point", angle=-0.15, flip=False, bend=1)),
    "swivel": dict(near=dict(hand=(196, 972), kind="flat", angle=0.1, flip=False, bend=1),
                   far=dict(hand=(452, 772), kind="grip", angle=-0.3, flip=False, bend=1)),
    "drawer": dict(near=dict(hand=(196, 972), kind="flat", angle=0.1, flip=False, bend=1),
                   far=dict(hand=(372, 950), kind="relaxed", angle=0.2, flip=False, bend=-1)),
}
R_FAR = {
    "hang": dict(hand=(752, 1060), angle=1.75, flip=False, gun=True, bend=1),
    "aim": dict(hand=(618, 800), angle=math.pi, flip=True, gun=True, bend=-1),
    "aim2": dict(hand=(630, 836), angle=math.pi - 0.28, flip=True, gun=True, bend=-1),
    "droop": dict(hand=(664, 905), angle=math.pi - 0.75, flip=True, gun=True, bend=-1),
}
R_NEAR = {
    "bag": dict(hand=(906, 1078), angle=1.5, bag=True, kind="grip", flip=False, bend=-1),
    "bag_up": dict(hand=(760, 830), angle=2.6, bag=True, kind="grip", flip=False, bend=-1, bag_tilt=0.1),
    "bag_set": dict(hand=(650, 846), angle=2.9, bag=False, kind="grip", flip=False, bend=-1),
    "hang": dict(hand=(902, 1080), angle=1.6, kind="relaxed", flip=False, bend=-1),
    "reach": dict(hand=(640, 880), angle=3.4, kind="point", flip=True, bend=-1),
    "reach_mid": dict(hand=(760, 930), angle=3.0, kind="relaxed", flip=True, bend=-1),
}


def greg_pose(t):
    p = dict(G_BASE)
    p["mouth"] = speaking(t, "GREG")
    p["blink"] = blink(t, [1.6, 4.95, 10.55, 14.25, 21.10, 23.05, 26.55, 29.35])
    # gaze
    look = step(t, [(0, "rob"), (2.85, "bag"), (3.30, "rob"), (12.05, "screen"), (13.75, "rob"), (14.45, "screen"),
                    (15.00, "rob"), (25.45, "drawer"), (26.40, "rob"), (26.80, "drawerdown"), (27.60, "rob")])
    gaze = {"rob": ((0.55, -0.05), 0.0), "bag": ((0.75, 0.6), 0.45), "screen": ((0.75, 0.25), 0.15),
            "drawer": ((0.35, 0.75), 0.55), "drawerdown": ((0.4, 0.8), 0.6)}[look]
    p["look"], p["nod"] = gaze
    arms = step(t, [(0, "rest"), (12.10, "tap"), (12.30, "tap_in"), (12.42, "tap"), (12.60, "tap_in"), (12.72, "tap"),
                    (14.30, "swivel"), (15.00, "rest"), (25.50, "drawer"), (26.05, "rest")])
    p["arms"] = G_ARMS[arms]
    return p


def rob_pose(t):
    p = dict(R_BASE)
    # entrance: three quick steps in from the right
    if t < 0.62:
        u = t / 0.62
        p["x"] = 1010 - (1010 - R_BASE["x"]) * u
        p["step"] = [0.8, 0.8, -0.8, -0.8, 0.8, 0.8, -0.4, 0][min(7, int(t * FPS))]
        p["y"] = R_BASE["y"] + (4 if int(t * FPS) % 2 else 0)
    p["mouth"] = speaking(t, "ROBBER")
    p["blink"] = blink(t, [5.45, 9.95, 16.95, 20.95, 22.75, 30.70])
    far = step(t, [(0, "hang"), (0.50, "aim"), (17.0, "aim2"), (21.85, "droop")])
    near = step(t, [(0, "bag"), (2.45, "bag_up"), (2.62, "bag_set"), (2.95, "hang"), (23.45, "reach_mid"), (23.70, "reach"),
                    (25.45, "hang")])
    p["arms"] = dict(far=R_FAR[far], near=R_NEAR[near])
    # gaze & expression beats
    look = step(t, [(0, "greg"), (15.0, "screen"), (17.05, "greg"), (22.05, "screen"), (22.60, "greg"), (23.25, "screen"),
                    (25.45, "greg"), (25.55, "drawer"), (26.35, "greg")])
    gaze = {"greg": ((-0.6, -0.05), 0.0), "screen": ((-0.75, 0.55), 0.45), "drawer": ((-0.8, 0.65), 0.5)}[look]
    p["look"], p["nod"] = gaze
    # slow head turn up from the screen (MCU 16.85)
    if 16.85 <= t < 17.75:
        u = (t - 17.05) / 0.70
        u = max(0.0, min(1.0, u))
        u = round(u * 7) / 7  # limited animation: 7 drawings
        p["look"] = tuple(a + (b - a) * u for a, b in zip((-0.75, 0.55), (-0.6, -0.05)))
        p["nod"] = 0.45 * (1 - u)
    exp = step(t, [(0, "mean"), (5.25, "huh"), (6.45, "flat"), (8.30, "narrow"), (10.10, "flat"), (17.70, "incred"),
                   (19.45, "flat"), (21.85, "unsure"), (30.35, "blank")])
    p.update({
        "mean": dict(lid=0.42, brow_ang=0.55, brow_raise=0.0, frown=0.6),
        "huh": dict(lid=0.12, brow_ang=-0.35, brow_raise=0.8, frown=0.2),
        "flat": dict(lid=0.45, brow_ang=0.2, brow_raise=0.0, frown=0.5),
        "narrow": dict(lid=0.62, brow_ang=0.7, brow_raise=-0.2, frown=0.8),
        "incred": dict(lid=0.30, brow_ang=-0.1, brow_raise=0.5, frown=0.4),
        "unsure": dict(lid=0.40, brow_ang=-0.3, brow_raise=0.25, frown=0.55),
        "blank": dict(lid=0.38, brow_ang=0.0, brow_raise=0.1, frown=0.3),
    }[exp])
    return p


def props(t):
    sw = kf(t, [(0, 0.0), (14.45, 0.0), (14.95, 1.0)], ease=False)
    sw = round(sw * 3) / 3
    drawer = 1.0 if t >= 25.55 else 0.0
    screen = "thanks" if t >= 25.05 else "tip"
    return dict(swivel=sw, drawer=drawer, screen=screen, bag_on=t >= 2.62)


CAMS = {
    "WS": Camera(565, 1000, 1.2),
    "MCU_G": Camera(352, 772, 2.05),
    "MCU_R": Camera(770, 772, 2.05),
}


# ------------------------------------------------------------------ frame
def draw_world(ink, t):
    g, r, pr = greg_pose(t), rob_pose(t), props(t)
    draw_store(ink)
    draw_counter(ink)
    draw_greg_upper(ink, g, clip_y=COUNTER_TOP + 2)
    draw_register(ink, pr["swivel"], pr["drawer"], pr["screen"])
    greg_arm(ink, g, "near")
    greg_arm(ink, g, "far")
    if pr["bag_on"]:
        paper_bag(ink, BAG_POS, key="bagc", open_top=True)
    robber_arm(ink, r, "far")
    robber_body(ink, r)
    robber_head(ink, r)
    robber_arm(ink, r, "near")


def draw_subtitle(canvas, text):
    import skia
    f = font_obj("PatrickHand-Regular.ttf", 80)
    words = text.split()
    lines, cur = [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if f.measureText(trial) > 880 and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = trial
    lines.append(cur)
    if len(lines) == 2:  # balance the two lines instead of leaving an orphan word
        best = min(range(1, len(words)), key=lambda k: max(f.measureText(" ".join(words[:k])), f.measureText(" ".join(words[k:]))))
        lines = [" ".join(words[:best]), " ".join(words[best:])]
    y0 = 1565 - (len(lines) - 1) * 86
    halo = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*PAPER))
    halo.setStyle(skia.Paint.kStroke_Style)
    halo.setStrokeWidth(16)
    halo.setStrokeJoin(skia.Paint.kRound_Join)
    fill = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*INK))
    sp = f.measureText(" ")
    for i, ln in enumerate(lines):
        wdt = f.measureText(ln)
        x, y = 540 - wdt / 2, y0 + i * 86
        # draw word by word: a stroked space glyph leaves a dot
        for pass_paint in (halo, fill):
            cx = x
            for w_ in ln.split(" "):
                canvas.drawString(w_, cx, y, f, pass_paint)
                cx += f.measureText(w_) + sp


def render_frame(i):
    import skia
    t = to_old(i / FPS)   # animation runs in layout time
    st, name = shot_at(t)
    if name == "BLACK":
        surf = skia.Surface(W, H)
        c = surf.getCanvas()
        c.clear(skia.ColorBLACK)
        a, b, txt = SFX_CAPTION
        if a <= t < b:
            f = font_obj("PatrickHand-Regular.ttf", 60)
            p = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(0.62, 0.58, 0.53, 1))
            c.drawString(txt, 540 - f.measureText(txt) / 2, 1565, f, p)
        return surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)
    surf, c = new_frame()
    if name == "INSERT":
        ink = Ink(c, Camera(), i)
        draw_insert(ink, t, st)
        if st > 20:
            draw_finger(ink, t)
    elif name == "ECU_R":
        r = rob_pose(t)
        tn = r["turn"]
        cam = Camera(r["x"] + tn * 0.30 * r["R"], r["y"] - 0.06 * r["R"], 8.6)
        ink = Ink(c, cam, i, wobble=1.6)
        draw_world(ink, t)
    else:
        ink = Ink(c, CAMS[name], i)
        draw_world(ink, t)
    sub = subtitle(t)
    if sub and not sub.startswith("["):
        draw_subtitle(c, sub)
    return finish(surf, c)


# ------------------------------------------------------------------ audio
def build_audio(path):
    sr = 48000
    n = int(to_new(END) * sr)
    mix = np.zeros(n, np.float32)

    def place(x, t, gain=1.0, src_sr=sr):
        if src_sr != sr:
            idx = np.arange(0, len(x), src_sr / sr)
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        a = int(t * sr)
        b = min(n, a + len(x))
        if a < n:
            mix[a:b] += x[: b - a] * gain

    black_t = [s for s, nme in SHOTS if nme == "BLACK"][0]
    room = sfx.room_tone(to_new(black_t), sr)
    place(room, 0, 1.0)
    for lid, spk, _ in LINES:
        x, xsr = sf.read(os.path.join(VOICES, f"{lid}.wav"), dtype="float32")
        x = x / (np.abs(x).max() + 1e-9) * 0.62
        place(x, to_new(START[lid]), 1.0, xsr)
    _place = place

    def place(x, t, gain=1.0, src_sr=sr):  # SFX cue times are written in layout time
        _place(x, to_new(t), gain, src_sr)
    for k, ts in enumerate((0.05, 0.22, 0.40, 0.57)):
        place(sfx.footstep(sr, k), ts, 0.30)
    place(sfx.paper_bag(sr), 2.62, 0.45)
    for ts in (12.30, 12.60):
        place(sfx.pos_beep(sr), ts, 0.16)
    place(sfx.swivel(sr), 14.50, 0.20)
    place(sfx.pos_beep(sr), 24.90, 0.16)
    place(sfx.drawer(sr), 25.55, 0.50)
    place(sfx.chip_insert(sr), 31.62, 0.40)
    place(sfx.decline(sr), 32.30, 0.30)
    peak = np.abs(mix).max()
    if peak > 0.97:
        mix *= 0.97 / peak
    sf.write(path, mix, sr)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--from", dest="t0", type=float, default=0)
    ap.add_argument("--to", dest="t1", type=float, default=END)
    args = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    load_audio()
    if args.stills:
        from PIL import Image
        times = [0.3, 1.5, 4.0, 5.9, 7.0, 9.0, 11.3, 13.0, 14.8, 15.8, 17.2, 18.3, 20.0, 22.3, 24.5, 24.95, 25.9, 27.2, 29.0, 30.7, 32.5]
        ims = [Image.fromarray(render_frame(int(round(to_new(tt) * FPS)))).convert("RGB").resize((270, 480)) for tt in times]
        cols = 7
        sheet = Image.new("RGB", (cols * 270, ((len(ims) + cols - 1) // cols) * 480), "white")
        for k, im in enumerate(ims):
            sheet.paste(im, ((k % cols) * 270, (k // cols) * 480))
        sheet.save(os.path.join(BUILD, "contact.png"))
        print("contact sheet ->", os.path.join(BUILD, "contact.png"))
        return
    wav = os.path.join(BUILD, "mix.wav")
    build_audio(wav)
    out = os.path.join(BUILD, "FunctioningAdults_E001_TheRobbery.mp4")
    end = to_new(END)
    nframes = int(round(end * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-framerate", str(FPS),
           "-i", "-", "-i", wav, "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-tune", "animation", "-af", "volume=5.5dB,alimiter=limit=0.66:level=false", "-c:a", "aac", "-b:a", "192k", "-t", f"{end:.3f}", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(nframes):
        proc.stdin.write(render_frame(i).tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{nframes}", flush=True)
    proc.stdin.close()
    proc.wait()
    write_srt(os.path.join(BUILD, "FunctioningAdults_E001_TheRobbery.srt"))
    print("done ->", out)


def write_srt(path):
    def ts(x):
        h, m = int(x // 3600), int(x % 3600 // 60)
        s = x % 60
        return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"
    a, b, cap = SFX_CAPTION
    rows = [(to_new(START[l]), to_new(START[l]) + DUR[l] + 0.22, txt) for l, _, txt in LINES] + [(to_new(a), to_new(b), cap)]
    rows = [(a, min(b, rows[k + 1][0] - 0.02) if k + 1 < len(rows) else b, txt) for k, (a, b, txt) in enumerate(rows)]
    with open(path, "w") as f:
        for k, (a, b, txt) in enumerate(rows, 1):
            f.write(f"{k}\n{ts(a)} --> {ts(b)}\n{txt}\n\n")


if __name__ == "__main__":
    main()
