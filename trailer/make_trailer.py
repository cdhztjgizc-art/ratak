#!/usr/bin/env python3
"""
Generátor traileru pro hru ČERNÝ OREL.

Obraz i zvuk se generují procedurálně (PIL + numpy + scipy), takže nejsou potřeba
žádné externí assety. Texty, herní módy, zbraně a attachmenty se upravují v config.json.

Použití:
    python3 make_trailer.py                  # celý trailer -> output/cerny_orel_trailer.mp4
    python3 make_trailer.py --stills 3 12.5  # náhledové PNG snímky v zadaných časech (s)
    python3 make_trailer.py --audio-only     # jen zvuková stopa (output/trailer_audio.wav)

Struktura traileru (sekundy):
    0–6    intro (texty, atmosféra)
    6–10   logo
    10–22  gameplay (3 záběry)
    22–34  herní módy
    34–45  zbrojírna
    45–57  attachmenty
    57–59  rychlý střih
    59–66  finále s logem
"""
import argparse
import json
import math
import os
import subprocess
import sys
import wave
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
DURATION = 66.0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
    CFG = json.load(fh)

GOLD = (232, 168, 61)
GOLD_LIGHT = (255, 214, 140)
WHITE = (240, 240, 236)
GRAY = (150, 152, 158)
RED = (214, 52, 46)
GREEN = (110, 214, 120)
BLACK = (0, 0, 0)

# Časová osa
T_LOGO, T_GAME, T_MODES, T_ARMORY, T_ATT, T_MONTAGE, T_FINAL = 6.0, 10.0, 22.0, 34.0, 45.0, 57.0, 59.0
MODE_START, MODE_LEN = 24.0, 2.5
WEAPON_START, WEAPON_LEN = 36.0, 3.0
ATT_START, ATT_STEP, ATT_FLY = 46.5, 1.8, 0.45
CUTS = [T_GAME, 14.0, 18.0, T_MODES, T_ARMORY, T_ATT]


# --------------------------------------------------------------------------- utility
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def prog(t, a, b):
    return clamp((t - a) / (b - a))


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in(t):
    t = clamp(t)
    return t ** 3


def ease_io(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def window(t, a, b, fin=0.3, fout=0.3):
    return clamp(min(prog(t, a, a + fin), 1 - prog(t, b - fout, b)))


def mixc(c1, c2, t):
    return tuple(int(lerp(c1[k], c2[k], t)) for k in range(3))


_STATIC = {}


def static(name, fn):
    if name not in _STATIC:
        _STATIC[name] = fn()
    return _STATIC[name]


# --------------------------------------------------------------------------- text
@lru_cache(maxsize=None)
def font(size, weight="Black"):
    for path in (f"/usr/share/fonts/opentype/inter/InterDisplay-{weight}.otf",
                 f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_w(txt, size, weight="Black", tracking=0):
    f = font(size, weight)
    return sum(f.getlength(c) for c in txt) + tracking * max(len(txt) - 1, 0)


def fit_size(txt, max_w, size, weight="Black", tracking=0):
    while size > 12 and text_w(txt, size, weight, tracking) > max_w:
        size -= 2
    return size


@lru_cache(maxsize=2048)
def text_mask(txt, size, weight, tracking, pad, blur):
    f = font(size, weight)
    asc, desc = f.getmetrics()
    w = int(math.ceil(text_w(txt, size, weight, tracking))) + 2 * pad
    m = Image.new("L", (max(w, 1), asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(m)
    x = pad
    for c in txt:
        d.text((x, pad), c, font=f, fill=255)
        x += f.getlength(c) + tracking
    if blur:
        m = m.filter(ImageFilter.GaussianBlur(blur))
    return m


def paste_color(frame, color, xy, mask, alpha=1.0):
    if alpha <= 0.004:
        return
    if alpha < 0.996:
        mask = mask.point(lambda v, a=alpha: int(v * a))
    x, y = int(xy[0]), int(xy[1])
    frame.paste(color, (x, y, x + mask.size[0], y + mask.size[1]), mask)


def put_text(frame, txt, pos, size, color=WHITE, weight="Black", tracking=0, anchor="mm",
             alpha=1.0, glow=0, glow_strength=0.9):
    """Vykreslí text s proložením; svislé zarovnání se počítá podle výšky verzálek."""
    tracking = int(round(tracking))
    pad = int(glow * 2.5) if glow else 0
    m = text_mask(txt, size, weight, tracking, pad, 0)
    f = font(size, weight)
    cap_top, cap_bot = f.getbbox("H")[1], f.getbbox("H")[3]
    w = m.size[0]
    x, y = pos
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w - pad
    else:
        x -= pad
    if anchor[1] == "m":
        y -= pad + (cap_top + cap_bot) / 2
    elif anchor[1] == "b":
        y -= pad + cap_bot
    else:
        y -= pad + cap_top
    if glow:
        paste_color(frame, color, (x, y), text_mask(txt, size, weight, tracking, pad, glow), alpha * glow_strength)
    paste_color(frame, color, (x, y), m, alpha)
    return (x + pad, y + pad + cap_top, x + w - pad, y + pad + cap_bot)


# --------------------------------------------------------------------------- shared textures
def vgrad(w, h, top, bottom):
    t = np.linspace(0, 1, h)[:, None, None]
    arr = (np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t)
    return Image.fromarray(np.repeat(arr, w, axis=1).astype(np.uint8), "RGB")


def radial_mask(w, h, cx, cy, r, power=2.0, gain=255):
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    v = np.clip(1 - d, 0, 1) ** power
    return Image.fromarray((v * gain).astype(np.uint8), "L")


def vignette_mask():
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    v = np.clip(1.15 - 0.55 * d ** 2.2, 0, 1)
    return Image.fromarray((v * 255).astype(np.uint8), "L")


def grain_frames():
    rng = np.random.default_rng(7)
    out = []
    for _ in range(6):
        n = np.clip(rng.normal(14, 5, (H, W)), 0, 28).astype(np.uint8)
        g = Image.fromarray(n, "L")
        out.append(Image.merge("RGB", (g, g, g)))
    return out


def fog_mask():
    rng = np.random.default_rng(11)
    small = Image.fromarray((rng.random((14, 40)) * 255).astype(np.uint8), "L")
    big = small.resize((W + 900, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(40))
    arr = np.asarray(big).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return Image.fromarray((arr ** 1.6 * 255).astype(np.uint8), "L")


def draw_fog(frame, t, color=(60, 62, 70), alpha=0.35, speed=40):
    fog = static("fog", fog_mask)
    ox = int((t * speed) % 900)
    paste_color(frame, color, (0, 0), fog.crop((ox, 0, ox + W, H)), alpha)


def draw_embers(frame, t, n=70, seed=5, alpha=1.0, speed=1.0):
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(0, W, n), rng.uniform(0, H, n)
    vy, vx = rng.uniform(25, 95, n) * speed, rng.uniform(-25, 25, n)
    sz, ph = rng.uniform(1.4, 4.2, n), rng.uniform(0, 6.28, n)
    d = ImageDraw.Draw(frame, "RGBA")
    for k in range(n):
        x = (xs[k] + vx[k] * t + 30 * math.sin(t * 0.7 + ph[k])) % W
        y = (ys[k] - vy[k] * t) % (H + 40) - 20
        fl = 0.5 + 0.5 * math.sin(t * 5 + ph[k] * 3)
        a = int(255 * alpha * (0.3 + 0.7 * fl))
        r = sz[k]
        d.ellipse((x - r * 3.5, y - r * 3.5, x + r * 3.5, y + r * 3.5), fill=(255, 130, 40, int(a * 0.10)))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 196, 120, a))


def flash(frame, color, a):
    if a <= 0.004:
        return frame
    return Image.blend(frame, Image.new("RGB", frame.size, color), clamp(a))


def glitch(frame, amount, seed):
    arr = np.asarray(frame).copy()
    rng = np.random.default_rng(seed)
    for _ in range(int(8 * amount) + 1):
        y = int(rng.integers(0, H - 60))
        hh = int(rng.integers(6, 70))
        dx = int(rng.integers(-120, 120) * amount)
        arr[y:y + hh] = np.roll(arr[y:y + hh], dx, axis=1)
    s = int(14 * amount)
    if s:
        arr[:, :, 0] = np.roll(arr[:, :, 0], s, axis=1)
        arr[:, :, 2] = np.roll(arr[:, :, 2], -s, axis=1)
    return Image.fromarray(arr)


def letterbox(frame, k=1.0):
    bar = int(138 * k)
    if bar > 0:
        d = ImageDraw.Draw(frame)
        d.rectangle((0, 0, W, bar), fill=BLACK)
        d.rectangle((0, H - bar, W, H), fill=BLACK)


# --------------------------------------------------------------------------- emblem (orel)
def _emblem_base():
    S = 2  # supersampling
    m = Image.new("L", (1000 * S, 1000 * S), 0)
    d = ImageDraw.Draw(m)

    def P(pts, fill=255, mirror=False):
        pp = [((1000 - x) if mirror else x, y) for x, y in pts]
        d.polygon([(x * S, y * S) for x, y in pp], fill=fill)

    feathers = [((560, 350), (566, 396), (930, 168)), ((566, 386), (571, 432), (968, 282)),
                ((569, 422), (569, 468), (955, 392)), ((566, 458), (560, 504), (905, 494)),
                ((558, 494), (550, 538), (822, 584)), ((548, 528), (534, 568), (718, 652))]
    for mirror in (False, True):
        P([(530, 330), (650, 296), (612, 430), (548, 450)], mirror=mirror)
        for a, b, tip in feathers:
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            dx, dy = tip[0] - mx, tip[1] - my
            ln = math.hypot(dx, dy)
            nx, ny = -dy / ln, dx / ln
            notch = (tip[0] - 0.2 * dx + nx * 20, tip[1] - 0.2 * dy + ny * 20)
            P([a, tip, notch, b], mirror=mirror)
    P([(455, 320), (540, 325), (562, 400), (552, 520), (520, 600), (500, 632), (480, 600), (448, 520), (438, 400)])
    P([(462, 588), (538, 588), (588, 722), (540, 700), (500, 748), (460, 700), (412, 722)])
    P([(452, 332), (460, 262), (488, 228), (530, 216), (566, 230), (618, 256), (626, 284),
       (606, 308), (600, 286), (572, 280), (546, 294), (540, 338)])
    P([(520, 256), (552, 246), (542, 264)], fill=0)  # oko
    # dělicí rýhy v tělě
    d.line([(500 * S, 360 * S), (500 * S, 590 * S)], fill=0, width=6 * S)
    m = m.resize((1000, 1000), Image.LANCZOS)
    grad = vgrad(1000, 1000, GOLD_LIGHT, (160, 98, 26))
    edge = ImageChops.subtract(m, m.filter(ImageFilter.MinFilter(7)))
    grad.paste((255, 236, 190), (0, 0, 1000, 1000), edge)
    img = grad.convert("RGBA")
    img.putalpha(m)
    return img


@lru_cache(maxsize=96)
def emblem(size):
    return static("emblem", _emblem_base).resize((size, size), Image.LANCZOS)


def paste_emblem(frame, cx, cy, size, alpha=1.0, glow=True):
    size = max(int(size), 8)
    e = emblem(size)
    x, y = int(cx - size / 2), int(cy - size / 2)
    a = e.getchannel("A")
    if glow:
        g = a.resize((max(size // 4, 1),) * 2).filter(ImageFilter.GaussianBlur(6)).resize((size, size))
        paste_color(frame, (255, 150, 40), (x, y), g, 0.55 * alpha)
    if alpha < 0.996:
        a = a.point(lambda v: int(v * alpha))
    frame.paste(e.convert("RGB"), (x, y), a)


# --------------------------------------------------------------------------- zbraně
BW, BH = 1700, 640
ROLES = {"body": (46, 48, 53), "part": (64, 67, 73), "dark": (26, 27, 30), "light": (110, 113, 120),
         "accent": GOLD, "red": (235, 40, 40), "lens": (70, 150, 200)}
EDGE = (92, 95, 102)


def R(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def weapon_shapes(kind):
    """Seznam (slot, typ, body, role, šířka). slot None = základ zbraně."""
    s = []

    def add(pts, role="part", slot=None, kind_="poly", w=0):
        s.append((slot, kind_, pts, role, w))

    if kind == "ar":
        add([(650, 312), (735, 312), (752, 380), (778, 470), (700, 484), (676, 392)], "part", "stdmag")
        for y in (350, 390, 430):
            add([(672, y + 6), (752, y - 4)], "dark", "stdmag", "line", 3)
        add([(650, 312), (735, 312), (760, 400), (802, 556), (716, 574), (680, 410)], "part", "mag")
        add([(710, 566), (804, 550), (810, 572), (714, 590)], "dark", "mag")
        for y in (360, 410, 460, 510):
            add([(678 + (y - 360) * 0.2, y + 6), (760 + (y - 360) * 0.25, y - 6)], "dark", "mag", "line", 3)
        add([(70, 250), (120, 232), (330, 226), (330, 312), (250, 316), (120, 352), (70, 356)], "body")
        add(R(56, 240, 80, 362), "dark")
        add([(150, 264), (290, 258), (290, 286), (170, 302)], "dark")
        add(R(300, 238, 340, 292), "part")
        add([(330, 250), (762, 250), (762, 312), (600, 316), (560, 332), (330, 322)], "body")
        add(R(330, 205, 770, 255), "part")
        add(R(470, 224, 560, 244), "dark")
        add(R(318, 212, 350, 226), "light")
        add(R(350, 186, 760, 206), "dark")
        for x in range(356, 750, 22):
            add(R(x, 178, x + 12, 190), "part")
        add(R(420, 274, 470, 284), "accent")
        add([(770, 208), (1110, 214), (1110, 296), (770, 302)], "body")
        for x in range(800, 1080, 48):
            add(R(x, 242, x + 30, 262), "dark")
        add(R(780, 196, 1100, 210), "dark")
        add([(1080, 226), (1120, 226), (1120, 276), (1080, 276)], "part")
        add(R(1110, 240, 1310, 262), "dark")
        add(R(1300, 230, 1352, 272), "part", "brake")
        add(R(1314, 236, 1320, 266), "dark", "brake")
        add(R(1330, 236, 1336, 266), "dark", "brake")
        add([(1060, 206), (1072, 160), (1090, 160), (1098, 206)], "part")
        add([(372, 186), (380, 152), (420, 152), (428, 186)], "part")
        add([(565, 316), (625, 316), (612, 440), (552, 440), (545, 420)], "part")
        add([(540, 322), (540, 348), (640, 348), (660, 320)], "body", None, "line", 8)
        add([(585, 318), (593, 318), (602, 340), (593, 343)], "light")
        # attachmenty
        add(R(500, 160, 640, 186), "dark", "optic")
        add([(510, 160), (522, 110), (628, 110), (640, 160)], "part", "optic")
        add(R(530, 122, 616, 150), "dark", "optic")
        add(R(626, 118, 640, 156), "red", "optic")
        add([(522, 112), (628, 112)], "light", "optic", "line", 4)
        add(R(1300, 222, 1600, 280), "part", "muzzle")
        for x in (1332, 1560):
            add(R(x, 222, x + 8, 280), "dark", "muzzle")
        add(R(1592, 228, 1610, 274), "dark", "muzzle")
        add([(1310, 226), (1590, 226)], "light", "muzzle", "line", 4)
        add(R(890, 294, 962, 306), "dark", "grip")
        add([(900, 304), (952, 304), (946, 414), (906, 414)], "part", "grip")
        for y in (330, 356, 382):
            add([(904, y), (948, y)], "dark", "grip", "line", 3)
        add(R(978, 296, 1092, 326), "part", "laser")
        add(R(1086, 302, 1100, 320), "red", "laser")
        add(R(990, 304, 1040, 312), "dark", "laser")
    elif kind == "sniper":
        add(R(590, 298, 650, 360), "part")
        add([(40, 240), (90, 222), (380, 236), (380, 320), (300, 322), (190, 348), (90, 364), (40, 366)], "body")
        add(R(28, 232, 46, 372), "dark")
        add(R(160, 206, 330, 238), "part")
        add([(190, 266), (300, 262), (300, 296), (210, 312)], "dark")
        add(R(380, 220, 700, 300), "part")
        add([(700, 226), (1000, 232), (1000, 290), (700, 298)], "body")
        for x in range(730, 980, 46):
            add(R(x, 250, x + 28, 268), "dark")
        add(R(1000, 244, 1560, 262), "dark")
        add(R(1550, 234, 1604, 272), "part")
        add(R(1566, 240, 1572, 266), "dark")
        add([(470, 300), (525, 300), (515, 420), (462, 420)], "part")
        add([(450, 306), (450, 334), (540, 334), (560, 304)], "body", None, "line", 8)
        add([(560, 300), (604, 338)], "light", None, "line", 10)
        add([(590, 326), (616, 326), (616, 350), (590, 350)], "light")
        add([(968, 292), (938, 412)], "dark", None, "line", 10)
        add([(982, 292), (1012, 412)], "dark", None, "line", 10)
        add(R(500, 182, 530, 222), "dark")
        add(R(800, 182, 830, 222), "dark")
        add(R(440, 150, 880, 184), "part")
        add([(880, 150), (944, 132), (944, 202), (880, 184)], "part")
        add([(440, 150), (398, 140), (398, 194), (440, 184)], "part")
        add(R(640, 122, 678, 150), "part")
        add(R(936, 140, 946, 194), "lens")
        add([(446, 154), (876, 154)], "light", None, "line", 4)
        add(R(420, 268, 470, 278), "accent")
    else:  # smg
        add([(640, 300), (700, 300), (714, 500), (654, 500)], "part")
        for y in (340, 380, 420, 460):
            add([(646, y), (704, y)], "dark", None, "line", 3)
        add([(140, 230), (380, 238)], "part", None, "line", 12)
        add([(140, 302), (380, 292)], "part", None, "line", 12)
        add(R(126, 220, 152, 314), "dark")
        add(R(380, 212, 820, 300), "body")
        add(R(420, 196, 800, 214), "dark")
        for x in range(426, 794, 22):
            add(R(x, 188, x + 12, 198), "part")
        add(R(520, 230, 600, 250), "dark")
        add(R(820, 228, 960, 284), "part")
        for x in range(836, 950, 30):
            add(R(x, 244, x + 18, 266), "dark")
        add(R(960, 246, 1030, 264), "dark")
        add(R(1020, 238, 1052, 272), "part")
        add([(870, 284), (912, 284), (904, 336), (876, 336)], "part")
        add([(520, 300), (575, 300), (565, 420), (510, 420)], "part")
        add([(500, 306), (500, 334), (600, 334), (618, 304)], "body", None, "line", 8)
        add([(432, 196), (438, 168), (458, 168), (464, 196)], "part")
        add([(776, 196), (782, 168), (798, 168), (804, 196)], "part")
        add(R(420, 266, 470, 276), "accent")
    return s


@lru_cache(maxsize=64)
def weapon_sprite(kind, width, attached=(), only=None, hide=(), style="solid"):
    SS = 2
    img = Image.new("RGBA", (BW * SS, BH * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for slot, typ, pts, role, w in weapon_shapes(kind):
        if only is not None:
            if slot != only:
                continue
        elif slot == "stdmag":
            if "mag" in attached or "mag" in hide:
                continue
        elif slot == "brake":
            if "muzzle" in attached or "muzzle" in hide:
                continue
        elif slot is not None and slot not in attached:
            continue
        pp = [(x * SS, y * SS) for x, y in pts]
        if style == "blueprint":
            col = GOLD if role in ("red", "accent") else (14, 20, 28)
            if typ == "line":
                d.line(pp, fill=GOLD, width=max(2, w))
            else:
                d.polygon(pp, fill=col + (255,), outline=GOLD, width=4)
        else:
            col = ROLES[role]
            if typ == "line":
                d.line(pp, fill=col, width=w * SS, joint="curve")
            else:
                d.polygon(pp, fill=col, outline=EDGE if role not in ("red", "accent", "lens") else None, width=3)
    h = int(round(width * BH / BW))
    return img.resize((width, h), Image.LANCZOS)


# --------------------------------------------------------------------------- vojáci (siluety)
SOLDIER_PARTS = [
    [(-0.15, -0.80), (0.15, -0.80), (0.13, -0.47), (-0.13, -0.47)],
    [(-0.21, -0.78), (-0.14, -0.79), (-0.14, -0.55), (-0.21, -0.57)],
    [(-0.13, -0.50), (0.13, -0.50), (0.12, -0.40), (-0.12, -0.40)],
    [(0.08, -0.79), (0.16, -0.75), (0.27, -0.66), (0.20, -0.62)],
    [(-0.22, -0.69), (0.42, -0.72), (0.42, -0.68), (-0.22, -0.64)],
    [(-0.05, -0.80), (0.05, -0.80), (0.04, -0.84), (-0.04, -0.84)],
]


def soldier_polys(stride):
    head = [(0.095 * math.cos(a), -0.905 + 0.092 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 14)[:-1]]
    s = 0.09 * stride
    legs = [[(-0.12, -0.42), (-0.01, -0.42), (-0.03 + s, 0), (-0.13 + s, 0)],
            [(0.01, -0.42), (0.12, -0.42), (0.13 - s, 0), (0.03 - s, 0)]]
    return SOLDIER_PARTS + [head] + legs


def soldier_mask(h, stride=0.0, angle=0.0, facing=1):
    """Vrátí (mask, offset_x, offset_y) relativně k pozici chodidel."""
    SS = 2
    size = int(h * 2.4) + 4
    m = Image.new("L", (size * SS, size * SS), 0)
    d = ImageDraw.Draw(m)
    ca, sa = math.cos(angle), math.sin(angle)
    c = size * SS / 2
    for poly in soldier_polys(stride):
        pts = []
        for x, y in poly:
            x *= facing
            rx, ry = x * ca - y * sa, x * sa + y * ca
            pts.append((c + rx * h * SS, c + ry * h * SS))
        d.polygon(pts, fill=255)
    m = m.resize((size, size), Image.LANCZOS)
    return m, -size / 2, -size / 2


def draw_soldier(frame, x, foot, h, stride=0.0, angle=0.0, facing=1, alpha=1.0, rim=(200, 110, 50),
                 body=(14, 13, 15)):
    if alpha <= 0.01 or h < 4:
        return
    m, ox, oy = soldier_mask(h, stride, angle, facing)
    if rim:
        paste_color(frame, rim, (x + ox + 2, foot + oy - 1), m, alpha * 0.8)
    paste_color(frame, body, (x + ox, foot + oy), m, alpha)


# --------------------------------------------------------------------------- herní svět
PALETTES = {
    "dusk": dict(top=(16, 20, 32), hz=(206, 112, 54), ground=(40, 30, 26), ground2=(12, 10, 10),
                 sun=(255, 176, 96), far=(118, 70, 52), near=(22, 18, 20), win=(255, 170, 80)),
    "night": dict(top=(4, 8, 16), hz=(40, 62, 86), ground=(18, 24, 30), ground2=(6, 8, 10),
                  sun=(150, 190, 230), far=(30, 42, 58), near=(10, 12, 16), win=(255, 210, 140)),
    "fire": dict(top=(22, 14, 14), hz=(160, 64, 30), ground=(38, 24, 20), ground2=(10, 8, 8),
                 sun=(255, 120, 50), far=(90, 44, 30), near=(18, 14, 14), win=(255, 140, 60)),
}
WW, WH, HZ = 3400, H + 120, 620  # svět, horizont ve světových souřadnicích


def build_world(kind):
    pal = PALETTES[kind]
    rng = np.random.default_rng({"dusk": 1, "night": 2, "fire": 3}[kind])
    sky = np.asarray(vgrad(1, HZ, pal["top"], pal["hz"])).astype(np.float32)
    ground = np.asarray(vgrad(1, WH - HZ, pal["ground"], pal["ground2"])).astype(np.float32)
    col = np.concatenate([sky, ground], axis=0)
    arr = np.repeat(col, WW, axis=1)
    sx, sy = WW * 0.42, HZ - 70
    yy, xx = np.mgrid[0:WH, 0:WW]
    glow = np.clip(1 - np.sqrt((xx - sx) ** 2 + ((yy - sy) * 1.8) ** 2) / 1300, 0, 1) ** 2.2
    arr += glow[:, :, None] * np.array(pal["sun"])[None, None, :] * 0.55
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img, "RGBA")
    if kind == "night":
        for _ in range(260):
            x, y = rng.uniform(0, WW), rng.uniform(0, HZ - 120)
            b = int(rng.uniform(60, 220))
            d.point((x, y), fill=(b, b, b + 20))
        d.ellipse((2200, 120, 2290, 210), fill=(220, 226, 236))
    # hory
    xs = np.arange(0, WW + 40, 40)
    ph = rng.uniform(0, 6.28, 3)
    hs = 70 + 40 * np.sin(xs / 380 + ph[0]) + 25 * np.sin(xs / 140 + ph[1]) + 12 * np.sin(xs / 55 + ph[2])
    d.polygon([(0, HZ)] + [(x, HZ - h) for x, h in zip(xs, hs)] + [(WW, HZ)], fill=mixc(pal["hz"], pal["top"], 0.45))
    # vzdálené město
    x = 0
    while x < WW:
        w = rng.uniform(50, 170)
        h = rng.uniform(60, 240)
        top = HZ - h
        if rng.random() < 0.35:
            d.polygon([(x, HZ), (x, top + rng.uniform(0, 40)), (x + w * 0.4, top), (x + w * 0.7, top + rng.uniform(20, 70)),
                       (x + w, top + rng.uniform(0, 30)), (x + w, HZ)], fill=pal["far"])
        else:
            d.rectangle((x, top, x + w, HZ), fill=pal["far"])
        x += w + rng.uniform(-10, 40)
    # opar u horizontu
    for k in range(40):
        a = int(110 * (1 - k / 40))
        d.line([(0, HZ - 60 + k * 3), (WW, HZ - 60 + k * 3)], fill=pal["hz"] + (a // 3,), width=3)
    # silnice
    vx = WW / 2
    d.polygon([(vx - 20, HZ), (vx + 20, HZ), (vx + 1900, WH), (vx - 1900, WH)], fill=mixc(pal["ground"], (0, 0, 0), 0.35))
    for k in range(-6, 7):
        d.line([(vx + k * 6, HZ), (vx + k * 600, WH)], fill=pal["hz"] + (22,), width=2)
    # blízké ruiny
    x = rng.uniform(-100, 100)
    while x < WW:
        w = rng.uniform(160, 320)
        h = rng.uniform(220, 470)
        base = HZ + rng.uniform(10, 40)
        top = base - h
        pts = [(x, base), (x, top + rng.uniform(0, 60))]
        nseg = 5
        for k in range(1, nseg):
            pts.append((x + w * k / nseg, top + rng.uniform(0, 90) * (1 if rng.random() < 0.6 else 2)))
        pts += [(x + w, top + rng.uniform(0, 60)), (x + w, base)]
        d.polygon(pts, fill=pal["near"])
        for wy in np.arange(top + 70, base - 30, 46):
            for wx in np.arange(x + 18, x + w - 30, 38):
                r = rng.random()
                if r < 0.07:
                    d.rectangle((wx, wy, wx + 16, wy + 24), fill=pal["win"] + (200,))
                elif r < 0.75:
                    d.rectangle((wx, wy, wx + 16, wy + 24), fill=(0, 0, 0, 90))
        x += w + rng.uniform(260, 620)
    # sutiny v popředí
    for _ in range(26):
        cx, cy = rng.uniform(0, WW), HZ + rng.uniform(60, 420)
        s = rng.uniform(20, 70) * (1 + (cy - HZ) / 300)
        pts = [(cx + s * math.cos(a) * rng.uniform(0.6, 1.2), cy + s * 0.45 * math.sin(a) * rng.uniform(0.6, 1.2))
               for a in np.linspace(math.pi, 2 * math.pi, 7)]
        d.polygon(pts + [(cx + s, cy), (cx - s, cy)], fill=(8, 7, 8, 230))
    return img


def world(kind):
    return static("world_" + kind, lambda: build_world(kind))


# --------------------------------------------------------------------------- viewmodel (zbraň v ruce)
VM_M = np.array([1180.0, 712.0])
VM_R = np.array([1840.0, 1150.0])


def _viewmodel():
    SS = 2
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    axis = VM_R - VM_M
    up = np.array([0.548, -0.837])
    down = -up

    def P(s, off):
        p = VM_M + axis * s + up * off
        return (p[0] * SS, p[1] * SS)

    def quad(sa, sb, wa, wb, oa=0.0, ob=0.0, fill=(46, 48, 53), rim=True):
        pts = [P(sa, oa + wa / 2), P(sb, ob + wb / 2), P(sb, ob - wb / 2), P(sa, oa - wa / 2)]
        d.polygon(pts, fill=fill, outline=(84, 86, 92), width=3)
        if rim:
            d.line([P(sa, oa + wa / 2), P(sb, ob + wb / 2)], fill=(170, 108, 60), width=5)

    quad(0.0, 0.05, 22, 24, fill=(60, 62, 68))
    quad(0.04, 0.14, 12, 16, fill=(26, 27, 30))
    quad(0.14, 0.52, 34, 100, 0, 6)
    for k in range(6):
        s = 0.18 + k * 0.055
        quad(s, s + 0.03, 8 + s * 30, 9 + s * 30, -2, -2, fill=(24, 25, 28), rim=False)
    quad(0.14, 0.52, 7, 16, 21, 62, fill=(30, 31, 34))
    quad(0.155, 0.17, 30, 30, 34, 38, fill=(60, 62, 68))
    quad(0.52, 1.0, 100, 200, 6, 18, fill=(52, 54, 60))
    quad(0.53, 0.66, 30, 40, 74, 92, fill=(36, 37, 40))
    quad(0.60, 0.66, 14, 16, 100, 116, fill=(70, 72, 78))
    # zásobník
    a = VM_M + axis * 0.60 + down * 40
    b = VM_M + axis * 0.70 + down * 50
    pts = [a, b, b + down * 190 + axis * 0.02, a + down * 175 + axis * 0.04]
    d.polygon([(p[0] * SS, p[1] * SS) for p in pts], fill=(58, 60, 66), outline=(84, 86, 92), width=3)
    # rukavice na předpažbí
    c = VM_M + axis * 0.31
    g = [c + up * 10 - axis * 0.06, c + up * 24 + axis * 0.06, c + down * 34 + axis * 0.10,
         c + down * 92 + axis * 0.07, c + down * 98 - axis * 0.08, c + down * 30 - axis * 0.09]
    d.polygon([(p[0] * SS, p[1] * SS) for p in g], fill=(30, 28, 27), outline=(66, 60, 56), width=3)
    for k in range(3):
        p0 = c + up * (18 - k * 10) - axis * (0.03 - k * 0.02)
        d.line([(p0[0] * SS, p0[1] * SS), ((p0 + down * 30)[0] * SS, (p0 + down * 30)[1] * SS)], fill=(56, 50, 46), width=4)
    img = img.resize((W, H), Image.LANCZOS)
    bbox = img.getbbox()
    return img.crop(bbox), bbox[:2]


def _flash_sprite(seed):
    rng = np.random.default_rng(seed)
    S = 360
    glow = radial_mask(S, S, S / 2, S / 2, S / 2, 1.8, 200)
    star = Image.new("L", (S * 2, S * 2), 0)
    d = ImageDraw.Draw(star)
    pts = []
    n = 7
    rot = rng.uniform(0, 6.28)
    for k in range(n * 2):
        a = rot + k * math.pi / n
        r = (rng.uniform(0.55, 0.95) if k % 2 == 0 else 0.12) * S
        pts.append((S + r * math.cos(a), S + r * 0.75 * math.sin(a)))
    d.polygon(pts, fill=255)
    star = star.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(2))
    return ImageChops.lighter(glow, star)


# --------------------------------------------------------------------------- gameplay data
VICTIMS = ["VLK_12", "STÍN", "KOBRA_7", "RAVEN", "BÝK_09", "DUCH", "SKORPION", "LIS_33"]


def burst(hit, n=3, gap=0.09):
    return [hit - gap * (n - 1 - k) for k in range(n)]


SHOT_A = dict(
    enemies=[dict(x=1500, h=190, hit=1.0, sway=1), dict(x=2100, h=150, hit=2.0, sway=-1), dict(x=1760, h=230, hit=3.1, sway=1)],
    holds=[(0.75, 1.12, 0), (1.72, 2.12, 1), (2.82, 3.25, 2)],
    start_pan=300,
)
SHOT_A["shots"] = sorted(sum((burst(e["hit"]) for e in SHOT_A["enemies"]), []))

SPRAY = [2.25 + 0.0875 * k for k in range(13)]
SHOT_C = dict(
    barrel=dict(x=1700, h=60),
    blown=[dict(x=1610, h=170, dir=-1), dict(x=1790, h=160, dir=1), dict(x=1885, h=150, dir=1)],
    enemies=[dict(x=2350, h=200, hit=2.6, sway=1), dict(x=2620, h=150, hit=3.3, sway=-1)],
    holds=[(0.6, 1.05, "barrel"), (2.35, 2.62, 0), (3.02, 3.4, 1)],
    start_pan=520,
    explode=1.0,
)
SHOT_C["shots"] = burst(1.0) + SPRAY
SNIPER_SHOT = 2.4
THERMAL = (255, 232, 196)  # tepelná stopa v dalekohledu


def gameplay_events():
    ev = [(T_GAME + s, "ar") for s in SHOT_A["shots"]]
    ev.append((14.0 + SNIPER_SHOT, "sniper"))
    ev += [(18.0 + s, "ar") for s in SHOT_C["shots"]]
    ev.append((18.0 + SHOT_C["explode"] + 0.02, "explosion"))
    return ev


def enemy_x(e, t):
    tt = min(t, e["hit"]) if "hit" in e else t
    return e["x"] + 22 * math.sin(tt * 1.6 + e["x"]) * e.get("sway", 1)


def pan_at(t, shot, targets):
    holds = shot["holds"]
    vals = [(a, b, targets[idx]) for a, b, idx in holds]
    if t < vals[0][0]:
        return lerp(shot["start_pan"], vals[0][2](vals[0][0]), ease_io(t / vals[0][0]))
    for k, (a, b, fn) in enumerate(vals):
        if a <= t <= b:
            return fn(t)
        if k + 1 < len(vals) and b < t < vals[k + 1][0]:
            na, _, nfn = vals[k + 1]
            return lerp(fn(b), nfn(na), ease_io((t - b) / (na - b)))
    a, b, fn = vals[-1]
    return fn(b) + 40 * ease_io(prog(t, b, b + 1.2))


def foot_y(h):
    return HZ + 0.32 * h + 20


# --------------------------------------------------------------------------- HUD
def draw_hud(frame, t_local, ammo, mag, kills, weapon_name, minimap_dots=(), show_vm_hud=True):
    d = ImageDraw.Draw(frame, "RGBA")
    # minimapa
    cx, cy, r = 150, 150, 110
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(8, 10, 12, 170), outline=(200, 200, 200, 90), width=2)
    for k in range(-2, 3):
        d.line([(cx + k * 40, cy - math.sqrt(max(r * r - (k * 40) ** 2, 0))),
                (cx + k * 40, cy + math.sqrt(max(r * r - (k * 40) ** 2, 0)))], fill=(255, 255, 255, 18))
        d.line([(cx - math.sqrt(max(r * r - (k * 40) ** 2, 0)), cy + k * 40),
                (cx + math.sqrt(max(r * r - (k * 40) ** 2, 0)), cy + k * 40)], fill=(255, 255, 255, 18))
    a = t_local * 2.4
    d.line([(cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a))], fill=GOLD + (90,), width=2)
    for dx, dy, alive in minimap_dots:
        if dx * dx + dy * dy < (r - 8) ** 2:
            d.ellipse((cx + dx - 5, cy + dy - 5, cx + dx + 5, cy + dy + 5), fill=(RED if alive else GRAY) + (230,))
    d.polygon([(cx, cy - 12), (cx + 8, cy + 8), (cx, cy + 3), (cx - 8, cy + 8)], fill=GOLD)
    # skóre
    d.rectangle((W / 2 - 190, 24, W / 2 + 190, 84), fill=(8, 10, 12, 160))
    d.rectangle((W / 2 - 190, 24, W / 2 - 184, 84), fill=GOLD)
    d.rectangle((W / 2 + 184, 24, W / 2 + 190, 84), fill=RED)
    put_text(frame, "ALFA", (W / 2 - 170, 54), 22, GOLD, "Bold", 3, "lm")
    put_text(frame, str(9 + kills), (W / 2 - 62, 54), 36, WHITE, "Black", 0, "mm")
    put_text(frame, "02:%02d" % (41 - int(t_local)), (W / 2, 54), 22, GRAY, "SemiBold", 1, "mm")
    put_text(frame, "7", (W / 2 + 62, 54), 36, WHITE, "Black", 0, "mm")
    put_text(frame, "BRAVO", (W / 2 + 170, 54), 22, RED, "Bold", 3, "rm")
    # zdraví
    d.rectangle((70, 1000, 390, 1012), fill=(255, 255, 255, 40))
    d.rectangle((70, 1000, 390, 1012), fill=(235, 235, 230, 220))
    put_text(frame, "ZDRAVÍ", (70, 976), 20, GRAY, "Bold", 4, "lb")
    put_text(frame, "100", (390, 976), 28, WHITE, "Black", 0, "rb")
    # munice
    if show_vm_hud:
        put_text(frame, weapon_name, (1860, 936), 20, GRAY, "Bold", 4, "rb")
        put_text(frame, "/ %d" % (mag * 4), (1860, 1010), 28, GRAY, "Bold", 0, "rb")
        put_text(frame, str(max(ammo, 0)), (1752, 1010), 74, WHITE if ammo > 5 else RED, "Black", 0, "rb")


def draw_killfeed(frame, entries, t_local, weapon):
    """entries: list of (t_hit, victim, headshot)"""
    y = 120
    for t_hit, victim, hs in entries:
        if t_local < t_hit:
            continue
        k = ease_out(prog(t_local, t_hit, t_hit + 0.2))
        x = 1880 + (1 - k) * 120
        parts = [("ČERNÝ_OREL", GOLD), (" [%s]%s " % (weapon, " ⦿" if hs else ""), GRAY), (victim, RED)]
        widths = [text_w(p, 24, "Bold") for p, _ in parts]
        tot = sum(widths)
        d = ImageDraw.Draw(frame, "RGBA")
        d.rectangle((x - tot - 20, y - 22, x + 14, y + 22), fill=(6, 8, 10, int(160 * k)))
        xx = x - tot
        for (p, c), wdt in zip(parts, widths):
            put_text(frame, p, (xx, y), 24, c, "Bold", 0, "lm", alpha=k)
            xx += wdt
        y += 52


def draw_crosshair(frame, spread=0.0, color=(240, 240, 240)):
    d = ImageDraw.Draw(frame, "RGBA")
    cx, cy = W / 2, H / 2
    g = 10 + spread * 26
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        d.line([(cx + dx * g, cy + dy * g), (cx + dx * (g + 18), cy + dy * (g + 18))], fill=color + (230,), width=3)
    d.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color + (230,))


def draw_hitmarker(frame, k, color=(255, 255, 255)):
    if k <= 0:
        return
    d = ImageDraw.Draw(frame, "RGBA")
    cx, cy = W / 2, H / 2
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        d.line([(cx + sx * 12, cy + sy * 12), (cx + sx * 26, cy + sy * 26)], fill=color + (int(255 * k),), width=4)


def caption(frame, txt, t, a, b, y=820):
    k = window(t, a, b, 0.25, 0.3)
    if k <= 0:
        return
    x = 110 - (1 - ease_out(prog(t, a, a + 0.45))) * 80
    size = 60
    tw = text_w(txt, size, "Black", 6)
    d = ImageDraw.Draw(frame, "RGBA")
    d.rectangle((x - 24, y - 52, x + tw + 40, y + 52), fill=(0, 0, 0, int(150 * k)))
    d.rectangle((x - 24, y - 52, x - 14, y + 52), fill=GOLD + (int(255 * k),))
    put_text(frame, txt, (x + 8, y), size, WHITE, "Black", 6, "lm", alpha=k)


def last_shot_dt(t, shots):
    past = [t - s for s in shots if s <= t + 1e-6]
    return min(past) if past else 99.0


def draw_muzzle(frame, t, shots, offset):
    dt = last_shot_dt(t, shots)
    if dt < 0.055:
        idx = int(round(t * 100)) % 3
        spr = static("flash%d" % idx, lambda: _flash_sprite(idx))
        mx, my = VM_M + offset
        s = spr.size[0]
        paste_color(frame, (255, 200, 120), (mx - s / 2, my - s / 2), spr, 1.0)
        paste_color(frame, (255, 250, 230), (mx - s / 4, my - s / 4), spr.resize((s // 2, s // 2)), 1.0)
        d = ImageDraw.Draw(frame, "RGBA")
        d.line([(mx, my), (W / 2 + 4, H / 2 + 6)], fill=(255, 220, 140, 150), width=3)
        return flash(frame, (255, 150, 60), 0.07)
    return frame


def paste_viewmodel(frame, t, shots, extra=(0, 0)):
    vm, (vx, vy) = static("viewmodel", _viewmodel)
    dt = last_shot_dt(t, shots)
    rec = math.exp(-dt / 0.07) if dt < 1 else 0.0
    bob = (8 * math.sin(t * 2.1), 6 * math.sin(t * 4.2))
    off = np.array([rec * 22 + bob[0] + extra[0], rec * 34 + bob[1] + extra[1]])
    frame.paste(vm, (int(vx + off[0]), int(vy + off[1])), vm)
    return off


# --------------------------------------------------------------------------- scény
def scene_intro(t):
    f = Image.new("RGB", (W, H), (4, 4, 6))
    draw_fog(f, t, (50, 46, 46), 0.35 + 0.1 * math.sin(t))
    draw_embers(f, t, 60, alpha=min(1, t / 1.5))
    lines = CFG["intro_lines"]
    for i, line in enumerate(lines):
        a = 0.3 + i * 1.9
        b = a + 1.75
        k = window(t, a, b, 0.45, 0.4)
        if k > 0:
            last = i == len(lines) - 1
            col = GOLD if last else WHITE
            track = 6 + 10 * prog(t, a, b)
            put_text(f, line.upper(), (W / 2, H / 2), 60 if last else 52, col, "SemiBold" if not last else "Bold",
                     track, "mm", alpha=k, glow=14 if last else 0, glow_strength=0.6)
    letterbox(f)
    return f


def scene_logo(t):
    lt = t - T_LOGO
    f = Image.new("RGB", (W, H), (4, 4, 6))
    draw_fog(f, t, (70, 50, 30), 0.3)
    d = ImageDraw.Draw(f, "RGBA")
    # záře za logem
    glow = static("logo_glow", lambda: radial_mask(W, H, W / 2, 430, 700, 2.0, 120))
    paste_color(f, (120, 60, 20), (0, 0), glow, ease_out(prog(lt, 0, 0.6)))
    zoom = 1 + 0.05 * lt + 0.5 * ease_in(prog(lt, 3.55, 4.0))
    size = 440 * zoom * (1 + 0.25 * (1 - ease_out(prog(lt, 0, 0.35))))
    paste_emblem(f, W / 2, 420 - 30 * ease_in(prog(lt, 3.55, 4.0)), size, alpha=ease_out(prog(lt, 0, 0.12)))
    # titulek s odhalením zleva doprava
    title = CFG["title"]
    size_t = 150
    pad = 30
    tr = int(lerp(46, 26, ease_out(prog(lt, 0.3, 2.5))))
    m = text_mask(title, size_t, "Black", tr, pad, 0)
    gm = text_mask(title, size_t, "Black", tr, pad, 18)
    reveal = ease_io(prog(lt, 0.35, 1.4))
    mw = m.size[0]
    ramp = np.clip((reveal * (mw + 200) - np.arange(mw)) / 200, 0, 1)[None, :]
    rm = Image.fromarray((np.asarray(m) * ramp).astype(np.uint8), "L")
    rg = Image.fromarray((np.asarray(gm) * ramp).astype(np.uint8), "L")
    f_ = font(size_t, "Black")
    ct, cb = f_.getbbox("H")[1], f_.getbbox("H")[3]
    x0, y0 = W / 2 - mw / 2, 790 - pad - (ct + cb) / 2
    fade = 1 - prog(lt, 3.6, 4.0)
    paste_color(f, GOLD, (x0, y0), rg, 0.9 * fade)
    paste_color(f, WHITE, (x0, y0), rm, fade)
    put_text(f, "OFICIÁLNÍ TRAILER", (W / 2, 880), 26, GRAY, "SemiBold", 18, "mm", alpha=window(lt, 1.6, 3.7, 0.5, 0.3))
    # rázová vlna při úderu
    p = prog(lt, 0, 0.7)
    if 0 < p < 1:
        r = 60 + 900 * ease_out(p)
        d.ellipse((W / 2 - r, 420 - r, W / 2 + r, 420 + r), outline=GOLD_LIGHT + (int(200 * (1 - p)),), width=6)
    draw_embers(f, t, 50, seed=9, speed=1.6)
    letterbox(f)
    f = flash(f, (255, 236, 200), 0.9 * math.exp(-lt / 0.12))
    return f


def soldier_stride(t, ph=0.0):
    return 0.6 * math.sin(t * 7 + ph)


def blood(frame, x, y, dt, seed):
    if not (0 <= dt < 0.45):
        return
    rng = np.random.default_rng(seed)
    d = ImageDraw.Draw(frame, "RGBA")
    for _ in range(14):
        vx, vy = rng.uniform(-220, 220), rng.uniform(-260, 60)
        px, py = x + vx * dt, y + vy * dt + 700 * dt * dt
        r = rng.uniform(2, 5) * (1 - dt / 0.45)
        d.ellipse((px - r, py - r, px + r, py + r), fill=(150, 18, 18, int(230 * (1 - dt / 0.45))))


def shot_assault(t):
    lt = t - T_GAME
    sh = SHOT_A
    enemies = sh["enemies"]
    targets = [(lambda e: (lambda tt: enemy_x(e, tt) - W / 2))(e) for e in enemies]
    pan = pan_at(lt, sh, targets)
    sway_y = 6 * math.sin(lt * 1.3)
    dts = last_shot_dt(lt, sh["shots"])
    shake = (math.exp(-dts / 0.05) * 6) if dts < 1 else 0
    rng = np.random.default_rng(int(t * 1000))
    px, py = pan + rng.uniform(-shake, shake), 60 + sway_y + rng.uniform(-shake, shake)
    f = world("dusk").crop((int(px), int(py), int(px) + W, int(py) + H))
    dots = []
    entries = []
    for i, e in enumerate(enemies):
        ex = enemy_x(e, lt) - px
        fy = foot_y(e["h"]) - py
        dth = lt - e["hit"]
        if dth < 0:
            draw_soldier(f, ex, fy, e["h"], soldier_stride(lt, i) * 0.4, 0, -1 if ex > W / 2 else 1)
            dots.append(((ex - W / 2) / 12, -(e["h"] - 100) * 0.5, True))
        else:
            ang = -1.3 * ease_in(prog(dth, 0, 0.45)) * (1 if i % 2 else -1)
            draw_soldier(f, ex, fy, e["h"], 0, ang, 1, alpha=1 - prog(dth, 2.2, 2.8))
            blood(f, ex, fy - e["h"] * 0.6, dth, i)
            entries.append((e["hit"], VICTIMS[i], i == 2))
            dots.append(((ex - W / 2) / 12, -(e["h"] - 100) * 0.5, False))
    off = paste_viewmodel(f, lt, sh["shots"])
    f = draw_muzzle(f, lt, sh["shots"], off)
    fired = sum(1 for s in sh["shots"] if s <= lt)
    draw_crosshair(f, math.exp(-dts / 0.12) if dts < 1 else 0)
    hm = max([1 - prog(lt - e["hit"], 0, 0.22) for e in enemies if lt >= e["hit"]] + [0])
    draw_hitmarker(f, hm)
    draw_killfeed(f, entries, lt, CFG["weapons"][0]["name"])
    draw_hud(f, lt, 30 - fired, 30, len(entries), CFG["weapons"][0]["name"], dots)
    caption(f, CFG["gameplay_captions"][0], lt, 0.4, 3.7)
    return f


def shot_scope(t):
    lt = t - 14.0
    Z = 2.3
    ex = 1640 + 30 * lt
    h = 80
    fy = foot_y(h) + 40
    head = (ex + 4, fy - h * 0.905)
    calm = prog(lt, 1.3, 2.25)
    sway = (1 - calm) * 1.0 + 0.08
    off_x = 260 * (1 - ease_io(prog(lt, 0, 1.4))) + 26 * sway * math.sin(lt * 1.7)
    off_y = -60 * (1 - ease_io(prog(lt, 0, 1.4))) + 18 * sway * math.sin(lt * 2.3 + 1)
    rec = 0.0
    if lt >= SNIPER_SHOT:
        rec = math.exp(-(lt - SNIPER_SHOT) / 0.25)
        off_y -= 70 * rec
        off_x = 0
    cx, cy = head[0] + off_x, head[1] + off_y
    if lt >= SNIPER_SHOT:
        cx, cy = head_fixed = (1640 + 30 * SNIPER_SHOT + 4, fy - h * 0.905 - 70 * rec)
    hw, hh = W / (2 * Z), H / (2 * Z)
    f = world("night").crop((int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))).resize((W, H), Image.BILINEAR)

    def S(p):
        return ((p[0] - cx) * Z + W / 2, (p[1] - cy) * Z + H / 2)

    if lt < SNIPER_SHOT:
        sx, sy = S((ex, fy))
        draw_soldier(f, sx, sy, h * Z, soldier_stride(lt), 0, 1, rim=(255, 120, 40), body=THERMAL)
    else:
        dth = lt - SNIPER_SHOT
        exs = 1640 + 30 * SNIPER_SHOT
        sx, sy = S((exs, fy))
        draw_soldier(f, sx, sy, h * Z, 0, 1.4 * ease_in(prog(dth, 0, 0.5)), 1,
                     rim=(255, 120, 40), body=THERMAL, alpha=1 - prog(dth, 1.4, 1.6))
        hx, hy = S((exs + 4, fy - h * 0.905))
        blood(f, hx, hy, dth * 0.8, 3)
    # dalekohled
    mask = static("scope", lambda: ImageChops.invert(radial_mask(W, H, W / 2, H / 2, 470, 0.08, 255)))
    f.paste((0, 0, 0), (0, 0, W, H), mask)
    d = ImageDraw.Draw(f, "RGBA")
    d.line([(W / 2 - 470, H / 2), (W / 2 + 470, H / 2)], fill=(0, 0, 0, 230), width=2)
    d.line([(W / 2, H / 2 - 470), (W / 2, H / 2 + 470)], fill=(0, 0, 0, 230), width=2)
    for k in range(1, 9):
        for s in (-1, 1):
            d.line([(W / 2 + s * k * 40, H / 2 - 7), (W / 2 + s * k * 40, H / 2 + 7)], fill=(0, 0, 0, 220), width=2)
            d.line([(W / 2 - 7, H / 2 + s * k * 40), (W / 2 + 7, H / 2 + s * k * 40)], fill=(0, 0, 0, 220), width=2)
    d.line([(W / 2 - 470, H / 2), (W / 2 - 140, H / 2)], fill=(0, 0, 0, 255), width=7)
    d.line([(W / 2 + 140, H / 2), (W / 2 + 470, H / 2)], fill=(0, 0, 0, 255), width=7)
    d.line([(W / 2, H / 2 + 140), (W / 2, H / 2 + 470)], fill=(0, 0, 0, 255), width=7)
    d.ellipse((W / 2 - 3, H / 2 - 3, W / 2 + 3, H / 2 + 3), fill=(230, 40, 40, 255))
    # texty
    if 1.2 < lt < SNIPER_SHOT:
        put_text(f, "ZADRŽ DECH", (W / 2, H / 2 - 230), 22, (210, 230, 240), "Bold", 8, "mm",
                 alpha=0.6 + 0.4 * math.sin(lt * 10))
    if lt >= SNIPER_SHOT:
        k = window(lt, SNIPER_SHOT + 0.15, 4.2, 0.2, 0.3)
        sc = 1 + 0.4 * (1 - ease_out(prog(lt, SNIPER_SHOT + 0.15, SNIPER_SHOT + 0.4)))
        put_text(f, "ZÁSAH DO HLAVY", (1690, 500), int(44 * sc), RED, "Black", 4, "mm", alpha=k, glow=10)
        put_text(f, "+150", (1690, 570), 40, GOLD, "Black", 2, "mm", alpha=k)
        draw_hitmarker(f, 1 - prog(lt - SNIPER_SHOT, 0, 0.3), (255, 60, 60))
    entries = [(SNIPER_SHOT, VICTIMS[3], True)]
    draw_killfeed(f, entries, lt, CFG["weapons"][1]["name"])
    draw_hud(f, lt + 4, 5 if lt < SNIPER_SHOT else 4, 5, 3 + (lt >= SNIPER_SHOT), CFG["weapons"][1]["name"], ())
    caption(f, CFG["gameplay_captions"][1], lt, 0.3, 3.7)
    if lt >= SNIPER_SHOT:
        f = flash(f, (255, 245, 230), 0.55 * math.exp(-(lt - SNIPER_SHOT) / 0.06))
    return f


def shot_chaos(t):
    lt = t - 18.0
    sh = SHOT_C
    bx = sh["barrel"]["x"]
    targets = {"barrel": lambda tt: bx - W / 2}
    for i, e in enumerate(sh["enemies"]):
        targets[i] = (lambda e: (lambda tt: enemy_x(e, tt) - W / 2))(e)
    pan = pan_at(lt, sh, targets)
    te = lt - sh["explode"]
    shake = 0.0
    if te >= 0:
        shake = 46 * math.exp(-te / 0.3)
    dts = last_shot_dt(lt, sh["shots"])
    if dts < 1:
        shake += 5 * math.exp(-dts / 0.05)
    rng = np.random.default_rng(int(t * 1000) + 5)
    px, py = pan + rng.uniform(-shake, shake), 60 + 5 * math.sin(lt * 1.4) + rng.uniform(-shake, shake)
    f = world("fire").crop((int(px), int(py), int(px) + W, int(py) + H))
    d = ImageDraw.Draw(f, "RGBA")
    bfy = foot_y(150) - py
    bsx = bx - px
    # sud
    if te < 0:
        d.rectangle((bsx - 26, bfy - 70, bsx + 26, bfy), fill=(120, 30, 24), outline=(40, 10, 8), width=3)
        for yy in (bfy - 50, bfy - 22):
            d.line([(bsx - 26, yy), (bsx + 26, yy)], fill=(60, 14, 10), width=4)
        d.polygon([(bsx - 10, bfy - 46), (bsx + 10, bfy - 46), (bsx, bfy - 28)], fill=(240, 200, 40))
    # kouřový sloup
    if te > 0:
        rs = np.random.default_rng(21)
        for k in range(26):
            age = te - k * 0.05
            if age <= 0:
                continue
            ox = rs.uniform(-60, 60) + age * rs.uniform(-30, 30)
            oy = -age * rs.uniform(90, 170)
            r = 40 + age * 70
            a = int(150 * clamp(1 - age / 3.5))
            c = int(rs.uniform(30, 55))
            d.ellipse((bsx + ox - r, bfy - 40 + oy - r, bsx + ox + r, bfy - 40 + oy + r), fill=(c, c - 4, c - 6, a))
    # odhození vojáci
    for i, e in enumerate(sh["blown"]):
        ex = e["x"] - px
        fy = foot_y(e["h"]) - py
        if te < 0:
            draw_soldier(f, ex, fy, e["h"], soldier_stride(lt, i) * 0.3, 0, e["dir"] * -1)
        else:
            vx = e["dir"] * (300 + 90 * i)
            vy = -620 - 60 * i
            g = 1700
            tt = min(te, 1.2)
            x = ex + vx * tt
            y = fy + vy * tt + 0.5 * g * tt * tt
            land = fy + 30
            y = min(y, land)
            spin = e["dir"] * min(te, 0.9) * 5
            draw_soldier(f, x, y, e["h"], 0, spin, 1, alpha=1 - prog(te, 1.6, 2.2), rim=(255, 120, 40))
    # ohnivá koule
    if 0 <= te < 1.6:
        rs = np.random.default_rng(33)
        p = te / 1.6
        # vrstvené obláčky (kouřový okraj, žhavé jádro), rozmazané do měkkého tvaru
        LS = 1000
        layer = Image.new("RGBA", (LS, LS), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        lcx, lcy = LS / 2, LS / 2 + 120
        heat = clamp(1 - te / 0.8)
        fade_a = clamp(1.4 - p * 1.1)
        puffs = []
        for k in range(18):
            ang = rs.uniform(0, 6.28)
            dist = rs.uniform(0, 170) * ease_out(prog(te, 0, 0.45))
            r = rs.uniform(60, 150) * ease_out(prog(te, 0, 0.22)) * (1 + 0.7 * p)
            rise = te * rs.uniform(60, 160)
            puffs.append((lcx + math.cos(ang) * dist, lcy + math.sin(ang) * dist * 0.55 - rise, r))
        shells = [(1.0, mixc((40, 30, 28), (120, 50, 20), heat), 0.9),
                  (0.78, mixc((70, 46, 36), (230, 96, 24), heat), 0.95),
                  (0.55, mixc((90, 60, 44), (255, 170, 60), heat), 1.0),
                  (0.32, mixc((110, 80, 60), (255, 244, 210), heat), 1.0)]
        for scale, col, aa in shells:
            for x_, y_, r in puffs:
                rr = r * scale
                oy = (1 - scale) * r * 0.35
                ld.ellipse((x_ - rr, y_ - rr - oy, x_ + rr, y_ + rr - oy), fill=col + (int(255 * aa * fade_a),))
        layer = layer.filter(ImageFilter.GaussianBlur(7 + 10 * p))
        f.paste(layer.convert("RGB"), (int(bsx - lcx), int(bfy - 80 - lcy)), layer.getchannel("A"))
        if te < 0.9:
            gl = static("boom_glow", lambda: radial_mask(1400, 1000, 700, 500, 700, 2.2, 255))
            paste_color(f, (255, 140, 50), (bsx - 700, bfy - 580), gl, 0.45 * clamp(1 - te / 0.9))
        d = ImageDraw.Draw(f, "RGBA")
        for k in range(40):
            ang = rs.uniform(math.pi * 1.05, math.pi * 1.95)
            sp = rs.uniform(500, 1300)
            x = bsx + math.cos(ang) * sp * te
            y = bfy - 60 + math.sin(ang) * sp * te + 1200 * te * te
            d.line([(x, y), (x - math.cos(ang) * 18, y - math.sin(ang) * 18 - 8)], fill=(255, 200, 120, int(255 * clamp(1 - te))), width=3)
    entries = [(sh["explode"] + 0.05, VICTIMS[4 + k], False) for k in range(3)]
    dots = []
    for i, e in enumerate(sh["enemies"]):
        ex = enemy_x(e, lt) - px
        fy = foot_y(e["h"]) - py
        dth = lt - e["hit"]
        if dth < 0:
            draw_soldier(f, ex, fy, e["h"], soldier_stride(lt, i + 3), 0, -1 if ex > W / 2 else 1, rim=(255, 120, 40))
            dots.append(((ex - W / 2) / 12, -40, True))
        else:
            draw_soldier(f, ex, fy, e["h"], 0, -1.3 * ease_in(prog(dth, 0, 0.45)), 1, rim=(255, 120, 40))
            blood(f, ex, fy - e["h"] * 0.6, dth, i + 7)
            entries.append((e["hit"], VICTIMS[7 - i], False))
    off = paste_viewmodel(f, lt, sh["shots"], extra=(rng.uniform(-shake, shake) * 0.3, rng.uniform(-shake, shake) * 0.3))
    f = draw_muzzle(f, lt, sh["shots"], off)
    if te >= 0:
        f = flash(f, (255, 200, 140), 0.75 * math.exp(-te / 0.12))
        f = Image.blend(f, Image.new("RGB", (W, H), (255, 110, 30)), 0.25 * math.exp(-te / 0.6))
    draw_crosshair(f, math.exp(-dts / 0.12) if dts < 1 else 0)
    hm = max([1 - prog(lt - e["hit"], 0, 0.22) for e in sh["enemies"] if lt >= e["hit"]] + [0])
    draw_hitmarker(f, hm)
    draw_killfeed(f, entries, lt, CFG["weapons"][0]["name"])
    fired = sum(1 for s in sh["shots"] if s <= lt)
    draw_hud(f, lt + 9, 30 - fired, 30, 4 + len([e for e in entries if e[0] <= lt]), CFG["weapons"][0]["name"], dots)
    if te >= 0.25:
        k = window(te, 0.25, 1.8, 0.15, 0.4)
        sc = 1 + 0.5 * (1 - ease_out(prog(te, 0.25, 0.45)))
        put_text(f, "TROJITÉ ZABITÍ", (W / 2, 330), int(54 * sc), GOLD, "Black", 6, "mm", alpha=k, glow=12)
        put_text(f, "+300", (W / 2, 400), 38, WHITE, "Black", 2, "mm", alpha=k)
    caption(f, CFG["gameplay_captions"][2], lt, 1.9, 3.85)
    return f


def scene_gameplay(t):
    if t < 14.0:
        return shot_assault(t)
    if t < 18.0:
        return shot_scope(t)
    return shot_chaos(t)


def stripes():
    m = Image.new("L", (W + 400, H), 0)
    d = ImageDraw.Draw(m)
    for x in range(-H, W + 400, 80):
        d.polygon([(x, H), (x + 30, H), (x + 30 + H, 0), (x + H, 0)], fill=255)
    return m


def section_bg(t, base=((14, 15, 19), (3, 3, 5))):
    f = static("bg_grad", lambda: vgrad(W, H, base[0], base[1])).copy()
    st = static("stripes", stripes)
    ox = int((t * 60) % 80)
    paste_color(f, GOLD, (0, 0), st.crop((ox, 0, ox + W, H)), 0.025)
    return f


def section_title(f, t, a, title, sub):
    lt = t - a
    k = window(lt, 0, 2.0, 0.25, 0.3)
    tr = lerp(60, 30, ease_out(prog(lt, 0, 1.6)))
    put_text(f, title, (W / 2, H / 2 - 20), 150, WHITE, "Black", tr, "mm", alpha=k, glow=20, glow_strength=0.5)
    d = ImageDraw.Draw(f, "RGBA")
    lw = 420 * ease_out(prog(lt, 0.15, 0.7))
    d.rectangle((W / 2 - lw, H / 2 + 80, W / 2 + lw, H / 2 + 84), fill=GOLD + (int(255 * k),))
    put_text(f, sub, (W / 2, H / 2 + 140), 30, GOLD, "Bold", 16, "mm", alpha=k * ease_out(prog(lt, 0.3, 0.8)))


@lru_cache(maxsize=16)
def icon_mask(kind, size):
    SS = 3
    S = size * SS
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    c = S / 2
    lw = int(S * 0.06)
    if kind == "crosshair":
        r = S * 0.34
        d.ellipse((c - r, c - r, c + r, c + r), outline=255, width=lw)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d.line([(c + dx * S * 0.18, c + dy * S * 0.18), (c + dx * S * 0.48, c + dy * S * 0.48)], fill=255, width=lw)
        d.ellipse((c - lw, c - lw, c + lw, c + lw), fill=255)
    elif kind == "flag":
        d.rectangle((S * 0.26, S * 0.12, S * 0.26 + lw, S * 0.9), fill=255)
        pts = [(S * 0.26 + lw, S * 0.14)]
        for k in range(11):
            x = S * 0.26 + lw + k * S * 0.05
            pts.append((x, S * 0.14 + math.sin(k / 10 * 6.28) * S * 0.035))
        for k in range(10, -1, -1):
            x = S * 0.26 + lw + k * S * 0.05
            pts.append((x, S * 0.5 + math.sin(k / 10 * 6.28) * S * 0.035))
        d.polygon(pts, fill=255)
    elif kind == "bomb":
        r = S * 0.3
        cx, cy = c - S * 0.05, c + S * 0.1
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        d.rectangle((cx + r * 0.45, cy - r * 1.05, cx + r * 0.85, cy - r * 0.65), fill=255)
        d.arc((cx + r * 0.6, cy - r * 1.9, cx + r * 1.6, cy - r * 0.8), 180, 300, fill=255, width=lw)
        sx, sy = cx + r * 1.42, cy - r * 1.72
        for a in range(0, 360, 45):
            ln = S * (0.09 if a % 90 == 0 else 0.05)
            d.line([(sx, sy), (sx + math.cos(math.radians(a)) * ln, sy + math.sin(math.radians(a)) * ln)], fill=255, width=lw // 2)
        d.ellipse((cx - r * 0.55, cy - r * 0.55, cx - r * 0.2, cy - r * 0.2), fill=0)
    else:  # skull
        d.ellipse((S * 0.2, S * 0.1, S * 0.8, S * 0.66), fill=255)
        d.rounded_rectangle((S * 0.32, S * 0.5, S * 0.68, S * 0.84), radius=S * 0.05, fill=255)
        d.ellipse((S * 0.29, S * 0.34, S * 0.45, S * 0.5), fill=0)
        d.ellipse((S * 0.55, S * 0.34, S * 0.71, S * 0.5), fill=0)
        d.polygon([(S * 0.5, S * 0.52), (S * 0.45, S * 0.6), (S * 0.55, S * 0.6)], fill=0)
        for x in (0.42, 0.5, 0.58):
            d.line([(S * x, S * 0.68), (S * x, S * 0.84)], fill=0, width=lw // 2)
    return m.resize((size, size), Image.LANCZOS)


@lru_cache(maxsize=16)
def outline_text(txt, size):
    m = text_mask(txt, size, "Black", 0, 10, 0)
    return ImageChops.subtract(m, m.filter(ImageFilter.MinFilter(7)))


def hexagon(d, cx, cy, r, rot, color, width):
    pts = [(cx + r * math.cos(rot + k * math.pi / 3), cy + r * math.sin(rot + k * math.pi / 3)) for k in range(6)]
    d.line(pts + [pts[0]], fill=color, width=width, joint="curve")


def scene_modes(t):
    f = section_bg(t)
    if t < MODE_START:
        section_title(f, t, T_MODES, "HERNÍ MÓDY", "VYBER SI SVŮJ BOJ")
        return f
    modes = CFG["modes"]
    i = min(int((t - MODE_START) // MODE_LEN), len(modes) - 1)
    mode = modes[i]
    lt = t - (MODE_START + i * MODE_LEN)
    enter = ease_out(prog(lt, 0, 0.42))
    leave = ease_in(prog(lt, MODE_LEN - 0.28, MODE_LEN)) if i < len(modes) - 1 else ease_in(prog(lt, MODE_LEN - 0.2, MODE_LEN))
    dx = (1 - enter) * 700 - leave * 900
    k = enter * (1 - leave)
    d = ImageDraw.Draw(f, "RGBA")
    # velké číslo v pozadí
    num = outline_text("%02d" % (i + 1), 560)
    paste_color(f, GOLD, (1180 + dx * 0.5, 120), num, 0.16 * k)
    # ikona
    icx, icy = 470 + dx * 0.8, 540
    hexagon(d, icx, icy, 210, t * 0.4, GOLD + (int(220 * k),), 4)
    hexagon(d, icx, icy, 240, -t * 0.25, GOLD + (int(70 * k),), 2)
    glow = static("icon_glow", lambda: radial_mask(500, 500, 250, 250, 250, 2.0, 140))
    paste_color(f, (140, 80, 20), (icx - 250, icy - 250), glow, k)
    im = icon_mask(mode.get("icon", "crosshair"), 230)
    paste_color(f, WHITE, (icx - 115, icy - 115), im, k)
    # texty
    tx = 790 + dx
    size = fit_size(mode["name"], 1060, 104, "Black", 2)
    put_text(f, "MÓD %02d" % (i + 1), (tx, 400), 26, GOLD, "Bold", 10, "lm", alpha=k)
    put_text(f, mode["name"], (tx, 480), size, WHITE, "Black", 2, "lm", alpha=k)
    put_text(f, mode["desc"], (tx, 575), 36, (196, 198, 204), "Medium", 1, "lm", alpha=k * ease_out(prog(lt, 0.15, 0.6)))
    cw = text_w(mode["players"], 28, "Black", 4) + 48
    d.rectangle((tx, 634, tx + cw, 690), fill=GOLD + (int(255 * k),))
    put_text(f, mode["players"], (tx + 24, 662), 28, (12, 12, 14), "Black", 4, "lm", alpha=k)
    # hlavička a průběh
    put_text(f, "HERNÍ MÓDY", (90, 90), 24, GOLD, "Bold", 10, "lm")
    for j in range(len(modes)):
        x0 = 90 + j * 150
        d.rectangle((x0, 990, x0 + 130, 996), fill=(255, 255, 255, 40))
        if j < i:
            d.rectangle((x0, 990, x0 + 130, 996), fill=GOLD)
        elif j == i:
            d.rectangle((x0, 990, x0 + 130 * prog(lt, 0, MODE_LEN), 996), fill=GOLD)
    put_text(f, "%02d / %02d" % (i + 1, len(modes)), (1830, 993), 24, GRAY, "Bold", 4, "rm")
    return f


def scene_armory(t):
    f = section_bg(t, ((12, 12, 14), (2, 2, 3)))
    if t < WEAPON_START:
        section_title(f, t, T_ARMORY, "ZBROJÍRNA", "ARZENÁL ČERNÉHO ORLA")
        return f
    weapons = CFG["weapons"]
    i = min(int((t - WEAPON_START) // WEAPON_LEN), len(weapons) - 1)
    wpn = weapons[i]
    lt = t - (WEAPON_START + i * WEAPON_LEN)
    enter = ease_out(prog(lt, 0, 0.45))
    leave = ease_in(prog(lt, WEAPON_LEN - 0.3, WEAPON_LEN)) if i < len(weapons) - 1 else 0
    k = enter * (1 - leave)
    spot = static("spot", lambda: radial_mask(W, H, 700, 500, 820, 1.6, 150))
    paste_color(f, (120, 78, 30), (0, 0), spot, 0.9)
    d = ImageDraw.Draw(f, "RGBA")
    # podlaha / stín
    shadow = static("shadow", lambda: radial_mask(1100, 120, 550, 60, 550, 1.4, 255).resize((1100, 120)))
    paste_color(f, (0, 0, 0), (150, 700), shadow, 0.8 * k)
    d.line([(80, 760), (1260, 760)], fill=GOLD + (int(60 * k),), width=2)
    spr = weapon_sprite(wpn.get("kind", "ar"), 1150)
    wx = 125 + (1 - enter) * -500 + leave * 600 + 30 * prog(lt, 0, WEAPON_LEN)
    wy = 520 - spr.size[1] / 2 - 10 * math.sin(lt * 1.2)
    a = spr.getchannel("A")
    if k < 0.996:
        a = a.point(lambda v: int(v * k))
    f.paste(spr, (int(wx), int(wy)), a)
    # světelný odlesk přes zbraň
    sweep = prog(lt, 0.4, 1.3)
    if 0 < sweep < 1:
        sxp = int(wx - 200 + sweep * 1500)
        band = Image.new("L", (160, spr.size[1]), 0)
        ImageDraw.Draw(band).polygon([(60, 0), (120, 0), (100, spr.size[1]), (40, spr.size[1])], fill=255)
        band = band.filter(ImageFilter.GaussianBlur(14))
        clip = ImageChops.multiply(band, spr.getchannel("A").crop((sxp - int(wx), 0, sxp - int(wx) + 160, spr.size[1])))
        paste_color(f, (255, 240, 210), (sxp, int(wy)), clip, 0.5 * k)
    # panel
    px = 1340 + (1 - enter) * 300
    put_text(f, wpn["class"], (px, 290), 28, GOLD, "Bold", 8, "lm", alpha=k)
    nsize = fit_size(wpn["name"], 520, 92, "Black", 2)
    put_text(f, wpn["name"], (px, 370), nsize, WHITE, "Black", 2, "lm", alpha=k)
    d.rectangle((px, 430, px + 500, 433), fill=(255, 255, 255, int(50 * k)))
    for j, (label, val) in enumerate(wpn["stats"].items()):
        y = 490 + j * 82
        put_text(f, label, (px, y), 22, GRAY, "Bold", 5, "lm", alpha=k)
        fill = val * ease_out(prog(lt, 0.3 + 0.1 * j, 1.1 + 0.1 * j))
        put_text(f, str(int(round(fill * 100))), (px + 500, y), 26, WHITE, "Black", 0, "rm", alpha=k)
        d.rectangle((px, y + 26, px + 500, y + 36), fill=(255, 255, 255, int(28 * k)))
        d.rectangle((px, y + 26, px + 500 * fill, y + 36), fill=GOLD + (int(255 * k),))
    # výběr zbraní dole
    put_text(f, "ZBROJÍRNA", (90, 90), 24, GOLD, "Bold", 10, "lm")
    for j, w2 in enumerate(weapons):
        bx = 90 + j * 300
        active = j == i
        d.rectangle((bx, 880, bx + 270, 1010), fill=(255, 255, 255, 18 if not active else 30),
                    outline=GOLD + (255,) if active else (255, 255, 255, 50), width=3 if active else 1)
        mini = weapon_sprite(w2.get("kind", "ar"), 230)
        f.paste(mini, (bx + 20, 900), mini)
        put_text(f, w2["name"], (bx + 16, 990), 20, WHITE if active else GRAY, "Bold", 2, "lm")
    return f


ATT_LAYOUT = {  # kotva ve souřadnicích zbraně, pozice popisku, zarovnání, směr příletu
    "optic": ((575, 112), (640, 245), "r", (0, -1)),
    "muzzle": ((1450, 222), (1470, 330), "l", (1, 0)),
    "laser": ((1040, 326), (1470, 720), "l", (1, 0.3)),
    "grip": ((926, 414), (1080, 860), "l", (0, 1)),
    "mag": ((760, 570), (780, 860), "r", (-0.3, 1)),
}
ATT_STATS = ["PŘESNOST", "KONTROLA", "MÍŘENÍ", "KAPACITA"]
ATT_BASE = [0.52, 0.48, 0.50, 0.40]
ATT_DELTA = {"optic": [0.06, 0, 0.22, 0], "muzzle": [0.04, 0.08, 0, 0], "grip": [0, 0.26, 0.02, 0],
             "mag": [0, 0, -0.05, 0.36], "laser": [0.2, 0, 0.04, 0]}
ATT_W = 1250
ATT_K = ATT_W / BW
ATT_X0, ATT_Y0 = 960 - 850 * ATT_K - 60, 330


def att_screen(p):
    return (ATT_X0 + p[0] * ATT_K, ATT_Y0 + p[1] * ATT_K)


def blueprint_bg():
    f = Image.new("RGB", (W, H), (8, 12, 18))
    d = ImageDraw.Draw(f, "RGBA")
    for x in range(0, W, 40):
        d.line([(x, 0), (x, H)], fill=GOLD + (18 if x % 200 else 40,), width=1)
    for y in range(0, H, 40):
        d.line([(0, y), (W, y)], fill=GOLD + (18 if y % 200 else 40,), width=1)
    return f


def scene_attachments(t):
    atts = [a for a in CFG["attachments"] if a["slot"] in ATT_LAYOUT]
    if t < ATT_START:
        f = section_bg(t)
        section_title(f, t, T_ATT, "ATTACHMENTY", "PŘIZPŮSOB SI ZBRAŇ")
        return f
    f = static("blueprint", blueprint_bg).copy()
    d = ImageDraw.Draw(f, "RGBA")
    lt = t - ATT_START
    step = int(lt // ATT_STEP)
    end = ATT_START + len(atts) * ATT_STEP
    push = ease_io(prog(t, end, T_MONTAGE))
    attached = tuple(a["slot"] for a in atts[:min(step, len(atts))])
    flying = atts[step] if step < len(atts) else None
    st = lt - step * ATT_STEP
    hide = (flying["slot"],) if flying and flying["slot"] in ("mag", "muzzle") and st > 0.1 else ()
    # zbraň
    base = weapon_sprite("ar", ATT_W, attached, None, hide)
    bp = weapon_sprite("ar", ATT_W, ("optic", "muzzle", "grip", "mag", "laser"), None, (), "blueprint")
    sc = 1 + 0.04 * push
    if sc != 1:
        base = base.resize((int(base.size[0] * sc), int(base.size[1] * sc)))
    ox = ATT_X0 - (base.size[0] - ATT_W) / 2
    oy = ATT_Y0 - (base.size[1] - BH * ATT_K) / 2
    if flying:
        ga = bp.getchannel("A").point(lambda v: int(v * 0.18))
        f.paste(bp, (int(ATT_X0), int(ATT_Y0)), ga)
    f.paste(base, (int(ox), int(oy)), base)
    # laserový paprsek
    if "laser" in attached:
        lx, ly = att_screen((1100, 311))
        for wdt, al in ((9, 40), (3, 200)):
            d.line([(lx, ly), (W, ly + 30)], fill=(255, 40, 40, al), width=wdt)
    # přilétající attachment
    if flying:
        slot = flying["slot"]
        anchor, lpos, align, dirv = ATT_LAYOUT[slot]
        p = ease_out(prog(st, 0, ATT_FLY))
        dist = (1 - p) * 650
        sp = weapon_sprite("ar", ATT_W, (), slot)
        f.paste(sp, (int(ATT_X0 + dirv[0] * dist), int(ATT_Y0 + dirv[1] * dist)), sp)
        if st >= ATT_FLY:
            ax, ay = att_screen(anchor)
            rp = prog(st, ATT_FLY, ATT_FLY + 0.45)
            r = 20 + 140 * ease_out(rp)
            d.ellipse((ax - r, ay - r, ax + r, ay + r), outline=GOLD_LIGHT + (int(220 * (1 - rp)),), width=4)
    # popisky
    shown = list(atts[:min(step, len(atts))]) + ([flying] if flying and st >= ATT_FLY else [])
    for j, a in enumerate(shown):
        anchor, (lx, ly), align, _ = ATT_LAYOUT[a["slot"]]
        t_snap = ATT_START + j * ATT_STEP + ATT_FLY
        k = ease_out(prog(t, t_snap, t_snap + 0.35))
        fade = 1 - prog(t, end, end + 0.35)
        if fade <= 0:
            continue
        ax, ay = att_screen(anchor)
        nw = max(text_w(a["name"], 34, "Black", 3), text_w(a["effect"], 22, "Bold", 3))
        d.ellipse((ax - 7, ay - 7, ax + 7, ay + 7), fill=GOLD)
        ex = lerp(ax, lx, k)
        ey = lerp(ay, ly, k)
        d.line([(ax, ay), (ex, ey)], fill=GOLD + (230,), width=3)
        if k > 0.5:
            kk = (k - 0.5) * 2
            sgn = 1 if align == "l" else -1
            d.line([(lx, ly), (lx + sgn * (nw + 20) * kk, ly)], fill=GOLD + (230,), width=3)
            tx = lx + 10 if align == "l" else lx - 10
            anc = "lb" if align == "l" else "rb"
            put_text(f, a["name"], (tx, ly - 14), 34, WHITE, "Black", 3, anc, alpha=kk * fade)
            put_text(f, a["effect"], (tx, ly + 16), 22, GOLD, "Bold", 3, "lt", alpha=kk * fade) if align == "l" else \
                put_text(f, a["effect"], (tx, ly + 16), 22, GOLD, "Bold", 3, "rt", alpha=kk * fade)
    # statistiky
    vals = list(ATT_BASE)
    gain = [0.0] * 4
    for j, a in enumerate(atts):
        t_snap = ATT_START + j * ATT_STEP + ATT_FLY
        kk = ease_out(prog(t, t_snap, t_snap + 0.5))
        for s in range(4):
            dv = ATT_DELTA[a["slot"]][s] * kk
            vals[s] += dv
            if 0 < t - t_snap < 1.0 and ATT_DELTA[a["slot"]][s] > 0:
                gain[s] = 1 - prog(t - t_snap, 0.4, 1.0)
    d.rectangle((0, 950, W, H), fill=(4, 6, 10, 210))
    d.line([(0, 950), (W, 950)], fill=GOLD + (120,), width=2)
    for s in range(4):
        x0 = 110 + s * 440
        put_text(f, ATT_STATS[s], (x0, 990), 20, GRAY, "Bold", 5, "lm")
        val = clamp(vals[s])
        put_text(f, str(int(round(val * 100))), (x0 + 360, 990), 24, GREEN if gain[s] > 0 else WHITE, "Black", 0, "rm")
        d.rectangle((x0, 1016, x0 + 360, 1026), fill=(255, 255, 255, 30))
        d.rectangle((x0, 1016, x0 + 360 * val, 1026), fill=GOLD)
        if gain[s] > 0:
            d.rectangle((x0, 1016, x0 + 360 * val, 1026), fill=GREEN + (int(200 * gain[s]),))
    put_text(f, "ATTACHMENTY", (90, 90), 24, GOLD, "Bold", 10, "lm")
    put_text(f, "%s  //  %d / %d" % (CFG["weapons"][0]["name"], len(shown), len(atts)), (1830, 90), 24, GRAY, "Bold", 4, "rm")
    if push > 0:
        put_text(f, "TVOJE ZBRAŇ. TVŮJ STYL.", (W / 2, 200), 56, WHITE, "Black", 8, "mm", alpha=ease_out(prog(t, end, end + 0.4)), glow=12, glow_strength=0.5)
    return f


MONTAGE = [(11.2, None), (25.3, None), (16.3, None), (37.6, None), (19.15, None), (56.2, None),
           (30.3, None), (40.6, None), (20.75, None)]


def render_scene(t):
    if t < T_LOGO:
        return scene_intro(t)
    if t < T_GAME:
        return scene_logo(t)
    if t < T_MODES:
        return scene_gameplay(t)
    if t < T_ARMORY:
        return scene_modes(t)
    if t < T_ATT:
        return scene_armory(t)
    if t < T_MONTAGE:
        return scene_attachments(t)
    if t < T_FINAL:
        return scene_montage(t)
    return scene_final(t)


def scene_montage(t):
    lt = t - T_MONTAGE
    cut_len = 0.2
    idx = int(lt // cut_len)
    if idx >= len(MONTAGE):
        f = Image.new("RGB", (W, H), (0, 0, 0))
        return f
    src, _ = MONTAGE[idx]
    ct = lt - idx * cut_len
    f = render_scene(src + ct)
    z = 1.08 + 0.05 * ct / cut_len
    cw, ch = int(W / z), int(H / z)
    f = f.crop(((W - cw) // 2, (H - ch) // 2, (W - cw) // 2 + cw, (H - ch) // 2 + ch)).resize((W, H), Image.BILINEAR)
    f = ImageEnhance.Contrast(f).enhance(1.35)
    f = ImageEnhance.Color(f).enhance(0.75)
    f = Image.blend(f, Image.new("RGB", (W, H), (120, 70, 20)), 0.12)
    f = flash(f, (255, 255, 255), 0.6 * math.exp(-ct / 0.04))
    return f


def scene_final(t):
    lt = t - T_FINAL
    f = Image.new("RGB", (W, H), (3, 3, 4))
    draw_fog(f, t, (80, 52, 26), 0.28, 25)
    glow = static("final_glow", lambda: radial_mask(W, H, W / 2, 380, 760, 2.0, 140))
    paste_color(f, (130, 66, 18), (0, 0), glow, ease_out(prog(lt, 0, 0.5)))
    d = ImageDraw.Draw(f, "RGBA")
    p = prog(lt, 0, 0.9)
    if 0 < p < 1:
        for k, col in enumerate((GOLD_LIGHT, GOLD)):
            r = 80 + 1300 * ease_out(clamp(p - k * 0.08))
            d.ellipse((W / 2 - r, 380 - r, W / 2 + r, 380 + r), outline=col + (int(220 * (1 - p)),), width=8 - k * 3)
    size = 480 * (1 + 0.45 * (1 - ease_out(prog(lt, 0, 0.3)))) * (1 + 0.012 * lt)
    paste_emblem(f, W / 2, 380, size, alpha=ease_out(prog(lt, 0, 0.08)))
    tk = ease_out(prog(lt, 0.35, 1.2))
    put_text(f, CFG["title"], (W / 2, 740), 150, WHITE, "Black", lerp(70, 26, tk), "mm", alpha=tk, glow=22, glow_strength=0.55)
    put_text(f, CFG["tagline"], (W / 2, 845), 38, GOLD, "SemiBold", 16, "mm", alpha=ease_out(prog(lt, 1.6, 2.4)))
    rk = ease_out(prog(lt, 3.0, 3.6))
    if rk > 0:
        rw = text_w(CFG["release"], 34, "Black", 12) + 70
        d.rectangle((W / 2 - rw / 2, 905, W / 2 + rw / 2, 965), outline=GOLD + (int(255 * rk),), width=3)
        put_text(f, CFG["release"], (W / 2, 935), 34, WHITE, "Black", 12, "mm", alpha=rk)
    draw_embers(f, t, 80, seed=13, speed=1.2)
    f = flash(f, (255, 240, 210), 0.95 * math.exp(-lt / 0.15))
    f = flash(f, (0, 0, 0), prog(lt, 6.0, 7.0))
    return f


# --------------------------------------------------------------------------- přechody a postprocess
def postprocess(f, t, i):
    for c in CUTS:
        dt = t - c
        if -0.07 <= dt < 0.12:
            f = glitch(f, 1 - abs(dt) / 0.12, i)
        if 0 <= dt < 0.25:
            f = flash(f, (255, 236, 200), 0.6 * math.exp(-dt / 0.06))
    if T_MODES - 0.3 <= t < T_MODES:
        f = flash(f, (0, 0, 0), prog(t, T_MODES - 0.3, T_MODES) * 0.8)
    v = static("vignette", vignette_mask)
    f = Image.composite(f, Image.new("RGB", (W, H), (0, 0, 0)), v)
    g = static("grain", grain_frames)[i % 6]
    f = ImageChops.add(f, g, 1.0, -14)
    if t < 0.6:
        f = flash(f, (0, 0, 0), 1 - prog(t, 0, 0.6))
    return f


def render_frame(i):
    t = i / FPS
    return postprocess(render_scene(t), t, i).tobytes()


# --------------------------------------------------------------------------- zvuk
SR = 44100


def make_audio(path):
    from scipy.signal import butter, fftconvolve, sosfilt

    n = int(SR * (DURATION + 1))
    mus = np.zeros((n, 2))
    sfx = np.zeros((n, 2))
    rng = np.random.default_rng(42)

    def tt(dur):
        return np.arange(int(dur * SR)) / SR

    def lp(x, fc, order=2):
        return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)

    def hp(x, fc, order=2):
        return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)

    def bp(x, lo, hi, order=2):
        return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)

    def noise(dur):
        return rng.standard_normal(int(dur * SR))

    def saw(freq, dur, phase=0.0):
        x = tt(dur)
        return 2 * ((x * freq + phase) % 1) - 1

    def add(bus, sig, t0, gain=1.0, pan=0.0):
        i0 = int(t0 * SR)
        if i0 >= n or i0 < 0:
            return
        sig = sig[: n - i0] * gain
        gl, gr = math.cos((pan + 1) * math.pi / 4) * 1.414, math.sin((pan + 1) * math.pi / 4) * 1.414
        bus[i0:i0 + len(sig), 0] += sig * gl
        bus[i0:i0 + len(sig), 1] += sig * gr

    def env(dur, attack, tau):
        x = tt(dur)
        return np.minimum(x / max(attack, 1e-4), 1) * np.exp(-x / tau)

    def kick(g=1.0):
        x = tt(0.6)
        fr = 45 + 110 * np.exp(-x / 0.035)
        s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-x / 0.22)
        s[:200] += hp(noise(200 / SR), 2000)[:200] * 0.3
        return s * g

    def snare():
        x = tt(0.3)
        return bp(noise(0.3), 1200, 7000) * np.exp(-x / 0.08) * 0.7 + np.sin(2 * np.pi * 190 * x) * np.exp(-x / 0.05) * 0.5

    def hat():
        return hp(noise(0.06), 8000) * np.exp(-tt(0.06) / 0.015) * 0.5

    def braam(dur=4.5):
        x = tt(dur)
        s = np.zeros_like(x)
        for fq in (41.2, 55.0, 82.4, 110.0, 164.8):
            for det in (-0.004, 0.0, 0.005):
                s += saw(fq * (1 + det), dur, rng.random())
        s = lp(s, 700, 4) / 10
        s *= np.minimum(x / 0.02, 1) * np.exp(-x / 1.9)
        sub = np.sin(2 * np.pi * np.cumsum(38 + 90 * np.exp(-x / 0.08)) / SR) * np.exp(-x / 1.2)
        boom = lp(noise(dur), 300) * np.exp(-x / 0.5) * 0.8
        return s + sub * 0.9 + boom

    def riser(dur):
        x = tt(dur)
        nz = noise(dur)
        chunks = 40
        out = np.zeros_like(nz)
        L = len(nz) // chunks
        for c in range(chunks):
            fc = 300 * (20 ** (c / chunks))
            seg = nz[max(c * L - 2000, 0): (c + 1) * L]
            fs_ = bp(seg, fc, min(fc * 2.2, 18000))
            out[c * L:(c + 1) * L] = fs_[-L:]
        sweep = np.sin(2 * np.pi * np.cumsum(120 + 900 * (x / dur) ** 2) / SR) * 0.25
        return (out * 0.6 + sweep) * (x / dur) ** 2.2

    def whoosh(dur=0.5):
        x = tt(dur)
        nz = noise(dur)
        out = np.zeros_like(nz)
        chunks = 12
        L = len(nz) // chunks
        for c in range(chunks):
            fc = 400 + 2200 * math.sin(math.pi * c / chunks)
            seg = nz[max(c * L - 1000, 0):(c + 1) * L]
            out[c * L:(c + 1) * L] = bp(seg, fc, fc * 2)[-L:]
        return out * np.sin(np.pi * x / dur) ** 2 * 0.8

    def gunshot(seed):
        r = np.random.default_rng(seed)
        x = tt(0.5)
        body = lp(r.standard_normal(len(x)), 4500) * np.exp(-x / 0.045)
        crack = hp(r.standard_normal(len(x)), 3000) * np.exp(-x / 0.006)
        thump = np.sin(2 * np.pi * np.cumsum(60 + 120 * np.exp(-x / 0.02)) / SR) * np.exp(-x / 0.07)
        tail = lp(r.standard_normal(len(x)), 1200) * np.exp(-x / 0.18) * 0.25
        return body * 0.7 + crack * 0.4 + thump * 0.9 + tail

    def sniper():
        x = tt(2.5)
        body = lp(noise(2.5), 3500) * np.exp(-x / 0.09)
        crack = hp(noise(2.5), 2500) * np.exp(-x / 0.01)
        thump = np.sin(2 * np.pi * np.cumsum(45 + 150 * np.exp(-x / 0.03)) / SR) * np.exp(-x / 0.25)
        tail = lp(noise(2.5), 900) * np.exp(-x / 0.7) * 0.4
        return body + crack * 0.6 + thump * 1.2 + tail

    def explosion():
        x = tt(4.0)
        body = lp(noise(4.0), 700, 3) * np.minimum(x / 0.005, 1) * np.exp(-x / 0.7) * 2.2
        sub = np.sin(2 * np.pi * np.cumsum(28 + 60 * np.exp(-x / 0.25)) / SR) * np.exp(-x / 1.0) * 1.2
        cr = hp(noise(4.0) * (rng.random(len(x)) < 0.004), 2000) * np.exp(-x / 1.2) * 3
        return body + sub + cr

    def click():
        x = tt(0.25)
        s = sum(np.sin(2 * np.pi * fq * x) * np.exp(-x / 0.025) for fq in (2100, 3300, 5200)) / 3
        s += np.sin(2 * np.pi * 120 * x) * np.exp(-x / 0.05) * 0.8
        s[:300] += hp(noise(300 / SR), 3000) * 0.6
        return s

    def boom():
        x = tt(2.0)
        return np.sin(2 * np.pi * np.cumsum(32 + 40 * np.exp(-x / 0.1)) / SR) * np.exp(-x / 0.7) + lp(noise(2.0), 220) * np.exp(-x / 0.4) * 0.6

    def note(fq, dur, cutoff=900, tau=0.2, voices=2):
        x = tt(dur)
        s = sum(saw(fq * (1 + 0.004 * v), dur, rng.random()) for v in range(voices)) / voices
        return lp(s, cutoff) * np.minimum(x / 0.005, 1) * np.exp(-x / tau)

    # --- intro + logo
    xd = tt(10.5)
    drone = (np.sin(2 * np.pi * 55 * xd) + 0.6 * np.sin(2 * np.pi * 82.4 * xd + 1) +
             0.5 * lp(noise(10.5), 160)) * np.minimum(xd / 3, 1) * (0.7 + 0.3 * np.sin(xd * 1.3))
    drone *= np.clip((10.5 - xd) / 0.6, 0, 1)
    add(mus, drone, 0, 0.18)
    for k in range(len(CFG["intro_lines"])):
        add(sfx, boom(), 0.3 + 1.9 * k, 0.55)
    add(sfx, riser(1.7), 4.3, 0.6)
    add(sfx, braam(), T_LOGO, 0.9)
    for k in range(14):
        add(mus, hat() * (0.3 + k / 14), 7.5 + 0.18 * k, 0.6, (-0.3 if k % 2 else 0.3))
    add(sfx, riser(1.0), 9.0, 0.45)
    add(sfx, whoosh(0.6), 9.5, 0.6)

    # --- hudba 10–57 s (120 BPM)
    beat = 0.5
    roots = [55.0, 43.65, 49.0, 41.2]
    chords = [(220.0, 261.6, 329.6), (174.6, 220.0, 261.6), (196.0, 246.9, 293.7), (164.8, 207.7, 246.9)]
    t0 = T_GAME
    nbeats = int((T_MONTAGE - t0) / beat)
    for b in range(nbeats):
        tb = t0 + b * beat
        bar = b // 4
        root = roots[bar % 4]
        sec = 0 if tb < T_MODES else 1 if tb < T_ARMORY else 2 if tb < T_ATT else 3
        add(mus, kick(), tb, 0.85)
        if b % 4 in (1, 3):
            add(mus, snare(), tb, 0.45 + 0.1 * (sec >= 2))
        if sec >= 1:
            for h8 in range(2 if sec < 3 else 4):
                add(mus, hat(), tb + h8 * beat / (2 if sec < 3 else 4), 0.35, 0.25 if h8 % 2 else -0.25)
        pattern = [1, 1, 2, 1, 1, 2, 1, 1.5]
        for e8 in range(2):
            fq = root * pattern[(b * 2 + e8) % 8]
            add(mus, note(fq, beat / 2, 380, 0.12, 2), tb + e8 * beat / 2, 0.42)
        if sec in (1, 3):
            ch = chords[bar % 4]
            for s16 in range(4):
                fq = ch[(b * 4 + s16) % 3] * (2 if (b + s16) % 5 == 0 else 1)
                add(mus, note(fq, 0.2, 2600, 0.07, 1), tb + s16 * beat / 4, 0.11, 0.4 if s16 % 2 else -0.4)
        if b % 4 == 0 and sec >= 0:
            for fq in chords[bar % 4]:
                x = tt(2.2)
                pad = lp(saw(fq / 2, 2.2) + saw(fq / 2 * 1.006, 2.2, 0.3), 900) * np.minimum(x / 0.4, 1) * np.clip((2.2 - x) / 0.4, 0, 1)
                add(mus, pad, tb, 0.05 if sec < 2 else 0.08, -0.5)
                add(mus, pad, tb + 0.012, 0.05 if sec < 2 else 0.08, 0.5)
    for k in range(8):
        add(mus, snare(), T_MONTAGE - 1.0 + k * 0.125, 0.3 + k * 0.05)

    # --- efekty
    for tb, kind in gameplay_events():
        if kind == "ar":
            add(sfx, gunshot(int(tb * 1000)), tb, 0.55, 0.15)
        elif kind == "sniper":
            add(sfx, sniper(), tb, 0.95)
        else:
            add(sfx, explosion(), tb, 0.9)
    for c in (T_GAME, 14.0, 18.0, T_MODES, T_ARMORY, T_ATT):
        add(sfx, boom(), c, 0.5)
        add(sfx, whoosh(0.45), c - 0.35, 0.45)
    for k in range(len(CFG["modes"])):
        add(sfx, whoosh(0.5), MODE_START + k * MODE_LEN - 0.1, 0.5, 0.3)
    for k in range(len(CFG["weapons"])):
        add(sfx, whoosh(0.55), WEAPON_START + k * WEAPON_LEN - 0.1, 0.5, -0.3)
        add(sfx, click(), WEAPON_START + k * WEAPON_LEN + 0.45, 0.5)
    for k in range(len(CFG["attachments"])):
        ts = ATT_START + k * ATT_STEP
        add(sfx, whoosh(0.4), ts, 0.45, 0.2)
        add(sfx, click(), ts + ATT_FLY, 0.75)
    # montáž
    add(sfx, riser(2.0), T_MONTAGE - 0.2, 0.55)
    for k in range(len(MONTAGE)):
        add(sfx, kick(1.0), T_MONTAGE + k * 0.2, 0.45 + 0.04 * k)
        add(sfx, gunshot(900 + k), T_MONTAGE + k * 0.2, 0.25)
    add(sfx, braam(6.5), T_FINAL, 1.0)
    add(sfx, boom(), T_FINAL, 0.8)
    xr = tt(6.0)
    ring = (np.sin(2 * np.pi * 880 * xr) + 0.5 * np.sin(2 * np.pi * 1320 * xr)) * np.exp(-xr / 2.0) * np.minimum(xr / 0.3, 1)
    add(sfx, ring, T_FINAL + 0.2, 0.05)
    add(sfx, boom(), T_FINAL + 3.0, 0.4)
    xd = tt(7.0)
    add(mus, (np.sin(2 * np.pi * 55 * xd) + 0.5 * lp(noise(7.0), 140)) * np.exp(-xd / 3), T_FINAL, 0.2)

    # ztišení hudby pod velkými momenty a mix
    duck = np.ones(n)
    for c in (T_LOGO, T_FINAL, 14.0 + SNIPER_SHOT, 18.0 + SHOT_C["explode"]):
        i0 = int(c * SR)
        L = int(1.2 * SR)
        duck[i0:i0 + L] = np.minimum(duck[i0:i0 + L], 0.35 + 0.65 * np.linspace(0, 1, L) ** 2)
    mus *= duck[:, None]
    mus[int((T_FINAL - 0.2) * SR):int(T_FINAL * SR)] *= 0.0
    ir_x = np.arange(int(1.6 * SR)) / SR
    ir = lp(rng.standard_normal(len(ir_x)), 3000) * np.exp(-ir_x / 0.45)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = np.stack([fftconvolve(sfx[:, c] + 0.5 * mus[:, c], ir)[:n] for c in range(2)], axis=1)
    mix = mus * 0.9 + sfx + wet * 0.22
    mix = np.tanh(mix * 1.3) / np.tanh(1.3)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 0.89
    fade = np.clip((DURATION - np.arange(n) / SR) / 1.0, 0, 1)
    mix *= fade[:, None]
    data = (mix[: int(DURATION * SR)] * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(data.tobytes())


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stills:
        for s in args.stills:
            i = int(round(s * FPS))
            img = Image.frombytes("RGB", (W, H), render_frame(i))
            p = os.path.join(OUT_DIR, "still_%05.2f.png" % s)
            img.save(p)
            print(p)
        return
    wav = os.path.join(OUT_DIR, "trailer_audio.wav")
    print("Generuji zvuk…", flush=True)
    make_audio(wav)
    if args.audio_only:
        return
    out = os.path.join(OUT_DIR, "cerny_orel_trailer.mp4")
    nframes = int(DURATION * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-b:v", "6M", "-maxrate", "8M", "-bufsize", "12M",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(args.workers) as pool:
        for k, buf in enumerate(pool.imap(render_frame, range(nframes), chunksize=4)):
            proc.stdin.write(buf)
            if k % 60 == 0:
                print(f"  snímek {k}/{nframes}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("Hotovo:", out)


if __name__ == "__main__":
    main()
