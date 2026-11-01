"""Synthetic training data: render Khojki text with Noto Sans/Serif Khojki, degrade it to
look like a 1900s letterpress scan, cut it into glyph clusters with the SAME code used at
inference, and label each cluster.

Labels are 'piece labels'. Most units (consonant + any above/below marks) are one cluster;
some split (the AA bar, GA, NYA, YA, ...). Pieces that look the same as another unit
(e.g. every AA bar) are unified, and PIECE_TABLE records how piece sequences map back to
Unicode. That table is the 'Unicode mapping' stage (src/mapping.py).
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .clusters import band, clusters, normalize_patch
from .khojki import from_tokens
from .render import FONTS, font

CONSONANTS = ("ka kha ga gga gha nga ca cha ja jja nya tta ttha dda ddha nna ta tha da ddda dha na "
              "pa pha ba bba bha ma ya ra la va sa ha lla qa").split()
# vowel signs / marks attested in the printed books (rarer combinations removed: they cost
# accuracy on the common ones -- see experiments in RETRO.md)
MODS = ["", "-aa", "-i", "-ii", "-u", "-e", "-ai", "-o", "-au", "-m", "-aa -m", "-e -m", "-u -m",
        "-nukta", "-shadda"]
INDEP = ["a", "aa", "i", "u", "e", "ai", "o", "au", "a -m", "aa -m", "e -m", "u -m"]
SYMBOLS = [str(d) for d in range(10)] + [":", "|", "||", "§", "(", ")"]


def units() -> list[str]:
    return [f"{c} {m}".strip() for c in CONSONANTS for m in MODS] + INDEP + SYMBOLS


# ---------------------------------------------------------------------------------------
REF_CONTEXT = ("va -e -m", "ja -u")  # tall mark before, descender after -> realistic line band


def _clean_pieces(tok: str, fp: str, size: int = 48) -> list[np.ndarray]:
    """Pieces of `tok` rendered cleanly between REF_CONTEXT units, in the full line band."""
    st = random.getstate()
    random.seed(1)  # no random spaces
    img, xr = sample_line([REF_CONTEXT[0], tok, REF_CONTEXT[1]], fp, size)
    random.setstate(st)
    boxes, ink, body = clusters(img)
    if not boxes:
        return []
    y0, y1 = band(boxes, body, ink)
    a, b = xr[1]
    return [normalize_patch(ink, bx, y0, y1) for bx in boxes if a - 2 <= (bx[0] + bx[2]) / 2 <= b + 2]


def render_padded(text: str, fp: str, size: int) -> Image.Image:
    f = font(fp, size)
    l, t, r, b = f.getbbox(text)
    asc, desc = f.getmetrics()
    im = Image.new("L", (r - l + 20, asc + desc + 20), 255)
    ImageDraw.Draw(im).text((10 - l, 10), text, font=f, fill=0)
    return im


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def build_piece_table(sim_thresh: float = 0.85) -> tuple[dict[str, list[str]], list[str]]:
    """unit token -> list of piece labels (consistent across fonts where possible)."""
    fp = FONTS[0]  # reference font for the decomposition (Noto Sans Khojki)
    singles: dict[str, np.ndarray] = {}
    multi: dict[str, list[np.ndarray]] = {}
    for u in units():
        p = _clean_pieces(u, fp)
        if len(p) == 1:
            singles[u] = p[0]
        elif len(p) > 1:
            multi[u] = p
    bar = _clean_pieces("ka -aa", fp)[-1]
    refs = dict(singles)
    refs["BAR"] = bar
    table: dict[str, list[str]] = {u: [u] for u in singles}
    for u, pieces in multi.items():
        labs = []
        for i, p in enumerate(pieces):
            toks = u.split()
            if i == len(pieces) - 1 and any(t in ("-aa", "-o", "-au", "aa") for t in toks) and _sim(p, bar) > 0.7:
                labs.append("BAR")       # trailing AA bar (possibly with a mark): always BAR
                continue
            best, score = None, sim_thresh
            for name, r in refs.items():
                s = _sim(p, r)
                if s > score:
                    best, score = name, s
            labs.append(best if best else f"{u}#{i}")
            if not best:
                refs[f"{u}#{i}"] = p
        table[u] = labs
    labels = sorted({l for v in table.values() for l in v})
    # alias for non-initial pieces seen on their own (e.g. NYA's colon-like tail without the
    # NYA body): the most similar stand-alone unit
    alias = {}
    for name, p in refs.items():
        if "#" in name and not name.endswith("#0"):
            s, best = max((_sim(p, r), n) for n, r in refs.items() if n in (":", "|", "BAR"))
            if s > 0.6:
                alias[name] = best
    table["__alias__"] = alias  # type: ignore[assignment]
    return table, labels


# ---------------------------------------------------------------------------------------
def degrade(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make a clean render look like a low-resolution letterpress scan."""
    img = gray.astype(np.float32)
    # ink spread: letterpress strokes are much heavier than Noto Regular (kernel ~ size)
    k = max(1, int(round(rng.uniform(0.01, 0.05) * img.shape[0])))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    img = cv2.erode(img, ker) if rng.random() < 0.85 else cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    if rng.random() < 0.7:                                     # shear (italic-ish type / skew)
        sh = rng.uniform(-0.12, 0.12)
        h, w = img.shape
        M = np.float32([[1, sh, -sh * h / 2], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.6))
    s = rng.uniform(0.35, 0.8)                                 # low-res capture
    h, w = img.shape
    img = cv2.resize(cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA),
                     (w, h), interpolation=cv2.INTER_LINEAR)
    paper = rng.uniform(150, 235)
    inkv = rng.uniform(0, 70)
    img = inkv + (img / 255.0) * (paper - inkv)
    img += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 18), img.shape)
    if rng.random() < 0.4:                                     # specks
        for _ in range(rng.randrange(1, 6)):
            y, x = rng.randrange(img.shape[0]), rng.randrange(img.shape[1])
            cv2.circle(img, (x, y), rng.choice([1, 1, 2]), inkv, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


SPACE_P = 0.15
CONTEXT = ["ka", "va -e", "ja -u", "ma -aa", "ra", "sa -m", "pa -u", ":", "ta", "da", "4", "kha -aa"]


def sample_line(units_seq: list[str], fp: str, size: int):
    """Render tokens separated by optional spaces; return image + per-unit x-ranges."""
    f = font(fp, size)
    xs, text = [], ""
    for u in units_seq:
        s = from_tokens(u)
        x0 = f.getlength(text) if text else 0.0
        text += s
        xs.append((x0, f.getlength(text)))
        if random.random() < SPACE_P:
            text += " "
    l, t, r, b = f.getbbox(text)
    asc, desc = f.getmetrics()
    W = int(max(r, xs[-1][1])) + 30
    im = Image.new("L", (W, asc + desc + 24), 255)
    ImageDraw.Draw(im).text((12, 12), text, font=f, fill=0)
    return np.array(im), [(a + 12, b + 12) for a, b in xs]


def generate(n_per_unit: int, table: dict[str, list[str]], seed: int = 0, verbose: bool = True,
             do_degrade: bool = True):
    rng = random.Random(seed)
    random.seed(seed)
    X, y = [], []
    us = [u for u in table if not u.startswith("__")]
    for it in range(n_per_unit):
        rng.shuffle(us)
        for u in us:
            fp = rng.choice(FONTS)
            size = rng.randrange(28, 64)
            seq = [rng.choice(CONTEXT), u, rng.choice(CONTEXT)]
            img, xr = sample_line(seq, fp, size)
            if do_degrade:
                img = degrade(img, rng)
            boxes, ink, body = clusters(img)
            if not boxes:
                continue
            ly0, ly1 = band(boxes, body, ink)
            a, b = xr[1]
            mine = [bx for bx in boxes if a - 2 <= (bx[0] + bx[2]) / 2 <= b + 2]
            if len(mine) != len(table[u]):
                continue  # degradation merged/split pieces: skip rather than mislabel
            for bx, lab in zip(mine, table[u]):
                X.append(normalize_patch(ink, bx, ly0, ly1).ravel())
                y.append(lab)
        if verbose:
            print(f"  pass {it + 1}/{n_per_unit}: {len(y)} samples", flush=True)
    return np.array(X, np.float32), np.array(y)
