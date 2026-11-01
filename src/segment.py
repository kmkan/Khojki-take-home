"""Stage 1: page image -> text-line boxes.

Classical, training-free approach (works on printed pages; see retro for limits):
  1. grayscale + Sauvola binarisation (robust to the yellowed / stained paper)
  2. remove ruled lines and page frames with long morphological openings
  3. drop specks and huge blobs (stains, ornaments) by connected-component stats
  4. smear components horizontally (kernel ~ 1.2 x estimated glyph height) so the
     characters of one line merge, but table columns (wide gutters) stay separate
  5. each merged blob = one line box; boxes are sorted column-major, top-to-bottom
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


@dataclass
class Line:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


def binarize(gray: np.ndarray, window: int = 41, k: float = 0.2) -> np.ndarray:
    """Return uint8 mask, 255 = ink."""
    t = threshold_sauvola(gray, window_size=window, k=k)
    return ((gray < t) * 255).astype(np.uint8)


def remove_rules(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove ruled lines. Returns (ink without rules, vertical-rule mask)."""
    h, w = ink.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 14), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 18)))
    vrules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk) | vrules
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
    return cv2.subtract(ink, rules), vrules


def clean_components(ink: np.ndarray) -> tuple[np.ndarray, float]:
    """Drop specks / giant blobs. Returns cleaned mask and median glyph height."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    hs = st[1:, cv2.CC_STAT_HEIGHT]
    areas = st[1:, cv2.CC_STAT_AREA]
    plausible = hs[(areas > 15) & (hs > 4)]
    med_h = float(np.median(plausible)) if len(plausible) else 12.0
    keep = np.zeros(n, bool)
    for i in range(1, n):
        a, hh, ww = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_WIDTH]
        if a < 6:                                   # dust
            continue
        if hh > 6 * med_h or ww > 25 * med_h:       # stains / ornaments / leftover rules
            continue
        if hh > 3 * med_h and ww < 0.25 * hh:       # vertical rule fragments
            continue
        if ww > 4 * med_h and hh < 0.2 * ww and a < 0.5 * ww * hh and hh < 0.4 * med_h:  # horiz. rule bits
            continue
        keep[i] = True
    return (keep[lab] * 255).astype(np.uint8), med_h


def find_lines(gray: np.ndarray, min_width_factor: float = 0.6) -> tuple[list[Line], np.ndarray]:
    ink = binarize(gray)
    ink, vrules = remove_rules(ink)
    ink, xh = clean_components(ink)
    kx = max(3, int(round(1.2 * xh)))
    ky = max(1, int(round(0.15 * xh)))
    smear = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    # column rules act as barriers: text on either side must not merge
    barrier = cv2.dilate(vrules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(xh))))
    smear = cv2.subtract(smear, barrier)
    n, lab, st, _ = cv2.connectedComponentsWithStats(smear, 8)
    lines: list[Line] = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 0.5 * xh or w < min_width_factor * xh:
            continue
        # shrink back to real ink extent
        sub = ink[y:y + h, x:x + w] & ((lab[y:y + h, x:x + w] == i) * 255).astype(np.uint8)
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            continue
        pad = int(0.25 * xh)
        lines.append(Line(max(0, x + xs.min() - pad), max(0, y + ys.min() - pad),
                          min(gray.shape[1], x + xs.max() + pad + 1),
                          min(gray.shape[0], y + ys.max() + pad + 1)))
    lines = [s for l in lines for s in _split_tall(l, ink, xh)]
    lines = [l for l in lines if _ink_count(ink, l) > 2 * xh * xh * 0.3]   # drop specks
    lines = _merge_vertical_fragments(lines, xh)
    lines = _merge_same_row(lines, xh, vrules)
    lines = _drop_nested(lines)
    lines = [l for l in lines if l.w >= 1.5 * xh]
    return _reading_order(lines), ink


def _drop_nested(lines: list[Line]) -> list[Line]:
    """Remove boxes that are (almost) fully inside a bigger box."""
    keep = []
    for i, a in enumerate(lines):
        inside = False
        for j, b in enumerate(lines):
            if i == j or b.w * b.h < a.w * a.h:
                continue
            ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
            iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
            if ix * iy > 0.7 * a.w * a.h and (b.w * b.h > a.w * a.h or j < i):
                inside = True
                break
        if not inside:
            keep.append(a)
    return keep


def _rule_between(vrules: np.ndarray, a: Line, b: Line) -> bool:
    x0, x1 = sorted((a.x1, b.x0))
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        x0, x1 = min(a.x1, b.x0) - 2, max(a.x1, b.x0) + 2
    return bool(vrules[max(0, y0):y1, max(0, x0 - 3):x1 + 3].any())


def _merge_same_row(lines: list[Line], xh: float, vrules: np.ndarray) -> list[Line]:
    """Join boxes on the same baseline separated by a word-sized gap (e.g. a verse number)."""
    changed = True
    while changed:
        changed = False
        lines.sort(key=lambda l: l.x0)
        for i, a in enumerate(lines):
            for j in range(i + 1, len(lines)):
                b = lines[j]
                yov = min(a.y1, b.y1) - max(a.y0, b.y0)
                gap = b.x0 - a.x1
                if yov > 0.5 * min(a.h, b.h) and -xh < gap < 1.3 * xh and not _rule_between(vrules, a, b):
                    a.x0, a.y0, a.x1, a.y1 = min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1)
                    lines.pop(j)
                    changed = True
                    break
            if changed:
                break
    return lines


def _reading_order(lines: list[Line]) -> list[Line]:
    """Group boxes into rows by vertical overlap, then left-to-right within a row."""
    rows: list[list[Line]] = []
    for l in sorted(lines, key=lambda l: (l.y0 + l.y1) / 2):
        c = (l.y0 + l.y1) / 2
        if rows:
            r = rows[-1]
            ry0, ry1 = min(x.y0 for x in r), max(x.y1 for x in r)
            if ry0 <= c <= ry1:
                r.append(l)
                continue
        rows.append([l])
    return [l for r in rows for l in sorted(r, key=lambda l: l.x0)]


def _ink_count(ink: np.ndarray, l: Line) -> int:
    return int((ink[l.y0:l.y1, l.x0:l.x1] > 0).sum())


def _split_tall(l: Line, ink: np.ndarray, xh: float) -> list[Line]:
    """A box taller than ~2 line heights holds several lines: cut at projection minima."""
    if l.h < 2.6 * xh:
        return [l]
    prof = (ink[l.y0:l.y1, l.x0:l.x1] > 0).sum(1).astype(float)
    prof = np.convolve(prof, np.ones(3) / 3, 'same')
    thr = 0.08 * prof.max()
    cuts, inside, start = [], False, 0
    for y, v in enumerate(prof):
        if v > thr and not inside:
            inside, start = True, y
        elif v <= thr and inside:
            inside = False
            if y - start > 0.4 * xh:
                cuts.append((start, y))
    if inside and len(prof) - start > 0.4 * xh:
        cuts.append((start, len(prof)))
    if len(cuts) <= 1:
        return [l]
    out = []
    for a, b in cuts:
        sub = ink[l.y0 + a:l.y0 + b, l.x0:l.x1]
        xs = np.nonzero(sub.sum(0))[0]
        pad = int(0.2 * xh)
        out.append(Line(l.x0 + xs.min(), max(0, l.y0 + a - pad), l.x0 + xs.max() + 1, min(ink.shape[0], l.y0 + b + pad)))
    return out


def _merge_vertical_fragments(lines: list[Line], xh: float) -> list[Line]:
    """Vowel signs / anusvara above the line can form their own thin blob; fold them in."""
    lines = sorted(lines, key=lambda l: l.h, reverse=True)
    out: list[Line] = []
    for l in lines:
        merged = False
        for o in out:
            xov = min(l.x1, o.x1) - max(l.x0, o.x0)
            vgap = max(o.y0 - l.y1, l.y0 - o.y1)
            if xov > 0.6 * min(l.w, o.w) and vgap < 0.35 * xh and l.h < 0.6 * o.h:
                o.x0, o.y0, o.x1, o.y1 = min(o.x0, l.x0), min(o.y0, l.y0), max(o.x1, l.x1), max(o.y1, l.y1)
                merged = True
                break
        if not merged:
            out.append(Line(l.x0, l.y0, l.x1, l.y1))
    return out


def draw_overlay(img_bgr: np.ndarray, lines: list[Line]) -> np.ndarray:
    vis = img_bgr.copy()
    for k, l in enumerate(lines):
        cv2.rectangle(vis, (l.x0, l.y0), (l.x1, l.y1), (0, 0, 255), 2)
        cv2.putText(vis, str(k), (l.x0, max(12, l.y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return vis
