"""Split a line image into glyph clusters (left-to-right).

A cluster = connected components that overlap horizontally (a letter plus the marks
above/below it, the two dots of the word separator, ...). Used identically on synthetic
renders (training) and on real crops (inference), so the classifier sees the same kind
of input in both cases.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def ink_mask(gray: np.ndarray) -> np.ndarray:
    if gray.max() - gray.min() < 30:
        return np.zeros_like(gray, np.uint8)
    win = max(15, (min(gray.shape) // 2) | 1)
    t = threshold_sauvola(gray, window_size=win, k=0.25)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (((gray < t) & (gray < otsu + 10)) * 255).astype(np.uint8)


def clusters(gray: np.ndarray, overlap: float = 0.3) -> tuple[list[tuple[int, int, int, int]], np.ndarray, float]:
    """Return boxes (x0,y0,x1,y1) of clusters, the ink mask, and the line's body height."""
    ink = ink_mask(gray)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = [tuple(st[i]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= max(4, ink.shape[0] // 25)]
    if not comps:
        return [], ink, float(gray.shape[0])
    body = float(np.median([c[3] for c in comps if c[3] > 3] or [gray.shape[0] / 2]))
    boxes = [[x, y, x + w, y + h] for x, y, w, h, a in comps]
    boxes.sort(key=lambda b: b[0])
    merged: list[list[int]] = []
    for b in boxes:
        if merged:
            m = merged[-1]
            ov = min(m[2], b[2]) - max(m[0], b[0])
            narrow = min(m[2] - m[0], b[2] - b[0])
            if ov > overlap * narrow:
                m[0], m[1], m[2], m[3] = min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])
                continue
        merged.append(list(b))
    # drop specks and ruled-line remnants at the crop edges
    W = ink.shape[1]
    keep = []
    for m in merged:
        x0, y0, x1, y1 = m
        w, h = x1 - x0, y1 - y0
        area = int((ink[y0:y1, x0:x1] > 0).sum())
        if area < 0.05 * body * body and h < 0.4 * body and w < 0.4 * body:
            continue                                            # speck
        if (x0 <= 2 or x1 >= W - 2) and w < 0.35 * body and h > 1.3 * body:
            continue                                            # column rule cut by the crop
        keep.append(tuple(m))
    return keep, ink, body


def band(boxes, body: float, ink: np.ndarray) -> tuple[int, int]:
    """Vertical band for patch normalisation, from the line's horizontal ink profile:
    the 'core' = longest run of rows holding >= 30% of the peak ink; the band is the core
    centre +- 1.6 core heights. Stable regardless of which marks / dots the line has."""
    x0 = min(b[0] for b in boxes)
    x1 = max(b[2] for b in boxes)
    prof = (ink[:, x0:x1] > 0).sum(1).astype(float)
    if prof.max() == 0:
        return 0, ink.shape[0]
    on = prof >= 0.3 * prof.max()
    best, cur, bs, s0 = 0, 0, 0, 0
    for y, v in enumerate(on):
        if v:
            if cur == 0:
                s0 = y
            cur += 1
            if cur > best:
                best, bs = cur, s0
        else:
            cur = 0
    core = max(best, 1)
    yc = bs + core / 2
    return int(round(yc - 1.6 * core)), int(round(yc + 1.6 * core))


def gaps(boxes: list[tuple[int, int, int, int]]) -> list[int]:
    return [boxes[i + 1][0] - boxes[i][2] for i in range(len(boxes) - 1)]


PATCH_H, PATCH_W = 32, 40


def normalize_patch(ink: np.ndarray, box, line_y0: int, line_y1: int) -> np.ndarray:
    """Cut the cluster's columns over the full line band (so marks above / below and the
    dots of the word separator keep their vertical position), scale band height to
    PATCH_H keeping aspect ratio, centre in a PATCH_W-wide canvas. float32 in [0,1]."""
    x0, _, x1, _ = box
    y0, y1 = int(line_y0), int(line_y1)
    padt, padb = max(0, -y0), max(0, y1 - ink.shape[0])
    p = ink[max(0, y0):min(ink.shape[0], y1), x0:x1].astype(np.float32) / 255.0
    if padt or padb:
        p = np.pad(p, ((padt, padb), (0, 0)))
    h, w = p.shape
    if h == 0 or w == 0:
        return np.zeros((PATCH_H, PATCH_W), np.float32)
    s = PATCH_H / h
    nw = max(1, min(PATCH_W, int(round(w * s))))
    p = cv2.resize(p, (nw, PATCH_H), interpolation=cv2.INTER_AREA)
    out = np.zeros((PATCH_H, PATCH_W), np.float32)
    out[:, (PATCH_W - nw) // 2:(PATCH_W - nw) // 2 + nw] = p
    return out
