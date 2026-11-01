"""Character error rate (code-point level) and edit-operation listing."""
from __future__ import annotations

import unicodedata as ud


def align(ref: str, hyp: str):
    """Levenshtein alignment over code points. Returns (distance, ops) where ops are
    ('=', r, h) | ('S', r, h) | ('D', r, '') | ('I', '', h)."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", hyp[j - 1]))
            j -= 1
    return D[n][m], ops[::-1]


def cer(ref: str, hyp: str) -> tuple[int, int]:
    return align(ref, hyp)[0], len(ref)


def short(c: str) -> str:
    if not c:
        return "∅"
    if c == " ":
        return "SPACE"
    return ud.name(c, f"U+{ord(c):04X}").replace("KHOJKI ", "").replace("LETTER ", "").replace("GUJARATI ", "GU ")
