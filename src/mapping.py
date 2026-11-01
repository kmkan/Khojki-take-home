"""Stage 3: piece labels -> Khojki Unicode (UTF-8).

1. Re-assemble multi-piece units with the piece table (greedy longest match), e.g.
   [ka -e#.. , BAR] -> 'ka -o',  [a, BAR] -> 'aa',  [ka, BAR] -> 'ka -aa'.
   Only table entries containing at least one piece that is NOT a stand-alone unit are used,
   so e.g. ['2', ':'] is read as digit+separator rather than as NYA.
   Pieces may only be joined when the gap between them is small.
2. Insert spaces at large gaps (but never right after a word separator).
3. Convert tokens to code points, NFC, validate.
"""
from __future__ import annotations

import unicodedata as ud

from .khojki import from_tokens, normalize, validate


VOWEL_TOKENS = {"-aa", "-i", "-ii", "-u", "-e", "-ai", "-o", "-au", "-vocalicr"}
CONSONANT_TOKENS = set(("ka kha ga gga gha nga ca cha ja jja nya tta ttha dda ddha nna ta tha da ddda dha "
                        "na pa pha ba bba bha ma ya ra la va sa ha lla qa").split())


class Mapper:
    def __init__(self, table: dict[str, list[str]]):
        self.alias = dict(table.get("__alias__", {}))
        table = {k: v for k, v in table.items() if not k.startswith("__")}
        self.table = table
        standalone = {u for u, labs in table.items() if len(labs) == 1 and labs[0] == u}
        self.seqs: dict[tuple[str, ...], str] = {}
        for u, labs in table.items():
            if len(labs) > 1 and any(l not in standalone for l in labs):
                self.seqs.setdefault(tuple(labs), u)
        self.maxlen = max((len(k) for k in self.seqs), default=1)
        self.standalone = standalone

    def to_tokens(self, labels: list[str], gaps: list[float], body: float,
                  space_gap: float = 0.55, join_gap: float = 0.35, prov: list | None = None) -> list[str]:
        """prov (optional list) receives, per output token, 'rec' if the token is a classifier
        label used as-is, or 'map' if a mapping rule produced it (space, piece assembly,
        alias, dropped/stray bar)."""
        prov = prov if prov is not None else []
        # orphan non-initial pieces (no preceding piece of the same unit) -> alias
        labels = list(labels)
        aliased: set[int] = set()
        for k, l in enumerate(labels):
            if l in self.alias and (k == 0 or labels[k - 1].split("#")[0] != l.split("#")[0]):
                labels[k] = self.alias[l]
                aliased.add(k)
        out: list[str] = []
        i = 0
        while i < len(labels):
            hit = None
            for L in range(min(self.maxlen, len(labels) - i), 1, -1):
                key = tuple(labels[i:i + L])
                if key in self.seqs and all(g < join_gap * body for g in gaps[i:i + L - 1]):
                    hit = (L, self.seqs[key])
                    break
            if not hit and "#" in labels[i]:
                # orphan piece of a split unit (e.g. GA's left part): swallow the following
                # piece(s) of the same unit or bars, up to the unit's piece count
                parent = labels[i].split("#")[0]
                n = len(self.table.get(parent, [1]))
                j = i + 1
                while j < len(labels) and j - i < n and gaps[j - 1] < join_gap * body and (
                        labels[j] == "BAR" or labels[j].split("#")[0] == parent):
                    j += 1
                unit = parent
                if j - i < n and self.table.get(parent, [None])[-1] == "BAR":
                    # the AA bar this piece expects is missing: drop the vowel it encodes
                    unit = _drop_bar(parent)
                hit = (j - i, unit)
            src = "map"
            if hit:
                unit, step = hit[1], hit[0]
            else:
                src = "map" if i in aliased else "rec"
                lab = labels[i]
                if lab in self.standalone:
                    unit = lab
                elif lab == "BAR":          # stray AA bar: attach as vowel sign
                    prev = out[-1].split() if out else []
                    has_vowel = any(t in VOWEL_TOKENS for t in prev)
                    unit = "-aa" if prev and prev[0] in CONSONANT_TOKENS and not has_vowel else None
                    src = "map"
                else:                       # orphan piece of a split unit: use its parent unit
                    unit = lab.split("#")[0]
                    src = "map"
                step = 1
            if i > 0 and gaps[i - 1] > space_gap * body and out and out[-1] != ":" and unit and unit != ":":
                out.append("_")
                prov.append("map")
            if unit:
                out.append(unit)
                prov.append(src)
            i += step
        return out

    def to_unicode(self, labels: list[str], gaps: list[float], body: float, **kw) -> str:
        return self.to_unicode_prov(labels, gaps, body, **kw)[0]

    def to_unicode_prov(self, labels, gaps, body, **kw) -> tuple[str, list[str]]:
        """Unicode text plus per-code-point provenance ('rec' / 'map')."""
        prov: list[str] = []
        toks = self.to_tokens(labels, gaps, body, prov=prov, **kw)
        text, cprov = "", []
        for t, p in zip(toks, prov):
            s = from_tokens(t)
            text += s
            cprov += [p] * len(s)
        assert normalize(text) == text and ud.normalize("NFC", text) == text
        return text, cprov


def _drop_bar(unit: str) -> str:
    toks = unit.split()
    rep = {"-aa": None, "-o": "-e", "-au": "-ai", "aa": "a"}
    out = []
    for t in toks:
        if t in rep:
            if rep[t]:
                out.append(rep[t])
        else:
            out.append(t)
    return " ".join(out) or unit


def check(text: str) -> list[str]:
    return validate(text)
