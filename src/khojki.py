"""Khojki Unicode helpers: inventory, validation, normalisation.

Encoding policy (see RETRO.md):
  * letters / vowel signs / marks : Khojki block U+11200-U+11241 (Unicode 15)
  * word separator ':'            : U+1123A KHOJKI WORD SEPARATOR (never ASCII ':')
  * digits                        : Gujarati digits U+0AE6-U+0AEF. Unicode has no Khojki
                                    digits; Gujarati digits carry scx=Gujr Khoj.
  * danda / double danda          : U+11238 / U+11239
  * space                         : U+0020, only where the print shows a visible gap
"""
from __future__ import annotations

import unicodedata as ud

KHOJKI = [chr(c) for c in range(0x11200, 0x11242) if ud.category(chr(c)) != "Cn"]
LETTERS = [c for c in KHOJKI if ud.category(c) == "Lo"]
VOWEL_SIGNS = [chr(c) for c in range(0x1122C, 0x11234)] + ["\U00011241"]
MARKS = [c for c in KHOJKI if ud.category(c) in ("Mn", "Mc")]
DIGITS = [chr(c) for c in range(0x0AE6, 0x0AF0)]  # Gujarati digits 0-9
WORD_SEP = "\U0001123A"
DANDA, DOUBLE_DANDA = "\U00011238", "\U00011239"
VIRAMA = "\U00011235"
OTHER = [" ", "(", ")", "-"]

ALPHABET = sorted(set(KHOJKI + DIGITS + OTHER))

# Characters people (and LLMs) commonly type by mistake, mapped to the policy code point.
CONFUSABLES = {
    ":": WORD_SEP, "ः": WORD_SEP,           # ASCII colon, Devanagari visarga
    **{str(i): DIGITS[i] for i in range(10)},    # ASCII digits
    **{chr(0x0966 + i): DIGITS[i] for i in range(10)},  # Devanagari digits
    "।": DANDA, "॥": DOUBLE_DANDA,
}


def normalize(text: str) -> str:
    """NFC, map known confusables to the policy code points, collapse whitespace."""
    text = ud.normalize("NFC", text)
    text = "".join(CONFUSABLES.get(c, c) for c in text)
    return " ".join(text.split())


def validate(text: str) -> list[str]:
    """Return a list of problems (empty list == valid)."""
    problems = []
    allowed = set(ALPHABET)
    prev = " "
    for i, c in enumerate(text):
        if c not in allowed:
            problems.append(f"pos {i}: U+{ord(c):04X} {ud.name(c, '?')} not in allowed inventory")
        if c in MARKS and (prev not in LETTERS and prev not in MARKS):
            problems.append(f"pos {i}: mark U+{ord(c):04X} {ud.name(c)} has no base letter")
        if c in VOWEL_SIGNS and prev in VOWEL_SIGNS:
            problems.append(f"pos {i}: two vowel signs in a row")
        prev = c
    if ud.normalize("NFC", text) != text:
        problems.append("text is not NFC")
    return problems


def describe(text: str) -> str:
    return " ".join(f"U+{ord(c):04X}" if c != " " else "_" for c in text)


def names(text: str) -> list[str]:
    return [ud.name(c, "SPACE").replace("KHOJKI ", "") for c in text]


# --- token transliteration (for writing / auditing transcriptions) -------------------
# Independent letters by lowercase short name, dependent signs prefixed with '-',
# ':' word separator, '_' space, digits 0-9 -> Gujarati digits.
_TOK = {}
for _c in KHOJKI:
    _n = ud.name(_c).replace("KHOJKI ", "")
    if _n.startswith("LETTER "):
        _TOK[_n[7:].lower().replace(" ", "")] = _c
    elif _n.startswith("VOWEL SIGN "):
        _TOK["-" + _n[11:].lower().replace(" ", "")] = _c
_TOK.update({"-anusvara": "\U00011234", "-m": "\U00011234", "-virama": VIRAMA, "-nukta": "\U00011236",
             "-shadda": "\U00011237", "-sukun": "\U0001123E", ":": WORD_SEP, "|": DANDA, "||": DOUBLE_DANDA,
             "_": " ", "(": "(", ")": ")", "§": "\U0001123B", "-": "-"})
_TOK.update({str(i): DIGITS[i] for i in range(10)})
_REV = {v: k for k, v in _TOK.items() if k not in ("-m",)}


def from_tokens(s: str) -> str:
    """'4 _ ja -u ga :'  ->  Khojki Unicode string."""
    out = []
    for t in s.split():
        if t not in _TOK:
            raise KeyError(f"unknown token {t!r}")
        out.append(_TOK[t])
    return "".join(out)


def to_tokens(text: str) -> str:
    return " ".join(_REV.get(c, f"<U+{ord(c):04X}>") for c in text)
