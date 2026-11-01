"""Render Unicode Khojki text to images with Noto Khojki fonts (HarfBuzz shaping via Raqm)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
FONTS = sorted(str(p) for p in FONT_DIR.glob("Noto*Khojki*.ttf"))
LAYOUT = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
_cache: dict = {}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path, size, layout_engine=LAYOUT)
    return _cache[key]


def render_line(text: str, font_path: str | None = None, size: int = 48, pad: int = 12) -> Image.Image:
    f = font(font_path or FONTS[0], size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 255)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=0)
    return im
