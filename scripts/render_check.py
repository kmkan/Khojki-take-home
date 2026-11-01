"""Visual check of transcriptions: scanned line crop on top, Noto rendering of the
transcription underneath, code points listed. Use it to verify every test line.

    python scripts/render_check.py data/test_lines.tsv results/render_check
"""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.khojki import names, validate  # noqa: E402
from src.render import render_line  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(tsv: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    small = ImageFont.load_default(size=14)
    with open(tsv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        crop = Image.open(ROOT / r["image"]).convert("L")
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        ren = render_line(r["crop_text"], size=int(crop.height * 0.55))
        lab = " | ".join(names(r["crop_text"]))
        W = max(crop.width, ren.width, 900)
        H = crop.height + ren.height + 70
        sheet = Image.new("L", (W, H), 255)
        sheet.paste(crop, (0, 0))
        sheet.paste(ren, (0, crop.height + 5))
        d = ImageDraw.Draw(sheet)
        for i in range(0, len(lab), 120):
            d.text((4, crop.height + ren.height + 10 + (i // 120) * 16), lab[i:i + 120], fill=0, font=small)
        sheet.save(out / f"{r['id']}.png")
        probs = validate(r["crop_text"])
        print(r["id"], "OK" if not probs else probs)


if __name__ == "__main__":
    main(*sys.argv[1:3])
