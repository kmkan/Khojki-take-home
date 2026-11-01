"""End-to-end Khojki OCR: page images -> line boxes -> recognition -> Khojki Unicode (UTF-8).

    python run.py data/pages/page_28.png data/pages/page_30.png --out results/output
    python run.py --all --out results/output          # every page in data/pages

Writes, per page: <page>.txt (one line of Khojki Unicode per detected line, reading order),
<page>.json (boxes + text), <page>_lines.png (detected lines overlay).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.khojki import validate  # noqa: E402
from src.recognize import Recognizer  # noqa: E402
from src.segment import draw_overlay, find_lines  # noqa: E402


def process(path: Path, rec: Recognizer, out: Path) -> dict:
    t0 = time.time()
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plain covers / blank leaves: almost no pixels darker than their local background.
    # Conservative on purpose (skips pp. 8, 27, 35); the stain p.16 and textured blank p.25
    # are NOT caught -- a global threshold that catches them also drops orange text pages.
    g = gray.astype("float32")
    dark = float(((g - cv2.GaussianBlur(g, (0, 0), 15)) < -25).mean())
    no_text = dark < 0.04
    lines = [] if no_text else find_lines(gray)[0]
    t1 = time.time()
    res = []
    for k, l in enumerate(lines):
        text = rec.recognize(gray[l.y0:l.y1, l.x0:l.x1])
        res.append({"line": k, "box": [int(l.x0), int(l.y0), int(l.x1), int(l.y1)], "text": text, "problems": validate(text)})
    t2 = time.time()
    stem = path.stem
    (out / f"{stem}.txt").write_text("\n".join(r["text"] for r in res) + "\n", encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps({"page": path.name, "no_text": no_text, "lines": res, "seconds": {
        "segmentation": round(t1 - t0, 3), "recognition": round(t2 - t1, 3)}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    cv2.imwrite(str(out / f"{stem}_lines.png"), draw_overlay(img, lines))
    return {"page": path.name, "lines": len(lines), "seg_s": t1 - t0, "rec_s": t2 - t1, "no_text": no_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=str(ROOT / "models/recognizer.pkl"))
    ap.add_argument("--out", default=str(ROOT / "results/output"))
    a = ap.parse_args()
    pages = sorted((ROOT / "data/pages").glob("*.png")) if a.all else [Path(p) for p in a.pages]
    if not pages:
        ap.error("give page images or --all")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recognizer.load(a.model)
    for p in pages:
        s = process(p, rec, out)
        tag = "  (no text: skipped)" if s["no_text"] else ""
        print(f"{s['page']}: {s['lines']} lines, seg {s['seg_s']:.2f}s, rec {s['rec_s']:.2f}s{tag}")


if __name__ == "__main__":
    main()
