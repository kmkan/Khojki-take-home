"""Extract the page images from the sample PDF (embedded scans, native resolution).

    python scripts/extract_pages.py path/to/sample_khojki_script_variations.pdf

Writes data/pages/page_01.png ... page_47.png. The scans are not committed to the repo
(the brief asks that they are used for this exercise only).
"""
import sys
from pathlib import Path

import pypdf

out = Path(__file__).resolve().parent.parent / "data" / "pages"
out.mkdir(parents=True, exist_ok=True)
reader = pypdf.PdfReader(sys.argv[1])
for i, page in enumerate(reader.pages, 1):
    page.images[0].image.convert("RGB").save(out / f"page_{i:02d}.png")
print(f"wrote {len(reader.pages)} pages to {out}")
