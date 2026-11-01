"""Builds data/test_lines.tsv from the token transcriptions below.

page_tokens : the full line as printed on the page (ground truth)
crop_tokens : what the automatically detected crop contains (differs only when line
              detection lost something, e.g. a verse number) -> isolates segmentation errors
Transcribed diplomatically (by glyph shape, following the Unicode chart / Noto glyphs, which
are themselves modelled on Laljibhai Devraj's metal type used to print this 1904 book).
'?' in `uncertain` = characters to double-check against the script chart.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.khojki import from_tokens, validate  # noqa: E402

ROWS = [
    # id, page_tokens, crop_tokens (None = same), gloss, uncertain
    ("p28_l01", "sa -m kha ya ma -aa : sa bha da -aa : va tha -aa :", None, "heading", "KHA, YA, BHA in words 1-2"),
    ("p28_l03", "4 _ ja -u ga :", None, "4 jug (ages)", ""),
    ("p28_l04", "4 _ va -e ddda :", None, "4 ved", "glyph = DDDA (Devraj form of /d/)"),
    ("p28_l05", "1 _ ka ra ma -aa :", None, "1 Krta(?)", "3rd glyph matches MA; word suggests TA"),
    ("p28_l06", "1 _ ra ga :", None, "1 Rag (Rig)", ""),
    ("p28_l07", "2 _ ta -e ma -aa :", None, "2 Treta", "2nd glyph matches MA; word suggests TA"),
    ("p28_l08", "2 _ ja -u ja ra :", None, "2 Jujar (Yajur)", ""),
    ("p28_l09", "3 _ da -u aa pa -u ra :", None, "3 Duapur", "first letter DA vs JA"),
    ("p28_l10", "2 _ sa -aa -m ma :", None, "3 Sam (printed numeral looks like 2)", "mark above SA: anusvara?"),
    ("p28_l11", "4 _ ka la ja -u ga :", None, "4 Kaljug", ""),
    ("p28_l12", "4 _ a tha ra :", None, "4 Atharv", "no VA visible"),
    ("p28_l21", "4 _ kha -aa -m nna -ii :", None, "4 khani", "NNA vs NA; anusvara"),
    ("p28_l22", "4 _ pa -u ra sa -aa ra tha :", None, "4 purusarth", ""),
    ("p28_l24", "1 _ da ra ma :", "da ra ma :", "1 dharam", ""),
    ("p28_l25", "2 _ ja -aa ra -e ja -aa :", None, "2 jareja", ""),
    ("p28_l26", "2 _ a ra tha :", "a ra tha :", "2 arath", ""),
    ("p28_l28", "3 _ ka -aa ma :", None, "3 kam", ""),
    ("p28_l29", "4 _ u ddda ma -o ja -aa :", None, "4 (u)dmoja", ""),
    ("p28_l30", "4 _ ma -o kha sa -shadda :", None, "4 mokhs", "SA vs KHA; shadda vs nukta"),
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "test_lines.tsv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "image", "page_text", "crop_text", "page_tokens", "gloss", "uncertain"])
        for rid, pt, ct, gloss, unc in ROWS:
            page_text, crop_text = from_tokens(pt), from_tokens(ct or pt)
            for t in (page_text, crop_text):
                assert not validate(t), (rid, validate(t))
            w.writerow([rid, f"data/lines/{rid}.png", page_text, crop_text, pt, gloss, unc])
    print(f"wrote {len(ROWS)} lines -> {out}")


if __name__ == "__main__":
    main()
