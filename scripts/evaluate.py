"""Evaluate on the hand-transcribed test lines and attribute errors to pipeline stages.

    python scripts/evaluate.py [--model models/recognizer.pkl] [--out results]

Per line we have:
  page_text : ground truth for the line as printed on the page
  crop_text : ground truth for what the auto-detected crop contains
  pred      : pipeline output for the crop
Stage attribution (edit operations):
  line detection    = edit(page_text, crop_text)   (characters lost/added by the crop)
  Unicode mapping   = errors in edit(crop_text, pred) on code points produced by a mapping
                      rule (spaces, piece re-assembly, aliases, AA-bar handling)
  recognition       = remaining errors in edit(crop_text, pred) (classifier label wrong)
(Stage errors sum to >= end-to-end errors; they are counted on separate alignments.)
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.khojki import ALPHABET, to_tokens, validate  # noqa: E402
from src.metrics import align, short  # noqa: E402
from src.recognize import Recognizer  # noqa: E402


def stage_errors(rec: Recognizer, gray, crop_text: str):
    """Split edit(crop_text, pred) into recognition vs Unicode-mapping errors using the
    provenance of each output code point: substitutions / insertions of code points that a
    mapping rule produced (spaces, re-assembled pieces, aliases, dropped / stray AA bars) and
    deleted SPACEs count as mapping errors; everything else is recognition."""
    labs, gp, body, _ = rec.labels(gray)
    pred, prov = rec.mapper.to_unicode_prov(labs, gp, body) if labs else ("", [])
    _, ops = align(crop_text, pred)
    j = 0
    rec_e = map_e = 0
    for op, r, h in ops:
        if op in ("=", "S", "I"):
            src = prov[j]
            j += 1
        else:
            src = "map" if r == " " else "rec"
        if op == "=":
            continue
        if src == "map":
            map_e += 1
        else:
            rec_e += 1
    return pred, rec_e, map_e, ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/recognizer.pkl"))
    ap.add_argument("--tsv", default=str(ROOT / "data/test_lines.tsv"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--no-prior", action="store_true", help="disable the digit layout prior")
    a = ap.parse_args()
    rec = Recognizer.load(a.model)
    if a.no_prior:
        rec.DIGIT_PRIOR = 0.0
    rows = list(csv.DictReader(open(a.tsv, encoding="utf-8"), delimiter="\t"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tot = Counter()
    confusions = Counter()
    per_line = []
    for r in rows:
        gray = cv2.imread(str(ROOT / r["image"]), cv2.IMREAD_GRAYSCALE)
        pred, rec_e, map_e, ops = stage_errors(rec, gray, r["crop_text"])
        seg_e = align(r["page_text"], r["crop_text"])[0]
        end_e = align(r["page_text"], pred)[0]
        tot["chars"] += len(r["page_text"])
        tot["errors"] += end_e
        tot["seg"] += seg_e
        tot["map"] += map_e
        tot["rec"] += rec_e
        for op, x, y in ops:
            if op != "=":
                confusions[(short(x), short(y))] += 1
        per_line.append({"id": r["id"], "ref": r["page_text"], "pred": pred, "ref_tokens": r["page_tokens"],
                         "pred_tokens": to_tokens(pred), "chars": len(r["page_text"]), "errors": end_e,
                         "cer": round(end_e / max(1, len(r["page_text"])), 3),
                         "seg_errors": seg_e, "map_errors": map_e, "rec_errors": rec_e,
                         "valid": not validate(pred)})
    cer_total = tot["errors"] / tot["chars"]
    # 95% bootstrap CI over lines (the test set is tiny; report the uncertainty)
    import random
    rnd = random.Random(0)
    boots = []
    for _ in range(2000):
        s = [rnd.choice(per_line) for _ in per_line]
        boots.append(sum(p["errors"] for p in s) / max(1, sum(p["chars"] for p in s)))
    boots.sort()
    ci = [round(boots[50], 3), round(boots[1949], 3)]
    summary = {"lines": len(rows), "chars": tot["chars"], "errors": tot["errors"], "CER": round(cer_total, 4),
               "CER_95CI_bootstrap": ci,
               "errors_by_stage": {"line_detection": tot["seg"], "recognition": tot["rec"],
                                   "unicode_mapping": tot["map"]},
               "outputs_valid": sum(p["valid"] for p in per_line),
               "top_confusions": [{"ref": k[0], "hyp": k[1], "count": v} for k, v in confusions.most_common(15)]}
    (out / "eval.json").write_text(json.dumps({"summary": summary, "lines": per_line}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    with open(out / "predictions.tsv", "w", encoding="utf-8") as fh:
        fh.write("id\tCER\tref\tpred\tref_tokens\tpred_tokens\n")
        for p in per_line:
            fh.write(f"{p['id']}\t{p['cer']}\t{p['ref']}\t{p['pred']}\t{p['ref_tokens']}\t{p['pred_tokens']}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    # sanity: every reference uses only the policy inventory
    assert all(set(r["page_text"]) <= set(ALPHABET) for r in rows)


if __name__ == "__main__":
    main()
