"""OPTIONAL vision-language-model baseline (NOT run for the reported results: the build
environment had no API access). Sends each test-line crop to Claude, normalises the answer
with the same encoding policy (src/khojki.normalize) and scores it with the same CER.

    pip install anthropic
    export ANTHROPIC_API_KEY=...          # never commit keys
    python scripts/vlm_baseline.py [--model claude-sonnet-5-5]

Writes results/vlm_predictions.tsv and prints CER. Check its output against the script
chart: VLMs often answer in Gujarati/Devanagari code points or ASCII ':'; normalise() maps
the known confusables and validate() flags the rest.
"""
import argparse
import base64
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.khojki import normalize, validate  # noqa: E402
from src.metrics import cer  # noqa: E402

PROMPT = (
    "This image is one line of printed Khojki script (Bombay, c. 1904). Transcribe it into "
    "Unicode Khojki (block U+11200-U+1124F). Use U+1123A KHOJKI WORD SEPARATOR for the colon-like "
    "mark, Gujarati digits (U+0AE6-U+0AEF) for numerals, and a normal space only where there is a "
    "visible gap. Output ONLY the transcription, nothing else."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    a = ap.parse_args()
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rows = list(csv.DictReader(open(ROOT / "data/test_lines.tsv", encoding="utf-8"), delimiter="\t"))
    E = N = 0
    with open(ROOT / "results/vlm_predictions.tsv", "w", encoding="utf-8") as fh:
        fh.write("id\tref\tpred_raw\tpred\tproblems\n")
        for r in rows:
            img = base64.b64encode((ROOT / r["image"]).read_bytes()).decode()
            msg = client.messages.create(model=a.model, max_tokens=200, messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
                {"type": "text", "text": PROMPT}]}])
            raw = msg.content[0].text.strip()
            pred = normalize(raw)
            e, n = cer(r["page_text"], pred)
            E, N = E + e, N + n
            fh.write(f"{r['id']}\t{r['page_text']}\t{raw}\t{pred}\t{validate(pred)}\n")
            print(r["id"], round(e / n, 3), validate(pred) or "")
    print(f"VLM CER: {E / N:.3f} ({E}/{N})")


if __name__ == "__main__":
    main()
