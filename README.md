# Khojki OCR
CER on 19 test lines from page 28 (1904 book): 45.9%. Details in `results/eval.json` and `RETRO.md`.

## How it works

1. Find lines (`src/segment.py`): binarise, remove ruled lines, group ink into lines.
2. Recognise (`src/recognize.py`): cut each line into glyphs and classify them with a small neural net trained on fake scans made from the Noto Khojki font (`src/synth.py`).
3. Convert to Unicode (`src/mapping.py`): join glyphs that print in pieces, add spaces, check the output is valid.

## Setup

```
pip install -r requirements.txt
python scripts/extract_pages.py <sample pages PDF>
```

## Running

```
python run.py --all            # all pages, output in results/output/
python scripts/evaluate.py     # CER and error breakdown
python scripts/train.py        # retrain the model (~6 min)
```

## Test lines

The 19 test lines and their Unicode transcriptions are in `data/test_lines.tsv`, and the line images are in `data/lines/`. Transcriptions were checked by hand against the script chart. `results/render_check/` shows each line with its transcription rendered underneath.

## Limitations

- Small test set (146 characters), so the CER is rough.
- Some digits and letters look almost the same in this type (૪ and PHA).
- Line detection struggles with faint table lines and blank textured pages.
- Not usable on handwriting yet.

## Use of AI

I used Claude heavily on this. It wrote most of the code (segmentation, synthetic data, classifier, Unicode mapping, evaluation). I decided on the entire structure of the pipeline, and ran/tested/verified the pipeline on my machine. The retro has an example of where the AI got Khojki wrong.