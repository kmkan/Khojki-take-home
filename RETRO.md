# Retro

Tested on 19 printed lines from page 28 (1904 book), 146 characters. CER 45.9%. 

Error breakdown by stage: line detection 4, character recognition 53, Unicode mapping 11. Most recognition errors are lookalike letters (e.g. MA read as TA) and digits read as letters (૪ looks like PHA in this type).

Experiments: first run was 69.9% CER. Normalising glyph size by estimated body height made it much worse (112%). Splitting the classifier into base letter + vowel marks got it to 56.8%. More synthetic data plus a rule that standalone clusters are probably numbers got it to 45.9% (51.4% without that rule). kNN and a plain MLP scored about 49%. Using less data or no image degradation made no real difference, which is within noise on this small test set.

Encoding: Khojki block for letters and vowel signs (U+1122C–U+11233, stored after the consonant). Word separator U+1123A rather than a colon. Digits are Gujarati (U+0AE6–U+0AEF) because Unicode has no Khojki digits. Output is checked against an allowed-character list, marks must follow a letter, and text must be NFC.

Bottleneck: real labelled lines. Roughly 1,000–2,000 lines from the printed books to train a proper line-level model (CNN-LSTM/CTC) instead of classifying glyph by glyph. Handwriting would need several thousand lines per scribe.

Scale: about 0.33 s per page on a laptop CPU, so about 1 hour per 10,000 pages. Labelling is the real cost: checking 3 lines took 260 s (about 87 s per line), so checking 1,500 lines would take about 36 hours. Handwritten pages are much worse: model confidence drops from 0.76 on print to 0.56, and the line detection breaks on dense, slanted pages.

AI tools: the AI assistant said the pages contained "Khojki digits". Listing the Khojki Unicode block showed there are no digit characters, and the Unicode proposal for Khojki (N3978) says Gujarati digits are used.