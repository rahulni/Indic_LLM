# Final submission — this folder is the complete submission

Everything a grader needs is inside `final_submission/`; nothing outside it is required.

```
index.html            the widget (V1 / V2 / V3 / Methodology) — self-contained, ~2.6 MB
files/
  hf_best_tokenizer.json            V2 — SUBMITTED tokenizer (HuggingFace format) · score 26,778.2
  hf_byte_fallback_tokenizer.json   V1 — faithful HF alternative · 21,639.3
  tokenizer.json + faithful_bpe.py  V3 — custom format + its encoder code · 35,095.6
  evaluate_hf.py / evaluate.py      one-command evaluation (HF / custom)
  validate.py + expected_metrics.json   full package audit vs published numbers
  corpus/*.faithful.txt (+ meta)    the exact corpus snapshots behind every number
  build_corpus.py                   exact Wikipedia extraction process
  train.py / train_hf.py            training code
  results.json                      V3 metrics · tokens_all.txt/.json  full vocab export
  SUBMISSION.md                     writeup (add the hosted URL there after deploying)
  VALIDATION_REPORT.md              what was validated and how to re-run it
build_final.py / final_template.html  page generator (rebuild needs the parent repo; not needed for grading)
DEPLOY.md                           this file
```

## Deploy (assignment deliverable: the widget URL)

Drag-and-drop this **whole folder** at https://app.netlify.com/drop
(or `netlify deploy --dir . --prod` from this folder). `index.html` becomes the site
root; all `files/` links on the page work immediately. Paste the resulting URL into
`files/SUBMISSION.md`. Any static host (GitHub Pages, etc.) works the same way.

## Grader verification — all from `files/`

```bash
cd files
pip install regex tokenizers

python evaluate_hf.py hf_best_tokenizer.json            # V2 (submitted) -> 26,778.2
python evaluate_hf.py hf_byte_fallback_tokenizer.json   # V1             -> 21,639.3
python evaluate.py                                      # V3             -> 35,095.6 + round-trip proof
python validate.py                                      # audits all three vs published numbers
```

## Changing which version is presented as submitted

One line in `build_final.py`: `SUBMITTED_ID = "v2"` (→ `"v1"` / `"v3"`), then
`python build_final.py`. Rebuilding requires the parent repo (corpus + tokenizer
sources); update `files/SUBMISSION.md` to match.
