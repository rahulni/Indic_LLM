# ERA V5 · Assignment 2 — Multilingual BPE tokenizer for India's Wikipedia page

**Languages:** English · Hindi · Telugu · **Marathi** (4th language of choice)
**Shared vocabulary:** 10,000 tokens · **Metric:** X = tokens / faithful-units, constraint X ≤ 1.2
**Score formula:** 1000 / (X_max − X_min)

**Submitted tokenizer:** [`hf_best_tokenizer.json`](hf_best_tokenizer.json) (V2, HuggingFace
format — BPE + Metaspace + NFKC + [UNK]) · **Self-score: 26,778.2** · all four X ≤ 1.2 ✓

**Widget:** open [`../index.html`](../index.html) in any browser (or the hosted URL below).
It shows the ratios, token statistics, calculations and self-score for **three** runnable
versions of the tokenizer, lets you browse and **download** every tokenizer and the full
token list, includes a live encode/decode demo, and walks the complete encode → decode
pipeline step by step for all four languages.

**Hosted widget URL:** _add Netlify URL here after deploying this folder_

---

## Results (submitted tokenizer, V2)

| Language | Page | Tokens | Faithful units | X | ≤ 1.2 |
|---|---|---:|---:|---:|:--:|
| English | India | 118,289 | 186,367 | 0.6347 | ✓ |
| Hindi | भारत | 55,160 | 88,359 | 0.6243 | ✓ |
| Telugu | భారతదేశం | 24,012 | 36,293 | 0.6616 | ✓ |
| Marathi | भारत | 18,867 | 29,766 | 0.6338 | ✓ |

```
sorted:  X_max = 0.6616 (Telugu)   X_min = 0.6243 (Hindi)
spread = 0.6616 − 0.6243 = 0.0373
score  = 1000 / 0.0373 = 26,778.2
```

A *faithful unit* is one contiguous letter/mark/number run OR one visible
punctuation/symbol character (regex `[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]`).

## The three versions in this package

| | Format | Score | Character-faithful | Stock HF tooling |
|---|---|---:|:--:|:--:|
| **V2 · submitted** — `hf_best_tokenizer.json` | HuggingFace (NFKC + [UNK]) | **26,778.2** | ✗ | ✓ |
| V1 — `hf_byte_fallback_tokenizer.json` | HuggingFace (byte-fallback) | 21,639.3 | ✓ | ✓ |
| V3 — `tokenizer.json` + `faithful_bpe.py` | custom faithful-bpe/1 | 35,095.6 | ✓ (any input) | ✗ (encoder included) |

V2 is submitted because it can be loaded and scored end-to-end with standard HuggingFace
tooling. V1 gives the same stock-tooling verification with an exact round-trip. V3 is the
from-scratch engine (akshara units, ▁ marker, 256 byte-fallback tokens) — highest score,
`decode(encode(x)) == x` for any input; since its JSON format is custom, its encoder code
ships alongside (`faithful_bpe.py`), plus a JavaScript port inside the widget.

## Verify every number yourself (from this folder)

```bash
pip install regex tokenizers

python evaluate_hf.py hf_best_tokenizer.json            # V2 (submitted) -> 26,778.2
python evaluate_hf.py hf_byte_fallback_tokenizer.json   # V1             -> 21,639.3
python evaluate.py                                      # V3             -> 35,095.6 + round-trip proof
python validate.py                                      # full audit: all three vs published numbers
```

## Corpus (exact extraction process)

`build_corpus.py` fetches each language's *India* page from the Wikipedia REST API
(`https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}`) and converts it to
**faithful Markdown** with `markdownify` — every visible character kept (links, URLs,
tables, references, navboxes, categories); only script/style/meta machinery stripped.
The exact snapshots used for all numbers are committed under `corpus/` with fetch
metadata (`*.meta.json`). Live pages drift; the snapshots are canonical.

Training details, design rationale and per-language worked examples are in the widget's
**Methodology** tab. Weight search: V1/V2 use en×3 hi×3 te×5 mr×7; V3 uses en×1 hi×1 te×2 mr×3.

## Files

```
../index.html                     the widget (self-contained)
hf_best_tokenizer.json            V2 — SUBMITTED tokenizer (HuggingFace format)
hf_byte_fallback_tokenizer.json   V1 — faithful HF alternative
tokenizer.json                    V3 — custom format tokenizer
faithful_bpe.py                   V3 encoder/decoder (required for the custom format)
evaluate_hf.py / evaluate.py      one-command evaluation (HF versions / custom version)
validate.py + expected_metrics.json   full package audit vs published numbers
build_corpus.py                   corpus extraction process
train.py / train_hf.py            training code (custom / HuggingFace)
corpus/                           committed corpus snapshots + metadata
results.json                      V3 metrics in machine-readable form
tokens_all.txt / tokens_all.json  full 10,000-token export
VALIDATION_REPORT.md              what was validated and how to re-run it
```
