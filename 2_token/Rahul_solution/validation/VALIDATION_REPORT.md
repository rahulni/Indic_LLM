# Validation report — Rahul_solution widget numbers & submission checklist

*Validated 2026-07-16 with `validation/validate.py` (60/60 checks passed; machine-readable
results in `validation/validation_results.json`). Nothing outside `validation/` was modified.*

## 1. Are the widget numbers correct? — YES

`widget_template.html` contains **no numbers** (only `__RESULTS__` / `__TOKENIZER__` /
`__TOKENS__` / `__HF__` placeholders); the numbers live in the built `widget.html`.
Verified end-to-end:

- The JSON blobs embedded in `widget.html` are **byte-identical** to `results.json`,
  `tokenizer.json`, `tokens/tokens_all.json`, and `hf_results.json`.
- Re-encoding all four committed corpora with the shipped `tokenizer.json`
  (via the shipped encoder `faithful_bpe.py`) reproduces **every reported number exactly**:

  | Lang | Tokens | Faithful units | X (faithful) | Word ratio |
  |---|---:|---:|---:|---:|
  | en | 123,020 ✓ | 186,367 ✓ | 0.660095 ✓ | 1.375801 ✓ |
  | hi | 56,474 ✓ | 88,359 ✓ | 0.639143 ✓ | 1.339802 ✓ |
  | te | 24,037 ✓ | 36,293 ✓ | 0.662304 ✓ | 1.492147 ✓ |
  | mr | 18,866 ✓ | 29,766 ✓ | 0.633810 ✓ | 1.315254 ✓ |

  spread = 0.0284936379 ✓ · **score = 35,095.55 ✓** · all faithful ratios ≤ 1.2 ✓
- Faithful-unit counts were recomputed with the **reference evaluator's own regex**
  (copied from `Sravan_solution/evaluate_tokenizer.py`), not Rahul's code — same numbers.
- Faithfulness holds: `decode(encode(x)) == x` exactly on all four corpora and on
  out-of-corpus probes (accents, CJK/Hangul, emoji, literal `▁`, tabs/newlines).
- Vocab bookkeeping: 10,000 = 256 byte + 2,178 base + 7,566 merges; token export has
  exactly 10,000 contiguous ids.

### About validating with the Sravan_solution tokenizer

The two tokenizers are **not the same**, so Sravan's tokenizer cannot reproduce Rahul's
numbers directly:

| | Sravan_solution | Rahul_solution |
|---|---|---|
| Format | HuggingFace `tokenizers` JSON | custom `faithful-bpe/1` JSON |
| 4th language | **Maithili** (mai) | **Marathi** (mr) |
| Weights | en3 hi4 te4 mai2 | en1 hi1 te2 mr3 |

Instead the validation uses Sravan's solution two ways:

1. **Methodology proof:** running Sravan's `tokenizer.json` on Sravan's corpus with this
   validator reproduces Sravan's `metrics.json` exactly (all token counts, units, and the
   6502.56 score) — so the metric used to check Rahul's numbers is provably identical to
   the reference evaluator's.
2. **Context run:** Sravan's tokenizer on Rahul's corpus gives en 0.5977 / hi 0.5793 /
   te 0.6731 (same regime as Rahul's; en/hi corpora snapshots are effectively identical —
   identical unit counts 186,367 / 88,359) and mr 1.0857 (high, as expected: Sravan never
   trained on Marathi). Additionally, Rahul's independent HuggingFace cross-check
   (`hf_tokenizer.json`) reproduces `hf_results.json` exactly.

## 2. Checklist coverage ("Notes for Students")

| Requirement | Covered? | Where |
|---|---|---|
| Exact tokenizer file | ✓ | `tokenizer.json` (+ download button in widget) |
| Code / clear method to build it | ✓ | `faithful_bpe.py`, `train.py`, `train_log.txt` |
| Exact Wikipedia corpus extraction process | ✓ | `build_corpus.py` + committed `corpus/` snapshots (`*.raw.html`, `*.faithful.txt`, `*.meta.json` with source URL & timestamp) |
| Token counts for all four languages | ✓ | `results.json`, widget table, `SUBMISSION.md` |
| Fertility ratios | ✓ | faithful + word ratios in all three places |
| Raw score calculation | ✓ | widget "calc" panel + `SUBMISSION.md` (spread → 1000/spread) |
| Live widget letting grader inspect/download tokenizer | ✓ (local) | `widget.html`: token browser, search, downloads (.txt/.json/tokenizer), **live JS encode/decode demo** |
| Encoder code for the custom JSON format | ✓ | `faithful_bpe.py` (Python) **and** the JS port inside the widget — both verified to produce identical IDs |

## 3. Original-assignment deliverables & gaps

| Deliverable | Status |
|---|---|
| Widget with ratios, statistics, calculations, self-score | ✓ `widget.html` |
| See & download the tokenizer (list of all tokens) | ✓ browse + 3 download buttons |
| **URL for the widget on Netlify (or other host)** | ✗ **NOT DONE YET** — `widget.html` is self-contained (no external requests), so it can be dropped on Netlify as-is; the URL must then be added to `SUBMISSION.md` |
| "We'll run your tokenizer ourselves" | ✓ reproducible: committed corpus + encoder code + this validation |

**Caveat worth knowing (already disclosed in `SUBMISSION.md` / `results.json`):** under the
literal assignment wording "tokens / words", the ratios are 1.32–1.49, i.e. **above 1.2**
for all four languages; the ≤ 1.2 claim holds under the **faithful-units** denominator,
which is the denominator the reference solution (`Sravan_solution`) itself grades with
(its ratios 0.58–0.73 are likewise faithful-unit based, and its own word ratios would also
exceed 1.2). The submission reports both metrics transparently.

## 4. How to re-run this validation

```bash
cd Rahul_solution/validation
python validate.py        # needs: pip install regex tokenizers  (~2 min)
```
