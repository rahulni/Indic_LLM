# Validation report — this submission package

Every number shown in `../index.html` (and in `SUBMISSION.md`) is recomputed, not
copied: the widget is built by `../build_final.py`, which re-encodes the committed
corpus with the actual tokenizer files and injects the results. This package then
ships its own auditor so anyone can re-verify from this folder alone.

## Re-run the audit

```bash
pip install regex tokenizers
python validate.py        # ~2-3 minutes; exit code 0 = all checks passed
```

## What `validate.py` checks

For each of the three tokenizers (V1 `hf_byte_fallback_tokenizer.json`,
V2 `hf_best_tokenizer.json` — submitted, V3 `tokenizer.json` via `faithful_bpe.py`):

- **Token counts** per language — re-encoded from `corpus/*.faithful.txt`, compared
  exactly against the published numbers (`expected_metrics.json`, the same data
  embedded in the widget).
- **Faithful-unit counts** — recomputed with the metric regex
  `[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]`.
- **Fertility ratios, spread, score** — recomputed and compared (X ≤ 1.2 enforced
  for all four languages).
- **Vocabulary size** — exactly 10,000 for every version.
- **Round-trip behaviour** — `decode(encode(x))` vs `x` on the full corpus of all
  four languages and on probe strings (accents, CJK/Hangul, emoji, superscripts,
  a literal ▁). Results must match the behaviour published in the widget: V1 and V3
  round-trip the corpus exactly; V3 round-trips *any* probe exactly; V2 (NFKC +
  [UNK]) is expectedly lossy on unseen characters — the widget displays this openly.
- **Cross-file consistency** — `results.json` agrees with `expected_metrics.json`.

Last run: ALL CHECKS PASSED (see console output of `python validate.py`).

## Independent implementation notes

- The auditor computes faithful units with its own copy of the metric regex and
  counts tokens through the public encoder APIs — it does not reuse the widget's
  build pipeline.
- The live encode/decode demo in the widget is a JavaScript port of the V3 encoder,
  verified to produce identical token IDs to the Python implementation.
- The corpus snapshots under `corpus/` are the exact texts behind every number;
  `build_corpus.py` documents (and can re-run) the extraction process.
