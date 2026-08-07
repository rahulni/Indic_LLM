# Corpus provenance

Everything under `corpus/` is vendored and committed so the demo runs with no
network access and no dependency on paths outside this directory. Regenerate with
`python tools/vendor_corpus.py`.

All files are written LF-normalised and NFC-normalised. Their bytes are the
identity of every shard downstream, so `.gitattributes` marks `corpus/**` as
binary to stop git rewriting line endings on checkout.

## Sources

| Lane | Origin | License | Notes |
|---|---|---|---|
| `web`, `multiling`, `indic`, `longctx` | Wikipedia plain-text extract via MediaWiki API (fetched by 2_token/fetch_data.py) | CC-BY-SA-4.0 | Plain-text extracts, split on blank lines into paragraph documents. |
| `code` | This repository's own Python files (sessions 2, 4, 5) | Course work, author-retained | Real source files, vendored whole so file boundaries are genuine. |
| `reasoning`, `agentic` | **Authored for this demo** (`tools/corpus_seed.py`) | Course work, author-retained | See the honesty note below. |

## Honesty note on the authored lanes

The `reasoning` and `agentic` documents were **written by hand for this
submission**. They are not harvested data and are not presented as such.

They exist because those two lanes are the only ones whose *structure* is
load-bearing: a reasoning trace needs a prompt to mask out of the loss, and an
agentic trace additionally needs tool observations, which the model must read but
must never be trained to predict. Each document carries explicit `role_spans`, and
`tdes/masks.py` keys its loss mask off exactly those roles.

## Indic tiers are a documented stand-in, not a verification claim

Session 5 distinguishes *verified* from *unverified* Indic supply (Sangraha
Verified means human-verified native content) and forbids the unverified tier
from substituting for the verified part of the protected floor.

**We have no human verification available for this corpus, and we do not claim
any.** The vendored Wikipedia text is also uniformly high-purity -- over 80% of
documents score exactly 1.0 on script purity -- so thresholding purity alone
would yield a 98/2 split: an arbitrary number dressed up as a measurement.

Instead `indic_tier` is assigned by splitting the corpus at the **median of a
composite quality score** that combines three genuinely computed signals:
script purity, length adequacy, and prose density (running prose versus
list-like extracts). Both `script_purity` and `quality_score` are stored on every
document, so the assignment can be re-derived and checked by a test.

This is a stand-in chosen so Session 5's tier *rule* can actually be exercised.
It is not a statement that any document here was verified by a human.

Threshold used: `0.938364` (57 verified / 56 unverified).

## Splits

Splits are carved deterministically per lane and are disjoint by construction --
a document is moved out of the train pool, never copied. Held-out documents carry
a distinctive canary string so a leak is detectable by content scan and not only
by content hash.

**Deduplication runs before the split, and that ordering matters.** An earlier
version split first. Because `en.txt` and `en_prose.txt` share 23 identical
paragraphs, a duplicate pair could be separated by the split -- one copy into
validation, its twin left in train -- and deduplicating inside the training pool
afterwards could never see it. That produced a real train/validation leak, which
the firewall's n-gram check caught at 100% overlap. The firewall is the right
safety net; collapsing duplicates across the whole pool first is the actual fix.

| Split | Documents | Canary | Permission |
|---|---|---|---|
| train | 234 | - | admitted to training |
| validation | 20 | `ZZQX-VALIDATION-CANARY-B48D6E05-READ-ONLY` | readable for evaluation, **never gradient-bearing** |
| eval | 20 | `ZZQX-EVAL-CANARY-7F3A9C21-DO-NOT-TRAIN` | **never read during training** |

## Train documents by lane

| Lane | Documents | Packing policy |
|---|---|---|
| `agentic` | 8 | `structure_preserving` |
| `code` | 59 | `best_fit` |
| `indic` | 87 | `greedy` |
| `longctx` | 28 | `long_context` |
| `multiling` | 22 | `concat_chop` |
| `reasoning` | 10 | `structure_preserving` |
| `web` | 20 | `concat_chop` |

## Document schema

```json
{"doc_id": "...", "lane": "indic", "text": "...", "chars": 512, "words": 78,
 "content_sha256": "...", "language": "tel", "script": "Telugu",
 "source": "...", "source_file": "...", "license": "CC-BY-SA-4.0",
 "provenance_tier": "public-encyclopaedic", "indic_tier": "verified",
 "script_purity": 0.9931, "never_train": false, "split": "train"}
```
