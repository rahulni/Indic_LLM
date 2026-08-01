"""Generate site/data.js from results.json + the duplicate-structure analysis.

Kept as a script rather than a README one-liner so the widget's data file has
one obvious, repeatable source. Not a pipeline stage: it moves no documents and
is excluded from the cleaning-script hash.

  python build_site_data.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import ASSIGNMENT_DIR

RESULTS = os.path.join(ASSIGNMENT_DIR, "results.json")
DEDUP_ANALYSIS = os.path.join(ASSIGNMENT_DIR, "analysis_duplicate_structure.json")
FERTILITY_ANALYSIS = os.path.join(ASSIGNMENT_DIR, "analysis_tokenizer_fertility.json")
OUT = os.path.join(ASSIGNMENT_DIR, "site", "data.js")


def _merge(data: dict, path: str, key: str, label: str) -> None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data[key] = json.load(f)
    else:
        print(f"[build] warning: no {os.path.basename(path)} - the {label} panel will be omitted")


def main() -> None:
    with open(RESULTS, encoding="utf-8") as f:
        data = json.load(f)

    _merge(data, DEDUP_ANALYSIS, "duplicate_structure", "threshold-sweep")
    _merge(data, FERTILITY_ANALYSIS, "tokenizer_fertility", "tokenizer-fertility")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("const RESULTS = ")
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write(";\n")

    t = data.get("totals", {})
    print(f"[build] wrote {OUT}")
    print(
        f"[build] {t.get('corpora_count')} corpora, "
        f"{t.get('final_docs', 0):,} final docs, "
        f"{t.get('final_tokens_cl100k', 0):,} final tokens, "
        f"all_deterministic={t.get('all_deterministic')}"
    )
    for cid in data.get("corpus_order", []):
        c = data["corpora"][cid]
        m = c["final_manifest"]
        print(
            f"         {cid}: {c['raw_input']['doc_count']:,} -> {m['final_doc_count']:,} docs, "
            f"{m['final_token_count_cl100k']:,} tokens, "
            f"{m.get('shards_blocked', 0)}/{m.get('shards_total', 1)} shards blocked"
        )


if __name__ == "__main__":
    main()
