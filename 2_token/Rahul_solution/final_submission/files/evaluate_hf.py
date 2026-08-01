#!/usr/bin/env python3
"""
Instructor helper: evaluate ANY HuggingFace-format tokenizer.json on the committed
India corpus (en/hi/te/mr) with the reference metric.

This is the reference solution's evaluate_tokenizer.py adapted only in two places:
the 4th language is Marathi (mr, not mai) and the tokenizer path is an argument.

Usage:
    pip install tokenizers regex
    python evaluate_hf.py hf_byte_fallback_tokenizer.json
    python evaluate_hf.py hf_best_tokenizer.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "mr"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "hf_byte_fallback_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(ROOT / path))
    rows = {}
    for code in LANGS:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        units = len(FAITHFUL_UNIT_RE.findall(text))
        tokens = len(tokenizer.encode(text).ids)
        rows[code] = {"tokens": tokens, "faithful_units": units, "ratio": tokens / units}
    ratios = [r["ratio"] for r in rows.values()]
    spread = max(ratios) - min(ratios)
    print(json.dumps({"tokenizer": path, "rows": rows, "spread": spread,
                      "score": 1000 / spread,
                      "all_within_1_2": all(r <= 1.2 for r in ratios)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
