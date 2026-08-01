#!/usr/bin/env python3
"""
Inject the trained tokenizer + metrics into widget_template.html to produce a
self-contained widget.html (no external requests; the encode/decode demo is a JS
port of the tokenizer).

Run (after train.py):
    python build_widget.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    tokenizer_full = json.loads((ROOT / "tokenizer.json").read_text(encoding="utf-8"))
    tokens = json.loads((ROOT / "tokens" / "tokens_all.json").read_text(encoding="utf-8"))
    hf = None
    hf_path = ROOT / "hf_results.json"
    if hf_path.exists():
        hf = json.loads(hf_path.read_text(encoding="utf-8"))

    # the widget's JS encoder only needs these fields
    tokenizer = {
        "unit_mode": tokenizer_full["unit_mode"],
        "pretok_mode": tokenizer_full["pretok_mode"],
        "base_units": tokenizer_full["base_units"],
        "merges": tokenizer_full["merges"],
    }

    template = (ROOT / "widget_template.html").read_text(encoding="utf-8")
    html = (template
            .replace("__RESULTS__", json.dumps(results, ensure_ascii=False))
            .replace("__TOKENIZER__", json.dumps(tokenizer, ensure_ascii=False))
            .replace("__TOKENS__", json.dumps(tokens, ensure_ascii=False))
            .replace("__HF__", json.dumps(hf, ensure_ascii=False)))

    out = ROOT / "widget.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
