#!/usr/bin/env python3
"""
Evaluate the saved faithful-BPE tokenizer (V3, tokenizer.json):
  * recompute both fertility metrics (faithful + word) and the score,
  * PROVE faithfulness: decode(encode(text)) == text on every corpus AND on
    out-of-corpus probes (unseen scripts, emoji, a literal marker) -- the exact
    case an [UNK]-based tokenizer fails.

Run (from this folder):
    python evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from faithful_bpe import FaithfulBPE

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "mr"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}

PROBES = [
    "India's population is 1,428,627,663.",
    "El niño comió jalapeños en Málaga.",          # unseen Latin accents
    "東京 Tokyo 日本 — 서울 — 🇮🇳🐯",                  # unseen CJK/Hangul/emoji
    "भारत एक देश है।\nline\tbreaks",
    "literal ▁ marker U+2581 must survive",
]


def main() -> int:
    tok = FaithfulBPE.load(str(ROOT / "tokenizer.json"))
    print(f"loaded tokenizer: vocab={len(tok.id_to_sym)}  "
          f"pretok_mode={tok.pretok_mode}  unit_mode={tok.unit_mode}\n")

    # ---- fertility metrics ----
    rows = {}
    for c in LANGS:
        text = (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8")
        a = tok.analyze(text)
        rows[c] = {**a, "faithful_ratio": a["total_tokens"] / a["faithful_units"],
                   "word_ratio": a["total_tokens"] / a["words"]}
        print(f"  {NAMES[c]:8s} faithful={rows[c]['faithful_ratio']:.4f}  "
              f"word={rows[c]['word_ratio']:.4f}  tokens={a['total_tokens']}  units={a['faithful_units']}")

    frs = [rows[c]["faithful_ratio"] for c in LANGS]
    spread = max(frs) - min(frs)
    print(f"\n  faithful spread = {spread:.4f}   SCORE = {1000/spread:.1f}")

    # ---- faithfulness: corpus round-trip ----
    print("\n  round-trip on corpus:")
    all_ok = True
    for c in LANGS:
        text = (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8")
        ok = tok.decode(tok.encode(text)) == text
        all_ok &= ok
        print(f"    {NAMES[c]:8s} exact={ok}")

    # ---- faithfulness: out-of-corpus probes ----
    print("\n  round-trip on out-of-corpus probes (an [UNK]-based tokenizer drops these):")
    for s in PROBES:
        dec = tok.decode(tok.encode(s))
        ok = dec == s
        all_ok &= ok
        print(f"    exact={ok}  {s!r}")
        if not ok:
            print(f"       GOT: {dec!r}")

    print(f"\n  ALL ROUND-TRIPS EXACT: {all_ok}")
    if not all_ok:
        raise SystemExit("FAITHFULNESS FAILURE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
