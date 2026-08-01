#!/usr/bin/env python3
"""
Train the shared 10,000-token faithful BPE tokenizer for India's Wikipedia page in
English, Hindi, Telugu and Marathi, and search per-language training weights to
minimise the faithful fertility spread (score = 1000 / (X_max - X_min)).

The PRIMARY tokenizer uses whitespace-mode pre-tokenization, which keeps URLs/markup
whole and compresses as efficiently as the reference (English ~115k tokens vs the
reference's 111k) -- an honest, high-quality tokenizer.

We ALSO record a "metric-gaming" demonstration: class-mode pre-tokenization clusters
the four ratios (because the corpora share ~50% identical markup) and pushes the score
far higher WHILE PRODUCING MORE TOKENS (worse compression). That shows the score
formula rewards ratio-clustering, not tokenizer quality -- a shortcoming of the metric.

Outputs (in this folder):
    tokenizer.json         primary (whitespace) faithful-BPE tokenizer
    results.json           metrics + calculations for the widget
    tokens/tokens_all.txt  full vocab, one token per line
    tokens/tokens_all.json vocab with ids + kind

Run:  python train.py            (focused search, ~12 min)
      python train.py --fast     (single baseline run, quick)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from faithful_bpe import FaithfulBPE

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TOKENS_DIR = ROOT / "tokens"

LANGS = ["en", "hi", "te", "mr"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}
LOCAL = {"en": "English", "hi": "हिन्दी", "te": "తెలుగు", "mr": "मराठी"}
TITLE = {"en": "India", "hi": "भारत", "te": "భారతదేశం", "mr": "भारत"}
VOCAB = 10000

TEXTS = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}

# curated whitespace-mode weight candidates (en, hi, te, mr): pull the higher-ratio
# languages (Telugu, Marathi) down by giving their corpora more repetitions.
WS_CANDIDATES = [
    {"en": 1, "hi": 1, "te": 1, "mr": 1},
    {"en": 1, "hi": 1, "te": 2, "mr": 2},
    {"en": 1, "hi": 1, "te": 2, "mr": 3},
    {"en": 1, "hi": 1, "te": 3, "mr": 3},
    {"en": 1, "hi": 1, "te": 3, "mr": 4},
    {"en": 2, "hi": 1, "te": 3, "mr": 4},
    {"en": 1, "hi": 1, "te": 4, "mr": 5},
]


def train_tokenizer(weights, pretok_mode, unit_mode="akshara", min_pair_freq=2):
    tok = FaithfulBPE(unit_mode=unit_mode, pretok_mode=pretok_mode)
    tok.train([(TEXTS[c], weights[c]) for c in LANGS], target_vocab=VOCAB, min_pair_freq=min_pair_freq)
    return tok


def measure(tok):
    rows = {}
    for c in LANGS:
        a = tok.analyze(TEXTS[c])
        rows[c] = {
            **a,
            "faithful_ratio": a["total_tokens"] / a["faithful_units"],
            "word_ratio": a["total_tokens"] / a["words"],
        }
    frs = [rows[c]["faithful_ratio"] for c in LANGS]
    spread = max(frs) - min(frs)
    return rows, spread, 1000.0 / spread


def search_whitespace(candidates):
    best = None
    for w in candidates:
        t0 = time.time()
        tok = train_tokenizer(w, "whitespace")
        rows, spread, score = measure(tok)
        print(f"  ws w={list(w.values())} spread={spread:.4f} score={score:.1f}  ({time.time()-t0:.0f}s)")
        if best is None or score > best[2]:
            best = (w, tok, score, rows, spread)
    return best


def build_results(tok, rows, weights, spread, score, pretok_mode, gaming):
    ranked = sorted(LANGS, key=lambda c: rows[c]["faithful_ratio"], reverse=True)
    x_labels = {c: f"X{i+1}" for i, c in enumerate(LANGS)}
    word_ratios = {c: rows[c]["word_ratio"] for c in LANGS}
    word_spread = max(word_ratios.values()) - min(word_ratios.values())
    return {
        "meta": {
            "assignment": "ERA V5 - shared 10k BPE for India Wikipedia (en/hi/te/mr)",
            "vocab_budget": VOCAB,
            "engine": "from-scratch faithful BPE (256 byte-fallback tokens; decode(encode(x))==x)",
            "unit_mode": tok.unit_mode,
            "pretok_mode": pretok_mode,
            "weights": weights,
            "headline_metric": "faithful_ratio = total_tokens / faithful_units",
            "faithful_unit_policy": "one contiguous letter/mark/number run OR one visible punctuation/symbol char",
            "word_metric": "word_ratio = total_tokens / words (words = letter/mark/number runs)",
            "score_formula": "1000 / (X_max - X_min) over the four faithful_ratios",
        },
        "vocab": {
            "total": len(tok.id_to_sym),
            "byte_tokens": 256,
            "base_units": len(tok.base_units),
            "merges": len(tok.merges),
        },
        "languages": [
            {
                "code": c, "x_label": x_labels[c], "name": NAMES[c],
                "local_name": LOCAL[c], "page_title": TITLE[c],
                "chars": len(TEXTS[c]), "words": rows[c]["words"],
                "faithful_units": rows[c]["faithful_units"],
                "total_tokens": rows[c]["total_tokens"],
                "faithful_ratio": rows[c]["faithful_ratio"],
                "word_ratio": rows[c]["word_ratio"],
            } for c in LANGS
        ],
        "sorted_by_faithful_ratio": [
            {"code": c, "name": NAMES[c], "faithful_ratio": rows[c]["faithful_ratio"]} for c in ranked
        ],
        "faithful": {
            "x_max": max(rows[c]["faithful_ratio"] for c in LANGS),
            "x_min": min(rows[c]["faithful_ratio"] for c in LANGS),
            "spread": spread, "score": score,
            "all_within_1_2": all(rows[c]["faithful_ratio"] <= 1.2 for c in LANGS),
        },
        "word": {
            "x_max": max(word_ratios.values()), "x_min": min(word_ratios.values()),
            "spread": word_spread, "score": 1000.0 / word_spread,
            "all_within_1_2": all(v <= 1.2 for v in word_ratios.values()),
        },
        "metric_gaming": gaming,
        "reference_score": 6502.56,
    }


def gaming_demo():
    """Class-mode pre-tokenization: higher score, but MORE tokens (worse compression)."""
    w = {"en": 1, "hi": 1, "te": 2, "mr": 2}
    tok = train_tokenizer(w, "class")
    rows, spread, score = measure(tok)
    return {
        "note": "Same corpus, class-mode pre-tokenization + tuned weights. The score is far higher "
                "yet total tokens are HIGHER than the primary tokenizer -- so the metric rewards "
                "ratio-clustering (shared markup), not compression quality.",
        "pretok_mode": "class", "weights": w,
        "spread": spread, "score": score,
        "total_tokens": {c: rows[c]["total_tokens"] for c in LANGS},
        "faithful_ratio": {c: rows[c]["faithful_ratio"] for c in LANGS},
    }


def export_tokens(tok):
    TOKENS_DIR.mkdir(exist_ok=True)
    toks = tok.token_list()
    (TOKENS_DIR / "tokens_all.json").write_text(json.dumps(toks, ensure_ascii=False), encoding="utf-8")

    def vis(s):  # keep one token per line: escape control chars in the .txt view
        return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")

    (TOKENS_DIR / "tokens_all.txt").write_text(
        "\n".join(f"{t['id']}\t{t['kind']}\t{vis(t['token'])}" for t in toks) + "\n", encoding="utf-8")


def main() -> int:
    fast = "--fast" in sys.argv
    t0 = time.time()
    print("Searching whitespace-mode weights (primary tokenizer)...")
    cands = WS_CANDIDATES[:2] if fast else WS_CANDIDATES
    weights, tok, score, rows, spread = search_whitespace(cands)

    print("Recording class-mode metric-gaming demonstration...")
    gaming = gaming_demo()

    results = build_results(tok, rows, weights, spread, score, "whitespace", gaming)
    tok.save(str(ROOT / "tokenizer.json"))
    (ROOT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    export_tokens(tok)

    print(f"\nPRIMARY (whitespace)  weights={weights}")
    for c in LANGS:
        r = rows[c]
        print(f"  {NAMES[c]:8s} faithful={r['faithful_ratio']:.4f}  word={r['word_ratio']:.4f}"
              f"  tokens={r['total_tokens']}  units={r['faithful_units']}")
    print(f"  faithful spread={spread:.4f}  SCORE={score:.1f}   (reference = 6502.56)")
    print(f"  metric-gaming (class mode) score={gaming['score']:.1f} "
          f"with MORE tokens (en={gaming['total_tokens']['en']} vs {rows['en']['total_tokens']})")
    print(f"  done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
