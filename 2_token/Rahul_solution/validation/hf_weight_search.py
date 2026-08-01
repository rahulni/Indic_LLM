#!/usr/bin/env python3
"""
Experiment: what score could an HF-format (HuggingFace `tokenizers`) submission get?

Context: Rahul's submitted tokenizer is a custom faithful-bpe format (score 35,095.55)
which the grader's stock HF tooling cannot load. Shipping an HF-format tokenizer.json
instead would make it loadable by Sravan_solution/evaluate_tokenizer.py directly --
but the existing hf_tokenizer.json (untuned weights 3/4/4/2) only scores 4,483.7.

This script searches per-language corpus weights using the SAME HF config as
train_hf.py / the reference (BPE + Metaspace(never) + NFKC, vocab 10000,
min_frequency=1) and reports the best achievable score. For the winning weights it
also trains a byte-fallback variant (no [UNK], no NFKC) to check whether an
HF-format tokenizer could keep the submission's faithfulness guarantee.

Writes hf_weight_search.json next to this file. Read-only w.r.t. all
pre-existing Rahul_solution files.

Run:  python hf_weight_search.py     (~2-5 min)
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Metaspace as MetaspaceDecoder, Sequence as DecoderSequence
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

HERE = Path(__file__).resolve().parent
RAHUL = HERE.parent
CORPUS = RAHUL / "corpus"
LANGS = ["en", "hi", "te", "mr"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

TEXTS = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}
UNITS = {c: len(FAITHFUL_UNIT_RE.findall(TEXTS[c])) for c in LANGS}

# candidates: baseline ratios (w=3,4,4,2) are en .616 hi .580 te .682 mr .803 --
# Marathi needs much more training weight, Hindi relatively less.
CANDIDATES = [
    {"en": 3, "hi": 4, "te": 4, "mr": 2},   # existing hf_results.json baseline
    {"en": 1, "hi": 1, "te": 1, "mr": 1},
    {"en": 1, "hi": 1, "te": 2, "mr": 2},
    {"en": 1, "hi": 1, "te": 2, "mr": 3},   # custom tokenizer's winning weights
    {"en": 1, "hi": 1, "te": 2, "mr": 4},
    {"en": 1, "hi": 1, "te": 3, "mr": 4},
    {"en": 1, "hi": 1, "te": 3, "mr": 5},
    {"en": 2, "hi": 1, "te": 3, "mr": 4},
    {"en": 2, "hi": 1, "te": 3, "mr": 5},
    {"en": 2, "hi": 1, "te": 4, "mr": 6},
    {"en": 2, "hi": 2, "te": 3, "mr": 5},
    {"en": 3, "hi": 2, "te": 4, "mr": 6},
    {"en": 1, "hi": 2, "te": 3, "mr": 4},
    {"en": 3, "hi": 3, "te": 5, "mr": 7},
]


def train_reference_style(weights) -> Tokenizer:
    """Identical config to train_hf.py (mirrors the reference solution)."""
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tok.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="never")
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for c in LANGS:
            p = Path(tmp) / f"{c}.txt"
            p.write_text(TEXTS[c], encoding="utf-8")
            files += [str(p)] * weights[c]
        tok.train(files, BpeTrainer(vocab_size=10000, min_frequency=1, special_tokens=["[UNK]"]))
    return tok


def train_byte_fallback(weights) -> Tokenizer:
    """Faithful-leaning HF variant: byte fallback instead of [UNK], no NFKC."""
    tok = Tokenizer(BPE(byte_fallback=True))
    tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tok.decoder = DecoderSequence([ByteFallback(), MetaspaceDecoder(replacement="▁", prepend_scheme="never")])
    byte_tokens = [f"<0x{i:02X}>" for i in range(256)]
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for c in LANGS:
            p = Path(tmp) / f"{c}.txt"
            p.write_text(TEXTS[c], encoding="utf-8")
            files += [str(p)] * weights[c]
        tok.train(files, BpeTrainer(vocab_size=10000, min_frequency=1, special_tokens=byte_tokens))
    return tok


def evaluate(tok: Tokenizer) -> dict:
    rows = {}
    for c in LANGS:
        n = len(tok.encode(TEXTS[c]).ids)
        rows[c] = {"tokens": n, "faithful_units": UNITS[c], "ratio": n / UNITS[c]}
    ratios = [rows[c]["ratio"] for c in LANGS]
    spread = max(ratios) - min(ratios)
    return {"rows": rows, "spread": spread, "score": 1000.0 / spread,
            "all_within_1_2": all(r <= 1.2 for r in ratios)}


def roundtrip_ok(tok: Tokenizer, samples: list[str]) -> dict:
    # skip_special_tokens=False so <0xNN> byte tokens reach the ByteFallback decoder
    out = {}
    for s in samples:
        dec = tok.decode(tok.encode(s).ids, skip_special_tokens=False)
        out[s[:36]] = (dec == s)
    return out


def main() -> int:
    print(f"corpus units: {UNITS}\n")
    print("Searching reference-style HF config (BPE + Metaspace + NFKC + [UNK]):")
    runs = []
    best = None
    for w in CANDIDATES:
        t0 = time.time()
        tok = train_reference_style(w)
        ev = evaluate(tok)
        runs.append({"weights": w, **{k: ev[k] for k in ("spread", "score", "all_within_1_2")},
                     "ratios": {c: ev["rows"][c]["ratio"] for c in LANGS}})
        print(f"  w={list(w.values())}  ratios=" +
              " ".join(f"{c}:{ev['rows'][c]['ratio']:.4f}" for c in LANGS) +
              f"  spread={ev['spread']:.4f}  score={ev['score']:8.1f}  ({time.time()-t0:.0f}s)")
        if best is None or ev["score"] > best[1]["score"]:
            best = (w, ev, tok)
    bw, bev, btok = best
    print(f"\nBEST reference-style: weights={bw}  score={bev['score']:.1f}")

    probes = [
        "India's population is 1,428,627,663.",
        "El niño comió jalapeños en Málaga.",
        "東京 Tokyo 日本 — 서울 — 🇮🇳🐯",
        "H₂O and E=mc² and ² superscript",
    ]
    print("\nRound-trip of best reference-style tokenizer (expected to FAIL on unseen chars/NFKC):")
    rt_ref = roundtrip_ok(btok, probes)
    for k, v in rt_ref.items():
        print(f"  exact={v}  {k!r}")

    print("\nByte-fallback HF variant at the best weights (no [UNK], no NFKC):")
    bf_result = None
    try:
        bftok = train_byte_fallback(bw)
        bfev = evaluate(bftok)
        print("  ratios=" + " ".join(f"{c}:{bfev['rows'][c]['ratio']:.4f}" for c in LANGS) +
              f"  spread={bfev['spread']:.4f}  score={bfev['score']:.1f}")
        rt_bf = roundtrip_ok(bftok, probes)
        for k, v in rt_bf.items():
            print(f"  exact={v}  {k!r}")
        bf_path = HERE / "hf_byte_fallback_tokenizer.json"
        bftok.save(str(bf_path))
        bf_result = {"weights": bw, "spread": bfev["spread"], "score": bfev["score"],
                     "ratios": {c: bfev["rows"][c]["ratio"] for c in LANGS},
                     "roundtrip": rt_bf, "saved_to": bf_path.name}
    except Exception as e:
        print(f"  byte-fallback variant failed: {e}")
        bf_result = {"error": str(e)}

    best_path = HERE / "hf_best_tokenizer.json"
    btok.save(str(best_path))

    out = {
        "config": "BPE + Metaspace(never) + NFKC + [UNK], vocab 10000, min_frequency 1 (mirrors train_hf.py / reference)",
        "runs": runs,
        "best": {"weights": bw, "spread": bev["spread"], "score": bev["score"],
                 "ratios": {c: bev["rows"][c]["ratio"] for c in LANGS},
                 "tokens": {c: bev["rows"][c]["tokens"] for c in LANGS},
                 "roundtrip": rt_ref, "saved_to": best_path.name},
        "byte_fallback_variant": bf_result,
        "custom_submission_score_for_comparison": 35095.55,
        "existing_hf_cross_check_score": 4483.71,
        "reference_score": 6502.56,
    }
    (HERE / "hf_weight_search.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"\nwrote {HERE / 'hf_weight_search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
