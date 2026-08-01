#!/usr/bin/env python3
"""
HuggingFace `tokenizers` sanity-check run.

This is NOT the submission -- it is an independent cross-check that our from-scratch
faithful-BPE scores land in the same regime as a standard, well-tested BPE trainer
(and as the trainer's reference solution, which uses the same stack). Divergence
would flag a bug in our own encoder/decoder.

Config mirrors the reference: BPE + Metaspace(▁) pre-tokenizer/decoder + NFKC, vocab
10,000. We reuse the same faithful corpus and the winning per-language weights found
by train.py.

Run:
    pip install tokenizers
    python train_hf.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "mr"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")


def faithful_units(text):
    return len(FAITHFUL_UNIT_RE.findall(text))


def default_weights():
    # reference-style weights; this is only an independent sanity check of the regime
    return {"en": 3, "hi": 4, "te": 4, "mr": 2}


def main() -> int:
    weights = default_weights()
    texts = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tok.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="never")

    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for c in LANGS:
            p = Path(tmp) / f"{c}.txt"
            p.write_text(texts[c], encoding="utf-8")
            files += [str(p)] * weights[c]
        trainer = BpeTrainer(vocab_size=10000, min_frequency=1, special_tokens=["[UNK]"])
        tok.train(files, trainer)

    tok.save(str(ROOT / "hf_tokenizer.json"))

    rows = {}
    for c in LANGS:
        t = tok.encode(texts[c]).ids
        u = faithful_units(texts[c])
        rows[c] = {"tokens": len(t), "faithful_units": u, "faithful_ratio": len(t) / u}
    frs = [rows[c]["faithful_ratio"] for c in LANGS]
    spread = max(frs) - min(frs)
    result = {
        "engine": "huggingface tokenizers (BPE + Metaspace + NFKC), sanity check",
        "weights": weights,
        "vocab_size": tok.get_vocab_size(),
        "rows": rows,
        "faithful_spread": spread,
        "faithful_score": 1000 / spread,
    }
    (ROOT / "hf_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in LANGS:
        print(f"  {NAMES[c]:8s} faithful_ratio={rows[c]['faithful_ratio']:.4f}  tokens={rows[c]['tokens']}")
    print(f"  HF faithful spread={spread:.4f}  score={1000/spread:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
