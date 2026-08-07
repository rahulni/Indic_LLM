# -*- coding: utf-8 -*-
"""Train, freeze, load and verify the tokenizer.

Session 2 defined the tokenizer contract: the same raw text must always produce
the same token ids. This stage has to make that contract *enforceable*, because
a shard is only meaningful relative to one exact tokenizer -- if the tokenizer
changes, the integers in every shard silently change meaning while every hash
still looks fine.

So the tokenizer is frozen to disk with its own digest, and
:func:`load_frozen` re-verifies that digest on every load. There is no code path
that uses a tokenizer without checking it.

One rule worth stating explicitly: the tokenizer is trained on the **train split
only**. A tokenizer fitted on evaluation text has already leaked -- it has spent
merge budget on the exact strings the benchmark will ask about -- and no
downstream firewall can undo that.
"""
from __future__ import annotations

import os
from collections import defaultdict

from ..config import SPECIAL_TOKENS, UNK_ID
from ..hashing import hash_obj, read_json, read_text, write_json, write_text
from .bpe import BPE, pretokenize

TOKENIZER_JSON = "tokenizer.json"
TOKENIZER_SHA = "tokenizer.sha256"


class TokenizerIntegrityError(RuntimeError):
    """Raised when a tokenizer's contents do not match its recorded digest."""


def _pretoken_freqs(texts: list[str]) -> dict[str, int]:
    freqs: dict[str, int] = defaultdict(int)
    for t in texts:
        for pt, _ in pretokenize(t, strip_joiners=True):
            freqs[pt] += 1
    return dict(freqs)


def train_and_freeze(train_texts: list[str], vocab_size: int,
                     out_dir: str, *, unit_mode: str = "byte",
                     reuse: bool = True) -> tuple[BPE, dict]:
    """Train on ``train_texts`` and write the frozen artifact + its digest.

    When a frozen tokenizer is already present, verifies and reuses it instead of
    retraining. Training is deterministic, so retraining produced the identical
    file -- but paying for 8,192 merges on every run is waste, and reuse is what
    "frozen" is supposed to mean. A mismatch in vocabulary size or unit mode
    retrains rather than silently returning the wrong tokenizer.
    """
    if reuse:
        try:
            tok, digest = load_frozen(out_dir)
        except (TokenizerIntegrityError, OSError):
            pass
        else:
            if tok.vocab_size == vocab_size and tok.unit_mode == unit_mode:
                return tok, {
                    "tokenizer_hash": digest,
                    "vocab_size": tok.vocab_size,
                    "n_base": len(tok.base_alphabet),
                    "n_merges": len(tok.merges),
                    "stopped_early": tok.vocab_size < vocab_size,
                    "unit_mode": unit_mode,
                    "distinct_pretokens": None,
                    "trained_on_documents": len(train_texts),
                    "reused_frozen_artifact": True,
                    "note": ("loaded and verified the committed frozen "
                             "tokenizer; training is deterministic, so this is "
                             "the artifact a retrain would have produced"),
                }
    freqs = _pretoken_freqs(train_texts)
    tok = BPE(unit_mode=unit_mode, specials=list(SPECIAL_TOKENS))
    stats = tok.train(freqs, target_vocab=vocab_size)

    payload = tok.to_dict()
    digest = hash_obj(payload)
    write_json(os.path.join(out_dir, TOKENIZER_JSON), payload, canonical=True)
    write_text(os.path.join(out_dir, TOKENIZER_SHA), digest)

    stats.update({
        "tokenizer_hash": digest,
        "distinct_pretokens": len(freqs),
        "unit_mode": unit_mode,
        "trained_on_documents": len(train_texts),
    })
    return tok, stats


def load_frozen(out_dir: str) -> tuple[BPE, str]:
    """Load and verify. Raises rather than returning a tokenizer we cannot trust."""
    jpath = os.path.join(out_dir, TOKENIZER_JSON)
    spath = os.path.join(out_dir, TOKENIZER_SHA)
    if not (os.path.exists(jpath) and os.path.exists(spath)):
        raise TokenizerIntegrityError(
            f"frozen tokenizer missing in {out_dir!r} "
            f"(need {TOKENIZER_JSON} and {TOKENIZER_SHA})"
        )
    payload = read_json(jpath)
    recorded = read_text(spath).strip()
    actual = hash_obj(payload)
    if actual != recorded:
        raise TokenizerIntegrityError(
            f"tokenizer hash mismatch: file hashes to {actual}, "
            f"but {TOKENIZER_SHA} records {recorded}. The shards were tokenised "
            f"with a different tokenizer than this one."
        )
    return BPE.from_dict(payload), actual


def encode(tok: BPE, text: str) -> list[int]:
    return tok.encode(text, unk_id=UNK_ID)


# ---------------------------------------------------------------------------
# Fertility
# ---------------------------------------------------------------------------

def fertility(tok: BPE, texts: list[str]) -> dict:
    """Tokens per word, plus the unknown rate.

    Fertility is the number that connects the tokenizer to the training budget.
    A language at 13 tokens per word costs ten times the compute of one at 1.3
    for the same content, fits a tenth as much into a context window, and gets
    chopped across sequence boundaries far more often. Session 5 targets ~2.0
    for Indic; this reports what we actually achieve at each vocabulary size.
    """
    n_words = n_word_tokens = n_tokens = n_unk = 0
    for t in texts:
        for pt, is_word in pretokenize(t, strip_joiners=True):
            syms = tok.encode_pretoken(pt)
            n_tokens += len(syms)
            n_unk += sum(1 for s in syms if s not in tok.tok2id)
            if is_word:
                n_words += 1
                n_word_tokens += len(syms)
    return {
        "words": n_words,
        "word_tokens": n_word_tokens,
        "total_tokens": n_tokens,
        "fertility": round(n_word_tokens / n_words, 4) if n_words else 0.0,
        "unk_tokens": n_unk,
        "unk_rate": round(n_unk / n_tokens, 6) if n_tokens else 0.0,
    }


def fertility_sweep(train_texts: list[str], by_language: dict[str, list[str]],
                    vocab_sizes: list[int]) -> dict:
    """Fertility per language across vocabulary sizes.

    Training a tokenizer is cheap even at sizes where *model* training is not,
    so the curve is affordable to measure and it is the evidence behind the
    claim that the demo's small vocabulary is a compute choice rather than a
    tokenizer failure.
    """
    freqs = _pretoken_freqs(train_texts)
    rows = []
    for v in sorted(vocab_sizes):
        tok = BPE(unit_mode="byte", specials=list(SPECIAL_TOKENS))
        try:
            tok.train(freqs, target_vocab=v)
        except ValueError:
            continue
        entry = {"vocab_size": tok.vocab_size, "requested": v, "by_language": {}}
        for lang, texts in sorted(by_language.items()):
            entry["by_language"][lang] = fertility(tok, texts)
        rows.append(entry)
    return {
        "note": (
            "Fertility is tokens per word. Lower is better. Session 5 targets ~2.0 "
            "for Indic languages; session 4 MEASURED cl100k at 13.268 tok/word on "
            "Telugu and MuRIL at 2.296 on the same corpus."
        ),
        "sweep": rows,
    }
