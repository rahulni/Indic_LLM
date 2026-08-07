# -*- coding: utf-8 -*-
"""Indic-aware byte-pair encoding.

Vendored and adapted from ``2_token/bpe.py`` upstream in this project, so this
submission is self-contained. The parts that matter for Indic are kept intact:

  * **NFC normalisation** -- the same syllable written precomposed or as
    base + combining mark must produce the same token ids.
  * **Akshara segmentation** -- base units are extended grapheme clusters
    (``regex`` ``\\X``), so a Devanagari or Telugu conjunct such as ``क्ष`` is one
    base symbol rather than three codepoints. Without this, Indic words start
    two to three times further from a single token than Latin words do, and the
    fertility gap widens for a reason that has nothing to do with the language.
  * **Joiner stripping** -- ZWJ/ZWNJ are invisible but create spurious spelling
    variants, which fragments merge statistics.
  * **Mark-preserving pretokenisation** -- a "word" is a maximal run of
    Letter/Mark characters, so vowel signs stay attached to their base letter.

What is added here on top of session 2: integer ids with reserved special
tokens, full ``encode``/``decode``, and JSON serialisation, because a training
system needs the tokenizer to be a frozen, hashable artifact rather than a
live object.
"""
from __future__ import annotations

import heapq
import unicodedata
from collections import defaultdict

import regex as _re  # for \X grapheme-cluster (akshara) segmentation

# End-of-word marker. A single codepoint above U+00FF so it can never collide
# with a byte symbol in byte mode, where every base unit is chr(0)..chr(255).
END = "Ā"

# Zero-width non-joiner / joiner.
_ZW = {"‌", "‍"}


def to_units(pt: str, mode: str = "byte") -> list[str]:
    """Segment a pretoken into base symbols.

    Three modes, and the choice has real consequences for Indic:

    ``byte``
        Each UTF-8 byte becomes one symbol, so the base alphabet is exactly 256
        no matter what the corpus contains. This is what cl100k, GPT-2 and
        Llama do. It is the default here for two reasons: the vocabulary size
        becomes an exact, freely chosen number rather than a corpus artifact,
        and no input can ever be unrepresentable.

        The cost lands on Indic. A Telugu character is three UTF-8 bytes, so at
        a small vocabulary -- before merges have had budget to rebuild
        syllables -- Telugu starts three times further from a single token than
        ASCII does. That is not a bug in this implementation; it is precisely
        the effect session 4 measured when cl100k scored 13.268 tokens/word on
        Telugu against 1.563 on English. The fertility sweep exists to show the
        merges buying that distance back as the budget grows.

    ``akshara``
        Base units are extended grapheme clusters, so a conjunct such as
        ``క్ష`` is one symbol from the start. Much better for Indic, but the
        base alphabet is then corpus-determined -- measured at **2,118** on this
        corpus -- so no vocabulary below that size is expressible at all.

    ``char``
        Unicode codepoints. Measured at 320 base symbols here. A middle ground,
        but still corpus-dependent and it can meet input it cannot represent.
    """
    if mode == "akshara":
        return _re.findall(r"\X", pt)
    if mode == "char":
        return list(pt)
    if mode == "byte":
        return [chr(b) for b in pt.encode("utf-8")]
    raise ValueError(f"unknown unit_mode {mode!r}")


def byte_symbol_to_text(sym: str) -> str:
    """Turn a byte-mode symbol back into text.

    Merged symbols are strings of chr(0)..chr(255), so the bytes come back with
    ``latin-1``. A merge can land mid-codepoint, which is normal in byte-level
    BPE, so decoding is tolerant rather than strict.
    """
    sym = sym.replace(END, "")
    if not sym:
        return ""
    try:
        return bytes(ord(c) & 0xFF for c in sym).decode("utf-8", errors="replace")
    except ValueError:
        return sym


def _char_class(ch: str) -> str:
    cat = unicodedata.category(ch)
    if cat[0] in ("L", "M"):
        return "W"          # letter or combining mark -> part of a word
    if cat == "Nd":
        return "D"          # decimal digit
    if ch.isspace():
        return "S"
    return "O"              # punctuation / symbol / other


def pretokenize(text: str, lowercase: bool = True,
                strip_joiners: bool = True) -> list[tuple[str, bool]]:
    """Return ``[(pretoken, is_word), ...]``.

    ``is_word`` marks runs of letters/marks. Digits and punctuation are
    tokenised too -- the vocabulary has to be complete -- but they are not
    counted as words when computing fertility, because tokens-per-word is only
    meaningful over actual words.
    """
    text = unicodedata.normalize("NFC", text)
    if strip_joiners:
        text = "".join(ch for ch in text if ch not in _ZW)
    if lowercase:
        text = unicodedata.normalize("NFC", text.casefold())

    out: list[tuple[str, bool]] = []
    cur: list[str] = []
    cur_cls: str | None = None

    def flush() -> None:
        nonlocal cur, cur_cls
        if cur:
            out.append(("".join(cur), cur_cls == "W"))
        cur, cur_cls = [], None

    for ch in text:
        cls = _char_class(ch)
        if cls == "S":
            flush()
        elif cls == "O":
            flush()
            out.append((ch, False))
        else:
            if cls != cur_cls:
                flush()
                cur_cls = cls
            cur.append(ch)
    flush()
    return out


class BPE:
    """Classic BPE over akshara base units, with ids.

    Training order is fully determined by ``(-frequency, pair)``: ties break on
    the pair's string comparison, never on dict or set iteration order. That
    matters because the tokenizer hash is part of every shard manifest -- a
    tokenizer that retrains to a different vocabulary from identical input would
    invalidate the reproducibility claim.
    """

    def __init__(self, unit_mode: str = "byte", specials: list[str] | None = None):
        self.unit_mode = unit_mode
        self.specials: list[str] = list(specials or [])
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}
        self.base_alphabet: list[str] = []
        self.vocab: list[str] = []
        self.tok2id: dict[str, int] = {}
        self.id2tok: list[str] = []

    # -- training ----------------------------------------------------------

    def train(self, pretoken_freqs: dict[str, int], target_vocab: int,
              min_pair_freq: int = 2) -> dict:
        budget = target_vocab - len(self.specials)
        if budget < 1:
            raise ValueError("target_vocab must exceed the number of special tokens")

        words: dict[int, list[str]] = {}
        freqs: dict[int, int] = {}
        # sorted() so the word ids do not depend on dict insertion order
        for i, pt in enumerate(sorted(pretoken_freqs)):
            words[i] = to_units(pt, self.unit_mode) + [END]
            freqs[i] = pretoken_freqs[pt]

        if self.unit_mode == "byte":
            # Fixed and complete: every possible input byte has a symbol, so
            # nothing is ever unrepresentable and the base size does not drift
            # with the corpus.
            alphabet = {chr(b) for b in range(256)} | {END}
        else:
            alphabet = set()
            for syms in words.values():
                alphabet.update(syms)
        self.base_alphabet = sorted(alphabet)

        if len(self.base_alphabet) > budget:
            raise ValueError(
                f"unit_mode={self.unit_mode!r} needs {len(self.base_alphabet)} base "
                f"symbols but the vocabulary budget is only {budget}. "
                f"Use unit_mode='byte' (256 base) or raise target_vocab."
            )

        vocab: set[str] = set(alphabet)
        pair_freq: dict[tuple[str, str], int] = defaultdict(int)
        pair_words: dict[tuple[str, str], set[int]] = defaultdict(set)
        for i, syms in words.items():
            f = freqs[i]
            for pair in zip(syms, syms[1:]):
                pair_freq[pair] += f
                pair_words[pair].add(i)

        # Lazy max-heap; stale entries are re-verified against pair_freq.
        heap = [(-c, p) for p, c in sorted(pair_freq.items())]
        heapq.heapify(heap)

        merges: list[tuple[str, str]] = []
        while len(vocab) < budget and heap:
            neg, pair = heapq.heappop(heap)
            cur = pair_freq.get(pair, 0)
            if -neg != cur:
                continue
            if cur < min_pair_freq:
                break
            a, b = pair
            vocab.add(a + b)
            merges.append(pair)

            for i in sorted(pair_words.get(pair, ())):   # sorted: no set-order dependence
                syms = words[i]
                if not any(syms[j] == a and syms[j + 1] == b for j in range(len(syms) - 1)):
                    continue
                f = freqs[i]
                for p in zip(syms, syms[1:]):
                    pair_freq[p] -= f
                    if pair_freq[p] <= 0:
                        pair_freq.pop(p, None)
                merged: list[str] = []
                j = 0
                while j < len(syms):
                    if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                        merged.append(a + b)
                        j += 2
                    else:
                        merged.append(syms[j])
                        j += 1
                words[i] = merged
                for p in zip(merged, merged[1:]):
                    pair_freq[p] += f
                    pair_words[p].add(i)
                    heapq.heappush(heap, (-pair_freq[p], p))
            pair_freq.pop(pair, None)

        self.merges = merges
        self.ranks = {p: r for r, p in enumerate(merges)}
        self.vocab = self.base_alphabet + [a + b for (a, b) in merges]
        self._build_ids()
        return {
            "n_base": len(self.base_alphabet),
            "n_merges": len(merges),
            "vocab_size": len(self.id2tok),
            "stopped_early": len(vocab) < budget,
        }

    def _build_ids(self) -> None:
        self.id2tok = list(self.specials) + list(self.vocab)
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}

    # -- encoding ----------------------------------------------------------

    def encode_pretoken(self, pt: str) -> list[str]:
        syms = to_units(pt, self.unit_mode) + [END]
        if len(syms) == 1:
            return syms
        while True:
            best_rank, best_i = None, -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            syms = syms[:best_i] + [syms[best_i] + syms[best_i + 1]] + syms[best_i + 2:]
        return syms

    def encode(self, text: str, unk_id: int = 0) -> list[int]:
        ids: list[int] = []
        for pt, _is_word in pretokenize(text, strip_joiners=True):
            for sym in self.encode_pretoken(pt):
                ids.append(self.tok2id.get(sym, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Best-effort reconstruction.

        Lossy by construction: pretokenisation casefolds, strips joiners and
        discards inter-token whitespace, so decode exists for *inspection* --
        the decoded previews in the learning ledger -- and never for
        round-tripping. In byte mode the symbols are reassembled into one byte
        string before decoding, so a merge that lands mid-codepoint still
        resolves correctly instead of producing a replacement character.
        """
        if self.unit_mode == "byte":
            buf = bytearray()
            for i in ids:
                if not (0 <= i < len(self.id2tok)):
                    continue
                t = self.id2tok[i]
                if i < len(self.specials):
                    buf.extend(t.encode("utf-8"))
                    continue
                if t.endswith(END):
                    buf.extend(ord(c) & 0xFF for c in t[:-1])
                    buf.extend(b" ")
                else:
                    buf.extend(ord(c) & 0xFF for c in t)
            return buf.decode("utf-8", errors="replace")
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(self.id2tok):
                out.append(self.id2tok[i].replace(END, " "))
        return "".join(out)

    def decode_token(self, tid: int) -> str:
        """A short printable form of one token, for ledger previews."""
        if not (0 <= tid < len(self.id2tok)):
            return "<oob>"
        t = self.id2tok[tid]
        if tid < len(self.specials):
            return t
        if self.unit_mode == "byte":
            text = byte_symbol_to_text(t)
            return (text + "_") if t.endswith(END) else text
        return t.replace(END, "_")

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": "tdes-bpe/1",
            "unit_mode": self.unit_mode,
            "specials": self.specials,
            "base_alphabet": self.base_alphabet,
            "merges": [list(m) for m in self.merges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPE":
        t = cls(unit_mode=d["unit_mode"], specials=d["specials"])
        t.base_alphabet = list(d["base_alphabet"])
        t.merges = [tuple(m) for m in d["merges"]]
        t.ranks = {p: r for r, p in enumerate(t.merges)}
        t.vocab = t.base_alphabet + [a + b for (a, b) in t.merges]
        t._build_ids()
        return t

    @property
    def vocab_size(self) -> int:
        return len(self.id2tok)
