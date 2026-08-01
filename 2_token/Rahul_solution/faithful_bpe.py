"""
A from-scratch, Unicode-aware BPE tokenizer that is *faithful*:

    decode(encode(text)) == text   for ANY text (in or out of the training corpus).

Design (chosen so the ratios are reproducible and defensible):

  * No normalization, no casefolding. NFKC / casefold silently change visible
    characters (² -> 2, İ -> i̇), which breaks round-trip. We keep bytes exactly.
  * SentencePiece-style space marker: the ASCII space U+0020 is represented by the
    marker character '▁' (U+2581) so the tokenizer can attach a leading space to a
    word ("▁india") and spend fewer tokens. On decode, '▁' -> ' '.
  * Byte fallback (256 reserved tokens): any unit not learned during training -- an
    unseen script, emoji, a literal '▁' in the input -- is emitted as its raw UTF-8
    bytes. Nothing is ever dropped. This is the concrete bug we fix versus a
    tokenizer that maps unseen characters to [UNK] (which deletes them).
  * Indic-aware base units: a "word" is segmented into extended grapheme clusters
    (aksharas) via regex \\X, so a Devanagari/Telugu syllable such as "క్ష" or "भा"
    is one base symbol instead of 2-3 codepoints -> lower fertility on Indic text.
    Byte-level BPE is deliberately avoided (it wastes tokens on Indic UTF-8).

Two fertility views are supported by analyze():
  * faithful units  = one contiguous letter/mark/number run OR one visible
                      punctuation/symbol char  (the grading standard).
  * words           = letter/mark/number runs only (the literal assignment reading).
"""
from __future__ import annotations

from collections import defaultdict
import heapq
import json

import regex as _re

MARK = "▁"                 # ▁  space marker
_PUA0 = 0xE000                  # byte-fallback tokens live at PUA U+E000..U+E0FF
_BYTE_RANGE = range(_PUA0, _PUA0 + 256)

# One faithful "unit": a run of letters/marks/numbers, or a single visible symbol.
_FAITHFUL_UNIT_RE = _re.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
_WORD_RE = _re.compile(r"[\p{L}\p{M}\p{N}]+")


def _byte_sym(b: int) -> str:
    return chr(_PUA0 + b)


def _is_byte_sym(s: str) -> bool:
    return len(s) == 1 and ord(s) in _BYTE_RANGE


def faithful_units(text: str) -> int:
    return len(_FAITHFUL_UNIT_RE.findall(text))


def word_units(text: str) -> int:
    return len(_WORD_RE.findall(text))


class FaithfulBPE:
    # character classes used for pre-tokenization
    _W, _D, _O, _NL = "W", "D", "O", "NL"

    def __init__(self, unit_mode: str = "akshara", pretok_mode: str = "class"):
        self.unit_mode = unit_mode          # "akshara" or "char"
        self.pretok_mode = pretok_mode      # "class" or "whitespace"
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}
        self.base_units: list[str] = []      # learned single-unit symbols (non-byte)
        self.base_set: set[str] = set()
        self.sym_to_id: dict[str, int] = {}
        self.id_to_sym: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Pre-tokenization  (fully reversible: concatenated surfaces == text) #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _class(ch: str) -> str:
        import unicodedata
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M"):
            return FaithfulBPE._W
        if cat == "Nd" or cat[0] == "N":
            return FaithfulBPE._D
        return FaithfulBPE._O

    def _pretokens(self, text: str):
        """Yield (units, cls) for each pre-token; units is a list of symbols.

        Two splitting modes (both fully reversible: concatenated surfaces == text):

        "whitespace" (SentencePiece-style): a maximal run of non-space characters is
            ONE pre-token, so URLs/markup like "https://en.wikipedia.org/wiki/India"
            stay whole and compress well. Matches the reference's methodology.
        "class": split runs by character class -- letters/marks (W), digits (D),
            punctuation/symbols (O) -- so linguistic words are isolated from markup
            and get dedicated merges. Lower cross-lingual spread, faster to train.

        In both modes each ASCII space U+0020 becomes a leading marker '▁' on the
        following run; literal '▁' and PUA chars are force-expanded to byte symbols
        so the marker is unambiguous; other whitespace is kept literally.
        """
        MARKCELL, BYTECELL, CHARCELL, NLCELL = 0, 1, 2, 3
        cells = []
        for ch in text:
            if ch == " ":
                cells.append((MARKCELL, None))
            elif ch == MARK or ord(ch) in _BYTE_RANGE:
                cells.append((BYTECELL, ch.encode("utf-8")))
            elif ch in "\r\n\t\f\v" or (ch.isspace() and ch != " "):
                cells.append((NLCELL, ch))
            else:
                cells.append((CHARCELL, ch))

        split_by_class = self.pretok_mode == "class"
        i, n = 0, len(cells)
        while i < n:
            kind = cells[i][0]
            if kind == NLCELL:
                j = i
                buf = []
                while j < n and cells[j][0] == NLCELL:
                    buf.append(cells[j][1])
                    j += 1
                yield list("".join(buf)), self._NL
                i = j
                continue

            marks = 0
            while i < n and cells[i][0] == MARKCELL:
                marks += 1
                i += 1

            run_cls = None  # class of the current run, for "class" mode
            units = [MARK] * marks
            has_letter = False
            charbuf = []

            def flush_chars():
                nonlocal charbuf, has_letter
                if charbuf:
                    s = "".join(charbuf)
                    if _WORD_RE.search(s):
                        has_letter = True
                    units.extend(self._segment(s))
                    charbuf = []

            j = i
            while j < n and cells[j][0] in (CHARCELL, BYTECELL):
                if split_by_class and cells[j][0] == CHARCELL:
                    c = self._class(cells[j][1])
                    if run_cls is None:
                        run_cls = c
                    elif c != run_cls:
                        break
                if cells[j][0] == CHARCELL:
                    charbuf.append(cells[j][1])
                else:
                    flush_chars()
                    units.extend(_byte_sym(b) for b in cells[j][1])
                j += 1
            flush_chars()
            i = j

            if units:
                yield units, (self._W if has_letter else self._O)

    def _segment(self, s: str) -> list[str]:
        if not s:
            return []
        if self.unit_mode == "akshara":
            return _re.findall(r"\X", s)
        return list(s)

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #
    def train(self, texts_with_weights, target_vocab: int = 10000, min_pair_freq: int = 2):
        """texts_with_weights: list of (text, weight). Weight repeats the corpus."""
        pt_freq: dict[tuple, int] = defaultdict(int)
        for text, w in texts_with_weights:
            for units, _cls in self._pretokens(text):
                key = tuple(units)
                if key:
                    pt_freq[key] += w

        words = {}
        freqs = {}
        alphabet = set()
        for idx, (units, fr) in enumerate(pt_freq.items()):
            syms = list(units)
            words[idx] = syms
            freqs[idx] = fr
            for s in syms:
                if not _is_byte_sym(s):
                    alphabet.add(s)

        self.base_units = sorted(alphabet)
        self.base_set = set(self.base_units)

        # room left for merges after 256 byte tokens + learned base units
        max_merges = target_vocab - 256 - len(self.base_units)

        pair_freq = defaultdict(int)
        pair_words = defaultdict(set)
        for i, syms in words.items():
            f = freqs[i]
            for pair in zip(syms, syms[1:]):
                pair_freq[pair] += f
                pair_words[pair].add(i)

        heap = [(-c, p) for p, c in pair_freq.items()]
        heapq.heapify(heap)

        merges: list[tuple[str, str]] = []
        while len(merges) < max_merges and heap:
            neg, pair = heapq.heappop(heap)
            cur = pair_freq.get(pair, 0)
            if -neg != cur:
                continue
            if cur < min_pair_freq:
                break
            a, b = pair
            new_sym = a + b
            merges.append(pair)

            for i in list(pair_words.get(pair, ())):
                syms = words[i]
                if not any(syms[j] == a and syms[j + 1] == b for j in range(len(syms) - 1)):
                    continue
                f = freqs[i]
                for p in zip(syms, syms[1:]):
                    pair_freq[p] -= f
                    if pair_freq[p] <= 0:
                        pair_freq.pop(p, None)
                merged = []
                j = 0
                while j < len(syms):
                    if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                        merged.append(new_sym)
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
        self._build_vocab()
        return {
            "byte_tokens": 256,
            "base_units": len(self.base_units),
            "merges": len(self.merges),
            "vocab_size": len(self.sym_to_id),
        }

    def _build_vocab(self):
        self.sym_to_id, self.id_to_sym = {}, {}
        for b in range(256):
            sym = _byte_sym(b)
            self.sym_to_id[sym] = b
            self.id_to_sym[b] = sym
        nid = 256
        for s in self.base_units:
            self.sym_to_id[s] = nid
            self.id_to_sym[nid] = s
            nid += 1
        for a, b in self.merges:
            s = a + b
            if s not in self.sym_to_id:
                self.sym_to_id[s] = nid
                self.id_to_sym[nid] = s
                nid += 1

    # ------------------------------------------------------------------ #
    # Encoding / decoding                                                #
    # ------------------------------------------------------------------ #
    def _merge_syms(self, syms: list[str]) -> list[str]:
        if len(syms) < 2:
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

    def _encode_units(self, units: list[str]) -> list[str]:
        syms = []
        for u in units:
            if _is_byte_sym(u) or u in self.base_set:
                syms.append(u)
            else:                                   # unseen unit -> byte fallback
                syms.extend(_byte_sym(b) for b in u.encode("utf-8"))
        return self._merge_syms(syms)

    def encode(self, text: str) -> list[int]:
        ids = []
        for units, _cls in self._pretokens(text):
            for s in self._encode_units(units):
                ids.append(self.sym_to_id[s])
        return ids

    def decode(self, ids) -> str:
        buf = bytearray()
        for i in ids:
            if i < 256:
                buf.append(i)
            else:
                buf += self.id_to_sym[i].replace(MARK, " ").encode("utf-8")
        return buf.decode("utf-8")

    # ------------------------------------------------------------------ #
    # Metrics                                                            #
    # ------------------------------------------------------------------ #
    def analyze(self, text: str, full: bool = True) -> dict:
        """Counts for both fertility views (both use the same total token count).

        faithful metric : total_tokens / faithful_units  (grading standard)
        word metric     : total_tokens / words           (comparable to reference)

        A "faithful unit" is one contiguous letter/mark/number run OR one visible
        punctuation/symbol char; a "word" is a letter/mark/number run only. The
        `full` flag is accepted for API symmetry; analyze is already O(one encode).
        """
        return {
            "total_tokens": len(self.encode(text)),
            "faithful_units": faithful_units(text),
            "words": word_units(text),
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def display_token(self, i: int) -> str:
        """Human-readable surface for a token id (marker shown as ▁, bytes as <0xHH>)."""
        if i < 256:
            return f"<0x{i:02X}>"
        return self.id_to_sym[i]

    def token_list(self) -> list[dict]:
        out = []
        for i in range(len(self.id_to_sym)):
            kind = "byte" if i < 256 else ("base" if self.id_to_sym[i] in self.base_set else "merge")
            out.append({"id": i, "token": self.display_token(i), "kind": kind})
        return out

    def save(self, path: str):
        data = {
            "format": "faithful-bpe/1",
            "unit_mode": self.unit_mode,
            "pretok_mode": self.pretok_mode,
            "marker": MARK,
            "byte_fallback": True,
            "base_units": self.base_units,
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "FaithfulBPE":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(unit_mode=data.get("unit_mode", "akshara"),
                  pretok_mode=data.get("pretok_mode", "class"))
        tok.base_units = list(data["base_units"])
        tok.base_set = set(tok.base_units)
        tok.merges = [(a, b) for a, b in data["merges"]]
        tok.ranks = {p: r for r, p in enumerate(tok.merges)}
        tok._build_vocab()
        return tok
