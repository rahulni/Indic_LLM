"""A small, from-scratch BPE tokenizer (Unicode-aware, multilingual).

Design choices (documented so the ratios are reproducible / defensible):
  * Text is NFC-normalized and casefolded (case carries no info in Indic scripts;
    folding English/Spanish reduces vocab pressure so the budget is spent on
    real subwords rather than capitalisation variants).
  * Pre-tokenization keeps Unicode combining marks (vowel signs etc.) attached to
    their base letter, so Devanagari / Telugu / Kannada syllables stay intact.
    A "word" = a maximal run of Letter/Mark chars. Digit-runs and punctuation are
    tokenized too (so the vocab is complete) but are NOT counted as words.
  * Classic byte-pair-encoding with a `</w>` end-of-word marker; merges never
    cross a pre-token boundary.

The public metric is *fertility*  X = (word tokens) / (words)  -- average number
of tokens per word. Lower = better compression. ~1.2 or less is the English goal.
"""
from collections import defaultdict
import heapq
import unicodedata

import regex as _re  # for \X grapheme-cluster (akshara) segmentation

END = "</w>"

# Zero-width joiner / non-joiner: invisible, but produce spurious spelling
# variants in Indic text. Stripping them collapses variants -> more repeated
# forms -> more useful merges (see normalize_indic / strip_joiners).
_ZW = {"‌", "‍"}  # ZWNJ, ZWJ


def to_units(pt: str, mode: str = "char"):
    """Segment a pre-token into base symbols.

    mode="char"    -> Unicode codepoints (classic BPE base units).
    mode="akshara" -> extended grapheme clusters (regex \\X). An Indic syllable
                      such as "క్ష" or "भा" becomes ONE base symbol instead of
                      2-3 codepoints, so rare words start far closer to a single
                      token -> lower fertility. Latin stays ~codepoints.
    """
    if mode == "akshara":
        return _re.findall(r"\X", pt)
    return list(pt)


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------
def _char_class(ch: str) -> str:
    cat = unicodedata.category(ch)
    if cat[0] in ("L", "M"):
        return "W"          # letter or combining mark  -> part of a word
    if cat == "Nd":
        return "D"          # decimal digit
    if ch.isspace():
        return "S"
    return "O"              # punctuation / symbol / other


def pretokenize(text: str, lowercase: bool = True, strip_joiners: bool = False):
    """Return a list of (pretoken:str, is_word:bool)."""
    text = unicodedata.normalize("NFC", text)
    if strip_joiners:
        text = "".join(ch for ch in text if ch not in _ZW)
    if lowercase:
        text = text.casefold()
        text = unicodedata.normalize("NFC", text)
    out = []
    cur = []
    cur_cls = None

    def flush():
        nonlocal cur, cur_cls
        if cur:
            out.append(("".join(cur), cur_cls == "W"))
        cur = []
        cur_cls = None

    for ch in text:
        cls = _char_class(ch)
        if cls == "S":
            flush()
        elif cls == "O":
            flush()
            out.append((ch, False))
        else:  # W or D
            if cls != cur_cls:
                flush()
                cur_cls = cls
            cur.append(ch)
    flush()
    return out


# ---------------------------------------------------------------------------
# BPE training
# ---------------------------------------------------------------------------
class BPETokenizer:
    def __init__(self, unit_mode: str = "char"):
        self.unit_mode = unit_mode       # "char" or "akshara"
        self.merges = []                 # ordered list of (a, b)
        self.ranks = {}                  # (a, b) -> rank
        self.base_alphabet = []          # sorted base symbols
        self.vocab = []                  # base_alphabet + merged tokens (in order)

    def train(self, pretoken_freqs: dict, target_vocab: int, min_pair_freq: int = 2):
        """`pretoken_freqs`: {pretoken_string: frequency} over the whole corpus."""
        words = {}    # id -> list[symbol]
        freqs = {}    # id -> frequency
        for i, (pt, fr) in enumerate(pretoken_freqs.items()):
            words[i] = to_units(pt, self.unit_mode) + [END]
            freqs[i] = fr

        vocab = set()
        for syms in words.values():
            vocab.update(syms)
        self.base_alphabet = sorted(vocab)

        pair_freq = defaultdict(int)
        pair_words = defaultdict(set)     # (a,b) -> superset of word-ids containing it
        for i, syms in words.items():
            f = freqs[i]
            for pair in zip(syms, syms[1:]):
                pair_freq[pair] += f
                pair_words[pair].add(i)

        # lazy max-heap of (-freq, pair); entries are re-verified against pair_freq
        heap = [(-c, p) for p, c in pair_freq.items()]
        heapq.heapify(heap)

        merges = []
        n_base = len(vocab)
        while len(vocab) < target_vocab and heap:
            neg, pair = heapq.heappop(heap)
            cur = pair_freq.get(pair, 0)
            if -neg != cur:               # stale heap entry
                continue
            if cur < min_pair_freq:
                break
            a, b = pair
            new_sym = a + b
            vocab.add(new_sym)
            merges.append(pair)

            affected = [i for i in pair_words.get(pair, ()) ]
            for i in affected:
                syms = words[i]
                # does this word actually still contain the pair?
                if not any(syms[j] == a and syms[j + 1] == b for j in range(len(syms) - 1)):
                    continue
                f = freqs[i]
                for p in zip(syms, syms[1:]):          # remove old contributions
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
                for p in zip(merged, merged[1:]):      # add new contributions
                    pair_freq[p] += f
                    pair_words[p].add(i)
                    heapq.heappush(heap, (-pair_freq[p], p))
            pair_freq.pop(pair, None)

        self.merges = merges
        self.ranks = {p: r for r, p in enumerate(merges)}
        self.vocab = self.base_alphabet + [a + b for (a, b) in merges]
        return {"n_base": n_base, "n_merges": len(merges), "vocab_size": len(self.vocab)}

    # -----------------------------------------------------------------
    def encode_pretoken(self, pt: str):
        syms = to_units(pt, self.unit_mode) + [END]
        if len(syms) == 1:
            return syms
        while True:
            best_rank = None
            best_i = -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            syms = syms[:best_i] + [syms[best_i] + syms[best_i + 1]] + syms[best_i + 2:]
        return syms
