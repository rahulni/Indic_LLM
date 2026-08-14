"""tiny_shakespeare corpus prep for Track B's word-level LM.

Downloads (and caches) Karpathy's tiny_shakespeare dataset, splits it into
word tokens, builds a closed word vocabulary (rest -> <UNK>), and reports the
EXACT measured byte/word counts -- never a remembered figure, per the
project's citation-verification discipline.

The natural corpus barely has words anywhere near the 32-character cap that
motivates this track (see the measured length histogram this module
produces), so a synthetic long-word supplement is generated separately here
and kept clearly labeled -- it is never blended into natural-corpus
statistics such as perplexity.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "tinyshakespeare.txt"

WORD_VOCAB_SIZE = 3000  # closed vocabulary for the softmax head; rest -> <UNK>
CONTEXT_LEN = 16
UNK = "<unk>"

_WORD_RE = re.compile(r"[a-zA-Z']+|[.,;:!?\"()\-]")


def fetch_raw_text() -> str:
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, CACHE_PATH)
    return CACHE_PATH.read_text(encoding="utf-8")


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def build_vocab(word_tokens: list[str], vocab_size: int = WORD_VOCAB_SIZE) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    for w in word_tokens:
        counts[w] = counts.get(w, 0) + 1
    most_common = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: vocab_size - 1]
    id_to_word = [UNK] + [w for w, _ in most_common]
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    return word_to_id, id_to_word


def build_alphabet(word_tokens: list[str]) -> tuple[dict[str, int], list[str]]:
    chars = sorted({c for w in word_tokens for c in w})
    char_to_id = {c: i for i, c in enumerate(chars)}
    return char_to_id, chars


def length_histogram(words: list[str]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for w in words:
        hist[len(w)] = hist.get(len(w), 0) + 1
    return hist


def synthetic_long_words(
    real_words: list[str], alphabet: list[str], rng: np.random.Generator, n: int = 200
) -> list[dict]:
    """Two clearly-labeled synthetic sources, both >32 characters:
    (a) concatenated real-word compounds, (b) uniformly random letter
    strings drawn from the corpus alphabet. Never mixed into the real
    training/eval corpus -- consumed only by the >32-char robustness check.
    """
    out = []
    long_reals = [w for w in real_words if len(w) >= 3]
    for _ in range(n // 2):
        parts = rng.choice(long_reals, size=rng.integers(4, 8))
        compound = "".join(parts)
        out.append({"word": compound, "source": "compound", "length": len(compound)})
    letters = [c for c in alphabet if c.isalpha()]
    for _ in range(n - len(out)):
        length = int(rng.integers(33, 61))
        word = "".join(rng.choice(letters, size=length))
        out.append({"word": word, "source": "random", "length": length})
    return [w for w in out if w["length"] > 32]


def make_windows(word_ids: np.ndarray, context_len: int = CONTEXT_LEN) -> np.ndarray:
    """Non-overlapping (context_len + 1)-word windows: input is window[:-1],
    target is window[1:]. Simple and fast; plenty of data even so (a ~200k
    word stream yields ~12k windows at context_len=16)."""
    n_windows = len(word_ids) // (context_len + 1)
    trimmed = word_ids[: n_windows * (context_len + 1)]
    return trimmed.reshape(n_windows, context_len + 1)


def load_corpus(seed: int = 0) -> dict:
    raw_text = fetch_raw_text()
    word_tokens = tokenize_words(raw_text)
    word_to_id, id_to_word = build_vocab(word_tokens)
    char_to_id, chars = build_alphabet(word_tokens)
    word_ids = np.array([word_to_id.get(w, 0) for w in word_tokens], dtype=np.int64)

    n_val = max(1, len(word_ids) // 10)
    train_ids, val_ids = word_ids[:-n_val], word_ids[-n_val:]

    rng = np.random.default_rng(seed)
    alpha_words = [w for w in word_tokens if w.isalpha()]
    synth = synthetic_long_words(alpha_words, chars, rng)

    return {
        "raw_bytes": len(raw_text.encode("utf-8")),
        "n_word_tokens": len(word_tokens),
        "n_unique_words": len(set(word_tokens)),
        "vocab_size": len(id_to_word),
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "alphabet": chars,
        "char_to_id": char_to_id,
        "length_histogram": length_histogram([w for w in word_tokens if w.isalpha()]),
        "max_natural_word_length": max((len(w) for w in word_tokens if w.isalpha()), default=0),
        "train_windows": make_windows(train_ids),
        "val_windows": make_windows(val_ids),
        "synthetic_long_words": synth,
    }


if __name__ == "__main__":
    data = load_corpus()
    print(f"raw corpus: {data['raw_bytes']:,} bytes (measured)")
    print(f"word tokens: {data['n_word_tokens']:,}  unique words: {data['n_unique_words']:,}")
    print(f"closed vocab size: {data['vocab_size']} (top words + <unk>)")
    print(f"alphabet ({len(data['alphabet'])} chars): {''.join(data['alphabet'])!r}")
    print(f"max natural word length: {data['max_natural_word_length']} characters")
    print(f"train windows: {data['train_windows'].shape}  val windows: {data['val_windows'].shape}")
    print(f"synthetic long words generated: {len(data['synthetic_long_words'])} (all > 32 chars)")
