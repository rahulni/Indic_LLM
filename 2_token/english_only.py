"""ENGLISH ONLY fertility experiments. Measures how low English can honestly go,
and whether extracting captions helps or hurts. All tokenizers here are trained on
English alone (a dedicated 10k budget) unless noted.

fertility X = word_tokens / words  (measured on the same English text).
"""
import os
import sys
from collections import Counter

from bpe import BPETokenizer, pretokenize

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
VOCAB = 10_000

APOS = {"'", "’", "-"}  # ' ’ -


def load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return f.read()


def glue_punct(pretoks):
    """Merge word ' / - word into a single word (India's, well-regulated)."""
    out, i = [], 0
    while i < len(pretoks):
        pt, isw = pretoks[i]
        if isw and i + 2 < len(pretoks):
            mid, midw = pretoks[i + 1]
            nxt, nxtw = pretoks[i + 2]
            if not midw and mid in APOS and nxtw:
                out.append((pt + mid + nxt, True))
                i += 3
                continue
        out.append((pt, isw))
        i += 1
    return out


def measure(text, vocab=VOCAB, mpf=2, glue=False):
    pts = pretokenize(text)
    if glue:
        pts = glue_punct(pts)
    freqs = Counter(pt for pt, _ in pts)
    tok = BPETokenizer(unit_mode="char")
    info = tok.train(dict(freqs), vocab, min_pair_freq=mpf)
    cache, words, wtoks = {}, 0, 0
    for pt, isw in pts:
        n = cache.get(pt)
        if n is None:
            n = len(tok.encode_pretoken(pt))
            cache[pt] = n
        if isw:
            words += 1
            wtoks += n
    return wtoks / words if words else 0.0, words, info["vocab_size"]


def row(name, text, **kw):
    x, w, v = measure(text, **kw)
    tag = "  <-- < 1.2 !" if x < 1.2 else ""
    print(f"{name:40s} X={x:6.3f}  words={w:6d}  vocab={v:5d}{tag}")


def main():
    prose = load("en_prose.txt")
    full = load("en_full.txt")
    print("== English-only tokenizer, dedicated 10k budget ==")
    row("prose only", prose)
    row("prose + captions + tables (your idea)", full)
    row("full, glue punctuation (India's/well-regd)", full, glue=True)
    row("prose, glue punctuation", prose, glue=True)
    print("\n== budget sensitivity (prose only) ==")
    for v in (4000, 6000, 8000, 10000):
        row(f"prose, vocab={v}", prose, vocab=v)
    print("\n== overfit reference ==")
    row("prose, min_pair_freq=1 (memorises)", prose, mpf=1)


if __name__ == "__main__":
    main()
