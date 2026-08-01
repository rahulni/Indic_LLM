"""Train a simple English BPE tokenizer and compute X1 for Wikipedia's India page.

X1 is reported in two forms because assignment handouts often use "vocab" to
mean either all word occurrences or the set of unique word types:

1. occurrence_x1 = total BPE pieces across the article / total word occurrences
2. type_x1       = total BPE pieces across unique words / unique word count
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


END_OF_WORD = "</w>"
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*", re.IGNORECASE)


def load_wikipedia_extract(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    page = data["query"]["pages"][0]
    return page["title"], page["extract"]


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def extract_words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def initial_symbols(word: str) -> tuple[str, ...]:
    return tuple(word) + (END_OF_WORD,)


def pair_counts(vocab: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for symbols, frequency in vocab.items():
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += frequency
    return counts


def merge_pair(
    vocab: Counter[tuple[str, ...]], pair: tuple[str, str]
) -> Counter[tuple[str, ...]]:
    left, right = pair
    merged_symbol = left + right
    merged_vocab: Counter[tuple[str, ...]] = Counter()

    for symbols, frequency in vocab.items():
        next_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                next_symbols.append(merged_symbol)
                i += 2
            else:
                next_symbols.append(symbols[i])
                i += 1
        merged_vocab[tuple(next_symbols)] += frequency

    return merged_vocab


def train_bpe(
    words: list[str], target_vocab_size: int, min_pair_frequency: int
) -> tuple[list[tuple[str, str]], int]:
    word_frequencies = Counter(words)
    vocab: Counter[tuple[str, ...]] = Counter(
        {initial_symbols(word): frequency for word, frequency in word_frequencies.items()}
    )
    base_symbols = {symbol for symbols in vocab for symbol in symbols}
    merges: list[tuple[str, str]] = []

    while len(base_symbols) + len(merges) < target_vocab_size:
        pairs = pair_counts(vocab)
        if not pairs:
            break

        best_pair, best_frequency = max(pairs.items(), key=lambda item: (item[1], item[0]))
        if best_frequency < min_pair_frequency:
            break

        merges.append(best_pair)
        vocab = merge_pair(vocab, best_pair)

    return merges, len(base_symbols) + len(merges)


def encode_word(word: str, merge_ranks: dict[tuple[str, str], int]) -> list[str]:
    symbols = list(initial_symbols(word))

    while len(symbols) > 1:
        ranked_pairs = [
            (merge_ranks[pair], i, pair)
            for i, pair in enumerate(zip(symbols, symbols[1:]))
            if pair in merge_ranks
        ]
        if not ranked_pairs:
            break

        _, _, best_pair = min(ranked_pairs)
        symbols = list(merge_pair(Counter({tuple(symbols): 1}), best_pair).keys())[0]

    pieces: list[str] = []
    for symbol in symbols:
        if symbol == END_OF_WORD:
            continue
        if symbol.endswith(END_OF_WORD):
            symbol = symbol[: -len(END_OF_WORD)]
        if symbol:
            pieces.append(symbol)
    return pieces


def encoded_piece_count(words: list[str], merges: list[tuple[str, str]]) -> int:
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
    word_frequencies = Counter(words)
    return sum(
        len(encode_word(word, merge_ranks)) * frequency
        for word, frequency in word_frequencies.items()
    )


def encoded_unique_piece_count(words: list[str], merges: list[tuple[str, str]]) -> int:
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
    return sum(len(encode_word(word, merge_ranks)) for word in set(words))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/india_en_extract.json"),
        help="Wikipedia API JSON extract for the English India page.",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=5000,
        help="English BPE vocabulary budget, including base characters and merges.",
    )
    parser.add_argument(
        "--min-pair-frequency",
        type=int,
        default=1,
        help="Stop learning merges once the best remaining pair appears fewer times.",
    )
    args = parser.parse_args()

    title, text = load_wikipedia_extract(args.input)
    words = extract_words(text)
    unique_words = set(words)
    merges, final_vocab_size = train_bpe(
        words, args.target_vocab_size, args.min_pair_frequency
    )

    occurrence_pieces = encoded_piece_count(words, merges)
    unique_pieces = encoded_unique_piece_count(words, merges)
    occurrence_x1 = occurrence_pieces / len(words)
    type_x1 = unique_pieces / len(unique_words)

    print(f"page_title: {title}")
    print(f"word_occurrences: {len(words)}")
    print(f"unique_word_types: {len(unique_words)}")
    print(f"target_bpe_vocab_size: {args.target_vocab_size}")
    print(f"final_bpe_vocab_size: {final_vocab_size}")
    print(f"learned_merges: {len(merges)}")
    print(f"min_pair_frequency: {args.min_pair_frequency}")
    print(f"occurrence_bpe_pieces: {occurrence_pieces}")
    print(f"occurrence_x1: {occurrence_x1:.6f}")
    print(f"type_bpe_pieces: {unique_pieces}")
    print(f"type_x1: {type_x1:.6f}")
    print(f"occurrence_pass_1.2: {occurrence_x1 <= 1.2}")
    print(f"type_pass_1.2: {type_x1 <= 1.2}")


if __name__ == "__main__":
    main()
