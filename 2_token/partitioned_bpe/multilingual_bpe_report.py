"""Optimize BPE vocab allocations for India Wikipedia pages in four languages.

The tokenizer design used here is a language-partitioned, word-internal BPE:

- Each language gets its own BPE merge list.
- The four language vocabularies are counted together against one global cap.
- BPE merges are learned inside words only, never across whitespace.

This is a practical way to optimize the assignment score, because the score
only depends on how tightly the four average-pieces-per-word ratios match.
"""

from __future__ import annotations

import argparse
import bisect
import html
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EN_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*", re.IGNORECASE)


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    name: str
    local_name: str
    path: Path


@dataclass(frozen=True)
class BpeOption:
    vocab_size: int
    pieces: int
    ratio: float


@dataclass(frozen=True)
class LanguageCurve:
    config: LanguageConfig
    title: str
    words: list[str]
    unique_words: int
    base_vocab_size: int
    options_desc: list[BpeOption]
    ratios_asc: list[float]
    options_asc: list[BpeOption]

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass(frozen=True)
class SelectedLanguage:
    curve: LanguageCurve
    option: BpeOption


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def is_word_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "M", "N"}


def extract_unicode_words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char in normalize_text(text):
        if is_word_char(char):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def extract_words(text: str, code: str) -> list[str]:
    normalized = normalize_text(text)
    if code == "en":
        return EN_WORD_RE.findall(normalized)
    return extract_unicode_words(normalized)


def load_wikipedia_extract(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    page = data["query"]["pages"][0]
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"No Wikipedia extract found in {path}")
    return page["title"], page["extract"]


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


def build_curve(config: LanguageConfig, max_vocab_size: int) -> LanguageCurve:
    title, text = load_wikipedia_extract(config.path)
    words = extract_words(text, config.code)
    if not words:
        raise ValueError(f"No words extracted for {config.code}")

    word_frequencies = Counter(words)
    vocab: Counter[tuple[str, ...]] = Counter(
        {tuple(word): frequency for word, frequency in word_frequencies.items()}
    )

    base_symbols = {symbol for symbols in vocab for symbol in symbols}
    base_vocab_size = len(base_symbols)
    current_vocab_size = base_vocab_size
    current_pieces = sum(len(word) * frequency for word, frequency in word_frequencies.items())
    options: list[BpeOption] = [
        BpeOption(current_vocab_size, current_pieces, current_pieces / len(words))
    ]

    while current_vocab_size < max_vocab_size and current_pieces > len(words):
        pairs = pair_counts(vocab)
        if not pairs:
            break

        best_pair, best_frequency = max(pairs.items(), key=lambda item: (item[1], item[0]))
        vocab = merge_pair(vocab, best_pair)
        current_vocab_size += 1
        current_pieces -= best_frequency
        options.append(
            BpeOption(current_vocab_size, current_pieces, current_pieces / len(words))
        )

    # Keep one option per ratio; if a plateau exists, the first option uses less vocab.
    deduped_desc: list[BpeOption] = []
    seen_ratios: set[tuple[int, int]] = set()
    for option in options:
        key = (option.pieces, len(words))
        if key not in seen_ratios:
            deduped_desc.append(option)
            seen_ratios.add(key)

    options_asc = list(reversed(deduped_desc))
    ratios_asc = [option.ratio for option in options_asc]

    return LanguageCurve(
        config=config,
        title=title,
        words=words,
        unique_words=len(word_frequencies),
        base_vocab_size=base_vocab_size,
        options_desc=deduped_desc,
        ratios_asc=ratios_asc,
        options_asc=options_asc,
    )


def min_vocab_option_in_interval(
    curve: LanguageCurve, low: float, high: float, english_limit: float | None
) -> BpeOption | None:
    strict_english_limit = (
        math.nextafter(english_limit, -math.inf) if english_limit is not None else None
    )
    effective_high = (
        min(high, strict_english_limit) if strict_english_limit is not None else high
    )
    if effective_high < low:
        return None

    # Descending-ratio options have ascending vocab sizes. The first option at or
    # below high is therefore the minimum-vocab option in the interval.
    for option in curve.options_desc:
            if option.ratio <= effective_high:
                return option if option.ratio >= low else None
    return None


def feasible_for_width(
    curves: list[LanguageCurve],
    candidate_lows: list[float],
    width: float,
    vocab_cap: int,
    english_limit: float,
) -> list[SelectedLanguage] | None:
    best_selection: list[SelectedLanguage] | None = None
    best_total = math.inf

    for low in candidate_lows:
        high = low + width
        selection: list[SelectedLanguage] = []
        total_vocab = 0

        for curve in curves:
            limit = english_limit if curve.config.code == "en" else None
            option = min_vocab_option_in_interval(curve, low, high, limit)
            if option is None:
                break
            selection.append(SelectedLanguage(curve, option))
            total_vocab += option.vocab_size
        else:
            if total_vocab <= vocab_cap and total_vocab < best_total:
                best_selection = selection
                best_total = total_vocab

    return best_selection


def optimize_allocations(
    curves: list[LanguageCurve], vocab_cap: int, english_limit: float
) -> list[SelectedLanguage]:
    candidate_lows = sorted(
        {
            option.ratio
            for curve in curves
            for option in curve.options_desc
            if option.ratio <= english_limit
        }
    )

    low_width = 0.0
    high_width = max(curve.options_desc[0].ratio for curve in curves)
    best_selection: list[SelectedLanguage] | None = None

    for _ in range(50):
        mid = (low_width + high_width) / 2
        selection = feasible_for_width(
            curves, candidate_lows, mid, vocab_cap, english_limit
        )
        if selection is None:
            low_width = mid
        else:
            best_selection = selection
            high_width = mid

    final_selection = feasible_for_width(
        curves, candidate_lows, high_width + 1e-12, vocab_cap, english_limit
    )
    if final_selection is None:
        raise RuntimeError("No feasible allocation found")

    # Prefer a fuller tokenizer if another option has effectively the same score:
    # search a tiny wider band and choose the allocation with the lowest max ratio,
    # then highest total vocab. The denominator remains unchanged at display scale.
    best_selection = final_selection
    best_key = selection_key(best_selection)
    for multiplier in (1.000001, 1.00001, 1.0001):
        selection = feasible_for_width(
            curves, candidate_lows, high_width * multiplier + 1e-12, vocab_cap, english_limit
        )
        if selection is None:
            continue
        key = selection_key(selection)
        if key < best_key:
            best_selection = selection
            best_key = key

    return best_selection


def selection_key(selection: list[SelectedLanguage]) -> tuple[float, float, int]:
    ratios = [item.option.ratio for item in selection]
    spread = max(ratios) - min(ratios)
    return (spread, max(ratios), -sum(item.option.vocab_size for item in selection))


def assignment_score(selection: list[SelectedLanguage]) -> float:
    ratios = [item.option.ratio for item in selection]
    spread = max(ratios) - min(ratios)
    return math.inf if spread == 0 else 1000 / spread


def render_html(
    selection: list[SelectedLanguage],
    vocab_cap: int,
    english_limit: float,
    output_path: Path,
) -> None:
    total_vocab = sum(item.option.vocab_size for item in selection)
    ratios = [item.option.ratio for item in selection]
    spread = max(ratios) - min(ratios)
    score = assignment_score(selection)
    sorted_selection = sorted(selection, key=lambda item: item.option.ratio, reverse=True)
    x_labels = {"en": "X1", "hi": "X2", "te": "X3", "mr": "X4"}

    rows = []
    for rank, item in enumerate(sorted_selection, 1):
        curve = item.curve
        option = item.option
        rows.append(
            f"""
            <tr>
              <td>{rank}</td>
              <td>{x_labels[curve.config.code]}</td>
              <td>{html.escape(curve.config.name)} <span>{html.escape(curve.config.local_name)}</span></td>
              <td>{html.escape(curve.config.code)}</td>
              <td>{html.escape(curve.title)}</td>
              <td>{curve.word_count:,}</td>
              <td>{curve.unique_words:,}</td>
              <td>{option.vocab_size:,}</td>
              <td>{option.pieces:,}</td>
              <td><strong>{option.ratio:.6f}</strong></td>
            </tr>
            """
        )

    english_ratio = next(
        item.option.ratio for item in selection if item.curve.config.code == "en"
    )
    pass_total = total_vocab < 10000
    pass_english = english_ratio < english_limit
    score_text = "infinite" if math.isinf(score) else f"{score:,.2f}"

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>India Wikipedia BPE Score</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #607080;
      --line: #d9e1e8;
      --panel: #f7f9fb;
      --accent: #0f766e;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 32px auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .sub {{
      margin: 0 0 24px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
    }}
    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .metric .value {{
      display: block;
      font-size: 22px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      font-size: 14px;
    }}
    th {{
      background: #edf3f7;
      color: #26323f;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    td:nth-child(1),
    td:nth-child(2),
    td:nth-child(4),
    td:nth-child(n+6) {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    td:nth-child(3),
    td:nth-child(5) {{
      text-align: left;
    }}
    td span {{
      color: var(--muted);
      margin-left: 4px;
    }}
    .status {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 16px 0 0;
    }}
    .pill {{
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 13px;
      color: #ffffff;
      background: var(--accent);
    }}
    .pill.warn {{ background: var(--warn); }}
    .note {{
      margin-top: 18px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }}
    @media (max-width: 760px) {{
      main {{ width: min(100% - 20px, 1120px); margin: 20px auto; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table {{ display: block; overflow-x: auto; }}
      th, td {{ white-space: nowrap; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>India Wikipedia BPE Score</h1>
    <p class="sub">Language-partitioned word-internal BPE allocation under a shared vocabulary cap.</p>

    <section class="metrics" aria-label="Summary metrics">
      <div class="metric"><span class="label">Total vocab entries</span><span class="value">{total_vocab:,}</span></div>
      <div class="metric"><span class="label">Ratio spread</span><span class="value">{spread:.9f}</span></div>
      <div class="metric"><span class="label">Assignment score</span><span class="value">{score_text}</span></div>
      <div class="metric"><span class="label">English X</span><span class="value">{english_ratio:.6f}</span></div>
    </section>

    <table>
      <thead>
        <tr>
          <th>Sort</th>
          <th>X ID</th>
          <th>Language</th>
          <th>Code</th>
          <th>Page</th>
          <th>Words</th>
          <th>Unique Words</th>
          <th>BPE Vocab Entries</th>
          <th>Encoded Token Occurrences</th>
          <th>X = Tokens / Words</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>

    <div class="status">
      <span class="pill{' warn' if not pass_total else ''}">Total vocab entries &lt; 10000: {str(pass_total)}</span>
      <span class="pill{' warn' if not pass_english else ''}">English X &lt; {english_limit}: {str(pass_english)}</span>
    </div>

    <p class="note">Score formula: 1000 / (largest X - smallest X). The 10,000 limit is applied to tokenizer vocabulary entries. Encoded token occurrences are the tokens produced when the Wikipedia page is tokenized, so they may exceed 10,000 for a long article.</p>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_json(selection: list[SelectedLanguage], output_path: Path) -> None:
    ratios = [item.option.ratio for item in selection]
    x_labels = {"en": "X1", "hi": "X2", "te": "X3", "mr": "X4"}
    data = {
        "total_vocab": sum(item.option.vocab_size for item in selection),
        "spread": max(ratios) - min(ratios),
        "assignment_score": assignment_score(selection),
        "languages": [
            {
                "code": item.curve.config.code,
                "x_label": x_labels[item.curve.config.code],
                "language": item.curve.config.name,
                "local_name": item.curve.config.local_name,
                "page_title": item.curve.title,
                "word_count": item.curve.word_count,
                "unique_words": item.curve.unique_words,
                "base_vocab_size": item.curve.base_vocab_size,
                "selected_bpe_vocab": item.option.vocab_size,
                "bpe_tokens": item.option.pieces,
                "encoded_token_occurrences": item.option.pieces,
                "ratio": item.option.ratio,
            }
            for item in sorted(selection, key=lambda selected: selected.option.ratio, reverse=True)
        ],
    }
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-cap", type=int, default=9999)
    parser.add_argument("--english-limit", type=float, default=1.2)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    configs = [
        LanguageConfig("en", "English", "English", Path("data/india_en_extract.json")),
        LanguageConfig("hi", "Hindi", "हिन्दी", Path("data/india_hi_extract.json")),
        LanguageConfig("te", "Telugu", "తెలుగు", Path("data/india_te_extract.json")),
        LanguageConfig("mr", "Marathi", "मराठी", Path("data/india_mr_extract.json")),
    ]

    curves = [build_curve(config, args.vocab_cap) for config in configs]
    selection = optimize_allocations(curves, args.vocab_cap, args.english_limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_html(
        selection,
        args.vocab_cap,
        args.english_limit,
        args.output_dir / "multilingual_bpe_report.html",
    )
    write_json(selection, args.output_dir / "multilingual_bpe_report.json")

    total_vocab = sum(item.option.vocab_size for item in selection)
    ratios = [item.option.ratio for item in selection]
    score = assignment_score(selection)
    print(f"total_vocab: {total_vocab}")
    print(f"spread: {max(ratios) - min(ratios):.12f}")
    print(f"assignment_score: {score:.6f}")
    for item in sorted(selection, key=lambda selected: selected.option.ratio, reverse=True):
        print(
            f"{item.curve.config.code}: vocab={item.option.vocab_size} "
            f"words={item.curve.word_count} pieces={item.option.pieces} "
            f"x={item.option.ratio:.9f}"
        )
    print(f"html: {args.output_dir / 'multilingual_bpe_report.html'}")
    print(f"json: {args.output_dir / 'multilingual_bpe_report.json'}")


if __name__ == "__main__":
    main()
