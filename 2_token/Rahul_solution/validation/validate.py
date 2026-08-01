#!/usr/bin/env python3
"""
Independent validation of the numbers reported by Rahul_solution's widget.

This script does NOT reuse Rahul_solution's metric code paths for the checks:
faithful-unit counts are computed with the reference evaluator's own regex
(copied verbatim from Sravan_solution/evaluate_tokenizer.py), and token counts
come from re-encoding the committed corpus with the shipped tokenizer files.

Checks
------
A. widget.html consistency: the RESULTS / TOKENIZER / TOKENS blobs embedded in
   widget.html are exactly results.json / tokenizer.json / tokens_all.json
   (widget_template.html itself contains only placeholders, no numbers).
B. Recompute Rahul's numbers: encode each corpus with tokenizer.json
   (loaded through the shipped encoder code, faithful_bpe.py), count faithful
   units with the reference regex, recompute ratios / spread / score and
   compare with results.json.
C. Faithfulness: decode(encode(x)) == x on all four corpora + probe strings.
D. Methodology proof: run Sravan_solution/tokenizer.json (HuggingFace format)
   on Sravan's own corpus (en/hi/te/mai) and reproduce Sravan's metrics.json,
   showing this validator's metric == the reference evaluator's metric.
E. Cross-check: Rahul's hf_tokenizer.json (independent HuggingFace encoder,
   same corpus) must reproduce hf_results.json.
F. Context: Sravan's tokenizer applied to Rahul's corpus. The two tokenizers
   are DIFFERENT (Sravan's 4th language is Maithili, Rahul's is Marathi), so
   this cannot reproduce Rahul's numbers -- it only shows the en/hi/te ratios
   sit in the same regime.
G. Vocab bookkeeping: vocab == 10000 == 256 bytes + base units + merges;
   token export has 10000 contiguous ids.

Run:
    python validate.py            (from this folder; ~2-4 min, pure python encode)

Writes validation_results.json next to this file. Read-only with respect to
every pre-existing file in Rahul_solution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import regex
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent          # Rahul_solution/validation
RAHUL = HERE.parent                             # Rahul_solution
SRAVAN = RAHUL.parent / "Sravan_solution"

sys.path.insert(0, str(RAHUL))
from faithful_bpe import FaithfulBPE            # the shipped encoder code

# --- reference metric, copied verbatim from Sravan_solution/evaluate_tokenizer.py ---
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")

RAHUL_LANGS = ["en", "hi", "te", "mr"]
SRAVAN_LANGS = ["en", "hi", "te", "mai"]

PROBES = [
    "India's population is 1,428,627,663.",
    "El niño comió jalapeños en Málaga.",
    "東京 Tokyo 日本 — 서울 — 🇮🇳🐯",
    "भारत एक देश है।\nline\tbreaks",
    "literal ▁ marker U+2581 must survive",
]

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_js_const(html: str, name: str):
    """Parse the JSON value assigned to `const <name> = ...;` in the widget."""
    marker = f"const {name} = "
    start = html.index(marker) + len(marker)
    value, _ = json.JSONDecoder().raw_decode(html[start:start + 40_000_000])
    return value


def fertility(tokens: int, units: int) -> float:
    return tokens / units


def spread_score(ratios: dict[str, float]) -> tuple[float, float]:
    spread = max(ratios.values()) - min(ratios.values())
    return spread, 1000.0 / spread


def main() -> int:
    results = read_json(RAHUL / "results.json")
    tokenizer_file = read_json(RAHUL / "tokenizer.json")
    tokens_export = read_json(RAHUL / "tokens" / "tokens_all.json")
    hf_results = read_json(RAHUL / "hf_results.json")
    corpus = {c: (RAHUL / "corpus" / f"{c}.faithful.txt").read_text(encoding="utf-8")
              for c in RAHUL_LANGS}

    # ---------------------------------------------------------------- A
    print("\nA. widget.html embeds exactly the shipped data files")
    html = (RAHUL / "widget.html").read_text(encoding="utf-8")
    w_results = extract_js_const(html, "RESULTS")
    w_tok = extract_js_const(html, "TOKENIZER")
    w_tokens = extract_js_const(html, "TOKENS")
    w_hf = extract_js_const(html, "HF")
    check("RESULTS in widget.html == results.json", w_results == results)
    check("TOKENIZER in widget.html == tokenizer.json (base_units+merges)",
          w_tok["base_units"] == tokenizer_file["base_units"]
          and w_tok["merges"] == tokenizer_file["merges"]
          and w_tok["unit_mode"] == tokenizer_file["unit_mode"]
          and w_tok["pretok_mode"] == tokenizer_file["pretok_mode"])
    check("TOKENS in widget.html == tokens/tokens_all.json", w_tokens == tokens_export)
    check("HF cross-check in widget.html == hf_results.json", w_hf == hf_results)
    check("no unreplaced placeholders in widget.html",
          "__RESULTS__" not in html and "__TOKENIZER__" not in html
          and "__TOKENS__" not in html and "__HF__" not in html)
    tmpl = (RAHUL / "widget_template.html").read_text(encoding="utf-8")
    check("widget_template.html holds placeholders only (no baked-in numbers)",
          all(p in tmpl for p in ("__RESULTS__", "__TOKENIZER__", "__TOKENS__", "__HF__")))

    # ---------------------------------------------------------------- B
    print("\nB. Recompute reported numbers from tokenizer.json + corpus/")
    tok = FaithfulBPE.load(str(RAHUL / "tokenizer.json"))
    reported = {l["code"]: l for l in results["languages"]}
    ratios: dict[str, float] = {}
    ids_by_lang: dict[str, list[int]] = {}
    for c in RAHUL_LANGS:
        ids = tok.encode(corpus[c])
        ids_by_lang[c] = ids
        n_tokens = len(ids)
        n_units = len(FAITHFUL_UNIT_RE.findall(corpus[c]))
        n_words = len(WORD_RE.findall(corpus[c]))
        ratios[c] = fertility(n_tokens, n_units)
        word_ratio = fertility(n_tokens, n_words)
        r = reported[c]
        check(f"{c}: token count == reported", n_tokens == r["total_tokens"],
              f"recomputed {n_tokens} vs reported {r['total_tokens']}")
        check(f"{c}: faithful units (reference regex) == reported",
              n_units == r["faithful_units"],
              f"recomputed {n_units} vs reported {r['faithful_units']}")
        check(f"{c}: words == reported", n_words == r["words"])
        check(f"{c}: faithful ratio == reported",
              abs(ratios[c] - r["faithful_ratio"]) < 1e-9,
              f"{ratios[c]:.6f}")
        check(f"{c}: word ratio == reported",
              abs(word_ratio - r["word_ratio"]) < 1e-9, f"{word_ratio:.6f}")
        check(f"{c}: faithful ratio <= 1.2", ratios[c] <= 1.2)
    spread, score = spread_score(ratios)
    check("spread == reported", abs(spread - results["faithful"]["spread"]) < 1e-9,
          f"{spread:.10f}")
    check("score == reported", abs(score - results["faithful"]["score"]) < 1e-6,
          f"{score:.2f} vs reported {results['faithful']['score']:.2f}")

    # ---------------------------------------------------------------- C
    print("\nC. Faithfulness: decode(encode(x)) == x")
    for c in RAHUL_LANGS:
        check(f"{c}: corpus round-trip exact", tok.decode(ids_by_lang[c]) == corpus[c])
    for s in PROBES:
        check(f"probe round-trip exact: {s[:34]!r}", tok.decode(tok.encode(s)) == s)

    # ---------------------------------------------------------------- D
    print("\nD. Methodology proof: reproduce Sravan_solution/metrics.json")
    sravan_metrics = read_json(SRAVAN / "metrics.json")
    sravan_tok = Tokenizer.from_file(str(SRAVAN / "tokenizer.json"))
    s_ratios: dict[str, float] = {}
    for c in SRAVAN_LANGS:
        text = (SRAVAN / "corpus" / f"{c}.faithful.txt").read_text(encoding="utf-8")
        n_tokens = len(sravan_tok.encode(text).ids)
        n_units = len(FAITHFUL_UNIT_RE.findall(text))
        s_ratios[c] = fertility(n_tokens, n_units)
        check(f"{c}: Sravan token count reproduced",
              n_tokens == sravan_metrics["token_counts"][c],
              f"{n_tokens} vs {sravan_metrics['token_counts'][c]}")
        check(f"{c}: Sravan faithful units reproduced",
              n_units == sravan_metrics["faithful_units"][c])
    s_spread, s_score = spread_score(s_ratios)
    check("Sravan score reproduced (6502.56)",
          abs(s_score - sravan_metrics["score"]) < 1e-6, f"{s_score:.2f}")

    # ---------------------------------------------------------------- E
    print("\nE. Cross-check: Rahul hf_tokenizer.json reproduces hf_results.json")
    hf_tok = Tokenizer.from_file(str(RAHUL / "hf_tokenizer.json"))
    for c in RAHUL_LANGS:
        n_tokens = len(hf_tok.encode(corpus[c]).ids)
        check(f"{c}: HF cross-check token count reproduced",
              n_tokens == hf_results["rows"][c]["tokens"],
              f"{n_tokens} vs {hf_results['rows'][c]['tokens']}")

    # ---------------------------------------------------------------- F
    print("\nF. Context: Sravan tokenizer on Rahul corpus (different tokenizer, not a validation)")
    f_rows = {}
    for c in RAHUL_LANGS:
        n_tokens = len(sravan_tok.encode(corpus[c]).ids)
        n_units = len(FAITHFUL_UNIT_RE.findall(corpus[c]))
        f_rows[c] = {"tokens": n_tokens, "faithful_units": n_units,
                     "ratio": fertility(n_tokens, n_units)}
        print(f"       {c}: tokens={n_tokens}  units={n_units}  ratio={f_rows[c]['ratio']:.4f}")

    # ---------------------------------------------------------------- G
    print("\nG. Vocab bookkeeping")
    v = results["vocab"]
    check("vocab total == 10000", len(tok.id_to_sym) == 10000, f"{len(tok.id_to_sym)}")
    check("256 bytes + base units + merges == 10000",
          256 + len(tok.base_units) + len(tok.merges) == 10000,
          f"256 + {len(tok.base_units)} + {len(tok.merges)}")
    check("results.json vocab block matches tokenizer",
          v["total"] == len(tok.id_to_sym) and v["base_units"] == len(tok.base_units)
          and v["merges"] == len(tok.merges))
    check("tokens_all.json has 10000 contiguous ids",
          len(tokens_export) == 10000
          and [t["id"] for t in tokens_export] == list(range(10000)))

    # ---------------------------------------------------------------- summary
    out = {
        "validated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "recomputed_rahul": {
            c: {"tokens": len(ids_by_lang[c]),
                "faithful_units": len(FAITHFUL_UNIT_RE.findall(corpus[c])),
                "faithful_ratio": ratios[c]} for c in RAHUL_LANGS
        },
        "recomputed_rahul_spread": spread,
        "recomputed_rahul_score": score,
        "reproduced_sravan_score": s_score,
        "sravan_tokenizer_on_rahul_corpus_context": f_rows,
        "failures": FAILURES,
        "all_passed": not FAILURES,
    }
    (HERE / "validation_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    print(f"recomputed score = {score:.2f}  (reported {results['faithful']['score']:.2f})")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
