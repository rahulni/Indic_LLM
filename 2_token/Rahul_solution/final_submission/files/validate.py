#!/usr/bin/env python3
"""
Self-contained audit of this submission package. Run it from this folder:

    pip install regex tokenizers
    python validate.py

It re-runs all THREE tokenizers on the committed corpus and checks every number
against the published metrics (results.json + expected_metrics.json — the same
data embedded in ../index.html):

  V1  hf_byte_fallback_tokenizer.json   (HuggingFace format, byte-fallback)
  V2  hf_best_tokenizer.json            (HuggingFace format, NFKC + [UNK]) — submitted
  V3  tokenizer.json                    (custom faithful-bpe/1, encoder: faithful_bpe.py)

Checks per version: per-language token counts, faithful-unit counts, fertility
ratios, spread, score, the X ≤ 1.2 constraint, vocabulary size, and round-trip
behaviour (decode(encode(x)) vs x) on the full corpus and on probe strings.
Exit code 0 = every check passed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from faithful_bpe import FaithfulBPE

LANGS = ["en", "hi", "te", "mr"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    expected = json.loads((ROOT / "expected_metrics.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    corpus = {c: (ROOT / "corpus" / f"{c}.faithful.txt").read_text(encoding="utf-8")
              for c in LANGS}
    units = {c: len(FAITHFUL_UNIT_RE.findall(corpus[c])) for c in LANGS}
    probes = expected["probes"]

    encoders = {}
    for vid in ("v1", "v2", "v3"):
        exp = expected["versions"][vid]
        path = ROOT / exp["tokenizer_file"]
        if vid == "v3":
            tok = FaithfulBPE.load(str(path))
            encoders[vid] = {
                "count": lambda t, tok=tok: len(tok.encode(t)),
                "rt": lambda t, tok=tok: tok.decode(tok.encode(t)) == t,
                "vocab": len(tok.id_to_sym),
            }
        else:
            tok = Tokenizer.from_file(str(path))
            encoders[vid] = {
                "count": lambda t, tok=tok: len(tok.encode(t).ids),
                "rt": lambda t, tok=tok: tok.decode(tok.encode(t).ids, skip_special_tokens=False) == t,
                "vocab": tok.get_vocab_size(),
            }

    for vid in ("v1", "v2", "v3"):
        exp = expected["versions"][vid]
        enc = encoders[vid]
        star = "  << submitted" if vid == expected["submitted"] else ""
        print(f"\n{vid.upper()} — {exp['name']}{star}")
        check(f"{vid}: vocabulary size == {expected['vocab_size']}",
              enc["vocab"] == expected["vocab_size"], str(enc["vocab"]))
        ratios = {}
        for c in LANGS:
            n = enc["count"](corpus[c])
            ratios[c] = n / units[c]
            row = exp["rows"][c]
            check(f"{vid}/{c}: tokens == published", n == row["tokens"],
                  f"recomputed {n} vs published {row['tokens']}")
            check(f"{vid}/{c}: faithful units == published", units[c] == row["units"])
            check(f"{vid}/{c}: ratio == published",
                  abs(ratios[c] - row["ratio"]) < 1e-9, f"{ratios[c]:.6f}")
            check(f"{vid}/{c}: ratio <= 1.2", ratios[c] <= 1.2)
        spread = max(ratios.values()) - min(ratios.values())
        score = 1000.0 / spread
        check(f"{vid}: spread == published", abs(spread - exp["spread"]) < 1e-9, f"{spread:.6f}")
        check(f"{vid}: score == published", abs(score - exp["score"]) < 1e-6, f"{score:.2f}")
        for c in LANGS:
            ok = enc["rt"](corpus[c])
            check(f"{vid}/{c}: corpus round-trip matches published behaviour",
                  ok == exp["rt_corpus"][c], f"exact={ok}")
        for p in probes:
            ok = enc["rt"](p)
            check(f"{vid}: probe round-trip matches published ({p[:28]!r})",
                  ok == exp["rt_probes"][p], f"exact={ok}")

    # results.json (the V3 metrics file) must agree with expected_metrics.json
    print("\nCross-file consistency")
    v3 = expected["versions"]["v3"]
    check("results.json score == expected v3 score",
          abs(results["faithful"]["score"] - v3["score"]) < 1e-9)
    check("results.json tokens == expected v3 tokens",
          all(l["total_tokens"] == v3["rows"][l["code"]]["tokens"] for l in results["languages"]))

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + '; '.join(FAILURES)}")
    for vid in ("v1", "v2", "v3"):
        e = expected["versions"][vid]
        star = " *" if vid == expected["submitted"] else ""
        print(f"  {vid.upper()}{star}: score {e['score']:.2f}  spread {e['spread']:.4f}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
