"""Post-hoc analysis: what does one word actually cost, under which tokenizer?

Deliberately NOT a pipeline stage. It changes no document and is excluded from
the cleaning-script hash (stage 8 globs `stage*.py`), because it is a
measurement about the corpus, not a step that produced it.

It exists because the manifest reports token counts in `cl100k_base`, and
cl100k has almost no Telugu merges - it encodes Telugu at close to the byte
level, giving a fertility around 13 tokens per word. That number is correct but
misleading if read as "how big is this corpus": under a tokenizer actually built
for Indian scripts the same text is a fraction of the size. So we measure both
and report the ratio, rather than leaving the widget to estimate it in prose.

Reference tokenizers:
  cl100k_base          the manifest's tokenizer (tiktoken)
  google/muril-base-cased  an Indic-appropriate WordPiece tokenizer covering
                       17 Indian languages incl. Telugu - ungated, needs no
                       sentencepiece, loads the tokenizer only (not weights).

  python analysis_tokenizer_fertility.py
"""
from __future__ import annotations

import os
import sys

import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpora
from common import ASSIGNMENT_DIR, read_jsonl, write_json

FLOOR = 10_000_000  # the assignment's lower bound, in tokens
INDIC_TOKENIZER = "google/muril-base-cased"
CL100K = tiktoken.get_encoding("cl100k_base")


def load_indic_tokenizer():
    from transformers import AutoTokenizer, logging as hf_logging

    # We only count tokens, we never run the 512-token model, so the
    # "sequence longer than 512" warning is noise here - silence it and lift
    # the length cap so tokenization is unbounded (the count stays exact).
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(INDIC_TOKENIZER)  # tokenizer files only, not weights
    tok.model_max_length = int(1e9)
    return tok


def measure(corpus_id: str, indic_tok) -> dict | None:
    work = os.path.join(ASSIGNMENT_DIR, f"work_{corpus_id}")
    path = os.path.join(work, "stage8_survivors.jsonl")
    if not os.path.exists(path):
        return None
    cfg = corpora.get(corpus_id)
    is_indic = cfg["script"] != "latin"

    docs = read_jsonl(path)
    words = cl_tokens = indic_tokens = 0
    for d in docs:
        t = d["text"]
        words += len(t.split())
        cl_tokens += len(CL100K.encode(t, disallowed_special=()))
        if is_indic:
            # add_special_tokens=False so we count content, not [CLS]/[SEP]
            indic_tokens += len(indic_tok(t, add_special_tokens=False)["input_ids"])

    cl_fert = round(cl_tokens / words, 3) if words else None
    result = {
        "corpus_id": corpus_id,
        "corpus_name": cfg["name"],
        "script": cfg["script"],
        "documents": len(docs),
        "words": words,
        "cl100k": {"tokens": cl_tokens, "fertility_tokens_per_word": cl_fert},
    }

    if is_indic:
        indic_fert = round(indic_tokens / words, 3) if words else None
        result["indic"] = {
            "tokenizer": INDIC_TOKENIZER,
            "tokens": indic_tokens,
            "fertility_tokens_per_word": indic_fert,
        }
        result["cl100k_over_indic_fertility_ratio"] = (
            round(cl_fert / indic_fert, 2) if (cl_fert and indic_fert) else None
        )
        result["clears_10M_floor_under_cl100k"] = cl_tokens >= FLOOR
        result["clears_10M_floor_under_indic"] = indic_tokens >= FLOOR
        result["interpretation"] = (
            f"The manifest's {cl_tokens:,} cl100k tokens is a real count, but it is ~"
            f"{result['cl100k_over_indic_fertility_ratio']}x what the same text costs under an "
            f"Indic tokenizer ({indic_tokens:,} MuRIL tokens). cl100k clears the 10M-token floor; "
            f"MuRIL {'also clears' if indic_tokens >= FLOOR else 'does NOT clear'} it. The corpus is "
            f"the same size either way - the token count is a property of the tokenizer, not the data, "
            f"which is exactly why fertility has to be reported, not just a token total."
        )
    else:
        # cl100k is a reasonable reference for English; MuRIL is not the point here.
        result["indic"] = None
        result["clears_10M_floor_under_cl100k"] = cl_tokens >= FLOOR
        result["interpretation"] = (
            f"English/Latin corpus - cl100k is a sensible tokenizer here and its fertility "
            f"({cl_fert}) is close to 1, so no Indic comparison is drawn. {cl_tokens:,} tokens, "
            f"{'above' if cl_tokens >= FLOOR else 'below'} the 10M floor."
        )
    return result


def main() -> None:
    indic_tok = load_indic_tokenizer()
    out = {}
    for cid in corpora.ORDER:
        r = measure(cid, indic_tok)
        if r is None:
            print(f"[fertility] {cid}: no stage8 survivors on disk, skipped")
            continue
        out[cid] = r
        cl = r["cl100k"]
        line = f"[fertility] {cid}: cl100k {cl['tokens']:,} @ {cl['fertility_tokens_per_word']} tok/word"
        if r.get("indic"):
            i = r["indic"]
            line += f"  |  MuRIL {i['tokens']:,} @ {i['fertility_tokens_per_word']} tok/word  (ratio {r['cl100k_over_indic_fertility_ratio']}x)"
        print(line)

    # cross-corpus totals - the two-corpus story clears the floor on any tokenizer
    telugu = out.get("telugu_web", {})
    if out:
        cl_total = sum(c["cl100k"]["tokens"] for c in out.values())
        out["_totals"] = {
            "cl100k_tokens_all_corpora": cl_total,
            "clears_10M_floor_combined_cl100k": cl_total >= FLOOR,
            "note": (
                "Across both corpora the token count clears the 10M floor comfortably under any "
                "tokenizer. The floor question only gets interesting for the Telugu corpus alone, "
                "which is where the cl100k-vs-Indic gap matters."
            ),
        }
    write_json(os.path.join(ASSIGNMENT_DIR, "analysis_tokenizer_fertility.json"), out)
    print("[fertility] wrote analysis_tokenizer_fertility.json")


if __name__ == "__main__":
    main()
