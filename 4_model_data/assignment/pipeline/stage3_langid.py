"""Stage 3: Language ID and validation.

The standard warning is specific: trusting the folder/label a document
arrived in ("everything under tel/ is Telugu") quietly pollutes the
per-language pool, and the exact bug it calls out is a two-letter code
("te") used where a three-letter code ("tel") was expected. Sangraha's
own folder layout uses the three-letter code, so we detect each
document's language at runtime with py3langid (which speaks the
two-letter ISO 639-1 family) and translate through an explicit mapping
table rather than assuming they line up - which is exactly the fix,
not just a callout.
"""
from __future__ import annotations

import os

import py3langid

import common
from common import StageTimer, make_report, read_jsonl, write_jsonl, write_json

# py3langid speaks ISO 639-1; Sangraha's folder names speak ISO 639-3.
# This table *is* the te/tel fix - translate, don't assume they match.
ISO_639_1_TO_3 = {
    "te": "tel", "en": "eng", "hi": "hin", "ta": "tam", "kn": "kan",
    "ml": "mal", "bn": "ben", "ur": "urd", "mr": "mar", "gu": "guj",
    "pa": "pan", "or": "ori", "as": "asm", "ne": "nep", "sa": "san",
    "sd": "snd", "ks": "kas",
}

MIN_CHARS_FOR_CONFIDENT_ID = 40  # below this, short/boilerplate docs are unreliable to judge
MAX_CHARS_FOR_LANGID = 4000  # langid needs only a prefix; very long docs overflow its uint16 feature counts


def run(input_path: str) -> dict:
    timer = StageTimer("langid")
    cfg = common.corpus()
    corpus_claimed = cfg["claimed_lang"]
    docs = read_jsonl(input_path)

    survivors = []
    mismatches = []
    too_short_to_judge = 0
    match_count = 0
    detected_lang_counts: dict[str, int] = {}

    for d in docs:
        text = d["text"]
        claimed = d.get("claimed_lang") or corpus_claimed

        if len(text) < MIN_CHARS_FOR_CONFIDENT_ID:
            too_short_to_judge += 1
            d2 = dict(d)
            d2["langid_detected"] = None
            d2["langid_status"] = "too_short_to_judge_kept"
            survivors.append(d2)
            continue

        detected_2letter, logprob = py3langid.classify(text[:MAX_CHARS_FOR_LANGID])
        detected_3letter = ISO_639_1_TO_3.get(detected_2letter, detected_2letter)
        detected_lang_counts[detected_3letter] = detected_lang_counts.get(detected_3letter, 0) + 1

        is_match = detected_3letter == claimed
        d2 = dict(d)
        d2["langid_detected"] = detected_3letter
        d2["langid_logprob"] = round(float(logprob), 2)

        if is_match:
            match_count += 1
            d2["langid_status"] = "match"
            survivors.append(d2)
        else:
            # Confident mismatch -> drop from the pool (mislabeled document).
            d2["langid_status"] = "mismatch_dropped"
            if len(mismatches) < 12:
                mismatches.append(
                    {
                        "doc_id": d["doc_id"],
                        "claimed_lang": claimed,
                        "detected_lang": detected_3letter,
                        "logprob": round(float(logprob), 2),
                        "text_preview": text[:160],
                    }
                )

    write_jsonl(common.work_path("stage3_survivors.jsonl"), survivors)

    report = make_report(
        stage_num=3,
        stage_name="Language ID",
        input_docs=len(docs),
        output_docs=len(survivors),
        elapsed_s=timer.done(),
        extra={
            "note": (
                "Runtime detection via py3langid (ISO 639-1) translated through an explicit "
                "639-1 -> 639-3 map before comparing against the folder-claimed label, "
                "which is the concrete fix for the te/tel code-mismatch bug."
            ),
            "corpus_id": cfg["id"],
            "claimed_lang": corpus_claimed,
            "docs_matching_claim": match_count,
            "docs_too_short_to_judge_kept_asis": too_short_to_judge,
            "docs_mismatched_and_dropped": len(docs) - len(survivors),
            "detected_language_distribution": detected_lang_counts,
        },
        examples=mismatches,
    )
    write_json(common.work_path("stage3_report.json"), report)
    return report


if __name__ == "__main__":
    r = run(common.work_path("stage2_survivors.jsonl"))
    print(f"[stage3] {r['input_docs']} -> {r['output_docs']} docs ({r['survival_pct']}% survive)")
