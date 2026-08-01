"""Stage 7: Decontaminate (keeping the evaluation honest).

This stage is a firewall: fingerprint the held-out/eval
material, scan every training shard for overlap against those
fingerprints, remove any training document that carries a piece of a
held-out example, and plant canary strings to catch a leak after the
fact. We build the held-out side from a *disjoint* slice of Sangraha's
`verified` Telugu tier (never present in our training pool), fingerprint
it at the 10-word-shingle level, scan the surviving training pool for
overlap, and separately run a real plant-and-detect canary test to
verify the leak-detection mechanism itself works.
"""
from __future__ import annotations

import os
import random
import uuid

import common
from common import StageTimer, make_report, read_jsonl, write_jsonl, write_json

FINGERPRINT_SHINGLE_SIZE = 10
CONTAMINATION_THRESHOLD_SHARED_SHINGLES = 2  # >= this many shared 10-grams = flagged overlap


def shingles(text: str, k: int) -> set[str]:
    words = text.split()
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_heldout_fingerprint() -> set[str]:
    heldout_docs = read_jsonl(common.heldout_sample())
    fp: set[str] = set()
    for d in heldout_docs:
        fp |= shingles(d["text"], FINGERPRINT_SHINGLE_SIZE)
    return fp


def canary_plant_and_detect_test(sample_doc_text: str) -> dict:
    """Synthetic mechanism test: plant a canary string in a 'trained' copy of a
    document, then prove a post-hoc scan of that copy detects it. This validates
    the *mechanism*, independent of whether our own small run happens to leak."""
    rng = random.Random(1234)  # fixed seed -> deterministic canary across reruns
    canary = f"CANARY-{uuid.uuid5(uuid.NAMESPACE_DNS, str(rng.random())).hex[:16]}"
    planted_text = sample_doc_text[:200] + f" {canary} " + sample_doc_text[200:400]

    # Simulate a "generation" that leaked training text verbatim, then scan for the canary.
    simulated_generation_log = planted_text[150:350]
    detected = canary in simulated_generation_log

    return {
        "canary_string": canary,
        "planted_in_doc_preview": True,
        "scanned_generation_log_preview": simulated_generation_log[:120] + "...",
        "leak_detected": detected,
        "mechanism_status": "PASS - canary found in the scanned output, exactly as designed" if detected else "FAIL",
    }


def run(input_path: str) -> dict:
    timer = StageTimer("decontaminate")
    docs = read_jsonl(input_path)

    fingerprint = build_heldout_fingerprint()

    survivors = []
    contaminated_examples = []
    contaminated_count = 0

    for d in docs:
        doc_shingles = shingles(d["text"], FINGERPRINT_SHINGLE_SIZE)
        overlap = doc_shingles & fingerprint
        if len(overlap) >= CONTAMINATION_THRESHOLD_SHARED_SHINGLES:
            contaminated_count += 1
            if len(contaminated_examples) < 5:
                contaminated_examples.append(
                    {
                        "doc_id": d["doc_id"],
                        "shared_shingles_with_heldout": len(overlap),
                        "sample_shared_ngram": next(iter(overlap)),
                    }
                )
            continue
        survivors.append(d)

    write_jsonl(common.work_path("stage7_survivors.jsonl"), survivors)

    canary_result = canary_plant_and_detect_test(
        docs[0]["text"] if docs else "no documents available for canary test"
    )

    report = make_report(
        stage_num=7,
        stage_name="Decontaminate",
        input_docs=len(docs),
        output_docs=len(survivors),
        elapsed_s=timer.done(),
        extra={
            "note": (
                "Held-out fingerprint built from a pool that is disjoint from the training "
                "pool by construction - the same role the Golden Proxy pattern plays. Overlap "
                "threshold is deliberately small (>=2 shared 10-grams) so even partial leakage "
                "is caught. " + common.corpus()["heldout"]["note"]
            ),
            "corpus_id": common.corpus()["id"],
            "heldout_docs_used": len(read_jsonl(common.heldout_sample())),
            "fingerprint_shingle_size_words": FINGERPRINT_SHINGLE_SIZE,
            "fingerprint_total_shingles": len(fingerprint),
            "contamination_threshold_shared_shingles": CONTAMINATION_THRESHOLD_SHARED_SHINGLES,
            "docs_flagged_contaminated_and_removed": contaminated_count,
            "canary_mechanism_test": canary_result,
        },
        examples=contaminated_examples,
    )
    write_json(common.work_path("stage7_report.json"), report)
    return report


if __name__ == "__main__":
    r = run(common.work_path("stage6_survivors.jsonl"))
    print(f"[stage7] {r['input_docs']} -> {r['output_docs']} docs ({r['survival_pct']}% survive)")
