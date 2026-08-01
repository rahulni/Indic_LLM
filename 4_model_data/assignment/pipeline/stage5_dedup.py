"""Stage 5: Deduplicate.

Global, not per-shard - every surviving document is compared against
the whole pool, in one pass, which is the exact mechanism standard practice
walks through: shingle each document, summarize the shingle set with a
MinHash signature, and use LSH banding to only compare documents that
land in the same bucket instead of an all-pairs O(n^2) comparison.
Sangraha's own `unverified` tier is the corpus the course's own audit
names as having had zero dedup - so this stage is a direct callback,
run for real rather than simulated.
"""
from __future__ import annotations

import os

from datasketch import MinHash, MinHashLSH

import common
from common import StageTimer, make_report, read_jsonl, write_jsonl, write_json, sha256_text

NUM_PERM = 128
LSH_THRESHOLD = 0.75
SHINGLE_SIZE = 5  # word-level 5-grams
PROBE_THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]


def word_shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_minhash(shingles: set[str]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def threshold_sensitivity_probe() -> list[dict]:
    """Real-document mechanism check, independent of whether these specific
    documents survive to this stage in the live pipeline. A random ~8% sample
    of a 150k-document shard has low odds of containing *both* halves of any
    one duplicate pair, so we deliberately seeded a handful of documents found
    via a prefix-collision scan of the full shard (see raw_sample/telugu_raw.jsonl,
    `included_reason`) and report their true similarity here - this is what a
    MinHash/LSH threshold probe demonstrates: the same pair can be a "catch"
    or a "miss" purely depending on where the threshold is set."""
    seeded = [d for d in read_jsonl(common.raw_sample()) if d.get("included_reason")]
    groups: dict[str, list[dict]] = {}
    for d in seeded:
        groups.setdefault(d["text"][:120], []).append(d)

    all_probes = []
    for prefix, members in groups.items():
        if len(members) < 2:
            continue
        # every pair within the group, not just the first two - more real candidates
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                sa, sb = word_shingles(a["text"]), word_shingles(b["text"])
                if not (sa or sb):
                    continue
                true_jaccard = len(sa & sb) / len(sa | sb)
                mha, mhb = build_minhash(sa), build_minhash(sb)
                est_jaccard = mha.jaccard(mhb)
                verdicts = {str(t): bool(est_jaccard >= t) for t in PROBE_THRESHOLDS}
                all_probes.append(
                    {
                        "doc_id_a": a["doc_id"],
                        "doc_id_b": b["doc_id"],
                        "shared_prefix": prefix[:60],
                        "words_a": len(a["text"].split()),
                        "words_b": len(b["text"].split()),
                        "true_jaccard_shingle_sets": round(true_jaccard, 4),
                        "estimated_jaccard_minhash": round(float(est_jaccard), 4),
                        "verdict_by_threshold": verdicts,
                    }
                )

    # Surface the most illustrative real cases: sorted by true similarity, descending,
    # so the highest-overlap (most borderline-to-threshold) genuine pairs lead.
    all_probes.sort(key=lambda p: p["true_jaccard_shingle_sets"], reverse=True)
    return all_probes[:8]


def natural_threshold_probe(near_pairs: list[dict], docs: list[dict]) -> list[dict]:
    """"Would this pair be caught at threshold T?" - answered against pairs the
    corpus really contains. Sorted by true similarity so the borderline cases,
    the ones where the threshold choice actually decides the outcome, lead."""
    by_id = {d["doc_id"]: d for d in docs}
    probes = []
    for n in near_pairs:
        a = by_id.get(n["doc_id"])
        b = by_id.get(n["near_duplicate_of"])
        if not (a and b):
            continue
        est = n["estimated_jaccard_minhash"]
        probes.append(
            {
                "doc_id_a": n["doc_id"],
                "doc_id_b": n["near_duplicate_of"],
                "source_a": (a.get("source_key") or a.get("shard")),
                "source_b": (b.get("source_key") or b.get("shard")),
                "cross_source": (a.get("source_key") != b.get("source_key")),
                "shared_prefix": a["text"][:60],
                "words_a": len(a["text"].split()),
                "words_b": len(b["text"].split()),
                "true_jaccard_shingle_sets": n["true_jaccard_shingle_sets"],
                "estimated_jaccard_minhash": est,
                "verdict_by_threshold": {str(t): bool(est >= t) for t in PROBE_THRESHOLDS},
            }
        )
    # Borderline first: closest to the operating threshold is the most informative.
    probes.sort(key=lambda p: abs(p["true_jaccard_shingle_sets"] - LSH_THRESHOLD))
    return probes[:8]


def dedup_pass(docs: list[dict]) -> dict:
    """One complete exact + near duplicate pass over whatever pool it is given.

    Used twice: once over the whole corpus (the global pass that actually
    produces survivors), and once per source shard in isolation. Running the
    identical code both ways is the point - the difference in what they remove
    is not an artifact of two different implementations, it is the cost of not
    having a global view."""
    docs = sorted(docs, key=lambda d: d["doc_id"])

    seen_hashes: dict[str, str] = {}
    exact_dupes: list[dict] = []
    after_exact: list[dict] = []
    for d in docs:
        h = sha256_text(d["text"])
        if h in seen_hashes:
            exact_dupes.append(
                {
                    "doc_id": d["doc_id"],
                    "duplicate_of": seen_hashes[h],
                    "source_key": d.get("source_key"),
                }
            )
        else:
            seen_hashes[h] = d["doc_id"]
            after_exact.append(d)

    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    minhashes: dict[str, MinHash] = {}
    shingle_cache: dict[str, set[str]] = {}
    survivors: list[dict] = []
    near_pairs: list[dict] = []

    for d in after_exact:
        shingles = word_shingles(d["text"])
        mh = build_minhash(shingles)
        matches = lsh.query(mh)
        if matches:
            rep_id = matches[0]
            rep_shingles = shingle_cache[rep_id]
            true_j = (
                len(shingles & rep_shingles) / len(shingles | rep_shingles)
                if (shingles or rep_shingles)
                else 0.0
            )
            near_pairs.append(
                {
                    "doc_id": d["doc_id"],
                    "near_duplicate_of": rep_id,
                    "source_key": d.get("source_key"),
                    "estimated_jaccard_minhash": round(float(mh.jaccard(minhashes[rep_id])), 4),
                    "true_jaccard_shingle_sets": round(true_j, 4),
                    "text_a": d["text"],
                    "text_b": None,
                }
            )
            continue
        lsh.insert(d["doc_id"], mh)
        minhashes[d["doc_id"]] = mh
        shingle_cache[d["doc_id"]] = shingles
        survivors.append(d)

    return {
        "survivors": survivors,
        "exact_dupes": exact_dupes,
        "near_pairs": near_pairs,
        "removed": len(docs) - len(survivors),
        "lsh_b": lsh.b,
        "lsh_r": lsh.r,
    }


def local_vs_global(docs: list[dict], global_removed: int) -> dict:
    """Standard practice's global-deduplication argument, measured instead of asserted.

    Every source shard is deduplicated on its own, exactly as a student working
    alone on their own shard would do it. The shards then look clean. The global
    pass over the merged pool removes more - and the gap is precisely the
    duplication that no local pass could ever have seen."""
    by_source: dict[str, list[dict]] = {}
    for d in docs:
        by_source.setdefault(d.get("source_key") or d.get("shard") or "single_shard", []).append(d)

    per_source = []
    local_total_removed = 0
    for key in sorted(by_source):
        res = dedup_pass(by_source[key])
        local_total_removed += res["removed"]
        per_source.append(
            {
                "source_key": key,
                "docs": len(by_source[key]),
                "removed_by_local_pass": res["removed"],
                "exact": len(res["exact_dupes"]),
                "near": len(res["near_pairs"]),
            }
        )

    return {
        "shards_deduplicated_independently": per_source,
        "total_removed_if_only_local_dedup": local_total_removed,
        "total_removed_by_global_pass": global_removed,
        "duplicates_only_a_global_pass_can_find": global_removed - local_total_removed,
        "note": (
            "Each shard above was deduplicated in isolation with the identical code the global "
            "pass uses. The final row is the whole argument for one large-memory "
            "machine holding the entire corpus at once: those documents are invisible to every "
            "local pass, no matter how carefully each shard owner does their job."
        ),
    }


def run(input_path: str) -> dict:
    timer = StageTimer("deduplicate")
    docs = read_jsonl(input_path)
    # Deterministic global ordering, independent of upstream file order.
    docs.sort(key=lambda d: d["doc_id"])

    # The real, corpus-wide pass. Same code path the per-shard comparison uses.
    result = dedup_pass(docs)
    survivors = result["survivors"]
    exact_dupes = result["exact_dupes"]
    near_pairs = result["near_pairs"]

    write_jsonl(common.work_path("stage5_survivors.jsonl"), survivors)

    b, r = result["lsh_b"], result["lsh_r"]

    # How many of the removals crossed a source boundary? Those are exactly the
    # ones no per-shard pass could have caught.
    id_to_source = {d["doc_id"]: d.get("source_key") for d in docs}
    cross_source_exact = sum(
        1 for e in exact_dupes
        if id_to_source.get(e["doc_id"]) != id_to_source.get(e["duplicate_of"])
    )
    cross_source_near = sum(
        1 for n in near_pairs
        if id_to_source.get(n["doc_id"]) != id_to_source.get(n["near_duplicate_of"])
    )

    scale_comparison = local_vs_global(docs, result["removed"])

    # Threshold sensitivity, measured on pairs this run actually found rather
    # than on candidates seeded in from outside. If the corpus genuinely
    # contains no near-duplicates, that is reported instead of manufactured.
    sensitivity_probes = natural_threshold_probe(near_pairs, docs)
    seeded_probes = threshold_sensitivity_probe()

    report = make_report(
        stage_num=5,
        stage_name="Deduplicate",
        input_docs=len(docs),
        output_docs=len(survivors),
        elapsed_s=timer.done(),
        extra={
            "note": (
                "Global pass over the whole surviving pool (not per-shard), exactly the "
                "mechanism standard practice names as missing from Sangraha's original ingestion."
            ),
            "method": "MinHash + LSH banding",
            "num_perm": NUM_PERM,
            "lsh_threshold": LSH_THRESHOLD,
            "lsh_bands_b": b,
            "lsh_rows_per_band_r": r,
            "shingle_size_words": SHINGLE_SIZE,
            "exact_duplicates_removed": len(exact_dupes),
            "near_duplicates_removed": len(near_pairs),
            "total_removed": result["removed"],
            "cross_source_exact_duplicates": cross_source_exact,
            "cross_source_near_duplicates": cross_source_near,
            "scale_local_vs_global": scale_comparison,
            "threshold_sensitivity_probe": sensitivity_probes,
            "threshold_probe_note": (
                "Measured on near-duplicate pairs this run actually found in the corpus, not on "
                "candidates seeded in from outside. The sample is drawn as a contiguous slice "
                "precisely so that duplicate pairs survive into it - a uniform random sub-sample "
                "keeps both halves of a given pair with probability p-squared (~0.6% at p=0.08), "
                "which is why an earlier random draw of this same shard found almost nothing."
            ),
            "seeded_probe_legacy": seeded_probes,
        },
        examples=exact_dupes[:6] + [
            {k: v for k, v in n.items() if k not in ("text_a", "text_b")} for n in near_pairs[:4]
        ],
    )
    write_json(common.work_path("stage5_report.json"), report)
    return report


if __name__ == "__main__":
    r = run(common.work_path("stage4_survivors.jsonl"))
    print(f"[stage5] {r['input_docs']} -> {r['output_docs']} docs ({r['survival_pct']}% survive)")
