# -*- coding: utf-8 -*-
"""Shard manifests and the admission gate.

A manifest is a shard's identity card. The course page lists what it must
record; this adds the few fields the curriculum needs (`curriculum_stage`,
`indic_tier`, `epoch_count`) and the ones session 4 established
(`pii_status`, per-file cleaning hashes).

The gate is the point of the whole exercise. A shard with no tokenizer hash has
token ids that mean nothing in particular; a shard with unknown cleaning
lineage cannot be shown to have been screened; a shard overlapping evaluation
data poisons every benchmark it touches. Any of those is a refusal, not a
warning -- `validate()` returns reasons and the caller drops the shard.

Timestamps deliberately do **not** appear in the hashed body. Two identical
runs must produce identical manifests, and a clock value would break that for a
reason unrelated to correctness. Wall-clock lives in ``run_meta.json``.
"""
from __future__ import annotations

from .hashing import hash_obj

# A shard missing any of these is not admitted. These are the minimum the
# lecture names: "we are not going to train on a shard that does not have the
# hash, the cleaning hash, dedup, eval and PII status."
REQUIRED_FIELDS = [
    "shard_id",
    "content_sha256",
    "tokenizer_hash",
    "cleaning_pipeline_hash",
    "dedup_status",
    "eval_overlap_status",
    "pii_status",
    "license",
    "capability_lane",
]

ALLOWED_LICENSES = {
    "CC-BY-SA-4.0",
    "CC-BY-4.0",
    "course-work (author-retained)",
    "public-domain",
}


def build_manifest(shard: dict, *, tokenizer_hash: str, cleaning_hash: str,
                   cleaning_per_file: dict, dedup_report: dict, pii_report: dict,
                   curriculum_stage: str, eval_overlap: dict,
                   parent_shard_ids: list[str] | None = None) -> dict:
    spans = shard["spans"]
    langs: dict[str, int] = {}
    scripts: dict[str, int] = {}
    tiers: dict[str, int] = {}
    licenses: dict[str, int] = {}
    for s in spans:
        if s.get("language"):
            langs[s["language"]] = langs.get(s["language"], 0) + 1
        if s.get("script"):
            scripts[s["script"]] = scripts.get(s["script"], 0) + 1
        if s.get("indic_tier"):
            tiers[s["indic_tier"]] = tiers.get(s["indic_tier"], 0) + 1
        if s.get("license"):
            licenses[s["license"]] = licenses.get(s["license"], 0) + 1

    body = {
        "shard_id": shard["shard_id"],
        "content_sha256": shard["content_sha256"],
        "spans_sha256": shard["spans_sha256"],
        "tokenizer_hash": tokenizer_hash,
        "capability_lane": shard["lane"],
        "curriculum_stage": curriculum_stage,
        "token_count": shard["n_tokens"],
        "document_count": shard["n_documents"],
        "document_ids": sorted(shard["doc_ids"]),
        "source_ids": sorted({s.get("source_file") for s in spans if s.get("source_file")}),
        "language_breakdown": dict(sorted(langs.items())),
        "script_breakdown": dict(sorted(scripts.items())),
        "indic_tier_breakdown": dict(sorted(tiers.items())),
        "license": (sorted(licenses)[0] if len(licenses) == 1
                    else "MIXED:" + ",".join(sorted(licenses))),
        "license_breakdown": dict(sorted(licenses.items())),
        "provenance_tier": "vendored-corpus",
        "cleaning_pipeline_hash": cleaning_hash,
        "cleaning_pipeline_per_file": cleaning_per_file,
        # Measured, not asserted -- see dedup.py and pii.py.
        "dedup_status": {
            "status": "DEDUPLICATED",
            "method": dedup_report["method"],
            "exact_removed": dedup_report["exact_duplicates_removed"],
            "near_removed": dedup_report["near_duplicates_removed"],
        },
        "pii_status": {
            "status": "SCREENED",
            "method": pii_report["method"],
            "documents_redacted": pii_report["documents_redacted"],
            "matches_by_category": pii_report["matches_by_category"],
        },
        "contamination_status": eval_overlap["contamination_status"],
        "eval_overlap_status": eval_overlap["eval_overlap_status"],
        "eval_overlap_detail": eval_overlap.get("detail", {}),
        "parent_shard_ids": sorted(parent_shard_ids or []),
        "epoch_count": 0.0,
        "never_train": False,
    }
    body["manifest_sha256"] = hash_obj(body)
    return body


def validate(manifest: dict) -> tuple[bool, list[str]]:
    """The admission gate. Returns ``(admitted, reasons)``."""
    reasons: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in manifest or manifest[field] in (None, "", {}):
            reasons.append(f"missing required field: {field}")

    if manifest.get("never_train"):
        reasons.append("shard is flagged never_train")

    lic = manifest.get("license", "")
    if lic.startswith("MIXED:"):
        for part in lic[len("MIXED:"):].split(","):
            if part and part not in ALLOWED_LICENSES:
                reasons.append(f"license not permitted for training: {part}")
    elif lic and lic not in ALLOWED_LICENSES:
        reasons.append(f"license not permitted for training: {lic}")

    if manifest.get("eval_overlap_status") not in (None, "CLEAN"):
        reasons.append(f"evaluation overlap: {manifest.get('eval_overlap_status')}")
    if manifest.get("contamination_status") not in (None, "CLEAN"):
        reasons.append(f"contamination: {manifest.get('contamination_status')}")

    if not manifest.get("token_count"):
        reasons.append("shard contains no tokens")

    # The manifest must not have been edited after it was sealed.
    if "manifest_sha256" in manifest:
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if hash_obj(body) != manifest["manifest_sha256"]:
            reasons.append("manifest_sha256 does not match the manifest body")

    return (not reasons), reasons


def summarise(manifests: list[dict]) -> dict:
    by_lane: dict[str, dict] = {}
    for m in manifests:
        lane = m["capability_lane"]
        e = by_lane.setdefault(lane, {"shards": 0, "tokens": 0, "documents": 0})
        e["shards"] += 1
        e["tokens"] += m["token_count"]
        e["documents"] += m["document_count"]
    return {
        "shards": len(manifests),
        "tokens": sum(m["token_count"] for m in manifests),
        "documents": sum(m["document_count"] for m in manifests),
        "by_lane": dict(sorted(by_lane.items())),
    }
