"""Stage 8: Manifest (reproducibility and provenance).

Emits one provenance record for the final surviving shard: source,
license, contributor, the cleaning script's own hash (so a later run
can prove the code hasn't changed), a content hash computed from the
cleaned text itself (not a running counter - deterministic by
construction), a real token count, and the language breakdown. No text
is removed here; a shard with unknown provenance would simply be
blocked from shipping.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import os

import tiktoken

import common
from common import PIPELINE_DIR, StageTimer, make_report, read_jsonl, write_jsonl, write_json, sha256_file

CONTRIBUTOR = "Indic data-cleaning pipeline"

# Licenses the corpus may ship under and still be trainable. Anything outside
# this set - including a missing license - blocks the shard. Best practice is
# explicit that an unknown license is a blocking condition, not a warning.
ALLOWED_LICENSES = {"CC-BY-4.0", "cc-by-4.0", "apache-2.0", "mit", "cc0-1.0", "odc-by"}
# Copyleft licenses are not "unknown", but they are not safely trainable either
# without a deliberate legal decision, so they get their own status rather than
# being quietly waved through with the permissive ones.
COPYLEFT_LICENSES = {"agpl-3.0", "gpl-3.0", "gpl-2.0", "lgpl-3.0"}


def license_status(license_value: str | None) -> tuple[str, str]:
    """Returns (status, reason). This is the provenance gate."""
    if not license_value:
        return "BLOCKED", "No license declared by the publisher - unknown provenance cannot ship."
    if license_value in COPYLEFT_LICENSES:
        return (
            "BLOCKED",
            f"{license_value} is copyleft; training on it without a deliberate legal decision "
            "would put the whole corpus's licensing in question.",
        )
    if license_value.lower() in {l.lower() for l in ALLOWED_LICENSES}:
        return "CLEAN", f"{license_value} permits training use with attribution."
    return "BLOCKED", f"{license_value} is not on the allow-list and has not been reviewed."


def cleaning_script_hash() -> dict:
    """Hash every stage script that actually ran, so a later re-run can prove
    the code producing this manifest hasn't changed - the same check widget 9
    demonstrates. Sorted glob -> deterministic across OS/filesystem ordering."""
    stage_files = sorted(
        glob.glob(os.path.join(PIPELINE_DIR, "stage*.py"))
        + [os.path.join(PIPELINE_DIR, "common.py"), os.path.join(PIPELINE_DIR, "corpora.py")]
    )
    per_file = {os.path.basename(p): sha256_file(p) for p in stage_files}
    combined = hashlib.sha256("".join(per_file[k] for k in sorted(per_file)).encode("utf-8")).hexdigest()
    return {"per_file_sha256": per_file, "combined_script_hash_sha256": combined}


def build_shard_manifests(docs: list[dict], enc, script_hash_info: dict, timestamp: str) -> list[dict]:
    """One manifest per source shard, which is what good provenance actually asks for -
    a single corpus-wide record cannot carry a per-shard license or block one
    source while letting the others through."""
    cfg = common.corpus()
    licenses = {s["key"]: s.get("license") for s in cfg["sources"]}
    hf_ids = {s["key"]: s["hf_id"] for s in cfg["sources"]}

    by_shard: dict[str, list[dict]] = {}
    for d in docs:
        by_shard.setdefault(d.get("source_key") or cfg["sources"][0]["key"], []).append(d)

    manifests = []
    for key in sorted(by_shard):
        shard_docs = sorted(by_shard[key], key=lambda d: d["doc_id"])
        text = "␟".join(d["text"] for d in shard_docs)
        tokens = sum(d.get("n_tokens_final_cl100k", 0) for d in shard_docs)
        words = sum(len(d["text"].split()) for d in shard_docs)
        lang_breakdown: dict[str, int] = {}
        for d in shard_docs:
            lang = d.get("langid_detected") or d.get("claimed_lang") or cfg["claimed_lang"]
            lang_breakdown[lang] = lang_breakdown.get(lang, 0) + 1

        lic = licenses.get(key)
        status, reason = license_status(lic)
        manifests.append(
            {
                "shard_id": key,
                "source": hf_ids.get(key, key),
                "license": lic,
                "license_status": status,
                "license_reason": reason,
                "ships": status == "CLEAN",
                "contributor": CONTRIBUTOR,
                "cleaning_script_hash_sha256": script_hash_info["combined_script_hash_sha256"],
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "doc_count": len(shard_docs),
                "token_count_cl100k": tokens,
                "word_count": words,
                "fertility_ratio_tokens_per_word": round(tokens / words, 3) if words else None,
                "language_breakdown": lang_breakdown,
                "timestamp_utc": timestamp,
            }
        )
    return manifests


def run(input_path: str, timestamp_override: str | None = None) -> dict:
    timer = StageTimer("manifest")
    cfg = common.corpus()
    docs = read_jsonl(input_path)
    docs.sort(key=lambda d: d["doc_id"])  # deterministic ordering before hashing

    enc = tiktoken.get_encoding("cl100k_base")
    lang_breakdown: dict[str, int] = {}
    total_tokens = 0
    total_words = 0

    for d in docs:
        lang = d.get("langid_detected") or d.get("claimed_lang") or cfg["claimed_lang"]
        lang_breakdown[lang] = lang_breakdown.get(lang, 0) + 1
        n_tok = len(enc.encode(d["text"], disallowed_special=()))
        d["n_tokens_final_cl100k"] = n_tok
        total_tokens += n_tok
        total_words += len(d["text"].split())

    write_jsonl(common.work_path("stage8_survivors.jsonl"), docs)

    # Content hash computed from the final cleaned+sorted text itself, so re-running
    # the identical pipeline on the identical input reproduces an identical hash.
    concatenated = "␟".join(d["text"] for d in docs)  # unit-separator join, deterministic
    content_sha256 = hashlib.sha256(concatenated.encode("utf-8")).hexdigest()

    script_hash_info = cleaning_script_hash()
    timestamp = timestamp_override or datetime.datetime.now(datetime.timezone.utc).isoformat()

    fertility_ratio = round(total_tokens / total_words, 3) if total_words else None

    shard_manifests = build_shard_manifests(docs, enc, script_hash_info, timestamp)
    blocked = [m for m in shard_manifests if not m["ships"]]
    corpus_license = cfg["sources"][0].get("license") if len(cfg["sources"]) == 1 else "per-shard (see shard_manifests)"

    manifest = {
        "corpus_id": cfg["id"],
        "corpus_name": cfg["name"],
        "source": (
            f"{cfg['sources'][0]['hf_id']} ({cfg['sources'][0].get('data_files') or 'train'})"
            if len(cfg["sources"]) == 1
            else f"{len(cfg['sources'])} sources - see shard_manifests"
        ),
        "license": corpus_license,
        "shard_manifests": shard_manifests,
        "shards_total": len(shard_manifests),
        "shards_blocked": len(blocked),
        "shards_blocked_ids": [m["shard_id"] for m in blocked],
        "tokens_blocked_from_shipping": sum(m["token_count_cl100k"] for m in blocked),
        "gating_note": (
            "Every shard carries its own license and its own status. A BLOCKED shard is counted "
            "in the statistics above but does not ship - which is the point of the gate: the "
            "corpus is not a single undifferentiated blob, and one bad source should cost you "
            "that source, not the whole run."
        ),
        "contributor": CONTRIBUTOR,
        "cleaning_script_name": "cleaning pipeline (stages 0-8)",
        "cleaning_script_hash_sha256": script_hash_info["combined_script_hash_sha256"],
        "cleaning_script_per_file_hash": script_hash_info["per_file_sha256"],
        "content_sha256": content_sha256,
        "final_doc_count": len(docs),
        "final_token_count_cl100k": total_tokens,
        "final_word_count": total_words,
        "fertility_ratio_tokens_per_word": fertility_ratio,
        "language_breakdown": lang_breakdown,
        "timestamp_utc": timestamp,
    }
    write_json(common.work_path("manifest.json"), manifest)

    report = make_report(
        stage_num=8,
        stage_name="Manifest",
        input_docs=len(docs),
        output_docs=len(docs),  # nothing removed - this stage only adds provenance
        elapsed_s=timer.done(),
        extra={
            "note": (
                "No text is removed at this stage. The manifest is the object the gating rule "
                "enforces - a shard without one does not ship. content_sha256 and the token "
                "count are both computed fresh here, not carried over from any earlier estimate."
            ),
            "manifest": manifest,
        },
        examples=[],
    )
    write_json(common.work_path("stage8_report.json"), report)
    return report


if __name__ == "__main__":
    r = run(common.work_path("stage7_survivors.jsonl"))
    print(f"[stage8] manifest written; final tokens = {r['extra']['manifest']['final_token_count_cl100k']}")
