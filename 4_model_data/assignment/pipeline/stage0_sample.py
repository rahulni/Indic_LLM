"""Stage 0: Sample.

The provenance chain has to start at the download, not at a frozen file
somebody produced once and forgot the recipe for. This stage is the
recipe: it draws both corpora from Hugging Face, records exactly which
rows it took and in what order, cross-checks each source's declared
license against what the HF API reports live, and writes a sampling
report next to the data. Its hash is folded into the manifest along with
the other stage scripts, so "which code produced this corpus" has a
complete answer.

Sampling strategy is contiguous, deliberately. The first round of this
pipeline used a uniform random ~8% draw of a 150k-document shard and then
found essentially no duplicates - which is arithmetic, not luck: if you
keep each document independently with probability p, you keep both halves
of a given duplicate pair with probability p-squared, about 0.6% at
p=0.08. Near-duplicates in a crawl also cluster by site and crawl order,
so a contiguous slice keeps pairs together where a scattered draw pulls
them apart.

Run directly to re-draw:  python stage0_sample.py [corpus_id ...]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

import tiktoken

import common
import corpora
from common import StageTimer, write_json, sha256_text

ENC = tiktoken.get_encoding("cl100k_base")


def live_license(hf_id: str) -> str | None:
    """What the Hugging Face API says right now, independent of what our
    registry claims. A disagreement between the two is worth surfacing."""
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/datasets/{hf_id}",
            headers={"User-Agent": "era-v5-session4-assignment"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            info = json.load(r)
        return (info.get("cardData") or {}).get("license")
    except Exception:
        return None


def n_tokens(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def _stream(hf_id: str, data_files: str | None):
    from datasets import load_dataset

    kwargs = {"split": "train", "streaming": True}
    if data_files:
        kwargs["data_files"] = data_files
    return load_dataset(hf_id, **kwargs)


def draw_source(
    src: dict,
    target_tokens: int,
    skip_rows: int = 0,
    max_rows: int | None = None,
) -> tuple[list[dict], dict]:
    """Take a contiguous run of rows from the head of a source (after
    `skip_rows`), stopping at whichever of target_tokens / max_rows comes
    first. Returns the docs plus a provenance record of the draw."""
    text_field = src.get("text_field", "text")
    docs: list[dict] = []
    tokens = 0
    rows_scanned = 0
    rows_skipped_empty = 0

    ds = _stream(src["hf_id"], src.get("data_files"))
    for i, row in enumerate(ds):
        if i < skip_rows:
            continue
        rows_scanned += 1
        text = (row.get(text_field) or "").strip()
        if not text:
            rows_skipped_empty += 1
            continue
        tok = n_tokens(text)
        docs.append(
            {
                # Identity of the RECORD, not of the text. Two records from two
                # different sources can carry byte-identical text - that is the
                # cross-source duplication we want stage 5 to find, so it must
                # survive ingestion as two distinct records rather than collapse
                # here by accident.
                "doc_id": sha256_text(f"{src['key']}#{i}#{text[:256]}"),
                "text_sha256": sha256_text(text),
                "text": text,
                "claimed_lang": src.get("claimed_lang", ""),
                "source": src["hf_id"],
                "source_key": src["key"],
                "tier": src.get("data_files") or "train",
                "shard": src.get("data_files") or f"{src['hf_id']}:train",
                "source_row_index": i,
                "n_tokens_cl100k_raw": tok,
            }
        )
        tokens += tok
        if tokens >= target_tokens:
            break
        if max_rows is not None and len(docs) >= max_rows:
            break

    declared = src.get("license")
    live = live_license(src["hf_id"])
    record = {
        "source_key": src["key"],
        "hf_id": src["hf_id"],
        "data_files": src.get("data_files"),
        "draw_strategy": "contiguous_from_head" if not skip_rows else f"contiguous_after_row_{skip_rows}",
        "first_row_index": docs[0]["source_row_index"] if docs else None,
        "last_row_index": docs[-1]["source_row_index"] if docs else None,
        "rows_scanned": rows_scanned,
        "rows_skipped_empty": rows_skipped_empty,
        "docs_taken": len(docs),
        "tokens_cl100k": tokens,
        "license_declared_in_registry": declared,
        "license_reported_by_hf_api": live,
        "license_agrees": (declared or None) == (live or None),
        "license_status": "declared" if (declared or live) else "MISSING - manifest gating must block this shard",
    }
    return docs, record


def sample_corpus(corpus_id: str) -> dict:
    timer = StageTimer(f"sample:{corpus_id}")
    cfg = corpora.get(corpus_id)
    sampling = cfg["sampling"]
    target = sampling["target_tokens"]
    sources = cfg["sources"]

    all_docs: list[dict] = []
    source_records = []

    # Every source draws against its own budget, independently. A single shared
    # budget consumed in source order silently starves whatever is listed last,
    # which is how the first attempt at this ended up with zero rows from the
    # source that makes cross-source deduplication measurable.
    for src in sources:
        src = dict(src)
        src.setdefault("claimed_lang", cfg["claimed_lang"])
        docs, rec = draw_source(
            src,
            target_tokens=src.get("target_tokens", target),
            max_rows=src.get("max_rows"),
            # Session-5 addition. draw_source has always supported skip_rows -
            # the held-out draw below uses it - but training sources could not
            # reach it. telugu_verified needs it: rows 0-299 of verified/tel are
            # telugu_web's frozen held-out set, so training must start past them.
            # Defaults to 0, so every pre-existing corpus draws exactly as before.
            skip_rows=src.get("skip_rows", 0),
        )
        all_docs.extend(docs)
        source_records.append(rec)
        print(f"  [{corpus_id}] {src['key']}: {len(docs)} docs, {rec['tokens_cl100k']:,} tokens")

    # --- held-out pool, disjoint by construction -------------------------
    ho_cfg = cfg["heldout"]
    ho_src = {
        "key": "heldout",
        "hf_id": ho_cfg["hf_id"],
        "data_files": ho_cfg.get("data_files"),
        "text_field": "text",
        "claimed_lang": cfg["claimed_lang"],
    }
    skip = 0
    if ho_cfg.get("offset_after_training_slice"):
        # start past the last training row we took from this same source
        same = [r for r in source_records if r.get("hf_id") == ho_cfg["hf_id"] and r.get("last_row_index") is not None]
        skip = (max(r["last_row_index"] for r in same) + 1) if same else 0
    heldout_docs, heldout_rec = draw_source(
        ho_src, target_tokens=10**12, skip_rows=skip, max_rows=ho_cfg["n_docs"]
    )

    train_ids = {d["doc_id"] for d in all_docs}
    leaked = [d for d in heldout_docs if d["doc_id"] in train_ids]
    heldout_docs = [d for d in heldout_docs if d["doc_id"] not in train_ids]

    raw_p = corpora.raw_path(corpus_id)
    ho_p = corpora.heldout_path(corpus_id)
    os.makedirs(os.path.dirname(raw_p), exist_ok=True)
    from common import write_jsonl

    write_jsonl(raw_p, all_docs)
    write_jsonl(ho_p, heldout_docs)

    report = {
        "corpus_id": corpus_id,
        "corpus_name": cfg["name"],
        "sampling_strategy": sampling["strategy"],
        "sampling_reason": sampling["reason"],
        "target_tokens_cl100k": target,
        "actual_tokens_cl100k": sum(d["n_tokens_cl100k_raw"] for d in all_docs),
        "actual_doc_count": len(all_docs),
        "sources": source_records,
        "heldout": {
            **heldout_rec,
            "docs_after_removing_any_train_overlap": len(heldout_docs),
            "docs_dropped_because_also_in_training_pool": len(leaked),
            "note": ho_cfg["note"],
        },
        "raw_file": os.path.basename(raw_p),
        "heldout_file": os.path.basename(ho_p),
        "elapsed_s": timer.done(),
        "determinism_note": (
            "No random draw is involved. The slice is contiguous from a fixed offset, so "
            "re-running this stage against the same upstream revision reproduces the same "
            "documents in the same order."
        ),
    }
    write_json(os.path.join(corpora.RAW_DIR, f"sampling_report_{corpus_id}.json"), report)
    return report


def main(argv: list[str]) -> None:
    targets = argv[1:] or corpora.ORDER
    # Merge with whatever is already on disk, so sampling one corpus on its own
    # does not silently erase the other's provenance record.
    combined_path = os.path.join(corpora.RAW_DIR, "sampling_report.json")
    reports = {}
    if os.path.exists(combined_path):
        try:
            with open(combined_path, encoding="utf-8") as f:
                reports = json.load(f)
        except Exception:
            reports = {}
    for cid in targets:
        print(f"[stage0] sampling {cid} ...")
        reports[cid] = sample_corpus(cid)
        r = reports[cid]
        print(f"[stage0] {cid}: {r['actual_doc_count']:,} docs, {r['actual_tokens_cl100k']:,} tokens")
    write_json(os.path.join(corpora.RAW_DIR, "sampling_report.json"), reports)


if __name__ == "__main__":
    main(sys.argv)
