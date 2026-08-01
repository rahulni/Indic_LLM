"""Orchestrator.

Runs stages 1-8 over every registered corpus and rolls every stage report
into one assignment/results.json - the single source of truth the widget
reads from. Nothing in the front end is allowed to show a number this file
did not produce.

The determinism check re-executes the whole pipeline a second time and
diffs the final content hash and script hash. That second run happens in a
**fresh subprocess with a different PYTHONHASHSEED**, which is the part
that makes it a real test: two passes inside one process share a hash
seed, so any accidental dependence on set or dict iteration order would
reproduce identically and the check would pass while proving nothing.

  python run_all.py                    # every corpus
  python run_all.py telugu_web         # just one
"""
from __future__ import annotations

import datetime
import json
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import corpora
import stage1_extract
import stage2_normalize
import stage3_langid
import stage4_quality
import stage5_dedup
import stage6_pii
import stage7_decontaminate
import stage8_manifest
from common import ASSIGNMENT_DIR, read_jsonl, write_json

THIS_FILE = os.path.abspath(__file__)


def run_pipeline_once(corpus_id: str, work_dir: str) -> list[dict]:
    common.set_context(corpus_id, work_dir)
    reports = []
    reports.append(stage1_extract.run())
    reports.append(stage2_normalize.run(common.work_path("stage1_survivors.jsonl")))
    reports.append(stage3_langid.run(common.work_path("stage2_survivors.jsonl")))
    reports.append(stage4_quality.run(common.work_path("stage3_survivors.jsonl")))
    reports.append(stage5_dedup.run(common.work_path("stage4_survivors.jsonl")))
    reports.append(stage6_pii.run(common.work_path("stage5_survivors.jsonl")))
    reports.append(stage7_decontaminate.run(common.work_path("stage6_survivors.jsonl")))
    reports.append(stage8_manifest.run(common.work_path("stage7_survivors.jsonl")))
    for r in reports:
        print(f"    [{corpus_id}] stage {r['stage_num']} {r['stage_name']}: "
              f"{r['input_docs']} -> {r['output_docs']} ({r['survival_pct']}%) in {r['elapsed_s']}s", flush=True)
    return reports


def determinism_subprocess(corpus_id: str, work_dir: str) -> list[dict]:
    """Second pass, in a genuinely separate interpreter with a different hash
    seed. If any stage depended on hash ordering, this is where it shows up."""
    out_path = os.path.join(work_dir, "_reports.json")
    env = dict(os.environ)
    parent_seed = env.get("PYTHONHASHSEED")
    seed = str(random.randint(1, 4_000_000_000))
    while seed == parent_seed:
        seed = str(random.randint(1, 4_000_000_000))
    env["PYTHONHASHSEED"] = seed
    print(f"  [{corpus_id}] determinism re-run in subprocess (PYTHONHASHSEED={seed}) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, THIS_FILE, "--single", corpus_id, work_dir, out_path],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"determinism subprocess failed for {corpus_id}")
    with open(out_path, encoding="utf-8") as f:
        return json.load(f), seed


def process_corpus(corpus_id: str) -> dict:
    cfg = corpora.get(corpus_id)
    t0 = time.time()
    print(f"\n=== {corpus_id}: {cfg['name']} ===", flush=True)

    raw_docs = read_jsonl(corpora.raw_path(corpus_id))
    raw_tokens = sum(d["n_tokens_cl100k_raw"] for d in raw_docs)
    print(f"  raw: {len(raw_docs):,} docs, {raw_tokens:,} tokens (cl100k_base)", flush=True)

    work_dir = os.path.join(ASSIGNMENT_DIR, f"work_{corpus_id}")
    reports = run_pipeline_once(corpus_id, work_dir)

    funnel = [{"stage": 0, "name": "Raw input", "docs": len(raw_docs), "tokens": raw_tokens}]
    for r in reports:
        funnel.append(
            {
                "stage": r["stage_num"],
                "name": r["stage_name"],
                "docs": r["output_docs"],
                "docs_removed": r["docs_removed"],
                "survival_pct_this_stage": r["survival_pct"],
            }
        )
    funnel[-1]["tokens"] = reports[-1]["extra"]["manifest"]["final_token_count_cl100k"]

    det_dir = os.path.join(ASSIGNMENT_DIR, f"work_determinism_{corpus_id}")
    if os.path.isdir(det_dir):
        shutil.rmtree(det_dir)
    reports_2, seed = determinism_subprocess(corpus_id, det_dir)

    m1 = reports[-1]["extra"]["manifest"]
    m2 = reports_2[-1]["extra"]["manifest"]
    determinism = {
        "run1_content_sha256": m1["content_sha256"],
        "run2_content_sha256": m2["content_sha256"],
        "run1_script_hash": m1["cleaning_script_hash_sha256"],
        "run2_script_hash": m2["cleaning_script_hash_sha256"],
        "run1_final_doc_count": m1["final_doc_count"],
        "run2_final_doc_count": m2["final_doc_count"],
        "content_hash_matches": m1["content_sha256"] == m2["content_sha256"],
        "script_hash_matches": m1["cleaning_script_hash_sha256"] == m2["cleaning_script_hash_sha256"],
        "doc_count_matches": m1["final_doc_count"] == m2["final_doc_count"],
        "run2_executed_in_separate_process": True,
        "run2_pythonhashseed": seed,
        "method_note": (
            "The second run is a separate interpreter launched with a different PYTHONHASHSEED. "
            "Two passes inside one process share a hash seed, so any accidental dependence on "
            "set or dict iteration order would reproduce identically and the check would pass "
            "without proving anything. This version can actually fail."
        ),
    }
    determinism["fully_reproducible"] = all(
        [determinism["content_hash_matches"], determinism["script_hash_matches"], determinism["doc_count_matches"]]
    )
    shutil.rmtree(det_dir, ignore_errors=True)

    sampling_path = os.path.join(corpora.RAW_DIR, f"sampling_report_{corpus_id}.json")
    sampling = json.load(open(sampling_path, encoding="utf-8")) if os.path.exists(sampling_path) else None

    print(f"  [{corpus_id}] determinism: fully_reproducible = {determinism['fully_reproducible']}", flush=True)

    return {
        "corpus_id": corpus_id,
        "corpus_name": cfg["name"],
        "corpus_kind": cfg["kind"],
        "hf_url": cfg["hf_url"],
        "elapsed_s": round(time.time() - t0, 2),
        "sampling_report": sampling,
        "raw_input": {"doc_count": len(raw_docs), "token_count_cl100k": raw_tokens},
        "stage_reports": reports,
        "survival_funnel": funnel,
        "determinism_check": determinism,
        "final_manifest": m1,
    }


def main() -> None:
    # Child mode: one corpus, one work dir, dump reports and exit.
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        corpus_id, work_dir, out_path = sys.argv[2], sys.argv[3], sys.argv[4]
        reports = run_pipeline_once(corpus_id, work_dir)
        write_json(out_path, reports)
        return

    t0 = time.time()
    targets = [a for a in sys.argv[1:] if not a.startswith("-")] or corpora.ORDER
    results = {}
    for cid in targets:
        results[cid] = process_corpus(cid)

    totals = {
        "corpora_count": len(results),
        "raw_docs": sum(r["raw_input"]["doc_count"] for r in results.values()),
        "raw_tokens_cl100k": sum(r["raw_input"]["token_count_cl100k"] for r in results.values()),
        "final_docs": sum(r["final_manifest"]["final_doc_count"] for r in results.values()),
        "final_tokens_cl100k": sum(r["final_manifest"]["final_token_count_cl100k"] for r in results.values()),
        "all_deterministic": all(r["determinism_check"]["fully_reproducible"] for r in results.values()),
    }

    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pipeline_elapsed_s": round(time.time() - t0, 2),
        "corpus_order": [c for c in corpora.ORDER if c in results],
        "corpora": results,
        "totals": totals,
    }
    results_path = os.path.join(ASSIGNMENT_DIR, "results.json")
    write_json(results_path, out)
    print(f"\nWrote {results_path}")
    print(f"Totals: {totals['final_docs']:,} docs / {totals['final_tokens_cl100k']:,} tokens "
          f"across {totals['corpora_count']} corpora; all_deterministic={totals['all_deterministic']}")
    print(f"Total wall clock: {out['pipeline_elapsed_s']}s")


if __name__ == "__main__":
    main()
