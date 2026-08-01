# -*- coding: utf-8 -*-
"""
run_verified.py - clean Sangraha Verified (Telugu), the tier the mixture audit
says we should have been cleaning all along.

Session 4 cleaned the *unverified* tier. At a 3T budget that tier supplies 11.1%
of the Indic lane and is barred from the anneal; Verified supplies 88.9% and is
the only tier the cooldown accepts (MIXTURE_PLAN.md §7, §16).

This drives session 4's existing 8-stage pipeline over the new corpus and writes
its own results file. **Session 4's results.json is never touched** - it is an
already-submitted artifact.

    python run_verified.py --check-only   # sample + prove disjointness, no stages
    python run_verified.py                # full pipeline

Disjointness matters here and is not assumed: telugu_web's held-out set is drawn
from verified/tel rows 0-299. The registry sets skip_rows=300, and this script
additionally asserts that no doc_id in the new training slice appears in
telugu_heldout.jsonl. An offset is a claim; the assertion is the evidence.
"""

import argparse
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIGNMENT = os.path.join(HERE, "..", "..", "4_model_data", "assignment")
PIPELINE = os.path.join(ASSIGNMENT, "pipeline")
sys.path.insert(0, PIPELINE)

CORPUS = "telugu_verified"
OUT = os.path.join(HERE, "results_verified.json")

import corpora  # noqa: E402


def heldout_doc_ids():
    """doc_ids already frozen as telugu_web's Golden Proxy."""
    path = corpora.heldout_path("telugu_web")
    ids = set()
    if not os.path.exists(path):
        return ids, path
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("doc_id"):
                ids.add(d["doc_id"])
    return ids, path


def assert_disjoint(raw_path):
    """The evidence, not the claim."""
    frozen, frozen_path = heldout_doc_ids()
    if not frozen:
        raise SystemExit(f"could not read frozen held-out ids from {frozen_path}")

    train_ids, rows = set(), []
    with io.open(raw_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            train_ids.add(d.get("doc_id"))
            if d.get("source_row_index") is not None:
                rows.append(d["source_row_index"])

    overlap = train_ids & frozen
    lo = min(rows) if rows else None
    print(f"  frozen held-out doc_ids (telugu_web): {len(frozen):,}")
    print(f"  new training doc_ids:                 {len(train_ids):,}")
    print(f"  lowest source_row_index in training:  {lo}")
    print(f"  overlap:                              {len(overlap)}")
    if overlap:
        raise SystemExit(
            f"CONTAMINATION: {len(overlap)} training docs are in telugu_web's "
            f"held-out set. Raise skip_rows above {lo}.")
    if lo is not None and lo < 300:
        raise SystemExit(
            f"training slice starts at row {lo}, inside the held-out range 0-299")
    print("  DISJOINT - training slice does not touch the frozen Golden Proxy")
    return dict(frozen_heldout_ids=len(frozen), training_ids=len(train_ids),
                lowest_training_row=lo, overlap=len(overlap), disjoint=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="sample and prove disjointness, then stop")
    args = ap.parse_args()

    t0 = time.time()
    cfg = corpora.get(CORPUS)
    print(f"corpus: {cfg['name']}")
    print(f"  source     : {cfg['sources'][0]['data_files']}")
    print(f"  skip_rows  : {cfg['sources'][0]['skip_rows']}")
    print(f"  target     : {cfg['sampling']['target_tokens']:,} cl100k tokens")
    print()

    raw_path = corpora.raw_path(CORPUS)
    if not os.path.exists(raw_path):
        print("== stage 0: draw the sample ==")
        import stage0_sample
        stage0_sample.sample_corpus(CORPUS)
        print()
    else:
        print(f"== stage 0: reusing existing {os.path.basename(raw_path)} ==\n")

    print("== disjointness check ==")
    disjoint = assert_disjoint(raw_path)
    print()

    if args.check_only:
        print(f"check-only, stopping. {time.time()-t0:.1f}s")
        return 0

    print("== stages 1-8 (plus the determinism re-run) ==")
    import run_all

    # process_corpus(), NOT main(). main() writes ASSIGNMENT/results.json, which
    # is session 4's submitted artifact. process_corpus returns the same per-
    # corpus dict that main() would have assembled, and writes nothing there.
    result = run_all.process_corpus(CORPUS)

    payload = dict(
        corpus=CORPUS, name=cfg["name"],
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        elapsed_s=time.time() - t0,
        disjointness=disjoint,
        quality_classifier_caveat=cfg.get("quality_classifier_caveat"),
        result=result,
    )
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False, default=str)
    print(f"\nwrote {os.path.normpath(OUT)}  ({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
