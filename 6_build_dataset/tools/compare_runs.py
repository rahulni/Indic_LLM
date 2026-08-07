# -*- coding: utf-8 -*-
"""Compare two artifact directories for determinism.

    python tools/compare_runs.py DIR_A DIR_B

Compares the things that must be identical -- shard content hashes, batch ids,
batch content hashes, manifest bodies -- and deliberately ignores the things
that must not be (wall-clock, throughput rates, host details). Raw file diffing
would fail on timing noise and prove nothing.

Exit code 0 when the runs agree.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tdes.hashing import read_json, read_jsonl        # noqa: E402


def load(d: str) -> dict:
    cons = read_jsonl(os.path.join(d, "ledgers", "consumption.jsonl"),
                      tolerate_torn_tail=True)
    mdir = os.path.join(d, "manifests")
    skip = {"index.json", "mixture_schedule.json"}
    manifests = {}
    for n in sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []:
        if n.endswith(".json") and n not in skip:
            m = read_json(os.path.join(mdir, n))
            if "shard_id" in m:
                manifests[m["shard_id"]] = m
    return {
        "shard_hashes": {k: m["content_sha256"] for k, m in manifests.items()},
        "manifest_hashes": {k: m["manifest_sha256"] for k, m in manifests.items()},
        "tokenizer": read_json(os.path.join(d, "reports.json"))
                     .get("tokenizer", {}).get("tokenizer_hash"),
        "batch_ids": [r["batch_id"] for r in cons],
        "content_hashes": [r["batch_content_hash"] for r in cons],
        "loss_mask_hashes": [r["loss_mask_hash"] for r in cons],
        "steps": [r["global_step"] for r in cons],
        "records": len(cons),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    problems = []
    for key in ("tokenizer", "records", "shard_hashes", "manifest_hashes",
                "batch_ids", "content_hashes", "loss_mask_hashes", "steps"):
        if a[key] != b[key]:
            problems.append(key)

    width = max(len(k) for k in a)
    for key in a:
        same = a[key] == b[key]
        n = len(a[key]) if isinstance(a[key], (list, dict)) else 1
        print(f"  {key:<{width}}  {'OK  ' if same else 'DIFF'}  ({n} values)")

    if problems:
        print(f"\nFAIL: runs differ in {problems}")
        for key in problems:
            if isinstance(a[key], list):
                for i, (x, y) in enumerate(zip(a[key], b[key])):
                    if x != y:
                        print(f"  first {key} difference at index {i}: {x} != {y}")
                        break
        return 1
    print("\nPASS: both runs produced identical shards, manifests and batches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
