"""Post-hoc analysis: does this corpus contain duplicates at all?

Deliberately NOT a pipeline stage. It changes no document and is excluded
from the cleaning-script hash (stage 8 globs `stage*.py`), because it is
evidence about the corpus rather than a step that produced it.

It exists to answer a question the dedup stage cannot answer on its own.
When a global MinHash/LSH pass at threshold 0.75 removes nothing, there are
two very different explanations - the corpus has no duplicates, or the
pipeline is not finding the ones that are there - and reporting "0" without
distinguishing them is exactly the kind of unexamined number this whole
exercise is about. So we sweep the threshold down and report the curve.

  python analysis_duplicate_structure.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

from datasketch import MinHash, MinHashLSH

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpora
from common import ASSIGNMENT_DIR, read_jsonl, write_json

SWEEP = [0.9, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3]
NUM_PERM = 128
SHINGLE = 5
PREFIX_CHARS = 120


def shingles(text: str, k: int = SHINGLE) -> set[str]:
    w = text.split()
    if len(w) < k:
        return {text} if text else set()
    return {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}


def analyse(corpus_id: str) -> dict | None:
    work = os.path.join(ASSIGNMENT_DIR, f"work_{corpus_id}")
    path = os.path.join(work, "stage4_survivors.jsonl")
    if not os.path.exists(path):
        return None
    docs = read_jsonl(path)

    mhs = []
    for d in docs:
        m = MinHash(num_perm=NUM_PERM)
        for s in shingles(d["text"]):
            m.update(s.encode("utf-8"))
        mhs.append(m)

    curve = []
    for th in SWEEP:
        lsh = MinHashLSH(threshold=th, num_perm=NUM_PERM)
        hits = 0
        for i, m in enumerate(mhs):
            if lsh.query(m):
                hits += 1
            else:
                lsh.insert(str(i), m)
        curve.append({"threshold": th, "documents_flagged": hits})

    # A cheap, independent signal: documents that open identically. These are
    # NOT duplicates - they are shared boilerplate leads - and showing that the
    # two measures disagree is the point.
    prefixes = Counter(d["text"][:PREFIX_CHARS] for d in docs)
    groups = [(p, n) for p, n in prefixes.items() if n > 1]
    groups.sort(key=lambda x: -x[1])

    return {
        "corpus_id": corpus_id,
        "documents_analysed": len(docs),
        "stage_analysed": "stage 4 survivors (post quality filter, pre dedup)",
        "threshold_sweep": curve,
        "operating_threshold": 0.75,
        "shared_prefix_groups": len(groups),
        "docs_sharing_a_prefix": sum(n for _, n in groups),
        "shared_prefix_examples": [{"prefix": p[:90], "documents": n} for p, n in groups[:5]],
        "interpretation": (
            "The sweep is the evidence that a zero at the operating threshold means the corpus is "
            "clean rather than the detector being blind. Where the curve stays flat at zero well "
            "below the operating threshold, there is genuinely nothing there to remove. Where it "
            "climbs steeply just under the threshold, the threshold - not the corpus - is deciding "
            "the outcome, and that is a tuning decision someone has to own. Shared opening lines "
            "are counted separately on purpose: documents that begin identically and then diverge "
            "are templated leads, not duplicates, and a prefix check would wrongly delete them."
        ),
    }


def main() -> None:
    out = {}
    for cid in corpora.ORDER:
        r = analyse(cid)
        if r is None:
            print(f"[analysis] {cid}: no stage4 survivors on disk, skipped")
            continue
        out[cid] = r
        flat = [c for c in r["threshold_sweep"] if c["documents_flagged"] == 0]
        print(
            f"[analysis] {cid}: {r['documents_analysed']:,} docs; "
            f"zero duplicates down to threshold {min(c['threshold'] for c in flat) if flat else 'n/a'}; "
            f"{r['shared_prefix_groups']} shared-prefix groups"
        )
    write_json(os.path.join(ASSIGNMENT_DIR, "analysis_duplicate_structure.json"), out)
    print(f"[analysis] wrote analysis_duplicate_structure.json")


if __name__ == "__main__":
    main()
