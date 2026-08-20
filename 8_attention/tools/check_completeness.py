#!/usr/bin/env python3
"""Assert the app covers everything it is required to cover, honestly.

The instructor's stated failure mode for this assignment: your agent builds something,
you don't know it missed something, and you score zero. This script is the answer to
that. It checks two independent lists, because either one alone is insufficient:

  1. The written assignment's required mechanisms.
  2. The mechanisms actually taught in the session, recovered from the transcript audit.

List 2 matters because the transcript teaches GSA (Gated Slot Attention), which the
written list omits entirely. Checking only the assignment would have silently dropped a
mechanism that was covered in class - the exact failure this script exists to prevent.

It also enforces the content rule that every mechanism must state what it costs. A
technique written up with only advantages has not been understood, so an empty `costs`
list is a hard failure rather than a warning.

Usage:  python tools/check_completeness.py [--json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MECHANISMS = ROOT / "app" / "src" / "data" / "mechanisms.json"

# The written assignment's explicit list, in the order it was given.
REQUIRED_BY_ASSIGNMENT = {
    "standard attention": "scaled-dot-product-attention",
    "absolute learned positions": "learned-absolute-positions",
    "sinusoidal": "sinusoidal-positions",
    "RoPE": "rope",
    "ALiBi": "alibi",
    "MQA": "mqa",
    "GQA": "gqa",
    "sliding window": "sliding-window-production",
    "attention sinks": "attention-sinks",
    "NTK-aware scaling": "ntk-aware-scaling",
    "YaRN": "yarn",
    "linear attention": "linear-attention",
    "the delta rule": "delta-rule",
    "Gated DeltaNet": "gated-deltanet",
    "MLA": "mla",
    "sparse and top-k attention": "topk-attention",
    "compressed/sparse attention as DeepSeek does it": "nsa",
    "DroPE": "drope",
}

# Recovered from the transcript audit. Keyed by the term the instructor actually used,
# because the point of this list is to prove the session itself is covered.
TAUGHT_IN_CLASS = {
    "KQV / scaled dot product": "scaled-dot-product-attention",
    "MHA / heads": "multi-head-attention",
    "position embedding": "learned-absolute-positions",
    "rope": "rope",
    "drop rope": "drope",
    "linear attention": "linear-attention",
    "delta rule": "delta-rule",
    "delta net / 'Delta Knight'": "deltanet-parallel",
    "gated delta net": "gated-deltanet",
    "GSA": "gsa",
    "GQA / 'group query attention'": "gqa",
    "MQA": "mqa",
    "sparse attention": "sparse-transformer",
    "top k": "topk-attention",
    "DeepSeek compression / 'low rank indexer'": "dsa",
}

# Fields that carry the actual explanation. The assignment is graded on these, so an
# entry that exists but says nothing is treated as missing.
#
# `plain` is the jargon-free register the page opens in. It is enforced here for the same
# reason `costs` is: the site renders it by default, so an entry without one would show a
# blank where its explanation should be. Presence is checkable; quality is not, so these
# are written deliberately rather than generated.
REQUIRED_PROSE = ("problem", "mechanism", "pick_when", "plain")


def main() -> int:
    as_json = "--json" in sys.argv
    mechanisms = json.loads(MECHANISMS.read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in mechanisms}

    failures: list[str] = []

    ids = [m["id"] for m in mechanisms]
    duplicates = {i for i in ids if ids.count(i) > 1}
    for dupe in sorted(duplicates):
        failures.append(f"duplicate id: {dupe}")

    for label, mid in sorted(REQUIRED_BY_ASSIGNMENT.items()):
        if mid not in by_id:
            failures.append(f"assignment requires '{label}' but id {mid!r} is missing")

    for label, mid in sorted(TAUGHT_IN_CLASS.items()):
        if mid not in by_id:
            failures.append(f"taught in class as '{label}' but id {mid!r} is missing")

    # Content quality rules, applied to every entry, not just required ones.
    for m in mechanisms:
        mid = m["id"]
        if not m.get("costs"):
            failures.append(
                f"{mid}: no costs listed. Every mechanism is a trade; a technique with "
                "only pros has not been understood."
            )
        if not m.get("buys"):
            failures.append(f"{mid}: no buys listed")
        for field in REQUIRED_PROSE:
            if not (m.get(field) or "").strip():
                failures.append(f"{mid}: empty {field}")
        if not m.get("era"):
            failures.append(f"{mid}: no era, so it cannot be placed in the story")
        if not m.get("category"):
            failures.append(f"{mid}: no category")

    covered_class = sum(1 for m in mechanisms if m.get("covered_status") == "definitely_covered")
    extension = sum(1 for m in mechanisms if m.get("covered_status") == "not_in_transcript")

    if as_json:
        print(json.dumps({"total": len(mechanisms), "failures": failures}, indent=2))
    else:
        print(f"mechanisms total                  : {len(mechanisms)}")
        print(f"required by the written assignment: {len(REQUIRED_BY_ASSIGNMENT)}")
        print(f"taught in class (transcript audit): {len(TAUGHT_IN_CLASS)}")
        print(f"  of which definitely covered     : {covered_class}")
        print(f"  extension beyond the session    : {extension}")
        if failures:
            print(f"\nFAILED ({len(failures)}):")
            for f in failures:
                print(f"  - {f}")
        else:
            print("\nOK - both lists covered, every mechanism states what it costs.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
