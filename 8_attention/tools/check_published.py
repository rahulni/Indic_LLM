#!/usr/bin/env python3
"""Assert the published site was built from the content currently in the repository.

The failure this guards against is real and easy: edit `mechanisms.json`, forget to
rebuild and republish, and the live site quietly shows yesterday's data while the
repository shows today's.

It does *not* try to prove the committed bundle is byte-identical to a fresh build. An
earlier version did, and failed on every CI run for a reason that had nothing to do with
staleness: this repository is authored on Windows, where the working tree is CRLF; git
stores LF; the Linux runner therefore compiles different source bytes and emits a
different content hash. Line endings, npm resolution and per-platform esbuild binaries
all move those bytes, and none of them mean the site is out of date.

So this compares the thing that actually matters - a fingerprint of the content files,
with line endings normalised - and separately checks that the assets `index.html` points
at are really there.

Usage:  python tools/check_published.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index.html"
ASSETS = ROOT / "assets"
BUILD_INFO = ROOT / "build-info.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_site import fingerprint  # noqa: E402  - shares one definition on purpose

REBUILD = (
    "Rebuild and republish:\n"
    "    cd app && npm run build\n"
    "    cd .. && python tools/publish_site.py"
)


def main() -> int:
    failures: list[str] = []

    for path in (INDEX, BUILD_INFO):
        if not path.exists():
            print(f"error: {path.name} is missing. {REBUILD}")
            return 1

    info = json.loads(BUILD_INFO.read_text(encoding="utf-8"))
    current = fingerprint()
    recorded = info.get("data_fingerprint")

    if recorded != current:
        failures.append(
            "The published site was built from different content than the repository "
            "currently holds.\n"
            f"       recorded: {recorded}\n"
            f"       current : {current}"
        )
    else:
        print(f"data fingerprint matches ({current[:12]})")

    # Every asset index.html references must exist, or the live page loads a blank shell.
    referenced = set(re.findall(r"/Indic_LLM/8_attention/(assets/[^\"']+)", INDEX.read_text(encoding="utf-8")))
    if not referenced:
        failures.append("index.html references no assets under /Indic_LLM/8_attention/assets/")

    for rel in sorted(referenced):
        if (ROOT / rel).exists():
            print(f"referenced asset present: {rel}")
        else:
            failures.append(f"index.html references {rel}, which is not committed")

    # And nothing orphaned: a stale bundle left behind means the repository grows and it
    # is no longer obvious which file the live site actually loads.
    on_disk = {f"assets/{p.name}" for p in ASSETS.iterdir()} if ASSETS.exists() else set()
    for orphan in sorted(on_disk - referenced):
        failures.append(f"{orphan} is committed but nothing references it")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        print(f"\n{REBUILD}")
        return 1

    print("\nOK - the published site matches the repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
