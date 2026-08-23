#!/usr/bin/env python3
"""Check that every URL this project points at still resolves.

Adding further-reading links creates a maintenance surface the rest of the project did
not have: a paper's date cannot rot, but a repository can be renamed and a blog can go
away. This is the guard for that.

The distinction it draws matters more than the check itself:

  * A **4xx** is a real broken link - the resource is gone or the URL was wrong. That
    fails the build, because it is the same class of error as a wrong date. During
    development a guessed repository URL for Multi-Token Attention returned 404, which is
    exactly what this catches.

  * A **5xx, a timeout, or a connection error** is the internet having a bad day. Those
    are reported and do not fail the build. A CI job that goes red because GitHub was
    briefly slow trains people to ignore red CI, which costs more than it saves.

  * A **429** is rate limiting, not breakage, and is treated as transient.

Usage:
    python tools/check_links.py            # check everything
    python tools/check_links.py --quiet    # only report problems
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MECHANISMS = ROOT / "app" / "src" / "data" / "mechanisms.json"
RESOURCES = ROOT / "app" / "src" / "data" / "resources.ts"

TIMEOUT = 25


def collect() -> dict[str, list[str]]:
    """Every URL in the project, mapped to the places that reference it."""
    where: dict[str, list[str]] = {}

    def add(url: str, source: str) -> None:
        where.setdefault(url, []).append(source)

    for m in json.loads(MECHANISMS.read_text(encoding="utf-8")):
        if m.get("url"):
            add(m["url"], f"{m['id']} (source)")
        for r in m.get("reading", []):
            add(r["url"], f"{m['id']} (reading)")

    # resources.ts is TypeScript, so the URLs are pulled out by pattern rather than by
    # importing it. A stricter parse would mean a Node dependency for no real gain.
    for url in re.findall(r"url:\s*'([^']+)'", RESOURCES.read_text(encoding="utf-8")):
        add(url, "resources.ts")

    return where


def status(url: str) -> tuple[int | None, str]:
    """HTTP status for a URL, or None if the request could not be completed.

    curl rather than urllib: this machine's Python has an incomplete CA bundle, and curl
    carries its own. Some hosts reject HEAD, so this issues a GET but discards the body.
    """
    try:
        out = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-L", "--max-time", str(TIMEOUT),
             "-w", "%{http_code}", "-A", "Mozilla/5.0 (link-check)", url],
            capture_output=True, text=True, timeout=TIMEOUT + 10,
        )
        code = int(out.stdout.strip() or 0)
        return (code if code else None), ""
    except Exception as err:  # noqa: BLE001 - any failure here is "could not reach"
        return None, str(err)


def main() -> int:
    quiet = "--quiet" in sys.argv
    where = collect()

    broken: list[str] = []
    transient: list[str] = []
    counts: Counter[str] = Counter()

    for url in sorted(where):
        code, err = status(url)
        refs = ", ".join(where[url])

        if code is None:
            counts["unreachable"] += 1
            transient.append(f"{url}\n      could not connect ({err[:80]}) - referenced by {refs}")
        elif code == 429:
            counts["rate-limited"] += 1
            transient.append(f"{url}\n      429 rate limited - referenced by {refs}")
        elif 400 <= code < 500:
            counts["broken"] += 1
            broken.append(f"{url}\n      HTTP {code} - referenced by {refs}")
        elif code >= 500:
            counts["server-error"] += 1
            transient.append(f"{url}\n      HTTP {code} - referenced by {refs}")
        else:
            counts["ok"] += 1
            if not quiet:
                print(f"  {code}  {url}")

    print(f"\nchecked {len(where)} unique URLs")
    for k in ("ok", "broken", "server-error", "rate-limited", "unreachable"):
        if counts[k]:
            print(f"  {k:14} {counts[k]}")

    if transient:
        print(f"\nTransient, not failing the build ({len(transient)}):")
        for t in transient:
            print(f"  - {t}")

    if broken:
        print(f"\nBROKEN ({len(broken)}):")
        for b in broken:
            print(f"  - {b}")
        print("\nA 4xx means the resource moved or the URL was wrong. Fix or remove it.")
        return 1

    print("\nOK - no broken links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
