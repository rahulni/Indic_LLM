#!/usr/bin/env python3
"""Check that every URL this project points at still resolves.

Adding further-reading links created a maintenance surface the rest of the project did
not have: a paper's date cannot rot, but a repository can be renamed and a blog can go
away. This is the guard for that.

The rule is **fail only on certainty**. A checker that cries wolf gets ignored, and an
ignored check is worse than no check at all.

  * **404 and 410 fail the build.** Those mean the resource is genuinely not there, which
    is the same class of error as a wrong date. During development a guessed repository
    URL for Multi-Token Attention returned 404, which is exactly what this is for.

  * **401, 403 and 429 are reported, not failed.** They mean "not for you", not "gone".
    Reddit and Substack both refuse datacenter IPs, so those URLs open fine in a browser
    and return 403 from a CI runner. Treating that as a broken link is simply wrong - and
    it is what made this workflow red on every run until it was fixed.

  * **5xx, timeouts and connection errors are reported, not failed.** That is the internet
    having a bad day. CI that goes red because GitHub was briefly slow trains people to
    ignore red CI, which costs more than it saves.

  * **Anything else is reported, not failed.** Ambiguous is not the same as broken.

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

# The only statuses that prove a link is dead.
GONE = {404, 410}

# Access control and bot protection: the resource exists, this client was refused.
BLOCKED = {401, 403, 429}


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

    # resources.ts is TypeScript, so URLs are pulled out by pattern rather than by
    # importing it. A stricter parse would mean a Node dependency for no real gain.
    for url in re.findall(r"url:\s*'([^']+)'", RESOURCES.read_text(encoding="utf-8")):
        add(url, "resources.ts")

    return where


def status(url: str) -> tuple[int | None, str]:
    """HTTP status for a URL, or None if the request could not be completed.

    curl rather than urllib: this machine's Python has an incomplete CA bundle and curl
    carries its own. Some hosts reject HEAD, so this issues a GET and discards the body.
    """
    try:
        out = subprocess.run(
            [
                "curl", "-s", "-o", "/dev/null", "-L",
                "--max-time", str(TIMEOUT),
                "-w", "%{http_code}",
                "-A", "Mozilla/5.0 (compatible; attention-timeline link check)",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT + 10,
        )
        code = int(out.stdout.strip() or 0)
        return (code if code else None), ""
    except Exception as err:  # noqa: BLE001 - any failure here means "could not reach"
        return None, str(err)


def main() -> int:
    quiet = "--quiet" in sys.argv
    where = collect()

    broken: list[str] = []
    reported: list[str] = []
    counts: Counter[str] = Counter()

    for url in sorted(where):
        code, err = status(url)
        refs = ", ".join(where[url])

        if code is None:
            counts["unreachable"] += 1
            reported.append(f"{url}\n      could not connect ({err[:80]}) - {refs}")
        elif code in GONE:
            counts["broken"] += 1
            broken.append(f"{url}\n      HTTP {code} - referenced by {refs}")
        elif code in BLOCKED:
            # Bot protection, not breakage. These open fine in a browser.
            counts["blocked"] += 1
            reported.append(f"{url}\n      HTTP {code} refused for automated clients - {refs}")
        elif code >= 400:
            counts["other"] += 1
            reported.append(f"{url}\n      HTTP {code} - {refs}")
        else:
            counts["ok"] += 1
            if not quiet:
                print(f"  {code}  {url}")

    print(f"\nchecked {len(where)} unique URLs")
    for key in ("ok", "broken", "blocked", "other", "unreachable"):
        if counts[key]:
            print(f"  {key:12} {counts[key]}")

    if reported:
        print(f"\nReported, not failing the build ({len(reported)}):")
        for r in reported:
            print(f"  - {r}")

    if broken:
        print(f"\nBROKEN ({len(broken)}):")
        for b in broken:
            print(f"  - {b}")
        print("\nA 404 or 410 means the resource is gone. Fix or remove the link.")
        return 1

    print("\nOK - nothing is gone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
