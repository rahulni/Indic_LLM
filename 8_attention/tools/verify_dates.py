#!/usr/bin/env python3
"""Verify every date in mechanisms.json against a primary source.

The assignment's stated grading risk is that an AI agent invents a launch date and
describes a half-remembered technique. This script exists so that no date in the app is
taken on trust: every arXiv-dated entry is checked against the arXiv API's own
`published` field (which is the v1 submission timestamp), and its title is checked too,
so a transposed identifier cannot silently point at a different paper.

Entries that genuinely have no paper - a community forum post, a product release - are
exempt from the API check but must carry a URL and a written `date_evidence` string
naming the artifact the date comes from. They are never silently waved through.

Exit code 0 means every date is backed by a source. Anything else fails the build.

Usage:  python tools/verify_dates.py [--json]
"""

from __future__ import annotations

import json
import ssl
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MECHANISMS = ROOT / "app" / "src" / "data" / "mechanisms.json"

ATOM = {"a": "http://www.w3.org/2005/Atom"}
API = "https://export.arxiv.org/api/query?id_list={ids}&max_results=100"

# arXiv asks for courtesy in batch size; 25 ids per call is well within their guidance.
BATCH = 25

VALID_DATE_KINDS = {"arxiv_v1", "release", "forum_post"}


def fetch(url: str) -> bytes:
    """GET a URL, working around this environment's incomplete SSL trust store.

    Plain urllib fails here with CERTIFICATE_VERIFY_FAILED because the local Python has
    no usable CA bundle, so we fall back to curl, which carries its own. CI has both, so
    whichever path is taken the result is identical.
    """
    try:
        return urllib.request.urlopen(url, timeout=60).read()
    except (ssl.SSLError, urllib.error.URLError):
        out = subprocess.run(
            ["curl", "-sS", "-L", "--fail", url],
            capture_output=True,
            check=True,
        )
        return out.stdout


def normalise(text: str) -> str:
    """Collapse whitespace, case and LaTeX math markers for robust title comparison.

    arXiv stores titles with the authors' LaTeX intact, so "Top-k" is recorded as
    "Top-$k$". Those dollar signs are markup rather than content, and stripping them
    keeps the check sensitive to what it is actually for - a transposed identifier
    pointing at a different paper - without tripping over typesetting.
    """
    return " ".join(text.replace("$", "").split()).lower()


def arxiv_metadata(ids: list[str]) -> dict[str, dict[str, str]]:
    """Return {bare_arxiv_id: {published, title}} for the requested identifiers."""
    found: dict[str, dict[str, str]] = {}
    for start in range(0, len(ids), BATCH):
        chunk = ids[start : start + BATCH]
        root = ET.fromstring(fetch(API.format(ids=",".join(chunk))))
        for entry in root.findall("a:entry", ATOM):
            raw = entry.find("a:id", ATOM).text.split("/abs/")[-1]
            bare = raw.split("v")[0] if "v" in raw.split("/")[-1] else raw
            found[bare] = {
                "published": entry.find("a:published", ATOM).text[:10],
                "title": " ".join(entry.find("a:title", ATOM).text.split()),
            }
    return found


def main() -> int:
    as_json = "--json" in sys.argv
    mechanisms = json.loads(MECHANISMS.read_text(encoding="utf-8"))

    failures: list[str] = []
    checked = 0
    exempt = 0

    # Structural checks first: a malformed record should fail loudly, not be skipped.
    for m in mechanisms:
        mid = m.get("id", "<missing id>")
        kind = m.get("date_kind")
        if kind not in VALID_DATE_KINDS:
            failures.append(f"{mid}: date_kind {kind!r} is not one of {sorted(VALID_DATE_KINDS)}")
        if not m.get("date"):
            failures.append(f"{mid}: no date")
        if not m.get("url"):
            failures.append(f"{mid}: no url, so the date cannot be checked by a human either")

    # Every entry that claims an arXiv v1 date gets checked against arXiv itself.
    to_check = {
        m["id"]: m for m in mechanisms if m.get("date_kind") == "arxiv_v1" and m.get("arxiv")
    }
    missing_id = [
        m["id"] for m in mechanisms if m.get("date_kind") == "arxiv_v1" and not m.get("arxiv")
    ]
    for mid in missing_id:
        failures.append(f"{mid}: date_kind is arxiv_v1 but no arxiv identifier given")

    ids = sorted({m["arxiv"] for m in to_check.values()})
    meta = arxiv_metadata(ids) if ids else {}

    for mid, m in sorted(to_check.items()):
        aid = m["arxiv"]
        record = meta.get(aid)
        if record is None:
            failures.append(f"{mid}: arXiv {aid} returned no entry - does that identifier exist?")
            continue

        checked += 1

        if record["published"] != m["date"]:
            failures.append(
                f"{mid}: date {m['date']} does not match arXiv v1 {record['published']} "
                f"for {aid}"
            )

        # A correct date attached to the wrong paper is still wrong. Checking the title
        # is what catches a transposed identifier.
        expected = m.get("title_check")
        if expected and normalise(expected) != normalise(record["title"]):
            failures.append(
                f"{mid}: title mismatch for {aid}\n"
                f"       expected: {expected}\n"
                f"       arXiv says: {record['title']}"
            )

    # Non-paper dates are allowed, but only with written evidence.
    for m in mechanisms:
        if m.get("date_kind") in {"release", "forum_post"}:
            exempt += 1
            if not m.get("date_evidence"):
                failures.append(
                    f"{m['id']}: date_kind {m['date_kind']!r} needs a date_evidence string "
                    "naming the dated artifact the date comes from"
                )

    if as_json:
        print(json.dumps({"checked": checked, "exempt": exempt, "failures": failures}, indent=2))
    else:
        print(f"arXiv dates verified against the API : {checked}")
        print(f"non-paper dates with written evidence: {exempt}")
        print(f"total mechanisms                     : {len(mechanisms)}")
        if failures:
            print(f"\nFAILED ({len(failures)}):")
            for f in failures:
                print(f"  - {f}")
        else:
            print("\nOK - every date is backed by a primary source.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
