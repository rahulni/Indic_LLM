# -*- coding: utf-8 -*-
"""PII screening.

The manifest carries ``pii_status``. As with dedup, the field is only worth
anything if something computed it, so this runs a real regex screen and records
real counts.

Scope is deliberately narrow and stated plainly: this is a **pattern screen**,
not identity resolution. It catches the shapes that leak most often in scraped
Indian web text -- the categories session 4's ``stage6_pii.py`` targets and that
the DPDP-Act analysis in session 3 calls out. It will not catch a name in
running prose, and nothing here claims it does.

Findings are redacted in place rather than causing the document to be dropped.
Dropping would silently shrink the Indic lane, which is exactly the lane the
protected floor exists to defend.
"""
from __future__ import annotations

import re

# Ordered: more specific patterns first, so a PAN is not first matched as a
# generic alphanumeric run.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Aadhaar: 12 digits, conventionally grouped 4-4-4 with spaces.
    #
    # An earlier version allowed '-' as the group separator and matched
    # "2007-2008 2008-2009 2009-2010" in a Hindi finance table -- a year range,
    # not an identifier. Hyphens are now excluded both as a separator and on
    # either boundary, so a number that is part of a hyphenated range can no
    # longer be mistaken for an identifier. Digits must also not run on past
    # twelve, which rules out longer figures.
    ("aadhaar_shaped", re.compile(r"(?<![\d-])\d{4} ?\d{4} ?\d{4}(?![\d-])")),
    # PAN: five letters, four digits, one letter.
    ("pan_shaped", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    # Indian mobile: optional +91, then 6-9 followed by nine digits.
    ("phone_in", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)")),
    ("ipv4", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
]

REDACTION = {
    "email": "<redacted:email>",
    "aadhaar_shaped": "<redacted:aadhaar>",
    "pan_shaped": "<redacted:pan>",
    "phone_in": "<redacted:phone>",
    "ipv4": "<redacted:ip>",
}


def scan(text: str) -> dict[str, int]:
    """Count matches per category without modifying the text."""
    return {name: n for name, pat in PATTERNS
            if (n := len(pat.findall(text))) > 0}


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace matches with category placeholders. Returns (text, counts)."""
    counts: dict[str, int] = {}
    for name, pat in PATTERNS:
        text, n = pat.subn(REDACTION[name], text)
        if n:
            counts[name] = n
    return text, counts


def screen_documents(docs: list[dict]) -> dict:
    """Screen and redact a document list in stable order.

    Returns the (possibly modified) documents and a report. A document whose
    text changed gets a fresh ``content_sha256`` -- the redacted text is a
    different document and must not keep the original's identity.
    """
    from .hashing import sha256_text

    out: list[dict] = []
    totals: dict[str, int] = {}
    affected: list[dict] = []
    for d in sorted(docs, key=lambda x: x["doc_id"]):
        new_text, counts = redact(d["text"])
        if counts:
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            affected.append({"doc_id": d["doc_id"], "categories": counts})
            d = dict(d, text=new_text, content_sha256=sha256_text(new_text),
                     chars=len(new_text), pii_redacted=True)
        else:
            d = dict(d, pii_redacted=False)
        out.append(d)

    return {
        "documents": out,
        "report": {
            "documents_scanned": len(docs),
            "documents_redacted": len(affected),
            "matches_by_category": dict(sorted(totals.items())),
            "affected": affected[:50],
            "categories": [name for name, _ in PATTERNS],
            "method": "regex pattern screen with in-place redaction",
            "scope_note": (
                "This is a pattern screen for the shapes that leak most often in "
                "scraped Indian web text (email, Aadhaar-shaped, PAN-shaped, "
                "Indian mobile, IPv4). It is not identity resolution and does not "
                "detect names in running prose."
            ),
        },
    }
