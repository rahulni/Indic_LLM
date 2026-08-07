# -*- coding: utf-8 -*-
"""The evaluation and validation firewall.

Evaluation data needs a manifest too. The difference is permission: training
shards are admitted into the stream, test shards are registered precisely so
they can be kept out of it.

Three permission levels, and the distinction between the last two is the one
that is easy to get wrong:

===========  ==========================  ===============================
split        readable during training?   may produce gradients?
===========  ==========================  ===============================
``train``    yes                         yes
``validation`` yes -- that is its job    **no, never**
``eval``     no                          no
===========  ==========================  ===============================

Enforcement is deliberately duplicated. The shard writer refuses never-train
documents (``shards.py``), and this module independently re-checks every
candidate before it can reach a batch. A single copy-paste error should not be
able to defeat the whole mechanism, and the lecture is explicit that the check
belongs on both sides.

Detection uses three signals, because a hash alone is brittle -- one edited
character defeats it:

* **content hash**  -- exact copies
* **canary string** -- a distinctive marker embedded in every held-out document
* **n-gram fingerprint** -- overlapping word 8-grams, which survives light editing
"""
from __future__ import annotations

from .determinism import canonical_text
from .hashing import sha256_text

NGRAM = 8
NGRAM_OVERLAP_THRESHOLD = 0.10   # fraction of a candidate's n-grams seen in held-out data


class EvalFirewallViolation(RuntimeError):
    """Raised when held-out data reaches a loss-bearing path."""


def _ngrams(text: str, n: int = NGRAM) -> set[str]:
    words = canonical_text(text).lower().split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class EvalRegistry:
    """Registry of held-out data, with an access log.

    The access log exists because the page asks for it, and because
    "validation was read 12 times during training and eval was read 0 times" is
    exactly the evidence that the two are being treated differently.
    """

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.hashes: dict[str, str] = {}          # content hash -> doc_id
        self.canaries: dict[str, str] = {}        # canary string -> doc_id
        self.ngrams: dict[str, str] = {}          # n-gram -> doc_id
        self.access_log: list[dict] = []
        self.blocked: list[dict] = []

    # -- registration ------------------------------------------------------

    def register(self, doc: dict, split: str) -> None:
        if not doc.get("never_train"):
            raise EvalFirewallViolation(
                f"{doc['doc_id']} registered as {split} but never_train is not set"
            )
        entry = {
            "doc_id": doc["doc_id"],
            "split": split,
            "content_sha256": doc["content_sha256"],
            "benchmark_id": doc.get("benchmark_id"),
            "version_tag": "demo-v1",
            "canary": doc.get("canary"),
            "lane": doc.get("lane"),
            "never_train": True,
            "gradient_bearing_allowed": False,
            "readable_during_training": split == "validation",
        }
        self.entries[doc["doc_id"]] = entry
        self.hashes[doc["content_sha256"]] = doc["doc_id"]
        if doc.get("canary"):
            self.canaries[doc["canary"]] = doc["doc_id"]
        for g in _ngrams(doc["text"]):
            self.ngrams.setdefault(g, doc["doc_id"])

    def register_all(self, heldout: dict[str, list[dict]]) -> dict:
        for split, docs in sorted(heldout.items()):
            for d in sorted(docs, key=lambda x: x["doc_id"]):
                self.register(d, split)
        return self.summary()

    # -- enforcement -------------------------------------------------------

    def check_document(self, doc: dict) -> dict:
        """Screen one candidate. Returns a verdict; never raises."""
        text = doc.get("text", "")
        reasons: list[str] = []
        matched: list[str] = []

        if doc.get("never_train"):
            reasons.append("document is flagged never_train")

        h = doc.get("content_sha256") or sha256_text(text)
        if h in self.hashes:
            reasons.append("exact content hash matches a held-out document")
            matched.append(self.hashes[h])

        for canary, owner in self.canaries.items():
            if canary in text:
                reasons.append("held-out canary string present in text")
                matched.append(owner)
                break

        grams = _ngrams(text)
        if grams:
            hits = [self.ngrams[g] for g in grams if g in self.ngrams]
            overlap = len(hits) / len(grams)
            if overlap >= NGRAM_OVERLAP_THRESHOLD:
                reasons.append(
                    f"{overlap:.1%} of {NGRAM}-grams overlap held-out data "
                    f"(threshold {NGRAM_OVERLAP_THRESHOLD:.0%})"
                )
                matched.extend(hits[:3])

        verdict = {
            "doc_id": doc.get("doc_id"),
            "blocked": bool(reasons),
            "reasons": reasons,
            "matched_heldout": sorted(set(matched)),
        }
        if reasons:
            self.blocked.append(verdict)
        return verdict

    def screen_candidates(self, docs: list[dict]) -> dict:
        """Screen a list of candidates. Returns admitted docs and the blocks."""
        admitted, blocked = [], []
        for d in sorted(docs, key=lambda x: x.get("doc_id", "")):
            v = self.check_document(d)
            (blocked if v["blocked"] else admitted).append(d if not v["blocked"] else v)
        return {"admitted": admitted, "blocked": blocked}

    def assert_not_gradient_bearing(self, doc_ids: list[str], *, where: str) -> None:
        """Hard stop. Called on the batch path with the doc ids about to bear loss."""
        bad = [d for d in doc_ids if d in self.entries]
        if bad:
            raise EvalFirewallViolation(
                f"held-out documents reached a loss-bearing batch in {where}: {sorted(bad)}"
            )

    # -- access accounting -------------------------------------------------

    def note_access(self, doc_ids: list[str], *, purpose: str, step: int | None = None) -> None:
        """Record a read of held-out data.

        ``purpose='validation_probe'`` is legitimate. Any access with
        ``purpose='training'`` is a bug, and the log is where it becomes visible.
        """
        for did in sorted(set(doc_ids)):
            e = self.entries.get(did)
            if e is None:
                continue
            self.access_log.append({
                "doc_id": did, "split": e["split"],
                "purpose": purpose, "global_step": step,
                "gradient_bearing": False,
            })

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        by_split: dict[str, int] = {}
        for e in self.entries.values():
            by_split[e["split"]] = by_split.get(e["split"], 0) + 1
        reads: dict[str, int] = {}
        for a in self.access_log:
            reads[a["split"]] = reads.get(a["split"], 0) + 1
        return {
            "registered": len(self.entries),
            "by_split": dict(sorted(by_split.items())),
            "fingerprints": {
                "content_hashes": len(self.hashes),
                "canaries": len(self.canaries),
                "ngrams": len(self.ngrams),
                "ngram_size": NGRAM,
                "ngram_overlap_threshold": NGRAM_OVERLAP_THRESHOLD,
            },
            "blocked_count": len(self.blocked),
            "blocked": self.blocked[:50],
            "access_log_entries": len(self.access_log),
            "reads_by_split": dict(sorted(reads.items())),
            "gradient_bearing_reads": sum(1 for a in self.access_log if a["gradient_bearing"]),
        }

    def registry_records(self) -> list[dict]:
        return [self.entries[k] for k in sorted(self.entries)]
