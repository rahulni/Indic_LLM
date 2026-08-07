# -*- coding: utf-8 -*-
"""Load the vendored corpus and run the admission pipeline over it.

    corpus/*/documents.jsonl
      -> load  (provenance, lane, language, tier, split)
      -> PII screen + redact
      -> deduplicate (exact, then near)
      -> admitted documents  +  a report per stage

Held-out documents (``split`` in {eval, validation}) are loaded through the same
code path, because the firewall can only block what it knows exists. They are
kept strictly separate from the training pool and carry ``never_train=True``.
"""
from __future__ import annotations

import os

from .config import LANES
from .dedup import deduplicate
from .hashing import read_jsonl
from .pii import screen_documents

TRAIN_LANES = list(LANES)
HELDOUT_SPLITS = ["validation", "eval"]


class CorpusError(RuntimeError):
    pass


def _load_split(corpus_dir: str, sub: str) -> list[dict]:
    path = os.path.join(corpus_dir, sub, "documents.jsonl")
    if not os.path.exists(path):
        return []
    return read_jsonl(path)


def load_raw(corpus_dir: str, *, docs_per_lane_cap: int | None = None) -> dict:
    """Load train lanes and held-out splits without transforming them."""
    train: list[dict] = []
    per_lane: dict[str, int] = {}
    for lane in TRAIN_LANES:
        docs = sorted(_load_split(corpus_dir, lane), key=lambda d: d["doc_id"])
        if docs_per_lane_cap is not None:
            docs = docs[:docs_per_lane_cap]
        for d in docs:
            if d.get("split") != "train" or d.get("never_train"):
                raise CorpusError(
                    f"{d['doc_id']} sits in a train lane but is marked "
                    f"split={d.get('split')!r} never_train={d.get('never_train')!r}"
                )
        per_lane[lane] = len(docs)
        train.extend(docs)

    heldout: dict[str, list[dict]] = {}
    for split in HELDOUT_SPLITS:
        docs = sorted(_load_split(corpus_dir, split), key=lambda d: d["doc_id"])
        for d in docs:
            if not d.get("never_train"):
                raise CorpusError(f"{d['doc_id']} is in {split} but never_train is not set")
        heldout[split] = docs

    if not train:
        raise CorpusError(
            f"no training documents under {corpus_dir!r}. "
            f"Run `python tools/vendor_corpus.py` to build the corpus."
        )
    return {"train": train, "heldout": heldout, "per_lane": per_lane}


def admit(corpus_dir: str, *, docs_per_lane_cap: int | None = None) -> dict:
    """Full admission: load, screen for PII, deduplicate.

    Order matters. PII redaction runs *before* dedup, because redaction can turn
    two documents that differed only in a phone number into genuine duplicates,
    and we would rather catch that than train on both.
    """
    raw = load_raw(corpus_dir, docs_per_lane_cap=docs_per_lane_cap)

    pii_out = screen_documents(raw["train"])
    dd_out = deduplicate(pii_out["documents"])
    admitted = dd_out["documents"]

    # Held-out data is screened for PII too -- it is read during evaluation --
    # but never deduplicated against the training pool, because removing an eval
    # document because it resembles a training document would quietly delete
    # the very contamination we want the firewall to detect.
    heldout: dict[str, list[dict]] = {}
    heldout_pii: dict[str, dict] = {}
    for split, docs in raw["heldout"].items():
        out = screen_documents(docs)
        heldout[split] = out["documents"]
        heldout_pii[split] = out["report"]

    by_lane: dict[str, list[dict]] = {}
    for d in admitted:
        by_lane.setdefault(d["lane"], []).append(d)
    for lane in by_lane:
        by_lane[lane].sort(key=lambda d: d["doc_id"])

    return {
        "documents": admitted,
        "by_lane": by_lane,
        "heldout": heldout,
        "reports": {
            "loaded": {
                "train_documents": len(raw["train"]),
                "per_lane": raw["per_lane"],
                "validation_documents": len(raw["heldout"].get("validation", [])),
                "eval_documents": len(raw["heldout"].get("eval", [])),
            },
            "pii": pii_out["report"],
            "pii_heldout": heldout_pii,
            "dedup": dd_out["report"],
            "admitted": {
                "documents": len(admitted),
                "per_lane": {k: len(v) for k, v in sorted(by_lane.items())},
                "total_chars": sum(d["chars"] for d in admitted),
                "total_words": sum(d["words"] for d in admitted),
            },
        },
    }


def indic_tier_counts(docs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in docs:
        t = d.get("indic_tier")
        if t:
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))
