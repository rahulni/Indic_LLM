# -*- coding: utf-8 -*-
"""Immutable tokenized shards.

A shard is a sealed object: a flat little-endian array of token ids, an index of
the document spans inside it, and a manifest whose ``content_sha256`` is the
hash of the token bytes. Once written it is never modified -- editing a shard
mints a new one with a new hash and a ``parent_shard_ids`` lineage.

Two properties are enforced here rather than documented and hoped for:

* **Idempotence.** Rebuilding a shard from the same documents and the same
  tokenizer must produce byte-identical output. The writer rebuilds in memory
  and compares before it will overwrite, so a drift shows up as a build failure
  rather than as a silent change under a manifest that still looks valid.

* **Never-train exclusion.** The writer refuses any document flagged
  ``never_train``. This is the first of the firewall's two sides; the batch
  builder re-checks independently, because a single copy error should not be
  able to defeat the whole mechanism.

Role spans (for SFT and agentic data) are converted from character offsets to
*token* offsets here, since this is the only place that knows the tokenization.
Storing them in the corpus as character offsets keeps the corpus independent of
any particular tokenizer.
"""
from __future__ import annotations

import os
import sys
from array import array

from .config import BOS_ID, EOS_ID, TOOL_CALL_ID, TOOL_OBS_ID, UNK_ID
from .hashing import (atomic_write_bytes, ensure_dir, hash_obj, read_json,
                      sha256_bytes, write_json)

SHARD_EXT = ".shard"
SPANS_EXT = ".spans.json"

# Roles whose tokens are context for the model but never a prediction target.
CONTEXT_ONLY_ROLES = {"user", "tool_obs"}


class ShardImmutabilityError(RuntimeError):
    """Raised when rebuilding a shard does not reproduce its bytes."""


class NeverTrainViolation(RuntimeError):
    """Raised when a held-out document reaches the shard writer."""


def _array_typecode(vocab_size: int) -> str:
    return "H" if vocab_size <= 0xFFFF else "I"


def _to_bytes(ids: list[int], vocab_size: int) -> bytes:
    """Serialise token ids little-endian regardless of host byte order.

    Without the byteswap a big-endian machine would produce different bytes for
    the same tokens, so the shard hash -- and therefore the manifest -- would
    depend on the CPU that built it.
    """
    a = array(_array_typecode(vocab_size), ids)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def _from_bytes(blob: bytes, vocab_size: int) -> list[int]:
    a = array(_array_typecode(vocab_size))
    a.frombytes(blob)
    if sys.byteorder == "big":
        a.byteswap()
    return list(a)


def tokenize_document(tok, doc: dict) -> dict:
    """Tokenize one document into ids plus token-level role spans.

    Each role turn is encoded separately so the boundary between prompt and
    response lands exactly on a token boundary. Encoding the whole document and
    then trying to map character offsets back would put the boundary inside a
    token whenever a merge straddles it, and the loss mask would be wrong by one
    token in a way nothing downstream could detect.
    """
    text = doc["text"]
    role_spans = doc.get("role_spans")
    ids: list[int] = [BOS_ID]
    token_roles: list[dict] = []

    if role_spans:
        for span in role_spans:
            chunk = text[span["start"]:span["end"]]
            role = span["role"]
            start = len(ids)
            if role == "tool_call":
                ids.append(TOOL_CALL_ID)
            elif role == "tool_obs":
                ids.append(TOOL_OBS_ID)
            ids.extend(tok.encode(chunk, unk_id=UNK_ID))
            token_roles.append({"role": role, "start": start, "end": len(ids)})
    else:
        start = len(ids)
        ids.extend(tok.encode(text, unk_id=UNK_ID))
        token_roles.append({"role": "text", "start": start, "end": len(ids)})

    # EOS marks the end of the DOCUMENT, not the end of a sentence. It is the
    # boundary the attention mask and the packer both key off.
    ids.append(EOS_ID)
    return {
        "doc_id": doc["doc_id"],
        "token_ids": ids,
        "n_tokens": len(ids),
        "token_roles": token_roles,
        "atomic_unit": bool(doc.get("atomic_unit")),
    }


def build_shards(docs: list[dict], tok, *, vocab_size: int, lane: str,
                 target_tokens: int, out_dir: str,
                 tokenizer_hash: str) -> list[dict]:
    """Group documents into shards of roughly ``target_tokens`` and write them.

    Documents are taken in sorted ``doc_id`` order and never split across
    shards: a shard boundary that ran through the middle of a document would
    make the span index -- and every audit built on it -- ambiguous.
    """
    for d in docs:
        if d.get("never_train"):
            raise NeverTrainViolation(
                f"{d['doc_id']} is flagged never_train and must not enter a training shard"
            )

    ensure_dir(out_dir)
    docs = sorted(docs, key=lambda d: d["doc_id"])

    shards: list[dict] = []
    batch: list[dict] = []
    n_tokens = 0

    def flush() -> None:
        nonlocal batch, n_tokens
        if not batch:
            return
        idx = len(shards)
        shards.append(_write_shard(batch, lane, idx, out_dir,
                                   vocab_size=vocab_size, tokenizer_hash=tokenizer_hash))
        batch, n_tokens = [], 0

    for d in docs:
        enc = tokenize_document(tok, d)
        enc["_doc"] = d
        batch.append(enc)
        n_tokens += enc["n_tokens"]
        if n_tokens >= target_tokens:
            flush()
    flush()
    return shards


def _write_shard(encoded: list[dict], lane: str, index: int, out_dir: str, *,
                 vocab_size: int, tokenizer_hash: str) -> dict:
    shard_id = f"{lane}_{index:04d}"
    ids: list[int] = []
    spans: list[dict] = []
    for enc in encoded:
        start = len(ids)
        ids.extend(enc["token_ids"])
        doc = enc["_doc"]
        spans.append({
            "doc_id": enc["doc_id"],
            "start": start,
            "end": len(ids),
            "n_tokens": enc["n_tokens"],
            # Roles are stored relative to the shard, so a packer never has to
            # know where the document started.
            "token_roles": [{"role": r["role"],
                             "start": start + r["start"],
                             "end": start + r["end"]} for r in enc["token_roles"]],
            "atomic_unit": enc["atomic_unit"],
            "language": doc.get("language"),
            "script": doc.get("script"),
            "indic_tier": doc.get("indic_tier"),
            "license": doc.get("license"),
            "source_file": doc.get("source_file"),
        })

    blob = _to_bytes(ids, vocab_size)
    content_hash = sha256_bytes(blob)
    shard_path = os.path.join(out_dir, shard_id + SHARD_EXT)
    spans_path = os.path.join(out_dir, shard_id + SPANS_EXT)

    # Immutability check: if the shard already exists, it must match exactly.
    if os.path.exists(shard_path):
        with open(shard_path, "rb") as f:
            existing = f.read()
        if existing != blob:
            raise ShardImmutabilityError(
                f"{shard_id} already exists with a different content hash "
                f"({sha256_bytes(existing)} on disk vs {content_hash} rebuilt). "
                f"A shard is immutable -- write a new shard id instead."
            )
    else:
        atomic_write_bytes(shard_path, blob)

    spans_payload = {
        "shard_id": shard_id,
        "lane": lane,
        "vocab_size": vocab_size,
        "tokenizer_hash": tokenizer_hash,
        "typecode": _array_typecode(vocab_size),
        "byte_order": "little",
        "n_tokens": len(ids),
        "spans": spans,
    }
    write_json(spans_path, spans_payload, canonical=True)

    return {
        "shard_id": shard_id,
        "lane": lane,
        "path": shard_path,
        "spans_path": spans_path,
        "content_sha256": content_hash,
        "spans_sha256": hash_obj(spans_payload),
        "n_tokens": len(ids),
        "n_documents": len(spans),
        "doc_ids": [s["doc_id"] for s in spans],
        "spans": spans,
    }


def load_shard_tokens(shard_path: str, vocab_size: int) -> list[int]:
    with open(shard_path, "rb") as f:
        return _from_bytes(f.read(), vocab_size)


def load_shard_spans(spans_path: str) -> dict:
    return read_json(spans_path)


def verify_shard(shard_path: str, expected_hash: str) -> bool:
    """Re-hash a shard on disk. Used by the audit and by resume."""
    with open(shard_path, "rb") as f:
        return sha256_bytes(f.read()) == expected_hash
