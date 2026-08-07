# -*- coding: utf-8 -*-
"""Hashing and JSON I/O.

Every identity in this system is a sha256 over canonical bytes. Nothing hashes
text in text mode, and nothing hashes a structure without first putting it into
a canonical JSON form, because ``json.dumps`` with default settings will happily
emit two different byte strings for the same dict.

The read/write helpers follow the shape used by the session-4 cleaning pipeline
(``4_model_data/assignment/pipeline/common.py``) so manifests produced here are
comparable with the ones produced there.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from typing import Any, Iterable, Iterator

from .determinism import canonical_bytes, canonical_text

CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hash of the *canonical* form -- LF newlines, NFC. See determinism.py."""
    return hashlib.sha256(canonical_bytes(text)).hexdigest()


def sha256_file(path: str) -> str:
    """Hash of raw file bytes, read in binary. Never text mode."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """The one JSON form we hash.

    ``sort_keys`` removes dict-order dependence, the compact separators remove
    whitespace drift, and ``ensure_ascii=False`` keeps Indic text as real
    characters so the same document hashes the same whether or not it went
    through an ASCII-escaping round trip.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def hash_many(parts: Iterable[Any]) -> str:
    """Hash an ordered sequence of parts with explicit separators.

    Length-prefixing each part prevents the ambiguity where ``("ab", "c")`` and
    ``("a", "bc")`` would otherwise produce the same digest -- which matters
    because batch ids are built from concatenated shard ids and offsets.
    """
    h = hashlib.sha256()
    for p in parts:
        b = canonical_json(p).encode("utf-8") if not isinstance(p, (bytes, bytearray)) else bytes(p)
        h.update(str(len(b)).encode("ascii"))
        h.update(b"\x1f")
        h.update(b)
        h.update(b"\x1e")
    return h.hexdigest()


def hash_source_tree(root: str, suffixes: tuple[str, ...] = (".py",)) -> dict:
    """Per-file hashes of the implementation itself.

    Mirrors ``stage8_manifest.py``'s ``cleaning_script_per_file_hash``: a shard
    records which code produced it, so a manifest can be matched back to an
    exact revision of the pipeline rather than a version string somebody
    remembered to bump.
    """
    per_file: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(suffixes):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                per_file[rel] = sha256_file(full)
    return {
        "per_file": per_file,
        "combined": hash_obj(per_file),
        "file_count": len(per_file),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any, *, canonical: bool = False) -> str:
    """Write JSON and return its content hash.

    ``canonical=True`` writes the exact bytes we hash (no indentation). Use it
    for anything whose file bytes are compared between runs; use the default
    indented form for artifacts a human reads.
    """
    ensure_dir(os.path.dirname(path))
    text = canonical_json(obj) if canonical else json.dumps(
        obj, sort_keys=True, indent=2, ensure_ascii=False
    )
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        if not canonical:
            f.write("\n")
    return sha256_text(text)


def read_json(path: str) -> Any:
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[dict]) -> str:
    ensure_dir(os.path.dirname(path))
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(canonical_json(r) + "\n")
    return sha256_file(path)


def read_jsonl(path: str, *, tolerate_torn_tail: bool = False) -> list[dict]:
    """Read JSONL.

    ``tolerate_torn_tail`` exists because a crash mid-append can leave a partial
    final line. The consumption ledger reader needs to survive that and report
    it rather than raising, since recovering from exactly that situation is the
    thing crash-recovery has to prove.
    """
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with io.open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_torn_tail and i == len(lines) - 1:
                break
            raise
    return rows


def iter_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_text(path: str, text: str) -> str:
    """Write canonical text (LF, NFC) and return its hash."""
    ensure_dir(os.path.dirname(path))
    text = canonical_text(text)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return sha256_text(text)


def read_text(path: str) -> str:
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        return canonical_text(f.read())


def atomic_write_bytes(path: str, data: bytes, *, retries: int = 5) -> str:
    """Write bytes atomically: temp file -> fsync -> os.replace.

    A crash during a checkpoint write must never leave a half-written
    checkpoint, because resume would then load garbage and the recovery
    evidence would be meaningless. ``os.replace`` is atomic on both POSIX and
    Windows; ``os.rename`` is not, on Windows, when the target exists.

    The retry exists because of a measured failure, not a hypothetical one. On
    Windows, replacing a *large* existing file can fail with ``WinError 5``
    (access denied) when something else -- an antivirus scanner, an indexer --
    still holds a handle to it moments after it was written. That never showed up
    on the stdlib backend's ~1MB checkpoints, and it killed a transformer run at
    the last step on a 124MB weights sidecar. Backing off and retrying is the
    standard mitigation; the write stays atomic because ``os.replace`` either
    happens or does not.
    """
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    delay = 0.1
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == retries - 1:
                # Leave the temp file in place: it is the complete, fsync'd data,
                # and deleting it would destroy the only good copy.
                raise
            time.sleep(delay)
            delay *= 2
    return sha256_bytes(data)
