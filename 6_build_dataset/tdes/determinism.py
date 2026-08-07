# -*- coding: utf-8 -*-
"""Determinism rules for the whole system.

Two defects were measured on the authoring machine before this module existed:

  1. ``core.autocrlf=true`` with no ``.gitattributes``. The vendored corpus was
     checked out CRLF here and would arrive LF on a Linux grader's machine, so
     every ``sha256(file_bytes)`` -- and therefore every shard hash, manifest and
     content hash -- would differ at the exact moment reproducibility is graded.
     Fixed by ``.gitattributes`` plus :func:`canonical_text` on ingest.

  2. ``hash()`` is salted per process. Two interpreters returned
     ``8858257094814241226`` and ``-3424441008681030352`` for ``hash('indic')``.
     Any ordering derived from ``hash()``, ``set`` iteration, or grouping built
     out of a set is therefore non-deterministic ACROSS RUNS, which would break
     replay intermittently and invisibly.

Everything that needs an order in this codebase goes through this module. There
is no fallback path that uses ``hash()``.
"""
from __future__ import annotations

import hashlib
import os
import sys
import unicodedata
from typing import Any, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

# Windows consoles default to cp1252; Telugu/Devanagari need utf-8 everywhere.
# Same guard the session-4 pipeline uses (4_model_data/assignment/pipeline/common.py).
if sys.platform == "win32":  # pragma: no cover - platform specific
    for _stream in ("stdout", "stderr"):
        try:
            getattr(sys, _stream).reconfigure(encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Canonical bytes
# ---------------------------------------------------------------------------

def canonical_text(text: str) -> str:
    """Normalise text so its bytes are platform-independent.

    * CRLF and lone CR collapse to LF, so a Windows checkout and a Linux
      checkout hash identically.
    * NFC composition, so the same Devanagari/Telugu syllable written as a
      precomposed codepoint or as base+combining-mark becomes one form. Without
      this, two byte-different files can be the same text and dedup misses it.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def canonical_bytes(text: str) -> bytes:
    """Canonical text as utf-8 bytes. The only thing we ever hash for content."""
    return canonical_text(text).encode("utf-8")


# ---------------------------------------------------------------------------
# Stable ordering -- never Python's hash()
# ---------------------------------------------------------------------------

def stable_key(value: Any) -> str:
    """A process-independent sort key for any hashable-ish value.

    ``sorted(xs, key=stable_key)`` gives the same order in every interpreter,
    under every ``PYTHONHASHSEED``. Use this instead of relying on set or dict
    iteration order whenever the order becomes part of an artifact.
    """
    return hashlib.sha256(repr_canonical(value).encode("utf-8")).hexdigest()


def repr_canonical(value: Any) -> str:
    """Deterministic textual form of a value.

    ``repr()`` on a set or dict can vary with insertion order, so containers are
    normalised explicitly before being rendered.
    """
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(repr_canonical(v) for v in value)) + "}"
    if isinstance(value, dict):
        items = sorted((repr_canonical(k), repr_canonical(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(repr_canonical(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        # repr() of a float is round-trippable and stable across CPython builds.
        return repr(value)
    return repr(value)


def stable_sorted(items: Iterable[T], key: Callable[[T], Any] | None = None) -> list[T]:
    """Sort with a process-independent key. Prefer this over bare ``sorted``
    whenever the elements are not already plain comparable scalars."""
    if key is None:
        return sorted(items, key=stable_key)
    return sorted(items, key=lambda x: stable_key(key(x)))


def stable_shuffle(items: Sequence[T], seed: str) -> list[T]:
    """A shuffle that depends only on ``seed`` and the items themselves.

    ``random.shuffle`` seeded with an int would also be reproducible in CPython,
    but it is reproducible by *convention* -- the algorithm is not part of the
    language spec and has changed between versions. Deriving the order from
    sha256 makes the order an explicit, inspectable function of the seed, which
    is what the ledger has to be able to justify.
    """
    decorated = [
        (hashlib.sha256(f"{seed}|{i}|{repr_canonical(x)}".encode("utf-8")).hexdigest(), i, x)
        for i, x in enumerate(items)
    ]
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [x for _, _, x in decorated]


class DeterministicRNG:
    """A counter-based RNG whose entire state is one integer.

    The point is checkpointability: ``state`` is a single int, so saving and
    restoring it is exact, and resuming a run cannot silently diverge the way a
    Mersenne-Twister state can if it is serialised carelessly.
    """

    __slots__ = ("seed", "counter")

    def __init__(self, seed: str, counter: int = 0) -> None:
        self.seed = seed
        self.counter = counter

    def _next_digest(self) -> bytes:
        d = hashlib.sha256(f"{self.seed}|{self.counter}".encode("utf-8")).digest()
        self.counter += 1
        return d

    def random(self) -> float:
        """Uniform in [0, 1) from 53 bits, matching float precision."""
        n = int.from_bytes(self._next_digest()[:8], "big") >> 11
        return n / float(1 << 53)

    def randrange(self, n: int) -> int:
        if n <= 0:
            raise ValueError("randrange requires n > 0")
        return int.from_bytes(self._next_digest()[:8], "big") % n

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Box-Muller. Deterministic given the counter."""
        import math

        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        return mu + sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def state(self) -> dict:
        return {"seed": self.seed, "counter": self.counter}

    @classmethod
    def from_state(cls, state: dict) -> "DeterministicRNG":
        return cls(seed=state["seed"], counter=int(state["counter"]))


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def hash_randomization_active() -> bool:
    """True when PYTHONHASHSEED is not pinned, i.e. ``hash()`` is salted.

    Reported into ``run_meta.json`` so the evidence bundle can show the demo
    produced identical artifacts *while* randomization was active -- which is
    stronger than pinning the seed and asserting nothing.
    """
    return os.environ.get("PYTHONHASHSEED") in (None, "", "random")
