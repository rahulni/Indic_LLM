# -*- coding: utf-8 -*-
"""Run logging.

Two outputs from one call:

  * ``run.log``      -- the human sequence of events the assignment lists
  * ``events.jsonl`` -- the same events structured, so the evidence bundle can
                        be built by *reading artifacts* rather than by trusting
                        in-memory booleans

The ``[PASS] <token>`` strings are fixed by the assignment and are emitted only
through :meth:`RunLog.pass_` so a token cannot drift by being typed twice.
"""
from __future__ import annotations

import io
import os
import time
from typing import Any

from .hashing import canonical_json, ensure_dir

# The five tokens named in the assignment. Emitting one not on this list is a
# programming error -- a grader greps for these exact strings.
PASS_TOKENS = {
    "tokenizer_hash_verified",
    "eval_shard_blocked",
    "checkpoint_saved",
    "resume_next_batch_matched",
    "replay_hash_matched",
    # Additional tokens we assert beyond the required five.
    "manifest_gate_rejected_bad_shard",
    "protected_floor_held",
    "no_duplicate_or_skipped_batches",
    "rank_partition_disjoint",
    "fork_diverged_at_expected_step",
    "validation_never_gradient_bearing",
    "epoch_cap_enforced",
    "indic_verified_floor_held",
    "hash_seed_independent",
}

# The 13 events run.log must contain, in order.
REQUIRED_EVENTS = [
    "shards created",
    "manifests validated",
    "evaluation data blocked",
    "mixture compiled",
    "batches packed",
    "OPUS decisions recorded",
    "checkpoint saved",
    "crash simulated",
    "run resumed",
    "historical stream replayed",
    "branch forked",
    "audit completed",
    "performance measured",
]


class RunLog:
    """Append-only log. Never buffers across a phase boundary.

    Timestamps are wall-clock and therefore go in the *human* log and in
    ``events.jsonl`` only -- never into anything hashed. Two identical runs must
    produce identical manifests, and a clock value inside a manifest would break
    that for a reason unrelated to correctness.
    """

    def __init__(self, out_dir: str, *, echo: bool = True) -> None:
        ensure_dir(out_dir)
        self.out_dir = out_dir
        self.log_path = os.path.join(out_dir, "run.log")
        self.events_path = os.path.join(out_dir, "events.jsonl")
        self.echo = echo
        self.t0 = time.time()
        self.events: list[dict] = []
        self._passes: list[str] = []
        self._fails: list[str] = []
        # Truncate: a run.log must describe exactly one run.
        for p in (self.log_path, self.events_path):
            with io.open(p, "w", encoding="utf-8", newline="\n"):
                pass

    # -- low level ----------------------------------------------------------

    def _write(self, line: str) -> None:
        with io.open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    def _event(self, kind: str, name: str, detail: dict | None = None) -> None:
        rec = {
            "seq": len(self.events),
            "t": round(time.time() - self.t0, 4),
            "kind": kind,
            "name": name,
            "detail": detail or {},
        }
        self.events.append(rec)
        with io.open(self.events_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(canonical_json(rec) + "\n")

    # -- public -------------------------------------------------------------

    def section(self, title: str) -> None:
        self._write("")
        self._write("=" * 78)
        self._write(f"  {title}")
        self._write("=" * 78)
        self._event("section", title)

    def info(self, msg: str, **detail: Any) -> None:
        self._write(f"[{self._elapsed()}] {msg}")
        self._event("info", msg, detail)

    def event(self, name: str, **detail: Any) -> None:
        """One of the 13 required run.log events."""
        self._write(f"[{self._elapsed()}] EVENT  {name}")
        self._event("event", name, detail)

    def pass_(self, token: str, **detail: Any) -> None:
        if token not in PASS_TOKENS:
            raise ValueError(f"unregistered PASS token {token!r}")
        self._write(f"[PASS] {token}")
        self._passes.append(token)
        self._event("pass", token, detail)

    def fail(self, token: str, **detail: Any) -> None:
        self._write(f"[FAIL] {token}")
        self._fails.append(token)
        self._event("fail", token, detail)

    def check(self, token: str, ok: bool, **detail: Any) -> bool:
        """Assert-and-record. Returns ``ok`` so callers can branch."""
        (self.pass_ if ok else self.fail)(token, **detail)
        return ok

    def table(self, title: str, rows: list[tuple], headers: tuple) -> None:
        """A small fixed-width table. Makes run.log readable without a viewer."""
        cols = len(headers)
        widths = [len(str(h)) for h in headers]
        for r in rows:
            for i in range(cols):
                widths[i] = max(widths[i], len(str(r[i])))
        self._write(f"  {title}")
        self._write("  " + "  ".join(str(headers[i]).ljust(widths[i]) for i in range(cols)))
        self._write("  " + "  ".join("-" * widths[i] for i in range(cols)))
        for r in rows:
            self._write("  " + "  ".join(str(r[i]).ljust(widths[i]) for i in range(cols)))

    def _elapsed(self) -> str:
        return f"{time.time() - self.t0:7.2f}s"

    # -- summary ------------------------------------------------------------

    @property
    def passes(self) -> list[str]:
        return list(self._passes)

    @property
    def fails(self) -> list[str]:
        return list(self._fails)

    def missing_required_events(self) -> list[str]:
        seen = {e["name"] for e in self.events if e["kind"] == "event"}
        return [e for e in REQUIRED_EVENTS if e not in seen]

    def elapsed(self) -> float:
        return time.time() - self.t0
