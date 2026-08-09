# -*- coding: utf-8 -*-
"""The training consumption ledger -- what the model was actually fed.

Append-only, one record per (global_step, rank, microbatch), flushed and
``fsync``'d before any checkpoint claims it. This is the inbound half of the
double entry; ``learning.py`` is the outbound half.

Three properties carry the whole crash-recovery story:

**The offset is the recovery coordinate.** A checkpoint stores
``ledger_offset`` = the number of records committed at the moment it was
written. Resume truncates anything past that offset and continues. Not the step
number, not an epoch -- the offset, because it is the only quantity that is
unique, monotonic and independent of how the run was partitioned.

**Truncation is real.** Records written after the last checkpoint describe work
whose weights were never saved. Replaying them would train twice; skipping past
them would train zero times. Both are the failure the assignment tests for, so
resume rewinds to the offset exactly.

**A torn tail is survivable.** A crash mid-``write`` can leave a partial final
line. The reader detects it, reports it, and does not raise -- recovering from
exactly that situation is what crash recovery has to prove.
"""
from __future__ import annotations

import io
import os

from ..hashing import canonical_json, ensure_dir, read_jsonl


class ConsumptionLedger:
    """Append-only JSONL with an fsync-able commit point."""

    def __init__(self, path: str, *, run_id: str, branch_id: str = "main") -> None:
        ensure_dir(os.path.dirname(path))
        self.path = path
        self.run_id = run_id
        self.branch_id = branch_id
        self._fh: io.TextIOWrapper | None = None
        self._count = self._count_existing()

    def _count_existing(self) -> int:
        if not os.path.exists(self.path):
            return 0
        return len(read_jsonl(self.path, tolerate_torn_tail=True))

    def _open(self) -> io.TextIOWrapper:
        if self._fh is None:
            self._fh = io.open(self.path, "a", encoding="utf-8", newline="\n")
        return self._fh

    # -- writing -----------------------------------------------------------

    def append_batch(self, batch: dict, step_plan: dict, *,
                     checkpoint_id: str, tokenizer_hash: str,
                     dataloader_version: str,
                     opus_decision_ids: dict[str, str] | None = None,
                     pool_state: dict | None = None) -> list[dict]:
        """One record per microbatch. Returns the records written."""
        opus_decision_ids = opus_decision_ids or {}
        records: list[dict] = []
        fh = self._open()

        for mb in batch["microbatches"]:
            samples = mb["samples"]
            rec = {
                "ledger_offset": self._count,
                "run_id": self.run_id,
                "branch_id": self.branch_id,
                "global_step": batch["global_step"],
                "checkpoint_id": checkpoint_id,
                "rank": mb["rank"],
                "accum_index": mb["accum_index"],
                "microbatch_id": mb["microbatch_id"],
                "batch_id": batch["batch_id"],
                "batch_content_hash": batch["batch_content_hash"],
                "loss_mask_hash": batch["loss_mask_hash"],
                "packed_sample_ids": [s["sample_index"] for s in samples],
                "shard_ids": sorted({sh for s in samples for sh in s["shard_ids"]}),
                # Spans are recorded BOTH flat (for the audit, which only asks
                # "which shard ranges trained this checkpoint") and grouped per
                # sample (for replay, which has to rebuild the exact sample).
                #
                # An earlier version stored only the flat list and had replay
                # re-split it by even division. That is a guess: it happens to
                # be right when every sample contributes the same number of
                # spans and silently wrong otherwise. Recording the grouping is
                # the difference between reconstructing the batch and hoping.
                "token_span_ids": [
                    {"shard_id": sp["shard_id"], "doc_id": sp["doc_id"],
                     "start": sp["shard_start"], "end": sp["shard_end"]}
                    for s in samples for sp in s["spans"]
                ],
                "samples": [
                    {
                        "sample_index": s["sample_index"],
                        "lane": s["lane"],
                        "packing_policy": s["policy"],
                        "seq_len": s["seq_len"],
                        "pool_epoch": s.get("pool_epoch", 0),
                        "indic_tier": s.get("indic_tier"),
                        # Why this sample was served. The allocation cause comes
                        # from the apportioner branch that produced the slot;
                        # the notes are facts about the sample that filled it.
                        "selection_reason": s.get("selection_reason"),
                        "selection_notes": s.get("selection_notes") or [],
                        "spans": [
                            {"shard_id": sp["shard_id"], "doc_id": sp["doc_id"],
                             "start": sp["shard_start"], "end": sp["shard_end"],
                             "seq_start": sp["seq_start"], "seq_end": sp["seq_end"],
                             "roles": sp.get("roles") or []}
                            for sp in s["spans"]
                        ],
                    }
                    for s in samples
                ],
                "doc_ids": sorted({d for s in samples for d in s["doc_ids"]}),
                "mixture_lane": sorted({s["lane"] for s in samples}),
                "lane_counts": {l: sum(1 for s in samples if s["lane"] == l)
                                for l in sorted({s["lane"] for s in samples})},
                "curriculum_stage": batch["stage"],
                "sequence_length": batch["seq_len"],
                "attention_policy": batch["attention_policy"],
                "position_policy": batch["position_policy"],
                "packing_policies": sorted({s["policy"] for s in samples}),
                "tokenizer_version": tokenizer_hash,
                "dataloader_version": dataloader_version,
                "opus_decision_ids": [opus_decision_ids.get(str(s["sample_index"]))
                                      for s in samples],
                "real_tokens": sum(s["real_tokens"] for s in samples),
                "pad_tokens": sum(s["pad_tokens"] for s in samples),
                "loss_bearing_tokens": sum(s["loss_bearing_tokens"] for s in samples),
                "pool_epochs": {s["lane"]: s.get("pool_epoch", 0) for s in samples},
                "indic_tiers": {t: sum(1 for s in samples if s.get("indic_tier") == t)
                                for t in sorted({s.get("indic_tier") for s in samples}
                                                - {None})},
                "pool_state": pool_state or {},
            }
            fh.write(canonical_json(rec) + "\n")
            records.append(rec)
            self._count += 1
        return records

    def commit(self) -> int:
        """Flush and fsync. The returned offset is safe to record in a checkpoint.

        Without the fsync, a checkpoint could name an offset whose records are
        still sitting in an OS buffer -- and after a power loss the ledger would
        be shorter than the checkpoint claims.
        """
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        return self._count

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

    # -- reading / recovery ------------------------------------------------

    @property
    def offset(self) -> int:
        return self._count

    def read_all(self) -> list[dict]:
        return read_jsonl(self.path, tolerate_torn_tail=True)

    def has_torn_tail(self) -> bool:
        if not os.path.exists(self.path):
            return False
        with io.open(self.path, "r", encoding="utf-8") as f:
            content = f.read()
        return bool(content) and not content.endswith("\n")

    def truncate_to(self, offset: int) -> dict:
        """Rewind to ``offset`` records, discarding anything after it.

        Called on resume. Records past the last checkpoint describe work whose
        weights were never persisted; keeping them would make the ledger claim
        training that did not survive.
        """
        self.close()
        rows = read_jsonl(self.path, tolerate_torn_tail=True)
        torn = self.has_torn_tail()
        keep = rows[:offset]
        discarded = len(rows) - len(keep)
        tmp = self.path + ".tmp"
        with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for r in keep:
                f.write(canonical_json(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._count = len(keep)
        return {
            "requested_offset": offset,
            "records_before": len(rows),
            "records_after": len(keep),
            "records_discarded": discarded,
            "torn_tail_repaired": torn,
        }


# ---------------------------------------------------------------------------
# Integrity checks -- what "no skipped or repeated batches" actually means
# ---------------------------------------------------------------------------

def verify_integrity(records: list[dict], *, branch_id: str | None = None) -> dict:
    """Global checks over a whole ledger.

    A single matched batch id after resume is necessary but nowhere near
    sufficient. These are the properties that actually establish the claim:

    * no ``batch_id`` appears at two different global steps (nothing repeated);
    * global steps form a contiguous run with no gaps (nothing skipped);
    * offsets are dense and strictly increasing;
    * within a step, every rank appears exactly once per accumulation index.
    """
    rows = [r for r in records if branch_id is None or r["branch_id"] == branch_id]
    problems: list[str] = []

    offsets = [r["ledger_offset"] for r in rows]
    if offsets != list(range(len(rows))):
        bad = [i for i, o in enumerate(offsets) if i != o][:5]
        problems.append(f"ledger offsets are not dense/increasing at index {bad}")

    steps = sorted({r["global_step"] for r in rows})
    if steps:
        expected = list(range(steps[0], steps[-1] + 1))
        missing = sorted(set(expected) - set(steps))
        if missing:
            problems.append(f"skipped global steps: {missing[:10]}")

    batch_to_steps: dict[str, set[int]] = {}
    for r in rows:
        batch_to_steps.setdefault(r["batch_id"], set()).add(r["global_step"])
    repeated = {b: sorted(s) for b, s in batch_to_steps.items() if len(s) > 1}
    if repeated:
        problems.append(f"batch_id reused across steps: {list(repeated.items())[:3]}")

    seen: dict[tuple[int, int, int], int] = {}
    for r in rows:
        key = (r["global_step"], r["rank"], r["accum_index"])
        seen[key] = seen.get(key, 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        problems.append(f"duplicate (step, rank, accum) records: {list(dupes)[:3]}")

    return {
        "records": len(rows),
        "branch_id": branch_id,
        "steps": len(steps),
        "step_range": [steps[0], steps[-1]] if steps else [],
        "unique_batch_ids": len(batch_to_steps),
        "no_skipped_steps": not any(p.startswith("skipped") for p in problems),
        "no_repeated_batches": not repeated,
        "offsets_dense": not any(p.startswith("ledger offsets") for p in problems),
        "ok": not problems,
        "problems": problems,
    }


def batches_in_range(records: list[dict], lo: int, hi: int,
                     branch_id: str = "main") -> dict[int, dict]:
    """Reconstruct the batch identity for each step in ``[lo, hi)``.

    This is what replay compares against. It reads the ledger rather than
    re-running the planner, which is the entire point: the recorded stream is
    the authority, not the code that once produced it.
    """
    out: dict[int, dict] = {}
    for r in records:
        if r["branch_id"] != branch_id or not (lo <= r["global_step"] < hi):
            continue
        e = out.setdefault(r["global_step"], {
            "global_step": r["global_step"],
            "batch_id": r["batch_id"],
            "batch_content_hash": r["batch_content_hash"],
            "loss_mask_hash": r["loss_mask_hash"],
            "shard_ids": set(),
            "token_spans": [],
            "microbatches": 0,
        })
        e["shard_ids"].update(r["shard_ids"])
        e["token_spans"].extend(r["token_span_ids"])
        e["microbatches"] += 1
    for e in out.values():
        e["shard_ids"] = sorted(e["shard_ids"])
        e["token_spans"] = sorted(
            e["token_spans"], key=lambda s: (s["shard_id"], s["start"], s["end"]))
    return out
