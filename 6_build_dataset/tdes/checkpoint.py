# -*- coding: utf-8 -*-
"""Checkpoints.

"A checkpoint without a data position is incomplete." Six components, and the
last one is the one people forget:

===================  =====================================================
model weights        obvious
optimizer state      momentum. Resuming without it re-estimates the moments
                     from scratch and produces a visible loss spike
scheduler state      where on the learning-rate curve we are
RNG state            so a resumed run continues the same random sequence
dataloader state     each lane pool's cursor and epoch count
**ledger offset**    which consumed records this checkpoint's weights include
===================  =====================================================

Written atomically -- temp file, ``fsync``, ``os.replace`` -- because a crash
during a checkpoint write must never leave a half-written file. Resume would
load garbage and the recovery evidence would be meaningless. ``os.replace`` is
atomic on POSIX and Windows; ``os.rename`` is not, on Windows, when the target
already exists.

The checkpoint id is a hash of its own payload, so a checkpoint cannot be
edited after the fact without the id ceasing to match.
"""
from __future__ import annotations

import hashlib
import io
import os

from .hashing import (atomic_write_bytes, canonical_json, ensure_dir, hash_obj,
                      read_json)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class CheckpointError(RuntimeError):
    pass


def checkpoint_path(out_dir: str, global_step: int, branch_id: str = "main") -> str:
    suffix = "" if branch_id == "main" else f"_{branch_id}"
    return os.path.join(out_dir, f"ckpt{suffix}_step_{global_step:06d}.json")


def _weights_path(json_path: str) -> str:
    return json_path[:-len(".json")] + ".weights.pt"


def _model_payload(model, json_path: str) -> dict:
    """The ``model`` component of a checkpoint.

    The stdlib model's parameters are a few hundred kilobytes of nested lists and
    live inline in the JSON, where they are covered directly by
    ``checkpoint_id``. A transformer's are tens of megabytes of tensors, and JSON
    is the wrong container for them -- so those go to a binary sidecar and the
    envelope carries the sidecar's **digest**. ``checkpoint_id`` therefore still
    covers the weights, transitively: change one float in the sidecar and the
    recorded digest no longer matches, exactly as if it had been inline.
    """
    if getattr(model, "checkpoint_format", "json") != "binary":
        return model.state_dict()

    import torch                                   # only on the torch path
    blob = model.state_dict_for_checkpoint()
    wpath = _weights_path(json_path)
    ensure_dir(os.path.dirname(wpath) or ".")
    buf = io.BytesIO()
    torch.save(blob, buf)
    raw = buf.getvalue()
    digest = hashlib.sha256(raw).hexdigest()

    # Idempotent, like the shard writer: the same (step, branch) can legitimately
    # be saved twice -- the periodic save at the last step and the explicit final
    # save are the same checkpoint -- and rewriting 124MB to produce identical
    # bytes is pure waste. Verified by digest rather than assumed by mtime.
    if not (os.path.exists(wpath)
            and hashlib.sha256(read_bytes(wpath)).hexdigest() == digest):
        atomic_write_bytes(wpath, raw)             # temp -> fsync -> os.replace
    return {
        "format": "binary",
        "file": os.path.basename(wpath),
        "sha256": digest,
        "bytes": len(raw),
        "config": blob["config"],
        "tensors": sorted(blob["weights"]),
        "note": ("weights live beside this file; checkpoint_id covers them via "
                 "the sha256 above, so tampering with either is detected"),
    }


def _restore_model(model, comp: dict, json_path: str) -> None:
    """Inverse of :func:`_model_payload`, verifying the sidecar before trusting it."""
    if not (isinstance(comp, dict) and comp.get("format") == "binary"):
        model.load_state_dict(comp)
        return

    import torch
    wpath = os.path.join(os.path.dirname(json_path), comp["file"])
    if not os.path.exists(wpath):
        raise CheckpointError(f"checkpoint names weights {comp['file']!r} but "
                              f"{wpath!r} does not exist")
    raw = read_bytes(wpath)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != comp["sha256"]:
        raise CheckpointError(
            f"weights {wpath!r} hash to {actual} but the checkpoint records "
            f"{comp['sha256']}; they were modified after the checkpoint was written")
    blob = torch.load(io.BytesIO(raw), weights_only=False)
    model.load_state_dict_from_checkpoint(blob)


def save(out_dir: str, *, global_step: int, branch_id: str, run_id: str,
         model, schedule, rng_state: dict, pool_states: dict,
         ledger_offset: int, tokenizer_hash: str, config: dict,
         extra: dict | None = None) -> dict:
    """Write a checkpoint and return its metadata.

    ``ledger_offset`` must come from a ``ConsumptionLedger.commit()`` that has
    already fsync'd, or the checkpoint could name records that are still only in
    an OS buffer.
    """
    ensure_dir(out_dir)
    path = checkpoint_path(out_dir, global_step, branch_id)
    body = {
        "format": "tdes-checkpoint/1",
        "run_id": run_id,
        "branch_id": branch_id,
        "global_step": global_step,
        "model": _model_payload(model, path),
        "optimizer": {"kind": "sgd_momentum",
                      "note": "momentum lives inside model.state_dict()['momentum']"},
        "scheduler": schedule.state(),
        "rng_state": rng_state,
        "dataloader_state": pool_states,
        "ledger_offset": ledger_offset,
        "tokenizer_hash": tokenizer_hash,
        "config": config,
        "extra": extra or {},
    }
    ckpt_id = hash_obj(body)
    body["checkpoint_id"] = ckpt_id

    atomic_write_bytes(path, canonical_json(body).encode("utf-8"))

    weights_path = _weights_path(path)
    return {
        "checkpoint_id": ckpt_id,
        "path": path,
        "global_step": global_step,
        "branch_id": branch_id,
        "ledger_offset": ledger_offset,
        "tokenizer_hash": tokenizer_hash,
        "components": sorted(["model", "optimizer", "scheduler", "rng_state",
                              "dataloader_state", "ledger_offset"]),
        "bytes": os.path.getsize(path) + (os.path.getsize(weights_path)
                                          if os.path.exists(weights_path) else 0),
    }


def load(path: str) -> dict:
    """Load and verify a checkpoint's self-hash."""
    if not os.path.exists(path):
        raise CheckpointError(f"no checkpoint at {path!r}")
    body = read_json(path)
    recorded = body.pop("checkpoint_id", None)
    actual = hash_obj(body)
    if recorded is None:
        raise CheckpointError(f"{path!r} has no checkpoint_id")
    if actual != recorded:
        raise CheckpointError(
            f"checkpoint {path!r} was modified after it was written "
            f"(hashes to {actual}, records {recorded})")
    body["checkpoint_id"] = recorded
    # Added after the hash check, so it can never affect the digest: restore()
    # needs to know where the file was in order to find a weights sidecar.
    body["__path__"] = path
    return body


def restore(body: dict, *, model, schedule_cls, tokenizer_hash: str | None = None,
            path: str | None = None):
    """Rebuild model and schedule from a checkpoint body.

    The tokenizer hash is checked here because a checkpoint restored under a
    different tokenizer is not a resumption -- the token ids in the shards would
    mean something else, and the loss curve would be discontinuous for a reason
    no amount of ledger inspection would explain.
    """
    if tokenizer_hash is not None and body.get("tokenizer_hash") != tokenizer_hash:
        raise CheckpointError(
            f"checkpoint was written under tokenizer {body.get('tokenizer_hash')} "
            f"but the current tokenizer is {tokenizer_hash}; the shards' token "
            f"ids would not mean the same thing")
    _restore_model(model, body["model"], path or body.get("__path__", ""))
    schedule = schedule_cls.from_state(body["scheduler"])
    return {
        "model": model,
        "schedule": schedule,
        "global_step": body["global_step"],
        "branch_id": body["branch_id"],
        "ledger_offset": body["ledger_offset"],
        "rng_state": body["rng_state"],
        "dataloader_state": body["dataloader_state"],
        "checkpoint_id": body["checkpoint_id"],
    }


def list_checkpoints(out_dir: str, branch_id: str | None = None) -> list[dict]:
    if not os.path.isdir(out_dir):
        return []
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith("ckpt") and name.endswith(".json")):
            continue
        try:
            body = read_json(os.path.join(out_dir, name))
        except Exception:
            continue
        if branch_id is not None and body.get("branch_id") != branch_id:
            continue
        rows.append({
            "path": os.path.join(out_dir, name),
            "checkpoint_id": body.get("checkpoint_id"),
            "global_step": body.get("global_step"),
            "branch_id": body.get("branch_id"),
            "ledger_offset": body.get("ledger_offset"),
        })
    return sorted(rows, key=lambda r: (r["branch_id"] or "", r["global_step"] or 0))


def prune(out_dir: str, *, keep_last: int = 3, keep_steps: set[int] | None = None,
          branch_id: str = "main") -> dict:
    """Delete old checkpoints, keeping the most recent and any still referenced.

    The lecture tells the story of a run that crashed because a 200 GB model
    could not write its checkpoint -- the disk was full of checkpoints nobody
    had deleted. Retention is therefore part of the system, not an afterthought:
    keep the newest few plus any step a fork or replay still needs, and remove
    the rest.
    """
    keep_steps = set(keep_steps or ())
    rows = list_checkpoints(out_dir, branch_id)
    if len(rows) <= keep_last:
        return {"examined": len(rows), "removed": [], "kept": [r["global_step"] for r in rows]}
    protected = {r["global_step"] for r in rows[-keep_last:]} | keep_steps
    removed = []
    for r in rows:
        if r["global_step"] in protected:
            continue
        try:
            os.remove(r["path"])
            removed.append(r["global_step"])
        except OSError:
            continue
        # The weights sidecar is the large half; deleting the envelope alone
        # would leave the disk exactly as full as before, which is the failure
        # this function exists to prevent.
        w = _weights_path(r["path"])
        if os.path.exists(w):
            try:
                os.remove(w)
            except OSError:
                pass
    return {
        "examined": len(rows),
        "removed": sorted(removed),
        "kept": sorted(protected & {r["global_step"] for r in rows}),
        "keep_last": keep_last,
        "protected_by_fork_or_replay": sorted(keep_steps),
        "why": "a full checkpoint directory is how real runs crash at save time",
    }


def latest(out_dir: str, branch_id: str = "main") -> dict | None:
    rows = list_checkpoints(out_dir, branch_id)
    return rows[-1] if rows else None
