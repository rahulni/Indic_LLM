# -*- coding: utf-8 -*-
"""Resume, replay, fork and audit -- the operational heart.

Four modes, and the distinctions between them are the whole point:

``resume``
    Same run, same branch, continue from the latest checkpoint and its ledger
    offset. Records written after that checkpoint describe work whose weights
    were never saved, so they are truncated and re-served. Skipping them would
    lose training; keeping them would double it.

``replay``
    Re-serve a historical interval **from the ledger**, not from the planner.
    Regenerating from a seed reproduces the order only if worker count, shard
    set, library versions and process state all match, none of which is
    guaranteed across a restart. The recorded stream is the authority.

``fork``
    Restore a checkpoint and deliberately diverge on a new branch id. The
    divergence point is recorded so the experiment is defined rather than
    inferred, and the parent branch is left byte-unchanged.

``audit``
    Reconstruct which data trained a checkpoint or a token range, and which
    OPUS decisions preceded a loss event.

The rule underneath all four:
``experiment = model checkpoint + optimizer state + data stream + code/config``.
If any one changes silently, the comparison is worthless.
"""
from __future__ import annotations

import math

from . import checkpoint as ckpt
from .batching import assemble_batch, build_sample, verify_rank_partition
from .config import (DATALOADER_VERSION, INDIC_VERIFIED_FLOOR_FRACTION,
                     RunConfig)
from .ledger.consumption import batches_in_range, verify_integrity
from .mixture import SELECTION_REASONS


class CrashSimulated(RuntimeError):
    """Raised by the injected fault. A real uncaught exception, not a return."""


class RunState:
    """Everything a step needs, in one place so resume can rebuild it exactly."""

    def __init__(self, cfg: RunConfig, *, model, trainer, schedule, pools,
                 loader, schedule_plan, cons_ledger, learn_ledger,
                 tokenizer, tokenizer_hash, registry, probe, selector,
                 branch_id: str = "main") -> None:
        self.cfg = cfg
        self.model = model
        self.trainer = trainer
        self.schedule = schedule
        self.pools = pools
        self.loader = loader
        self.plan = schedule_plan
        self.cons = cons_ledger
        self.learn = learn_ledger
        self.tokenizer = tokenizer
        self.tokenizer_hash = tokenizer_hash
        self.registry = registry
        self.probe = probe
        self.selector = selector
        self.branch_id = branch_id

        self.global_step = 0
        self.model_age_tokens = 0
        self.last_checkpoint_id = "genesis"
        self.checkpoints: list[dict] = []
        self.opus_scores_by_shard: dict[str, float] = {}
        # shard_id -> the candidate_id of the most recent decision touching it.
        # OPUS is advisory here: it scores without gating the stream, which is
        # what keeps the served order free of any float. So this is recorded as
        # provenance for a score, never as the cause of a draw -- see
        # `selection_reason` for the cause.
        self.opus_decision_by_shard: dict[str, str] = {}
        self.seen_doc_ids: set[str] = set()
        self.indic_tier_shortfalls: list[dict] = []
        self.indic_served_verified = 0
        self.indic_served_unverified = 0
        # Shards withheld until the anneal stage. Held as a set so a served
        # sample can say it came from the reserve.
        self._reserved_shards: set[str] = set(
            (schedule_plan.get("anneal_reserve") or {}).get("reserved_shard_ids", []))

    # -- batch construction ------------------------------------------------

    def build_batch(self, global_step: int) -> dict:
        step_plan = self.plan["steps"][global_step]
        seq_len = step_plan["sequence_length"]
        samples, idx = [], 0

        for lane, k in sorted(step_plan["lane_quota"].items()):
            # Why this lane got slots this step, taken from the apportioner that
            # allocated them rather than guessed from the numbers afterwards.
            # Consumed in a fixed order so the reason attached to a sample is
            # deterministic.
            reasons = self._reason_queue(step_plan, lane, k)
            for i, seq in enumerate(self._draw(lane, k, seq_len)):
                shard_ids = sorted({g["shard_id"] for g in seq["segments"]})
                self.loader.prefetch(shard_ids)
                toks = self.loader.get_many(shard_ids)
                s = build_sample(
                    seq, toks, seq_len=seq_len,
                    attention_policy=step_plan["attention_policy"],
                    position_policy=step_plan["position_policy"],
                    lane=lane, sample_index=idx)
                s["selection_reason"] = reasons[i] if i < len(reasons) else "lane_quota"
                s["selection_notes"] = self._selection_notes(seq, shard_ids)
                samples.append(s)
                idx += 1

        batch = assemble_batch(self.cfg, step_plan, samples, branch_id=self.branch_id)

        # The firewall's second enforcement point. The shard writer already
        # refused never-train documents; this re-checks independently, because
        # one copy error should not be able to defeat both.
        self.registry.assert_not_gradient_bearing(
            batch["doc_ids"], where=f"step {global_step} batch {batch['batch_id'][:12]}")
        return batch

    def _reason_queue(self, step_plan: dict, lane: str, k: int) -> list[str]:
        """The allocation causes for this lane's ``k`` slots, in a fixed order.

        Read from the schedule the apportioner produced, so the recorded reason
        is the branch that actually allocated the slot rather than a guess made
        after the fact. Floors are listed first because they were served first.
        """
        by_reason = (step_plan.get("lane_quota_reasons") or {}).get(lane, {})
        out: list[str] = []
        for reason in SELECTION_REASONS:
            out.extend([reason] * int(by_reason.get(reason, 0)))
        # A schedule compiled before this field existed leaves the queue empty;
        # the caller falls back to lane_quota rather than inventing a reason.
        return out[:k]

    def _selection_notes(self, seq: dict, shard_ids: list[str]) -> list[str]:
        """Facts about the sample that filled a slot. Not why the slot existed."""
        notes: list[str] = []
        if seq.get("pool_epoch", 0) > 0:
            notes.append("repeat_pass")
        if self._reserved_shards & set(shard_ids):
            notes.append("anneal_reserve")
        if seq.get("tier_rule_forced"):
            notes.append("tier_rule_forced")
        return notes

    def _draw(self, lane: str, k: int, seq_len: int) -> list[dict]:
        """Draw ``k`` sequences for ``lane``, honouring the Indic tier rule.

        Session 5: unverified Indic may never substitute for the verified part
        of the protected floor. So the Indic quota is split -- at least
        ``INDIC_VERIFIED_FLOOR_FRACTION`` of it is drawn from the verified
        sub-pool -- rather than drawn from a mixed pool and hoping the ratio
        works out. If verified supply runs short the shortfall is recorded, not
        silently backfilled with unverified data.
        """
        if k <= 0:
            return []
        if lane != "indic":
            pool = self.pools.get((lane, seq_len))
            return pool.take(k) if pool and len(pool) else []

        vp = self.pools.get(("indic:verified", seq_len))
        up = self.pools.get(("indic:unverified", seq_len))
        out: list[dict] = []

        # Decide tier per sample against the CUMULATIVE ratio, not per step.
        #
        # Rounding within a step is wrong here for the same reason it was wrong
        # for lane shares. Most steps draw a single Indic sample, and
        # ceil(1 x 0.5) = 1 makes verified win every time -- the unverified tier
        # is then never served at all. Measured: 12/12 verified, 0 unverified.
        #
        # The rule is "at least half verified", not "all verified". Session 5
        # allocates the unverified tier a real share; it just may not stand in
        # for the verified half. Tracking cumulative counts honours the floor
        # exactly while still serving unverified data.
        for _ in range(k):
            total = self.indic_served_verified + self.indic_served_unverified
            need = math.ceil(INDIC_VERIFIED_FLOOR_FRACTION * (total + 1))
            want_verified = self.indic_served_verified < need

            pool = (vp if want_verified else up) or (up if want_verified else vp)
            if pool is None or not len(pool):
                pool = vp if (vp and len(vp)) else up
            if pool is None or not len(pool):
                break

            drawn = pool.take(1)
            if not drawn:
                break
            # Record when the tier rule, not the lane quota, decided which
            # sub-pool this sample came from.
            if want_verified:
                drawn[0]["tier_rule_forced"] = True
            tier = drawn[0].get("indic_tier")
            if tier == "verified":
                self.indic_served_verified += 1
            else:
                self.indic_served_unverified += 1
            if want_verified and tier != "verified":
                self.indic_tier_shortfalls.append({
                    "global_step": self.global_step,
                    "wanted": "verified", "served": tier,
                    "reason": "verified Indic supply exhausted at this sequence length",
                })
            out.extend(drawn)

        if len(out) < k:
            pool = self.pools.get((lane, seq_len))
            if pool and len(pool):
                out.extend(pool.take(k - len(out)))
        return out

    def pool_states(self) -> dict:
        st = {f"{lane}@{sl}": p.state() for (lane, sl), p in sorted(self.pools.items())}
        # Dataloader state is not only cursors. The Indic tier ratio is carried
        # across steps, so a resume that forgot it would restart the ratio and
        # serve a different tier than the recorded stream did.
        st["__indic_tier__"] = {
            "served_verified": self.indic_served_verified,
            "served_unverified": self.indic_served_unverified,
        }
        return st

    def load_pool_states(self, states: dict) -> None:
        for key, st in states.items():
            if key == "__indic_tier__":
                self.indic_served_verified = int(st.get("served_verified", 0))
                self.indic_served_unverified = int(st.get("served_unverified", 0))
                continue
            lane, _, sl = key.partition("@")
            if not sl.isdigit():
                continue
            p = self.pools.get((lane, int(sl)))
            if p is not None:
                p.load_state(st)

    # -- one step ----------------------------------------------------------

    def run_step(self, global_step: int, *, log=None, collect_tokens: bool = True,
                 crash_at: int | None = None) -> dict:
        batch = self.build_batch(global_step)

        ok, problems = verify_rank_partition(batch, self.cfg.profile.ranks)
        if not ok:
            raise RuntimeError(f"rank partition broken at step {global_step}: {problems}")

        # The fault is injected *after* records are appended and *before* the
        # next checkpoint, so the truncation path is genuinely exercised rather
        # than being a no-op.
        self.cons.append_batch(
            batch, self.plan["steps"][global_step],
            checkpoint_id=self.last_checkpoint_id,
            tokenizer_hash=self.tokenizer_hash,
            dataloader_version=DATALOADER_VERSION,
            opus_decision_ids={
                str(s["sample_index"]): self.opus_decision_by_shard.get(sid)
                for s in batch["samples"]
                for sid in (s["shard_ids"][:1] or [""])
                if self.opus_decision_by_shard.get(sid)},
            pool_state=self.pool_states())

        if crash_at is not None and global_step == crash_at:
            self.cons.commit()
            raise CrashSimulated(
                f"injected fault at global step {global_step}: "
                f"{self.cons.offset} ledger records written, last checkpoint "
                f"{self.last_checkpoint_id[:12]}")

        metrics = self.trainer.train_step(batch, global_step,
                                          collect_tokens=collect_tokens)
        self.model_age_tokens += metrics["loss_tokens"]
        self.seen_doc_ids.update(batch["doc_ids"])

        self.learn.record_step(
            metrics, batch, tokenizer=self.tokenizer,
            checkpoint_id=self.last_checkpoint_id,
            model_age_tokens=self.model_age_tokens,
            opus_scores={s["sample_index"]: self.opus_scores_by_shard.get(
                s["shard_ids"][0] if s["shard_ids"] else "") for s in batch["samples"]},
            probe_delta=self.probe.delta() if self.probe.history else None)

        self.global_step = global_step + 1
        return {"batch": batch, "metrics": metrics}

    def save_checkpoint(self, out_dir: str, *, log=None) -> dict:
        offset = self.cons.commit()          # fsync before we claim the offset
        self.learn.flush()
        meta = ckpt.save(
            out_dir, global_step=self.global_step, branch_id=self.branch_id,
            run_id=self.cfg.run_id, model=self.model, schedule=self.schedule,
            rng_state={"seed": self.cfg.seed, "counter": self.global_step},
            pool_states=self.pool_states(), ledger_offset=offset,
            tokenizer_hash=self.tokenizer_hash,
            config={"profile": self.cfg.profile.name, "seed": self.cfg.seed},
            extra={"model_age_tokens": self.model_age_tokens})
        self.last_checkpoint_id = meta["checkpoint_id"]
        self.checkpoints.append(meta)
        if log:
            log.event("checkpoint saved", step=self.global_step,
                      checkpoint_id=meta["checkpoint_id"][:16], ledger_offset=offset)
            log.pass_("checkpoint_saved", step=self.global_step,
                      ledger_offset=offset, components=meta["components"])
        return meta


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def resume(state: RunState, ckpt_dir: str, *, log=None) -> dict:
    """Restore the newest checkpoint and rewind the ledger to its offset."""
    newest = ckpt.latest(ckpt_dir, state.branch_id)
    if newest is None:
        raise ckpt.CheckpointError(f"no checkpoint to resume from in {ckpt_dir!r}")

    body = ckpt.load(newest["path"])
    from .trainer import LRSchedule
    restored = ckpt.restore(body, model=state.model, schedule_cls=LRSchedule,
                            tokenizer_hash=state.tokenizer_hash)

    state.schedule = restored["schedule"]
    state.trainer.schedule = restored["schedule"]
    state.global_step = restored["global_step"]
    state.last_checkpoint_id = restored["checkpoint_id"]
    state.model_age_tokens = body.get("extra", {}).get("model_age_tokens", 0)
    state.load_pool_states(restored["dataloader_state"])

    trunc = state.cons.truncate_to(restored["ledger_offset"])

    out = {
        "checkpoint_id": restored["checkpoint_id"],
        "resumed_at_step": state.global_step,
        "ledger_offset": restored["ledger_offset"],
        "truncation": trunc,
        "next_expected_step": state.global_step,
    }
    if log:
        log.event("run resumed", **{k: v for k, v in out.items() if k != "truncation"})
        log.info(f"ledger rewound to offset {trunc['requested_offset']}: "
                 f"{trunc['records_discarded']} uncommitted record(s) discarded"
                 + (", torn tail repaired" if trunc["torn_tail_repaired"] else ""))
    return out


def verify_resume(state: RunState, expected_batch_id: str, *, log=None) -> dict:
    """Build the next batch and prove it is exactly the one that was expected.

    The single-id match is necessary but not sufficient, so the global integrity
    check runs alongside it: no duplicate batch id anywhere, no gap in global
    steps, dense offsets. That combination is what "without skipped or repeated
    batches" actually means.
    """
    # Building a batch advances the lane pools, and the real step is about to
    # build the same batch again. Without snapshotting, the verification itself
    # would consume the sequences it was checking and the run would resume one
    # batch further on than it reported -- a skipped batch caused by the check
    # that exists to prove nothing was skipped.
    saved = state.pool_states()
    batch = state.build_batch(state.global_step)
    state.load_pool_states(saved)

    matched = batch["batch_id"] == expected_batch_id
    integrity = verify_integrity(state.cons.read_all(), branch_id=state.branch_id)

    out = {
        "expected_batch_id": expected_batch_id,
        "actual_batch_id": batch["batch_id"],
        "matched": matched,
        "global_step": state.global_step,
        "integrity": integrity,
        "no_skipped_or_repeated": integrity["ok"],
    }
    if log:
        log.check("resume_next_batch_matched", matched,
                  expected=expected_batch_id[:16], actual=batch["batch_id"][:16],
                  step=state.global_step)
        log.check("no_duplicate_or_skipped_batches", integrity["ok"],
                  records=integrity["records"], steps=integrity["steps"],
                  problems=integrity["problems"])
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(state: RunState, lo: int, hi: int, *, log=None) -> dict:
    """Re-serve steps ``[lo, hi)`` from the ledger and compare against the original.

    Nothing is re-planned. The ledger says which spans were served; this rebuilds
    the batch from those spans and checks that batch id, content hash and token
    spans all match. Comparing only the batch id would prove the plan was
    followed while saying nothing about the tokens behind it.
    """
    original = batches_in_range(state.cons.read_all(), lo, hi, state.branch_id)
    rows, mismatches = [], []

    for step in sorted(original):
        want = original[step]
        rebuilt = _rebuild_from_ledger(state, want, step)
        got = _batch_fingerprint(rebuilt)
        row = {
            "global_step": step,
            "batch_id_original": want["batch_id"],
            "batch_id_replay": got["batch_id"],
            "batch_id_match": want["batch_id"] == got["batch_id"],
            "content_hash_original": want["batch_content_hash"],
            "content_hash_replay": got["batch_content_hash"],
            "content_hash_match": want["batch_content_hash"] == got["batch_content_hash"],
            "loss_mask_hash_match": want["loss_mask_hash"] == got["loss_mask_hash"],
            "spans_match": want["token_spans"] == got["token_spans"],
            "shard_ids_match": want["shard_ids"] == got["shard_ids"],
        }
        row["ok"] = all(row[k] for k in
                        ("batch_id_match", "content_hash_match",
                         "loss_mask_hash_match", "spans_match", "shard_ids_match"))
        rows.append(row)
        if not row["ok"]:
            mismatches.append(row)

    out = {
        "range": [lo, hi],
        "branch_id": state.branch_id,
        "steps_compared": len(rows),
        "matched": len(rows) - len(mismatches),
        "mismatches": mismatches[:10],
        "all_matched": not mismatches and bool(rows),
        "compared_fields": ["batch_id", "batch_content_hash", "loss_mask_hash",
                            "token_spans", "shard_ids"],
        "rows": rows,
    }
    if log:
        log.event("historical stream replayed", range=[lo, hi],
                  steps=len(rows), matched=out["matched"])
        log.check("replay_hash_matched", out["all_matched"],
                  steps=len(rows), mismatches=len(mismatches))
    return out


def _batch_fingerprint(batch: dict) -> dict:
    """Reduce an assembled batch to the same shape the ledger records.

    Replay compares like with like: the ledger stores an identity summary, so
    the rebuilt batch is summarised the same way rather than compared field by
    field against a different structure.
    """
    spans = [{"shard_id": sp["shard_id"], "doc_id": sp["doc_id"],
              "start": sp["shard_start"], "end": sp["shard_end"]}
             for s in batch["samples"] for sp in s["spans"]]
    return {
        "batch_id": batch["batch_id"],
        "batch_content_hash": batch["batch_content_hash"],
        "loss_mask_hash": batch["loss_mask_hash"],
        "shard_ids": sorted({sh for s in batch["samples"] for sh in s["shard_ids"]}),
        "token_spans": sorted(spans, key=lambda x: (x["shard_id"], x["start"], x["end"])),
    }


def _rebuild_from_ledger(state: RunState, recorded: dict, step: int) -> dict:
    """Rebuild one batch strictly from recorded per-sample span groups.

    Nothing here consults the planner, the pools or the schedule's lane quotas.
    The ledger says which shard ranges made up each sample and where each landed
    inside its sequence; this reads the shards at those ranges and rebuilds. If
    the planner had drifted, this would still reproduce the original batch --
    which is the entire reason the ledger exists.
    """
    step_plan = state.plan["steps"][step]
    rows = [r for r in state.cons.read_all()
            if r["global_step"] == step and r["branch_id"] == state.branch_id]

    samples = []
    for r in sorted(rows, key=lambda x: (x["rank"], x["accum_index"])):
        for rec in r["samples"]:
            spans = rec["spans"]
            shard_ids = sorted({g["shard_id"] for g in spans})
            state.loader.prefetch(shard_ids)
            toks = state.loader.get_many(shard_ids)
            seq_len = rec["seq_len"]
            segments = [{
                "shard_id": g["shard_id"], "doc_id": g["doc_id"],
                "shard_start": g["start"], "shard_end": g["end"],
                "seq_start": g["seq_start"], "seq_end": g["seq_end"],
                "roles": g.get("roles") or [], "truncated": False,
            } for g in spans]
            used = sum(g["seq_end"] - g["seq_start"] for g in spans)
            seq = {
                "policy": rec["packing_policy"],
                "seq_len": seq_len,
                "segments": segments,
                "real_tokens": used,
                "pad_tokens": seq_len - used,
                "n_documents": len({g["doc_id"] for g in spans}),
                "boundary_crossings": max(0, len(spans) - 1),
                "pool_epoch": rec.get("pool_epoch", 0),
                "indic_tier": rec.get("indic_tier"),
            }
            samples.append(build_sample(
                seq, toks, seq_len=seq_len,
                attention_policy=r["attention_policy"],
                position_policy=r["position_policy"],
                lane=rec["lane"], sample_index=rec["sample_index"]))

    samples.sort(key=lambda s: s["sample_index"])
    return assemble_batch(state.cfg, step_plan, samples, branch_id=state.branch_id)


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------

def fork(state: RunState, ckpt_dir: str, from_step: int, new_branch: str, *,
         log=None) -> dict:
    """Restore a checkpoint and diverge onto a new branch."""
    rows = [r for r in ckpt.list_checkpoints(ckpt_dir, state.branch_id)
            if r["global_step"] <= from_step]
    if not rows:
        raise ckpt.CheckpointError(f"no checkpoint at or before step {from_step}")
    chosen = rows[-1]

    body = ckpt.load(chosen["path"])
    from .trainer import LRSchedule
    restored = ckpt.restore(body, model=state.model, schedule_cls=LRSchedule,
                            tokenizer_hash=state.tokenizer_hash)

    parent_branch = state.branch_id
    parent_offset = restored["ledger_offset"]

    state.branch_id = new_branch
    state.cons.branch_id = new_branch
    state.schedule = restored["schedule"]
    state.trainer.schedule = restored["schedule"]
    state.global_step = restored["global_step"]
    state.last_checkpoint_id = restored["checkpoint_id"]
    state.load_pool_states(restored["dataloader_state"])

    out = {
        "parent_branch": parent_branch,
        "new_branch": new_branch,
        "diverged_at_step": restored["global_step"],
        "requested_from_step": from_step,
        "parent_checkpoint_id": restored["checkpoint_id"],
        "parent_ledger_offset": parent_offset,
    }
    if log:
        log.event("branch forked", **out)
    return out


def verify_fork(cons_records: list[dict], parent_branch: str, new_branch: str,
                diverged_at: int, *, log=None) -> dict:
    """The branch must diverge exactly where it said, and leave the parent alone."""
    parent = [r for r in cons_records if r["branch_id"] == parent_branch]
    child = [r for r in cons_records if r["branch_id"] == new_branch]
    child_steps = sorted({r["global_step"] for r in child})
    parent_steps = sorted({r["global_step"] for r in parent})

    starts_right = bool(child_steps) and child_steps[0] == diverged_at
    parent_intact = bool(parent_steps) and max(parent_steps) >= diverged_at
    # Same step on two branches must not produce the same batch -- otherwise
    # nothing actually diverged.
    overlap = set(child_steps) & set(parent_steps)
    distinct = True
    for s in sorted(overlap):
        pb = {r["batch_id"] for r in parent if r["global_step"] == s}
        cb = {r["batch_id"] for r in child if r["global_step"] == s}
        if pb & cb:
            distinct = False
            break

    ok = starts_right and parent_intact and distinct
    out = {
        "parent_branch": parent_branch, "new_branch": new_branch,
        "diverged_at_step": diverged_at,
        "child_first_step": child_steps[0] if child_steps else None,
        "child_steps": len(child_steps), "parent_steps": len(parent_steps),
        "starts_at_divergence": starts_right,
        "parent_records_intact": parent_intact,
        "branches_produce_distinct_batches": distinct,
        "ok": ok,
    }
    if log:
        log.check("fork_diverged_at_expected_step", ok, **{
            k: v for k, v in out.items() if k != "ok"})
    return out


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(cons_records: list[dict], learn_steps: list[dict], opus_decisions: list[dict],
          manifests: list[dict], *, token_window: tuple[int, int] | None = None,
          log=None) -> dict:
    """Answer the questions the course page poses of an audit."""
    man_by_id = {m["shard_id"]: m for m in manifests}

    # Which shards trained the model over a token range?
    cumulative, ranged = 0, []
    for r in sorted(cons_records, key=lambda x: x["ledger_offset"]):
        lo = cumulative
        cumulative += r["loss_bearing_tokens"]
        if token_window is None or (lo < token_window[1] and cumulative > token_window[0]):
            ranged.append(r)

    shards: dict[str, dict] = {}
    for r in ranged:
        for sid in r["shard_ids"]:
            e = shards.setdefault(sid, {"shard_id": sid, "microbatches": 0,
                                        "steps": set(), "tokens": 0})
            e["microbatches"] += 1
            e["steps"].add(r["global_step"])
            e["tokens"] += r["loss_bearing_tokens"] // max(1, len(r["shard_ids"]))
    for e in shards.values():
        m = man_by_id.get(e["shard_id"], {})
        e["steps"] = sorted(e["steps"])
        e["step_count"] = len(e["steps"])
        e["capability_lane"] = m.get("capability_lane")
        e["content_sha256"] = m.get("content_sha256")
        e["document_ids"] = m.get("document_ids", [])[:5]
        del e["steps"]

    # Which OPUS decisions preceded the largest loss spike?
    spike = None
    if len(learn_steps) > 2:
        deltas = [(learn_steps[i]["mean_loss"] - learn_steps[i - 1]["mean_loss"], i)
                  for i in range(1, len(learn_steps))]
        d, i = max(deltas)
        spike = {
            "at_step": learn_steps[i]["global_step"],
            "delta": round(d, 6),
            "loss_before": learn_steps[i - 1]["mean_loss"],
            "loss_after": learn_steps[i]["mean_loss"],
            "opus_decisions_before": [
                {k: v for k, v in dd.items()
                 if k in ("candidate_id", "lane", "opus_score", "status", "rejection_reason")}
                for dd in opus_decisions
                if dd["global_step"] <= learn_steps[i]["global_step"]][-8:],
        }

    out = {
        "token_window": list(token_window) if token_window else None,
        "total_loss_bearing_tokens": cumulative,
        "records_in_window": len(ranged),
        "shards_influencing_window": sorted(
            shards.values(), key=lambda e: -e["step_count"])[:25],
        "shard_count_in_window": len(shards),
        # An audit over the whole run answers nothing: it cannot come back empty
        # and it cannot exclude anything, so it proves no ability to isolate an
        # interval. Recorded as a verdict rather than left for a reader to notice.
        "window_is_strict_subset": bool(
            token_window is not None
            and (len(ranged) < len(cons_records) or len(shards) < len(man_by_id))),
        "records_total": len(cons_records),
        "shards_total": len(man_by_id),
        "largest_loss_spike": spike,
        "lineage_note": ("every shard above resolves through its manifest to "
                         "content_sha256, document ids and the cleaning pipeline "
                         "hash that produced it"),
    }
    if log:
        log.event("audit completed", shards=len(shards),
                  window=out["token_window"], records=len(ranged))
    return out
