# -*- coding: utf-8 -*-
"""Compile the curriculum into per-step lane quotas.

Session 5 described the mixture in human terms -- lane shares, protected floors,
an anneal reserve, a sequence ladder. This stage has to turn that into a number
for every step: *at step 47, which lane does each of the six samples come from,
and how long is a sequence?*

Four things this has to get right, all of them named in the course page:

**Per-stage sequence length.** The stage record carries ``sequence_length``.
Phases A/B run short and C/D run long, with the sample count adjusted so tokens
per step stay roughly constant. Without this, "reserve long documents for the
long-context rung" has no rung to reserve them for.

**Crossfade.** Stage transitions ramp rather than step. A share that jumps from
16% to 32% at a single step is a discontinuity the model has to absorb all at
once; blending it over a warmup band is what session 5 means by annealing.

**Protected floors.** Enforced over a sliding window, not per step -- a floor of
1% cannot be met by a single step that only has six samples. The window is where
the guarantee lives.

**Anneal reserve.** A fraction of supply is withheld entirely until the final
stage. This is distinct from crossfade: crossfade blends *weights*, the reserve
withholds *shards* so they cannot be consumed early.
"""
from __future__ import annotations

import math

from .config import (ANNEAL_RESERVE_FRACTION, INDIC_VERIFIED_FLOOR_FRACTION,
                     LANES, STAGES, RunConfig, Stage)
from .determinism import stable_shuffle
from .scarcity import resolve as resolve_scarcity, summarise as summarise_scarcity


class LaneApportioner:
    """Apportion samples to lanes with **carry-over** between steps.

    Stateless largest-remainder apportionment does not work here, and the demo
    profile shows exactly why. With 6 samples per step, a lane weighted at 2%
    is entitled to 0.12 samples; it floors to zero and the remainder slots go to
    larger lanes. Repeat that every step and the lane receives **nothing** --
    which is what happened to ``longctx`` and ``multiling`` on the first
    compile, silently taking the 1.5% long-context floor to zero.

    The fix is to make entitlement cumulative. Each lane carries a fractional
    residual across steps; when the residual crosses one, the lane gets a
    sample. A 2% lane at 6 samples/step then receives one sample roughly every
    eight steps, which is the honest way to express a share smaller than one
    quantum of the batch.

    This is error-diffusion, and it is deterministic: the state is a plain dict
    of floats advanced in sorted key order, so two runs produce the same
    sequence. No dependence on ``hash()`` or dict insertion order.
    """

    def __init__(self) -> None:
        self.residual: dict[str, float] = {}

    def state(self) -> dict:
        return {k: round(v, 12) for k, v in sorted(self.residual.items())}

    def apportion(self, weights: dict[str, float], n: int,
                  floors: dict[str, float] | None = None) -> dict[str, int]:
        if n <= 0:
            return {}
        total = sum(weights.values()) or 1.0
        floors = floors or {}

        # Accumulate this step's entitlement on top of what was carried over.
        for lane in sorted(weights):
            self.residual[lane] = self.residual.get(lane, 0.0) + (weights[lane] / total) * n

        # Protected lanes are served first, so a floor can never be crowded out
        # by a larger lane's remainder.
        alloc: dict[str, int] = {}
        remaining = n
        for lane in sorted(floors, key=lambda l: (-floors[l], l)):
            if remaining <= 0:
                break
            if self.residual.get(lane, 0.0) >= 1.0:
                take = min(remaining, int(self.residual[lane]))
                if take:
                    alloc[lane] = alloc.get(lane, 0) + take
                    self.residual[lane] -= take
                    remaining -= take

        # Then everyone else, largest whole entitlement first.
        order = sorted(weights, key=lambda l: (-self.residual.get(l, 0.0), l))
        for lane in order:
            if remaining <= 0:
                break
            whole = int(self.residual.get(lane, 0.0))
            if whole >= 1:
                take = min(remaining, whole)
                alloc[lane] = alloc.get(lane, 0) + take
                self.residual[lane] -= take
                remaining -= take

        # Any slots left over go to the largest fractional residual. This keeps
        # the batch exactly full without letting rounding leak away capacity.
        while remaining > 0:
            lane = max(sorted(weights), key=lambda l: (self.residual.get(l, 0.0), l))
            alloc[lane] = alloc.get(lane, 0) + 1
            self.residual[lane] -= 1.0
            remaining -= 1

        return {k: v for k, v in sorted(alloc.items()) if v}


def _floor_window(profile) -> int:
    """How many steps a protected floor is measured over.

    A floor is a statement about a window, and the window has to be big enough
    for the floor to be *expressible*. With 4 samples per step, a 1% floor over
    6 steps means 24 x 0.01 = 0.24 samples -- the guarantee cannot be met by any
    integer number of samples, so the check fails on windows that are actually
    correct.

    The window is therefore derived, not guessed: it is the smallest number of
    steps in which the smallest active floor corresponds to at least one whole
    sample, with a safety factor of two so a single unlucky placement does not
    trip it.
    """
    floors = [f for st in STAGES for f in st.protected_floors.values() if f > 0]
    if not floors:
        return max(4, profile.checkpoint_every)
    smallest = min(floors)
    need = math.ceil(2.0 / (smallest * profile.samples_per_step))
    return max(profile.checkpoint_every, need)


def _floor_window_why(profile) -> dict:
    floors = [f for st in STAGES for f in st.protected_floors.values() if f > 0]
    smallest = min(floors) if floors else 0.0
    w = _floor_window(profile)
    return {
        "window_steps": w,
        "samples_per_step": profile.samples_per_step,
        "samples_in_window": w * profile.samples_per_step,
        "smallest_floor": smallest,
        "samples_the_smallest_floor_implies": round(
            smallest * w * profile.samples_per_step, 3),
        "rule": ("window = max(checkpoint_interval, ceil(2 / (smallest_floor x "
                 "samples_per_step))); a floor below one sample per window is "
                 "not a checkable guarantee"),
    }


def _blend(prev: dict[str, float], cur: dict[str, float], t: float) -> dict[str, float]:
    """Linear crossfade from ``prev`` to ``cur`` at position ``t`` in [0, 1]."""
    keys = sorted(set(prev) | set(cur))
    out = {k: (1.0 - t) * prev.get(k, 0.0) + t * cur.get(k, 0.0) for k in keys}
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def stage_step_bounds(total_steps: int) -> list[tuple[Stage, int, int]]:
    """Split the run into contiguous stage intervals by stage weight."""
    bounds: list[tuple[Stage, int, int]] = []
    cursor = 0
    for i, st in enumerate(STAGES):
        n = (total_steps - cursor if i == len(STAGES) - 1
             else max(1, round(total_steps * st.weight)))
        bounds.append((st, cursor, min(cursor + n, total_steps)))
        cursor += n
        if cursor >= total_steps:
            for st2 in STAGES[i + 1:]:
                bounds.append((st2, total_steps, total_steps))
            break
    return bounds


def reserve_anneal_shards(manifests: list[dict], seed: str,
                          fraction: float = ANNEAL_RESERVE_FRACTION) -> dict:
    """Withhold shards for the anneal stage.

    Session 5 reserves verified tiers only. Here that means: prefer shards with
    no unverified Indic content, so the final, highest-leverage stage is not
    spent on the weaker tier.
    """
    def is_verified(m: dict) -> bool:
        tiers = m.get("indic_tier_breakdown") or {}
        return "unverified" not in tiers

    eligible = sorted([m["shard_id"] for m in manifests if is_verified(m)])
    n = max(1, round(len(manifests) * max(fraction, 0.05))) if manifests else 0
    picked = set(stable_shuffle(eligible, seed=seed + "|anneal")[:n])
    return {
        "reserved_shard_ids": sorted(picked),
        "reserved_count": len(picked),
        "eligible_count": len(eligible),
        "fraction_requested": fraction,
        "rule": "verified tiers only; withheld from stages A-C and released in D",
    }


def compile_schedule(cfg: RunConfig, manifests: list[dict], *,
                     total_steps: int) -> dict:
    """Produce the per-step plan plus the reports that justify it."""
    profile = cfg.profile
    samples = profile.samples_per_step

    supply: dict[str, int] = {}
    shards_by_lane: dict[str, list[str]] = {}
    for m in manifests:
        lane = m["capability_lane"]
        supply[lane] = supply.get(lane, 0) + m["token_count"]
        shards_by_lane.setdefault(lane, []).append(m["shard_id"])
    for lane in shards_by_lane:
        shards_by_lane[lane].sort()

    anneal = reserve_anneal_shards(manifests, cfg.seed)
    reserved = set(anneal["reserved_shard_ids"])

    bounds = stage_step_bounds(total_steps)
    steps: list[dict] = []
    demand: dict[tuple[str, str], int] = {}
    # One apportioner for the whole run: residuals must carry across stage
    # boundaries too, or a small lane resets its entitlement at every stage.
    apportioner = LaneApportioner()

    for idx, (stage, lo, hi) in enumerate(bounds):
        prev_mix = bounds[idx - 1][0].mixture if idx > 0 else stage.mixture
        seq_len = cfg.seq_len_for(stage)
        span = max(1, hi - lo)
        warm = max(1, int(round(span * stage.warmup_frac)))

        for s in range(lo, hi):
            local = s - lo
            t = min(1.0, (local + 1) / warm)
            mix = _blend(prev_mix, stage.mixture, t) if local < warm else dict(stage.mixture)
            quota = apportioner.apportion(mix, samples, stage.protected_floors)

            for lane, k in quota.items():
                if k:
                    demand[(stage.stage_id, lane)] = (
                        demand.get((stage.stage_id, lane), 0) + k * seq_len)

            steps.append({
                "global_step": s,
                "stage": stage.stage_id,
                "stage_label": stage.label,
                "sequence_length": seq_len,
                "position_policy": cfg.position_policy_for(stage),
                "attention_policy": cfg.attention_policy,
                "in_warmup": local < warm,
                "warmup_t": round(t, 4),
                "lane_weights": {k: round(v, 6) for k, v in sorted(mix.items())},
                "lane_quota": {k: v for k, v in sorted(quota.items()) if v},
                "anneal_release": stage.anneal_reserve_only,
                "tokens_this_step": samples * seq_len,
            })

    # -- scarcity ---------------------------------------------------------
    #
    # Resolved twice, because the two questions are different and only one of
    # them determines the epoch count.
    #
    #   per-stage: does this stage's demand fit? Useful for reporting where the
    #              pressure sits, but it must NOT be the epoch authority --
    #              resolving each stage against the full lane supply resets the
    #              count at every stage boundary and hides real repetition. A
    #              lane consumed once in A and again in B has been seen twice.
    #
    #   whole-run: total demand against total supply. This is the number the
    #              epoch cap applies to.
    lane_total: dict[str, int] = {}
    for (stage_id, lane), want in demand.items():
        lane_total[lane] = lane_total.get(lane, 0) + want

    protected_lanes = {l for st in STAGES for l in st.protected_floors}
    decisions: list[dict] = []
    for lane, want in sorted(lane_total.items()):
        decisions.append(resolve_scarcity(
            lane, want, supply.get(lane, 0), stage_id="ALL",
            is_protected=lane in protected_lanes,
            later_stages_exist=False,
        ))

    per_stage: list[dict] = []
    for (stage_id, lane), want in sorted(demand.items()):
        stage = next(s for s in STAGES if s.stage_id == stage_id)
        have = supply.get(lane, 0)
        if stage.anneal_reserve_only:
            have = sum(m["token_count"] for m in manifests
                       if m["capability_lane"] == lane and m["shard_id"] in reserved) or have
        per_stage.append(resolve_scarcity(
            lane, want, have, stage_id=stage_id,
            is_protected=lane in stage.protected_floors,
            later_stages_exist=stage_id != STAGES[-1].stage_id,
        ))

    planned = _planned_shares(steps)
    return {
        "steps": steps,
        "stages": [{"stage": st.stage_id, "label": st.label, "weight": st.weight,
                    "step_start": lo, "step_end": hi,
                    "sequence_length": cfg.seq_len_for(st),
                    "mixture": st.mixture, "protected_floors": st.protected_floors,
                    "warmup_frac": st.warmup_frac,
                    "anneal_reserve_only": st.anneal_reserve_only}
                   for st, lo, hi in bounds],
        "supply_tokens_by_lane": dict(sorted(supply.items())),
        "shards_by_lane": {k: v for k, v in sorted(shards_by_lane.items())},
        "anneal_reserve": anneal,
        "scarcity": {"decisions": decisions,
                     "summary": summarise_scarcity(decisions),
                     "per_stage": per_stage,
                     "note": "epoch counts come from whole-run demand; the per_stage rows show where the pressure sits but reset at each stage boundary "
                             "and must not be read as epoch counts"},
        "planned_shares": planned,
        "total_steps": total_steps,
        "samples_per_step": samples,
        "protected_floor_window": _floor_window(profile),
        "protected_floor_window_rationale": _floor_window_why(profile),
        "indic_verified_floor_fraction": INDIC_VERIFIED_FLOOR_FRACTION,
        "lanes_from_session5_without_corpus": ["civic", "parallel"],
        "note": ("Session 5 lanes `civic` and `parallel` have no corpus here; "
                 "their share is folded into `web`. Phase weights, protected "
                 "floors and the sequence ladder are session 5's real values."),
    }


def _planned_shares(steps: list[dict]) -> dict:
    totals: dict[str, int] = {}
    grand = 0
    for s in steps:
        for lane, k in s["lane_quota"].items():
            tk = k * s["sequence_length"]
            totals[lane] = totals.get(lane, 0) + tk
            grand += tk
    return {
        "by_lane_tokens": dict(sorted(totals.items())),
        "by_lane_share": {k: round(v / grand, 6) for k, v in sorted(totals.items())} if grand else {},
        "total_tokens": grand,
    }


def verify_indic_tier_floor(cons_records: list[dict], *,
                            branch_id: str = "main") -> dict:
    """Unverified Indic must not substitute for the verified part of the floor.

    Session 5 states the rule and this checks it against what was actually
    served: of the Indic samples consumed, at least
    ``INDIC_VERIFIED_FLOOR_FRACTION`` must carry ``indic_tier == "verified"``.

    Checked over the whole run rather than per step, for the same reason floors
    are: with a handful of Indic samples in any one step, a 50% split is not
    expressible step by step. Where verified supply genuinely ran out the
    orchestrator records a shortfall rather than backfilling silently, so a
    failure here names a supply problem instead of a policy breach.
    """
    verified = unverified = 0
    for r in cons_records:
        if r.get("branch_id") != branch_id:
            continue
        for t, n in (r.get("indic_tiers") or {}).items():
            if t == "verified":
                verified += n
            elif t == "unverified":
                unverified += n
    total = verified + unverified
    share = (verified / total) if total else 0.0
    return {
        "indic_samples": total,
        "verified": verified,
        "unverified": unverified,
        "verified_share": round(share, 6),
        "required_share": INDIC_VERIFIED_FLOOR_FRACTION,
        "checkable": total > 0,
        "held": bool(total > 0 and share >= INDIC_VERIFIED_FLOOR_FRACTION - 1e-9),
        "rule": ("session 5: unverified Indic may never substitute for the "
                 "verified portion of the protected floor"),
    }


def verify_floors(consumed: list[dict], schedule: dict) -> dict:
    """Check protected floors over a sliding window of consumed samples.

    ``consumed`` rows are ``{"global_step", "lane", "samples", "indic_tier"}``.

    Two things this refuses to do, because both would be dishonest:

    * **Vacuously pass.** If the run is shorter than one window, zero windows
      get checked and "no violations" means nothing. That is reported as
      ``not_checkable``, never as held.
    * **Silently skip an inexpressible floor.** A 1% floor over a window holding
      40 samples implies 0.4 samples -- no integer allocation can satisfy it, so
      failing it would be a false alarm and passing it would be luck. Such
      floors are listed under ``not_expressible`` with the arithmetic, and
      excluded from the verdict rather than quietly counted as held.

    At the graded ``demo`` profile all four floors are expressible; at ``fast``
    only the largest is, and the report says so.
    """
    samples_per_step = schedule["samples_per_step"]
    total_steps = schedule["total_steps"]
    window = min(schedule["protected_floor_window"], max(4, total_steps // 3))
    stages = {s["stage"]: s for s in schedule["stages"]}

    all_floors: dict[str, float] = {}
    for s in schedule["stages"]:
        for lane, f in s.get("protected_floors", {}).items():
            all_floors[lane] = max(all_floors.get(lane, 0.0), f)
    expressible, not_expressible = {}, {}
    for lane, f in sorted(all_floors.items()):
        implied = f * window * samples_per_step
        (expressible if implied >= 1.0 else not_expressible)[lane] = {
            "floor": f, "samples_implied_by_floor": round(implied, 3),
            "window_steps": window, "samples_in_window": window * samples_per_step,
        }
    by_step: dict[int, dict[str, int]] = {}
    tiers: dict[int, dict[str, int]] = {}
    for r in consumed:
        by_step.setdefault(r["global_step"], {}).setdefault(r["lane"], 0)
        by_step[r["global_step"]][r["lane"]] += r.get("samples", 1)
        if r.get("indic_tier"):
            tiers.setdefault(r["global_step"], {}).setdefault(r["indic_tier"], 0)
            tiers[r["global_step"]][r["indic_tier"]] += r.get("samples", 1)

    steps_sorted = sorted(by_step)
    violations: list[dict] = []
    worst: dict[str, float] = {}
    windows_checked = 0

    for i in range(len(steps_sorted)):
        win = steps_sorted[i:i + window]
        if len(win) < window:
            break
        windows_checked += 1
        counts: dict[str, int] = {}
        total = 0
        for s in win:
            for lane, k in by_step[s].items():
                counts[lane] = counts.get(lane, 0) + k
                total += k
        if not total:
            continue
        stage_id = next((st["stage"] for st in schedule["stages"]
                         if st["step_start"] <= win[0] < st["step_end"]), None)
        floors = stages.get(stage_id, {}).get("protected_floors", {})
        for lane, floor in floors.items():
            share = counts.get(lane, 0) / total
            worst[lane] = min(worst.get(lane, 1.0), share)
            if lane not in expressible:
                continue        # cannot be satisfied by any integer allocation
            if share < floor:
                violations.append({"window_start": win[0], "window_end": win[-1],
                                   "lane": lane, "observed_share": round(share, 6),
                                   "floor": floor, "stage": stage_id})

    checkable = windows_checked > 0 and bool(expressible)
    return {
        "window_steps": window,
        "windows_checked": windows_checked,
        "samples_per_step": samples_per_step,
        "checked_floors": expressible,
        "not_expressible": not_expressible,
        "violations": violations[:20],
        "violation_count": len(violations),
        "worst_observed_share": {k: round(v, 6) for k, v in sorted(worst.items())},
        "checkable": checkable,
        "floors_held": bool(checkable and not violations),
        "verdict_note": (
            "floors_held is true only when at least one full window was checked "
            "AND at least one floor was expressible in it. Floors listed under "
            "not_expressible imply fewer than one sample per window, so no "
            "integer allocation can satisfy them; they are excluded from the "
            "verdict rather than counted as held."),
    }
