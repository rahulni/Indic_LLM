# -*- coding: utf-8 -*-
"""Cost: what the packing and checkpoint decisions are worth in rupees.

Not a graded requirement. It is here because it is the lens the lecture actually
teaches: the answer to "how often should we checkpoint" is not a loss threshold
or an epoch count, it is *how much GPU spend are you willing to lose to a crash*.
Padding, likewise, is not a tidiness issue -- it is a line item.

Every constant is named and sourced. The instance and hourly rate are the ones
quoted in the course lecture. The demo's own numbers are trivially small; the
point is the ratio, which is scale-invariant, and the per-1B-token projection
that makes it legible.
"""
from __future__ import annotations

from .config import COST


def build_report(perf: dict, *, checkpoint_every_steps: int, checkpoints: int,
                 resume_seconds: float, replay_seconds: float) -> dict:
    c = perf["raw_counters"]
    eff = perf["efficiency"]
    pos = c.get("positions_total", 0) or 1
    train_s = perf["train_seconds"] or 1e-9

    inr_per_hour = COST["inr_per_hour"]
    inr_per_second = inr_per_hour / 3600.0
    inr_this_run = train_s * inr_per_second

    pad = c.get("tokens_pad", 0)
    ctx = c.get("tokens_context_only", 0)
    pad_frac = pad / pos
    ctx_frac = ctx / pos

    # Projection: what the same fractions cost on a real run. Uses this run's
    # own measured positions-per-second, so it is an extrapolation of a measured
    # rate rather than an invented one.
    pos_per_s = pos / train_s
    seconds_per_billion = 1e9 / pos_per_s
    inr_per_billion = seconds_per_billion * inr_per_second

    per_step_seconds = train_s / max(1, perf["steps"])
    steps_at_risk = checkpoint_every_steps
    inr_at_risk = steps_at_risk * per_step_seconds * inr_per_second

    return {
        "constants": {
            "instance": COST["instance"],
            "inr_per_hour": inr_per_hour,
            "inr_per_second": round(inr_per_second, 8),
            "gpus": COST["gpus"],
            "source": COST["source"],
        },
        "this_run": {
            "train_seconds": round(train_s, 4),
            "inr_spent_equivalent": round(inr_this_run, 6),
            "positions_processed": pos,
            "positions_per_second": round(pos_per_s, 3),
        },
        "waste": {
            "pad_positions": pad,
            "pad_fraction": round(pad_frac, 6),
            "inr_on_padding": round(inr_this_run * pad_frac, 6),
            "context_only_positions": ctx,
            "context_only_fraction": round(ctx_frac, 6),
            "inr_on_context_only": round(inr_this_run * ctx_frac, 6),
            "note": ("context-only positions are not waste -- prompts and tool "
                     "observations must be read to be conditioned on. They are "
                     "reported beside padding because both are compute that "
                     "produces no gradient, and only one of them is avoidable."),
        },
        "projection_per_billion_positions": {
            "seconds": round(seconds_per_billion, 1),
            "inr": round(inr_per_billion, 2),
            "inr_lost_to_padding": round(inr_per_billion * pad_frac, 2),
            "basis": "this run's measured positions/second, extrapolated linearly",
        },
        "checkpoint_policy": {
            "interval_steps": checkpoint_every_steps,
            "checkpoints_written": checkpoints,
            "seconds_per_step": round(per_step_seconds, 6),
            "inr_at_risk_between_checkpoints": round(inr_at_risk, 6),
            "rule": ("checkpoint interval is a risk-in-rupees decision: pick the "
                     "amount of GPU spend you are willing to lose to a crash, and "
                     "checkpoint at that interval. Loss thresholds and epoch "
                     "counts do not answer the question."),
        },
        "recovery": {
            "resume_seconds": round(resume_seconds, 4),
            "resume_inr": round(resume_seconds * inr_per_second, 6),
            "replay_seconds": round(replay_seconds, 4),
            "replay_inr": round(replay_seconds * inr_per_second, 6),
        },
        "packing_savings": _packing_savings(perf, inr_per_billion),
    }


def _packing_savings(perf: dict, inr_per_billion: float) -> dict:
    """What the chosen packing policy saved against the pad-only baseline."""
    rows = []
    for lane, rep in perf.get("packing_by_lane", {}).items():
        by = rep.get("by_policy", {})
        base = by.get("pad_only", {})
        best_name = rep.get("policy_used")
        best = by.get(best_name, {})
        if not base or not best:
            continue
        b_seq = base.get("sequences", 0)
        x_seq = best.get("sequences", 0)
        if not b_seq:
            continue
        saved = (b_seq - x_seq) / b_seq
        rows.append({
            "lane": lane,
            "baseline_policy": "pad_only",
            "chosen_policy": best_name,
            "baseline_sequences": b_seq,
            "chosen_sequences": x_seq,
            "sequence_reduction": round(saved, 6),
            "inr_saved_per_billion_positions": round(inr_per_billion * max(0.0, saved), 2),
        })
    return {
        "by_lane": sorted(rows, key=lambda r: -r["sequence_reduction"]),
        "note": ("sequences are the unit of compute at a fixed sequence length, "
                 "so a reduction in sequence count is a proportional reduction "
                 "in spend for the same data"),
    }
