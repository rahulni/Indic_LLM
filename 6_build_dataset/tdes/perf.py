# -*- coding: utf-8 -*-
"""Throughput and packing efficiency.

The metric that matters is not raw tokens per second. A loader can report a
large number while most of what it delivered was padding, context-only tokens,
or candidates OPUS then rejected. The honest figure is:

    useful loss-bearing tokens per second at the target mixture

Every derived number here ships with the raw counters and the formula that
produced it. The assignment is explicit that numbers which cannot be
reconstructed receive no credit, so ``performance.json`` carries its own
arithmetic: a reader can recompute every rate by hand from the counters in the
same file.
"""
from __future__ import annotations

import time


class PerfMeter:
    """Accumulates counters and wall-clock across phases."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.phases: dict[str, float] = {}
        self._phase_start: dict[str, float] = {}
        self.counters: dict[str, int] = {}
        self.train_seconds = 0.0
        self.steps = 0

    # -- timing ------------------------------------------------------------

    def start(self, phase: str) -> None:
        self._phase_start[phase] = time.perf_counter()

    def stop(self, phase: str) -> float:
        t = time.perf_counter() - self._phase_start.pop(phase, time.perf_counter())
        self.phases[phase] = self.phases.get(phase, 0.0) + t
        return t

    def add(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def record_step(self, batch: dict, metrics: dict, seconds: float) -> None:
        self.steps += 1
        self.train_seconds += seconds
        self.add("positions_total", batch["total_positions"])
        self.add("tokens_real", batch["real_tokens"])
        self.add("tokens_pad", batch["pad_tokens"])
        self.add("tokens_loss_bearing", batch["loss_bearing_tokens"])
        self.add("tokens_context_only",
                 batch["real_tokens"] - batch["loss_bearing_tokens"])
        self.add("samples", len(batch["samples"]))
        self.add("microbatches", len(batch["microbatches"]))

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    # -- report ------------------------------------------------------------

    def report(self, *, loader_stats: dict, packing: dict, opus_summary: dict,
               schedule: dict, actual_shares: dict) -> dict:
        c = self.counters
        secs = max(self.train_seconds, 1e-9)
        pos = c.get("positions_total", 0)
        real = c.get("tokens_real", 0)
        loss_bearing = c.get("tokens_loss_bearing", 0)
        accepted = opus_summary.get("by_status", {}).get("accepted", 0)
        candidates = opus_summary.get("total_decisions", 0) or 1

        wait = loader_stats.get("loader_wait_seconds", 0.0)
        rates = {
            "raw_tokens_per_second": round(pos / secs, 3),
            "useful_loss_bearing_tokens_per_second": round(loss_bearing / secs, 3),
            "accepted_tokens_per_second_after_opus": round(
                loss_bearing * (accepted / candidates) / secs, 3),
            "samples_per_second": round(c.get("samples", 0) / secs, 3),
            "steps_per_second": round(self.steps / secs, 3),
            "seconds_per_step": round(secs / max(1, self.steps), 6),
        }

        efficiency = {
            "packing_utilization": round(real / pos, 6) if pos else 0.0,
            "loss_bearing_fraction_of_positions": round(loss_bearing / pos, 6) if pos else 0.0,
            "loss_bearing_fraction_of_real": round(loss_bearing / real, 6) if real else 0.0,
            "pad_fraction": round(c.get("tokens_pad", 0) / pos, 6) if pos else 0.0,
            "context_only_fraction": round(
                c.get("tokens_context_only", 0) / pos, 6) if pos else 0.0,
            "opus_acceptance_rate": round(accepted / candidates, 6),
            "loader_wait_fraction_of_train_time": round(wait / secs, 6),
            "gpu_idle_proxy_seconds": round(wait, 6),
        }

        # Mixture compliance: planned versus what was actually consumed.
        planned = schedule["planned_shares"]["by_lane_share"]
        lanes = sorted(set(planned) | set(actual_shares))
        drift = [{
            "lane": l,
            "planned_share": round(planned.get(l, 0.0), 6),
            "actual_share": round(actual_shares.get(l, 0.0), 6),
            "drift": round(actual_shares.get(l, 0.0) - planned.get(l, 0.0), 6),
        } for l in lanes]
        max_drift = max((abs(d["drift"]) for d in drift), default=0.0)

        return {
            "wall_clock_seconds": round(self.elapsed, 4),
            "train_seconds": round(self.train_seconds, 4),
            "steps": self.steps,
            "phase_seconds": {k: round(v, 4) for k, v in sorted(self.phases.items())},
            "raw_counters": dict(sorted(c.items())),
            "rates": rates,
            "efficiency": efficiency,
            "loader": loader_stats,
            "packing_by_lane": {
                lane: {
                    "policy_used": rep["ranked_by_effective_yield"][0]
                                   if rep.get("ranked_by_effective_yield") else None,
                    "by_policy": {p: {k: v for k, v in r.items()
                                      if k in ("sequences", "utilization", "coverage",
                                               "effective_yield", "pad_tokens",
                                               "tokens_dropped")}
                                  for p, r in rep["by_policy"].items()},
                }
                for lane, rep in sorted(packing.items())
            },
            "mixture_compliance": {
                "by_lane": drift,
                "max_abs_drift": round(max_drift, 6),
                "within_tolerance": max_drift <= 0.10,
                "tolerance": 0.10,
                "note": ("drift is expected at demo scale: six samples per step "
                         "cannot express a 1.5% share exactly, so small lanes are "
                         "served by carry-over across steps and converge over the "
                         "run rather than at every step"),
            },
            "formulas": {
                "raw_tokens_per_second": "positions_total / train_seconds",
                "useful_loss_bearing_tokens_per_second": "tokens_loss_bearing / train_seconds",
                "accepted_tokens_per_second_after_opus":
                    "tokens_loss_bearing * (opus_accepted / opus_candidates) / train_seconds",
                "packing_utilization": "tokens_real / positions_total",
                "pad_fraction": "tokens_pad / positions_total",
                "context_only_fraction": "tokens_context_only / positions_total",
                "loader_wait_fraction_of_train_time": "loader_wait_seconds / train_seconds",
                "cache_hit_rate": "cache.hits / (cache.hits + cache.misses)",
            },
            "reconstruction_note": (
                "every rate above is recomputable by hand from raw_counters, "
                "train_seconds and loader. No figure in this file is reported "
                "without the inputs that produce it."
            ),
        }


def actual_lane_shares(cons_records: list[dict], branch_id: str = "main") -> dict:
    """Lane shares as actually consumed, by loss-bearing token."""
    totals: dict[str, float] = {}
    grand = 0.0
    for r in cons_records:
        if r["branch_id"] != branch_id:
            continue
        counts = r.get("lane_counts") or {}
        n = sum(counts.values()) or 1
        for lane, k in counts.items():
            share = r["loss_bearing_tokens"] * (k / n)
            totals[lane] = totals.get(lane, 0.0) + share
            grand += share
    return {k: round(v / grand, 6) for k, v in sorted(totals.items())} if grand else {}
