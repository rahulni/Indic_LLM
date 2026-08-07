# -*- coding: utf-8 -*-
"""The learning ledger -- what came back out of the model.

This is the half that makes the system a double entry rather than a data loader
with good logging. The consumption ledger records what was sent; this records
what the model did with it, attached back to the source data.

The lecture is emphatic that this is the part nobody keeps, and that it can only
be produced *while* training -- recovering it afterwards means re-running the
same model over the same data at the same training state, which at real scale
never happens cleanly.

Per loss-bearing token: id, decoded preview, position, document, shard,
language, lane, EOS/boundary flag, cross-entropy, perplexity, model age,
checkpoint, stage, OPUS score, repeated-pass number.

Per shard: mean loss, high-perplexity clusters, loss delta before and after
exposure, gradient norm, gradient alignment, perplexity at EOS specifically, the
repeated-pass effect, and a usefulness classification.

The classification thresholds come from the lecture and are worth stating
because their *shape* is the interesting part -- suspiciously **low** loss is a
finding, not a success:

    <= 0.5   broken           boilerplate, duplication or a leak
    <= 1.2   already_learned  the model knows it; the compute buys little
    <  2.0   neutral
    >= 2.0   useful           ~3.4 was called a healthy mid-training shard
"""
from __future__ import annotations

import math
import os

from ..config import EOS_ID, USEFULNESS_THRESHOLDS, N_SPECIAL
from ..hashing import canonical_json, ensure_dir


def classify(mean_loss: float) -> str:
    t = USEFULNESS_THRESHOLDS
    if mean_loss <= t["broken_below"]:
        return "broken"
    if mean_loss <= t["already_learned_below"]:
        return "already_learned"
    if mean_loss < t["neutral_below"]:
        return "neutral"
    return "useful"


class LearningLedger:
    """Token-level trace plus per-shard and per-lane aggregation."""

    def __init__(self, token_path: str, shard_path: str, *,
                 max_token_records: int = 12000,
                 trace_windows: list[tuple[int, int]] | None = None) -> None:
        """Tiered storage, as the course page prescribes.

        A full token trace is the most valuable signal here and also the most
        expensive: at demo scale it is 22 MB, and at real scale it is
        unaffordable outright. The page's answer is tiering -- full traces for
        selected intervals, aggregates for everything else -- so that is what
        this does rather than writing everything and hoping.

        ``trace_windows`` are ``(lo, hi)`` step ranges kept at full resolution.
        Aggregation is unaffected: per-shard, per-lane and per-token-id
        statistics accumulate over **every** step regardless of what the trace
        keeps, so the shard report is complete for the whole run.
        """
        ensure_dir(os.path.dirname(token_path))
        self.token_path = token_path
        self.shard_path = shard_path
        self.max_token_records = max_token_records
        self.trace_windows = trace_windows or []
        self._tok_fh = None
        self.token_records_written = 0
        self.token_records_sampled_out = 0

        # Aggregates keep accumulating even once the token trace is capped or
        # outside a trace window, so the per-shard report stays complete for the
        # whole run. Tiering limits what is *written*, never what is *measured*.
        self.by_shard: dict[str, dict] = {}
        self.by_lane: dict[str, dict] = {}
        self.by_token_id: dict[int, dict] = {}
        self.eos_losses: list[float] = []
        self.step_history: list[dict] = []

    def _in_trace_window(self, step: int) -> bool:
        if not self.trace_windows:
            return True
        return any(lo <= step < hi for lo, hi in self.trace_windows)

    # -- writing -----------------------------------------------------------

    def _fh(self):
        if self._tok_fh is None:
            self._tok_fh = open(self.token_path, "a", encoding="utf-8", newline="\n")
        return self._tok_fh

    def record_step(self, metrics: dict, batch: dict, *, tokenizer,
                    checkpoint_id: str, model_age_tokens: int,
                    opus_scores: dict[int, float] | None = None,
                    probe_delta: dict | None = None) -> dict:
        """Attach a step's per-token losses back to their source spans."""
        opus_scores = opus_scores or {}
        by_index = {s["sample_index"]: s for s in batch["samples"]}
        fh = self._fh()

        for t in metrics["token_records"]:
            sample = by_index.get(t["sample_index"])
            if sample is None:
                continue
            pos = t["pos"]
            span = _span_at(sample, pos)
            loss = t["loss"]
            ppl = math.exp(min(loss, 700.0))
            tid = t["token_id"]
            is_eos = tid == EOS_ID

            if is_eos:
                self.eos_losses.append(loss)

            shard_id = span["shard_id"] if span else "?"
            doc_id = span["doc_id"] if span else "?"
            self._acc(self.by_shard, shard_id, loss, sample["lane"])
            self._acc(self.by_lane, sample["lane"], loss, sample["lane"])
            tk = self.by_token_id.setdefault(tid, {"n": 0, "sum": 0.0})
            tk["n"] += 1
            tk["sum"] += loss

            if (self.token_records_written < self.max_token_records
                    and self._in_trace_window(metrics["global_step"])):
                rec = {
                    "global_step": metrics["global_step"],
                    "checkpoint_id": checkpoint_id,
                    "sample_index": t["sample_index"],
                    "rank": t["rank"],
                    "position_in_packed_sequence": pos,
                    "token_id": tid,
                    "decoded": tokenizer.decode_token(tid),
                    "is_special": tid < N_SPECIAL,
                    "is_eos_boundary": is_eos,
                    "loss_mask": 1,
                    "cross_entropy": round(loss, 6),
                    "perplexity": round(ppl, 6),
                    "doc_id": doc_id,
                    "shard_id": shard_id,
                    "capability_lane": sample["lane"],
                    "language": span.get("language") if span else None,
                    "curriculum_stage": batch["stage"],
                    "model_age_tokens": model_age_tokens,
                    "opus_score": opus_scores.get(t["sample_index"]),
                    "repeated_pass": sample.get("pool_epoch", 0),
                }
                fh.write(canonical_json(rec) + "\n")
                self.token_records_written += 1
            else:
                self.token_records_sampled_out += 1

        step = {
            "global_step": metrics["global_step"],
            "mean_loss": round(metrics["mean_loss"], 6),
            "perplexity": round(metrics["perplexity"], 6),
            "grad_norm": round(metrics["grad_norm"], 6),
            "clipped": metrics["clipped"],
            "lr": round(metrics["lr"], 8),
            "loss_tokens": metrics["loss_tokens"],
            "stage": batch["stage"],
            "checkpoint_id": checkpoint_id,
            "model_age_tokens": model_age_tokens,
            "lane_counts": batch["lane_counts"],
            "pad_tokens": batch["pad_tokens"],
            "real_tokens": batch["real_tokens"],
            "probe_delta": probe_delta,
        }
        self.step_history.append(step)
        return step

    @staticmethod
    def _acc(store: dict, key: str, loss: float, lane: str) -> None:
        e = store.setdefault(key, {"n": 0, "sum": 0.0, "max": 0.0,
                                   "lane": lane, "high_ppl": 0})
        e["n"] += 1
        e["sum"] += loss
        e["max"] = max(e["max"], loss)
        if loss > 6.0:
            e["high_ppl"] += 1

    def flush(self) -> None:
        if self._tok_fh is not None:
            self._tok_fh.flush()
            os.fsync(self._tok_fh.fileno())

    def close(self) -> None:
        self.flush()
        if self._tok_fh is not None:
            self._tok_fh.close()
            self._tok_fh = None

    # -- reporting ---------------------------------------------------------

    def shard_report(self, *, gradient_alignment: dict[str, float] | None = None,
                     probe_deltas: dict[str, float] | None = None,
                     epochs: dict[str, int] | None = None) -> list[dict]:
        gradient_alignment = gradient_alignment or {}
        probe_deltas = probe_deltas or {}
        epochs = epochs or {}
        rows = []
        for shard_id, e in sorted(self.by_shard.items()):
            mean = e["sum"] / e["n"] if e["n"] else 0.0
            rows.append({
                "shard_id": shard_id,
                "capability_lane": e["lane"],
                "tokens_scored": e["n"],
                "mean_token_loss": round(mean, 6),
                "mean_perplexity": round(math.exp(min(mean, 700.0)), 6),
                "max_token_loss": round(e["max"], 6),
                "high_perplexity_tokens": e["high_ppl"],
                "high_perplexity_fraction": round(e["high_ppl"] / e["n"], 6) if e["n"] else 0.0,
                "gradient_alignment": gradient_alignment.get(shard_id),
                "loss_delta_before_after_exposure": probe_deltas.get(shard_id),
                "repeated_pass_number": epochs.get(shard_id, 0),
                "usefulness": classify(mean),
                "usefulness_rule": (
                    f"<= {USEFULNESS_THRESHOLDS['broken_below']} broken; "
                    f"<= {USEFULNESS_THRESHOLDS['already_learned_below']} already_learned; "
                    f"< {USEFULNESS_THRESHOLDS['neutral_below']} neutral; else useful"),
            })
        return rows

    def summary(self) -> dict:
        lanes = {}
        for lane, e in sorted(self.by_lane.items()):
            mean = e["sum"] / e["n"] if e["n"] else 0.0
            lanes[lane] = {
                "tokens": e["n"],
                "mean_loss": round(mean, 6),
                "mean_perplexity": round(math.exp(min(mean, 700.0)), 6),
                "usefulness": classify(mean),
            }
        hardest = sorted(
            ({"token_id": k, "n": v["n"], "mean_loss": round(v["sum"] / v["n"], 6)}
             for k, v in self.by_token_id.items() if v["n"] >= 3),
            key=lambda r: -r["mean_loss"])[:20]
        eos_mean = (sum(self.eos_losses) / len(self.eos_losses)) if self.eos_losses else None
        first = self.step_history[0]["mean_loss"] if self.step_history else None
        last = self.step_history[-1]["mean_loss"] if self.step_history else None
        return {
            "steps_recorded": len(self.step_history),
            "token_records_written": self.token_records_written,
            "token_records_sampled_out": self.token_records_sampled_out,
            "storage_tiering": {
                "full_trace_step_windows": self.trace_windows,
                "max_token_records": self.max_token_records,
                "policy": ("full token trace for selected step windows; per-shard, "
                           "per-lane and per-token-id aggregates accumulate over "
                           "EVERY step regardless, so the shard report is complete "
                           "for the whole run"),
            },
            "by_lane": lanes,
            "shards_touched": len(self.by_shard),
            "hardest_tokens": hardest,
            "eos": {
                "observations": len(self.eos_losses),
                "mean_loss": round(eos_mean, 6) if eos_mean is not None else None,
                "mean_perplexity": round(math.exp(min(eos_mean, 700.0)), 6)
                                   if eos_mean is not None else None,
                "why": ("perplexity at EOS specifically is how you tell whether the "
                        "boundary token is being learned; if the model is surprised "
                        "by EOS it has not learned where documents end"),
            },
            "loss_first_step": first,
            "loss_last_step": last,
            "loss_improvement": round(first - last, 6) if (first and last) else None,
            "usefulness_counts": _counts(
                classify(v["sum"] / v["n"]) for v in self.by_shard.values() if v["n"]),
        }


def _span_at(sample: dict, pos: int) -> dict | None:
    for sp in sample["spans"]:
        if sp["seq_start"] <= pos < sp["seq_end"]:
            return sp
    return None


def _counts(it) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))
