# -*- coding: utf-8 -*-
"""OPUS: gradient-aligned data selection, with a full audit trail.

The lecture describes the mechanism concretely, and at this scale it can be
implemented as described rather than approximated by a quality heuristic:

1. Run a **golden proxy** -- held-out data representing how we want the model to
   behave -- through the *current* model and take the gradient. That direction
   is "what this model still needs to learn".
2. For each candidate batch, take a **short prefix** of each sample (the
   lecture's "we only send the initial 512 tokens; that is good enough for this
   test") and compute its gradient.
3. Score by cosine similarity between the two. A candidate that would move the
   weights the proxy says need moving scores high.
4. Accept the top fraction. Record everything -- including the rejections.

Two things this buys beyond selection itself:

**``gradient_alignment`` for free.** The course page lists it as a learning-ledger
field "where available". Here it *is* available, because it is the same number
the selector already computed.

**Pool-exhaustion detection.** Because the acceptance *ratio* is fixed, the
selector always returns its quota and can never report that the pool has run
dry. Only the absolute scores can. A mean score collapsing over time means OPUS
is choosing the best of a bad set -- the diagnostic the lecture calls out, and
the reason the score distribution is tracked per round rather than just the
accept count.

Protected floors override rejection. A protected lane's batch is kept even when
it scores badly, and the override is recorded as its own status so the trail
never claims OPUS liked something it did not.
"""
from __future__ import annotations

from .trainer import cosine

REJECTION_REASONS = [
    "below_threshold",
    "stage_mismatch",
    "quota_pressure",
    "duplication",
    "already_learned",
]

STATUSES = ["accepted", "rejected", "deferred", "protected_override"]


def _prefix_sample(sample: dict, n_tokens: int) -> dict:
    """A cheap prefix view of a sample for scoring.

    Scoring the full sample would cost as much as training on it, which would
    defeat the point: the probe has to be small relative to the training it
    saves. Masks are sliced alongside the tokens so the prefix is a valid sample
    in its own right, not a truncated one that scores padding.
    """
    n = min(n_tokens, len(sample["tokens"]))
    return {
        "tokens": sample["tokens"][:n],
        "loss_mask": sample["loss_mask"][:n],
        "segment_ids": sample["segment_ids"][:n],
        "position_ids": sample["position_ids"][:n],
        "lane": sample["lane"],
        "doc_ids": sample["doc_ids"],
        "sample_index": sample["sample_index"],
    }


def score_candidate(model, samples: list[dict], proxy: list[float], *,
                    prefix_tokens: int, dims: int = 256) -> dict:
    """Cosine between a candidate's prefix gradient and the proxy direction."""
    model.zero_grad()
    n_scored = 0
    total_loss = 0.0
    for r in model.loss_batch([_prefix_sample(s, prefix_tokens) for s in samples],
                              backward=True, collect_tokens=False):
        total_loss += r["sum_loss"]
        n_scored += r["n_tokens"]
    g = model.grad_vector(dims)
    model.zero_grad()          # scoring must never contribute to an update
    return {
        "alignment": cosine(g, proxy),
        "prefix_tokens_scored": n_scored,
        "prefix_mean_loss": (total_loss / n_scored) if n_scored else 0.0,
    }


class OpusSelector:
    """Scores candidate batches and keeps four ledgers: accepted, rejected,
    deferred, protected."""

    def __init__(self, *, accept_ratio: float, prefix_tokens: int,
                 protected_lanes: set[str], proxy_version: str = "probe-v1",
                 dims: int = 256) -> None:
        self.accept_ratio = accept_ratio
        self.prefix_tokens = prefix_tokens
        self.protected_lanes = set(protected_lanes)
        self.proxy_version = proxy_version
        self.dims = dims
        self.decisions: list[dict] = []
        self.rounds: list[dict] = []
        self._counter = 0

    def select(self, model, candidates: list[dict], proxy: list[float], *,
               global_step: int, stage: str, checkpoint_id: str,
               seen_doc_ids: set[str] | None = None,
               model_loss: float | None = None) -> dict:
        """Score every candidate, then accept the top fraction.

        ``candidates`` are dicts of ``{"candidate_id", "lane", "samples",
        "shard_ids"}``.
        """
        seen_doc_ids = seen_doc_ids or set()
        scored: list[dict] = []

        for c in candidates:
            self._counter += 1
            s = score_candidate(model, c["samples"], proxy,
                                prefix_tokens=self.prefix_tokens, dims=self.dims)
            dup = bool(seen_doc_ids and
                       set(d for smp in c["samples"] for d in smp["doc_ids"]) <= seen_doc_ids)
            scored.append({
                "candidate_id": c.get("candidate_id", f"cand-{self._counter:06d}"),
                "lane": c["lane"],
                "shard_ids": sorted(c.get("shard_ids", [])),
                "curriculum_stage": stage,
                "global_step": global_step,
                "scoring_checkpoint": checkpoint_id,
                "proxy_version": self.proxy_version,
                "opus_score": round(s["alignment"], 8),
                "gradient_alignment": round(s["alignment"], 8),
                "prefix_mean_loss": round(s["prefix_mean_loss"], 6),
                "prefix_tokens_scored": s["prefix_tokens_scored"],
                "effective_token_estimate": sum(
                    sum(smp["loss_mask"]) for smp in c["samples"]),
                "is_protected": c["lane"] in self.protected_lanes,
                "_duplicate": dup,
            })

        # Rank by alignment. Ties break on candidate id so the order is stable.
        ranked = sorted(scored, key=lambda r: (-r["opus_score"], r["candidate_id"]))
        n_accept = max(1, int(round(len(ranked) * self.accept_ratio))) if ranked else 0
        threshold = ranked[n_accept - 1]["opus_score"] if n_accept else 0.0

        out: list[dict] = []
        for rank, r in enumerate(ranked):
            protected = r.pop("is_protected")
            dup = r.pop("_duplicate")
            if rank < n_accept:
                r.update(status="accepted", rejection_reason=None,
                         protected_floor_override=False)
            elif protected:
                # The floor is the point. A protected lane is kept even when the
                # selector dislikes it, and the trail says so explicitly rather
                # than pretending OPUS chose it.
                r.update(status="protected_override", rejection_reason=None,
                         protected_floor_override=True)
            else:
                if dup:
                    reason = "duplication"
                elif (model_loss is not None
                      and r["prefix_mean_loss"] < 0.5 * model_loss):
                    # Already easier than the model's running average: the
                    # concept is learned, so the gradient it buys is small.
                    reason = "already_learned"
                elif r["opus_score"] < 0:
                    reason = "stage_mismatch"
                else:
                    reason = "below_threshold"
                status = "deferred" if reason in ("already_learned", "stage_mismatch") else "rejected"
                r.update(status=status, rejection_reason=reason,
                         protected_floor_override=False)
            r["rank"] = rank
            r["accept_threshold"] = round(threshold, 8)
            out.append(r)

        self.decisions.extend(out)
        scores = [r["opus_score"] for r in out]
        rnd = {
            "global_step": global_step,
            "stage": stage,
            "candidates": len(out),
            "accepted": sum(1 for r in out if r["status"] == "accepted"),
            "rejected": sum(1 for r in out if r["status"] == "rejected"),
            "deferred": sum(1 for r in out if r["status"] == "deferred"),
            "protected_override": sum(1 for r in out if r["status"] == "protected_override"),
            "accept_threshold": round(threshold, 8),
            "score_mean": round(sum(scores) / len(scores), 8) if scores else 0.0,
            "score_max": round(max(scores), 8) if scores else 0.0,
            "score_min": round(min(scores), 8) if scores else 0.0,
            "accepted_score_mean": round(
                sum(r["opus_score"] for r in out if r["status"] == "accepted")
                / max(1, sum(1 for r in out if r["status"] == "accepted")), 8),
        }
        self.rounds.append(rnd)
        return {"decisions": out, "round": rnd,
                "accepted": [r for r in out if r["status"] in ("accepted", "protected_override")]}

    # -- reporting ---------------------------------------------------------

    def pool_health(self) -> dict:
        """Is the candidate pool exhausting?

        Compares the mean accepted score in the first third of rounds against
        the last third. A large drop at a fixed acceptance ratio means the
        selector is scraping rather than choosing -- the pool is spent, even
        though the accept *count* is unchanged.
        """
        if len(self.rounds) < 3:
            return {"available": False, "rounds": len(self.rounds)}
        k = max(1, len(self.rounds) // 3)
        early = [r["accepted_score_mean"] for r in self.rounds[:k]]
        late = [r["accepted_score_mean"] for r in self.rounds[-k:]]
        e, l = sum(early) / len(early), sum(late) / len(late)
        drop = (e - l) / abs(e) if abs(e) > 1e-9 else 0.0
        return {
            "available": True,
            "early_accepted_score_mean": round(e, 8),
            "late_accepted_score_mean": round(l, 8),
            "relative_drop": round(drop, 6),
            "pool_exhausting": drop > 0.5,
            "interpretation": (
                "acceptance ratio is fixed, so the accepted COUNT cannot reveal "
                "pool exhaustion; only the score distribution can. A large drop "
                "means OPUS is selecting the best of a bad set."
            ),
        }

    def summary(self) -> dict:
        by_status: dict[str, int] = {}
        by_reason: dict[str, int] = {}
        by_lane: dict[str, dict[str, int]] = {}
        for d in self.decisions:
            by_status[d["status"]] = by_status.get(d["status"], 0) + 1
            if d.get("rejection_reason"):
                by_reason[d["rejection_reason"]] = by_reason.get(d["rejection_reason"], 0) + 1
            lane = by_lane.setdefault(d["lane"], {})
            lane[d["status"]] = lane.get(d["status"], 0) + 1
        return {
            "total_decisions": len(self.decisions),
            "by_status": dict(sorted(by_status.items())),
            "by_rejection_reason": dict(sorted(by_reason.items())),
            "by_lane": {k: dict(sorted(v.items())) for k, v in sorted(by_lane.items())},
            "rounds": len(self.rounds),
            "accept_ratio": self.accept_ratio,
            "prefix_tokens": self.prefix_tokens,
            "proxy_version": self.proxy_version,
            "protected_lanes": sorted(self.protected_lanes),
            "pool_health": self.pool_health(),
            "method": ("cosine similarity between a candidate's prefix gradient "
                       "and the validation-probe gradient direction"),
        }
