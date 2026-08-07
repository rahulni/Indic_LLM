# -*- coding: utf-8 -*-
"""Turn packed sequences into batches, and give every batch an identity.

Two identities, and both are needed:

``batch_id``
    ``sha256(run ‖ branch ‖ step ‖ [(shard, span, sample)...])`` -- *which*
    data this batch claims to be.

``batch_content_hash``
    ``sha256(token_ids ‖ loss_mask ‖ attention_policy ‖ position_ids)`` -- what
    the batch actually *contains*.

Replay compares both. Matching only ``batch_id`` would prove the plan was
followed while saying nothing about whether the tokens behind it were the same;
matching only the content hash would prove the tokens matched while losing the
provenance. A pipeline that changed a shard's contents but kept its span layout
would pass the first check and fail the second, which is the point.

Rank partitioning is deterministic and disjoint: sample ``i`` of a step belongs
to rank ``i % world_size``. Every sample is served exactly once per step, and
the test suite asserts it rather than assuming it.
"""
from __future__ import annotations

from .config import PAD_ID, RunConfig
from .determinism import stable_shuffle
from .hashing import hash_many
from .masks import build_masks
from .packing import pack


class LaneSequencePool:
    """The deterministic, cyclic supply of packed sequences for one lane.

    Sequences are packed once per (lane, sequence length) and then served in a
    fixed order. When the pool is exhausted it wraps, and the wrap count is the
    lane's epoch number -- which is how the repetition budget is actually
    measured rather than estimated.
    """

    def __init__(self, lane: str, sequences: list[dict], seed: str) -> None:
        self.lane = lane
        self.sequences = stable_shuffle(sequences, seed=f"{seed}|pool|{lane}")
        self.cursor = 0
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.sequences)

    def take(self, n: int) -> list[dict]:
        out = []
        for _ in range(n):
            if not self.sequences:
                break
            if self.cursor >= len(self.sequences):
                self.cursor = 0
                self.epoch += 1
            seq = dict(self.sequences[self.cursor])
            seq["pool_index"] = self.cursor
            seq["pool_epoch"] = self.epoch
            out.append(seq)
            self.cursor += 1
        return out

    def state(self) -> dict:
        return {"lane": self.lane, "cursor": self.cursor, "epoch": self.epoch,
                "size": len(self.sequences)}

    def load_state(self, st: dict) -> None:
        self.cursor = int(st["cursor"])
        self.epoch = int(st["epoch"])


def sequence_tier(seq: dict, items_by_doc: dict[str, dict]) -> str | None:
    """The Indic tier of a packed sequence.

    A sequence is ``verified`` only when *every* document in it is verified.
    Mixing tiers inside one sequence and calling the result verified is exactly
    the substitution session 5 forbids, so a mixed sequence counts as
    unverified.
    """
    tiers = {items_by_doc.get(g["doc_id"], {}).get("indic_tier")
             for g in seq["segments"]}
    tiers.discard(None)
    if not tiers:
        return None
    return "verified" if tiers == {"verified"} else "unverified"


def build_pools(schedule: dict, shard_items: dict[str, list[dict]],
                lane_packing: dict[str, str], seed: str,
                seq_lens: list[int]) -> dict[tuple[str, int], LaneSequencePool]:
    """Pack every lane at every sequence length the schedule uses.

    The Indic lane additionally gets **tier sub-pools**. Session 5's rule is
    that unverified Indic may never substitute for the verified portion of the
    protected floor, and a rule that only appears in a report is not a rule --
    it has to be something the server can honour. Splitting the pool is what
    lets ``RunState.build_batch`` draw the verified share from verified data
    rather than hoping the mix works out.
    """
    pools: dict[tuple[str, int], LaneSequencePool] = {}
    for lane, items in sorted(shard_items.items()):
        policy = lane_packing.get(lane, "concat_chop")
        by_doc = {i["doc_id"]: i for i in items}
        for sl in sorted(set(seq_lens)):
            seqs = pack(items, sl, policy)
            if not seqs and policy == "long_context":
                # No document is long enough for this rung. Falling back keeps
                # the lane alive rather than silently dropping its floor, and
                # the substitution is visible in the sequence's policy field.
                seqs = pack(items, sl, "concat_chop")
                for s in seqs:
                    s["policy_fallback_from"] = "long_context"
            for s in seqs:
                s["indic_tier"] = sequence_tier(s, by_doc)
            pools[(lane, sl)] = LaneSequencePool(lane, seqs, seed)

            if lane == "indic":
                for tier in ("verified", "unverified"):
                    sub = [s for s in seqs if s.get("indic_tier") == tier]
                    if sub:
                        pools[(f"indic:{tier}", sl)] = LaneSequencePool(
                            f"indic:{tier}", sub, seed)
    return pools


def materialise_tokens(sequence: dict, shard_tokens: dict[str, list[int]],
                       seq_len: int) -> list[int]:
    """Lay a packed sequence out into a flat token array, padded to ``seq_len``."""
    out = [PAD_ID] * seq_len
    for seg in sequence["segments"]:
        src = shard_tokens[seg["shard_id"]]
        chunk = src[seg["shard_start"]:seg["shard_end"]]
        lo = seg["seq_start"]
        out[lo:lo + len(chunk)] = chunk
    return out


def build_sample(sequence: dict, shard_tokens: dict[str, list[int]], *,
                 seq_len: int, attention_policy: str, position_policy: str,
                 lane: str, sample_index: int) -> dict:
    tokens = materialise_tokens(sequence, shard_tokens, seq_len)
    masks = build_masks(sequence, tokens, attention_policy=attention_policy,
                        position_policy=position_policy, lane=lane)
    # Roles travel with the span. Without them the ledger cannot rebuild the
    # loss mask for SFT or agentic samples, and replay silently produces a
    # different batch -- which is exactly what the content hash caught.
    spans = [{"shard_id": s["shard_id"], "doc_id": s["doc_id"],
              "shard_start": s["shard_start"], "shard_end": s["shard_end"],
              "seq_start": s["seq_start"], "seq_end": s["seq_end"],
              "truncated": s["truncated"],
              "roles": s.get("roles") or []} for s in sequence["segments"]]
    return {
        "sample_index": sample_index,
        "lane": lane,
        "policy": sequence["policy"],
        "seq_len": seq_len,
        "tokens": tokens,
        "loss_mask": masks["loss_mask"],
        "segment_ids": masks["segment_ids"],
        "position_ids": masks["position_ids"],
        "attention_policy": attention_policy,
        "position_policy": position_policy,
        "spans": spans,
        "doc_ids": sorted({s["doc_id"] for s in spans}),
        "shard_ids": sorted({s["shard_id"] for s in spans}),
        "real_tokens": sequence["real_tokens"],
        "pad_tokens": sequence["pad_tokens"],
        "loss_bearing_tokens": masks["loss_bearing_tokens"],
        "pool_index": sequence.get("pool_index"),
        "pool_epoch": sequence.get("pool_epoch", 0),
        "indic_tier": sequence.get("indic_tier"),
    }


def compute_batch_id(run_id: str, branch_id: str, global_step: int,
                     samples: list[dict]) -> str:
    """Identity of the *plan*: which spans, in which order."""
    parts = [run_id, branch_id, global_step]
    for s in samples:
        for sp in s["spans"]:
            parts.append([sp["shard_id"], sp["shard_start"], sp["shard_end"],
                          s["sample_index"]])
    return hash_many(parts)


def compute_content_hash(samples: list[dict]) -> str:
    """Identity of the *contents*: tokens, masks and policies actually built."""
    parts = []
    for s in samples:
        parts.append(s["tokens"])
        parts.append(s["loss_mask"])
        parts.append(s["position_ids"])
        parts.append(s["attention_policy"])
        parts.append(s["segment_ids"])
    return hash_many(parts)


def assemble_batch(cfg: RunConfig, step_plan: dict, samples: list[dict], *,
                   branch_id: str) -> dict:
    """Group samples into per-rank microbatches and seal the batch."""
    profile = cfg.profile
    world = profile.ranks
    micro = profile.microbatch
    accum = profile.grad_accum

    # Round-robin over ranks: disjoint by construction, and every sample lands
    # exactly once. Asserted in tests/test_rank_partition_disjoint.
    by_rank: dict[int, list[dict]] = {r: [] for r in range(world)}
    for i, s in enumerate(samples):
        by_rank[i % world].append(s)

    microbatches = []
    for rank in range(world):
        rows = by_rank[rank]
        for acc in range(accum):
            chunk = rows[acc * micro:(acc + 1) * micro]
            if chunk:
                microbatches.append({
                    "rank": rank,
                    "accum_index": acc,
                    "microbatch_id": f"s{step_plan['global_step']:06d}_r{rank}_a{acc}",
                    "sample_indices": [c["sample_index"] for c in chunk],
                    "samples": chunk,
                })

    batch_id = compute_batch_id(cfg.run_id, branch_id,
                                step_plan["global_step"], samples)
    content_hash = compute_content_hash(samples)
    return {
        "run_id": cfg.run_id,
        "branch_id": branch_id,
        "global_step": step_plan["global_step"],
        "stage": step_plan["stage"],
        "seq_len": step_plan["sequence_length"],
        "batch_id": batch_id,
        "batch_content_hash": content_hash,
        "loss_mask_hash": hash_many([s["loss_mask"] for s in samples]),
        "attention_policy": step_plan["attention_policy"],
        "position_policy": step_plan["position_policy"],
        "microbatches": microbatches,
        "samples": samples,
        "lane_counts": _lane_counts(samples),
        "total_positions": sum(len(s["tokens"]) for s in samples),
        "real_tokens": sum(s["real_tokens"] for s in samples),
        "pad_tokens": sum(s["pad_tokens"] for s in samples),
        "loss_bearing_tokens": sum(s["loss_bearing_tokens"] for s in samples),
        "shard_ids": sorted({sh for s in samples for sh in s["shard_ids"]}),
        "doc_ids": sorted({d for s in samples for d in s["doc_ids"]}),
    }


def _lane_counts(samples: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        out[s["lane"]] = out.get(s["lane"], 0) + 1
    return dict(sorted(out.items()))


def verify_rank_partition(batch: dict, world_size: int) -> tuple[bool, list[str]]:
    """Every sample served exactly once, to exactly one rank."""
    problems = []
    seen: dict[int, list[int]] = {}
    for mb in batch["microbatches"]:
        for si in mb["sample_indices"]:
            seen.setdefault(si, []).append(mb["rank"])
    for si, ranks in sorted(seen.items()):
        if len(ranks) > 1:
            problems.append(f"sample {si} served to ranks {sorted(ranks)}")
    expected = {s["sample_index"] for s in batch["samples"]}
    missing = expected - set(seen)
    if missing:
        problems.append(f"samples never served: {sorted(missing)}")
    return (not problems), problems
