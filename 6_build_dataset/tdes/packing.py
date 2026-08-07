# -*- coding: utf-8 -*-
"""Packing policies.

Packing is where the context window becomes a budget. Every position left
unused is a token the model did not learn from, and every position filled with
padding is compute that was paid for and produced nothing.

Six policies, one per data type, because the right answer genuinely differs:

=====================  ===========================================================
``pad_only``           One document per sequence, pad the rest. Kept as the
                       baseline the utilization report measures everything else
                       against -- not because anyone would use it.
``concat_chop``        Join with EOS and cut fixed windows. Efficient for plain
                       prose, where a sequence boundary is only an engineering
                       boundary.
``greedy``             First sequence with room. Fast, order-dependent, leaves
                       holes.
``best_fit``           Best-fit-decreasing bin packing. Better utilization, and
                       it keeps an atomic unit whole when it fits.
``structure_preserving`` One trace per sequence, never merged. For SFT and
                       agentic data, where two unrelated trajectories sharing an
                       attention window teaches a transition that does not exist.
``long_context``       Only documents at least as long as the sequence. Reserved
                       for the late rung so a 128-token window is not spent on
                       64-token documents.
=====================  ===========================================================

Every policy returns the same shape, so the utilization report can compare them
directly on the same inputs.
"""
from __future__ import annotations

from .config import PAD_ID

POLICIES = ["pad_only", "concat_chop", "greedy", "best_fit",
            "structure_preserving", "long_context"]


def _segment(item: dict, tok_start: int, tok_end: int, seq_offset: int) -> dict:
    """One document's contribution to a packed sequence.

    ``tok_start``/``tok_end`` index into the shard; ``seq_offset`` is where the
    slice lands inside the sequence. Keeping both means an audit can go from a
    position in a batch back to a byte range in a shard.
    """
    roles = []
    for r in item.get("token_roles", []):
        lo, hi = max(r["start"], tok_start), min(r["end"], tok_end)
        if lo < hi:
            roles.append({"role": r["role"],
                          "start": seq_offset + (lo - tok_start),
                          "end": seq_offset + (hi - tok_start)})
    return {
        "shard_id": item["shard_id"],
        "doc_id": item["doc_id"],
        "shard_start": tok_start,
        "shard_end": tok_end,
        "seq_start": seq_offset,
        "seq_end": seq_offset + (tok_end - tok_start),
        "roles": roles,
        "truncated": (tok_start > item["start"]) or (tok_end < item["end"]),
    }


def _seq(segments: list[dict], seq_len: int, policy: str) -> dict:
    used = sum(s["seq_end"] - s["seq_start"] for s in segments)
    return {
        "policy": policy,
        "seq_len": seq_len,
        "segments": segments,
        "real_tokens": used,
        "pad_tokens": seq_len - used,
        "utilization": round(used / seq_len, 6) if seq_len else 0.0,
        "n_documents": len({s["doc_id"] for s in segments}),
        "boundary_crossings": max(0, len(segments) - 1),
    }


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def _one_item_per_sequence(items: list[dict], seq_len: int, policy: str) -> list[dict]:
    """One document per sequence; a long document spans several sequences.

    The distinction that matters: a sequence never holds *two* documents, but a
    document longer than the window is chunked across consecutive sequences
    rather than truncated.

    The first version truncated, and it cost 81% of the agentic lane -- traces
    run 250-600 tokens against a 64-token demo window, so "one per sequence"
    silently threw away everything past the first window. That is not what
    either policy means. ``pad_only`` is a baseline for measuring *padding*
    waste, not data loss, and structure preservation is about not letting
    unrelated examples share an attention window -- a single trace occupying
    several windows breaks no structure at all.
    """
    out = []
    for it in items:
        pos = it["start"]
        while pos < it["end"]:
            take = min(seq_len, it["end"] - pos)
            out.append(_seq([_segment(it, pos, pos + take, 0)], seq_len, policy))
            pos += take
    return out


def pack_pad_only(items: list[dict], seq_len: int) -> list[dict]:
    return _one_item_per_sequence(items, seq_len, "pad_only")


def pack_concat_chop(items: list[dict], seq_len: int) -> list[dict]:
    """Concatenate then cut. Documents already carry EOS, so the boundary
    survives the cut even when a window lands mid-document."""
    out: list[dict] = []
    cur: list[dict] = []
    offset = 0
    for it in items:
        pos = it["start"]
        while pos < it["end"]:
            take = min(seq_len - offset, it["end"] - pos)
            cur.append(_segment(it, pos, pos + take, offset))
            offset += take
            pos += take
            if offset >= seq_len:
                out.append(_seq(cur, seq_len, "concat_chop"))
                cur, offset = [], 0
    if cur:
        out.append(_seq(cur, seq_len, "concat_chop"))
    return out


def pack_greedy(items: list[dict], seq_len: int) -> list[dict]:
    """First sequence with room, in arrival order.

    Fast and simple, which is the point: at 10 trillion tokens the cost of
    building the dataset is itself a budget line. The price is holes, because
    document order is preserved rather than optimised.
    """
    bins: list[list[dict]] = []
    fill: list[int] = []
    for it in items:
        n = it["end"] - it["start"]
        if n > seq_len:                       # too long to place whole: chop it
            pos = it["start"]
            while pos < it["end"]:
                take = min(seq_len, it["end"] - pos)
                bins.append([_segment(it, pos, pos + take, 0)])
                fill.append(take)
                pos += take
            continue
        for b in range(len(bins)):
            if fill[b] + n <= seq_len:
                bins[b].append(_segment(it, it["start"], it["end"], fill[b]))
                fill[b] += n
                break
        else:
            bins.append([_segment(it, it["start"], it["end"], 0)])
            fill.append(n)
    return [_seq(b, seq_len, "greedy") for b in bins]


def pack_best_fit(items: list[dict], seq_len: int) -> list[dict]:
    """Best-fit-decreasing.

    Bin packing is NP-hard, so nobody solves it exactly at corpus scale. BFD is
    the standard approximation -- sort by decreasing size, then place each item
    into the *tightest* bin that still holds it -- and it is provably within a
    small constant of optimal while running in O(n log n).

    Atomic units (a whole function or class, in the code lane) are never split
    when they fit: cutting a function in half teaches the model that functions
    end arbitrarily.
    """
    order = sorted(items, key=lambda i: (-(i["end"] - i["start"]), i["doc_id"]))
    bins: list[list[dict]] = []
    fill: list[int] = []
    for it in order:
        n = it["end"] - it["start"]
        if n > seq_len:
            pos = it["start"]
            while pos < it["end"]:
                take = min(seq_len, it["end"] - pos)
                bins.append([_segment(it, pos, pos + take, 0)])
                fill.append(take)
                pos += take
            continue
        best, best_left = -1, seq_len + 1
        for b in range(len(bins)):
            left = seq_len - fill[b] - n
            if 0 <= left < best_left:
                best, best_left = b, left
        if best >= 0:
            bins[best].append(_segment(it, it["start"], it["end"], fill[best]))
            fill[best] += n
        else:
            bins.append([_segment(it, it["start"], it["end"], 0)])
            fill.append(n)
    return [_seq(b, seq_len, "best_fit") for b in bins]


def pack_structure_preserving(items: list[dict], seq_len: int) -> list[dict]:
    """Never merges two items; a long item spans consecutive sequences.

    The guarantee is *no two unrelated examples share an attention window*. Two
    traces in one window would teach the model that a tool observation from
    task A is reasonable context for a tool call in task B -- a transition that
    never happens in the real environment.

    What the guarantee does **not** require is that a trace fit in one window.
    Chunking a single trace across consecutive sequences preserves its structure
    perfectly; only merging breaks it.

    The padding this still leaves -- on the final chunk of each trace -- is the
    honest cost of the policy, and the utilization report shows it.
    """
    return _one_item_per_sequence(items, seq_len, "structure_preserving")


def pack_long_context(items: list[dict], seq_len: int, *,
                      min_fraction: float = 1.0) -> list[dict]:
    """Only documents long enough to fill the window.

    Short documents are *skipped*, not padded. A long-context rung is the most
    expensive compute in the run, and spending it on a 60-token document is the
    waste the reservation exists to prevent. The skipped items are reported so
    the caller can route them elsewhere rather than losing them.
    """
    need = int(seq_len * min_fraction)
    out, skipped = [], []
    for it in items:
        n = it["end"] - it["start"]
        if n < need:
            skipped.append(it["doc_id"])
            continue
        pos = it["start"]
        while it["end"] - pos >= need:
            out.append(_seq([_segment(it, pos, pos + seq_len, 0)], seq_len, "long_context"))
            pos += seq_len
    for s in out:
        s["skipped_short_documents"] = len(skipped)
    return out


_DISPATCH = {
    "pad_only": pack_pad_only,
    "concat_chop": pack_concat_chop,
    "greedy": pack_greedy,
    "best_fit": pack_best_fit,
    "structure_preserving": pack_structure_preserving,
    "long_context": pack_long_context,
}


def pack(items: list[dict], seq_len: int, policy: str) -> list[dict]:
    if policy not in _DISPATCH:
        raise ValueError(f"unknown packing policy {policy!r}; choose from {POLICIES}")
    return _DISPATCH[policy](items, seq_len)


# ---------------------------------------------------------------------------
# Utilization
# ---------------------------------------------------------------------------

def utilization(sequences: list[dict], input_tokens: int | None = None) -> dict:
    """Fill rate **and** coverage. Reporting one without the other is a lie.

    The first version of this reported only fill rate, and on the code lane at
    a 64-token sequence it declared ``pad_only`` 100% efficient. It was: every
    position it emitted held a real token. It reached that by truncating each
    document to 64 tokens and discarding the other 92% of the lane -- 48,849
    input tokens became 3,776 -- so a policy that threw away almost all the data
    scored better than one that kept it.

    Fill rate answers "was the compute I paid for used?".
    Coverage answers "did the data I have actually get trained on?".
    A policy is only good if both are high, so both are always reported and
    ``effective_yield`` is their product.
    """
    if not sequences:
        return {"sequences": 0, "real_tokens": 0, "pad_tokens": 0,
                "total_positions": 0, "utilization": 0.0, "truncations": 0,
                "boundary_crossings": 0, "input_tokens": input_tokens or 0,
                "tokens_dropped": input_tokens or 0, "coverage": 0.0,
                "effective_yield": 0.0}
    real = sum(s["real_tokens"] for s in sequences)
    pad = sum(s["pad_tokens"] for s in sequences)
    total = real + pad
    fill = round(real / total, 6) if total else 0.0
    row = {
        "sequences": len(sequences),
        "real_tokens": real,
        "pad_tokens": pad,
        "total_positions": total,
        "utilization": fill,
        "truncations": sum(1 for s in sequences for g in s["segments"] if g["truncated"]),
        "boundary_crossings": sum(s["boundary_crossings"] for s in sequences),
    }
    if input_tokens:
        # Distinct positions of the source actually represented, so a policy
        # that emits the same tokens twice cannot inflate its coverage.
        covered = 0
        seen: set[tuple[str, int, int]] = set()
        for s in sequences:
            for g in s["segments"]:
                key = (g["shard_id"], g["shard_start"], g["shard_end"])
                if key not in seen:
                    seen.add(key)
                    covered += g["shard_end"] - g["shard_start"]
        row.update({
            "input_tokens": input_tokens,
            "tokens_covered": covered,
            "tokens_dropped": max(0, input_tokens - covered),
            "coverage": round(min(1.0, covered / input_tokens), 6),
            "effective_yield": round(fill * min(1.0, covered / input_tokens), 6),
        })
    return row


def compare_policies(items: list[dict], seq_len: int,
                     policies: list[str] | None = None) -> dict:
    """Run every policy over the same items so the numbers are comparable.

    This is the evidence behind any claim about packing efficiency: the same
    inputs, the same sequence length, one row per policy, fill and coverage
    side by side.
    """
    input_tokens = sum(i["end"] - i["start"] for i in items)
    rows = {}
    for p in (policies or POLICIES):
        try:
            rows[p] = utilization(pack(items, seq_len, p), input_tokens)
        except Exception as e:                       # a policy may not apply
            rows[p] = {"error": str(e), "sequences": 0}
    base = rows.get("pad_only", {}).get("total_positions", 0)
    for r in rows.values():
        if base and r.get("total_positions"):
            r["positions_vs_pad_only"] = round(r["total_positions"] / base, 6)
    ranked = sorted((p for p, r in rows.items() if r.get("effective_yield") is not None),
                    key=lambda p: (-rows[p].get("effective_yield", 0), p))
    return {
        "seq_len": seq_len,
        "items": len(items),
        "input_tokens": input_tokens,
        "by_policy": rows,
        "ranked_by_effective_yield": ranked,
        "metric_note": ("utilization is fill rate (real / positions emitted); "
                        "coverage is the fraction of input tokens represented; "
                        "effective_yield is their product. A policy that "
                        "truncates can score 100% fill while dropping most of "
                        "the lane, so fill alone is never reported."),
    }
