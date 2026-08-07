# -*- coding: utf-8 -*-
"""Loss masks, attention masks and position ids.

The batch has to carry more than token ids -- it has to carry the *training
meaning* of those ids. Three separate things, often confused:

**Loss mask** -- may this position contribute to the gradient?
  Pretraining: every real token. SFT: the response only. Agentic: the model's
  own actions and tool calls only. A ``user`` turn and a ``tool_obs`` turn are
  context the model reads and must never be trained to *produce* -- train on
  tool observations and the model learns to invent environment responses
  instead of calling the tool.

**Attention mask** -- which earlier positions may this position see?
  ``document_causal`` builds block-diagonal segments so attention cannot cross
  a document boundary inside a packed sequence. The lecture describes the
  weaker, learned version (EOS is a token, back-propagation teaches the model
  to stop attending across it). Both work; this enforces it structurally as
  well, so cross-document attention is impossible rather than merely
  discouraged, and the packed neighbours of a sample cannot change its loss.

**Position ids** -- where does this token sit?
  ``reset_per_document`` restarts the count at each document, so a document
  packed second does not look like it began at position 400.
  ``continuous`` numbers straight through. Both are implemented and the policy
  travels with the batch, because a checkpoint resumed under a different
  position policy is not the same experiment.
"""
from __future__ import annotations

from .config import PAD_ID

# Roles the model reads but is never trained to produce.
CONTEXT_ONLY_ROLES = {"user", "tool_obs"}
# Roles that carry loss in SFT / agentic data.
TARGET_ROLES = {"assistant", "tool_call", "text"}

ATTENTION_POLICIES = ["document_causal", "causal"]
POSITION_POLICIES = ["reset_per_document", "continuous"]


def build_masks(sequence: dict, tokens: list[int], *,
                attention_policy: str = "document_causal",
                position_policy: str = "reset_per_document",
                lane: str = "") -> dict:
    """Produce ``loss_mask``, ``segment_ids`` and ``position_ids`` for one sequence.

    ``segment_ids`` is the compact form of the attention mask: two positions may
    attend to each other only if they share a segment id (and the earlier one
    precedes the later one). Storing N ids instead of an N x N boolean matrix
    keeps the ledger small while carrying exactly the same information.
    """
    seq_len = sequence["seq_len"]
    loss_mask = [0] * seq_len
    segment_ids = [0] * seq_len       # 0 is reserved for padding
    position_ids = [0] * seq_len

    for seg_idx, seg in enumerate(sequence["segments"], start=1):
        lo, hi = seg["seq_start"], seg["seq_end"]

        # Attention: one segment per document under document_causal, one shared
        # segment for the whole sequence under plain causal.
        sid = seg_idx if attention_policy == "document_causal" else 1
        for i in range(lo, hi):
            segment_ids[i] = sid

        # Positions.
        if position_policy == "reset_per_document":
            for k, i in enumerate(range(lo, hi)):
                position_ids[i] = k
        else:
            for i in range(lo, hi):
                position_ids[i] = i

        # Loss. Default is "everything real"; roles narrow it where they exist.
        roles = seg.get("roles") or []
        if not roles:
            for i in range(lo, hi):
                loss_mask[i] = 1
        else:
            for r in roles:
                bearing = 0 if r["role"] in CONTEXT_ONLY_ROLES else 1
                for i in range(max(lo, r["start"]), min(hi, r["end"])):
                    loss_mask[i] = bearing
            # Any part of the segment not covered by a role is plain text.
            covered = set()
            for r in roles:
                covered.update(range(max(lo, r["start"]), min(hi, r["end"])))
            for i in range(lo, hi):
                if i not in covered:
                    loss_mask[i] = 1

    # Padding never bears loss and never joins a segment.
    for i in range(seq_len):
        if i < len(tokens) and tokens[i] == PAD_ID:
            loss_mask[i] = 0
            segment_ids[i] = 0
        if i >= len(tokens):
            loss_mask[i] = 0
            segment_ids[i] = 0

    # A position may only bear loss if the token it predicts is one it could
    # legitimately predict. Two cases, and the second was a real defect found
    # while building the transformer backend:
    #
    #   * the final position of the sequence has no next token at all;
    #   * a position whose target lies in a *different segment* is being asked
    #     to predict the first token of the next document -- while the attention
    #     mask forbids it from seeing that document at all. That is training on
    #     noise. The packer inserts no separator between documents, so this
    #     happens at every internal boundary. Measured on the demo profile: 29
    #     of 45,170 scored positions (0.064%). Small, but it is exactly the
    #     off-by-one this section exists to prevent, and the n-gram backend hid
    #     it because a fully masked context still produces a loss.
    #
    # Derived from segment_ids rather than special-cased per policy: under plain
    # `causal` every real position shares one segment, so this reduces to
    # masking the position before padding -- which is also correct, since a PAD
    # target is not a prediction either.
    for i in range(seq_len - 1):
        if segment_ids[i] and segment_ids[i + 1] != segment_ids[i]:
            loss_mask[i] = 0
    if seq_len:
        loss_mask[seq_len - 1] = 0

    return {
        "loss_mask": loss_mask,
        "segment_ids": segment_ids,
        "position_ids": position_ids,
        "attention_policy": attention_policy,
        "position_policy": position_policy,
        "lane": lane,
        "loss_bearing_tokens": sum(loss_mask),
        "context_only_tokens": sum(
            1 for seg in sequence["segments"] for r in (seg.get("roles") or [])
            if r["role"] in CONTEXT_ONLY_ROLES for _ in range(r["start"], r["end"])
        ),
    }


def can_attend(segment_ids: list[int], i: int, j: int) -> bool:
    """May position ``i`` attend to position ``j``? Used by the model and tests."""
    if j > i:
        return False                       # causal
    si, sj = segment_ids[i], segment_ids[j]
    return si != 0 and si == sj            # never across a document, never padding


def materialise_attention_matrix(segment_ids: list[int]) -> list[list[bool]]:
    """The full N x N mask. Only for tests and the dashboard -- the runtime uses
    ``segment_ids`` directly, because materialising N x N is exactly the cost
    this representation exists to avoid."""
    n = len(segment_ids)
    return [[can_attend(segment_ids, i, j) for j in range(n)] for i in range(n)]


def summarise(mask_sets: list[dict]) -> dict:
    total = sum(len(m["loss_mask"]) for m in mask_sets)
    bearing = sum(m["loss_bearing_tokens"] for m in mask_sets)
    return {
        "sequences": len(mask_sets),
        "total_positions": total,
        "loss_bearing_tokens": bearing,
        "loss_bearing_fraction": round(bearing / total, 6) if total else 0.0,
        "attention_policies": sorted({m["attention_policy"] for m in mask_sets}),
        "position_policies": sorted({m["position_policy"] for m in mask_sets}),
    }
