# -*- coding: utf-8 -*-
"""Lane scarcity: what to do when a lane demands more than it has.

The course page names four options when a lane requests more tokens than exist:

    repeat existing data | generate synthetic data | reduce the lane share |
    move the share to a later stage

The policy that fires is *recorded*, not silently applied, because a run where
the Indic lane quietly shrank to fit supply looks identical in the loss curve to
a run where it was planned that way -- and only one of those is a decision.

The repetition ceiling comes from session 5's ``spec.py``: ``EPOCH_CAP = 4.0``,
from Muennighoff et al., "Scaling Data-Constrained Language Models"
(NeurIPS 2023, arXiv:2305.16264). The same paper fits a decay in which each
repetition past the first is worth less than fresh data, so the effective token
count is reported alongside the raw count rather than pretending a fourth pass
buys as much as the first.
"""
from __future__ import annotations

import math

from .config import EPOCH_CAP, EPOCH_DECAY_R_STAR

# In order of preference. Repetition is cheapest and reversible; reducing a
# lane's share is a real change to the curriculum and comes last.
POLICIES = ["repeat", "defer_to_later_stage", "reduce_share", "synthesize"]


def effective_tokens(unique_tokens: int, epochs: float,
                     r_star: float = EPOCH_DECAY_R_STAR) -> float:
    """Muennighoff et al. eq. 6: value delivered by ``epochs`` passes.

        D' = U + U * R* * (1 - exp(-R / R*)),   R = epochs - 1

    At the 4.0 cap this returns about 3.73 effective epochs, so the fourth pass
    is worth roughly a quarter of a fresh one. Physical cost is unchanged --
    you still spend compute on every pass -- but the *value* is discounted, and
    reporting both makes the discount visible instead of assumed away.
    """
    if unique_tokens <= 0 or epochs <= 0:
        return 0.0
    r = max(0.0, epochs - 1.0)
    return unique_tokens * (1.0 + r_star * (1.0 - math.exp(-r / r_star)))


def resolve(lane: str, demand_tokens: int, supply_tokens: int, *,
            stage_id: str, epoch_cap: float = EPOCH_CAP,
            is_protected: bool = False, later_stages_exist: bool = True) -> dict:
    """Decide how a lane's shortfall is covered. Pure function; records its reasoning."""
    if supply_tokens <= 0:
        return {
            "lane": lane, "stage": stage_id, "policy": "reduce_share",
            "demand_tokens": demand_tokens, "supply_tokens": 0,
            "satisfied_tokens": 0, "epochs": 0.0, "effective_tokens": 0.0,
            "shortfall_tokens": demand_tokens,
            "reason": "no shards available for this lane at this stage",
        }

    epochs_needed = demand_tokens / supply_tokens
    base = {
        "lane": lane, "stage": stage_id,
        "demand_tokens": demand_tokens, "supply_tokens": supply_tokens,
        "epochs_needed": round(epochs_needed, 4),
        "epoch_cap": epoch_cap,
        # Recorded on the decision because only a protected lane may exceed the
        # cap, and a checker should not have to re-derive which lanes those are
        # from the stage table to know whether a breach was legitimate.
        "is_protected": bool(is_protected),
    }

    if epochs_needed <= 1.0:
        return {**base, "policy": "none", "satisfied_tokens": demand_tokens,
                "epochs": round(epochs_needed, 4),
                "effective_tokens": float(demand_tokens),
                "shortfall_tokens": 0,
                "reason": "supply covers demand within a single pass"}

    if epochs_needed <= epoch_cap:
        return {**base, "policy": "repeat", "satisfied_tokens": demand_tokens,
                "epochs": round(epochs_needed, 4),
                "effective_tokens": round(effective_tokens(supply_tokens, epochs_needed), 1),
                "shortfall_tokens": 0,
                "reason": f"demand needs {epochs_needed:.2f} passes, within the "
                          f"{epoch_cap} epoch cap; repeating existing data"}

    # Past the cap. Serve what the cap allows and cover the rest by policy.
    capped = int(supply_tokens * epoch_cap)
    shortfall = demand_tokens - capped

    if is_protected:
        # A protected lane may not be quietly shrunk -- that is the entire point
        # of the floor. Over-repeat and flag it loudly instead.
        return {**base, "policy": "repeat_over_cap", "satisfied_tokens": demand_tokens,
                "epochs": round(epochs_needed, 4),
                "effective_tokens": round(effective_tokens(supply_tokens, epochs_needed), 1),
                "shortfall_tokens": 0,
                "warning": f"protected lane exceeds the {epoch_cap} epoch cap at "
                           f"{epochs_needed:.2f} passes; supply must be grown "
                           f"before a real run",
                "reason": "lane is protected, so its floor is honoured even past "
                          "the repetition cap; the cap breach is recorded"}

    policy = "defer_to_later_stage" if later_stages_exist else "reduce_share"
    return {**base, "policy": policy, "satisfied_tokens": capped,
            "epochs": epoch_cap,
            "effective_tokens": round(effective_tokens(supply_tokens, epoch_cap), 1),
            "shortfall_tokens": shortfall,
            "reason": f"demand needs {epochs_needed:.2f} passes, beyond the "
                      f"{epoch_cap} epoch cap; "
                      + ("deferring the excess to a later stage"
                         if policy == "defer_to_later_stage"
                         else "reducing this lane's share")}


def summarise(decisions: list[dict]) -> dict:
    by_policy: dict[str, int] = {}
    for d in decisions:
        by_policy[d["policy"]] = by_policy.get(d["policy"], 0) + 1
    fired = [d for d in decisions if d["policy"] != "none"]
    return {
        "decisions": len(decisions),
        "by_policy": dict(sorted(by_policy.items())),
        "lanes_needing_a_policy": sorted({d["lane"] for d in fired}),
        "total_shortfall_tokens": sum(d.get("shortfall_tokens", 0) for d in decisions),
        "cap_breaches": [d for d in decisions if d["policy"] == "repeat_over_cap"],
        "epoch_cap": EPOCH_CAP,
        "epoch_cap_source": ("session 5 spec.py EPOCH_CAP_DEFAULT; Muennighoff et al., "
                             "Scaling Data-Constrained Language Models, NeurIPS 2023 "
                             "(arXiv:2305.16264)"),
    }
