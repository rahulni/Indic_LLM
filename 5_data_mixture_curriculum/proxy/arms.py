# -*- coding: utf-8 -*-
"""
arms.py - turns the mixture spec into concrete, runnable proxy-arm configs.

Each arm is a full training configuration: per-lane sampling weights per phase,
the subsampled pool size for each lane, sequence-length ladder, LR schedule and
band-crossfade settings. Nothing here is hand-written per arm - every arm is a
declared mutation of the baseline, so an arm cannot silently drift from the plan
it is supposed to test.

    python arms.py                 # print all arm configs
    python arms.py --json out/     # write one JSON per arm
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plan"))

import spec      # noqa: E402
import audit     # noqa: E402


def baseline_config(scale="scale_1b"):
    r = audit.run()
    p = spec.PROXY[scale]
    factor = p["tokens_per_arm"] / r["budget"]

    lanes = {}
    for a in r["audited"]:
        lanes[a["id"]] = dict(
            label=a["label"],
            phase_shares=list(a["phases"]),
            whole_run_share=round(a["share"], 4),
            # [REVIEW FIX 1] pool is subsampled by the same factor as the
            # budget, so proxy epochs equal full-run epochs.
            pool_tokens=int(a["supply_usable"] * factor),
            trained_tokens=int(a["trained"] * factor),
            candidate_tokens=int(a["demand"] * factor),
            expected_epochs=round(a["epochs"], 4),
            selector_multiplier=round(a["selector_mult"], 4),
            loss_bearing_frac=spec.LOSS_BEARING[a["id"]],
        )

    return dict(
        arm_id="A0", arm_name="Baseline",
        params=int(p["params"]), tokens=int(p["tokens_per_arm"]),
        corpus_scale_factor=factor,
        tokens_per_param=p["tokens_per_arm"] / p["params"],
        phases=[dict(id=ph["id"], weight=ph["weight"],
                     tokens=int(ph["weight"] * p["tokens_per_arm"]))
                for ph in r["phases"]],
        lanes=lanes,
        floor={k: v["pct"] for k, v in spec.FLOOR.items()},
        selector=dict(spec.SELECTOR),
        lr=dict(spec.LR_SCHEDULE),
        seq_ladder=[dict(s) for s in r["seq_ladder"]],
        band_transition=dict(spec.BAND_TRANSITION),
        difficulty_bands=[dict(id=b["id"], phase_mix=list(b["phase_mix"]))
                          for b in r["difficulty_bands"]],
        length_bands=[dict(id=b["id"], budget=list(b["token_budget"]),
                           control=b["control"],
                           share=b["share_of_reasoning_lane"])
                      for b in r["length_bands"]],
    )


# Each arm is a mutation of the baseline. Keeping them as functions rather than
# hand-written configs is what stops an arm from testing something other than
# the single variable it claims to isolate.

def _arm_A1(c):
    """Agentic-starved: agentic to zero, freed share to web."""
    freed = list(c["lanes"]["agentic"]["phase_shares"])
    c["lanes"]["agentic"]["phase_shares"] = [0.0] * len(freed)
    c["lanes"]["web"]["phase_shares"] = [
        w + f for w, f in zip(c["lanes"]["web"]["phase_shares"], freed)]
    c["floor"].pop("agentic", None)
    return c


def _arm_A2(c):
    """Floor-off: selector may discard anything."""
    c["floor"] = {}
    return c


def _arm_A3(c):
    """Code-heavy: violate the repetition ceiling on purpose."""
    c["lanes"]["code"]["phase_shares"] = [16.0, 34.0, 46.0, 30.0]
    c["lanes"]["web"]["phase_shares"] = [68.0, 43.0, 20.0, 5.0]
    return c


def _arm_A4(c):
    """Static-mix: no phases, no anneal, same totals."""
    n = len(c["phases"])
    for lane in c["lanes"].values():
        lane["phase_shares"] = [lane["whole_run_share"]] * n
    c["band_transition"]["overlap_frac"] = 0.0
    c["lr"]["anneal_shape"] = "none - decay runs uniformly to zero across the whole run"
    return c


def _arm_A5(c):
    """Selector-off: train on every candidate batch."""
    c["selector"]["keep_fraction"] = 1.0
    c["selector"]["enabled"] = False
    for lane in c["lanes"].values():
        lane["selector_multiplier"] = 1.0
        lane["candidate_tokens"] = lane["trained_tokens"]
    return c


def _arm_A6(c):
    """Sharp-transitions: remove the crossfade."""
    c["band_transition"]["overlap_frac"] = 0.0
    c["band_transition"]["trip_action"] = "none - this arm exists to let it spike"
    return c


MUTATIONS = {"A1": _arm_A1, "A2": _arm_A2, "A3": _arm_A3,
             "A4": _arm_A4, "A5": _arm_A5, "A6": _arm_A6}


def recompute(c):
    """Re-derive every dependent field from the mutated phase shares.

    Without this, an arm that changes a lane's share still reports the
    baseline's trained tokens and epoch count - the config would look mutated
    while describing the wrong experiment."""
    weights = [ph["weight"] for ph in c["phases"]]
    kf = c["selector"].get("keep_fraction", 1.0)
    for lid, l in c["lanes"].items():
        share = sum(s * w for s, w in zip(l["phase_shares"], weights))
        l["whole_run_share"] = round(share, 4)
        floor = min(c["floor"].get(lid, 0.0), share)
        mult = ((floor + (share - floor) / kf) / share) if share > 0 else 1.0
        l["selector_multiplier"] = round(mult, 4)
        l["trained_tokens"] = int(share / 100.0 * c["tokens"])
        l["candidate_tokens"] = int(l["trained_tokens"] * mult)
        l["expected_epochs"] = (round(l["candidate_tokens"] / l["pool_tokens"], 4)
                                if l["pool_tokens"] else 0.0)
    return c


def all_arms(scale="scale_1b"):
    base = baseline_config(scale)
    out = [base]
    for a in spec.PROXY["arms"]:
        if a["id"] == "A0":
            continue
        c = copy.deepcopy(base)
        c["arm_id"], c["arm_name"] = a["id"], a["name"]
        c["change"], c["tests"] = a["change"], a["tests"]
        out.append(recompute(MUTATIONS[a["id"]](c)))
    return out


def validate(arms):
    """An arm whose lane shares do not sum to 100 is not testing what it says."""
    problems = []
    for c in arms:
        for i, ph in enumerate(c["phases"]):
            tot = sum(l["phase_shares"][i] for l in c["lanes"].values())
            if abs(tot - 100.0) > 1e-6:
                problems.append(f"{c['arm_id']} phase {ph['id']}: shares sum to {tot:.3f}")
        fl = sum(c["floor"].values())
        if fl >= 100.0:
            problems.append(f"{c['arm_id']}: floor {fl} >= 100")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="DIR", help="write one JSON per arm")
    ap.add_argument("--scale", default="scale_1b", choices=["scale_1b", "scale_3b"])
    args = ap.parse_args()

    arms = all_arms(args.scale)
    problems = validate(arms)

    for c in arms:
        print(f"{c['arm_id']:3s} {c['arm_name']:20s} "
              f"{c['params']/1e9:.0f}B params x {c['tokens']/1e9:.0f}B tokens  "
              f"({c['tokens_per_param']:.0f} tok/param)")
        for lid, l in c["lanes"].items():
            if l["whole_run_share"] < 0.01 and sum(l["phase_shares"]) == 0:
                continue
            print(f"      {lid:10s} pool={l['pool_tokens']/1e6:8.1f}M  "
                  f"trained={l['trained_tokens']/1e6:8.1f}M  "
                  f"epochs={l['expected_epochs']:6.2f}")

    if problems:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"\nvalidated {len(arms)} arms: all phase shares sum to 100")

    if args.json:
        os.makedirs(args.json, exist_ok=True)
        for c in arms:
            path = os.path.join(args.json, f"{c['arm_id']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=1)
        print(f"wrote {len(arms)} configs to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
