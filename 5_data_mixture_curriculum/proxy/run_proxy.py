# -*- coding: utf-8 -*-
"""
run_proxy.py - orchestrates the proxy screen.

    python run_proxy.py --dry-run              # validate everything, no GPU
    python run_proxy.py --stage pools          # build subsampled pools
    python run_proxy.py --stage train --arm A0
    python run_proxy.py --stage eval  --arm A0
    python run_proxy.py --stage decide         # apply the decision rules

--dry-run executes the full control flow with a stub trainer, so the plumbing,
the configs and the decision logic are all exercised without a GPU. Everything
it prints about cost and shape is real; only the loss numbers are fake, and they
are labelled as such.

Real runs need: torch, transformers, datasets, and the pools from --stage pools.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plan"))

import spec       # noqa: E402
import audit      # noqa: E402
import arms as armlib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")


# --------------------------------------------------------------------------
# Stage 1: pools
# --------------------------------------------------------------------------

def build_pools(cfg, dry_run):
    """Subsample each lane's corpus to the proxy scale.

    This is the step that makes arm A3 meaningful: without it every lane sits
    at ~0.005 epochs and repetition can never be observed."""
    plan = []
    for lid, l in cfg["lanes"].items():
        if l["pool_tokens"] <= 0:
            continue
        plan.append(dict(
            lane=lid, target_tokens=l["pool_tokens"],
            sources=[s["name"] for s in _lane_sources(lid)],
            expected_epochs=l["expected_epochs"],
            seq_min=spec.SEQ_MIN,
        ))
    if dry_run:
        total = sum(p["target_tokens"] for p in plan)
        print(f"  [dry-run] would build {len(plan)} lane pools, "
              f"{total/1e9:.2f}B tokens total")
        for p in plan:
            print(f"      {p['lane']:10s} {p['target_tokens']/1e6:9.1f}M  "
                  f"from {len(p['sources'])} source(s)")
        return plan
    raise NotImplementedError(
        "Real pool construction needs the cleaned corpora. Wire this to the "
        "session-4 pipeline output: for each lane, stream its sources, apply "
        "the same dedup/decontamination gates, and write shards until "
        "target_tokens is reached. Documents below SEQ_MIN are dropped, not "
        "padded.")


def _lane_sources(lane_id):
    return [i for i in spec.INVENTORY if i["lane"] == lane_id]


# --------------------------------------------------------------------------
# Stage 2: train
# --------------------------------------------------------------------------

def _phase_at(step_frac, phases):
    """Which phase a given fraction of training is in, plus crossfade weight."""
    acc = 0.0
    for i, ph in enumerate(phases):
        nxt = acc + ph["weight"]
        if step_frac < nxt or i == len(phases) - 1:
            return i, acc, nxt
        acc = nxt
    return len(phases) - 1, acc, 1.0


def mixture_at(cfg, step_frac):
    """Lane sampling weights at a point in training, with the band crossfade
    applied. This is the function the dataloader calls.

    [TRANSCRIPT REQUIREMENT] The crossfade is why a phase boundary is a ramp
    rather than a step: V4 saw loss and gradient-norm spikes at hard changes."""
    phases = cfg["phases"]
    idx, lo, hi = _phase_at(step_frac, phases)
    overlap = cfg["band_transition"]["overlap_frac"]

    weights = {lid: l["phase_shares"][idx] for lid, l in cfg["lanes"].items()}
    if overlap > 0 and idx + 1 < len(phases):
        span = hi - lo
        into_next = (step_frac - (hi - span * overlap)) / (span * overlap)
        if into_next > 0:
            t = min(1.0, into_next)
            for lid, l in cfg["lanes"].items():
                weights[lid] = (1 - t) * l["phase_shares"][idx] + t * l["phase_shares"][idx + 1]

    tot = sum(weights.values()) or 1.0
    return {k: v / tot for k, v in weights.items()}


def seq_len_at(cfg, step_frac):
    """[TRANSCRIPT REQUIREMENT] Sequence length is a property of the BATCH.
    All examples in a batch share it, and it never drops below SEQ_MIN."""
    idx, _, _ = _phase_at(step_frac, cfg["phases"])
    phase_id = cfg["phases"][idx]["id"]
    rungs = [r for r in cfg["seq_ladder"] if r["phase"] == phase_id]
    if not rungs:
        return spec.SEQ_MIN, 1
    # Deterministic round-robin weighted by share_of_phase.
    total = sum(r["share_of_phase"] for r in rungs)
    pick = (step_frac * 1000.0) % 100.0
    acc = 0.0
    for r in rungs:
        acc += 100.0 * r["share_of_phase"] / total
        if pick < acc:
            return r["seq_len"], r["batch_examples"]
    return rungs[-1]["seq_len"], rungs[-1]["batch_examples"]


def lr_at(cfg, step_frac):
    """[REVIEW FIX 8] Warmup -> cosine to 10% -> linear to zero across phase D.
    Without this the anneal is only a data change."""
    lr = cfg["lr"]
    warm = lr["warmup_frac"]
    anneal_start = 1.0 - cfg["phases"][-1]["weight"]

    if step_frac < warm:
        return lr["peak"] * (step_frac / warm)
    if step_frac < anneal_start:
        import math
        t = (step_frac - warm) / (anneal_start - warm)
        floor = lr["anneal_start_frac"]
        return lr["peak"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))
    t = (step_frac - anneal_start) / (1.0 - anneal_start)
    return lr["peak"] * lr["anneal_start_frac"] * (1.0 - t)


def train(cfg, dry_run, probes=9):
    print(f"  arm {cfg['arm_id']} ({cfg['arm_name']}): "
          f"{cfg['params']/1e9:.0f}B params, {cfg['tokens']/1e9:.0f}B tokens")
    print(f"    {'frac':>6s} {'phase':>5s} {'seq':>7s} {'batch':>6s} "
          f"{'lr':>10s}  top lanes")
    for i in range(probes):
        f = i / (probes - 1.0)
        idx, _, _ = _phase_at(f, cfg["phases"])
        mix = mixture_at(cfg, f)
        sl, be = seq_len_at(cfg, f)
        top = sorted(mix.items(), key=lambda kv: -kv[1])[:3]
        top_s = " ".join(f"{k}={v*100:.0f}%" for k, v in top)
        print(f"    {f:6.2f} {cfg['phases'][idx]['id']:>5s} {sl:7d} {be:6d} "
              f"{lr_at(cfg, f):10.2e}  {top_s}")
    if dry_run:
        return dict(arm=cfg["arm_id"], stub=True,
                    note="STUB - no model was trained; metrics below are placeholders")
    raise NotImplementedError(
        "Real training needs torch + transformers. The dataloader must call "
        "mixture_at(), seq_len_at() and lr_at() per step; batches must be "
        "length-homogeneous; loss must be masked per section 12 of the plan.")


# --------------------------------------------------------------------------
# Stage 3: eval
# --------------------------------------------------------------------------

def evaluate(cfg, dry_run):
    metrics = [m["metric"] for m in spec.PROXY["metrics"]]
    if dry_run:
        print(f"  [dry-run] arm {cfg['arm_id']}: would score {len(metrics)} metrics")
        for m in spec.PROXY["metrics"]:
            print(f"      {m['lane']:10s} {m['metric']}")
        return {m: None for m in metrics}
    raise NotImplementedError(
        "Real eval needs lm-evaluation-harness (MMLU, GSM8K), bigcode-evaluation-"
        "harness (HumanEval+, MBPP+), the BFCL runner, RULER, and FLORES-200 "
        "with chrF++. Indic results are reported per language, never averaged.")


# --------------------------------------------------------------------------
# Stage 4: decide
# --------------------------------------------------------------------------

def decide(results, dry_run):
    """Apply the pre-registered rules. They are evaluated mechanically so a
    result cannot be reinterpreted after the fact."""
    print("  Pre-registered decision rules:")
    for d in spec.PROXY["decisions"]:
        have = all(results.get(a) for a in ("A0",)) and not dry_run
        status = "PENDING - no results yet" if not have else "EVALUATE"
        print(f"    [{status}] {d['rule']}")
        print(f"        iff:  {d['iff']}")
        print(f"        else: {d['else_']}")
    if dry_run:
        print("\n  [dry-run] rules not evaluated - no real metrics exist.")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "pools", "train", "eval", "decide"])
    ap.add_argument("--arm", default=None)
    ap.add_argument("--scale", default="scale_1b", choices=["scale_1b", "scale_3b"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_cfgs = armlib.all_arms(args.scale)
    problems = armlib.validate(all_cfgs)
    if problems:
        print("arm validation failed:")
        for p in problems:
            print("  -", p)
        return 1
    cfgs = [c for c in all_cfgs if not args.arm or c["arm_id"] == args.arm]
    if not cfgs:
        print(f"no arm matching {args.arm}")
        return 1

    cost = audit.proxy_cost(spec.BUDGET_TOKENS)
    key = "arm_1b" if args.scale == "scale_1b" else "arm_3b"
    print(f"proxy screen: {len(cfgs)} arm(s) at {args.scale}, "
          f"{cost[key]['hours']:.1f} h/arm, "
          f"${cost[key]['usd']:,.0f}/arm, "
          f"${cost[key]['usd']*len(cfgs):,.0f} total")
    if args.dry_run:
        print("DRY RUN - no GPU, no model, no data. Control flow only.\n")

    os.makedirs(OUT, exist_ok=True)
    results = {}
    t0 = time.time()

    if args.stage in ("all", "pools"):
        print("\n== stage: pools ==")
        build_pools(cfgs[0], args.dry_run)
    if args.stage in ("all", "train"):
        print("\n== stage: train ==")
        for c in cfgs:
            results[c["arm_id"]] = train(c, args.dry_run)
    if args.stage in ("all", "eval"):
        print("\n== stage: eval ==")
        for c in cfgs:
            evaluate(c, args.dry_run)
    if args.stage in ("all", "decide"):
        print("\n== stage: decide ==")
        decide(results, args.dry_run)

    print(f"\nelapsed {time.time()-t0:.2f}s")
    if args.dry_run:
        path = os.path.join(OUT, "dry_run.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(arms=[c["arm_id"] for c in cfgs],
                           scale=args.scale, stub=True), f, indent=1)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
