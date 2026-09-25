"""Throughput, measured on a GPU whose clock will not hold still.

This file exists because of a measurement failure worth describing, since anyone repeating
this assignment on a laptop will hit it.

The batch ladder timed the storing baseline at 104,429 tokens/s early in the session, and
the *identical* probe at 51,948 tokens/s an hour later. Nothing in the code had changed.
The RTX 3070 Laptop had simply heated up: SM clock 1560 MHz against a 2100 MHz maximum,
with `clocks_throttle_reasons` reporting 0x24 - thermal slowdown plus power cap.

A 2x drift is larger than every effect this project is trying to measure. Worse, it is
*ordered*: a matrix run front to back gives its first variant a cold GPU and its last a hot
one, so whichever integrator runs first wins on speed regardless of its merits. In the
contaminated ladder this produced the nonsense that `coupling` and `midpoint` were *faster*
than the baseline they are built on.

Clock locking (`nvidia-smi -lgc`) needs privileges this user does not have, so the fix is
experimental design rather than configuration:

  1. **Warm to equilibrium first.** A fixed load runs until the clock stops falling, so no
     measurement happens on the way down from boost.
  2. **Balance the order (ABBA).** Every mode is measured once going forwards and once
     going backwards, and the two are averaged. Any drift that is roughly linear in time
     cancels, because each mode sits at the mirrored position in the second pass.
  3. **Record the conditions.** Clock, temperature and throttle reasons are captured with
     every measurement, so a reader can see how stable the machine was rather than trust
     that it was.

    python -m revlm.bench            # -> assets/throughput.json
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import torch

from . import metering as M
from .model import GPT, GPTConfig

MODES = ["store", "checkpoint", "euler", "midpoint", "coupling", "euler_implicit"]


def gpu_state() -> dict:
    """SM clock, temperature and throttle reasons, straight from the driver."""
    q = ("clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
         "clocks_throttle_reasons.active")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return {"sm_mhz": float(out[0]), "sm_max_mhz": float(out[1]),
                "temp_c": float(out[2]), "power_w": float(out[3]),
                "throttle": out[4].strip()}
    except Exception:
        return {}


def _build(mode, batch, cfg, euler_iters=4, ce_chunks=1):
    torch.manual_seed(0)
    m = GPT(cfg, mode=mode, euler_iters=euler_iters, ce_chunks=ce_chunks).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device="cuda")
    return m, opt, x


def warm_to_equilibrium(cfg, batch=32, seconds=90, log=print):
    """Run a fixed load until the clock stops falling, so nothing is timed on the way down."""
    m, opt, x = _build("store", batch, cfg)
    t0, last = time.time(), None
    while time.time() - t0 < seconds:
        for _ in range(20):
            m(x, x).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        st = gpu_state()
        if last is not None and abs(st.get("sm_mhz", 0) - last) < 30:
            log(f"  settled at {st.get('sm_mhz')} MHz, {st.get('temp_c')} C")
            break
        last = st.get("sm_mhz", 0)
    del m, opt, x
    torch.cuda.empty_cache()
    return gpu_state()


def measure(mode, cfg, batch, seconds=12.0, euler_iters=4, ce_chunks=1):
    """Tokens per second for one mode over a fixed wall-clock window."""
    m, opt, x = _build(mode, batch, cfg, euler_iters, ce_chunks)
    for _ in range(4):                       # per-mode kernel warm-up, not thermal
        m(x, x).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    before = gpu_state()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        m(x, x).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        n += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out = {"mode": mode, "batch": batch, "steps": n,
           "tokens_per_sec": n * batch * cfg.block_size / dt,
           "peak_alloc_gib": M.peak_gib(), "before": before, "after": gpu_state()}
    del m, opt, x
    torch.cuda.empty_cache()
    return out


def run(batch=32, seconds=12.0, log=print):
    cfg = GPTConfig()
    log("warming to thermal equilibrium")
    settled = warm_to_equilibrium(cfg, log=log)

    log("\npass 1 (forwards)")
    fwd = []
    for mode in MODES:
        r = measure(mode, cfg, batch, seconds)
        fwd.append(r)
        log(f"  {mode:<16}{r['tokens_per_sec']:>9,.0f} tok/s   "
            f"{r['after'].get('sm_mhz','?')} MHz  {r['after'].get('temp_c','?')} C")

    log("\npass 2 (backwards - cancels any drift that is linear in time)")
    bwd = []
    for mode in reversed(MODES):
        r = measure(mode, cfg, batch, seconds)
        bwd.append(r)
        log(f"  {mode:<16}{r['tokens_per_sec']:>9,.0f} tok/s   "
            f"{r['after'].get('sm_mhz','?')} MHz  {r['after'].get('temp_c','?')} C")

    merged = {}
    for r in fwd + bwd:
        merged.setdefault(r["mode"], []).append(r["tokens_per_sec"])
    base = sum(merged["store"]) / len(merged["store"])
    summary = []
    for mode in MODES:
        vals = merged[mode]
        mean = sum(vals) / len(vals)
        summary.append({"mode": mode, "tokens_per_sec": mean,
                        "passes": vals, "spread_pct": (max(vals) - min(vals)) / mean * 100,
                        "slower_than_baseline": base / mean})

    log(f"\n{'mode':<16}{'tok/s (mean)':>14}{'spread':>9}{'vs baseline':>13}")
    for s in summary:
        log(f"  {s['mode']:<14}{s['tokens_per_sec']:>14,.0f}{s['spread_pct']:>8.1f}%"
            f"{s['slower_than_baseline']:>12.2f}x")

    out = {"batch": batch, "window_seconds": seconds, "settled": settled,
           "forward_pass": fwd, "backward_pass": bwd, "summary": summary,
           "design": "ABBA order-balanced; each mode measured once per direction and "
                     "averaged, after warming to thermal equilibrium"}
    os.makedirs("assets", exist_ok=True)
    json.dump(out, open("assets/throughput.json", "w"), indent=2)
    log("\nwrote assets/throughput.json")
    return out


if __name__ == "__main__":
    run()
