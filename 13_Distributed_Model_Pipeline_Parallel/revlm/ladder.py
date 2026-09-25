"""Find the largest batch each variant can actually run, and chart the memory it costs.

Two things have to be right for "the largest batch that fits" to mean anything.

**A fresh subprocess per probe.** Once a CUDA allocation fails, the caching allocator is
left fragmented, and a later probe in the same process can fail at a batch size it would
otherwise have handled. An in-process ladder reports a maximum batch that is too low, and
reports it consistently enough to look trustworthy.

**A hard cap on the allocator.** This matters more, and it is easy to miss. On Windows
(WDDM) the driver will happily oversubscribe VRAM by paging to host RAM rather than
raising an error. A first run of this ladder "fitted" batch 96 of the storing baseline at
a reported peak of 11.98 GiB - on an 8.00 GiB card - and the only symptom was throughput
collapsing to 0.16x. Every one of those maxima was a fiction. So each probe caps the
caching allocator at the memory actually free on the device, which turns silent paging
back into the honest `OutOfMemoryError` it should have been. `degraded` flags any probe
whose throughput fell far below its mode's best, as a second line of defence.

    python -m revlm.ladder                      # run the full ladder
    python -m revlm.ladder --probe midpoint 64  # one probe (used internally)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

OOM_EXIT = 17


def probe_once(mode: str, batch: int, n_layer: int = 10, ce_chunks: int = 1,
               steps: int = 12, block_size: int = 512, vocab_size: int = 8192) -> dict:
    """The body of a single probe. Runs in its own process; never call this directly."""
    import torch

    from . import metering as M
    from .model import GPT, GPTConfig

    # cap the allocator at real free VRAM: on WDDM the driver pages instead of failing,
    # which turns an out-of-memory into a 5x slowdown that looks like a successful run
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(free / total)

    cfg = GPTConfig(n_layer=n_layer, vocab_size=vocab_size, block_size=block_size)
    torch.manual_seed(0)
    model = GPT(cfg, mode=mode, ce_chunks=ce_chunks).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device="cuda")

    for _ in range(2):
        model(x, y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    state = M.allocated_gib()
    M.reset_peak()
    timer = M.Timer(warmup=3)
    for _ in range(steps):
        timer.start()
        model(x, y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        timer.stop(batch * cfg.block_size)
    return {
        "mode": mode, "batch": batch, "n_layer": n_layer, "ce_chunks": ce_chunks,
        "peak_alloc_gib": M.peak_gib(), "peak_reserved_gib": M.reserved_peak_gib(),
        "state_gib": state, "per_batch_gib": max(0.0, M.peak_gib() - state),
        "tokens_per_sec": timer.tokens_per_sec,
        "vram_cap_gib": free / 2 ** 30,
        "analytic": M.analytic_activation_gib(cfg, batch, mode),
    }


def _run_probe(mode, batch, n_layer, ce_chunks, timeout=420):
    """Spawn one probe. Returns a dict, or None if it ran out of memory."""
    cmd = [sys.executable, "-m", "revlm.ladder", "--probe", mode, str(batch),
           str(n_layer), str(ce_chunks)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             env=env, cwd=os.path.dirname(os.path.dirname(__file__)))
    except subprocess.TimeoutExpired:
        return None
    if out.returncode == OOM_EXIT:
        return None
    if out.returncode != 0:
        raise RuntimeError(f"probe {mode}@{batch} failed:\n{out.stderr[-2000:]}")
    for line in reversed(out.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"probe {mode}@{batch} produced no result:\n{out.stdout[-1000:]}")


def ladder(modes, batches, n_layer=10, ce_chunks=1, log=print):
    """Walk each mode up the batch sizes until it runs out of memory."""
    rows, max_batch = [], {}
    for mode in modes:
        log(f"  {mode}:")
        mine = []
        for b in sorted(batches):
            res = _run_probe(mode, b, n_layer, ce_chunks)
            if res is None:
                log(f"    batch {b:4d}  OOM")
                break
            mine.append(res)
            max_batch[mode] = b
            log(f"    batch {b:4d}  peak {res['peak_alloc_gib']:5.2f} GiB  "
                f"{res['tokens_per_sec']:7.0f} tok/s")
        # second line of defence: a probe that "fits" but runs at a fraction of the mode's
        # best throughput did not really fit
        best = max((r["tokens_per_sec"] for r in mine), default=1.0)
        for r in mine:
            r["degraded"] = r["tokens_per_sec"] < 0.6 * best
            if r["degraded"]:
                log(f"    batch {r['batch']:4d}  DEGRADED "
                    f"({r['tokens_per_sec']/best:.2f}x of best) - not counted as fitting")
        healthy = [r["batch"] for r in mine if not r["degraded"]]
        if healthy:
            max_batch[mode] = max(healthy)
        rows.extend(mine)
    return {"rows": rows, "max_batch": max_batch,
            "n_layer": n_layer, "ce_chunks": ce_chunks}


def depth_sweep(modes, depths, batch=16, log=print):
    """Peak memory against depth: the claim that reversibility is flat in L."""
    rows = []
    for mode in modes:
        for L in depths:
            res = _run_probe(mode, batch, L, 1)
            if res is None:
                log(f"  {mode} L={L}: OOM")
                continue
            rows.append(res)
            log(f"  {mode:<11} L={L:3d}  peak {res['peak_alloc_gib']:5.2f} GiB")
    return rows


def main(argv):
    if len(argv) > 1 and argv[1] == "--probe":
        import torch

        mode, batch = argv[2], int(argv[3])
        n_layer = int(argv[4]) if len(argv) > 4 else 10
        ce_chunks = int(argv[5]) if len(argv) > 5 else 1
        try:
            print(json.dumps(probe_once(mode, batch, n_layer, ce_chunks)))
        except torch.OutOfMemoryError:
            sys.exit(OOM_EXIT)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                sys.exit(OOM_EXIT)
            raise
        return

    from .train import environment

    modes = ["store", "checkpoint", "euler", "midpoint", "coupling"]
    grid = [8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256]
    print("batch ladder (fresh process per probe)")
    out = ladder(modes, grid)
    out["environment"] = environment()
    print("\ndepth sweep")
    out["depth_rows"] = depth_sweep(["store", "checkpoint", "euler", "midpoint"],
                                    [4, 8, 16, 32])
    os.makedirs("assets", exist_ok=True)
    json.dump(out, open("assets/ladder.json", "w"), indent=2)
    print("\nmax batch per mode:", out["max_batch"])


if __name__ == "__main__":
    main(sys.argv)
