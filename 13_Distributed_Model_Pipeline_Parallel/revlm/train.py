"""One training run, instrumented.

Everything that makes runs comparable lives here: the same seed, the same sampler stream,
the same schedule shape, weight decay and dropout off everywhere, and a token budget rather
than a step budget - so a run at batch 192 and a run at batch 48 both see exactly 50M
tokens and the comparison is of integrators, not of how long each was allowed to train.
"""
from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import dataclass, field, asdict

import torch

from . import metering as M
from . import reversible as R
from .data import Batches, decode
from .model import GPT, GPTConfig


@dataclass
class RunSpec:
    name: str
    mode: str
    batch_size: int
    tokens: int = 50_000_000
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    euler_iters: int = 4
    ce_chunks: int = 1
    seed: int = 1234
    gamma_init: float = 0.1
    n_layer: int = 10
    note: str = ""
    tags: list = field(default_factory=list)

    def dict(self):
        return asdict(self)


def lr_at(step: int, total: int, spec: RunSpec) -> float:
    """Linear warm-up then cosine decay - the shape is identical in every run."""
    warm = max(1, int(total * spec.warmup_frac))
    if step < warm:
        return spec.lr * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    floor = spec.lr * spec.min_lr_frac
    return floor + 0.5 * (spec.lr - floor) * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model, val_path, block_size, batch_size, n_batches=20, seed=99):
    model.eval()
    batches = Batches(val_path, batch_size, block_size, seed=seed)
    total = 0.0
    for _ in range(n_batches):
        x, y = batches.next()
        total += model(x, y).item()
    model.train()
    return total / n_batches


@torch.no_grad()
def sample(model, tok_path, n_tokens=120, seed=7):
    model.eval()
    torch.manual_seed(seed)
    idx = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    out = model.generate(idx, max_new_tokens=n_tokens, temperature=0.8, top_k=40)
    model.train()
    return decode(tok_path, out[0].tolist())


def cap_allocator():
    """Cap the caching allocator at the VRAM actually free on the device.

    On Windows (WDDM) the driver oversubscribes VRAM by paging to host RAM instead of
    raising an error, so a run that does not fit does not fail - it just gets ~5x slower,
    and reports a peak larger than the card. Both the throughput and the memory number
    would then be fiction. Capping the allocator turns that back into a real
    OutOfMemoryError.
    """
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(free / total)
    return free / 2 ** 30


def train_one(spec: RunSpec, meta: dict, out_dir: str, log=print) -> dict:
    """Run `spec` to completion and return everything the report needs."""
    cap = cap_allocator()
    torch.manual_seed(spec.seed)
    cfg = GPTConfig(vocab_size=meta["vocab_size"], gamma_init=spec.gamma_init,
                    n_layer=spec.n_layer)
    model = GPT(cfg, mode=spec.mode, euler_iters=spec.euler_iters,
                ce_chunks=spec.ce_chunks).cuda()
    model.track_recon = False
    opt = torch.optim.AdamW(model.param_groups(), lr=spec.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay)

    steps = max(1, spec.tokens // (spec.batch_size * cfg.block_size))
    batches = Batches(meta["train_bin"], spec.batch_size, cfg.block_size, seed=spec.seed)
    timer = M.Timer(warmup=5)

    # the fixed floor: weights + gradients + Adam moments, measured with nothing live
    for _ in range(2):
        x, y = batches.next()
        model(x, y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    state_gib = M.allocated_gib()
    M.reset_peak()

    history, diag_hist = [], []
    t_start = time.time()
    log(f"  {spec.name}: {steps} steps x {spec.batch_size}x{cfg.block_size} tokens",
        flush=True)

    for step in range(steps):
        lr = lr_at(step, steps, spec)
        for g in opt.param_groups:
            g["lr"] = lr
        # Sample the reversibility diagnostics on exactly the cadence the history is
        # logged on. (They used to be sampled every steps//20 while history was written
        # every steps//50; those two coincide only where both divide the step, which for
        # 2034 steps meant step 0 and nowhere else - a whole run of "diagnostics" that was
        # a single sample repeated.)
        model.track_recon = (step % max(1, steps // 50) == 0)

        timer.start()
        x, y = batches.next()
        loss = model(x, y)
        loss.backward()
        if spec.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        timer.stop(spec.batch_size * cfg.block_size)

        if step % max(1, steps // 50) == 0 or step == steps - 1:
            lv = loss.item()
            history.append({"step": step, "tokens": (step + 1) * spec.batch_size * cfg.block_size,
                            "loss": lv, "lr": lr})
            if model.track_recon and model.diag:
                d = dict(model.diag)
                d["step"] = step
                # is the implicit-Euler contraction condition still holding?
                if spec.mode == "euler_implicit":
                    h = model.embed(x[:2])
                    d["lipschitz"] = max(R.contraction_estimate(b, h) for b in model.blocks[:3])
                diag_hist.append(d)
            log(f"    step {step:5d}/{steps}  loss {lv:6.3f}  "
                f"{timer.tokens_per_sec:7.0f} tok/s  peak {M.peak_gib():5.2f} GiB",
                flush=True)

    wall = time.time() - t_start
    peak_alloc, peak_res = M.peak_gib(), M.reserved_peak_gib()
    val = evaluate(model, meta["val_bin"], cfg.block_size, min(spec.batch_size, 16))
    try:
        text = sample(model, meta["tokenizer"])
    except Exception as exc:  # generation is a nicety, never a reason to lose a run
        text = f"<sampling failed: {exc}>"

    result = {
        "spec": spec.dict(),
        "params": sum(p.numel() for p in model.parameters()),
        "steps": steps,
        "tokens_seen": steps * spec.batch_size * cfg.block_size,
        "final_train_loss": history[-1]["loss"],
        "final_val_loss": val,
        "tokens_per_sec": timer.tokens_per_sec,
        "timed_steps": timer.counted_steps,
        "peak_alloc_gib": peak_alloc,
        "peak_reserved_gib": peak_res,
        "state_gib": state_gib,
        "per_batch_gib": max(0.0, peak_alloc - state_gib),
        "wall_seconds": wall,
        "vram_cap_gib": cap,
        "history": history,
        "diagnostics": diag_hist,
        "sample": text,
        "analytic": M.analytic_activation_gib(cfg, spec.batch_size, spec.mode),
    }
    log(f"    done: val {val:.4f}  {timer.tokens_per_sec:.0f} tok/s  "
        f"peak {peak_alloc:.2f} GiB  {wall/60:.1f} min")

    del model, opt, batches
    torch.cuda.empty_cache()
    return result


def environment() -> dict:
    p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "gpu": p.name if p else "cpu",
        "gpu_total_gib": round(p.total_memory / 2 ** 30, 2) if p else 0,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "compiled": False,  # no triton on Windows, so every run is eager - stated, not hidden
        "allocator_capped": True,  # see cap_allocator(): WDDM pages instead of OOMing
    }


def save(results: dict, out_dir: str, name: str = "results.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    json.dump(results, open(path, "w"), indent=2)
    return path
