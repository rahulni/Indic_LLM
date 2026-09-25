"""Measuring instruments: throughput, peak memory, and where the peak actually goes.

Two rules this file exists to enforce.

**Synchronise before you look at the clock.** CUDA calls are asynchronous, so a naive
`time.time()` around a training step times the *launch* of the work, not the work. Every
throughput number here brackets `torch.cuda.synchronize()`.

**Warm-up is not the run.** The first few steps pay for cuBLAS autotuning and allocator
growth. They are timed separately and excluded, and the excluded count is reported so the
number can be audited rather than trusted.
"""
from __future__ import annotations

import contextlib
import time

import torch


def cuda() -> bool:
    return torch.cuda.is_available()


def reset_peak():
    if cuda():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def allocated_gib() -> float:
    return torch.cuda.memory_allocated() / 2 ** 30 if cuda() else 0.0


def peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2 ** 30 if cuda() else 0.0


def reserved_peak_gib() -> float:
    """What the allocator took from the driver. This - not `allocated` - is what OOMs."""
    return torch.cuda.max_memory_reserved() / 2 ** 30 if cuda() else 0.0


@contextlib.contextmanager
def peak_scope():
    """`with peak_scope() as p: ...` then read `p['alloc']` / `p['reserved']` after."""
    reset_peak()
    out = {}
    try:
        yield out
    finally:
        if cuda():
            torch.cuda.synchronize()
        out["alloc"] = peak_gib()
        out["reserved"] = reserved_peak_gib()


class Timer:
    """Wall-clock over a synchronised region, warm-up excluded."""

    def __init__(self, warmup: int = 5):
        self.warmup, self.n, self.elapsed, self.tokens = warmup, 0, 0.0, 0
        self._t0 = None

    def start(self):
        if cuda():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self, tokens: int):
        if cuda():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self.n += 1
        if self.n > self.warmup:          # the first `warmup` steps are not the run
            self.elapsed += dt
            self.tokens += tokens
        return dt

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def counted_steps(self) -> int:
        return max(0, self.n - self.warmup)


def optimizer_state_gib(model, optimizer) -> float:
    """Parameters + gradients + Adam moments actually resident, in GiB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(p.grad.numel() * p.grad.element_size()
                 for p in model.parameters() if p.grad is not None)
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total / 2 ** 30


def decompose_peak(make_step, base_gib: float) -> dict:
    """Split a step's peak into the fixed floor and what the batch adds.

    `base_gib` is the steady-state floor (weights + grads + Adam states), measured between
    steps when nothing is live. Everything above it is what one batch costs, which is the
    only part reversibility can touch.
    """
    with peak_scope() as p:
        make_step()
    return {
        "peak_alloc_gib": p["alloc"],
        "peak_reserved_gib": p["reserved"],
        "state_gib": base_gib,
        "per_batch_gib": max(0.0, p["alloc"] - base_gib),
    }


def analytic_activation_gib(cfg, batch: int, mode: str, bytes_per_elem: int = 4) -> dict:
    """What the arithmetic says the residual stream should cost, before measuring it.

    Worth writing down before looking at the CUDA counters: if measurement and arithmetic
    disagree, one of them is wrong, and it is usually a mental model rather than a GPU.
    """
    bt = batch * cfg.block_size
    stream = bt * cfg.n_embd * bytes_per_elem / 2 ** 30      # one residual-stream tensor
    # a stored block keeps roughly a dozen intermediates of stream size (qkv is 3x, the
    # MLP hidden is 4x, plus the two LayerNorms and the residual adds)
    per_block_stored = stream * 12
    logits = bt * cfg.vocab_size * 4 / 2 ** 30               # fp32, the head's bill

    if mode in ("store",):
        act = per_block_stored * cfg.n_layer
    elif mode == "checkpoint":
        act = stream * cfg.n_layer + per_block_stored        # one input per layer + one live
    else:
        act = stream * 3 + per_block_stored                  # boundaries + one live layer
    return {"stream_gib": stream, "activations_gib": act, "logits_gib": logits,
            "total_gib": act + logits}
