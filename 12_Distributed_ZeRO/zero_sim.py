"""zero_sim: 32 virtual GPUs in one Python process, with ZeRO stages 0-3 running on them.

Read top to bottom. Each section builds on the one before it.

  1. VirtualGPU       a memory ledger per virtual GPU: what a real GPU would be holding
  2. ThreadComm       collectives between threads, charged with the ring cost model
  3. ring_all_reduce  the real ring algorithm, passed chunk by chunk between neighbours
  4. VirtualCluster   run one function on N threads, the way torchrun runs N processes
  5. TinyGPT          the demo model, cut into "units" (the granularity ZeRO shards at)
  6. Engine           one executor; DDP / ZeRO1 / ZeRO2 / ZeRO3 differ only in their hooks
  7. run_training     the driver the notebook and the tests call

Conventions
  * Psi: parameter count. P: the padded parameter count (each unit is rounded up to a
    multiple of N*64 so every GPU owns an equal, SIMD-aligned slice).
  * "bf16" precision is the ZeRO paper's mixed-precision recipe: bf16 weights and grads,
    plus an fp32 master copy and fp32 Adam m and v, i.e. 2 + 2 + 12 = 16 bytes per parameter.
    "fp32" precision has no master copy: 4 + 4 + 8 = 16 bytes per parameter.
"""
from __future__ import annotations

import gc
import logging
import math
import queue
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)   # "triton not found"

ALIGN = 64            # shard boundaries fall on multiples of 64 elements
MiB = 2 ** 20

# =============================================================================================
# 1. VirtualGPU: a memory ledger
# =============================================================================================

CATEGORIES = ("weights", "grads", "master", "adam_m", "adam_v", "activations", "temp")
MODEL_STATES = ("weights", "grads", "master", "adam_m", "adam_v")   # what ZeRO shards


class VirtualOOMError(RuntimeError):
    """Raised when an allocation would exceed a virtual GPU's capacity (like CUDA OOM)."""


class _ActHandle:
    """What autograd stores instead of a saved activation, so its bytes can be released
    on the GPU that owns them when autograd drops it (not on whichever thread runs __del__)."""
    __slots__ = ("gpu", "key", "t")

    def __init__(self, gpu, key, t):
        self.gpu, self.key, self.t = gpu, key, t

    def __del__(self):
        try:
            self.gpu._act_release(self.key)
        except Exception:        # interpreter shutdown
            pass


class VirtualGPU:
    """One virtual GPU. Tensors live in ordinary CPU (or real-GPU) memory; this object keeps
    the books: every buffer the training algorithm holds is recorded by name and category.

    With `capacity` set, an allocation that would not fit raises VirtualOOMError *before*
    the real tensor is created, so the ledger enforces the limit just as a real GPU would.
    """

    def __init__(self, rank: int, capacity: int | None = None, device: str = "cpu"):
        self.rank = rank
        self.capacity = capacity
        self.device = torch.device(device)
        self._lock = threading.RLock()
        self._held: dict[str, tuple[str, int]] = {}
        self._act_refs: dict[int, int] = {}
        self.by_cat = dict.fromkeys(CATEGORIES, 0)
        self.current = 0
        self.peak = 0
        self.peak_by_cat = dict(self.by_cat)
        self.cat_peak = dict(self.by_cat)        # each category's own maximum
        self.marks: dict[str, dict] = {}
        self.phase = "idle"
        self.timeline: list | None = None        # set to [] to record every alloc/free

    # -- bookkeeping -------------------------------------------------------------------------
    def alloc(self, name: str, nbytes: int, category: str) -> None:
        with self._lock:
            if name in self._held:
                raise KeyError(f"vGPU {self.rank}: '{name}' is already allocated")
            if self.capacity is not None and self.current + nbytes > self.capacity:
                raise VirtualOOMError(self._oom_message(name, nbytes, category))
            self._held[name] = (category, nbytes)
            self.by_cat[category] += nbytes
            self.current += nbytes
            if self.current > self.peak:
                self.peak = self.current
                self.peak_by_cat = dict(self.by_cat)
            if self.by_cat[category] > self.cat_peak[category]:
                self.cat_peak[category] = self.by_cat[category]
            self._event()

    def free(self, name: str) -> None:
        with self._lock:
            category, nbytes = self._held.pop(name)
            self.by_cat[category] -= nbytes
            self.current -= nbytes
            self._event()

    def holds(self, name: str) -> bool:
        return name in self._held

    def tensor(self, name, numel, dtype, category, zero=True) -> torch.Tensor:
        """Charge the ledger, then create the tensor (so an OOM never allocates)."""
        nbytes = numel * dtype.itemsize
        self.alloc(name, nbytes, category)
        make = torch.zeros if zero else torch.empty
        return make(numel, dtype=dtype, device=self.device)

    def reset_peak(self) -> None:
        with self._lock:
            self.peak = self.current
            self.peak_by_cat = dict(self.by_cat)
            self.cat_peak = dict(self.by_cat)

    def mark(self, label: str) -> None:
        """Snapshot the ledger at a named point of the step (e.g. 'end_of_backward')."""
        with self._lock:
            self.marks[label] = dict(self.by_cat)

    def model_state_bytes(self, snapshot: dict | None = None) -> int:
        snap = self.by_cat if snapshot is None else snapshot
        return sum(snap[c] for c in MODEL_STATES)

    def _event(self) -> None:
        if self.timeline is not None:
            b = self.by_cat
            self.timeline.append((self.phase, self.current, b["weights"], b["grads"],
                                  b["master"] + b["adam_m"] + b["adam_v"],
                                  b["activations"], b["temp"]))

    def _oom_message(self, name, nbytes, category) -> str:
        held = ", ".join(f"{c} {v / MiB:.2f}" for c, v in self.by_cat.items() if v)
        return (f"vGPU {self.rank} out of memory: tried to allocate {nbytes / MiB:.2f} MiB for "
                f"'{name}' ({category}). Capacity {self.capacity / MiB:.2f} MiB; "
                f"{self.current / MiB:.2f} MiB already held ({held or 'nothing'}); "
                f"{(self.capacity - self.current) / MiB:.2f} MiB free.")

    # -- activations: counted automatically as autograd saves them --------------------------
    def saved_tensor_hooks(self, is_param_storage):
        """Context manager: every tensor autograd saves for backward is charged to this GPU
        as 'activations' (once per storage), and released when autograd lets go of it.
        Weights are skipped: they are already on the books as weights."""
        gpu = self

        def pack(t):
            st = t.untyped_storage()
            n, ptr = st.nbytes(), st.data_ptr()
            if n == 0 or is_param_storage(ptr):
                return t
            gpu._act_acquire(ptr, n)
            return _ActHandle(gpu, ptr, t)

        def unpack(h):
            return h.t if type(h) is _ActHandle else h

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

    def _act_acquire(self, ptr: int, nbytes: int) -> None:
        with self._lock:
            count = self._act_refs.get(ptr, 0)
            if count == 0:
                self.alloc(f"act@{ptr:x}", nbytes, "activations")
            self._act_refs[ptr] = count + 1

    def _act_release(self, ptr: int) -> None:
        with self._lock:
            count = self._act_refs[ptr] - 1
            if count == 0:
                del self._act_refs[ptr]
                self.free(f"act@{ptr:x}")
            else:
                self._act_refs[ptr] = count


# =============================================================================================
# 2. ThreadComm: collectives between threads
# =============================================================================================

@dataclass
class CommStats:
    """Per-GPU communication counters. Bytes are what this GPU *sends* under the ring
    algorithm (the standard cost model), keyed by (tag, op)."""
    bytes: dict = field(default_factory=lambda: defaultdict(int))
    calls: dict = field(default_factory=lambda: defaultdict(int))
    ring_steps: int = 0
    seconds: float = 0.0
    ring_steps_by_tag: dict = field(default_factory=lambda: defaultdict(int))

    def copy(self) -> "CommStats":
        return CommStats(defaultdict(int, self.bytes), defaultdict(int, self.calls),
                         self.ring_steps, self.seconds, defaultdict(int, self.ring_steps_by_tag))

    def since(self, before: "CommStats") -> "CommStats":
        out = CommStats()
        for k, v in self.bytes.items():
            if v - before.bytes.get(k, 0):
                out.bytes[k] = v - before.bytes.get(k, 0)
        for k, v in self.calls.items():
            if v - before.calls.get(k, 0):
                out.calls[k] = v - before.calls.get(k, 0)
        out.ring_steps = self.ring_steps - before.ring_steps
        out.seconds = self.seconds - before.seconds
        for k, v in self.ring_steps_by_tag.items():
            if v - before.ring_steps_by_tag.get(k, 0):
                out.ring_steps_by_tag[k] = v - before.ring_steps_by_tag.get(k, 0)
        return out

    def total_bytes(self, exclude=("loss", "init")) -> int:
        return sum(v for (tag, _), v in self.bytes.items() if tag not in exclude)

    def total_ring_steps(self, exclude=("loss", "init")) -> int:
        """Sequential ring steps (the latency term of the alpha-beta model)."""
        return sum(v for tag, v in self.ring_steps_by_tag.items() if tag not in exclude)

    def total_calls(self, exclude=("loss", "init")) -> int:
        return sum(v for (tag, _), v in self.calls.items() if tag not in exclude)

    def by_op(self, exclude=("loss", "init")) -> dict:
        out = defaultdict(int)
        for (tag, op), v in self.bytes.items():
            if tag not in exclude:
                out[op] += v
        return dict(out)


class ThreadComm:
    """Collectives for `world` threads. Every GPU calls the same collective with its own
    tensor (SPMD, exactly like torch.distributed). Inputs are published in shared slots;
    a barrier makes sure every GPU has arrived before anyone reads, and a second barrier
    makes sure everyone has finished reading before anyone moves on and reuses its buffer.

    Reductions always sum in rank order, in fp32, then divide by N. all_reduce is built as
    reduce_scatter + all_gather, so DDP and ZeRO sum their gradients through the same code
    and get the same bits.
    """

    def __init__(self, world: int, timeout: float = 120.0):
        self.world = world
        self.timeout = timeout
        self._barrier = threading.Barrier(world, timeout=timeout)
        self._slots: list = [None] * world
        self._boxes = [queue.Queue() for _ in range(world)]
        self._aborted = threading.Event()
        self.stats = [CommStats() for _ in range(world)]

    def handle(self, rank: int) -> "RankComm":
        return RankComm(self, rank)

    def abort(self) -> None:
        self._aborted.set()
        self._barrier.abort()

    def _wait(self) -> None:
        self._barrier.wait()

    def _charge(self, rank, tag, op, nbytes, steps, t0) -> None:
        st = self.stats[rank]
        st.bytes[(tag, op)] += int(nbytes)
        st.calls[(tag, op)] += 1
        st.ring_steps += steps
        st.ring_steps_by_tag[tag] += steps
        st.seconds += time.perf_counter() - t0

    # -- the three collectives ZeRO is made of ------------------------------------------------
    def reduce_scatter(self, rank, full, out, tag, average=True, _charge=True):
        """full: this GPU's whole tensor (numel divisible by N). out: receives this GPU's
        1/N slice of the sum (or mean) over all GPUs."""
        t0 = time.perf_counter()
        N, n = self.world, full.numel()
        assert full.dim() == 1 and out.dim() == 1, "collectives work on flat (1-D) buffers"
        assert n % N == 0, f"reduce_scatter needs numel % world == 0 (got {n} % {N})"
        s = n // N
        lo = rank * s
        self._slots[rank] = full
        self._wait()                                             # everyone has published
        acc = self._slots[0][lo:lo + s].to(device=out.device, dtype=torch.float32, copy=True)
        for r in range(1, N):
            acc.add_(self._slots[r][lo:lo + s].to(device=out.device, dtype=torch.float32))
        if average:
            acc.div_(N)
        out.copy_(acc)
        self._wait()                                             # everyone has finished reading
        self._slots[rank] = None
        if _charge:
            self._charge(rank, tag, "reduce_scatter", (N - 1) / N * full.nbytes, N - 1, t0)
        return out

    def all_gather(self, rank, shard, out, tag, _charge=True):
        """shard: this GPU's 1/N slice. out: receives all N slices, concatenated in rank
        order. `shard` may be a view of this GPU's own slice of `out` (in-place gather)."""
        t0 = time.perf_counter()
        N, s = self.world, shard.numel()
        assert shard.dim() == 1 and out.dim() == 1, "collectives work on flat (1-D) buffers"
        assert out.numel() == N * s
        self._slots[rank] = shard
        self._wait()
        for r in range(N):
            dst = out[r * s:(r + 1) * s]
            src = self._slots[r]
            if dst.data_ptr() != src.data_ptr():                # skip copying onto itself
                dst.copy_(src)
        self._wait()
        self._slots[rank] = None
        if _charge:
            self._charge(rank, tag, "all_gather", (N - 1) * shard.nbytes, N - 1, t0)
        return out

    def all_reduce(self, rank, t, tag, average=True):
        """In place. Implemented as reduce-scatter then all-gather, which is also how the
        ring algorithm does it, and why its cost is exactly the sum of the two."""
        t0 = time.perf_counter()
        N, n = self.world, t.numel()
        padded = -(-n // N) * N
        in_place = padded == n and t.is_contiguous()
        work = t.view(-1) if in_place else torch.cat([t.reshape(-1), t.new_zeros(padded - n)])
        shard = torch.empty(padded // N, dtype=t.dtype, device=t.device)
        self.reduce_scatter(rank, work, shard, tag, average, _charge=False)
        self.all_gather(rank, shard, work, tag, _charge=False)
        if not in_place:                                # padded or non-contiguous: copy back
            t.copy_(work[:n].reshape(t.shape))
        self._charge(rank, tag, "all_reduce", 2 * (N - 1) / N * t.nbytes, 2 * (N - 1), t0)
        return t

    def broadcast(self, rank, t, src, tag):
        t0 = time.perf_counter()
        self._slots[rank] = t
        self._wait()
        if rank != src:
            t.copy_(self._slots[src])
        self._wait()
        self._slots[rank] = None
        self._charge(rank, tag, "broadcast", (self.world - 1) / self.world * t.nbytes,
                     self.world - 1, t0)
        return t

    def barrier(self, rank):
        self._wait()

    # -- point to point, used only by the hand-written ring ----------------------------------
    def send(self, rank, dst, payload, nbytes, tag="p2p"):
        """Non-blocking send of any Python object; `nbytes` is what goes on the wire."""
        self._boxes[dst].put((rank, payload))
        self.stats[rank].bytes[(tag, "send")] += int(nbytes)
        self.stats[rank].calls[(tag, "send")] += 1

    def recv(self, rank):
        deadline = time.perf_counter() + self.timeout
        while True:
            if self._aborted.is_set():
                raise threading.BrokenBarrierError("communicator aborted")
            try:
                return self._boxes[rank].get(timeout=0.2)
            except queue.Empty:
                if time.perf_counter() > deadline:
                    raise TimeoutError(f"rank {rank}: recv timed out")


class RankComm:
    """The communicator as one GPU sees it (like a torch.distributed process group)."""

    def __init__(self, comm: ThreadComm, rank: int):
        self._c, self.rank, self.world = comm, rank, comm.world

    @property
    def stats(self) -> CommStats:
        return self._c.stats[self.rank]

    def reduce_scatter(self, full, out, tag, average=True):
        return self._c.reduce_scatter(self.rank, full, out, tag, average)

    def all_gather(self, shard, out, tag):
        return self._c.all_gather(self.rank, shard, out, tag)

    def all_reduce(self, t, tag, average=True):
        return self._c.all_reduce(self.rank, t, tag, average)

    def broadcast(self, t, src, tag):
        return self._c.broadcast(self.rank, t, src, tag)

    def barrier(self):
        return self._c.barrier(self.rank)

    def send(self, dst, payload, nbytes, tag="p2p"):
        return self._c.send(self.rank, dst, payload, nbytes, tag)

    def recv(self):
        return self._c.recv(self.rank)


# =============================================================================================
# 3. The ring all-reduce, by hand
# =============================================================================================

def ring_all_reduce(comm: RankComm, t: torch.Tensor, trace: list | None = None) -> torch.Tensor:
    """Sum `t` over all GPUs using only neighbour-to-neighbour messages.

    The tensor is cut into N chunks. Phase 1 (reduce-scatter, N-1 steps): each GPU sends
    one chunk to its right neighbour and adds the chunk arriving from its left. Afterwards
    GPU r holds the complete sum of chunk (r+1) % N. Phase 2 (all-gather, N-1 steps): the
    completed chunks travel once more around the ring. Every GPU sends 2(N-1) chunks of
    size S/N, i.e. 2(N-1)/N * S bytes, however large N gets.

    trace, if given, receives (phase, step, contributions-per-chunk) after every step.
    """
    N, r = comm.world, comm.rank
    assert t.numel() % N == 0
    chunks = [c.clone() for c in t.reshape(-1).float().chunk(N)]
    have = [1] * N                       # how many GPUs' data each local chunk contains
    right = (r + 1) % N
    if trace is not None:
        trace.append(("start", 0, list(have)))
    for step in range(N - 1):            # phase 1: reduce-scatter
        send_i, recv_i = (r - step) % N, (r - step - 1) % N
        comm.send(right, (send_i, chunks[send_i].clone(), have[send_i]),
                  chunks[send_i].numel() * t.element_size(), tag="ring")
        _, (i, data, count) = comm.recv()
        assert i == recv_i
        chunks[recv_i].add_(data)
        have[recv_i] += count
        if trace is not None:
            trace.append(("reduce-scatter", step + 1, list(have)))
    for step in range(N - 1):            # phase 2: all-gather
        send_i, recv_i = (r + 1 - step) % N, (r - step) % N
        comm.send(right, (send_i, chunks[send_i].clone(), have[send_i]),
                  chunks[send_i].numel() * t.element_size(), tag="ring")
        _, (i, data, count) = comm.recv()
        assert i == recv_i
        chunks[recv_i] = data
        have[recv_i] = count
        if trace is not None:
            trace.append(("all-gather", step + 1, list(have)))
    return torch.cat(chunks).to(t.dtype).reshape(t.shape)


# =============================================================================================
# 4. VirtualCluster: N threads, one function
# =============================================================================================

class VirtualCluster:
    """Runs fn(rank, gpu, comm) on `world` threads, the way `torchrun --nproc N` runs a
    script N times. If any GPU raises, every barrier is aborted (nobody hangs) and the
    error from the lowest rank is re-raised, so the output is the same on every run."""

    def __init__(self, world: int, capacity: int | None = None, device: str = "cpu",
                 timeout: float = 120.0):
        self.world = world
        self.comm = ThreadComm(world, timeout)
        self.gpus = [VirtualGPU(r, capacity, device) for r in range(world)]

    def run(self, fn):
        results, errors = [None] * self.world, [None] * self.world

        def target(rank):
            try:
                results[rank] = fn(rank, self.gpus[rank], self.comm.handle(rank))
            except BaseException as e:           # noqa: BLE001  (propagated below)
                errors[rank] = e
                self.comm.abort()

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)                 # one core per virtual GPU, no oversubscription
        threads = [threading.Thread(target=target, args=(r,), daemon=True,
                                    name=f"vGPU-{r}") for r in range(self.world)]
        try:
            for th in threads:
                th.start()
            for th in threads:
                while th.is_alive():
                    th.join(0.2)
        except KeyboardInterrupt:
            self.comm.abort()
            raise
        finally:
            torch.set_num_threads(old_threads)

        failed = [(r, e) for r, e in enumerate(errors) if e is not None]
        if failed:
            primary = [(r, e) for r, e in failed if not isinstance(e, threading.BrokenBarrierError)]
            rank, err = (primary or failed)[0]
            err.add_note(f"(raised on vGPU {rank}; {len(failed)} of {self.world} GPUs failed)")
            raise err
        return results


# =============================================================================================
# 5. TinyGPT, cut into units
# =============================================================================================

@dataclass
class GPTConfig:
    vocab_size: int = 65
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 64


_MASKS: dict = {}


def _causal_mask(T, device):
    key = (T, str(device), threading.get_ident())      # one per virtual GPU, like real GPUs
    if key not in _MASKS:
        _MASKS[key] = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)
    return _MASKS[key]


class Embed(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        return self.wte(idx) + self.wpe(pos)


class Attention(nn.Module):
    """Causal self-attention written as explicit matmuls. (PyTorch's fused SDPA kernel on
    CPU is invisible to FlopCounterMode, so it would silently count as 0 FLOPs.)"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (z.view(B, T, H, C // H).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // H))
        att = att.masked_fill(_causal_mask(T, x.device), float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Head(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)   # untied from wte

    def forward(self, x):
        return self.head(self.ln_f(x))


def module_names(cfg: GPTConfig) -> list[str]:
    return ["embed"] + [f"block{i}" for i in range(cfg.n_layer)] + ["head"]


def build_module(cfg: GPTConfig, name: str) -> nn.Module:
    if name == "embed":
        return Embed(cfg)
    if name == "head":
        return Head(cfg)
    return Block(cfg)


class TinyGPT(nn.Module):
    """The same model as a plain single-device nn.Module: used for the "one big GPU"
    reference and to load a consolidated checkpoint for text generation."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.parts = nn.ModuleList([build_module(cfg, n) for n in module_names(cfg)])

    def forward(self, idx):
        x = idx
        for m in self.parts:
            x = m(x)
        return x

    def load_flat(self, flats: dict[str, torch.Tensor]) -> "TinyGPT":
        with torch.no_grad():
            for name, mod in zip(module_names(self.cfg), self.parts):
                off = 0
                for p in mod.parameters():
                    p.copy_(flats[name][off:off + p.numel()].view_as(p))
                    off += p.numel()
        return self

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, generator=None):
        for _ in range(n_new):
            logits = self(idx[:, -self.cfg.block_size:])[:, -1].float() / temperature
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def init_weights(cfg: GPTConfig, seed: int = 1337) -> dict[str, torch.Tensor]:
    """GPT-2 style init, as one fp32 flat vector per module (unpadded). Every virtual GPU
    starts from these same numbers, which is what DDP's initial broadcast guarantees."""
    g = torch.Generator().manual_seed(seed)
    flats = {}
    for name in module_names(cfg):
        with torch.device("meta"):
            mod = build_module(cfg, name)
        parts = []
        for pname, p in mod.named_parameters():
            if pname.endswith("proj.weight"):          # residual projections
                w = torch.randn(p.shape, generator=g) * (0.02 / math.sqrt(2 * cfg.n_layer))
            elif p.dim() >= 2:
                w = torch.randn(p.shape, generator=g) * 0.02
            elif pname.endswith("weight"):             # LayerNorm gain
                w = torch.ones(p.shape)
            else:                                      # LayerNorm bias
                w = torch.zeros(p.shape)
            parts.append(w.reshape(-1))
        flats[name] = torch.cat(parts)
    return flats


@dataclass
class UnitSpec:
    """A unit is the granularity at which ZeRO shards and gathers: one flat buffer."""
    name: str
    modules: list
    numel: int          # real parameters
    padded: int         # rounded up to a multiple of world * ALIGN

    def shard(self, world: int) -> int:
        return self.padded // world


def make_units(cfg: GPTConfig, world: int, wrap: str = "block") -> list[UnitSpec]:
    """wrap='block': embed | block0 | ... | head (like FSDP auto-wrapping each block).
    wrap='whole': the entire model is one unit (one giant gather in ZeRO-3)."""
    names = module_names(cfg)
    groups = [[n] for n in names] if wrap == "block" else [names]
    q = world * ALIGN
    specs = []
    for grp in groups:
        numel = 0
        for n in grp:
            with torch.device("meta"):
                numel += sum(p.numel() for p in build_module(cfg, n).parameters())
        specs.append(UnitSpec("+".join(grp) if len(grp) < 3 else "model", grp, numel,
                              -(-numel // q) * q))
    return specs


# =============================================================================================
# 6. Engine: one executor, four policies
# =============================================================================================

@dataclass
class AdamWConfig:
    lr: float = 3e-3
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01      # applied to every element, norms included, for simplicity


class _Unit:
    """Runtime state of one unit on one GPU."""

    def __init__(self, spec: UnitSpec, idx: int, is_last: bool, rank: int, world: int):
        self.spec, self.idx, self.is_last, self.name = spec, idx, is_last, spec.name
        self.lo, self.hi = rank * spec.shard(world), (rank + 1) * spec.shard(world)
        self.mods: list[nn.Module] = []
        self.params: list = []                  # (param, offset, numel, shape)
        self.flat_w = self.w_shard = None       # full weights / this GPU's weight slice (ZeRO-3)
        self.flat_g = self.g_shard = None       # full grads / this GPU's grad slice
        self.master = self.m = self.v = None    # fp32 optimizer state (full or slice)


class Engine:
    """Runs one data-parallel training step on one virtual GPU.

    The model is cut into units. Forward runs unit by unit; each unit's input is detached
    so its backward can be run on its own, in reverse order. That lets a stage do work
    *between* units (gather, release, reduce-scatter), which is where ZeRO lives.
    Subclasses only override the hooks and the optimizer step.
    """
    stage = -1
    label = "?"

    def __init__(self, rank, gpu: VirtualGPU, comm: RankComm, cfg: GPTConfig,
                 specs: list[UnitSpec], init: dict, precision="bf16", checkpoint=False,
                 opt: AdamWConfig | None = None):
        self.rank, self.gpu, self.comm, self.cfg = rank, gpu, comm, cfg
        self.N = comm.world
        self.mixed = precision == "bf16"
        self.w_dtype = torch.bfloat16 if self.mixed else torch.float32
        self.checkpoint = checkpoint
        self.opt = opt or AdamWConfig()
        self.t = 0                                   # optimizer step count
        self.count_flops = False
        self.probe = None                            # optional callback(label) at step milestones
        self._ckpt_ptrs: set = set()                 # storages booked as activation checkpoints
        self.flops = defaultdict(int)
        self.seconds = defaultdict(float)
        self.units = [_Unit(s, i, i == len(specs) - 1, rank, self.N) for i, s in enumerate(specs)]
        self._init = {s.name: torch.cat([init[m] for m in s.modules] +
                                        [torch.zeros(s.padded - s.numel)]) for s in specs}
        self.setup()
        del self._init

    # ---- helpers the stages use (each one keeps the ledger in step with reality) ------------
    def _new_full_weights(self, u: _Unit, category="weights"):
        """Allocate the unit's full flat weight buffer and make the modules' parameters
        views into it."""
        u.flat_w = self.gpu.tensor(f"w:{u.name}", u.spec.padded, self.w_dtype, category)
        u.flat_w.copy_(self._init[u.name])
        with torch.device("meta"):
            u.mods = [build_module(self.cfg, m) for m in u.spec.modules]
        off = 0
        for mod in u.mods:
            for qual, p in list(mod.named_parameters()):
                owner, _, pname = qual.rpartition(".")
                sub = mod.get_submodule(owner) if owner else mod
                n = p.numel()
                param = nn.Parameter(u.flat_w[off:off + n].view(p.shape))
                sub._parameters[pname] = param
                u.params.append((param, off, n, p.shape))
                off += n
            mod.register_forward_pre_hook(self._guard(u))

    def _guard(self, u):
        def hook(mod, args):
            if u.flat_w.untyped_storage().nbytes() == 0:
                raise RuntimeError(f"vGPU {self.rank}: unit '{u.name}' was released; "
                                   f"all-gather it before running it")
        return hook

    def _new_optimizer(self, u: _Unit, sharded: bool):
        n = u.hi - u.lo if sharded else u.spec.padded
        src = self._init[u.name][u.lo:u.hi] if sharded else self._init[u.name]
        if self.mixed:
            u.master = self.gpu.tensor(f"master:{u.name}", n, torch.float32, "master")
            u.master.copy_(src)
        u.m = self.gpu.tensor(f"adam_m:{u.name}", n, torch.float32, "adam_m")
        u.v = self.gpu.tensor(f"adam_v:{u.name}", n, torch.float32, "adam_v")

    def _new_full_grads(self, u: _Unit, category: str):
        """Allocate a zeroed full gradient buffer and point every param.grad into it, so
        autograd accumulates straight into the flat buffer (0 + g is exact)."""
        u.flat_g = self.gpu.tensor(f"g:{u.name}", u.spec.padded, self.w_dtype, category)
        for param, off, n, shape in u.params:
            param.grad = u.flat_g[off:off + n].view(shape)

    def _drop_full_grads(self, u: _Unit):
        for param, *_ in u.params:
            param.grad = None
        u.flat_g = None
        self.gpu.free(f"g:{u.name}")

    def _reduce_scatter_grads(self, u: _Unit):
        """Sum the unit's gradient over all GPUs; keep only my 1/N slice (averaged)."""
        n = u.hi - u.lo
        if u.g_shard is None:
            u.g_shard = self.gpu.tensor(f"g_shard:{u.name}", n, self.w_dtype, "grads")
            self.comm.reduce_scatter(u.flat_g, u.g_shard, tag="grad_sync")
        else:                                   # gradient accumulation: add this micro-batch
            tmp = self.gpu.tensor(f"tmp:rs:{u.name}", n, self.w_dtype, "temp", zero=False)
            self.comm.reduce_scatter(u.flat_g, tmp, tag="grad_sync")
            u.g_shard.add_(tmp)
            self.gpu.free(f"tmp:rs:{u.name}")

    def _adamw(self, u: _Unit, p32: torch.Tensor, grad: torch.Tensor, lr: float):
        """AdamW on flat fp32 tensors, using only plain elementwise ops (mul_, add_, div_,
        sqrt_). Every element is updated independently, so updating a 1/N slice gives
        bit-for-bit the same numbers as updating the whole vector."""
        b1, b2 = self.opt.betas
        n = p32.numel()
        g32 = self.gpu.tensor(f"tmp:g32:{u.name}", n, torch.float32, "temp", zero=False)
        work = self.gpu.tensor(f"tmp:adam:{u.name}", n, torch.float32, "temp", zero=False)
        g32.copy_(grad)
        u.m.mul_(b1)
        torch.mul(g32, 1 - b1, out=work)
        u.m.add_(work)
        u.v.mul_(b2)
        torch.mul(g32, g32, out=work)
        work.mul_(1 - b2)
        u.v.add_(work)
        torch.div(u.v, 1 - b2 ** self.t, out=work)      # denominator: sqrt(v_hat) + eps
        work.sqrt_()
        work.add_(self.opt.eps)
        torch.div(u.m, 1 - b1 ** self.t, out=g32)       # m_hat / denominator * lr
        g32.div_(work)
        g32.mul_(lr)
        p32.mul_(1 - lr * self.opt.weight_decay)        # decoupled weight decay
        p32.sub_(g32)
        self.gpu.free(f"tmp:g32:{u.name}")
        self.gpu.free(f"tmp:adam:{u.name}")

    def _booked_elsewhere(self, ptr: int) -> bool:
        """Storages the activation hook must skip: weights (booked as weights) and
        activation checkpoints (booked explicitly when checkpointing)."""
        return ptr in self._ckpt_ptrs or any(
            u.flat_w is not None and u.flat_w.untyped_storage().data_ptr() == ptr
            for u in self.units)

    def _probe(self, label: str):
        if self.probe is not None:
            self.probe(label)

    @contextmanager
    def _counting(self, phase: str):
        t0 = time.perf_counter()
        if self.count_flops:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                yield
            self.flops[phase] += fc.get_total_flops()
        else:
            yield
        self.seconds[phase] += time.perf_counter() - t0

    def _run_unit(self, u: _Unit, x, y, G):
        h = x
        for mod in u.mods:
            h = mod(h)
        if u.is_last:
            h = F.cross_entropy(h.float().view(-1, h.shape[-1]), y.view(-1)) / G
        return h

    def _check_grads(self, u: _Unit):
        if u.flat_g is None:
            return
        for param, off, n, shape in u.params:
            assert param.grad is not None and param.grad.data_ptr() == u.flat_g[off:].data_ptr(), \
                f"autograd re-allocated the grad of a param in '{u.name}'"

    # ---- one training step ----------------------------------------------------------------
    def train_step(self, micro_batches, lr) -> dict:
        gpu, G = self.gpu, len(micro_batches)
        gpu.reset_peak()
        self._probe("step_start")
        before = self.comm.stats.copy()
        loss_sum = 0.0
        for i, (x, y) in enumerate(micro_batches):
            loss_sum += self._forward_backward(x, y, G, first=(i == 0), last=(i == G - 1))
        gpu.phase = "end_of_backward"
        gpu.mark("end_of_backward")          # the moment the ZeRO formulas describe
        self._probe("end_of_backward")
        self.end_of_backward()
        gpu.phase = "step"
        self.t += 1
        with self._counting("step"):
            self.step(lr)
        gpu.phase = "idle"
        loss = torch.tensor([loss_sum], dtype=torch.float32, device=gpu.device)
        self.comm.all_reduce(loss, tag="loss")          # average over GPUs, for logging only
        return dict(loss=loss.item(), peak=gpu.peak, peak_by_cat=dict(gpu.peak_by_cat),
                    cat_peak=dict(gpu.cat_peak),
                    end_of_backward=dict(gpu.marks["end_of_backward"]),
                    comm=self.comm.stats.since(before),
                    activations_left=gpu.by_cat["activations"])

    def _forward_backward(self, x, y, G, first, last) -> float:
        acts = self.gpu.saved_tensor_hooks(self._booked_elsewhere)
        saved = []
        h = x
        for u in self.units:                                     # ---- forward
            self.gpu.phase = f"fwd:{u.name}"
            self.before_forward(u)
            x_in = h.detach().requires_grad_(True) if h.is_floating_point() else h
            if self.checkpoint and x_in.is_floating_point():    # the one tensor a checkpointed
                st = x_in.untyped_storage()                     # unit keeps: its input
                self.gpu.alloc(f"ckpt:{u.name}", st.nbytes(), "activations")
                self._ckpt_ptrs.add(st.data_ptr())
            grad_ctx = torch.no_grad() if self.checkpoint else nullcontext()
            with acts, grad_ctx, self._counting("forward"):
                out = self._run_unit(u, x_in, y, G)
            saved.append((x_in, out))
            self.after_forward(u)
            h = out
        loss_val = float(h.detach())
        self._probe("end_of_forward")
        if first:
            self.begin_backward()
        grad = None
        for k in reversed(range(len(self.units))):             # ---- backward
            u = self.units[k]
            x_in, out = saved[k]
            saved[k] = None
            self.gpu.phase = f"bwd:{u.name}"
            self.before_backward(u)
            if self.checkpoint:                                 # recompute the unit's forward
                with acts, self._counting("recompute"):
                    out = self._run_unit(u, x_in, y, G)
            with self._counting("backward"):
                torch.autograd.backward(out, grad)
            self._check_grads(u)
            grad = x_in.grad if x_in.is_floating_point() else None
            if self.checkpoint and x_in.is_floating_point():
                self._ckpt_ptrs.discard(x_in.untyped_storage().data_ptr())
                self.gpu.free(f"ckpt:{u.name}")
            del out, x_in
            self.after_backward(u, last)
        return loss_val

    # ---- the hooks: every stage is defined by what it does here ---------------------------
    def setup(self): raise NotImplementedError
    def before_forward(self, u): pass
    def after_forward(self, u): pass
    def begin_backward(self): pass
    def before_backward(self, u): pass
    def after_backward(self, u, last_micro): pass
    def end_of_backward(self): pass
    def step(self, lr): raise NotImplementedError

    # ---- checkpointing: each GPU saves what it owns ---------------------------------------
    def owned_state(self) -> dict:
        """The fp32 weights this GPU owns: {unit: (lo, hi, tensor)}. DDP owns everything;
        ZeRO GPUs own a 1/N slice. Consolidating = concatenating slices in rank order."""
        out = {}
        for u in self.units:
            if self.mixed:
                t = u.master                                  # full (DDP) or slice (ZeRO)
            elif self.stage == 0:
                t = u.flat_w
            elif self.stage == 3:
                t = u.w_shard
            else:
                t = u.flat_w[u.lo:u.hi]
            lo, hi = (0, u.spec.padded) if self.stage == 0 else (u.lo, u.hi)
            out[u.name] = (lo, hi, t.detach().float().cpu().clone())
        return out


class DDP(Engine):
    """ZeRO stage 0. Every GPU holds all weights, all gradients and the whole optimizer.
    Gradients are all-reduced unit by unit as backward produces them."""
    stage, label = 0, "DDP (ZeRO-0)"

    def setup(self):
        for u in self.units:
            self._new_full_weights(u)                        # 2Ψ weights
            self._new_optimizer(u, sharded=False)            # 12Ψ master + m + v

    def begin_backward(self):
        for u in self.units:
            self._new_full_grads(u, "grads")                 # 2Ψ grads

    def after_backward(self, u, last_micro):
        if last_micro:
            self.comm.all_reduce(u.flat_g, tag="grad_sync")  # 2Ψ of traffic in total

    def step(self, lr):
        for u in self.units:
            p32 = u.master if self.mixed else u.flat_w
            self._adamw(u, p32, u.flat_g, lr)                # Ψ elements updated per GPU
            if self.mixed:
                u.flat_w.copy_(u.master)
            self._drop_full_grads(u)


class ZeRO1(Engine):
    """Stage 1: shard the optimizer states. Each GPU keeps all weights and (during
    backward) all gradients, but only 1/N of the fp32 master + Adam m, v. After backward,
    gradients are reduce-scattered (each GPU receives the slice it owns), the GPU updates
    its slice, and an all-gather puts the updated weights back on every GPU."""
    stage, label = 1, "ZeRO-1"

    def setup(self):
        for u in self.units:
            self._new_full_weights(u)                        # 2Ψ weights
            self._new_optimizer(u, sharded=True)             # 12Ψ/N

    def begin_backward(self):
        for u in self.units:
            self._new_full_grads(u, "grads")                 # 2Ψ grads (full, for now)

    def end_of_backward(self):
        for u in self.units:
            self._reduce_scatter_grads(u)                    # Ψ of traffic
            self._drop_full_grads(u)

    def step(self, lr):
        for u in self.units:
            my_w = u.flat_w[u.lo:u.hi]
            p32 = u.master if self.mixed else my_w
            self._adamw(u, p32, u.g_shard, lr)               # Ψ/N elements updated per GPU
            if self.mixed:
                my_w.copy_(u.master)
            u.g_shard = None
            self.gpu.free(f"g_shard:{u.name}")
            self.comm.all_gather(my_w, u.flat_w, tag="param_gather")   # Ψ of traffic


class ZeRO2(ZeRO1):
    """Stage 2: also shard the gradients. As soon as a unit's backward finishes, its
    gradient is reduce-scattered and the full-size buffer is freed, so a GPU never holds
    more than one unit's full gradient at a time."""
    stage, label = 2, "ZeRO-2"

    def begin_backward(self):
        pass                                                 # no full-size gradients up front

    def before_backward(self, u):
        self._new_full_grads(u, "temp")                      # one unit, briefly

    def after_backward(self, u, last_micro):
        self._reduce_scatter_grads(u)                        # every micro-batch
        self._drop_full_grads(u)

    def end_of_backward(self):
        pass


class ZeRO3(Engine):
    """Stage 3: also shard the weights. A unit's full weights exist only while it runs:
    all-gathered just before its forward, freed right after, and gathered again for its
    backward. Freeing means shrinking the storage to 0 bytes; autograd's saved references
    point at that same storage, so re-gathering brings them back to life."""
    stage, label = 3, "ZeRO-3"

    def setup(self):
        for u in self.units:
            self._new_full_weights(u, category="temp")       # bind param views, then...
            u.w_shard = self.gpu.tensor(f"w_shard:{u.name}", u.hi - u.lo, self.w_dtype, "weights")
            u.w_shard.copy_(u.flat_w[u.lo:u.hi])             # 2Ψ/N weights
            self._release(u)                                 # ...drop the full copy
            self._new_optimizer(u, sharded=True)             # 12Ψ/N

    def _gather(self, u):
        self.gpu.alloc(f"w:{u.name}", u.flat_w.dtype.itemsize * u.spec.padded, "temp")
        u.flat_w.untyped_storage().resize_(u.flat_w.dtype.itemsize * u.spec.padded)
        self.comm.all_gather(u.w_shard, u.flat_w, tag="param_gather")

    def _release(self, u):
        u.flat_w.untyped_storage().resize_(0)
        self.gpu.free(f"w:{u.name}")

    def before_forward(self, u):
        self._gather(u)                                      # Ψ of traffic over the forward

    def after_forward(self, u):
        self._release(u)

    def before_backward(self, u):
        self._gather(u)                                      # Ψ again over the backward
        self._new_full_grads(u, "temp")

    def after_backward(self, u, last_micro):
        self._reduce_scatter_grads(u)                        # Ψ: the third one
        self._drop_full_grads(u)
        self._release(u)

    def step(self, lr):
        for u in self.units:
            p32 = u.master if self.mixed else u.w_shard
            self._adamw(u, p32, u.g_shard, lr)
            if self.mixed:
                u.w_shard.copy_(u.master)                    # weights stay sharded
            u.g_shard = None
            self.gpu.free(f"g_shard:{u.name}")


STAGES = {0: DDP, 1: ZeRO1, 2: ZeRO2, 3: ZeRO3}


def expected_model_state_bytes(stage: int, P: int, N: int, precision="bf16") -> int:
    """The ZeRO paper's formulas for the model states one GPU holds, at the end of
    backward (the moment every stage holds its largest set of model states)."""
    bw = bg = 2 if precision == "bf16" else 4
    K = 12 if precision == "bf16" else 8
    return {0: (bw + bg + K) * P,
            1: (bw + bg) * P + K * P // N,
            2: bw * P + (bg + K) * P // N,
            3: (bw + bg + K) * P // N}[stage]


def estimate_peak_bytes(stage, specs, N, act_bytes, precision="bf16") -> int:
    """An upper bound on one GPU's peak *ledger* memory: model states (the formula) +
    activations + the largest temporary buffer this implementation books. Activations do not
    depend on the ZeRO stage, so act_bytes is measured once on a single device. (Unbooked
    scratch - a collective's fp32 slice accumulator, autograd's transient input gradients -
    is outside the ledger and so outside this bound.)"""
    P = sum(s.padded for s in specs)
    u = max(s.padded for s in specs)
    bw = 2 if precision == "bf16" else 4
    states = expected_model_state_bytes(stage, P, N, precision)
    step_tmp = 8 * (u if stage == 0 else u // N)            # fp32 grad copy + Adam scratch
    transient = {0: 0, 1: 0, 2: bw * u, 3: 2 * bw * u}[stage]   # full grad (Z2), + weights (Z3)
    return states + max(act_bytes + transient, step_tmp)


# =============================================================================================
# 7. The driver
# =============================================================================================

class CharData:
    """Character-level text; batches are a pure function of (seed, step), so every stage,
    and every world size with the same global batch (world x micro_bsz), sees exactly the
    same sequences."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.uint8)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def global_starts(self, step, n_seqs, grad_accum, T, seed):
        g = torch.Generator().manual_seed(seed * 1_000_003 + step)
        return torch.randint(0, len(self.data) - T - 1, (grad_accum, n_seqs), generator=g)

    def micro_batches(self, step, rank, world, micro_bsz, grad_accum, T, seed, device="cpu"):
        starts = self.global_starts(step, world * micro_bsz, grad_accum, T, seed)
        out = []
        for gi in range(grad_accum):
            mine = starts[gi, rank * micro_bsz:(rank + 1) * micro_bsz]
            x = torch.stack([self.data[s:s + T] for s in mine.tolist()]).long()
            y = torch.stack([self.data[s + 1:s + T + 1] for s in mine.tolist()]).long()
            out.append((x.to(device), y.to(device)))
        return out


def lr_at(step, total, base, warmup, min_frac=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


@dataclass
class RunConfig:
    stage: int
    world: int = 32
    steps: int = 3
    micro_bsz: int = 1
    grad_accum: int = 1
    precision: str = "bf16"
    checkpoint: bool = False
    wrap: str = "block"
    capacity: int | None = None
    device: str = "cpu"
    seed: int = 1337
    lr: float = 3e-3
    warmup: int = 5
    flops_step: int | None = None          # run FlopCounterMode on this step (all GPUs)
    timeline_step: int | None = None       # record the ledger timeline on this step
    timeline_ranks: tuple = (0,)
    keep_state: bool = True


@dataclass
class RunResult:
    rc: RunConfig
    specs: list
    losses: list                          # global loss per step
    records: list                         # records[step][rank] -> dict from train_step
    flops: list                           # per rank: {phase: flops} on flops_step
    seconds: list                         # per rank: {phase: seconds} over the run
    timelines: dict                       # rank -> list of ledger events
    setup_by_cat: list                    # per rank: ledger right after setup
    states: list | None                   # per rank: owned_state() at the end
    opt_elements: list                    # per rank: number of optimizer elements
    comm_seconds: list

    @property
    def P(self) -> int:
        return sum(s.padded for s in self.specs)

    def consolidated(self) -> dict[str, torch.Tensor]:
        """Stitch every GPU's owned slices back into full fp32 unit vectors (what
        DeepSpeed's zero_to_fp32.py does with the per-rank checkpoint files)."""
        full = {}
        for s in self.specs:
            buf = torch.empty(s.padded)
            for st in self.states:
                if st is None:
                    continue
                lo, hi, t = st[s.name]
                buf[lo:hi] = t
            full[s.name] = buf
        return full

    def module_flats(self, cfg: GPTConfig) -> dict[str, torch.Tensor]:
        """Split the consolidated units back into per-module vectors (drops padding)."""
        out = {}
        for s, (name, buf) in zip(self.specs, self.consolidated().items()):
            off = 0
            for m in s.modules:
                with torch.device("meta"):
                    n = sum(p.numel() for p in build_module(cfg, m).parameters())
                out[m] = buf[off:off + n].clone()
                off += n
        return out


def run_training(rc: RunConfig, cfg: GPTConfig, data: CharData, init: dict | None = None) -> RunResult:
    """Train `rc.steps` steps of `rc.stage` on `rc.world` virtual GPUs. Returns everything
    the notebook plots and the tests assert on."""
    init = init if init is not None else init_weights(cfg, rc.seed)
    specs = make_units(cfg, rc.world, rc.wrap)
    cluster = VirtualCluster(rc.world, rc.capacity, rc.device)
    T = cfg.block_size

    def worker(rank, gpu: VirtualGPU, comm: RankComm):
        eng = STAGES[rc.stage](rank, gpu, comm, cfg, specs, init, rc.precision, rc.checkpoint)
        setup_by_cat = dict(gpu.by_cat)
        records = []
        for step in range(rc.steps):
            eng.count_flops = (step == rc.flops_step)
            if step == rc.timeline_step and rank in rc.timeline_ranks:
                gpu.timeline = []
            mb = data.micro_batches(step, rank, rc.world, rc.micro_bsz, rc.grad_accum, T,
                                    rc.seed, rc.device)
            rec = eng.train_step(mb, lr_at(step, rc.steps, rc.lr, rc.warmup))
            rec["timeline"] = gpu.timeline
            gpu.timeline = None
            records.append(rec)
        opt_el = sum(u.m.numel() for u in eng.units)
        state = eng.owned_state() if rc.keep_state else None
        return records, dict(eng.flops), dict(eng.seconds), setup_by_cat, state, opt_el, \
            comm.stats.seconds

    out = cluster.run(worker)
    del cluster
    gc.collect()      # engines hold reference cycles (module hooks); free their memory now
    records = [[out[r][0][s] for r in range(rc.world)] for s in range(rc.steps)]
    timelines = {}
    for s in range(rc.steps):
        for r in range(rc.world):
            tl = records[s][r].pop("timeline")
            if tl is not None:
                timelines[r] = tl
    states = [o[4] for o in out]
    if rc.keep_state and rc.stage == 0:            # DDP replicas must never drift apart
        for st in states[1:]:
            for k, (lo, hi, t) in st.items():
                assert torch.equal(t, states[0][k][2]), "DDP replicas diverged"
        states = [states[0]] + [None] * (rc.world - 1)
    return RunResult(rc, specs, [records[s][0]["loss"] for s in range(rc.steps)], records,
                     [o[1] for o in out], [o[2] for o in out], timelines,
                     [o[3] for o in out], states if rc.keep_state else None,
                     [o[5] for o in out], [o[6] for o in out])
