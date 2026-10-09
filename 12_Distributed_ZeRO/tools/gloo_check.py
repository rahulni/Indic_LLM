"""gloo_check: the same ZeRO-1 engine on real torch.distributed (gloo), next to the simulator.

zero_sim's collectives are home-made (threads, barriers, shared slots). This tool swaps them
for real ones. It launches `world` ordinary Python processes, joins them into a gloo process
group, and runs the *unchanged* zero_sim.ZeRO1 engine in each one. The engine talks through
TorchDistComm, an adapter with exactly the RankComm interface. Then the thread simulator runs
in this process with the same model, data, seed and learning rate, and the two are compared:
losses, consolidated fp32 weights, and the per-rank communication counters.

    python tools/gloo_check.py --world 2            # prints the result as JSON
    from tools.gloo_check import run_gloo_check     # from the notebook (folder on sys.path)

Bit-exact agreement is not promised: gloo adds the N contributions in its own order, the
simulator in rank order, and fp32 addition is not associative. At N = 2 there is only one
addition per element, so the two usually agree to the bit anyway.

Windows notes: plain subprocesses (not mp.spawn, which needs an importable __main__ and
misbehaves in Jupyter); a FileStore rendezvous (no TCPStore, so no libuv question); every
worker hides CUDA before importing torch and runs on one CPU thread.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "input.txt"

# The model the check trains, and the run settings. The run settings are zero_sim.RunConfig's
# defaults; they are passed explicitly to both the workers and the simulator, so the two sides
# cannot drift apart even if the defaults change.
DEFAULT_CFG = dict(vocab_size=65, n_embd=64, n_layer=2, n_head=2, block_size=32)
RUN = dict(seed=1337, lr=3e-3, warmup=5, micro_bsz=1, grad_accum=1)
RTOL, ATOL = 1e-5, 1e-6

# torch is imported lazily: a worker must hide CUDA before torch loads, and the launcher keeps
# its own memory small while the workers run.
torch = dist = zero_sim = None


def _load():
    global torch, dist, zero_sim
    if zero_sim is None:
        import torch as _torch
        import torch.distributed as _dist
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import zero_sim as _zero_sim
        torch, dist, zero_sim = _torch, _dist, _zero_sim
    return torch, dist, zero_sim


# =============================================================================================
# TorchDistComm: RankComm's interface on a real process group
# =============================================================================================

_TAG_SIZE, _TAG_DATA = 7001, 7002       # p2p message tags: a length header, then the payload
_UNSUPPORTED = (RuntimeError, NotImplementedError, AttributeError, TypeError, ValueError)


class TorchDistComm:
    """The communicator as one process sees it, on torch.distributed.

    Same methods, same op names and the same ring cost model as zero_sim.ThreadComm, so a
    CommStats from here can be compared key by key with one from the simulator.

    Reductions are done as fp32 SUM, then divide by N (gloo has no AVG, and its bf16
    support is unreliable), the same recipe ThreadComm uses. For each collective the first
    primitive in a fallback chain that works is used and remembered in `self.primitives`;
    the ones that failed are in `self.fallbacks` with their error. A primitive that is not
    supported fails on every rank before any data moves, so all ranks fall back together.
    """

    def __init__(self, group=None):
        _load()
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.stats = zero_sim.CommStats()
        self.primitives: dict[str, str] = {}
        self.fallbacks: dict[str, str] = {}
        self._pending: list = []            # isend work handles, kept with their buffers

    # -- bookkeeping ------------------------------------------------------------------------
    def _charge(self, tag, op, nbytes, steps, t0) -> None:
        st = self.stats
        st.bytes[(tag, op)] += int(nbytes)
        st.calls[(tag, op)] += 1
        st.ring_steps += steps
        st.seconds += time.perf_counter() - t0

    def _run(self, op: str, chain: list) -> None:
        """Run the first primitive of `chain` that works; later calls go straight to it."""
        chosen = self.primitives.get(op)
        if chosen is not None:
            dict(chain)[chosen]()
            return
        for name, fn in chain:
            try:
                fn()
            except _UNSUPPORTED as e:
                msg = str(e).strip().splitlines()
                self.fallbacks[name] = f"{type(e).__name__}: {msg[0][:200] if msg else ''}"
                continue
            self.primitives[op] = name
            return
        raise RuntimeError(f"rank {self.rank}: no working primitive for {op}: {self.fallbacks}")

    @staticmethod
    def _wire(t):
        """A contiguous 1-D copy to put on the wire: half-precision floats travel as fp32
        (exact both ways), everything else as itself."""
        wire = torch.float32 if t.dtype in (torch.bfloat16, torch.float16) else t.dtype
        return t.detach().reshape(-1).to(dtype=wire, copy=True).contiguous()

    # -- the collectives ----------------------------------------------------------------------
    def reduce_scatter(self, full, out, tag, average=True):
        t0 = time.perf_counter()
        N, n = self.world, full.numel()
        assert n % N == 0, f"reduce_scatter needs numel % world == 0 (got {n} % {N})"
        s = n // N
        src = full.detach().reshape(-1).to(torch.float32).contiguous()
        res = torch.empty(s, dtype=torch.float32, device=src.device)
        SUM = dist.ReduceOp.SUM

        def rs_tensor():
            dist.reduce_scatter_tensor(res, src, op=SUM, group=self.group)

        def rs_list():
            dist.reduce_scatter(res, list(src.chunk(N)), op=SUM, group=self.group)

        def ar_slice():
            buf = src.clone()
            dist.all_reduce(buf, op=SUM, group=self.group)
            res.copy_(buf[self.rank * s:(self.rank + 1) * s])

        self._run("reduce_scatter", [("reduce_scatter_tensor", rs_tensor),
                                     ("reduce_scatter", rs_list),
                                     ("all_reduce+slice", ar_slice)])
        if average:
            res.div_(N)
        out.copy_(res.view_as(out))
        self._charge(tag, "reduce_scatter", (N - 1) / N * full.nbytes, N - 1, t0)
        return out

    def all_gather(self, shard, out, tag):
        """`shard` may be a view of this rank's own slice of `out`: it is copied first."""
        t0 = time.perf_counter()
        N, s = self.world, shard.numel()
        assert out.numel() == N * s
        mine = self._wire(shard)
        direct = out.dtype == mine.dtype and out.is_contiguous()
        flat = out.view(-1) if direct else torch.empty(N * s, dtype=mine.dtype, device=mine.device)

        def ag_tensor():
            dist.all_gather_into_tensor(flat, mine, group=self.group)

        def ag_list():
            dist.all_gather(list(flat.chunk(N)), mine, group=self.group)

        self._run("all_gather", [("all_gather_into_tensor", ag_tensor),
                                 ("all_gather", ag_list)])
        if not direct:
            out.copy_(flat.view(out.shape))
        self._charge(tag, "all_gather", (N - 1) * shard.nbytes, N - 1, t0)
        return out

    def all_reduce(self, t, tag, average=True):
        t0 = time.perf_counter()
        N = self.world
        buf = t.detach().reshape(-1).to(dtype=torch.float32, copy=True).contiguous()
        self._run("all_reduce", [("all_reduce",
                                  lambda: dist.all_reduce(buf, op=dist.ReduceOp.SUM,
                                                          group=self.group))])
        if average:
            buf.div_(N)
        t.copy_(buf.view(t.shape))
        self._charge(tag, "all_reduce", 2 * (N - 1) / N * t.nbytes, 2 * (N - 1), t0)
        return t

    def broadcast(self, t, src, tag):
        t0 = time.perf_counter()
        buf = self._wire(t)
        self._run("broadcast", [("broadcast",
                                 lambda: dist.broadcast(buf, src, group=self.group))])
        t.copy_(buf.view(t.shape))
        self._charge(tag, "broadcast", (self.world - 1) / self.world * t.nbytes,
                     self.world - 1, t0)
        return t

    def barrier(self):
        dist.barrier(group=self.group)

    # -- point to point (so zero_sim.ring_all_reduce runs unchanged over real sockets) --------
    def send(self, dst, payload, nbytes, tag="p2p"):
        """Non-blocking, like ThreadComm.send: any picklable object; `nbytes` is what the
        ring model says goes on the wire (the pickle framing is not charged)."""
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        data = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        size = torch.tensor([data.numel()], dtype=torch.int64)
        works = [dist.isend(size, dst, group=self.group, tag=_TAG_SIZE),
                 dist.isend(data, dst, group=self.group, tag=_TAG_DATA)]
        self._pending.append((works, size, data))
        self.stats.bytes[(tag, "send")] += int(nbytes)
        self.stats.calls[(tag, "send")] += 1

    def recv(self):
        """Receive the next message from any rank: returns (src, payload)."""
        size = torch.zeros(1, dtype=torch.int64)
        src = dist.recv(size, src=None, group=self.group, tag=_TAG_SIZE)
        data = torch.empty(int(size.item()), dtype=torch.uint8)
        dist.recv(data, src=src, group=self.group, tag=_TAG_DATA)
        self._pending = [p for p in self._pending if not all(w.is_completed() for w in p[0])]
        return src, pickle.loads(data.numpy().tobytes())

    def close(self):
        for works, *_ in self._pending:
            for w in works:
                w.wait()
        self._pending.clear()


# =============================================================================================
# Worker: one process, one rank
# =============================================================================================

def _file_url(path) -> str:
    return "file:///" + str(Path(path).resolve()).replace("\\", "/").lstrip("/")


def _stats_json(st) -> dict:
    return dict(bytes={f"{t}:{o}": int(v) for (t, o), v in sorted(st.bytes.items())},
                calls={f"{t}:{o}": int(v) for (t, o), v in sorted(st.calls.items())},
                ring_steps=int(st.ring_steps), seconds=round(float(st.seconds), 4))


def _peak_rss_mb() -> float | None:
    """This process's peak resident memory, for sizing `world` against free RAM."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)
    except Exception:
        return None


def _ring_check(comm: TorchDistComm) -> dict:
    """zero_sim.ring_all_reduce, unchanged, over gloo point-to-point messages."""
    N, r = comm.world, comm.rank
    x = torch.arange(N * 256, dtype=torch.float32).mul_(0.37).add_(r).sin_()
    trace: list = []
    got = zero_sim.ring_all_reduce(comm, x, trace)
    ref = x.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    comm.close()
    return dict(allclose=bool(torch.allclose(got, ref, rtol=RTOL, atol=ATOL)),
                max_abs_diff=float((got - ref).abs().max()),
                bytes_sent=int(comm.stats.bytes[("ring", "send")]),
                bytes_expected=int(2 * (N - 1) / N * x.nbytes),
                final_contributions=trace[-1][2])


def _worker(a) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # before torch is imported
    t_start = time.perf_counter()
    _load()
    torch.set_num_threads(1)
    from datetime import timedelta
    dist.init_process_group("gloo", init_method=_file_url(a.store), rank=a.rank,
                            world_size=a.world, timeout=timedelta(seconds=a.pg_timeout))
    try:
        rank, N, K = a.rank, a.world, a.steps
        cfg = zero_sim.GPTConfig(vocab_size=a.vocab_size, n_embd=a.n_embd, n_layer=a.n_layer,
                                 n_head=a.n_head, block_size=a.block_size)
        init = zero_sim.init_weights(cfg, a.seed)
        specs = zero_sim.make_units(cfg, N)
        gpu = zero_sim.VirtualGPU(rank)
        comm = TorchDistComm()
        eng = zero_sim.ZeRO1(rank, gpu, comm, cfg, specs, init, precision="fp32")
        data = zero_sim.CharData(DATA.read_text(encoding="utf-8"))
        losses, lrs = [], []
        for step in range(K):
            mb = data.micro_batches(step, rank, N, micro_bsz=a.micro_bsz,
                                    grad_accum=a.grad_accum, T=cfg.block_size, seed=a.seed)
            lr = zero_sim.lr_at(step, K, a.lr, a.warmup)
            if rank == a.crash_rank and step == 1:          # failure-path test hook
                raise RuntimeError(f"rank {rank}: injected failure at step {step}")
            rec = eng.train_step(mb, lr)
            losses.append(float(rec["loss"]))
            lrs.append(lr)
        train_stats = comm.stats.copy()
        out = Path(a.out)
        torch.save(eng.owned_state(), out / f"rank{rank}.pt")
        ring = _ring_check(comm) if a.ring else None
        dist.barrier()
        result = dict(rank=rank, world=N, steps=K, losses=losses, lrs=lrs,
                      primitives=comm.primitives, fallbacks=comm.fallbacks,
                      comm=_stats_json(train_stats), ring=ring,
                      ledger_peak_bytes=int(gpu.peak), process_peak_mb=_peak_rss_mb(),
                      torch=torch.__version__, seconds=round(time.perf_counter() - t_start, 2))
        (out / f"rank{rank}.json").write_text(json.dumps(result, indent=1))
    finally:
        dist.destroy_process_group()


# =============================================================================================
# Launcher
# =============================================================================================

def _available_ram() -> float | None:
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(m)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return float(m.ullAvailPhys)
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _tail(path: Path, n: int = 1500) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-n:]
    except Exception as e:
        return f"<no log: {e}>"


def _stop(procs) -> None:
    for p in procs:
        if p.poll() is None:
            p.kill()
    for p in procs:
        try:
            p.wait(10)
        except Exception:
            pass


def _cfg_dict(cfg) -> dict:
    if cfg is None:
        return dict(DEFAULT_CFG)
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return dict(DEFAULT_CFG, **cfg)


def run_gloo_check(world: int | None = None, steps: int = 3, timeout: float = 240,
                   cfg=None, ring: bool = True, keep_files: bool = False,
                   _crash_rank: int | None = None) -> dict:
    """Train ZeRO-1 for `steps` steps on `world` real gloo processes and on the thread
    simulator, and compare. Never hangs: after `timeout` seconds every worker is killed.

    Returns a JSON-able dict. status is 'ok' (allclose), 'mismatch', 'failed', 'timeout'
    or 'skipped' (not enough free RAM; `reason` says why)."""
    t_start = time.perf_counter()
    avail = _available_ram()
    out: dict = dict(status=None, world=world, steps=steps,
                     ram_available_gb=None if avail is None else round(avail / 1e9, 2))
    if avail is not None and avail < 0.8e9:
        return dict(out, status="skipped", seconds=0.0,
                    reason=f"only {avail / 1e9:.2f} GB RAM available; each gloo worker "
                           f"loads its own torch (~0.3-0.5 GB), need >= 0.8 GB")
    if world is None:
        world = 4 if avail is not None and avail >= 2.5e9 else 2
    out["world"] = world
    cfg_d = _cfg_dict(cfg)
    out["config"] = dict(cfg_d, **RUN, precision="fp32", stage=1)

    tmp = Path(tempfile.mkdtemp(prefix="zero_gloo_"))
    store = tmp / "filestore"
    if store.exists():                   # a stale FileStore makes init hang or fail
        store.unlink()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    args = ["--world", str(world), "--store", str(store), "--out", str(tmp),
            "--steps", str(steps), "--pg-timeout", str(int(timeout))]
    for k, v in {**cfg_d, **RUN}.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    if not ring:
        args.append("--no-ring")
    if _crash_rank is not None:
        args += ["--crash-rank", str(_crash_rank)]

    procs, logs = [], []
    status = None
    try:
        for r in range(world):
            log = open(tmp / f"rank{r}.log", "w", encoding="utf-8")
            logs.append(log)
            procs.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", "--rank", str(r),
                 *args], cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, creationflags=flags))
        deadline = time.monotonic() + timeout
        while True:
            codes = [p.poll() for p in procs]
            if all(c is not None for c in codes):
                status = None if all(c == 0 for c in codes) else "failed"
                break
            if any(c not in (None, 0) for c in codes):
                status = "failed"                # the others would wait forever: stop them
                break
            if time.monotonic() > deadline:
                status = "timeout"
                break
            time.sleep(0.2)
    finally:
        _stop(procs)                             # also on KeyboardInterrupt (kernel interrupt)
        for log in logs:
            log.close()

    if status is not None:
        codes = [p.returncode for p in procs]
        bad = [r for r, c in enumerate(codes) if c != 0] or list(range(world))
        return dict(out, status=status, returncodes=codes, workdir=str(tmp),
                    stderr_tail={f"rank{r}": _tail(tmp / f"rank{r}.log") for r in bad[:2]},
                    seconds=round(time.perf_counter() - t_start, 1))

    try:
        out.update(_compare(tmp, world, steps, cfg_d, ring))
    except Exception as e:                       # noqa: BLE001  (reported, never raised)
        out.update(status="failed", error=f"{type(e).__name__}: {e}", workdir=str(tmp))
        keep_files = True
    out["seconds"] = round(time.perf_counter() - t_start, 1)
    if not keep_files:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        out["workdir"] = str(tmp)
    return out


def _compare(tmp: Path, world: int, steps: int, cfg_d: dict, ring: bool) -> dict:
    _load()
    ranks = [json.loads((tmp / f"rank{r}.json").read_text()) for r in range(world)]
    shards = [torch.load(tmp / f"rank{r}.pt", weights_only=True) for r in range(world)]

    # consolidate: concatenate every rank's owned slice, in rank order, per unit
    gloo_full = {}
    for name in shards[0]:
        parts, expect_lo = [], 0
        for st in shards:
            lo, hi, t = st[name]
            assert lo == expect_lo and hi - lo == t.numel(), f"{name}: slices do not tile"
            parts.append(t)
            expect_lo = hi
        gloo_full[name] = torch.cat(parts)

    # the thread simulator, same everything
    cfg = zero_sim.GPTConfig(**cfg_d)
    data = zero_sim.CharData(DATA.read_text(encoding="utf-8"))
    assert data.vocab_size <= cfg.vocab_size
    init = zero_sim.init_weights(cfg, RUN["seed"])
    rc = zero_sim.RunConfig(stage=1, world=world, steps=steps, precision="fp32", **RUN)
    res = zero_sim.run_training(rc, cfg, data, init)
    sim_full = res.consolidated()

    assert list(gloo_full) == list(sim_full), (list(gloo_full), list(sim_full))
    w_diff = max(float((gloo_full[k] - sim_full[k]).abs().max()) for k in sim_full)
    w_close = all(torch.allclose(gloo_full[k], sim_full[k], rtol=RTOL, atol=ATOL)
                  for k in sim_full)
    w_exact = all(torch.equal(gloo_full[k], sim_full[k]) for k in sim_full)
    lg = torch.tensor(ranks[0]["losses"], dtype=torch.float64)
    ls = torch.tensor(res.losses, dtype=torch.float64)
    l_diff = float((lg - ls).abs().max())
    l_close = bool(torch.allclose(lg, ls, rtol=RTOL, atol=ATOL))
    same_loss_all_ranks = all(r["losses"] == ranks[0]["losses"] for r in ranks)

    sim_bytes = []
    for r in range(world):
        tot: dict = {}
        for s in range(steps):
            for (tag, op), v in res.records[s][r]["comm"].bytes.items():
                tot[f"{tag}:{op}"] = tot.get(f"{tag}:{op}", 0) + int(v)
        sim_bytes.append(dict(sorted(tot.items())))
    gloo_bytes = [r["comm"]["bytes"] for r in ranks]

    prims = ranks[0]["primitives"]
    agree = all(r["primitives"] == prims for r in ranks)
    ok = w_close and l_close and same_loss_all_ranks and agree and gloo_bytes == sim_bytes
    result = dict(
        status="ok" if ok else "mismatch",
        primitives=prims if agree else [r["primitives"] for r in ranks],
        fallbacks=ranks[0]["fallbacks"],
        losses_gloo=ranks[0]["losses"],
        losses_sim=[float(x) for x in res.losses],
        max_abs_weight_diff=w_diff,
        max_abs_loss_diff=l_diff,
        allclose=bool(w_close and l_close),
        bit_exact_weights=bool(w_exact),
        comm_bytes_per_rank=dict(gloo=gloo_bytes, sim=sim_bytes, equal=gloo_bytes == sim_bytes),
        worker_seconds=[r["seconds"] for r in ranks],
        worker_peak_mb=[r["process_peak_mb"] for r in ranks],
        torch=ranks[0]["torch"],
    )
    if ring:
        result["ring_all_reduce_on_gloo"] = dict(
            allclose=all(r["ring"]["allclose"] for r in ranks),
            max_abs_diff=max(r["ring"]["max_abs_diff"] for r in ranks),
            bytes_sent_per_rank=[r["ring"]["bytes_sent"] for r in ranks],
            bytes_expected=ranks[0]["ring"]["bytes_expected"],
            final_contributions_rank0=ranks[0]["ring"]["final_contributions"])
    return result


# =============================================================================================
# CLI
# =============================================================================================

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--worker" in argv:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--world", type=int, default=None, help="default: 4 if RAM allows, else 2")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--timeout", type=float, default=240)
    p.add_argument("--no-ring", dest="ring", action="store_false",
                   help="skip running zero_sim.ring_all_reduce over gloo p2p")
    p.add_argument("--keep-files", action="store_true")
    # worker mode (used by the launcher)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    p.add_argument("--store", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.add_argument("--pg-timeout", type=int, default=240, help=argparse.SUPPRESS)
    p.add_argument("--crash-rank", type=int, default=-1, help=argparse.SUPPRESS)
    for k, v in {**DEFAULT_CFG, **RUN}.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v,
                       help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.worker:
        _worker(a)
        return 0
    cfg = {k: getattr(a, k) for k in DEFAULT_CFG}
    res = run_gloo_check(a.world, a.steps, a.timeout, cfg=cfg, ring=a.ring,
                         keep_files=a.keep_files)
    print(json.dumps(res, indent=2))
    return 0 if res["status"] in ("ok", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
