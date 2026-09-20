"""Tests for zero_sim's communication layer, ring all-reduce, memory ledger and cluster runner.

    python -m unittest tests.test_comm -v          (from the folder root)

Every collective is checked against a plain torch reference at N = 3, 4 and 32, and every
byte counter against the ring formulas: reduce-scatter and all-gather send (N-1)/N * S bytes
per GPU, all-reduce 2(N-1)/N * S, where S is the size of the full tensor.
"""
import copy
import os
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                   # noqa: E402
import torch.nn as nn                          # noqa: E402

import zero_sim as zs                          # noqa: E402
from zero_sim import MiB, VirtualCluster, VirtualGPU, VirtualOOMError, ring_all_reduce  # noqa: E402

WORLDS = (3, 4, 32)
TIMEOUT = 30.0      # barrier timeout: a broken test fails in seconds instead of hanging for 2 min


def rank_inputs(N, numel, seed, dtype=torch.float32):
    """Rank r's input: reproducible random numbers, different on every rank."""
    return [torch.randn(numel, generator=torch.Generator().manual_seed(1000 * seed + r)).to(dtype)
            for r in range(N)]


def rank_order_sum(xs):
    """The simulator's documented reduction: fp32, in rank order."""
    acc = xs[0].float().clone()
    for x in xs[1:]:
        acc.add_(x.float())
    return acc


def run(N, fn, **kw):
    cluster = VirtualCluster(N, timeout=TIMEOUT, **kw)
    return cluster.run(fn), cluster


class TestCollectives(unittest.TestCase):
    """reduce_scatter / all_gather / all_reduce / broadcast against plain torch."""

    def assertClose(self, a, b, msg=None):
        self.assertTrue(torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-6),
                        msg or f"max abs diff {(a.float() - b.float()).abs().max():.3g}")

    def assertEqualT(self, a, b, msg=None):
        self.assertTrue(torch.equal(a, b), msg or f"max abs diff {(a - b).abs().max():.3g}")

    def test_reduce_scatter(self):
        for N in WORLDS:
            with self.subTest(N=N):
                numel = N * 40
                s = numel // N
                xs = rank_inputs(N, numel, seed=1)

                def fn(rank, gpu, comm):
                    mean, total = torch.empty(s), torch.empty(s)
                    comm.reduce_scatter(xs[rank].clone(), mean, tag="rs")
                    comm.reduce_scatter(xs[rank].clone(), total, tag="rs", average=False)
                    return mean, total

                outs, _ = run(N, fn)
                ref, exact = torch.stack(xs).sum(0), rank_order_sum(xs)
                for r, (mean, total) in enumerate(outs):
                    sl = slice(r * s, (r + 1) * s)
                    self.assertClose(total, ref[sl])
                    self.assertClose(mean, ref[sl] / N)
                    self.assertEqualT(total, exact[sl])          # rank order, fp32 ...
                    self.assertEqualT(mean, exact[sl] / N)       # ... then divide by N

    def test_reduce_scatter_bf16_sums_in_fp32(self):
        N, numel = 4, 4 * 64
        s = numel // N
        xs = rank_inputs(N, numel, seed=11, dtype=torch.bfloat16)

        def fn(rank, gpu, comm):
            out = torch.empty(s, dtype=torch.bfloat16)
            return comm.reduce_scatter(xs[rank].clone(), out, tag="rs")

        outs, _ = run(N, fn)
        exact = rank_order_sum(xs) / N
        for r, out in enumerate(outs):
            self.assertEqualT(out, exact[r * s:(r + 1) * s].to(torch.bfloat16))

    def test_all_gather_including_in_place(self):
        for N in WORLDS:
            with self.subTest(N=N):
                s = 24
                shards = rank_inputs(N, s, seed=2)

                def fn(rank, gpu, comm):
                    out = torch.full((N * s,), float("nan"))
                    comm.all_gather(shards[rank], out, tag="ag")
                    # in place: the shard is a view of this GPU's own slice of the output
                    buf = torch.full((N * s,), float("nan"))
                    mine = buf[rank * s:(rank + 1) * s]
                    mine.copy_(shards[rank])
                    comm.all_gather(mine, buf, tag="ag")
                    return out, buf

                outs, _ = run(N, fn)
                ref = torch.cat(shards)
                for out, buf in outs:
                    self.assertEqualT(out, ref)
                    self.assertEqualT(buf, ref)

    def test_all_reduce(self):
        for N in WORLDS:
            with self.subTest(N=N):
                xs = rank_inputs(N, N * 16, seed=3)
                ys = rank_inputs(N, N * 5 + 3, seed=4)      # not divisible by N: padded path

                def fn(rank, gpu, comm):
                    a = xs[rank].clone()
                    comm.all_reduce(a, tag="ar")
                    b = xs[rank].clone()
                    comm.all_reduce(b, tag="ar", average=False)
                    c = ys[rank].clone()
                    comm.all_reduce(c, tag="ar", average=False)
                    d = xs[rank].clone().reshape(16, N)             # 2-D, contiguous
                    comm.all_reduce(d, tag="ar")
                    return a, b, c, d

                outs, _ = run(N, fn)
                sx, sy = torch.stack(xs).sum(0), torch.stack(ys).sum(0)
                ex, ey = rank_order_sum(xs), rank_order_sum(ys)
                for a, b, c, d in outs:
                    self.assertClose(a, sx / N)
                    self.assertClose(b, sx)
                    self.assertClose(c, sy)
                    self.assertEqual(d.shape, (16, N))
                    self.assertClose(d.reshape(-1), sx / N)
                    self.assertEqualT(a, ex / N)
                    self.assertEqualT(b, ex)
                    self.assertEqualT(c, ey)

    def test_all_reduce_non_contiguous(self):
        """Regression: all_reduce on a non-contiguous tensor must write the result back
        (it once reduced into a reshaped copy and silently left `t` unchanged)."""
        N = 4
        xs = [torch.arange(12.0).reshape(3, 4) + r for r in range(N)]

        def fn(rank, gpu, comm):
            t = xs[rank].clone().t()                    # (4, 3), non-contiguous
            comm.all_reduce(t, tag="ar", average=False)
            return t

        outs, _ = run(N, fn)
        ref = sum(xs).t()
        for t in outs:
            self.assertEqualT(t, ref)

    def test_broadcast(self):
        for N in WORLDS:
            with self.subTest(N=N):
                xs = rank_inputs(N, N * 8, seed=6)

                def fn(rank, gpu, comm):
                    t = xs[rank].clone()
                    return comm.broadcast(t, src=1, tag="bc")

                outs, _ = run(N, fn)
                for t in outs:
                    self.assertEqualT(t, xs[1])

    def test_all_reduce_is_reduce_scatter_then_all_gather(self):
        """Bit for bit: this is why DDP and ZeRO see the same gradient sums."""
        for N in WORLDS:
            with self.subTest(N=N):
                numel = N * 32
                xs = rank_inputs(N, numel, seed=7)

                def fn(rank, gpu, comm):
                    pairs = []
                    for dtype in (torch.float32, torch.bfloat16):
                        for avg in (True, False):
                            x = xs[rank].to(dtype)
                            a = x.clone()
                            comm.all_reduce(a, tag="ar", average=avg)
                            shard = torch.empty(numel // N, dtype=dtype)
                            comm.reduce_scatter(x.clone(), shard, tag="rs", average=avg)
                            b = torch.empty(numel, dtype=dtype)
                            comm.all_gather(shard, b, tag="ag")
                            pairs.append((a, b))
                    return pairs

                outs, _ = run(N, fn)
                for pairs in outs:
                    for a, b in pairs:
                        self.assertEqualT(a, b)


class TestCounters(unittest.TestCase):
    """Bytes and calls charged per GPU follow the ring cost model exactly."""

    def test_ring_formulas(self):
        for N in WORLDS:
            with self.subTest(N=N):
                numel = N * 64
                S = numel * 4                                # full tensor, fp32 bytes

                def fn(rank, gpu, comm):
                    x = torch.randn(numel)
                    comm.reduce_scatter(x.clone(), torch.empty(numel // N), tag="t_rs")
                    comm.reduce_scatter(x.to(torch.bfloat16), torch.empty(
                        numel // N, dtype=torch.bfloat16), tag="t_rs16")
                    comm.all_gather(torch.randn(numel // N), torch.empty(numel), tag="t_ag")
                    before = comm.stats.copy()
                    comm.all_reduce(x.clone(), tag="t_ar")
                    after_one = comm.stats.since(before)
                    comm.all_reduce(x.clone(), tag="t_ar")
                    comm.broadcast(x.clone(), src=0, tag="t_bc")
                    return after_one

                outs, cluster = run(N, fn)
                rs = (N - 1) * S // N
                ar = 2 * (N - 1) * S // N
                for r in range(N):
                    st = cluster.comm.stats[r]
                    self.assertEqual(st.bytes[("t_rs", "reduce_scatter")], rs)
                    self.assertEqual(st.bytes[("t_rs16", "reduce_scatter")], rs // 2)
                    self.assertEqual(st.bytes[("t_ag", "all_gather")], rs)
                    self.assertEqual(st.bytes[("t_ar", "all_reduce")], 2 * ar)
                    self.assertEqual(st.bytes[("t_bc", "broadcast")], rs)
                    self.assertEqual(ar, st.bytes[("t_rs", "reduce_scatter")]
                                     + st.bytes[("t_ag", "all_gather")])   # AR = RS + AG
                    self.assertEqual(dict(st.calls), {
                        ("t_rs", "reduce_scatter"): 1, ("t_rs16", "reduce_scatter"): 1,
                        ("t_ag", "all_gather"): 1, ("t_ar", "all_reduce"): 2,
                        ("t_bc", "broadcast"): 1})               # AR's inner RS+AG not counted
                    self.assertEqual(st.ring_steps, 3 * (N - 1) + 2 * 2 * (N - 1) + (N - 1))
                    one = outs[r]
                    self.assertEqual(dict(one.bytes), {("t_ar", "all_reduce"): ar})
                    self.assertEqual(dict(one.calls), {("t_ar", "all_reduce"): 1})
                    self.assertEqual(one.ring_steps, 2 * (N - 1))
                    self.assertEqual(st.by_op(exclude=())["all_reduce"], 2 * ar)
                    self.assertEqual(st.total_bytes(exclude=("t_bc",)),
                                     rs + rs // 2 + rs + 2 * ar)


class TestRingAllReduce(unittest.TestCase):
    """The hand-written ring: neighbour-to-neighbour messages only."""

    def test_ring_matches_library_and_formula(self):
        for N in (4, 32):
            with self.subTest(N=N):
                numel = N * 16
                S = numel * 4
                xs = rank_inputs(N, numel, seed=5)

                def fn(rank, gpu, comm):
                    trace = []
                    got = ring_all_reduce(comm, xs[rank].clone(), trace)
                    ref = xs[rank].clone()
                    comm.all_reduce(ref, tag="ref", average=False)
                    return got, ref, trace

                outs, cluster = run(N, fn)
                total = torch.stack(xs).sum(0)
                for r, (got, ref, trace) in enumerate(outs):
                    self.assertTrue(torch.allclose(got, ref, rtol=1e-5, atol=1e-5))
                    self.assertTrue(torch.allclose(got, total, rtol=1e-5, atol=1e-5))
                    st = cluster.comm.stats[r]
                    self.assertEqual(st.bytes[("ring", "send")], 2 * (N - 1) * S // N)
                    self.assertEqual(st.calls[("ring", "send")], 2 * (N - 1))
                    self.assertEqual(len(trace), 1 + 2 * (N - 1))
                    self.assertEqual(trace[0], ("start", 0, [1] * N))
                    phase, step, have = trace[N - 1]              # end of reduce-scatter
                    self.assertEqual((phase, step), ("reduce-scatter", N - 1))
                    self.assertEqual(have.count(N), 1)            # exactly one complete chunk
                    self.assertEqual(have[(r + 1) % N], N)        # ... the one the docstring says
                    self.assertEqual(trace[-1], ("all-gather", N - 1, [N] * N))


class TestVirtualGPU(unittest.TestCase):

    def test_alloc_free_peak(self):
        gpu = VirtualGPU(0)
        gpu.alloc("w", 4 * MiB, "weights")
        gpu.alloc("g", 3 * MiB, "grads")
        self.assertEqual(gpu.current, 7 * MiB)
        self.assertTrue(gpu.holds("g"))
        gpu.free("g")
        gpu.alloc("m", 1 * MiB, "adam_m")
        self.assertFalse(gpu.holds("g"))
        self.assertEqual(gpu.current, 5 * MiB)
        self.assertEqual(gpu.peak, 7 * MiB)
        self.assertEqual(gpu.peak_by_cat["weights"], 4 * MiB)
        self.assertEqual(gpu.peak_by_cat["grads"], 3 * MiB)
        self.assertEqual(gpu.peak_by_cat["adam_m"], 0)              # arrived after the peak
        self.assertEqual(gpu.by_cat["grads"], 0)
        self.assertEqual(gpu.model_state_bytes(), 5 * MiB)
        gpu.reset_peak()
        self.assertEqual(gpu.peak, 5 * MiB)
        with self.assertRaises(KeyError):
            gpu.alloc("w", 1, "weights")                            # names are unique
        t = gpu.tensor("bf", 1000, torch.bfloat16, "temp")
        self.assertEqual((t.numel(), t.dtype), (1000, torch.bfloat16))
        self.assertEqual(gpu.by_cat["temp"], 2000)

    def test_mark_is_a_snapshot(self):
        gpu = VirtualGPU(0)
        gpu.alloc("w", 100, "weights")
        gpu.mark("end_of_backward")
        gpu.alloc("g", 50, "grads")
        self.assertEqual(gpu.marks["end_of_backward"]["weights"], 100)
        self.assertEqual(gpu.marks["end_of_backward"]["grads"], 0)
        self.assertEqual(gpu.model_state_bytes(gpu.marks["end_of_backward"]), 100)

    def test_capacity_raises_before_allocating(self):
        gpu = VirtualGPU(5, capacity=10 * MiB)
        gpu.alloc("w", 4 * MiB, "weights")
        gpu.alloc("g", 6 * MiB, "grads")                            # exactly full: allowed
        gpu.free("g")
        boom = AssertionError("a real tensor was created before the capacity check")
        with mock.patch("torch.zeros", side_effect=boom), \
                mock.patch("torch.empty", side_effect=boom):
            with self.assertRaises(VirtualOOMError) as cm:
                gpu.tensor("adam_m", 7 * MiB // 4, torch.float32, "adam_m")
        self.assertEqual(gpu.current, 4 * MiB)                       # ledger untouched
        self.assertFalse(gpu.holds("adam_m"))
        msg = str(cm.exception)
        for part in ("vGPU 5", "7.00 MiB", "Capacity 10.00 MiB", "4.00 MiB already held",
                     "weights 4.00", "6.00 MiB free", "adam_m"):
            self.assertIn(part, msg)
        self.assertIsInstance(cm.exception, RuntimeError)            # like torch's OOM


class TestVirtualCluster(unittest.TestCase):

    def _assert_propagates(self, other_ranks_do, exc_type=VirtualOOMError, bad_rank=2):
        """Rank `bad_rank` of 8 fails while the others block; the original error must come
        out quickly, with a note naming the rank, and torch's thread count restored."""
        N = 8
        old = torch.get_num_threads()
        torch.set_num_threads(3)
        try:
            cluster = VirtualCluster(N, capacity=1 * MiB, timeout=60.0)

            def fn(rank, gpu, comm):
                if rank == bad_rank:
                    time.sleep(0.2)                                  # let the others block first
                    if exc_type is VirtualOOMError:
                        gpu.tensor("too_big", MiB, torch.float32, "weights")    # 4 MiB > 1 MiB
                    raise exc_type("boom")
                other_ranks_do(comm)

            t0 = time.perf_counter()
            with self.assertRaises(exc_type) as cm:
                cluster.run(fn)
            self.assertLess(time.perf_counter() - t0, 5.0)           # no deadlock
            notes = " ".join(getattr(cm.exception, "__notes__", []))
            self.assertIn(f"vGPU {bad_rank}", notes)
            self.assertIn(f"of {N} GPUs failed", notes)
            self.assertEqual(torch.get_num_threads(), 3)
            self.assertFalse(any(t.name.startswith("vGPU-") and t.is_alive()
                                 for t in threading.enumerate()))
            return cm.exception
        finally:
            torch.set_num_threads(old)

    def test_oom_while_others_wait_at_barrier(self):
        err = self._assert_propagates(lambda comm: comm.barrier())
        self.assertIn("vGPU 2 out of memory", str(err))

    def test_error_while_others_wait_in_a_collective(self):
        self._assert_propagates(lambda comm: comm.all_reduce(torch.ones(64), tag="x"),
                                exc_type=ValueError)

    def test_error_while_others_wait_in_recv(self):
        self._assert_propagates(lambda comm: comm.recv(), exc_type=KeyError)

    def test_lowest_rank_error_wins(self):
        def fn(rank, gpu, comm):
            if rank == 5:
                raise ValueError("rank 5")
            if rank == 2:
                raise KeyError("rank 2")
            comm.barrier()

        for _ in range(3):                                           # deterministic, every time
            with self.assertRaises(KeyError):
                VirtualCluster(8, timeout=TIMEOUT).run(fn)

    def test_one_thread_per_gpu_and_results_in_rank_order(self):
        old = torch.get_num_threads()
        out, _ = run(4, lambda rank, gpu, comm: (rank, torch.get_num_threads(), gpu.rank))
        self.assertEqual(out, [(r, 1, r) for r in range(4)])
        self.assertEqual(torch.get_num_threads(), old)


class TestActivationLedger(unittest.TestCase):

    @staticmethod
    def _block():
        torch.manual_seed(0)
        cfg = zs.GPTConfig(n_embd=32, n_head=2, block_size=16)
        return zs.Block(cfg), torch.randn(2, 16, 32)

    def test_activations_return_to_zero(self):
        gpu = VirtualGPU(0)
        block, x = self._block()
        x.requires_grad_(True)
        with gpu.saved_tensor_hooks(lambda ptr: False):
            out = block(x)
        during = gpu.by_cat["activations"]
        out.square().mean().backward()
        del out
        self.assertGreater(during, 0)
        self.assertEqual(gpu.by_cat["activations"], 0)
        self.assertEqual(gpu.current, 0)
        self.assertEqual(gpu.peak_by_cat["activations"], gpu.peak)
        self.assertEqual(gpu._act_refs, {})

    def test_weights_are_not_counted_as_activations(self):
        lin = nn.Linear(16, 16, bias=False)
        x = torch.randn(4, 16, requires_grad=True)
        peaks = []
        for skip in (lambda ptr: False,
                     lambda ptr: ptr == lin.weight.untyped_storage().data_ptr()):
            gpu = VirtualGPU(0)
            with gpu.saved_tensor_hooks(skip):
                y = lin(x)
            y.sum().backward()
            del y
            self.assertEqual(gpu.current, 0)
            peaks.append(gpu.peak)
        self.assertEqual(peaks[0] - peaks[1], lin.weight.untyped_storage().nbytes())

    def test_per_thread_ledgers(self):
        """Hooks are per thread: each virtual GPU is charged only for its own activations."""
        block, x = self._block()
        blocks = [copy.deepcopy(block) for _ in range(4)]         # one model per GPU

        def fn(rank, gpu, comm):
            xr = (x + rank).requires_grad_(True)
            with gpu.saved_tensor_hooks(lambda ptr: False):
                out = blocks[rank](xr)
            comm.barrier()                                           # all graphs alive at once
            during = gpu.by_cat["activations"]
            out.sum().backward()
            del out
            return during, gpu.by_cat["activations"]

        outs, _ = run(4, fn)
        self.assertEqual(len({d for d, _ in outs}), 1)                # same shapes, same bytes
        for during, after in outs:
            self.assertGreater(during, 0)
            self.assertEqual(after, 0)


if __name__ == "__main__":
    unittest.main()
