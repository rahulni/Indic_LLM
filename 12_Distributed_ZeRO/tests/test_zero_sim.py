"""Engine tests: the four ZeRO stages on virtual GPUs.

Run from the project folder:  python -m unittest tests.test_zero_sim -v
"""
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import zero_sim  # noqa: E402
from zero_sim import (MODEL_STATES, STAGES, CharData, GPTConfig, RunConfig,  # noqa: E402
                      TinyGPT, VirtualCluster, VirtualOOMError, estimate_peak_bytes,
                      expected_model_state_bytes, init_weights, make_units, run_training)

# The notebook's tolerances for "DDP on N GPUs == one GPU with the whole batch" (fp32).
LOSS_TOL, WEIGHT_TOL = 5e-6, 2e-5

random.seed(0)
TEXT = "".join(random.choice("abcdefghij klmnop\n") for _ in range(20000))
DATA = CharData(TEXT)
SMALL = GPTConfig(vocab_size=DATA.vocab_size, n_embd=32, n_layer=2, n_head=2, block_size=16)
INIT = init_weights(SMALL, 1337)


def run(stage, world=4, steps=2, **kw):
    return run_training(RunConfig(stage=stage, world=world, steps=steps, **kw), SMALL, DATA, INIT)


def model_states(rec):
    snap = rec["end_of_backward"]
    return sum(snap[c] for c in MODEL_STATES)


class TestEquivalence(unittest.TestCase):
    """ZeRO changes the memory layout, never the numbers."""

    def check_bit_exact(self, world, precision="bf16"):
        runs = {s: run(s, world, 3, precision=precision) for s in range(4)}
        base = runs[0].consolidated()
        for s in (1, 2, 3):
            self.assertEqual(runs[s].losses, runs[0].losses, f"stage {s} losses")
            other = runs[s].consolidated()
            for k in base:
                self.assertTrue(torch.equal(base[k], other[k]), f"stage {s} unit {k}")

    def test_bit_exact_n4_bf16(self):
        self.check_bit_exact(4)

    def test_bit_exact_n3_uneven(self):
        self.check_bit_exact(3)

    def test_bit_exact_n32(self):
        self.check_bit_exact(32)

    def test_bit_exact_fp32(self):
        self.check_bit_exact(4, "fp32")

    @staticmethod
    def ddp_vs_single(world=8, steps=3):
        a = run(0, world, steps, precision="fp32")
        b = run_training(RunConfig(stage=0, world=1, micro_bsz=world, steps=steps,
                                   precision="fp32"), SMALL, DATA, INIT)
        dl = max(abs(x - y) for x, y in zip(a.losses, b.losses))
        fa, fb = a.module_flats(SMALL), b.module_flats(SMALL)
        return dl, max((fa[k] - fb[k]).abs().max().item() for k in fa)

    def test_ddp_matches_single_device(self):
        dl, dw = self.ddp_vs_single()
        self.assertLess(dl, LOSS_TOL)
        self.assertLess(dw, WEIGHT_TOL)

    def test_missing_average_is_caught(self):
        """Mutation test: summing gradients instead of averaging them (the classic DDP bug)
        must fail the tolerances above. Adam is nearly scale-invariant, so loose tolerances
        would let it through."""
        orig = zero_sim.ThreadComm.all_reduce

        def buggy(self_, rank, t, tag, average=True):
            return orig(self_, rank, t, tag, average=False if tag == "grad_sync" else average)

        zero_sim.ThreadComm.all_reduce = buggy
        try:
            dl, dw = self.ddp_vs_single()
        finally:
            zero_sim.ThreadComm.all_reduce = orig
        self.assertTrue(dl >= LOSS_TOL or dw >= WEIGHT_TOL, (dl, dw))

    def test_checkpointing_changes_nothing_but_memory(self):
        for s in range(4):
            a, b = run(s, 4, 3), run(s, 4, 3, checkpoint=True)
            self.assertEqual(a.losses, b.losses, s)
            ca, cb = a.consolidated(), b.consolidated()
            self.assertTrue(all(torch.equal(ca[k], cb[k]) for k in ca), s)

    def test_grad_accumulation_equivalence(self):
        """With G > 1, ZeRO-1 still matches DDP bit for bit (both accumulate locally and
        reduce once). ZeRO-2/3 reduce every micro-batch and add the reduced slices, a
        different floating-point order, so they agree to rounding only."""
        runs = {s: run(s, 4, 3, grad_accum=2) for s in range(4)}
        c0 = runs[0].consolidated()
        self.assertEqual(runs[1].losses, runs[0].losses)
        self.assertTrue(all(torch.equal(c0[k], runs[1].consolidated()[k]) for k in c0))
        for s in (2, 3):
            for x, y in zip(runs[s].losses, runs[0].losses):
                self.assertAlmostEqual(x, y, delta=2e-2)


class TestMemory(unittest.TestCase):
    def test_model_states_equal_formula(self):
        for world in (1, 3, 4, 8):
            for prec in ("bf16", "fp32"):
                for s in range(4):
                    r = run(s, world, 1, precision=prec, keep_state=False)
                    for rec in r.records[0]:
                        self.assertEqual(model_states(rec),
                                         expected_model_state_bytes(s, r.P, world, prec),
                                         (world, prec, s))

    def test_no_activation_leak(self):
        for s in range(4):
            r = run(s, 4, 2, keep_state=False)
            for rec in r.records:
                for g in rec:
                    self.assertEqual(g["activations_left"], 0)
                    self.assertGreater(g["cat_peak"]["activations"], 0)

    def test_checkpointing_reduces_activations(self):
        a = run(3, 4, 1, keep_state=False)
        b = run(3, 4, 1, checkpoint=True, keep_state=False)
        act_b = b.records[0][0]["cat_peak"]["activations"]
        self.assertGreater(act_b, 0)                    # the kept unit inputs are booked
        self.assertLess(act_b, a.records[0][0]["cat_peak"]["activations"])
        self.assertEqual(b.records[0][0]["activations_left"], 0)

    def test_zero3_peak_below_estimate(self):
        act = run(0, 1, 1, keep_state=False).records[0][0]["cat_peak"]["activations"]
        specs = make_units(SMALL, 4)
        for s in range(4):
            peak = run(s, 4, 1, keep_state=False).records[0][0]["peak"]
            self.assertLessEqual(peak, estimate_peak_bytes(s, specs, 4, act), s)

    def test_oom_aborts_cleanly(self):
        with self.assertRaises(VirtualOOMError):
            run(0, 8, 1, capacity=20_000, keep_state=False)


class TestCommunication(unittest.TestCase):
    def test_bytes_and_calls_per_step(self):
        N = 4
        for s in range(4):
            r = run(s, N, 1, keep_state=False)
            comm = r.records[0][0]["comm"]
            unit = (N - 1) * 2 * r.P // N
            self.assertEqual(comm.total_bytes(), (3 if s == 3 else 2) * unit)
            U = len(r.specs)
            self.assertEqual(comm.total_calls(), {0: U, 1: 2 * U, 2: 2 * U, 3: 3 * U}[s])

    def test_grad_accumulation_traffic(self):
        N, G = 4, 3
        for s in range(4):
            r = run(s, N, 1, grad_accum=G, keep_state=False)
            unit = (N - 1) * 2 * r.P // N
            expected = {0: 2 * unit, 1: 2 * unit, 2: (G + 1) * unit, 3: 3 * G * unit}[s]
            self.assertEqual(r.records[0][0]["comm"].total_bytes(), expected, s)


class TestCompute(unittest.TestCase):
    def test_flops_equal_across_stages_and_formula(self):
        cfg = SMALL
        B, T, d, L, V = 1, cfg.block_size, cfg.n_embd, cfg.n_layer, cfg.vocab_size
        fwd = 2 * B * T * (12 * L * d * d + d * V) + 4 * L * B * T * T * d
        for s in range(4):
            r = run(s, 4, 1, flops_step=0, keep_state=False)
            self.assertEqual(r.flops[0]["forward"], fwd)
            self.assertEqual(r.flops[0]["backward"], 2 * fwd)

    def test_optimizer_elements(self):
        for s in range(4):
            r = run(s, 4, 1, keep_state=False)
            self.assertEqual(r.opt_elements[0], r.P if s == 0 else r.P // 4)


class TestZeRO3Mechanics(unittest.TestCase):
    def test_released_unit_refuses_to_run(self):
        specs = make_units(SMALL, 2)

        def fn(rank, gpu, comm):
            eng = STAGES[3](rank, gpu, comm, SMALL, specs, INIT)
            u = eng.units[1]
            self.assertEqual(u.flat_w.untyped_storage().nbytes(), 0)
            with self.assertRaises(RuntimeError):
                u.mods[0](torch.zeros(1, 4, SMALL.n_embd, dtype=torch.bfloat16))
            return True

        self.assertEqual(VirtualCluster(2).run(fn), [True, True])

    def test_consolidated_checkpoint_loads(self):
        r = run(3, 4, 2)
        model = TinyGPT(SMALL).load_flat(r.module_flats(SMALL))
        x = DATA.micro_batches(0, 0, 1, 2, 1, SMALL.block_size, 1337)[0][0]
        self.assertTrue(torch.isfinite(model(x)).all())
        ddp = run(0, 4, 2).module_flats(SMALL)
        for k, v in r.module_flats(SMALL).items():
            self.assertTrue(torch.equal(v, ddp[k]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
