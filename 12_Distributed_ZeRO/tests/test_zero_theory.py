"""Tests for zero_theory.py: the closed-form ZeRO maths (no torch needed).

Run from the project folder:  python -m unittest tests.test_zero_theory -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zero_theory as zt  # noqa: E402

A100, H100 = "A100-80GB", "H100-80GB"


class TestPaperNumbers(unittest.TestCase):
    def test_figure1_rounded_like_the_paper(self):
        f1 = zt.paper_figure1()
        got = {s: round(v, 1) for s, v in f1["gb"].items()}
        self.assertEqual(got, {0: 120.0, 1: 31.4, 2: 16.6, 3: 1.9})
        self.assertEqual(got, f1["paper_gb"])

    def test_figure1_exact(self):
        gb = zt.paper_figure1()["gb"]
        self.assertAlmostEqual(gb[1], 31.40625, places=9)
        self.assertAlmostEqual(gb[2], 16.640625, places=9)
        self.assertAlmostEqual(gb[3], 1.875, places=12)

    def test_trillion_on_1024_gpus(self):
        t = zt.paper_trillion_example()
        self.assertAlmostEqual(t["total_tb"], 16.0)
        self.assertAlmostEqual(t["per_gpu_gb"], 15.625)
        self.assertEqual(round(t["per_gpu_gb"], 1), 15.6)

    def test_plan_section5_toy_model_bytes(self):
        # PLAN.md section 5: padded Ψ' = 823,296 on N = 32, to the byte.
        psi, n = 823_296, 32
        want = {0: 13_172_736, 1: 3_601_920, 2: 2_006_784, 3: 411_648}
        for stage, b in want.items():
            self.assertEqual(zt.model_state_bytes(stage, psi, n), b)


class TestMemory(unittest.TestCase):
    PSI = 7e9

    def test_breakdown_sums_and_shards_one_more_state_per_stage(self):
        n = 8
        for stage in zt.STAGES:
            b = zt.model_state_breakdown(stage, self.PSI, n)
            self.assertAlmostEqual(sum(b.values()), zt.model_state_bytes(stage, self.PSI, n))
            self.assertEqual(b["optimizer"], 12 * self.PSI / (n if stage >= 1 else 1))
            self.assertEqual(b["grads"], 2 * self.PSI / (n if stage >= 2 else 1))
            self.assertEqual(b["weights"], 2 * self.PSI / (n if stage >= 3 else 1))

    def test_stage3_scales_exactly_as_one_over_n(self):
        for n in (1, 2, 4, 8, 16, 32, 64, 1024):
            self.assertEqual(zt.model_state_bytes(3, self.PSI, n) * n, 16 * self.PSI)

    def test_stage0_never_shrinks(self):
        for n in (1, 32, 1024):
            self.assertEqual(zt.model_state_bytes(0, self.PSI, n), 16 * self.PSI)

    def test_stage1_and_2_approach_4psi_and_2psi(self):
        ns = [2 ** k for k in range(21)]
        for stage, floor in ((1, 4 * self.PSI), (2, 2 * self.PSI)):
            vals = [zt.model_state_bytes(stage, self.PSI, n) for n in ns]
            self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])), "must decrease")
            self.assertTrue(all(v > floor for v in vals), "never below the floor")
            self.assertLess(vals[-1] / floor - 1, 1e-5)

    def test_all_stages_equal_at_n1(self):
        vals = {zt.model_state_bytes(s, self.PSI, 1) for s in zt.STAGES}
        self.assertEqual(vals, {16 * self.PSI})

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            zt.model_state_bytes(4, self.PSI, 8)
        with self.assertRaises(ValueError):
            zt.model_state_bytes(1, self.PSI, 0)


class TestCommunication(unittest.TestCase):
    def test_volumes(self):
        self.assertEqual([zt.comm_volume_psi(s) for s in zt.STAGES], [2, 2, 2, 3])

    def test_ring_factor(self):
        psi = 1e9
        for n in (2, 3, 4, 32, 1024):
            for s in zt.STAGES:
                got = zt.comm_bytes_per_gpu(s, psi, n, bytes_per_elem=2)
                self.assertAlmostEqual(got / (zt.comm_volume_psi(s) * psi * 2), (n - 1) / n)
        self.assertEqual(zt.comm_bytes_per_gpu(0, psi, 1), 0)

    def test_plan_section5_toy_bytes(self):
        psi, n = 823_296, 32
        self.assertEqual(round(zt.comm_bytes_per_gpu(0, psi, n) / 1e6, 2), 3.19)
        self.assertEqual(round(zt.comm_bytes_per_gpu(3, psi, n) / 1e6, 2), 4.79)
        self.assertAlmostEqual(zt.comm_bytes_per_gpu(3, psi, n) / zt.comm_bytes_per_gpu(1, psi, n), 1.5)

    def test_collective_calls_six_units(self):
        self.assertEqual([zt.collective_calls(s, 6) for s in zt.STAGES], [6, 12, 12, 18])
        self.assertEqual(zt.collective_calls_by_op(3, 6),
                         {"all_reduce": 0, "reduce_scatter": 6, "all_gather": 12})

    def test_ring_latency_steps_match_volume(self):
        # each Ψ of volume is one pass of N-1 ring hops per unit
        n, u = 32, 6
        for s in zt.STAGES:
            self.assertEqual(zt.ring_latency_steps(s, n, u), zt.comm_volume_psi(s) * u * (n - 1))

    def test_grad_accum_ordering(self):
        psi, n, G = 7e9, 32, 4
        b = {s: zt.grad_accum_comm_bytes(s, psi, n, G) for s in zt.STAGES}
        self.assertEqual(b[0], b[1])
        self.assertGreater(b[2], b[1])
        self.assertGreater(b[3], b[2])
        unit = zt.ring_factor(n) * psi * 2
        self.assertAlmostEqual(b[2] / unit, G + 1)
        self.assertAlmostEqual(b[3] / unit, 3 * G)
        # with G = 1 it collapses to the plain per-step volumes
        for s in zt.STAGES:
            self.assertEqual(zt.grad_accum_comm_bytes(s, psi, n, 1), zt.comm_bytes_per_gpu(s, psi, n))

    def test_grad_accum_calls(self):
        self.assertEqual(zt.collective_calls(2, 6, grad_accum=4), 4 * 6 + 6)
        self.assertEqual(zt.collective_calls(3, 6, grad_accum=4), 3 * 4 * 6)
        self.assertEqual(zt.collective_calls(1, 6, grad_accum=4), 12)

    def test_zero3_comm_per_token_ignores_accumulation(self):
        psi, n, tok = 7e9, 32, 4096
        r1 = zt.step_time(3, psi, n, tok, A100, alpha=0, grad_accum=1)["comm_over_compute"]
        r8 = zt.step_time(3, psi, n, 8 * tok, A100, alpha=0, grad_accum=8)["comm_over_compute"]
        self.assertAlmostEqual(r1, r8)
        d1 = zt.step_time(0, psi, n, tok, A100, alpha=0, grad_accum=1)["comm_over_compute"]
        d8 = zt.step_time(0, psi, n, 8 * tok, A100, alpha=0, grad_accum=8)["comm_over_compute"]
        self.assertAlmostEqual(d1 / d8, 8)


class TestActivations(unittest.TestCase):
    s, b, h, a = 4096, 2, 4096, 32

    def test_special_cases(self):
        s, b, h, a = self.s, self.b, self.h, self.a
        f = zt.activation_bytes_per_layer
        self.assertEqual(f(s, b, h, a, "full"), 2 * s * b * h)
        self.assertEqual(f(s, b, h, a, "selective"), 34 * s * b * h)
        self.assertAlmostEqual(f(s, b, h, a, "none"), s * b * h * (34 + 5 * a * s / h))
        # the only thing selective recomputation drops is the 5·a·s²·b attention-score term
        self.assertAlmostEqual(f(s, b, h, a, "none") - f(s, b, h, a, "selective"), 5 * a * s * s * b)

    def test_ordering_and_linearity_in_batch(self):
        f = zt.activation_bytes_per_layer
        vals = [f(self.s, self.b, self.h, self.a, m) for m in ("none", "selective", "full")]
        self.assertTrue(vals[0] > vals[1] > vals[2])
        for m in zt.RECOMPUTE_MODES:
            self.assertAlmostEqual(f(self.s, 4, self.h, self.a, m), 2 * f(self.s, 2, self.h, self.a, m))
        with self.assertRaises(ValueError):
            f(self.s, self.b, self.h, self.a, "sometimes")

    def test_model_total(self):
        m = zt.MODELS["LLaMA-2 7B"]
        per = zt.activation_bytes_per_layer(4096, 1, 4096, 32, "selective")
        self.assertEqual(zt.activation_bytes_model("LLaMA-2 7B", 1, recompute="selective"),
                         m["n_layers"] * per)
        full = zt.activation_bytes_model("LLaMA-2 7B", 1, recompute="full")
        self.assertEqual(full, 32 * 2 * 4096 * 4096
                         + zt.activation_bytes_per_layer(4096, 1, 4096, 32, "none"))


class TestModelsAndHardware(unittest.TestCase):
    def test_param_counts_match_architecture(self):
        self.assertEqual(zt.gpt2_param_count(48, 1600), zt.MODELS["GPT-2 XL"]["n_params"])
        for name in ("LLaMA-2 7B", "LLaMA-2 13B", "LLaMA-2 70B"):
            m = zt.MODELS[name]
            got = zt.llama_param_count(m["n_layers"], m["hidden"], m["ffn"], m["n_heads"],
                                       m["n_kv_heads"], m["vocab"])
            self.assertEqual(got, m["n_params"], name)

    def test_published_sizes(self):
        self.assertEqual(round(zt.MODELS["GPT-2 XL"]["n_params"] / 1e6), 1558)
        self.assertEqual(round(zt.MODELS["LLaMA-2 7B"]["n_params"] / 1e9, 1), 6.7)
        self.assertEqual(round(zt.MODELS["LLaMA-2 13B"]["n_params"] / 1e9, 1), 13.0)
        self.assertEqual(round(zt.MODELS["LLaMA-2 70B"]["n_params"] / 1e9, 1), 69.0)

    def test_hardware_constants(self):
        a, h = zt.HARDWARE[A100], zt.HARDWARE[H100]
        self.assertEqual(a["peak_bf16_flops"], 312e12)
        self.assertEqual(h["peak_bf16_flops"], 989e12)
        self.assertEqual((a["nvlink_bw"], h["nvlink_bw"]), (300e9, 450e9))   # 600/900 bidir
        self.assertEqual((a["inter_node_bw"], h["inter_node_bw"]), (25e9, 50e9))
        for hw in (a, h):
            self.assertEqual(hw["hbm_bytes"], 80e9)
            self.assertIn("datasheet", hw["source"])

    def test_effective_bandwidth(self):
        self.assertEqual(zt.effective_bandwidth(A100, 8), 300e9)
        self.assertEqual(zt.effective_bandwidth(A100, 32), 25e9)
        self.assertEqual(zt.effective_bandwidth(A100, 32, "rail"), 200e9)
        self.assertEqual(zt.effective_bandwidth(H100, 32, "rail"), 400e9)

    def test_70b_model_states_fit_only_with_zero3(self):
        psi = zt.MODELS["LLaMA-2 70B"]["n_params"]
        fits = [zt.model_state_bytes(s, psi, 32) <= 80e9 for s in zt.STAGES]
        self.assertEqual(fits, [False, False, False, True])


class TestTime(unittest.TestCase):
    def test_overlap_thresholds(self):
        expect = {A100: (3.3e3, 5.0e3), H100: (5.3e3, 7.9e3)}
        for hw, (t2, t3) in expect.items():
            for s in (0, 1, 2):
                self.assertAlmostEqual(zt.overlap_threshold_tokens(s, hw), t2, delta=60)
            self.assertAlmostEqual(zt.overlap_threshold_tokens(3, hw), t3, delta=60)
        self.assertAlmostEqual(zt.overlap_threshold_tokens(0, A100), 3328.0)
        self.assertAlmostEqual(zt.overlap_threshold_tokens(3, A100) / zt.overlap_threshold_tokens(0, A100), 1.5)

    def test_threshold_is_independent_of_psi(self):
        # At T* tokens, comm == compute for ANY model size: Ψ cancels.
        for hw in (A100, H100):
            for s in zt.STAGES:
                t_star = zt.overlap_threshold_tokens(s, hw, n=32)
                for psi in (1e8, 1.5e9, 7e9, 70e9, 1e12):
                    r = zt.step_time(s, psi, 32, t_star, hw, alpha=0)["comm_over_compute"]
                    self.assertAlmostEqual(r, 1.0, places=9)

    def test_step_time_parts(self):
        psi, n, tok = 7e9, 32, 4096
        t = zt.step_time(0, psi, n, tok, A100, alpha=0)
        self.assertAlmostEqual(t["compute_s"], 6 * psi * tok / (312e12 * 0.4))
        self.assertAlmostEqual(t["comm_s"], 2 * 31 / 32 * psi * 2 / 25e9)
        rc = zt.step_time(0, psi, n, tok, A100, recompute=True)
        self.assertAlmostEqual(rc["compute_s"] / t["compute_s"], 8 / 6)
        lat = zt.step_time(3, psi, n, tok, A100, alpha=10e-6, n_units=34)
        self.assertAlmostEqual(lat["latency_s"], 10e-6 * 3 * 34 * 31)
        self.assertEqual(zt.step_time(3, psi, 1, tok, A100)["comm_s"], 0)


class TestHybrid(unittest.TestCase):
    def test_hsdp_memory_and_bytes(self):
        psi, n, g = 7e9, 32, 8
        h = zt.hybrid_zero3(psi, n, g, hw=A100, alpha=0)
        self.assertEqual(h["memory_bytes"], 16 * psi / g)
        self.assertAlmostEqual(h["intra_node_bytes"], 3 * 7 / 8 * psi * 2)
        self.assertAlmostEqual(h["inter_node_bytes"], 2 * 3 / 4 * psi / g * 2)
        self.assertAlmostEqual(h["comm_s"], h["intra_node_bytes"] / 300e9 + h["inter_node_bytes"] / 25e9)
        self.assertLess(h["comm_s"], h["flat_zero3_comm_s"])
        with self.assertRaises(ValueError):
            zt.hybrid_zero3(psi, 12, 8)

    def test_hpz_reproduces_zeropp_114x(self):
        # ZeRO++: 100B params, 1024 GPUs, 16 GPUs per node -> 114x less memory than DP
        h = zt.zeropp_hpz(100e9, 1024, gpus_per_node=16)
        self.assertEqual(round(16 * 100e9 / h["memory_bytes"]), 114)
        # cross-node volume drops from 3Ψ to 2Ψ
        self.assertAlmostEqual(h["inter_node_bytes"] / zt.comm_bytes_per_gpu(3, 100e9, 1024), 2 / 3)


class TestTableAndDecision(unittest.TestCase):
    def test_rows_shape_and_70b(self):
        rows = zt.real_hardware_rows(32, A100)
        self.assertEqual(len(rows), len(zt.MODELS) * 6)
        r70 = {r["variant"]: r for r in rows if r["model"] == "LLaMA-2 70B"}
        self.assertEqual([r70[v]["states_gb"] <= 80 for v in ("DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3")],
                         [False, False, False, True])
        self.assertEqual(r70["ZeRO-3"]["fits_with"], "full")
        self.assertEqual(r70["HSDP"]["fits_with"], "no")
        full = {r["variant"]: r for r in zt.real_hardware_rows(32, A100, recompute="full")
                if r["model"] == "LLaMA-2 70B"}
        self.assertTrue(full["ZeRO-3"]["fits_80GB"])
        self.assertFalse(full["ZeRO-2"]["fits_80GB"])
        self.assertIn("LLaMA-2 70B", zt.rows_to_text(rows))

    def test_rows_compute_and_comm_consistent(self):
        for r in zt.real_hardware_rows(32, H100):
            self.assertAlmostEqual(r["comm_over_compute"], r["comm_s"] / r["compute_s"])
            self.assertAlmostEqual(r["total_gb"], r["states_gb"] + r["act_gb"])

    def test_decision(self):
        mem = 80e9
        self.assertEqual(zt.decision(1.5e9, 32, mem, 3e9), "DDP")
        self.assertEqual(zt.decision(6.74e9, 32, mem, 18e9), "ZeRO-1")
        self.assertEqual(zt.decision(13e9, 32, mem, 28.5e9), "ZeRO-2")
        self.assertEqual(zt.decision(69e9, 32, mem, 12e9), "ZeRO-3")
        self.assertEqual(zt.decision(69e9, 32, mem, 91e9), "ZeRO-3 + checkpointing/offload/TP/PP")


if __name__ == "__main__":
    unittest.main()
