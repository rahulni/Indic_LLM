"""CI-safe unittest wrapper around capacity_proof.py's checks.

Runs a much smaller sweep than the full report (fewer trials, fewer cells)
so the suite completes in well under a second, while still exercising the
exact same code path -- including the single-pair-unbind-is-exact assertion
that run_all() performs internally.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from capacity_proof import (  # noqa: E402
    build_fillers,
    build_roles,
    random_unitary_vector,
    sweep_cell,
    theoretical_decode_accuracy,
)


class TestHolographicBindingProof(unittest.TestCase):
    def test_unitary_vector_has_unit_magnitude_spectrum(self):
        rng = np.random.default_rng(0)
        for D in (8, 16, 63, 64):
            v = random_unitary_vector(D, rng)
            spec = np.fft.fft(v)
            self.assertTrue(np.allclose(np.abs(spec), 1.0, atol=1e-8), f"D={D} spectrum not unit magnitude")

    def test_single_pair_unbind_is_exact(self):
        rng = np.random.default_rng(0)
        D = 64
        roles = build_roles(4, D, rng)
        fillers = build_fillers(5, D, rng)
        bound = np.fft.ifft(np.fft.fft(roles[1]) * np.fft.fft(fillers[3])).real
        recovered = np.fft.ifft(np.fft.fft(bound) * np.conj(np.fft.fft(roles[1]))).real
        self.assertLess(np.max(np.abs(recovered - fillers[3])), 1e-8)

    def test_decode_accuracy_at_length_one_is_perfect(self):
        rng = np.random.default_rng(0)
        D = 128
        roles = build_roles(4, D, rng)
        fillers = build_fillers(10, D, rng)
        result = sweep_cell(L=1, D=D, roles=roles, fillers=fillers, n_trials=50, rng=rng)
        self.assertEqual(result["accuracy"], 1.0)

    def test_decode_accuracy_degrades_with_length(self):
        rng = np.random.default_rng(0)
        D = 64
        roles = build_roles(40, D, rng)
        fillers = build_fillers(10, D, rng)
        short = sweep_cell(L=2, D=D, roles=roles, fillers=fillers, n_trials=100, rng=rng)
        long = sweep_cell(L=40, D=D, roles=roles, fillers=fillers, n_trials=100, rng=rng)
        self.assertGreater(short["accuracy"], long["accuracy"])

    def test_theoretical_bound_tracks_measurement(self):
        """The derived SNR bound must actually predict the measurement, within
        a tolerance that acknowledges its stated Gaussian-independence
        approximation (expected to be mildly optimistic)."""
        rng = np.random.default_rng(0)
        D, alphabet_size = 192, 30
        roles = build_roles(40, D, rng)
        fillers = build_fillers(alphabet_size, D, rng)
        for L in (8, 20, 40):
            measured = sweep_cell(L=L, D=D, roles=roles, fillers=fillers, n_trials=200, rng=rng)["accuracy"]
            theory = theoretical_decode_accuracy(L, D, alphabet_size)
            self.assertAlmostEqual(
                measured, theory, delta=0.08, msg=f"L={L}: measured {measured:.3f} vs theory {theory:.3f}"
            )
            self.assertGreaterEqual(theory, measured - 0.02, "the bound should not be pessimistic")

    def test_shift_roles_are_unitary_and_exact(self):
        """The instructor's literal reading (character as a Fourier wave =
        deterministic shift role) must also give exact single-pair unbind."""
        rng = np.random.default_rng(0)
        D = 64
        roles = build_roles(8, D, rng, kind="shift")
        fillers = build_fillers(5, D, rng)
        spec = np.fft.fft(roles[3])
        self.assertTrue(np.allclose(np.abs(spec), 1.0, atol=1e-8))
        bound = np.fft.ifft(spec * np.fft.fft(fillers[2])).real
        recovered = np.fft.ifft(np.fft.fft(bound) * np.conj(spec)).real
        self.assertLess(np.max(np.abs(recovered - fillers[2])), 1e-8)

    def test_decode_accuracy_improves_with_dimension(self):
        rng = np.random.default_rng(0)
        L = 20
        roles_small = build_roles(L, 32, rng)
        fillers_small = build_fillers(10, 32, rng)
        roles_big = build_roles(L, 512, rng)
        fillers_big = build_fillers(10, 512, rng)
        small = sweep_cell(L=L, D=32, roles=roles_small, fillers=fillers_small, n_trials=100, rng=rng)
        big = sweep_cell(L=L, D=512, roles=roles_big, fillers=fillers_big, n_trials=100, rng=rng)
        self.assertGreater(big["accuracy"], small["accuracy"])


if __name__ == "__main__":
    unittest.main()
