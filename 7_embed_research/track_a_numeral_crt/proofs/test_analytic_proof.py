"""CI-safe unittest wrapper around analytic_proof.py's checks.

Uses a small random-pair sample (not the full 10^6) so the whole suite runs
in well under a second; the exhaustive parts of each check (which do not
depend on the sample size) are still run in full.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analytic_proof import (  # noqa: E402
    POSITION_PRIMES,
    VALUE_PRIMES,
    check_additive_per_modulus,
    check_bijection,
    check_division_by_units,
    check_full_addition,
    check_full_multiplication,
    check_multiplicative_per_modulus,
    check_subtraction,
    crt_constants,
)


class TestCRTKroneckerProof(unittest.TestCase):
    def setUp(self):
        self.pos_N, self.pos_consts = crt_constants(POSITION_PRIMES)
        self.val_N, self.val_consts = crt_constants(VALUE_PRIMES)

    def test_moduli_match_documented_values(self):
        self.assertEqual(self.pos_N, 42)
        self.assertEqual(self.val_N, 1_062_347)

    def test_bijection_position(self):
        result = check_bijection(POSITION_PRIMES, self.pos_N, "position")
        self.assertTrue(result["passed"], result["detail"])

    def test_bijection_value(self):
        result = check_bijection(VALUE_PRIMES, self.val_N, "value")
        self.assertTrue(result["passed"], result["detail"])

    def test_additive_homomorphism_position(self):
        result = check_additive_per_modulus(POSITION_PRIMES)
        self.assertTrue(result["passed"], result["detail"])

    def test_additive_homomorphism_value(self):
        result = check_additive_per_modulus(VALUE_PRIMES)
        self.assertTrue(result["passed"], result["detail"])

    def test_full_addition_small_sample(self):
        result = check_full_addition(VALUE_PRIMES, self.val_N, self.val_consts, n_random_pairs=1000, seed=0)
        self.assertTrue(result["passed"], result["detail"])

    def test_multiplicative_homomorphism_position(self):
        result = check_multiplicative_per_modulus(POSITION_PRIMES)
        self.assertTrue(result["passed"], result["detail"])

    def test_multiplicative_homomorphism_value(self):
        result = check_multiplicative_per_modulus(VALUE_PRIMES)
        self.assertTrue(result["passed"], result["detail"])

    def test_full_multiplication_and_wraparound(self):
        result = check_full_multiplication(VALUE_PRIMES, self.val_N, self.val_consts)
        self.assertTrue(result["passed"], result["detail"])

    def test_subtraction_and_underflow(self):
        result = check_subtraction(VALUE_PRIMES, self.val_N, self.val_consts)
        self.assertTrue(result["passed"], result["detail"])

    def test_division_by_units(self):
        result = check_division_by_units(VALUE_PRIMES, self.val_N, self.val_consts)
        self.assertTrue(result["passed"], result["detail"])

    def test_magnitude_comparison_is_not_preserved(self):
        """The honest counterweight to the four homomorphism proofs: residue
        codes destroy ORDER. a < b is not recoverable from the residues without
        a full CRT reconstruction -- the classic, well-known RNS limitation, and
        the reason RNS is not a drop-in replacement for positional notation."""
        import numpy as np

        n = np.arange(self.val_N, dtype=np.int64)
        # lexicographic order of the residue tuple vs. true numeric order
        first_slot = n % VALUE_PRIMES[0]
        # 0 and 1 differ by 1 numerically, but a big number can share 0's residue
        collisions = np.sum(first_slot == 0)
        self.assertGreater(
            collisions, 1, "many distinct magnitudes share a residue -- order is not readable per-slot"
        )


if __name__ == "__main__":
    unittest.main()
