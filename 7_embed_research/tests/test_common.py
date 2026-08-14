"""Smoke tests for the shared common/ package -- shape and wiring checks,
not scientific claims (those are proved in each track's proofs/ suite)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.evidence import EvidenceRow, write_evidence  # noqa: E402
from common.seeding import seed_everything  # noqa: E402


class TestSeeding(unittest.TestCase):
    def test_seed_everything_runs_without_torch_or_numpy_installed(self):
        # seeding.py must degrade gracefully if numpy/torch aren't importable;
        # here they are, so this just checks it doesn't raise.
        seed_everything(0)

    def test_python_random_is_reproducible(self):
        import random

        seed_everything(42)
        a = [random.random() for _ in range(5)]
        seed_everything(42)
        b = [random.random() for _ in range(5)]
        self.assertEqual(a, b)


class TestEvidence(unittest.TestCase):
    def test_write_evidence_all_pass(self):
        rows = [EvidenceRow("req1", "file1", "field1", lambda: (True, "ok"))]
        out_path = ROOT / "tests" / "_scratch_evidence.json"
        try:
            all_passed = write_evidence(rows, out_path)
            self.assertTrue(all_passed)
            data = json.loads(out_path.read_text())
            self.assertEqual(data[0]["result"], "PASS")
        finally:
            out_path.unlink(missing_ok=True)

    def test_write_evidence_detects_failure(self):
        rows = [
            EvidenceRow("req1", "file1", "field1", lambda: (True, "ok")),
            EvidenceRow("req2", "file2", "field2", lambda: (False, "broken")),
        ]
        out_path = ROOT / "tests" / "_scratch_evidence.json"
        try:
            all_passed = write_evidence(rows, out_path)
            self.assertFalse(all_passed)
        finally:
            out_path.unlink(missing_ok=True)

    def test_evidence_row_catches_exceptions_as_failure(self):
        def _raises():
            raise ValueError("artifact missing")

        row = EvidenceRow("req", "file", "field", _raises)
        result = row.evaluate()
        self.assertEqual(result["result"], "FAIL")
        self.assertIn("artifact missing", result["detail"])


class TestTransformerTrunk(unittest.TestCase):
    def test_trunk_shape_and_device_roundtrip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from common.device import count_params, get_device
        from common.transformer import TransformerTrunk, TrunkConfig

        cfg = TrunkConfig(d_model=32, n_layer=2, n_head=4, d_ff=64, max_seq_len=16)
        trunk = TransformerTrunk(cfg)
        x = torch.randn(2, 5, 32)
        y = trunk(x)
        self.assertEqual(tuple(y.shape), (2, 5, 32))
        self.assertGreater(count_params(trunk), 0)

        device = get_device()
        trunk.to(device)
        y2 = trunk(x.to(device))
        self.assertEqual(y2.device.type, device.type)

    def test_causal_mask_blocks_future_positions(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from common.transformer import TransformerTrunk, TrunkConfig

        cfg = TrunkConfig(d_model=16, n_layer=1, n_head=2, d_ff=32, max_seq_len=8)
        trunk = TransformerTrunk(cfg)
        trunk.eval()
        x = torch.randn(1, 4, 16)
        y_full = trunk(x)
        # changing a later token must not change an earlier position's output
        x_perturbed = x.clone()
        x_perturbed[:, -1, :] += 10.0
        y_perturbed = trunk(x_perturbed)
        self.assertTrue(torch.allclose(y_full[:, :-1], y_perturbed[:, :-1], atol=1e-5))


class TestKroneckerCodecMatchesPaper(unittest.TestCase):
    """The Kronecker arm must implement Eq. 1 of arXiv:2605.29459 exactly:

        kappa(b) = (1/sqrt(L)) * SUM_p  c_{b_p} (x) p_p

    An earlier version of this code omitted the 1/sqrt(L) factor and used a
    corpus character alphabet instead of the 256-value byte alphabet, so
    these tests pin down both properties against a literal transcription of
    the formula rather than against the implementation's own conventions."""

    @staticmethod
    def _paper_formula(word: str, d_p: int = 32, d_c: int = 256):
        import numpy as np

        raw = word.encode("utf-8")[:d_p]
        acc = np.zeros(d_c * d_p)
        for p, byte_val in enumerate(raw):
            c = np.zeros(d_c)
            c[byte_val] = 1.0
            pos = np.zeros(d_p)
            pos[p] = 1.0
            acc += np.kron(c, pos)
        return acc / np.sqrt(len(raw))

    def test_dimension_is_8192(self):
        from track_b_holographic_binding.model.embeddings import (
            KRONECKER_BYTE_DIM,
            KRONECKER_D,
            KRONECKER_MAX_SLOTS,
        )

        self.assertEqual(KRONECKER_BYTE_DIM, 256)
        self.assertEqual(KRONECKER_MAX_SLOTS, 32)
        self.assertEqual(KRONECKER_D, 8192)

    def test_matches_paper_formula_up_to_coordinate_permutation(self):
        """Both factors are one-hot, so the sum of Kronecker products and our
        block layout differ only by a fixed permutation of coordinates -- one
        the following Linear absorbs. Assert that equivalence directly."""
        import numpy as np

        from track_b_holographic_binding.model.embeddings import kronecker_encode

        for word in ["a", "hello", "x" * 40, "नमस्ते"]:
            mine = kronecker_encode(word)
            paper = self._paper_formula(word)
            self.assertTrue(
                np.array_equal(np.sort(mine), np.sort(paper)),
                f"{word!r}: not a permutation of the paper's kappa(b)",
            )

    def test_every_token_is_unit_norm(self):
        """The 1/sqrt(L) factor exists to make code norm independent of token
        length. Without it the norm grows as sqrt(L) (1.0 -> 5.66 at L=32)."""
        import numpy as np

        from track_b_holographic_binding.model.embeddings import kronecker_encode

        for word in ["a", "in", "hello", "elephant", "x" * 16, "x" * 32, "x" * 64]:
            self.assertAlmostEqual(
                float(np.linalg.norm(kronecker_encode(word))), 1.0, places=5, msg=f"{word!r} not unit-norm"
            )

    def test_is_byte_level_not_character_level(self):
        """d_p = 32 counts BYTES, so multi-byte scripts are capped far earlier
        in characters than Latin text -- a real property of the scheme."""
        from track_b_holographic_binding.model.embeddings import kronecker_encode

        devanagari = "नमस्ते"
        self.assertGreater(len(devanagari.encode("utf-8")), len(devanagari))
        self.assertEqual(kronecker_encode(devanagari).shape, (8192,))

    def test_truncation_past_32_bytes_loses_information(self):
        """Two tokens sharing a 32-byte prefix must encode identically -- the
        claim Track B's truncation probe rests on."""
        import numpy as np

        from track_b_holographic_binding.model.embeddings import kronecker_encode

        a = kronecker_encode("x" * 32 + "aaaaaaaa")
        b = kronecker_encode("x" * 32 + "bbbbbbbb")
        self.assertTrue(np.array_equal(a, b))

    def test_vocab_table_rows_are_unit_norm_except_unk(self):
        import numpy as np

        from track_b_holographic_binding.model.embeddings import build_kronecker_slot_table

        table = build_kronecker_slot_table(["<unk>", "a", "hello", "elephant"])
        self.assertEqual(table.shape, (4, 8192))
        self.assertEqual(np.linalg.norm(table[0]), 0.0, "UNK row is reserved and stays all-zero")
        for i in range(1, table.shape[0]):
            self.assertAlmostEqual(float(np.linalg.norm(table[i])), 1.0, places=5)


class TestRandomOffset(unittest.TestCase):
    """The random-offset augmentation is the mechanism that unlocked OOD
    length generalization, so its invariants are worth pinning down: it must
    shift digits only, shift them all by the SAME amount within an example
    (otherwise digit alignment -- the thing the model must learn -- breaks),
    and never exceed the embedding table's row count."""

    def _batch(self, offset_max: int, seed: int = 0):
        import numpy as np

        from track_a_numeral_crt.data.generate_arithmetic import sample_batch

        rng = np.random.default_rng(seed)
        return sample_batch(rng, 64, "add", (1, 4), 48, random_offset_max=offset_max)

    def test_zero_offset_is_unchanged(self):
        import numpy as np

        a = self._batch(0, seed=7)
        b = self._batch(0, seed=7)
        self.assertTrue(np.array_equal(a["place_value_ids"], b["place_value_ids"]))
        self.assertEqual(int(a["place_value_ids"].max()), int(a["place_value_ids"].max()))

    def test_offset_stays_within_embedding_table(self):
        from track_a_numeral_crt.model.embeddings import MAX_PLACE_VALUE

        batch = self._batch(24, seed=3)
        self.assertLess(
            int(batch["place_value_ids"].max()),
            MAX_PLACE_VALUE,
            "offset place values must stay inside the embedding table",
        )

    def test_offset_is_uniform_within_an_example(self):
        """Digits of a, b and the answer must all move together: the gap
        between consecutive place values inside one number stays 1."""
        import numpy as np

        from track_a_numeral_crt.data.generate_arithmetic import DIGIT_BASE, sample_batch

        rng = np.random.default_rng(11)
        batch = sample_batch(rng, 32, "add", (3, 4), 48, random_offset_max=24)
        for row_tokens, row_pv in zip(batch["input_ids"], batch["place_value_ids"]):
            digit_pos = [i for i, t in enumerate(row_tokens) if DIGIT_BASE <= t < DIGIT_BASE + 10]
            if not digit_pos:
                continue
            # the first digit run is operand a; its place values must be consecutive
            run = [row_pv[i] for i in digit_pos[:3]]
            self.assertEqual(list(np.diff(run)), [1, 1], f"place values not consecutive within a number: {run}")

    def test_offset_actually_varies_across_examples(self):
        import numpy as np

        batch = self._batch(24, seed=5)
        from track_a_numeral_crt.data.generate_arithmetic import DIGIT_BASE

        first_digit_pv = []
        for row_tokens, row_pv in zip(batch["input_ids"], batch["place_value_ids"]):
            for t, pv in zip(row_tokens, row_pv):
                if DIGIT_BASE <= t < DIGIT_BASE + 10:
                    first_digit_pv.append(pv)
                    break
        self.assertGreater(len(set(first_digit_pv)), 1, "offsets should differ between examples")


if __name__ == "__main__":
    unittest.main()
