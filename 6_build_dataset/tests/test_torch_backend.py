# -*- coding: utf-8 -*-
"""Tests for the transformer backend.

Skipped wholesale when torch is absent, so the stdlib submission path never
depends on them being runnable.

The important one is :meth:`TestDocumentMasking.test_packed_matches_unpacked`.
Every packing policy in this system assumes that putting several documents in
one sequence does not change what the model learns from any of them. Under the
n-gram backend that assumption was cheap to satisfy and hard to test
meaningfully. Under a transformer it is a real property of the attention mask,
and it either holds bitwise or the packing story is fiction. That test is the
reason to trust `tdes/packing.py`.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    HAVE_TORCH = False

from tdes.config import PAD_ID, SEP_ID                # noqa: E402
from tdes.masks import materialise_attention_matrix   # noqa: E402


def _sample(tokens, *, loss_mask=None, segment_ids=None, position_ids=None,
            lane="web", index=0):
    n = len(tokens)
    return {
        "tokens": list(tokens),
        "loss_mask": list(loss_mask) if loss_mask else [1] * n,
        "segment_ids": list(segment_ids) if segment_ids else [1] * n,
        "position_ids": list(position_ids) if position_ids else list(range(n)),
        "lane": lane,
        "doc_ids": [f"doc_{index}"],
        "shard_ids": [f"shard_{index}"],
        "sample_index": index,
    }


def _model(vocab=64, **kw):
    from tdes.model_torch import TorchLM
    kw.setdefault("d_model", 32)
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("seq_len", 64)
    kw.setdefault("device", "cpu")     # CPU so CI without a GPU runs this
    kw.setdefault("amp", False)
    return TorchLM(vocab, seed=7, **kw)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class TestDocumentMasking(unittest.TestCase):

    def test_mask_matches_the_existing_oracle(self):
        """The torch mask must equal masks.can_attend, position for position.

        `materialise_attention_matrix` was written for tests and the dashboard.
        Using it as the oracle here means the transformer's masking is checked
        against the same predicate the rest of the system documents.
        """
        from tdes.model_torch import attention_mask
        seg = [1, 1, 1, 2, 2, 0, 0]
        want = materialise_attention_matrix(seg)
        got = attention_mask(seg)
        self.assertEqual(got.shape, (len(seg), len(seg)))
        for i in range(len(seg)):
            for j in range(len(seg)):
                self.assertEqual(bool(got[i, j]), want[i][j],
                                 f"mask disagrees at ({i},{j}) for {seg}")

    # The three documents used by the equivalence checks below.
    DOCS = ([5, 6, 7, 8, 9], [11, 12, 13, 14], [21, 22, 23, 24])

    def _packed(self):
        """One sequence holding DOCS back to back, ids reset per document."""
        toks, segs, pos = [], [], []
        for k, d in enumerate(self.DOCS, start=1):
            toks += d
            segs += [k] * len(d)
            pos += list(range(len(d)))
        return _sample(toks, segment_ids=segs, position_ids=pos)

    def _max_divergence(self, m):
        """Largest |alone - packed| over *every* document, at matching positions.

        Comparing only the first document would be vacuous: a leading document
        can attend to nothing but itself under plain causal masking too, so a
        broken mask would not show up. A mutation test proved exactly that -- see
        :meth:`test_the_equivalence_check_can_actually_fail`. Every document is
        compared, and the offsets are where the teeth are.
        """
        packed = m.loss_batch([self._packed()], backward=False)[0]
        by_pos = {t["pos"]: t["loss"] for t in packed["per_token"]}
        worst, compared = 0.0, 0
        off = 0
        for d in self.DOCS:
            alone = m.loss_batch([_sample(d, segment_ids=[1] * len(d),
                                          position_ids=list(range(len(d))))],
                                 backward=False)[0]
            for t in alone["per_token"]:
                p = t["pos"] + off
                if p in by_pos:
                    worst = max(worst, abs(t["loss"] - by_pos[p]))
                    compared += 1
            off += len(d)
        return worst, compared

    def test_packed_matches_unpacked(self):
        """THE GATE.

        Every document's per-token losses must be identical whether it sits
        alone in a sequence or packed beside two others. If this fails, packing
        changes what the model learns and every utilisation number in the
        submission is measuring the wrong thing.
        """
        m = _model()
        worst, compared = self._max_divergence(m)
        self.assertGreaterEqual(compared, 6, "too few positions compared")
        self.assertLess(worst, 1e-6,
                        f"packing changed the loss by up to {worst:.3e}")

    def test_the_equivalence_check_can_actually_fail(self):
        """Anti-vacuity: break the mask and the gate must notice.

        Written because the first version of the gate passed even with document
        masking removed entirely. A test that cannot fail is not evidence, so the
        ability to fail is asserted here rather than assumed.
        """
        from tdes.model_torch import TorchLM
        m = _model()
        original = TorchLM._batch_mask
        try:
            def causal_only(self, segs):
                b, t = segs.shape
                c = torch.tril(torch.ones(t, t, dtype=torch.bool,
                                          device=segs.device))
                return c.unsqueeze(0).expand(b, t, t).unsqueeze(1)

            TorchLM._batch_mask = causal_only
            worst, compared = self._max_divergence(m)
        finally:
            TorchLM._batch_mask = original
        self.assertGreater(compared, 0)
        self.assertGreater(worst, 1e-4,
                           "removing document masking changed nothing, so the "
                           "gate proves nothing")
        # And the real mask must still be correct afterwards.
        self.assertLess(self._max_divergence(m)[0], 1e-6)

    def test_padding_does_not_produce_nan(self):
        """A padding row is fully masked; softmax over all -inf is NaN. The
        diagonal guard must keep a real position's loss finite."""
        m = _model()
        s = _sample([5, 6, 7, PAD_ID, PAD_ID],
                    loss_mask=[1, 1, 1, 0, 0],
                    segment_ids=[1, 1, 1, 0, 0],
                    position_ids=[0, 1, 2, 0, 0])
        r = m.loss_batch([s], backward=False)[0]
        self.assertGreater(r["n_tokens"], 0)
        self.assertTrue(math.isfinite(r["sum_loss"]), r["sum_loss"])
        for t in r["per_token"]:
            self.assertTrue(math.isfinite(t["loss"]))


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class TestLossSemantics(unittest.TestCase):

    def test_initial_loss_is_ln_v(self):
        """Same anchor as the stdlib backend: an uninformed model scores ln(V)."""
        vocab = 64
        m = _model(vocab)
        s = _sample([3, 9, 15, 21, 27, 33], segment_ids=[1] * 6)
        r = m.loss_batch([s], backward=False)[0]
        self.assertAlmostEqual(r["mean_loss"], math.log(vocab), delta=0.35,
                               msg=f"got {r['mean_loss']}, ln(V)={math.log(vocab)}")
        self.assertAlmostEqual(m.initial_loss(), math.log(vocab), places=10)

    def test_masked_positions_bear_no_gradient(self):
        """A prompt or tool observation is context, never a target. Scoring only
        masked-out positions must leave the gradient exactly zero."""
        m = _model()
        s = _sample([5, 6, 7, 8, 9], loss_mask=[0, 0, 0, 0, 0])
        m.zero_grad()
        r = m.loss_batch([s], backward=True)[0]
        self.assertEqual(r["n_tokens"], 0)
        self.assertEqual(r["sum_loss"], 0.0)
        total = sum(float(p.grad.abs().sum()) for _, p in m.named_parameters()
                    if p.grad is not None)
        self.assertEqual(total, 0.0, "context-only positions produced a gradient")

    def test_loss_mask_changes_the_gradient(self):
        """The complement of the above: unmasking must produce one."""
        m = _model()
        s = _sample([5, 6, 7, 8, 9], loss_mask=[1, 1, 1, 1, 1])
        m.zero_grad()
        m.loss_batch([s], backward=True)
        total = sum(float(p.grad.abs().sum()) for _, p in m.named_parameters()
                    if p.grad is not None)
        self.assertGreater(total, 0.0)

    def test_batching_matches_one_at_a_time(self):
        """loss_batch over N samples must equal N single-sample calls. The
        trainer relies on this to hand the model a whole step at once."""
        m = _model()
        a = _sample([5, 6, 7, 8], index=0)
        b = _sample([9, 10, 11, 12, 13], index=1)
        c = _sample([14, 15, 16], index=2)
        together = m.loss_batch([a, b, c], backward=False)
        apart = [m.loss_batch([s], backward=False)[0] for s in (a, b, c)]
        for i, (x, y) in enumerate(zip(together, apart)):
            self.assertEqual(x["n_tokens"], y["n_tokens"], f"sample {i}")
            self.assertAlmostEqual(x["sum_loss"], y["sum_loss"], places=5,
                                   msg=f"sample {i} differs when batched")

    def test_position_policy_is_load_bearing(self):
        """reset_per_document vs continuous must actually change the model.

        Under the n-gram backend this policy was recorded in every batch and
        changed nothing. Here it indexes the positional embedding, so recording
        it finally means something.
        """
        m = _model()
        toks = [5, 6, 7, 11, 12, 13]
        segs = [1, 1, 1, 2, 2, 2]
        reset = _sample(toks, segment_ids=segs, position_ids=[0, 1, 2, 0, 1, 2])
        cont = _sample(toks, segment_ids=segs, position_ids=[0, 1, 2, 3, 4, 5])
        lr = m.loss_batch([reset], backward=False)[0]["sum_loss"]
        lc = m.loss_batch([cont], backward=False)[0]["sum_loss"]
        self.assertNotAlmostEqual(lr, lc, places=6)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class TestGradientsAndState(unittest.TestCase):

    def test_gradcheck(self):
        """Autograd against finite differences in float64.

        The stdlib backend asserts hand-written backprop matches finite
        differences to ~1e-9. This is the same claim for this backend: the
        gradients are real, not plausible-looking.
        """
        from tdes.model_torch import TorchLM
        torch.manual_seed(0)
        m = TorchLM(11, d_model=8, n_layers=1, n_heads=2, seq_len=8,
                    device="cpu", amp=False, seed=1).double()
        tok = torch.tensor([[2, 3, 4, 5]])
        seg = torch.tensor([[1, 1, 1, 1]])
        pid = torch.tensor([[0, 1, 2, 3]])
        emb = m.tok(tok).detach().clone().requires_grad_(True)

        def f(e):
            x = e + m.pos(pid)
            mask = m._batch_mask(seg)
            for blk in m.blocks:
                x = blk(x, mask)
            return m.head(m.ln_f(x)).sum()

        self.assertTrue(torch.autograd.gradcheck(f, (emb,), eps=1e-6,
                                                 atol=1e-7, rtol=1e-4))

    def test_step_reports_real_grad_norm_and_clipping(self):
        m = _model()
        m.zero_grad()
        m.loss_batch([_sample([5, 6, 7, 8, 9])], backward=True)
        before = [p.detach().clone() for p in m.parameters()]
        out = m.step(0.1, momentum=0.9, clip=1.0)
        self.assertGreater(out["grad_norm"], 0.0)
        self.assertIn("clipped", out)
        moved = any(not torch.equal(b, p)
                    for b, p in zip(before, m.parameters()))
        self.assertTrue(moved, "step did not change any parameter")

    def test_grad_vector_projection_is_stable_and_nonzero(self):
        m = _model()
        m.zero_grad()
        m.loss_batch([_sample([5, 6, 7, 8, 9])], backward=True)
        v1 = m.grad_vector(64)
        v2 = m.grad_vector(64)
        self.assertEqual(len(v1), 64)
        self.assertEqual(v1, v2, "projection is not a pure function of the grads")
        self.assertGreater(sum(abs(x) for x in v1), 0.0)

    def test_state_roundtrip_restores_losses_and_momentum(self):
        m = _model()
        m.zero_grad()
        m.loss_batch([_sample([5, 6, 7, 8, 9])], backward=True)
        m.step(0.1)
        blob = m.state_dict_for_checkpoint()
        probe = _sample([5, 6, 7, 8, 9])
        want = m.loss_batch([probe], backward=False)[0]["sum_loss"]

        m2 = _model()
        self.assertNotAlmostEqual(
            m2.loss_batch([probe], backward=False)[0]["sum_loss"], want, places=6)
        m2.load_state_dict_from_checkpoint(blob)
        self.assertAlmostEqual(
            m2.loss_batch([probe], backward=False)[0]["sum_loss"], want, places=9)
        self.assertEqual(sorted(m2._momentum_buf), sorted(m._momentum_buf))

    def test_shape_mismatch_is_refused(self):
        m = _model(vocab=64)
        blob = m.state_dict_for_checkpoint()
        other = _model(vocab=64, d_model=64)
        with self.assertRaises(ValueError):
            other.load_state_dict_from_checkpoint(blob)

    def test_run_to_run_losses_are_identical(self):
        """Two freshly seeded models must agree exactly."""
        probe = _sample([5, 6, 7, 8, 9, 10, 11])
        a = _model().loss_batch([probe], backward=False)[0]
        b = _model().loss_batch([probe], backward=False)[0]
        self.assertEqual(a["sum_loss"], b["sum_loss"])
        self.assertEqual([t["loss"] for t in a["per_token"]],
                         [t["loss"] for t in b["per_token"]])


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class TestSeamConformance(unittest.TestCase):

    def test_satisfies_the_protocol(self):
        from tdes.lm import LanguageModel
        self.assertIsInstance(_model(), LanguageModel)

    def test_both_backends_agree_on_the_result_shape(self):
        """The ledger is built on this shape. If the two backends disagree, the
        learning ledger silently means different things per backend."""
        from tdes.model import NeuralLM
        s = _sample([5, 6, 7, 8, 9])
        t_out = _model(vocab=64).loss_batch([s], backward=False)[0]
        n_out = NeuralLM(64, 8, 16, 4, seed="t").loss_batch([s], backward=False)[0]
        self.assertEqual(set(t_out), set(n_out))
        self.assertEqual(t_out["n_tokens"], n_out["n_tokens"])
        self.assertEqual([t["pos"] for t in t_out["per_token"]],
                         [t["pos"] for t in n_out["per_token"]])
        self.assertEqual([t["token_id"] for t in t_out["per_token"]],
                         [t["token_id"] for t in n_out["per_token"]])

    def test_dropout_is_refused_rather_than_ignored(self):
        from tdes.model_torch import TorchLM
        with self.assertRaises(NotImplementedError):
            TorchLM(32, d_model=16, n_layers=1, n_heads=2, seq_len=16,
                    dropout=0.1, device="cpu", amp=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
