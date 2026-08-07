# -*- coding: utf-8 -*-
"""Invariant tests. Stdlib ``unittest`` -- no pytest, no pip install.

    python -m unittest discover -s tests -v

These assert the properties the system claims, and several of them exist
because the property broke during development and the test is what would have
caught it:

* ``test_structure_preserving_never_merges`` -- an earlier version truncated
  instead of chunking and dropped 81% of the agentic lane.
* ``test_replay_needs_roles_to_match`` -- role spans were missing from the
  ledger, so replayed SFT samples silently lost their prompt masking. The
  batch id matched; only the content hash caught it.
* ``test_apportioner_serves_small_lanes`` -- stateless apportionment starved
  every lane under one sample per step, taking a protected floor to zero.
* ``test_pii_patterns_do_not_match_year_ranges`` -- the Aadhaar pattern matched
  ``2007-2008 2008-2009`` in a Hindi finance table.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tdes import pii                                                  # noqa: E402
from tdes.batching import (LaneSequencePool, assemble_batch, build_sample,  # noqa: E402
                           compute_batch_id, compute_content_hash,
                           verify_rank_partition)
from tdes.config import (EOS_ID, PAD_ID, RunConfig, STAGES, get_profile)   # noqa: E402
from tdes.dedup import deduplicate, minhash                           # noqa: E402
from tdes.determinism import (DeterministicRNG, canonical_text,       # noqa: E402
                              stable_shuffle)
from tdes.firewall import EvalFirewallViolation, EvalRegistry         # noqa: E402
from tdes.hashing import hash_many, hash_obj, sha256_text             # noqa: E402
from tdes.ledger.consumption import verify_integrity                  # noqa: E402
from tdes.ledger.learning import classify                             # noqa: E402
from tdes.manifest import validate                                    # noqa: E402
from tdes.masks import build_masks, can_attend                        # noqa: E402
from tdes.mixture import LaneApportioner, _floor_window               # noqa: E402
from tdes.model import NeuralLM                                       # noqa: E402
from tdes.packing import compare_policies, pack, utilization          # noqa: E402
from tdes.scarcity import effective_tokens, resolve                   # noqa: E402


def make_item(shard, doc, start, end, roles=None, atomic=False):
    return {"shard_id": shard, "doc_id": doc, "start": start, "end": end,
            "n_tokens": end - start, "token_roles": roles or [],
            "atomic_unit": atomic}


# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):

    def test_canonical_text_collapses_line_endings(self):
        # The whole cross-platform hash story rests on this.
        self.assertEqual(sha256_text("a\r\nb"), sha256_text("a\nb"))
        self.assertEqual(sha256_text("a\rb"), sha256_text("a\nb"))

    def test_canonical_text_normalises_unicode(self):
        import unicodedata
        s = "क्षि"
        self.assertEqual(sha256_text(unicodedata.normalize("NFD", s)),
                         sha256_text(unicodedata.normalize("NFC", s)))

    def test_no_python_hash_dependence(self):
        """Ordering must not come from hash(), which is salted per process."""
        xs = ["indic", "code", "web", "agentic", "longctx"]
        a = stable_shuffle(xs, seed="s")
        b = stable_shuffle(list(reversed(xs)), seed="s")
        self.assertEqual(sorted(a), sorted(b))
        self.assertEqual(a, stable_shuffle(xs, seed="s"))
        self.assertNotEqual(a, stable_shuffle(xs, seed="other"))

    def test_hash_many_is_prefix_unambiguous(self):
        self.assertNotEqual(hash_many(["ab", "c"]), hash_many(["a", "bc"]))

    def test_canonical_json_is_order_independent(self):
        self.assertEqual(hash_obj({"a": 1, "b": 2}), hash_obj({"b": 2, "a": 1}))

    def test_rng_state_round_trips_exactly(self):
        r = DeterministicRNG("x")
        [r.random() for _ in range(5)]
        st = r.state()
        a = [r.random() for _ in range(4)]
        # One restored generator advanced four times -- not four fresh ones,
        # which would each replay the same first draw.
        restored = DeterministicRNG.from_state(st)
        b = [restored.random() for _ in range(4)]
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
class TestDedupAndPII(unittest.TestCase):

    def test_exact_duplicates_removed_deterministically(self):
        docs = [{"doc_id": f"d{i}", "text": t, "content_sha256": sha256_text(t)}
                for i, t in enumerate(["hello world " * 20, "hello world " * 20,
                                       "totally different content " * 20])]
        out = deduplicate(docs)
        self.assertEqual(out["report"]["exact_duplicates_removed"], 1)
        self.assertEqual(len(out["documents"]), 2)
        self.assertEqual([d["doc_id"] for d in out["documents"]],
                         [d["doc_id"] for d in deduplicate(docs)["documents"]])

    def test_minhash_is_process_independent(self):
        self.assertEqual(minhash("the quick brown fox " * 10),
                         minhash("the quick brown fox " * 10))

    def test_pii_patterns_fire_on_real_shapes(self):
        cases = {
            "email": "write to ravi.kumar@example.co.in today",
            "aadhaar_shaped": "id is 4321 8765 2109 on file",
            "pan_shaped": "PAN ABCDE1234F issued",
            "phone_in": "call +91 9876543210 now",
            "ipv4": "host 192.168.10.44 responded",
        }
        for cat, text in cases.items():
            with self.subTest(cat=cat):
                self.assertIn(cat, pii.scan(text), f"{cat} not detected in {text!r}")
                redacted, counts = pii.redact(text)
                self.assertIn(cat, counts)
                self.assertIn("<redacted:", redacted)

    def test_pii_patterns_do_not_match_year_ranges(self):
        """Regression: a Hindi finance table matched the Aadhaar pattern."""
        self.assertNotIn("aadhaar_shaped",
                         pii.scan("वित्त वर्ष 2007-2008 2008-2009 2009-2010 2011-2012"))


# ---------------------------------------------------------------------------
class TestPacking(unittest.TestCase):

    def setUp(self):
        self.items = [make_item("s0", f"d{i}", i * 100, i * 100 + n)
                      for i, n in enumerate([40, 90, 25, 150, 60, 30])]

    def test_all_policies_produce_full_coverage(self):
        rep = compare_policies(self.items, 64)
        for name in ("concat_chop", "greedy", "best_fit",
                     "structure_preserving", "pad_only"):
            with self.subTest(policy=name):
                self.assertGreaterEqual(rep["by_policy"][name]["coverage"], 0.999,
                                        f"{name} dropped input tokens")

    def test_structure_preserving_never_merges(self):
        """Two documents must never share one sequence. Long ones may span several."""
        for s in pack(self.items, 64, "structure_preserving"):
            self.assertEqual(len({g["doc_id"] for g in s["segments"]}), 1)

    def test_concat_chop_does_merge(self):
        """The contrast: concat_chop is expected to pack documents together."""
        seqs = pack(self.items, 64, "concat_chop")
        self.assertTrue(any(len({g["doc_id"] for g in s["segments"]}) > 1 for s in seqs))

    def test_utilization_arithmetic_is_reconstructible(self):
        for name in ("concat_chop", "best_fit"):
            seqs = pack(self.items, 64, name)
            u = utilization(seqs, sum(i["end"] - i["start"] for i in self.items))
            self.assertEqual(u["total_positions"], u["real_tokens"] + u["pad_tokens"])
            self.assertEqual(u["total_positions"], u["sequences"] * 64)
            self.assertAlmostEqual(u["utilization"],
                                   u["real_tokens"] / u["total_positions"], places=6)

    def test_long_context_skips_short_documents(self):
        seqs = pack(self.items, 64, "long_context")
        for s in seqs:
            self.assertEqual(s["real_tokens"], 64)

    def test_packing_is_deterministic(self):
        for name in ("greedy", "best_fit", "concat_chop"):
            a = pack(self.items, 64, name)
            b = pack(self.items, 64, name)
            self.assertEqual([s["segments"] for s in a], [s["segments"] for s in b])


# ---------------------------------------------------------------------------
class TestMasks(unittest.TestCase):

    def _packed(self):
        a = make_item("s0", "docA", 0, 30)
        b = make_item("s0", "docB", 30, 55)
        return pack([a, b], 64, "concat_chop")[0]

    def test_attention_never_crosses_a_document(self):
        seq = self._packed()
        m = build_masks(seq, [7] * 64)
        seg = m["segment_ids"]
        b0 = seq["segments"][1]["seq_start"]
        self.assertFalse(can_attend(seg, b0, b0 - 1), "attended across a boundary")
        self.assertTrue(can_attend(seg, b0 + 1, b0), "blocked inside its own document")

    def test_attention_is_causal(self):
        seq = self._packed()
        seg = build_masks(seq, [7] * 64)["segment_ids"]
        self.assertFalse(can_attend(seg, 3, 9))

    def test_position_ids_reset_per_document(self):
        seq = self._packed()
        m = build_masks(seq, [7] * 64, position_policy="reset_per_document")
        b0 = seq["segments"][1]["seq_start"]
        self.assertEqual(m["position_ids"][b0], 0)
        m2 = build_masks(seq, [7] * 64, position_policy="continuous")
        self.assertEqual(m2["position_ids"][b0], b0)

    def test_loss_mask_excludes_prompt_and_tool_observations(self):
        roles = [{"role": "user", "start": 0, "end": 10},
                 {"role": "assistant", "start": 10, "end": 20},
                 {"role": "tool_obs", "start": 20, "end": 30},
                 {"role": "tool_call", "start": 30, "end": 40}]
        it = make_item("s0", "trace", 0, 40, roles=roles)
        seq = pack([it], 64, "structure_preserving")[0]
        lm = build_masks(seq, [7] * 64)["loss_mask"]
        self.assertEqual(sum(lm[0:10]), 0, "user prompt bore loss")
        self.assertEqual(sum(lm[10:20]), 10, "assistant response did not bear loss")
        self.assertEqual(sum(lm[20:30]), 0, "tool observation bore loss")
        # 9, not 10: position 39 is the document's last token, so the token it
        # would predict is outside the document. This expectation used to be 10,
        # which encoded the boundary off-by-one that
        # test_no_scored_position_predicts_across_a_document now forbids.
        self.assertEqual(sum(lm[30:39]), 9, "tool call did not bear loss")
        self.assertEqual(lm[39], 0, "last token of the document bore loss")

    def test_padding_never_bears_loss(self):
        it = make_item("s0", "d", 0, 20)
        seq = pack([it], 64, "pad_only")[0]
        toks = [9] * 20 + [PAD_ID] * 44
        m = build_masks(seq, toks)
        self.assertEqual(sum(m["loss_mask"][20:]), 0)
        self.assertEqual(m["loss_mask"][-1], 0, "final position cannot be a target")
        self.assertEqual(m["loss_mask"][19], 0,
                         "the position before padding predicts PAD, not a token")

    def test_no_scored_position_predicts_across_a_document(self):
        """A boundary position must not bear loss.

        Its target is the first token of the *next* document, which the attention
        mask forbids it from seeing -- so training on it is training on noise.
        The packer inserts no separator between documents, so without this rule
        every internal boundary contributes one such position. Found while
        building the transformer backend; the n-gram backend hid it, because a
        fully masked context still yields a loss.
        """
        seq = self._packed()
        m = build_masks(seq, [7] * 64)
        lm, seg = m["loss_mask"], m["segment_ids"]
        b0 = seq["segments"][1]["seq_start"]
        self.assertEqual(lm[b0 - 1], 0, "boundary position bore loss")
        for i in range(63):
            if lm[i] and seg[i] and seg[i + 1] != seg[i]:
                self.fail(f"position {i} (segment {seg[i]}) bears loss but its "
                          f"target is in segment {seg[i + 1]}")

    def test_scored_count_equals_what_a_trainer_would_score(self):
        """``loss_bearing_tokens`` must equal the positions actually scored.

        Before the boundary rule these differed, so every loss-bearing rate in
        performance.json was very slightly overstated (measured: 45 positions of
        55,251). A metric that counts positions no trainer will ever score is
        the kind of number this system exists to refuse.
        """
        seq = self._packed()
        toks = [7] * 55 + [PAD_ID] * 9
        m = build_masks(seq, toks)
        would_score = sum(1 for i in range(63)
                          if m["loss_mask"][i] and toks[i + 1] != PAD_ID)
        self.assertEqual(m["loss_bearing_tokens"], would_score)


# ---------------------------------------------------------------------------
class TestMixture(unittest.TestCase):

    def test_apportioner_serves_small_lanes(self):
        """Regression: a 2% lane at 6 samples/step used to receive nothing."""
        ap = LaneApportioner()
        weights = {"web": 0.62, "code": 0.16, "indic": 0.08, "multiling": 0.05,
                   "reasoning": 0.05, "agentic": 0.02, "longctx": 0.02}
        floors = {"indic": 0.03, "longctx": 0.015, "agentic": 0.01}
        totals: dict[str, int] = {}
        for _ in range(200):
            for lane, k in ap.apportion(weights, 6, floors).items():
                totals[lane] = totals.get(lane, 0) + k
        for lane in weights:
            self.assertGreater(totals.get(lane, 0), 0, f"{lane} was starved")
        self.assertEqual(sum(totals.values()), 200 * 6, "batch not exactly filled")

    def test_apportioner_is_deterministic(self):
        w = {"a": 0.7, "b": 0.29, "c": 0.01}
        r1 = [LaneApportioner().apportion(w, 4) for _ in range(1)]
        ap1, ap2 = LaneApportioner(), LaneApportioner()
        s1 = [ap1.apportion(w, 4) for _ in range(50)]
        s2 = [ap2.apportion(w, 4) for _ in range(50)]
        self.assertEqual(s1, s2)

    def test_floor_window_makes_the_smallest_floor_expressible(self):
        for name in ("fast", "demo", "full"):
            with self.subTest(profile=name):
                p = get_profile(name)
                w = _floor_window(p)
                smallest = min(f for st in STAGES
                               for f in st.protected_floors.values() if f > 0)
                self.assertGreaterEqual(smallest * w * p.samples_per_step, 1.0)

    def test_epoch_cap_and_decay(self):
        d = resolve("indic", 1000, 100, stage_id="A", is_protected=False)
        self.assertEqual(d["policy"], "defer_to_later_stage")
        self.assertLessEqual(d["epochs"], 4.0)
        d2 = resolve("indic", 1000, 100, stage_id="A", is_protected=True)
        self.assertEqual(d2["policy"], "repeat_over_cap")
        self.assertIn("warning", d2)
        # Repetition is discounted: four passes are worth less than four.
        self.assertLess(effective_tokens(100, 4.0), 400.0)
        self.assertGreater(effective_tokens(100, 4.0), 100.0)

    def test_scarcity_within_one_pass_needs_no_policy(self):
        self.assertEqual(resolve("web", 50, 100, stage_id="A")["policy"], "none")


# ---------------------------------------------------------------------------
class TestFirewall(unittest.TestCase):

    def _registry(self):
        reg = EvalRegistry()
        doc = {"doc_id": "eval_1", "never_train": True,
               "text": "the capital of a country is a city " * 12 + " CANARY-XYZ-123",
               "content_sha256": sha256_text("eval-1"), "canary": "CANARY-XYZ-123",
               "lane": "web", "benchmark_id": "b1"}
        reg.register(doc, "eval")
        return reg, doc

    def test_blocks_exact_copy(self):
        reg, doc = self._registry()
        self.assertTrue(reg.check_document(doc)["blocked"])

    def test_blocks_edited_copy_via_canary_and_ngrams(self):
        reg, doc = self._registry()
        edited = dict(doc, never_train=False, content_sha256="0" * 64)
        v = reg.check_document(edited)
        self.assertTrue(v["blocked"])

    def test_admits_unrelated_document(self):
        reg, _ = self._registry()
        clean = {"doc_id": "train_1", "never_train": False,
                 "text": "quantum chromodynamics describes the strong interaction " * 10,
                 "content_sha256": sha256_text("train-1")}
        self.assertFalse(reg.check_document(clean)["blocked"])

    def test_gradient_bearing_access_raises(self):
        reg, doc = self._registry()
        with self.assertRaises(EvalFirewallViolation):
            reg.assert_not_gradient_bearing([doc["doc_id"]], where="test")

    def test_validation_reads_are_never_gradient_bearing(self):
        reg = EvalRegistry()
        doc = {"doc_id": "val_1", "never_train": True, "text": "x " * 40,
               "content_sha256": sha256_text("v"), "canary": None, "lane": "web"}
        reg.register(doc, "validation")
        reg.note_access(["val_1"], purpose="validation_probe", step=3)
        s = reg.summary()
        self.assertEqual(s["gradient_bearing_reads"], 0)
        self.assertEqual(s["reads_by_split"]["validation"], 1)


# ---------------------------------------------------------------------------
class TestManifestGate(unittest.TestCase):

    def _manifest(self):
        m = {"shard_id": "web_0000", "content_sha256": "a" * 64,
             "tokenizer_hash": "b" * 64, "cleaning_pipeline_hash": "c" * 64,
             "dedup_status": {"status": "DEDUPLICATED"},
             "eval_overlap_status": "CLEAN", "contamination_status": "CLEAN",
             "pii_status": {"status": "SCREENED"}, "license": "CC-BY-SA-4.0",
             "capability_lane": "web", "token_count": 4000, "never_train": False}
        m["manifest_sha256"] = hash_obj(m)
        return m

    def test_clean_manifest_is_admitted(self):
        self.assertTrue(validate(self._manifest())[0])

    def test_gate_rejects_each_defect(self):
        for field, value in [("tokenizer_hash", ""),
                             ("eval_overlap_status", "OVERLAP"),
                             ("contamination_status", "SUSPECT"),
                             ("license", "proprietary"),
                             ("token_count", 0),
                             ("never_train", True)]:
            with self.subTest(field=field):
                m = self._manifest()
                m[field] = value
                m["manifest_sha256"] = hash_obj(
                    {k: v for k, v in m.items() if k != "manifest_sha256"})
                ok, reasons = validate(m)
                self.assertFalse(ok, f"{field}={value!r} was admitted")
                self.assertTrue(reasons)

    def test_gate_detects_edit_after_sealing(self):
        m = self._manifest()
        m["token_count"] = 999999          # edited without re-sealing
        ok, reasons = validate(m)
        self.assertFalse(ok)
        self.assertTrue(any("manifest_sha256" in r for r in reasons))


# ---------------------------------------------------------------------------
class TestBatchIdentity(unittest.TestCase):

    def _batch(self, seed_token=7):
        cfg = RunConfig(profile=get_profile("fast"))
        items = [make_item("s0", f"d{i}", i * 80, i * 80 + 60) for i in range(4)]
        toks = {"s0": [seed_token] * 4000}
        seqs = pack(items, 64, "concat_chop")[:4]
        samples = [build_sample(s, toks, seq_len=64, attention_policy="document_causal",
                                position_policy="reset_per_document", lane="web",
                                sample_index=i) for i, s in enumerate(seqs)]
        plan = {"global_step": 5, "stage": "A", "sequence_length": 64,
                "attention_policy": "document_causal",
                "position_policy": "reset_per_document"}
        return cfg, assemble_batch(cfg, plan, samples, branch_id="main"), samples

    def test_batch_id_is_deterministic(self):
        _, b1, _ = self._batch()
        _, b2, _ = self._batch()
        self.assertEqual(b1["batch_id"], b2["batch_id"])
        self.assertEqual(b1["batch_content_hash"], b2["batch_content_hash"])

    def test_content_hash_detects_changed_tokens_when_spans_are_identical(self):
        """The reason two hashes exist rather than one."""
        _, b1, _ = self._batch(seed_token=7)
        _, b2, _ = self._batch(seed_token=9)
        self.assertEqual(b1["batch_id"], b2["batch_id"], "spans should be identical")
        self.assertNotEqual(b1["batch_content_hash"], b2["batch_content_hash"],
                            "different tokens produced the same content hash")

    def test_rank_partition_is_disjoint_and_complete(self):
        cfg, b, samples = self._batch()
        ok, problems = verify_rank_partition(b, cfg.profile.ranks)
        self.assertTrue(ok, problems)
        served = [i for mb in b["microbatches"] for i in mb["sample_indices"]]
        self.assertEqual(sorted(served), sorted(s["sample_index"] for s in samples))
        self.assertEqual(len(served), len(set(served)), "a sample was served twice")

    def test_replay_needs_roles_to_match(self):
        """Regression: roles missing from a span changed the loss mask."""
        roles = [{"role": "user", "start": 0, "end": 20},
                 {"role": "assistant", "start": 20, "end": 50}]
        it = make_item("s0", "trace", 0, 50, roles=roles)
        toks = {"s0": [7] * 4000}
        seq = pack([it], 64, "structure_preserving")[0]
        with_roles = build_sample(seq, toks, seq_len=64,
                                  attention_policy="document_causal",
                                  position_policy="reset_per_document",
                                  lane="agentic", sample_index=0)
        stripped = {**seq, "segments": [{**g, "roles": []} for g in seq["segments"]]}
        without = build_sample(stripped, toks, seq_len=64,
                               attention_policy="document_causal",
                               position_policy="reset_per_document",
                               lane="agentic", sample_index=0)
        self.assertNotEqual(compute_content_hash([with_roles]),
                            compute_content_hash([without]))
        self.assertEqual(compute_batch_id("r", "main", 0, [with_roles]),
                         compute_batch_id("r", "main", 0, [without]),
                         "spans are the same, so batch_id must be too")


# ---------------------------------------------------------------------------
class TestLedgerIntegrity(unittest.TestCase):

    def _rows(self, steps=6, ranks=2):
        rows, off = [], 0
        for s in range(steps):
            for r in range(ranks):
                rows.append({"ledger_offset": off, "branch_id": "main",
                             "global_step": s, "rank": r, "accum_index": 0,
                             "batch_id": f"b{s}"})
                off += 1
        return rows

    def test_clean_ledger_passes(self):
        self.assertTrue(verify_integrity(self._rows())["ok"])

    def test_detects_skipped_step(self):
        rows = [r for r in self._rows() if r["global_step"] != 3]
        for i, r in enumerate(rows):
            r["ledger_offset"] = i
        out = verify_integrity(rows)
        self.assertFalse(out["ok"])
        self.assertFalse(out["no_skipped_steps"])

    def test_detects_repeated_batch(self):
        rows = self._rows()
        rows[-1]["batch_id"] = "b0"          # a batch reused at a later step
        out = verify_integrity(rows)
        self.assertFalse(out["ok"])
        self.assertFalse(out["no_repeated_batches"])

    def test_detects_offset_gap(self):
        rows = self._rows()
        rows[4]["ledger_offset"] = 99
        out = verify_integrity(rows)
        self.assertFalse(out["ok"])
        self.assertFalse(out["offsets_dense"])


# ---------------------------------------------------------------------------
class TestModel(unittest.TestCase):

    def test_initial_loss_is_ln_vocab(self):
        import math
        import statistics
        m = NeuralLM(256, 8, 16, 3, seed="t")
        toks = [10 + i % 200 for i in range(40)]
        seg = [1] * 40
        losses = [m.forward_loss(m.build_context(toks, seg, i), toks[i])[0]
                  for i in range(3, 40)]
        self.assertAlmostEqual(statistics.mean(losses), math.log(256), delta=0.25)

    def test_gradients_match_finite_differences(self):
        """Proof the backprop is real rather than plausible-looking."""
        m = NeuralLM(48, 4, 8, 3, seed="g")
        toks = [3, 7, 11, 5, 9, 2, 8]
        ctx = m.build_context(toks, [1] * 7, 4)
        m.zero_grad()
        _, cache = m.forward_loss(ctx, toks[4])
        m.backward(cache)
        eps = 1e-6
        for o, j in [(5, 2), (11, 0), (30, 5)]:
            with self.subTest(w=(o, j)):
                orig = m.W2[o][j]
                m.W2[o][j] = orig + eps
                lp, _ = m.forward_loss(ctx, toks[4])
                m.W2[o][j] = orig - eps
                lm, _ = m.forward_loss(ctx, toks[4])
                m.W2[o][j] = orig
                self.assertAlmostEqual(m.gW2[o][j], (lp - lm) / (2 * eps), places=6)

    def test_attention_mask_changes_the_input_and_the_loss(self):
        """The mask must be load-bearing, not decorative."""
        m = NeuralLM(64, 4, 8, 4, seed="m")
        toks = [5, 6, 7, 8, 9, 10, 11, 12]
        same = m.build_context(toks, [1] * 8, 5)
        split = m.build_context(toks, [1, 1, 1, 2, 2, 2, 2, 2], 5)
        self.assertNotEqual(same, split)
        self.assertIn(-1, split, "cross-document slots were not masked")
        self.assertNotEqual(m.forward_loss(same, toks[5])[0],
                            m.forward_loss(split, toks[5])[0])

    def test_state_dict_round_trips(self):
        a = NeuralLM(32, 4, 8, 2, seed="a")
        a.zero_grad()
        _, c = a.forward_loss(a.build_context([1, 2, 3, 4], [1] * 4, 2), 3)
        a.backward(c)
        a.step(0.1)
        b = NeuralLM(32, 4, 8, 2, seed="different")
        b.load_state_dict(a.state_dict())
        self.assertEqual(a.W2, b.W2)
        self.assertEqual(a.E, b.E)


# ---------------------------------------------------------------------------
class TestUsefulnessClassification(unittest.TestCase):

    def test_thresholds_from_the_lecture(self):
        self.assertEqual(classify(0.3), "broken")
        self.assertEqual(classify(1.2), "already_learned")
        self.assertEqual(classify(1.8), "neutral")
        self.assertEqual(classify(3.4), "useful")


if __name__ == "__main__":
    unittest.main(verbosity=2)
