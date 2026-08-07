# -*- coding: utf-8 -*-
"""Recovery and policy tests that the main invariant suite does not cover.

These four were promised in the plan and were missing. Two of them cover code
paths that had **no** test at all:

* **Torn-tail recovery.** ``consumption.py`` claims a crash mid-``write`` can
  leave a partial final line and that the reader survives it. Nothing exercised
  that. Here a real torn line is written and the recovery path is driven.

* **The Indic tier rule.** ``INDIC_VERIFIED_FLOOR_FRACTION`` was declared in
  config and only ever *reported* -- a guarantee nothing enforced. Now the
  drawing logic honours it and this proves the verifier rejects a stream that
  violates it.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tdes.batching import LaneSequencePool, sequence_tier                # noqa: E402
from tdes.config import INDIC_VERIFIED_FLOOR_FRACTION                    # noqa: E402
from tdes.hashing import read_jsonl                                      # noqa: E402
from tdes.ledger.consumption import ConsumptionLedger, verify_integrity  # noqa: E402
from tdes.mixture import verify_indic_tier_floor                         # noqa: E402
from tdes.perf import PerfMeter, actual_lane_shares                      # noqa: E402
from tdes.scarcity import resolve, summarise                             # noqa: E402


def fake_batch(step: int, ranks=2, lane="web", tier=None) -> dict:
    samples = [{
        "sample_index": i, "lane": lane, "policy": "concat_chop", "seq_len": 8,
        "tokens": [1] * 8, "loss_mask": [1] * 8, "segment_ids": [1] * 8,
        "position_ids": list(range(8)), "spans": [{
            "shard_id": "s0", "doc_id": f"d{i}", "shard_start": i * 8,
            "shard_end": i * 8 + 8, "seq_start": 0, "seq_end": 8,
            "truncated": False, "roles": []}],
        "doc_ids": [f"d{i}"], "shard_ids": ["s0"], "real_tokens": 8,
        "pad_tokens": 0, "loss_bearing_tokens": 7, "pool_epoch": 0,
        "indic_tier": tier,
    } for i in range(ranks)]
    return {
        "run_id": "r", "branch_id": "main", "global_step": step, "stage": "A",
        "seq_len": 8, "batch_id": f"batch-{step:04d}",
        "batch_content_hash": f"content-{step:04d}",
        "loss_mask_hash": f"lm-{step:04d}",
        "attention_policy": "document_causal",
        "position_policy": "reset_per_document",
        "microbatches": [{"rank": r, "accum_index": 0,
                          "microbatch_id": f"s{step}_r{r}",
                          "sample_indices": [r], "samples": [samples[r]]}
                         for r in range(ranks)],
        "samples": samples, "lane_counts": {lane: ranks},
        "total_positions": 8 * ranks, "real_tokens": 8 * ranks,
        "pad_tokens": 0, "loss_bearing_tokens": 7 * ranks,
        "shard_ids": ["s0"], "doc_ids": [f"d{i}" for i in range(ranks)],
    }


class TestTornTailRecovery(unittest.TestCase):
    """A crash mid-write leaves a partial line. The ledger must survive it."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tdes-torn-")
        self.path = os.path.join(self.dir, "consumption.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write_records(self, n=6) -> ConsumptionLedger:
        led = ConsumptionLedger(self.path, run_id="r", branch_id="main")
        for step in range(n):
            led.append_batch(fake_batch(step), {}, checkpoint_id="c0",
                             tokenizer_hash="t", dataloader_version="v")
        led.commit()
        led.close()
        return led

    def _tear(self) -> None:
        """Append a genuinely truncated JSON line, as a killed process would."""
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write('{"ledger_offset": 99, "run_id": "r", "branch_id": "ma')

    def test_reader_survives_a_torn_tail(self):
        self._write_records()
        self._tear()
        rows = read_jsonl(self.path, tolerate_torn_tail=True)
        self.assertEqual(len(rows), 12, "complete records should still be readable")

    def test_reader_raises_on_a_torn_tail_when_not_tolerating(self):
        self._write_records()
        self._tear()
        with self.assertRaises(json.JSONDecodeError):
            read_jsonl(self.path, tolerate_torn_tail=False)

    def test_torn_tail_is_detected_and_reported(self):
        self._write_records()
        self._tear()
        led = ConsumptionLedger(self.path, run_id="r", branch_id="main")
        self.assertTrue(led.has_torn_tail())
        out = led.truncate_to(8)
        self.assertTrue(out["torn_tail_repaired"])
        self.assertEqual(out["records_after"], 8)
        self.assertFalse(led.has_torn_tail(), "repair left the file still torn")

    def test_recovery_leaves_a_ledger_that_passes_integrity(self):
        """The point of surviving a torn tail is a usable ledger afterwards."""
        self._write_records()
        self._tear()
        led = ConsumptionLedger(self.path, run_id="r", branch_id="main")
        led.truncate_to(8)                       # rewind to a checkpoint offset
        integ = verify_integrity(led.read_all(), branch_id="main")
        self.assertTrue(integ["ok"], integ["problems"])
        self.assertTrue(integ["offsets_dense"])
        self.assertTrue(integ["no_skipped_steps"])

    def test_appending_after_recovery_keeps_offsets_dense(self):
        self._write_records()
        self._tear()
        led = ConsumptionLedger(self.path, run_id="r", branch_id="main")
        led.truncate_to(8)
        led.append_batch(fake_batch(4), {}, checkpoint_id="c1",
                         tokenizer_hash="t", dataloader_version="v")
        led.commit()
        led.close()
        integ = verify_integrity(read_jsonl(self.path), branch_id="main")
        self.assertTrue(integ["offsets_dense"], integ["problems"])


class TestIndicTierFloor(unittest.TestCase):
    """Unverified Indic must not substitute for the verified part of the floor."""

    def _records(self, verified: int, unverified: int) -> list[dict]:
        return [{"branch_id": "main",
                 "indic_tiers": {"verified": verified, "unverified": unverified}}]

    def test_verified_majority_holds(self):
        out = verify_indic_tier_floor(self._records(6, 4))
        self.assertTrue(out["held"])
        self.assertAlmostEqual(out["verified_share"], 0.6)

    def test_indic_unverified_cannot_fill_verified_floor(self):
        out = verify_indic_tier_floor(self._records(2, 8))
        self.assertFalse(out["held"], "an 20/80 split satisfied a 50% floor")
        self.assertEqual(out["required_share"], INDIC_VERIFIED_FLOOR_FRACTION)

    def test_exactly_at_the_boundary_holds(self):
        out = verify_indic_tier_floor(self._records(5, 5))
        self.assertTrue(out["held"], "exactly 50% should satisfy a 50% floor")

    def test_no_indic_samples_is_not_checkable(self):
        out = verify_indic_tier_floor([{"branch_id": "main", "indic_tiers": {}}])
        self.assertFalse(out["checkable"])
        self.assertFalse(out["held"], "an unmeasurable rule must not report held")

    def test_other_branches_are_excluded(self):
        rows = [{"branch_id": "main", "indic_tiers": {"verified": 5, "unverified": 5}},
                {"branch_id": "exp-b", "indic_tiers": {"verified": 0, "unverified": 99}}]
        self.assertTrue(verify_indic_tier_floor(rows, branch_id="main")["held"])

    def test_a_mixed_sequence_counts_as_unverified(self):
        """Calling a tier-mixed sequence verified is the substitution itself."""
        by_doc = {"a": {"indic_tier": "verified"}, "b": {"indic_tier": "unverified"}}
        pure = {"segments": [{"doc_id": "a"}, {"doc_id": "a"}]}
        mixed = {"segments": [{"doc_id": "a"}, {"doc_id": "b"}]}
        none = {"segments": [{"doc_id": "zzz"}]}
        self.assertEqual(sequence_tier(pure, by_doc), "verified")
        self.assertEqual(sequence_tier(mixed, by_doc), "unverified")
        self.assertIsNone(sequence_tier(none, by_doc))

    def test_tier_sub_pools_serve_only_their_tier(self):
        seqs = [{"policy": "greedy", "seq_len": 8, "segments": [],
                 "real_tokens": 8, "pad_tokens": 0, "indic_tier": "verified"}] * 3
        pool = LaneSequencePool("indic:verified", seqs, "seed")
        drawn = pool.take(3)
        self.assertEqual(len(drawn), 3)
        self.assertTrue(all(d["indic_tier"] == "verified" for d in drawn))


class TestScarcityRecording(unittest.TestCase):

    def test_scarcity_policy_recorded(self):
        """Every decision names a policy and the reasoning behind it."""
        decisions = [
            resolve("web", 50, 100, stage_id="A"),
            resolve("indic", 300, 100, stage_id="A"),
            resolve("code", 900, 100, stage_id="A", later_stages_exist=True),
            resolve("agentic", 900, 100, stage_id="D", is_protected=True),
        ]
        for d in decisions:
            with self.subTest(lane=d["lane"]):
                self.assertIn("policy", d)
                self.assertIn("reason", d)
                self.assertTrue(d["reason"], "a policy without a reason is a guess")
        self.assertEqual([d["policy"] for d in decisions],
                         ["none", "repeat", "defer_to_later_stage", "repeat_over_cap"])

    def test_summary_surfaces_cap_breaches(self):
        s = summarise([resolve("indic", 900, 100, stage_id="A", is_protected=True)])
        self.assertEqual(len(s["cap_breaches"]), 1)
        self.assertIn("epoch_cap_source", s)

    def test_protected_lane_is_never_silently_shrunk(self):
        d = resolve("indic", 900, 100, stage_id="A", is_protected=True)
        self.assertEqual(d["shortfall_tokens"], 0)
        self.assertIn("warning", d, "a cap breach on a protected lane must be loud")


class TestPerfReconstructible(unittest.TestCase):

    def test_perf_numbers_reconstructible(self):
        """Every published rate must be recomputable from the raw counters."""
        meter = PerfMeter()
        for step in range(4):
            meter.record_step(fake_batch(step), {}, seconds=0.5)
        rep = meter.report(
            loader_stats={"loader_wait_seconds": 0.1,
                          "cache": {"hit_rate": 0.5}},
            packing={}, opus_summary={"by_status": {"accepted": 2},
                                      "total_decisions": 4},
            schedule={"planned_shares": {"by_lane_share": {"web": 1.0}}},
            actual_shares={"web": 1.0})
        c = rep["raw_counters"]
        secs = rep["train_seconds"]
        self.assertAlmostEqual(rep["rates"]["raw_tokens_per_second"],
                               c["positions_total"] / secs, places=3)
        self.assertAlmostEqual(rep["rates"]["useful_loss_bearing_tokens_per_second"],
                               c["tokens_loss_bearing"] / secs, places=3)
        self.assertAlmostEqual(rep["efficiency"]["packing_utilization"],
                               c["tokens_real"] / c["positions_total"], places=6)
        self.assertAlmostEqual(rep["efficiency"]["pad_fraction"],
                               c["tokens_pad"] / c["positions_total"], places=6)
        for key in rep["rates"]:
            if key in rep["formulas"]:
                self.assertTrue(rep["formulas"][key], f"{key} published without a formula")

    def test_actual_shares_sum_to_one(self):
        rows = [{"branch_id": "main", "lane_counts": {"web": 3, "indic": 1},
                 "loss_bearing_tokens": 40},
                {"branch_id": "main", "lane_counts": {"code": 2},
                 "loss_bearing_tokens": 20}]
        shares = actual_lane_shares(rows)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=6)

    def test_other_branches_excluded_from_shares(self):
        rows = [{"branch_id": "main", "lane_counts": {"web": 1},
                 "loss_bearing_tokens": 10},
                {"branch_id": "exp-b", "lane_counts": {"code": 1},
                 "loss_bearing_tokens": 999}]
        self.assertEqual(sorted(actual_lane_shares(rows)), ["web"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
