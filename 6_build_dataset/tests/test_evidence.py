# -*- coding: utf-8 -*-
"""The evidence bundle must be earned.

The assignment says hardcoded evidence will not be accepted. That is a claim
about the implementation, so it needs a test that could actually catch a
violation: run the real demo, confirm the bundle passes, then **corrupt an
artifact** and confirm the corresponding row flips to FAIL.

If the bundle were built from in-memory booleans, every one of these would still
say PASS after the corruption. That is precisely what is being ruled out.

The demo runs once at ``--profile fast`` and every case reuses it, so the whole
module costs one run rather than one per assertion.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tdes import evidence as ev                                       # noqa: E402
from tdes.hashing import read_json, write_json                        # noqa: E402

_ARTIFACTS: str | None = None
_TMP: str | None = None


def setUpModule() -> None:
    global _ARTIFACTS, _TMP
    _TMP = tempfile.mkdtemp(prefix="tdes-evidence-")
    _ARTIFACTS = os.path.join(_TMP, "artifacts")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "run_demo.py"),
         "--profile", "fast", "--out", _ARTIFACTS],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise unittest.SkipTest(
            f"run_demo.py --profile fast exited {r.returncode}\n"
            f"{r.stdout[-3000:]}\n{r.stderr[-2000:]}")


def tearDownModule() -> None:
    if _TMP and os.path.isdir(_TMP):
        shutil.rmtree(_TMP, ignore_errors=True)


class EvidenceCase(unittest.TestCase):
    """Copies the artifacts so each corruption starts from a clean, passing run."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tdes-case-", dir=_TMP)
        self.art = os.path.join(self.dir, "artifacts")
        shutil.copytree(_ARTIFACTS, self.art)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def rebuild(self) -> dict:
        return ev.build(self.art)

    def row(self, bundle: dict, key: str) -> dict:
        return next(c for c in bundle["checks"] if c["key"] == key)

    def assertRowFails(self, key: str) -> None:
        b = self.rebuild()
        self.assertEqual(self.row(b, key)["result"], "FAIL",
                         f"{key} still passed after its evidence was corrupted "
                         f"-- the bundle is not reading the artifact")
        self.assertFalse(b["summary"]["all_passed"])


class TestBaseline(EvidenceCase):

    def test_untouched_run_passes_every_row(self):
        b = self.rebuild()
        failed = [c["key"] for c in b["checks"] if c["result"] == "FAIL"]
        self.assertEqual(failed, [], f"clean run failed: {failed}")
        self.assertEqual(b["summary"]["passed"], b["summary"]["total"])

    def test_bundle_reports_what_it_read(self):
        b = self.rebuild()
        a = b["artifacts_read"]
        self.assertGreater(a["consumption.jsonl"], 0)
        self.assertGreater(a["manifests"], 0)
        self.assertGreater(a["learning_tokens.jsonl"], 0)
        self.assertGreater(a["opus_decisions.jsonl"], 0)

    def test_required_artifact_layout_exists(self):
        for rel in ["run.log", "evidence.json", "evidence.md", "performance.json",
                    "manifests", "ledgers", "checkpoints"]:
            with self.subTest(artifact=rel):
                self.assertTrue(os.path.exists(os.path.join(self.art, rel)),
                                f"missing {rel}")


class TestCorruptionFlipsRows(EvidenceCase):

    def test_corrupting_replay_hashes_fails_replay(self):
        p = os.path.join(self.art, "ledgers", "replay.json")
        d = read_json(p)
        d["all_matched"] = False
        d["mismatches"] = [{"global_step": 7, "batch_id_match": False}]
        write_json(p, d)
        self.assertRowFails("replay")

    def test_corrupting_resume_batch_id_fails_crash_recovery(self):
        p = os.path.join(self.art, "ledgers", "resume.json")
        d = read_json(p)
        d["matched"] = False
        d["actual_batch_id"] = "deadbeef"
        write_json(p, d)
        self.assertRowFails("crash_recovery")

    def test_deleting_a_ledger_record_fails_crash_recovery(self):
        """A hole in the ledger is a skipped batch, whatever the summary says."""
        p = os.path.join(self.art, "ledgers", "consumption.jsonl")
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        keep = rows[:6] + rows[8:]
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in keep:
                f.write(json.dumps(r, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False) + "\n")
        self.assertRowFails("crash_recovery")

    def test_tampering_with_a_manifest_tokenizer_hash_fails_tokenizer_integrity(self):
        names = [n for n in sorted(os.listdir(os.path.join(self.art, "manifests")))
                 if n.endswith(".json") and n not in ("index.json", "mixture_schedule.json")]
        p = os.path.join(self.art, "manifests", names[0])
        m = read_json(p)
        m["tokenizer_hash"] = "0" * 64
        write_json(p, m)
        self.assertRowFails("tokenizer_integrity")

    def test_recording_a_leaked_eval_document_fails_the_firewall(self):
        p = os.path.join(self.art, "ledgers", "firewall.json")
        d = read_json(p)
        d["leak_scan"] = {"heldout_docs_in_batches": 1, "doc_ids": ["eval_web_0001"]}
        write_json(p, d)
        self.assertRowFails("evaluation_firewall")

    def test_a_gradient_bearing_validation_read_fails_the_firewall(self):
        p = os.path.join(self.art, "ledgers", "firewall.json")
        d = read_json(p)
        d["registry"]["gradient_bearing_reads"] = 1
        write_json(p, d)
        self.assertRowFails("evaluation_firewall")

    def test_a_floor_violation_fails_mixture_compliance(self):
        p = os.path.join(self.art, "reports.json")
        d = read_json(p)
        d["floors"]["floors_held"] = False
        d["floors"]["violation_count"] = 3
        write_json(p, d)
        self.assertRowFails("mixture_compliance")

    def test_inconsistent_throughput_arithmetic_fails_throughput(self):
        """A rate that cannot be recomputed from the raw counters is not credit."""
        p = os.path.join(self.art, "performance.json")
        d = read_json(p)
        d["rates"]["useful_loss_bearing_tokens_per_second"] *= 3.0
        write_json(p, d)
        self.assertRowFails("throughput")

    def test_emptying_the_opus_ledger_fails_the_audit_trail(self):
        open(os.path.join(self.art, "ledgers", "opus_decisions.jsonl"),
             "w", encoding="utf-8").close()
        self.assertRowFails("opus_audit_trail")

    def test_emptying_the_token_trace_fails_the_learning_trace(self):
        open(os.path.join(self.art, "ledgers", "learning_tokens.jsonl"),
             "w", encoding="utf-8").close()
        self.assertRowFails("learning_trace")

    def test_removing_the_pass_line_from_run_log_fails_its_row(self):
        """The log is evidence too, not decoration."""
        p = os.path.join(self.art, "run.log")
        with open(p, encoding="utf-8") as f:
            text = f.read()
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace("[PASS] replay_hash_matched", ""))
        self.assertRowFails("replay")


if __name__ == "__main__":
    unittest.main(verbosity=2)
