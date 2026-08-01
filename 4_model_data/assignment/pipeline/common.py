"""Shared helpers for the Session 4 cleaning pipeline.

Every stage reads a JSONL file of documents and writes a JSONL file of
surviving documents plus a small JSON report. Reports are the single
source of truth for the widget - nothing in the front end is allowed
to show a number that isn't traceable back to one of these files.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# Windows consoles default to cp1252; Telugu text needs utf-8 everywhere.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def read_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def write_jsonl(path: str, docs: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class StageTimer:
    name: str
    t0: float = field(default_factory=time.time)

    def done(self) -> float:
        return round(time.time() - self.t0, 2)


def make_report(
    stage_num: int,
    stage_name: str,
    input_docs: int,
    output_docs: int,
    elapsed_s: float,
    extra: dict,
    examples: list,
) -> dict:
    survival_pct = round(100.0 * output_docs / input_docs, 3) if input_docs else 0.0
    return {
        "stage_num": stage_num,
        "stage_name": stage_name,
        "input_docs": input_docs,
        "output_docs": output_docs,
        "docs_removed": input_docs - output_docs,
        "survival_pct": survival_pct,
        "elapsed_s": elapsed_s,
        "extra": extra,
        "examples": examples,
    }


PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSIGNMENT_DIR = os.path.dirname(PIPELINE_DIR)
RAW_DIR = os.path.join(ASSIGNMENT_DIR, "raw_sample")

# --- Active run context -------------------------------------------------
# Set once per pipeline run by run_all.py. Stages read these through the
# accessor functions below rather than importing module-level constants, so
# that switching corpus or work directory does not require monkey-patching
# every stage module (which the previous version did, and which made the
# determinism check weaker than it looked).

_ACTIVE_CORPUS = "telugu_web"
_WORK_DIR = os.path.join(ASSIGNMENT_DIR, "work")


def set_context(corpus_id: str, work_dir: str) -> None:
    global _ACTIVE_CORPUS, _WORK_DIR
    _ACTIVE_CORPUS = corpus_id
    _WORK_DIR = work_dir
    os.makedirs(work_dir, exist_ok=True)


def corpus_id() -> str:
    return _ACTIVE_CORPUS


def corpus() -> dict:
    import corpora

    return corpora.get(_ACTIVE_CORPUS)


def work_dir() -> str:
    return _WORK_DIR


def work_path(filename: str) -> str:
    return os.path.join(_WORK_DIR, filename)


def raw_sample() -> str:
    import corpora

    return corpora.raw_path(_ACTIVE_CORPUS)


def heldout_sample() -> str:
    import corpora

    return corpora.heldout_path(_ACTIVE_CORPUS)
