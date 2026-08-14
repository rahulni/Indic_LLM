"""A tiny PASS/FAIL harness.

Each row is graded by re-reading an on-disk artifact file, never from
in-memory state left over from whatever produced it — so a run that silently
skipped work cannot fake a PASS, and corrupting an artifact flips its row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class EvidenceRow:
    requirement: str
    file: str
    compared: str
    check: Callable[[], tuple[bool, str]]

    def evaluate(self) -> dict:
        try:
            passed, detail = self.check()
        except Exception as exc:  # artifact missing/corrupt -> FAIL, not a crash
            passed, detail = False, f"error reading artifact: {exc}"
        return {
            "requirement": self.requirement,
            "result": "PASS" if passed else "FAIL",
            "file": self.file,
            "compared": self.compared,
            "detail": detail,
        }


def write_evidence(rows: list[EvidenceRow], out_path: Path) -> bool:
    results = [row.evaluate() for row in rows]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return all(r["result"] == "PASS" for r in results)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
