# -*- coding: utf-8 -*-
"""Ledger offset bookkeeping shared by the checkpoint and the orchestrator.

Small on purpose. The offset is the one number that ties model state to data
state, so it lives in exactly one place rather than being recomputed by whoever
needs it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class LedgerPosition:
    """Where a checkpoint sits in the consumed stream."""
    branch_id: str
    global_step: int
    ledger_offset: int
    records_committed: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerPosition":
        return cls(branch_id=d["branch_id"], global_step=int(d["global_step"]),
                   ledger_offset=int(d["ledger_offset"]),
                   records_committed=int(d["records_committed"]))

    def describe(self) -> str:
        return (f"branch={self.branch_id} step={self.global_step} "
                f"offset={self.ledger_offset}")
