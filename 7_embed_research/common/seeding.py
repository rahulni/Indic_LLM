"""Seed every RNG a script might touch.

torch and numpy are imported lazily so the pure-stdlib proof scripts (which
have no torch dependency, and run in well under a minute) can still call
this without requiring either to be installed.
"""
from __future__ import annotations

import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
