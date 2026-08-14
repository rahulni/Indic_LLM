"""Shared plotting helpers.

Every figure is written to disk as a PNG (for direct inspection / the
generated README) and also returned as a base64 data URI, so it can be
inlined straight into a self-contained dashboard.html with zero external
file references or network fetches.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_and_embed(fig: plt.Figure, out_path: Path, dpi: int = 150) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
