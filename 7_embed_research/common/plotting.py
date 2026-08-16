"""Shared plotting helpers.

Every figure is written to disk as a PNG (for direct inspection / the
generated README) and also returned as a base64 data URI, so it can be
inlined straight into a self-contained dashboard.html with zero external
file references or network fetches.

Figures are rendered once per theme. A PNG's ink is baked in, so unlike the
SVG charts -- which resolve `var(--series-N)` live and therefore follow the
theme toggle for free -- a single PNG cannot serve both a black and a white
page. `render_themed()` produces both and the dashboard shows whichever
matches the active theme.
"""
from __future__ import annotations

import base64
import io
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ink and series colours mirroring chart_kit.PALETTE_CSS, so a figure sits on
# the same surface as the card around it. The series hexes are the validated
# categorical palette -- both sets pass the six colour checks against their
# own surface, so do not hand-tune them here without re-validating.
FIGURE_THEMES: dict[str, dict] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "faint": "#c3c2b7",
        "grid": "#e1e0d9",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    },
    "dark": {
        "surface": "#141413",
        "ink": "#f2f2f0",
        "muted": "#c3c2b7",
        "faint": "#4a4a46",
        "grid": "#2c2c2a",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500"),
    },
}


def _rc(theme: dict) -> dict:
    """rcParams carrying a theme's ink, so titles, tick labels, axis labels and
    legends follow it without every plot function restating them."""
    return {
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["ink"],
        "axes.labelcolor": theme["muted"],
        "axes.titlecolor": theme["ink"],
        "axes.edgecolor": theme["faint"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "grid.color": theme["grid"],
        "legend.facecolor": theme["surface"],
        "legend.edgecolor": theme["faint"],
        "legend.framealpha": 0.9,
    }


def render_themed(
    build: Callable[[dict], plt.Figure], out_path: Path, dpi: int = 150
) -> dict[str, str]:
    """Calls `build(theme)` once per theme, returning {theme_name: data_uri}.

    The light PNG keeps the original filename so existing references to e.g.
    `clock_structure.png` stay valid; the dark one gains a `_dark` suffix.
    """
    out_path = Path(out_path)
    uris: dict[str, str] = {}
    for name, theme in FIGURE_THEMES.items():
        with plt.rc_context(_rc(theme)):
            fig = build(theme)
            suffix = "" if name == "light" else "_dark"
            path = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix}")
            uris[name] = save_and_embed(fig, path, dpi)
    return uris


def save_and_embed(fig: plt.Figure, out_path: Path, dpi: int = 150) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
