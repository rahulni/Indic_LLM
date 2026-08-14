"""Renders Track B's proof figures from capacity_proof.py's saved report.

  1. Decode accuracy vs. word length, one line per embedding dimension D --
     the headline capacity plot, deliberately crossing the 32-char cap.
  2. Interference visualization: mean cosine similarity to the TRUE filler
     vs. the best-matching DISTRACTOR filler as word length grows, at a
     fixed D -- makes the crosstalk mechanism visible, not just the
     pass/fail accuracy outcome.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.plotting import save_and_embed  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def plot_capacity_curve(report: dict) -> plt.Figure:
    cells = report["cells"]
    by_d: dict[int, list[tuple[int, float, float]]] = {}
    for c in cells:
        by_d.setdefault(c["D"], []).append((c["L"], c["accuracy"], c.get("theoretical_accuracy")))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis")
    d_values = report["d_sweep"]
    for i, d in enumerate(d_values):
        pts = sorted(by_d[d])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = cmap(i / max(len(d_values) - 1, 1))
        ax.plot(xs, ys, marker="o", markersize=4, color=color, label=f"D={d} (measured)")
        theory = [p[2] for p in pts]
        if all(t is not None for t in theory):
            ax.plot(xs, theory, color=color, linewidth=1, linestyle=":", alpha=0.8)
    ax.axvline(32, color="0.3", linestyle="--", linewidth=1, label="V1's 32-char cap")
    ax.plot([], [], color="0.35", linestyle=":", linewidth=1, label="derived SNR bound (dotted)")
    ax.set_xlabel("word length L (number of superposed bound pairs)")
    ax.set_ylabel("decode accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Capacity trades length against dimension — and matches the derived bound")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    return fig


def plot_role_comparison(report: dict) -> plt.Figure:
    """The instructor's literal "each character is a Fourier wave, just add
    them" (deterministic shift roles) against random-phase roles, at the
    model's actual d_model."""
    d = 192 if 192 in report["d_sweep"] else report["d_sweep"][len(report["d_sweep"]) // 2]
    rand_pts = sorted((c["L"], c["accuracy"]) for c in report["cells"] if c["D"] == d)
    shift_pts = sorted((c["L"], c["accuracy"]) for c in report["shift_role_cells"] if c["D"] == d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [p[0] for p in rand_pts], [p[1] for p in rand_pts],
        marker="o", markersize=4, color="#2a78d6", label="random-phase roles",
    )
    ax.plot(
        [p[0] for p in shift_pts], [p[1] for p in shift_pts],
        marker="o", markersize=4, color="#eb6834", label="shift roles (the literal reading)",
    )
    ax.axvline(32, color="0.3", linestyle="--", linewidth=1, label="V1's 32-char cap")
    ax.set_xlabel("word length L")
    ax.set_ylabel("decode accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Both role constructions work; the literal one is not worse (D={d})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_interference(report: dict) -> plt.Figure:
    cells = report["cells"]
    target_d = report["d_sweep"][len(report["d_sweep"]) // 2]  # a middle D, where the effect is visible
    pts = sorted((c["L"], c["mean_true_sim"], c["mean_best_distractor_sim"]) for c in cells if c["D"] == target_d)
    xs = [p[0] for p in pts]
    true_sims = [p[1] for p in pts]
    distractor_sims = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, true_sims, marker="o", markersize=4, color="#2e7d32", label="cosine sim to TRUE filler")
    ax.plot(xs, distractor_sims, marker="o", markersize=4, color="#c0392b", label="cosine sim to best DISTRACTOR")
    ax.axvline(32, color="0.3", linestyle="--", linewidth=1, label="V1's 32-char cap")
    ax.set_xlabel("word length L")
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(f"Crosstalk made visible (D={target_d}): the two curves converging IS the failure mode")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    report = json.loads((RESULTS_DIR / "capacity_proof_report.json").read_text())

    figures = {}
    fig1 = plot_capacity_curve(report)
    figures["capacity_curve"] = save_and_embed(fig1, RESULTS_DIR / "capacity_curve.png")

    fig2 = plot_interference(report)
    figures["interference"] = save_and_embed(fig2, RESULTS_DIR / "interference.png")

    fig3 = plot_role_comparison(report)
    figures["role_comparison"] = save_and_embed(fig3, RESULTS_DIR / "role_comparison.png")

    (RESULTS_DIR / "proof_figures.json").write_text(json.dumps(figures))
    print(f"wrote {len(figures)} figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
