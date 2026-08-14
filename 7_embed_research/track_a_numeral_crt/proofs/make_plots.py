"""Renders the two proof figures from analytic_proof.py's saved plot data.

  1. "Clock" structure: for each value-code prime p_i, plot n mod p_i as a
     point on the unit circle at angle 2*pi*r/p_i, colored by n. This is the
     visual signature of the circulant/DFT structure behind the additive
     homomorphism -- adding a constant is a rotation by a fixed angle.
  2. Shift-scatter: decode(shift(encode(a), k)) vs (a+k) mod N across a
     sample of a, for a few k. A perfect diagonal line is the proof made
     visible: zero error, not "usually small error".

Run after analytic_proof.py. Writes PNGs into results/ and prints the base64
data URIs to proof_figures.json for the dashboard builder to inline.
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


def plot_clock_structure(data: dict) -> plt.Figure:
    primes = data["primes"]
    n_sample = np.array(data["n_sample"])
    residues = np.array(data["residues"])  # (n_samples, len(primes))

    fig, axes = plt.subplots(1, len(primes), figsize=(3.2 * len(primes), 3.4), subplot_kw={"aspect": "equal"})
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=n_sample.min(), vmax=n_sample.max())
    for i, (ax, p) in enumerate(zip(axes, primes)):
        angles = 2 * np.pi * residues[:, i] / p
        x, y = np.cos(angles), np.sin(angles)
        ax.scatter(x, y, c=n_sample, cmap=cmap, norm=norm, s=10, alpha=0.6, linewidths=0)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="0.85", linewidth=1, zorder=0)
        ax.set_title(f"mod {p}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.7, label="n (value code)", pad=0.02)
    fig.suptitle("Each residue slot is literally a clock: n mod p sits at angle 2π(n mod p)/p", fontsize=11)
    return fig


def plot_shift_scatter(data: dict) -> plt.Figure:
    scatter = data["shift_scatter"]
    ks = list(scatter.keys())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.6 * len(ks), 3.6))
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        expected = np.array(scatter[k]["expected"])
        decoded = np.array(scatter[k]["decoded"])
        ax.scatter(expected, decoded, s=6, alpha=0.4, color="#3b6fa0", linewidths=0)
        lo, hi = expected.min(), expected.max()
        ax.plot([lo, hi], [lo, hi], color="#c0392b", linewidth=1, linestyle="--", label="y = x")
        ax.set_title(f"k = {k}\nmax |error| = {scatter[k]['max_abs_error']}", fontsize=10)
        ax.set_xlabel("(a + k) mod N")
        ax.set_ylabel("decode(shift(encode(a), k))")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Exact recovery: shifting the embedding IS modular addition", fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    data = json.loads((RESULTS_DIR / "proof_plot_data.json").read_text())

    figures = {}
    fig1 = plot_clock_structure(data)
    figures["clock_structure"] = save_and_embed(fig1, RESULTS_DIR / "clock_structure.png")

    fig2 = plot_shift_scatter(data)
    figures["shift_scatter"] = save_and_embed(fig2, RESULTS_DIR / "shift_scatter.png")

    (RESULTS_DIR / "proof_figures.json").write_text(json.dumps(figures))
    print(f"wrote {len(figures)} figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
