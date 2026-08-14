"""Standalone measurement of Track B's capacity/crosstalk claim.

No transformer, no training. Binding two symbols via circular convolution is,
by the Convolution Theorem, exactly elementwise multiplication of their DFT
representations:

    bind(role, filler) = ifft(fft(role) * fft(filler)).real

A word is composed by SUMMING (superposing) each character's role-bound
filler into a single FIXED-DIMENSION vector, with no length cap:

    word_vec = sum_i bind(role(i), filler(char_i))

Because ifft is linear, this sum can be computed by summing spectra first and
inverse-transforming once -- that identity is exploited below purely for
vectorized speed, it changes nothing mathematically.

Roles are constructed as UNITARY vectors (unit magnitude at every DFT
frequency, random phase -- literally points on a circle per frequency
component, i.e. Fourier waves, per the instructor's own framing). This makes
single-pair unbinding EXACT:

    unbind(bind(role, filler), role) == filler   exactly, no error term

because multiplying by the conjugate spectrum divides out a magnitude-1
factor. The interesting question -- and the whole point of this track -- is
what happens once MULTIPLE bound pairs are superposed into one word vector:
unbinding position i then recovers filler(char_i) plus interference from
every other bound pair, and that interference is what this script measures,
as a function of word length L and embedding dimension D.

Runtime budget: fully vectorized (batched FFT), well under 60 seconds.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.seeding import seed_everything  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ALPHABET_SIZE = 30  # representative size; the real corpus alphabet is fixed later in data/corpus.py
L_MAX = 40  # deliberately crosses the V1 32-character cap
L_SWEEP = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40]
D_SWEEP = [64, 128, 192, 256, 512, 1024]  # 192 is included because it is the model's actual d_model
N_TRIALS = 300


def theoretical_decode_accuracy(L: int, D: int, alphabet_size: int) -> float:
    """Our own derivation of the standard VSA/HRR signal-to-noise argument --
    NOT a formula copied from Plate, and deliberately not presented as one
    (see CITATIONS.md). Stated in full so a reader can check every step:

    Unbinding position i from a superposition of L bound pairs returns the
    true filler plus the sum of (L-1) interference terms, each of which is a
    convolution of unrelated random vectors and is therefore approximately
    zero-mean with the same per-component variance as a filler (1/D per
    component, since fillers are drawn N(0, 1/D)).

    Score the true filler by dot product: signal ~ ||filler||^2 ~ 1, and its
    variance is dominated by the interference projected onto that filler:
    Var ~ (L-1)/D. A distractor filler is uncorrelated with everything, so
    its score is ~N(0, L/D).

    Decode is correct when the true filler outscores all (alphabet_size - 1)
    distractors. With s = 1/sqrt((L-1)/D + eps) as the effective SNR, and
    treating the distractor scores as independent standard normals scaled by
    sqrt(L/D):

        P(correct) = E_x[ Phi(x * sqrt(D/L) + ...) ^ (alphabet_size - 1) ]

    approximated below by the closed form P = Phi(snr / sqrt(2))^(A-1) with
    snr = sqrt(D / max(L-1, 1)), integrating out via a Gaussian quadrature so
    the expectation over the true score's own fluctuation is not dropped.

    Assumptions (all stated, none hidden): roles are unitary so single-pair
    unbind is exact; fillers are iid Gaussian; interference terms are treated
    as independent Gaussians (they are not exactly independent, so this is an
    approximation, and it is expected to be optimistic at small D).
    """
    if L <= 1:
        return 1.0
    snr = math.sqrt(D / (L - 1))
    # numerically integrate over the true filler's own score fluctuation
    xs = np.linspace(-8, 8, 2001)
    pdf = np.exp(-0.5 * xs**2) / math.sqrt(2 * math.pi)
    # true score ~ snr + x (in units of the interference sd); each distractor ~ N(0,1)
    cdf_beats_one = 0.5 * (1 + erf_vec((snr + xs) / math.sqrt(2)))
    integrand = pdf * np.power(cdf_beats_one, alphabet_size - 1)
    return float(np.trapezoid(integrand, xs))


def erf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized erf via numpy (math.erf is scalar-only)."""
    return np.vectorize(math.erf)(x)


def shift_role_vector(D: int, position: int) -> np.ndarray:
    """The instructor's LITERAL framing -- "represent each character like a
    Fourier wave" -- as a deterministic role: position p is the unit impulse
    shifted to index p, whose DFT is the pure phase ramp exp(-2*pi*i*k*p/D).
    Unit magnitude at every frequency, so single-pair unbind is still exact;
    the difference from random-phase roles is that these roles are highly
    structured, so binding is just a circular SHIFT of the filler -- and
    shifted copies of similar fillers interfere in a correlated way."""
    v = np.zeros(D)
    v[position % D] = 1.0
    return v


def random_unitary_vector(D: int, rng: np.random.Generator) -> np.ndarray:
    """A real vector whose DFT has magnitude exactly 1 at every frequency
    (random phase) -- the exact-inverse-under-circular-convolution role
    vector construction."""
    spec = np.zeros(D, dtype=complex)
    if D % 2 == 0:
        n_free = D // 2 - 1
        spec[D // 2] = rng.choice([-1.0, 1.0])
    else:
        n_free = (D - 1) // 2
    spec[0] = 1.0
    phases = rng.uniform(0, 2 * np.pi, size=n_free)
    spec[1 : 1 + n_free] = np.exp(1j * phases)
    spec[D - n_free : D] = np.conj(spec[1 : 1 + n_free][::-1])
    return np.fft.ifft(spec).real


def build_roles(l_max: int, D: int, rng: np.random.Generator, kind: str = "random") -> np.ndarray:
    if kind == "shift":
        return np.stack([shift_role_vector(D, p) for p in range(l_max)])  # (l_max, D)
    return np.stack([random_unitary_vector(D, rng) for _ in range(l_max)])  # (l_max, D)


def build_fillers(alphabet_size: int, D: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 1.0 / np.sqrt(D), size=(alphabet_size, D))  # (alphabet_size, D)


def sweep_cell(
    L: int, D: int, roles: np.ndarray, fillers: np.ndarray, n_trials: int, rng: np.random.Generator
) -> dict:
    alphabet_size = fillers.shape[0]
    char_indices = rng.integers(0, alphabet_size, size=(n_trials, L))  # (N, L)

    fft_roles = np.fft.fft(roles[:L], axis=-1)  # (L, D) complex
    fft_fillers_used = np.fft.fft(fillers[char_indices], axis=-1)  # (N, L, D) complex

    # sum spectra first (ifft linearity), then a single inverse transform
    word_spec = (fft_fillers_used * fft_roles[None, :, :]).sum(axis=1)  # (N, D)
    word_vec = np.fft.ifft(word_spec, axis=-1).real  # (N, D)

    # unbind at every position simultaneously
    fft_word = np.fft.fft(word_vec, axis=-1)  # (N, D)
    est_spec = fft_word[:, None, :] * np.conj(fft_roles)[None, :, :]  # (N, L, D)
    estimates = np.fft.ifft(est_spec, axis=-1).real  # (N, L, D)

    filler_norms = np.linalg.norm(fillers, axis=1)  # (alphabet_size,)
    est_norms = np.linalg.norm(estimates, axis=-1)  # (N, L)
    sims = np.einsum("nld,ad->nla", estimates, fillers) / (
        est_norms[:, :, None] * filler_norms[None, None, :] + 1e-12
    )  # (N, L, alphabet_size)
    predicted = np.argmax(sims, axis=-1)  # (N, L)
    correct = predicted == char_indices  # (N, L)

    # interference visualization data: cosine similarity to the TRUE filler
    # vs. the best-matching DISTRACTOR filler, at every decode -- makes the
    # crosstalk mechanism visible rather than just the pass/fail outcome
    true_sims = np.take_along_axis(sims, char_indices[:, :, None], axis=-1)[:, :, 0]  # (N, L)
    sims_masked = sims.copy()
    np.put_along_axis(sims_masked, char_indices[:, :, None], -np.inf, axis=-1)
    best_distractor_sims = sims_masked.max(axis=-1)  # (N, L)

    return {
        "L": L,
        "D": D,
        "n_trials": n_trials,
        "n_decodes": int(correct.size),
        "accuracy": float(correct.mean()),
        "mean_true_sim": float(true_sims.mean()),
        "mean_best_distractor_sim": float(best_distractor_sims.mean()),
    }


def run_all(seed: int = 0) -> dict:
    seed_everything(seed)
    t0 = time.time()
    cells = []
    shift_cells = []
    for D in D_SWEEP:
        rng = np.random.default_rng(seed * 1000 + D)  # deterministic per-D roles/fillers
        roles = build_roles(L_MAX, D, rng)
        fillers = build_fillers(ALPHABET_SIZE, D, rng)

        # sanity check baked into the sweep itself: single-pair unbind must be exact
        single_rng = np.random.default_rng(seed)
        ci = single_rng.integers(0, ALPHABET_SIZE)
        pi = single_rng.integers(0, L_MAX)
        bound = np.fft.ifft(np.fft.fft(roles[pi]) * np.fft.fft(fillers[ci])).real
        recovered = np.fft.ifft(np.fft.fft(bound) * np.conj(np.fft.fft(roles[pi]))).real
        single_pair_max_error = float(np.max(np.abs(recovered - fillers[ci])))
        assert single_pair_max_error < 1e-8, "single-pair unbind should be exact for a unitary role vector"

        for L in L_SWEEP:
            cell = sweep_cell(L, D, roles, fillers, N_TRIALS, rng)
            cell["theoretical_accuracy"] = theoretical_decode_accuracy(L, D, ALPHABET_SIZE)
            cells.append(cell)

        # the instructor's literal "each character is a Fourier wave, just add
        # them" reading: deterministic shift roles instead of random phase
        shift_roles = build_roles(L_MAX, D, rng, kind="shift")
        shift_bound = np.fft.ifft(np.fft.fft(shift_roles[pi]) * np.fft.fft(fillers[ci])).real
        shift_recovered = np.fft.ifft(np.fft.fft(shift_bound) * np.conj(np.fft.fft(shift_roles[pi]))).real
        assert np.max(np.abs(shift_recovered - fillers[ci])) < 1e-8, "shift roles are unitary too"
        for L in L_SWEEP:
            shift_cells.append(sweep_cell(L, D, shift_roles, fillers, N_TRIALS, rng))

    elapsed = time.time() - t0
    report = {
        "seed": seed,
        "elapsed_seconds": elapsed,
        "alphabet_size": ALPHABET_SIZE,
        "l_max": L_MAX,
        "l_sweep": L_SWEEP,
        "d_sweep": D_SWEEP,
        "n_trials_per_cell": N_TRIALS,
        "single_pair_unbind_exact": True,
        "single_pair_unbind_exact_shift_roles": True,
        "cells": cells,
        "shift_role_cells": shift_cells,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "capacity_proof_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    report = run_all()
    print(f"Holographic binding -- capacity proof ({report['elapsed_seconds']:.2f}s)")
    print(f"  alphabet_size={report['alphabet_size']}  L up to {report['l_max']} (crosses the 32-char cap)")
    print("  single-pair unbind: EXACT (verified per D, max error < 1e-8) -- this is the provable core claim")
    print()
    print("  decode accuracy (word length L x embedding dim D):")
    header = "L".rjust(4) + "".join(f"D={d}".rjust(9) for d in report["d_sweep"])
    print("   " + header)
    by_l = {}
    for cell in report["cells"]:
        by_l.setdefault(cell["L"], {})[cell["D"]] = cell["accuracy"]
    for L in report["l_sweep"]:
        row = str(L).rjust(4)
        for d in report["d_sweep"]:
            row += f"{by_l[L][d]:.2%}".rjust(9)
        print("   " + row)

    print()
    print("  measured vs. our derived SNR bound (D=192, the model's actual d_model):")
    print("      L  measured  theory")
    for cell in report["cells"]:
        if cell["D"] == 192:
            print(f"   {cell['L']:>4}  {cell['accuracy']:>7.2%}  {cell['theoretical_accuracy']:>6.2%}")

    print()
    print("  random-phase roles vs. the instructor's literal shift roles (D=192):")
    print("      L    random     shift")
    shift_by_l = {c["L"]: c["accuracy"] for c in report["shift_role_cells"] if c["D"] == 192}
    for L in report["l_sweep"]:
        print(f"   {L:>4}  {by_l[L][192]:>8.2%}  {shift_by_l[L]:>8.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
