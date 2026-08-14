"""Standalone, exact proof of the Kronecker Numeral Embedding claim.

No transformer, no learning, no GPU. This script is the load-bearing proof
artifact for Track A: it verifies, directly, that representing an integer via
its residues modulo a set of pairwise-coprime primes (the Kronecker/CRT
decomposition Z/NZ ~= prod Z/p_i Z, i.e. a Residue Number System) makes
addition and multiplication mod N into fixed, independent, per-slot
operations on a one-hot embedding.

Two separate modulus sets are used on purpose -- do not merge them:

  POSITION code  primes = [2, 3, 7]              N = 42          width = 12
    Smallest coprime-prime set >= 32. Reused later as the transformer's
    *positional* embedding (digit place-value index only ever runs 0-31).

  VALUE code     primes = [11, 13, 17, 19, 23]    N = 1,062,347   width = 83
    sqrt(N) ~= 1030, so 3-digit x 3-digit multiplication never wraps. This is
    the set the exhaustive/exact arithmetic claims below are proven against.

What is proven, and how (exhaustive vs. sampled is always stated explicitly):

  1. CRT bijection: n -> (n mod p_1, ..., n mod p_k) is a bijection on
     [0, N) <-> prod [0, p_i). Verified directly (not cited), exhaustively,
     for both modulus sets.
  2. Additive homomorphism, per modulus, EXHAUSTIVE over all (p_i x p_i)
     (residue, shift) pairs: shifting the one-hot code by k positions equals
     the one-hot code of (residue + k) mod p_i. This is the literal
     "circulant permutation" claim, and it is what makes checks 3 and the
     scatter-plot demonstration meaningful rather than merely a residue-
     arithmetic coincidence: since encode/decode is a checked bijection per
     slot (this check), verifying addition at the residue level (check 3)
     is equivalent to verifying it at the one-hot-vector level.
  3. Full-pipeline addition: EXHAUSTIVE over all a in [0, N) against several
     fixed b, plus ~10^6 SAMPLED random (a, b) pairs across the full range.
  4. Multiplicative homomorphism via discrete log (Zech logarithm trick):
     since every modulus here is prime, (Z/p_i Z)* is cyclic, so multiplying
     residues reduces to ADDING their discrete logs. Verified EXHAUSTIVELY
     over all p_i^2 pairs per modulus, explicitly special-casing a=0 or b=0
     mod p_i (0 has no discrete log -- the product is 0 whenever either
     factor is 0, tested, not glossed over).
  5. Full-pipeline multiplication for a*b < N: EXHAUSTIVE over all pairs with
     a, b <= floor(sqrt(N)), plus an explicit WRAPAROUND demonstration once
     a*b >= N (the recovered value is (a*b) mod N, not a*b -- shown on
     purpose, not hidden).

Runtime budget: everything below runs in well under 60 seconds on a laptop
CPU, using vectorized numpy for the O(N) and O(N^2)-scale checks.
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

POSITION_PRIMES = [2, 3, 7]
VALUE_PRIMES = [11, 13, 17, 19, 23]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Modular arithmetic primitives (pure Python, exact integers)
# ---------------------------------------------------------------------------


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def modinv(a: int, m: int) -> int:
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse mod {m}")
    return x % m


def crt_constants(primes: list[int]) -> tuple[int, list[int]]:
    """N and, per prime, the fixed constant c_i = (N/p_i) * inverse(N/p_i, p_i) mod N
    such that decode(residues) = sum_i residues[i] * c_i mod N (Garner/CRT)."""
    N = math.prod(primes)
    consts = []
    for p in primes:
        Ni = N // p
        inv = modinv(Ni % p, p)
        consts.append((Ni * inv) % N)
    return N, consts


def find_primitive_root(p: int) -> int:
    """Brute force -- fine for p <= 23."""
    order_needed = p - 1
    for g in range(1, p):
        seen = set()
        val = 1
        ok = True
        for _ in range(max(order_needed, 1)):
            val = (val * g) % p
            if val in seen:
                ok = False
                break
            seen.add(val)
        if ok and val == 1 and len(seen) == order_needed:
            return g
    raise ValueError(f"no primitive root found for {p}")


def build_log_tables(p: int, g: int) -> tuple[np.ndarray, np.ndarray]:
    """log_arr[r] = discrete log of r base g, for r in [1, p-1]; log_arr[0] is
    an unused sentinel (0 has no discrete log). exp_arr[e] = g^e mod p.

    Since g has order p-1, val = g^e for e in [0, p-2] visits every nonzero
    residue exactly once, so the direct assignment log_arr[val] = e below
    never collides -- no sentinel/branch logic needed to get this right."""
    log_arr = np.zeros(p, dtype=np.int64)
    exp_arr = np.zeros(max(p - 1, 1), dtype=np.int64)
    val = 1
    for e in range(p - 1):
        exp_arr[e] = val
        log_arr[val] = e
        val = (val * g) % p
    return log_arr, exp_arr


# ---------------------------------------------------------------------------
# Vectorized one-hot encode / shift / decode (the literal embedding pipeline)
# ---------------------------------------------------------------------------


def encode_onehot(n_arr: np.ndarray, primes: list[int]) -> list[np.ndarray]:
    mats = []
    for p in primes:
        r = (n_arr % p).astype(np.int64)
        mat = np.zeros((len(n_arr), p), dtype=np.float32)
        mat[np.arange(len(n_arr)), r] = 1.0
        mats.append(mat)
    return mats


def shift_onehot(mats: list[np.ndarray], k: int, primes: list[int]) -> list[np.ndarray]:
    return [np.roll(mat, shift=int(k % p), axis=1) for mat, p in zip(mats, primes)]


def decode_onehot(mats: list[np.ndarray], primes: list[int], N: int, consts: list[int]) -> np.ndarray:
    residue_cols = [np.argmax(mat, axis=1).astype(np.int64) for mat in mats]
    return crt_decode_array(residue_cols, consts, N)


def crt_decode_array(residue_cols: list[np.ndarray], consts: list[int], N: int) -> np.ndarray:
    total = np.zeros_like(residue_cols[0], dtype=np.int64)
    for r_i, c_i in zip(residue_cols, consts):
        total = (total + r_i.astype(np.int64) * c_i) % N
    return total


def mod_mul_via_log_array(
    a_arr: np.ndarray, b_arr: np.ndarray, p: int, log_arr: np.ndarray, exp_arr: np.ndarray
) -> np.ndarray:
    ar = a_arr % p
    br = b_arr % p
    zero_mask = (ar == 0) | (br == 0)
    ea = log_arr[ar]
    eb = log_arr[br]
    e = (ea + eb) % max(p - 1, 1)
    prod = exp_arr[e]
    return np.where(zero_mask, 0, prod)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_bijection(primes: list[int], N: int, label: str) -> dict:
    n_arr = np.arange(N, dtype=np.int64)
    residues = np.stack([n_arr % p for p in primes], axis=1)
    # mixed-radix encode each residue row to a single int to test uniqueness
    mixed = np.zeros(N, dtype=np.int64)
    mult = 1
    for i, p in enumerate(primes):
        mixed += residues[:, i] * mult
        mult *= p
    n_unique = len(np.unique(mixed))
    passed = n_unique == N
    return {
        "check": f"crt_bijection[{label}]",
        "exhaustive": True,
        "n_tested": int(N),
        "passed": bool(passed),
        "detail": f"{n_unique}/{N} distinct residue tuples for N={N}, primes={primes}",
    }


def check_additive_per_modulus(primes: list[int]) -> dict:
    total_pairs = 0
    for p in primes:
        n_arr = np.arange(p, dtype=np.int64)
        mats = encode_onehot(n_arr, [p])[0]  # (p, p) one-hot of every residue
        for k in range(p):
            shifted = np.roll(mats, shift=k, axis=1)
            expected = encode_onehot((n_arr + k) % p, [p])[0]
            if not np.array_equal(shifted, expected):
                return {
                    "check": "additive_homomorphism_per_modulus",
                    "exhaustive": True,
                    "n_tested": total_pairs,
                    "passed": False,
                    "detail": f"mismatch at prime={p}, k={k}",
                }
            total_pairs += p
    return {
        "check": "additive_homomorphism_per_modulus",
        "exhaustive": True,
        "n_tested": total_pairs,
        "passed": True,
        "detail": f"all (residue, shift) pairs verified for primes={primes} "
        f"({total_pairs} pairs = sum(p_i^2))",
    }


def check_full_addition(primes: list[int], N: int, consts: list[int], n_random_pairs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_arr = np.arange(N, dtype=np.int64)

    # exhaustive over all a, for a handful of fixed b
    fixed_bs = [1, 41, 12345, N // 3, N - 1]
    for b in fixed_bs:
        a_res = [n_arr % p for p in primes]
        sum_res = [(r + b) % p for r, p in zip(a_res, primes)]
        decoded = crt_decode_array(sum_res, consts, N)
        expected = (n_arr + b) % N
        if not np.array_equal(decoded, expected):
            return {
                "check": "full_pipeline_addition",
                "exhaustive": False,
                "n_tested": 0,
                "passed": False,
                "detail": f"exhaustive-a check failed for fixed b={b}",
            }
    n_exhaustive = N * len(fixed_bs)

    # sampled random (a, b) pairs across the full range
    a_rand = rng.integers(0, N, size=n_random_pairs, dtype=np.int64)
    b_rand = rng.integers(0, N, size=n_random_pairs, dtype=np.int64)
    a_res = [a_rand % p for p in primes]
    b_res = [b_rand % p for p in primes]
    sum_res = [(ra + rb) % p for ra, rb, p in zip(a_res, b_res, primes)]
    decoded = crt_decode_array(sum_res, consts, N)
    expected = (a_rand + b_rand) % N
    sampled_ok = bool(np.array_equal(decoded, expected))

    return {
        "check": "full_pipeline_addition",
        "exhaustive": True,
        "n_tested": int(n_exhaustive),
        "passed": bool(sampled_ok),
        "detail": (
            f"exhaustive: all a in [0,{N}) x {len(fixed_bs)} fixed b values "
            f"({n_exhaustive} pairs) match; sampled: {n_random_pairs} random "
            f"(a,b) pairs {'match' if sampled_ok else 'MISMATCH'}"
        ),
        "n_sampled": int(n_random_pairs),
        "sampled_passed": sampled_ok,
    }


def check_multiplicative_per_modulus(primes: list[int]) -> dict:
    total_pairs = 0
    zero_cases_tested = 0
    for p in primes:
        g = find_primitive_root(p)
        log_arr, exp_arr = build_log_tables(p, g)
        a_arr = np.arange(p, dtype=np.int64)
        b_arr = np.arange(p, dtype=np.int64)
        A, B = np.meshgrid(a_arr, b_arr, indexing="ij")
        via_log = mod_mul_via_log_array(A.ravel(), B.ravel(), p, log_arr, exp_arr)
        ground_truth = (A.ravel() * B.ravel()) % p
        zero_cases_tested += int(np.sum((A.ravel() == 0) | (B.ravel() == 0)))
        if not np.array_equal(via_log, ground_truth):
            bad = np.argmax(via_log != ground_truth)
            return {
                "check": "multiplicative_homomorphism_via_discrete_log",
                "exhaustive": True,
                "n_tested": total_pairs,
                "passed": False,
                "detail": f"mismatch at prime={p}, pair index {bad}",
            }
        total_pairs += p * p
    return {
        "check": "multiplicative_homomorphism_via_discrete_log",
        "exhaustive": True,
        "n_tested": total_pairs,
        "passed": True,
        "detail": (
            f"all (a,b) residue pairs verified for primes={primes} "
            f"({total_pairs} pairs total, {zero_cases_tested} of them zero-cases "
            f"where a=0 or b=0 mod p, all correctly resolved to product 0)"
        ),
    }


def check_full_multiplication(primes: list[int], N: int, consts: list[int]) -> dict:
    log_exp = {}
    for p in primes:
        g = find_primitive_root(p)
        log_exp[p] = build_log_tables(p, g)

    limit = int(math.isqrt(N))  # a,b <= limit guarantees a*b < N
    a_arr = np.arange(limit + 1, dtype=np.int64)
    A, B = np.meshgrid(a_arr, a_arr, indexing="ij")
    Af, Bf = A.ravel(), B.ravel()

    prod_res = []
    for p in primes:
        log_arr, exp_arr = log_exp[p]
        prod_res.append(mod_mul_via_log_array(Af, Bf, p, log_arr, exp_arr))
    decoded = crt_decode_array(prod_res, consts, N)
    expected = Af * Bf  # < N by construction of `limit`, so no mod needed
    exhaustive_ok = bool(np.array_equal(decoded, expected))

    # explicit wraparound demonstration: a*b >= N on purpose
    a_wrap, b_wrap = 1500, 1500
    assert a_wrap * b_wrap >= N, "wraparound demo must actually exceed N"
    wrap_res = []
    for p in primes:
        log_arr, exp_arr = log_exp[p]
        wrap_res.append(
            mod_mul_via_log_array(np.array([a_wrap]), np.array([b_wrap]), p, log_arr, exp_arr)
        )
    wrap_decoded = int(crt_decode_array(wrap_res, consts, N)[0])
    wrap_expected_mod = (a_wrap * b_wrap) % N
    wrap_raw = a_wrap * b_wrap
    wrap_ok = wrap_decoded == wrap_expected_mod and wrap_decoded != wrap_raw

    return {
        "check": "full_pipeline_multiplication",
        "exhaustive": True,
        "n_tested": int((limit + 1) ** 2),
        "passed": bool(exhaustive_ok and wrap_ok),
        "detail": (
            f"exhaustive: all a,b in [0,{limit}] ({(limit + 1) ** 2} pairs, "
            f"a*b < N={N} guaranteed) {'match' if exhaustive_ok else 'MISMATCH'}; "
            f"wraparound demo: {a_wrap}*{b_wrap}={wrap_raw} (>= N), "
            f"recovered={wrap_decoded}, (a*b) mod N={wrap_expected_mod} "
            f"({'correctly wraps, does NOT equal raw product' if wrap_ok else 'FAILED'})"
        ),
    }


def check_subtraction(primes: list[int], N: int, consts: list[int]) -> dict:
    """Subtraction is the additive inverse: subtracting b is shifting each slot
    by (p - b mod p) mod p -- the same fixed circulant machinery as addition,
    just rotated the other way. Verified per-modulus exhaustively (all
    (residue, k) pairs), then full-pipeline over all a in [0, N) for fixed b.
    The result is (a - b) mod N: for a >= b that IS a - b; for a < b it wraps
    (negative numbers do not exist in [0, N) -- stated, not hidden)."""
    total_pairs = 0
    for p in primes:
        n_arr = np.arange(p, dtype=np.int64)
        mats = encode_onehot(n_arr, [p])[0]
        for k in range(p):
            inverse_shift = (p - k) % p
            shifted = np.roll(mats, shift=inverse_shift, axis=1)
            expected = encode_onehot((n_arr - k) % p, [p])[0]
            if not np.array_equal(shifted, expected):
                return {
                    "check": "subtractive_homomorphism_per_modulus",
                    "exhaustive": True,
                    "n_tested": total_pairs,
                    "passed": False,
                    "detail": f"mismatch at prime={p}, k={k}",
                }
            total_pairs += p

    n_arr = np.arange(N, dtype=np.int64)
    fixed_bs = [1, 41, 12345, N // 3, N - 1]
    for b in fixed_bs:
        a_res = [n_arr % p for p in primes]
        diff_res = [(r - b) % p for r, p in zip(a_res, primes)]
        decoded = crt_decode_array(diff_res, consts, N)
        expected = (n_arr - b) % N
        if not np.array_equal(decoded, expected):
            return {
                "check": "full_pipeline_subtraction",
                "exhaustive": False,
                "n_tested": total_pairs,
                "passed": False,
                "detail": f"full-pipeline check failed for fixed b={b}",
            }

    # explicit underflow demonstration: a < b wraps to (a - b) mod N, on purpose
    a_small, b_big = 5, 12
    under_res = [np.array([((a_small % p) - (b_big % p)) % p], dtype=np.int64) for p in primes]
    under_decoded = int(crt_decode_array(under_res, consts, N)[0])
    under_expected = (a_small - b_big) % N  # = N - 7
    under_ok = under_decoded == under_expected and under_decoded != a_small - b_big

    return {
        "check": "full_pipeline_subtraction",
        "exhaustive": True,
        "n_tested": int(total_pairs + N * len(fixed_bs)),
        "passed": bool(under_ok),
        "detail": (
            f"per-modulus: all (residue, k) pairs ({total_pairs}); full pipeline: all a in "
            f"[0,{N}) x {len(fixed_bs)} fixed b match; underflow demo: {a_small}-{b_big} "
            f"recovers {under_decoded} = (a-b) mod N (wraps, does NOT equal {a_small - b_big})"
        ),
    }


def check_division_by_units(primes: list[int], N: int, consts: list[int]) -> dict:
    """Division by b is multiplication by b's multiplicative inverse --
    realizable in log-space as SUBTRACTING b's discrete log (the negated-log
    trick), per slot. This is exact division only when b is a UNIT mod N
    (gcd(b, N) = 1, i.e. b shares no factor with any modulus) AND a is a
    multiple of b; otherwise the result is the ring element a * b^{-1} mod N,
    which is well-defined but is not integer division -- both cases are
    demonstrated explicitly. If gcd(b, N) > 1, b has no inverse at all and
    division is undefined: also demonstrated, not glossed over."""
    log_exp = {}
    for p in primes:
        g = find_primitive_root(p)
        log_exp[p] = build_log_tables(p, g)

    # per-modulus: for every nonzero b, dividing via log-subtraction inverts multiplication
    total_pairs = 0
    for p in primes:
        log_arr, exp_arr = log_exp[p]
        a_arr = np.arange(p, dtype=np.int64)
        for b in range(1, p):
            prod = (a_arr * b) % p
            # divide prod by b via negated discrete log: log(x) - log(b) mod (p-1)
            nonzero = prod != 0
            recovered = np.zeros_like(prod)
            e = (log_arr[prod[nonzero]] - log_arr[b]) % (p - 1)
            recovered[nonzero] = exp_arr[e]
            if not np.array_equal(recovered, a_arr):
                return {
                    "check": "division_by_units_via_negated_log",
                    "exhaustive": True,
                    "n_tested": total_pairs,
                    "passed": False,
                    "detail": f"mismatch at prime={p}, b={b}",
                }
            total_pairs += p

    # full-pipeline: exact integer division when b | a and gcd(b, N) = 1
    rng = np.random.default_rng(0)
    n_samples = 100_000
    b_candidates = np.array([b for b in range(2, 100) if math.gcd(b, N) == 1], dtype=np.int64)
    b_rand = rng.choice(b_candidates, size=n_samples)
    q_rand = rng.integers(0, 10_000, size=n_samples, dtype=np.int64)
    a_rand = (q_rand * b_rand) % N  # a is a multiple of b by construction

    quot_res = []
    for p in primes:
        log_arr, exp_arr = log_exp[p]
        ar = a_rand % p
        br = b_rand % p
        # b is a unit mod N so br != 0 for every slot; a may be 0 mod p (quotient residue 0)
        nonzero = ar != 0
        rec = np.zeros_like(ar)
        e = (log_arr[ar[nonzero]] - log_arr[br[nonzero]]) % (p - 1)
        rec[nonzero] = exp_arr[e]
        quot_res.append(rec)
    decoded = crt_decode_array(quot_res, consts, N)
    expected = q_rand % N
    sampled_ok = bool(np.array_equal(decoded, expected))

    # demonstration 1: b NOT a unit (shares a factor with N) -> no inverse exists
    b_nonunit = 11  # literally one of the moduli
    nonunit_has_no_inverse = math.gcd(b_nonunit, N) != 1

    # demonstration 2: b a unit but b does NOT divide a -> result is a*b^{-1} mod N, not integer division
    a_nd, b_nd = 7, 2  # 7/2 is not an integer
    inv2 = modinv(b_nd, N)
    ring_result = (a_nd * inv2) % N
    ring_demo_ok = ring_result != a_nd // b_nd and (ring_result * b_nd) % N == a_nd

    return {
        "check": "division_by_units_via_negated_log",
        "exhaustive": True,
        "n_tested": total_pairs,
        "passed": bool(sampled_ok and nonunit_has_no_inverse and ring_demo_ok),
        "detail": (
            f"per-modulus: all (a, b!=0) pairs ({total_pairs}) invert exactly; full pipeline: "
            f"{n_samples} sampled (b|a, gcd(b,N)=1) cases recover the exact quotient; "
            f"gcd({b_nonunit}, N)={math.gcd(b_nonunit, N)} so {b_nonunit} has NO inverse "
            f"(division undefined, demonstrated); 7*2^-1 mod N = {ring_result} != 3 "
            f"(a valid ring element, NOT integer division -- demonstrated, not hidden)"
        ),
        "n_sampled": n_samples,
        "sampled_passed": sampled_ok,
    }


# ---------------------------------------------------------------------------
# Plot-data generation (consumed by track_a_numeral_crt/proofs/make_plots.py)
# ---------------------------------------------------------------------------


def generate_plot_data(primes: list[int], N: int, consts: list[int], seed: int, n_samples: int = 3000) -> dict:
    rng = np.random.default_rng(seed)
    n_sample = rng.integers(0, N, size=n_samples, dtype=np.int64)
    residues = np.stack([n_sample % p for p in primes], axis=1)

    ks = [1, 100, 5000]
    a_sample = rng.integers(0, N, size=n_samples, dtype=np.int64)
    scatter = {}
    for k in ks:
        mats = encode_onehot(a_sample, primes)
        shifted = shift_onehot(mats, k, primes)
        decoded = decode_onehot(shifted, primes, N, consts)
        expected = (a_sample + k) % N
        scatter[str(k)] = {
            "expected": expected.tolist(),
            "decoded": decoded.tolist(),
            "max_abs_error": int(np.max(np.abs(decoded - expected))),
        }

    return {
        "primes": primes,
        "N": N,
        "n_sample": n_sample.tolist(),
        "residues": residues.tolist(),
        "shift_scatter": scatter,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all(seed: int = 0, n_random_pairs: int = 1_000_000) -> dict:
    seed_everything(seed)
    t0 = time.time()
    checks = []

    pos_N, pos_consts = crt_constants(POSITION_PRIMES)
    val_N, val_consts = crt_constants(VALUE_PRIMES)
    assert pos_N == 42 and val_N == 1_062_347, "modulus products drifted from the documented values"

    checks.append(check_bijection(POSITION_PRIMES, pos_N, "position"))
    checks.append(check_bijection(VALUE_PRIMES, val_N, "value"))
    checks.append(check_additive_per_modulus(POSITION_PRIMES))
    checks.append(check_additive_per_modulus(VALUE_PRIMES))
    checks.append(check_full_addition(VALUE_PRIMES, val_N, val_consts, n_random_pairs, seed))
    checks.append(check_multiplicative_per_modulus(POSITION_PRIMES))
    checks.append(check_multiplicative_per_modulus(VALUE_PRIMES))
    checks.append(check_full_multiplication(VALUE_PRIMES, val_N, val_consts))
    checks.append(check_subtraction(VALUE_PRIMES, val_N, val_consts))
    checks.append(check_division_by_units(VALUE_PRIMES, val_N, val_consts))

    elapsed = time.time() - t0
    all_passed = all(c["passed"] for c in checks)

    report = {
        "seed": seed,
        "elapsed_seconds": elapsed,
        "position_primes": POSITION_PRIMES,
        "position_N": pos_N,
        "value_primes": VALUE_PRIMES,
        "value_N": val_N,
        "all_passed": all_passed,
        "checks": checks,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "analytic_proof_report.json").write_text(json.dumps(report, indent=2))

    plot_data = generate_plot_data(VALUE_PRIMES, val_N, val_consts, seed)
    (RESULTS_DIR / "proof_plot_data.json").write_text(json.dumps(plot_data))

    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--samples", type=int, default=1_000_000, help="random (a,b) pairs for the sampled addition check"
    )
    args = parser.parse_args()

    report = run_all(seed=args.seed, n_random_pairs=args.samples)

    print(f"Kronecker Numeral Embedding -- analytic proof ({report['elapsed_seconds']:.2f}s)")
    print(f"  position code: primes={report['position_primes']} N={report['position_N']}")
    print(f"  value code:    primes={report['value_primes']} N={report['value_N']}")
    print()
    for c in report["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        exhaustive = "exhaustive" if c["exhaustive"] else "sampled"
        print(f"  [{status}] {c['check']} ({exhaustive}, n={c['n_tested']})")
        print(f"         {c['detail']}")
    print()
    print(f"ALL CHECKS {'PASSED' if report['all_passed'] else 'FAILED'}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
