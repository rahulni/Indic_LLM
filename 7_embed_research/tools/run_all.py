"""Orchestrates: proofs (always) -> evidence.json -> dashboards.

Training runs are NOT invoked by this script (they are opt-in, run
separately via each track's model/train.py, since they take minutes rather
than seconds) -- this script assumes results/*.json already exist for
whichever runs you want reflected in the evidence bundle, and grades
whatever is actually on disk. Missing runs simply show as FAIL rows rather
than being silently skipped.

    python tools/run_all.py

Exits non-zero if any evidence row fails.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.evidence import EvidenceRow, load_json, write_evidence  # noqa: E402

TRACK_A = ROOT / "track_a_numeral_crt"
TRACK_B = ROOT / "track_b_holographic_binding"


def run_proofs() -> None:
    print("=== running proofs ===")
    subprocess.run([sys.executable, str(TRACK_A / "proofs" / "analytic_proof.py")], check=True)
    subprocess.run([sys.executable, str(TRACK_A / "proofs" / "make_plots.py")], check=True)
    subprocess.run([sys.executable, str(TRACK_B / "proofs" / "capacity_proof.py")], check=True)
    subprocess.run([sys.executable, str(TRACK_B / "proofs" / "make_plots.py")], check=True)


def track_a_evidence_rows() -> list[EvidenceRow]:
    report_path = TRACK_A / "results" / "analytic_proof_report.json"
    rows = []

    def make_check(check_name: str):
        def _check():
            report = load_json(report_path)
            for c in report["checks"]:
                if c["check"] == check_name or c["check"].startswith(check_name):
                    return c["passed"], c["detail"]
            return False, f"check {check_name!r} not found in report"

        return _check

    for name in [
        "crt_bijection",
        "additive_homomorphism_per_modulus",
        "full_pipeline_addition",
        "multiplicative_homomorphism_via_discrete_log",
        "full_pipeline_multiplication",
        "full_pipeline_subtraction",
        "division_by_units_via_negated_log",
    ]:
        rows.append(
            EvidenceRow(
                requirement=f"Track A analytic proof: {name}",
                file=str(report_path.relative_to(ROOT)),
                compared="checks[].passed",
                check=make_check(name),
            )
        )

    for run_name in ["baseline_add", "baseline_mul", "abacus_add", "abacus_mul", "crt_add", "crt_mul"]:
        run_path = TRACK_A / "results" / f"{run_name}_default.json"

        def make_run_check(path: Path):
            def _check():
                if not path.exists():
                    return False, f"{path.name} does not exist -- run has not been executed"
                result = load_json(path)
                expected_lengths = {str(n) for n in range(1, 9)}
                actual = set(result.get("final_eval_by_length", {}).keys())
                ok = expected_lengths.issubset(actual)
                return ok, f"final_eval_by_length covers lengths {sorted(actual, key=int)}"

            return _check

        rows.append(
            EvidenceRow(
                requirement=f"Track A training run completed: {run_name} (default profile)",
                file=str(run_path.relative_to(ROOT)),
                compared="final_eval_by_length keys",
                check=make_run_check(run_path),
            )
        )

    # the random-offset experiment is the one that actually produced OOD
    # generalization, so it gets an evidence row asserting the effect is real
    # (not merely that the run happened)
    for arm in ["abacus", "crt"]:
        offset_path = TRACK_A / "results" / f"{arm}_offset_add_default.json"
        plain_path = TRACK_A / "results" / f"{arm}_add_default.json"

        def make_offset_check(off: Path, plain: Path, arm_name: str):
            def _check():
                if not off.exists():
                    return False, f"{off.name} does not exist -- offset run has not been executed"
                off_acc = load_json(off)["final_eval_by_length"]["5"]["accuracy"]
                if not plain.exists():
                    return False, f"{plain.name} missing -- cannot compare against the no-offset run"
                plain_acc = load_json(plain)["final_eval_by_length"]["5"]["accuracy"]
                return off_acc > plain_acc, (
                    f"{arm_name} OOD (length 5) accuracy: {off_acc:.1%} with random-offset training "
                    f"vs {plain_acc:.1%} without"
                )

            return _check

        rows.append(
            EvidenceRow(
                requirement=f"Track A: random-offset training improves OOD length generalization ({arm})",
                file=str(offset_path.relative_to(ROOT)),
                compared="final_eval_by_length['5'].accuracy vs the no-offset run",
                check=make_offset_check(offset_path, plain_path, arm),
            )
        )

    # the value-embedding experiment is the instructor's literal ask; its
    # evidence row records that it ran and reports the (negative) outcome
    for arm in ["crt_value", "learned"]:
        for task in ["add", "mul"]:
            vpath = TRACK_A / "results" / f"value_{arm}_{task}_default.json"

            def make_value_check(path: Path):
                def _check():
                    if not path.exists():
                        return False, f"{path.name} does not exist -- run has not been executed"
                    r = load_json(path)
                    acc = r["final_accuracy"]["accuracy"]
                    return "final_accuracy" in r, (
                        f"exact-match {acc:.1%} (n={r['final_accuracy']['n']}), "
                        f"embedding params {r['n_params_embedding']:,}"
                    )

                return _check

            rows.append(
                EvidenceRow(
                    requirement=f"Track A value-embedding run completed: {arm} / {task}",
                    file=str(vpath.relative_to(ROOT)),
                    compared="final_accuracy.accuracy",
                    check=make_value_check(vpath),
                )
            )
    return rows


def track_b_evidence_rows() -> list[EvidenceRow]:
    report_path = TRACK_B / "results" / "capacity_proof_report.json"
    rows = [
        EvidenceRow(
            requirement="Track B analytic proof: single-pair unbind is exact",
            file=str(report_path.relative_to(ROOT)),
            compared="single_pair_unbind_exact",
            check=lambda: (load_json(report_path)["single_pair_unbind_exact"], "assertion checked inside run_all()"),
        ),
        EvidenceRow(
            requirement="Track B analytic proof: decode accuracy at L=1 is perfect for every D",
            file=str(report_path.relative_to(ROOT)),
            compared="cells[L==1].accuracy",
            check=lambda: (
                all(c["accuracy"] == 1.0 for c in load_json(report_path)["cells"] if c["L"] == 1),
                "checked cells where L == 1 across the full D sweep",
            ),
        ),
    ]

    def check_theory_tracks_measurement():
        report = load_json(report_path)
        cells = [c for c in report["cells"] if "theoretical_accuracy" in c]
        if not cells:
            return False, "no theoretical_accuracy recorded -- the derived bound was not computed"
        deltas = [abs(c["accuracy"] - c["theoretical_accuracy"]) for c in cells]
        worst = max(deltas)
        return worst < 0.12, (
            f"our derived SNR bound predicts measured decode accuracy across all {len(cells)} "
            f"(L, D) cells to within {worst:.1%} worst-case, {sum(deltas) / len(deltas):.1%} mean"
        )

    rows.append(
        EvidenceRow(
            requirement="Track B: the derived capacity bound actually predicts the measurements",
            file=str(report_path.relative_to(ROOT)),
            compared="cells[].accuracy vs cells[].theoretical_accuracy",
            check=check_theory_tracks_measurement,
        )
    )

    def check_shift_roles():
        report = load_json(report_path)
        shift = report.get("shift_role_cells")
        if not shift:
            return False, "shift-role sweep missing"
        at_one = [c["accuracy"] for c in shift if c["L"] == 1]
        return all(a == 1.0 for a in at_one), (
            f"the instructor's literal 'character as a Fourier wave' (deterministic shift roles) "
            f"was swept across {len(shift)} (L, D) cells and is exact at L=1 for every D"
        )

    rows.append(
        EvidenceRow(
            requirement="Track B: the literal shift-role construction was tested, not just described",
            file=str(report_path.relative_to(ROOT)),
            compared="shift_role_cells[]",
            check=check_shift_roles,
        )
    )

    for arm in ["kronecker", "holographic"]:
        run_path = TRACK_B / "results" / f"{arm}_default.json"

        def make_run_check(path: Path):
            def _check():
                if not path.exists():
                    return False, f"{path.name} does not exist -- run has not been executed"
                result = load_json(path)
                ok = "final_val_perplexity" in result and "truncation_information_loss" in result
                return ok, f"final_val_perplexity={result.get('final_val_perplexity')}"

            return _check

        rows.append(
            EvidenceRow(
                requirement=f"Track B training run completed: {arm} (default profile)",
                file=str(run_path.relative_to(ROOT)),
                compared="final_val_perplexity, truncation_information_loss",
                check=make_run_check(run_path),
            )
        )

    kron_path = TRACK_B / "results" / "kronecker_default.json"

    def check_truncation_is_total():
        if not kron_path.exists():
            return False, "kronecker_default.json does not exist -- run has not been executed"
        sim = load_json(kron_path)["truncation_information_loss"]["mean_cosine_similarity"]
        return sim > 0.999, f"mean_cosine_similarity={sim:.6f} (expect ~1.0: identical after truncation)"

    rows.append(
        EvidenceRow(
            requirement="Track B: Kronecker arm's 32-char truncation is measurably total (similarity == 1.0)",
            file=str(kron_path.relative_to(ROOT)),
            compared="truncation_information_loss.mean_cosine_similarity",
            check=check_truncation_is_total,
        )
    )

    dress_path = TRACK_B / "results" / "holographic_dress_default.json"
    plain_holo_path = TRACK_B / "results" / "holographic_default.json"

    def check_dress_ablation():
        if not dress_path.exists():
            return False, "holographic_dress_default.json does not exist -- ablation not run"
        dressed = load_json(dress_path)["final_val_perplexity"]
        plain = load_json(plain_holo_path)["final_val_perplexity"] if plain_holo_path.exists() else None
        return True, (
            f"dressed holographic perplexity {dressed:.2f}"
            + (f" vs plain holographic {plain:.2f}" if plain is not None else "")
            + " -- tests whether the Kronecker arm's edge is explained by having a learned projection"
        )

    rows.append(
        EvidenceRow(
            requirement="Track B: the perplexity gap was probed with a learned-dressing ablation",
            file=str(dress_path.relative_to(ROOT)),
            compared="final_val_perplexity, dressed vs plain",
            check=check_dress_ablation,
        )
    )
    return rows


def seed_coverage_rows() -> list[EvidenceRow]:
    """Headline claims must rest on more than one seed. GPU runs here are not
    bit-reproducible (see ARCHITECTURE.md), so this row asserts that the
    aggregate actually covers multiple seeds rather than silently reporting a
    single lucky run."""
    agg_path = ROOT / "submission_artifacts" / "seed_aggregate.json"

    def _check():
        if not agg_path.exists():
            return False, "seed_aggregate.json missing -- run tools/aggregate_seeds.py"
        agg = load_json(agg_path)
        counts = {k: v["n_seeds"] for k, v in agg["track_a"].items()}
        counts.update({f"B/{k}": v["n_seeds"] for k, v in agg["track_b"].items()})
        multi = [k for k, n in counts.items() if n >= 2]
        return len(multi) == len(counts), (
            "seeds per config: " + ", ".join(f"{k}={n}" for k, n in counts.items())
        )

    return [
        EvidenceRow(
            requirement="Headline results are reported over multiple seeds, not a single run",
            file="submission_artifacts/seed_aggregate.json",
            compared="n_seeds per config",
            check=_check,
        )
    ]


def main() -> int:
    run_proofs()

    print("=== aggregating seeds ===")
    subprocess.run([sys.executable, str(ROOT / "tools" / "aggregate_seeds.py")], check=True)

    print("=== building evidence ===")
    rows = track_a_evidence_rows() + track_b_evidence_rows() + seed_coverage_rows()
    all_passed = write_evidence(rows, ROOT / "submission_artifacts" / "evidence.json")
    for row in rows:
        result = row.evaluate()
        print(f"  [{result['result']}] {result['requirement']}")

    print("=== building dashboards ===")
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_dashboard.py"), "--track", "a"], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_dashboard.py"), "--track", "b"], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_index.py")], check=True)

    print(f"\n{'ALL EVIDENCE PASSED' if all_passed else 'SOME EVIDENCE FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
