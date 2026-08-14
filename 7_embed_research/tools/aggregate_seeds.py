"""Aggregates multi-seed runs into mean +/- std, so the READMEs and dashboards
report a distribution rather than a single lucky (or unlucky) seed.

Reads whatever seed files exist on disk and says exactly how many it found --
it never assumes a seed count, so a partial sweep is reported as partial
rather than silently averaged over fewer runs than claimed.

    python tools/aggregate_seeds.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACK_A = ROOT / "track_a_numeral_crt"
TRACK_B = ROOT / "track_b_holographic_binding"

# (label, filename stem without the _seedN suffix)
TRACK_A_CONFIGS = [
    ("baseline", "baseline_add_default"),
    ("abacus", "abacus_add_default"),
    ("crt", "crt_add_default"),
    ("abacus+offset", "abacus_offset_add_default"),
    ("crt+offset", "crt_offset_add_default"),
]
TRACK_B_CONFIGS = [("kronecker", "kronecker_default"), ("holographic", "holographic_default")]


def seed_files(results_dir: Path, stem: str) -> list[Path]:
    """seed 0 has no suffix (historical naming); seeds 1+ carry _seedN."""
    found = []
    base = results_dir / f"{stem}.json"
    if base.exists():
        found.append(base)
    found.extend(sorted(results_dir.glob(f"{stem}_seed*.json")))
    return found


def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate_track_a() -> dict:
    out = {}
    for label, stem in TRACK_A_CONFIGS:
        paths = seed_files(TRACK_A / "results", stem)
        if not paths:
            continue
        runs = [json.loads(p.read_text()) for p in paths]
        by_length = {}
        for length in sorted({int(k) for r in runs for k in r["final_eval_by_length"]}):
            accs = [
                r["final_eval_by_length"][str(length)]["accuracy"]
                for r in runs
                if str(length) in r["final_eval_by_length"]
            ]
            by_length[str(length)] = mean_std(accs)
        out[label] = {
            "seeds": [r["seed"] for r in runs],
            "n_seeds": len(runs),
            "eval_n_per_length": runs[0]["final_eval_by_length"]["1"]["n"],
            "n_params": runs[0]["n_params"],
            "random_offset_max": runs[0].get("random_offset_max", 0),
            "accuracy_by_length": by_length,
        }
    return out


def aggregate_track_b() -> dict:
    out = {}
    for label, stem in TRACK_B_CONFIGS:
        paths = seed_files(TRACK_B / "results", stem)
        if not paths:
            continue
        runs = [json.loads(p.read_text()) for p in paths]
        out[label] = {
            "seeds": [r["seed"] for r in runs],
            "n_seeds": len(runs),
            "val_perplexity": mean_std([r["final_val_perplexity"] for r in runs]),
            "n_params_total": runs[0]["n_params_total"],
            "n_params_embedding_learned": runs[0]["n_params_embedding_learned"],
            "truncation_cosine_similarity": mean_std(
                [r["truncation_information_loss"]["mean_cosine_similarity"] for r in runs]
            ),
        }
    return out


def main() -> int:
    report = {"track_a": aggregate_track_a(), "track_b": aggregate_track_b()}
    out_path = ROOT / "submission_artifacts" / "seed_aggregate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("Track A -- exact-match accuracy by test digit-length (mean +/- std over seeds)")
    for label, data in report["track_a"].items():
        print(f"\n  {label}  (seeds {data['seeds']}, n={data['eval_n_per_length']}/length)")
        row = "    "
        for length, stats in data["accuracy_by_length"].items():
            row += f"L{length}: {stats['mean']:.1%}+/-{stats['std']:.1%}   "
        print(row)

    print("\nTrack B -- validation perplexity (mean +/- std over seeds)")
    for label, data in report["track_b"].items():
        p = data["val_perplexity"]
        t = data["truncation_cosine_similarity"]
        print(
            f"  {label:14s} seeds {data['seeds']}  ppl {p['mean']:.2f}+/-{p['std']:.2f}  "
            f"truncation-sim {t['mean']:.4f}  learned-embed-params {data['n_params_embedding_learned']:,}"
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
