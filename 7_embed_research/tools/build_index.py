"""Renders the top-level submission_artifacts/index.html landing page:
links + summary cards for both tracks. Run after both tracks' dashboards
have been built.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.evidence import load_json  # noqa: E402
from tools.chart_kit import PALETTE_CSS  # noqa: E402

TRACK_A = ROOT / "track_a_numeral_crt"
TRACK_B = ROOT / "track_b_holographic_binding"


def summarize_track_a() -> dict:
    results_dir = TRACK_A / "results"
    proof = load_json(results_dir / "analytic_proof_report.json")
    all_pass = all(c["passed"] for c in proof["checks"])
    return {
        "title": "Track A — Kronecker Numeral Embeddings",
        "tagline": "Arithmetic-preserving embeddings via a Residue Number System (CRT) decomposition.",
        "proof_status": f"{sum(c['passed'] for c in proof['checks'])}/{len(proof['checks'])} checks pass"
        + (" — exact on the full tested range" if all_pass else " — SEE FAILURES"),
        "link": "../track_a_numeral_crt/submission_artifacts/dashboard.html",
    }


def summarize_track_b() -> dict:
    results_dir = TRACK_B / "results"
    proof = load_json(results_dir / "capacity_proof_report.json")
    return {
        "title": "Track B — Holographic/Fourier Binding",
        "tagline": "Circular-convolution superposition replacing the fixed 32-slot Kronecker/tensor-product cap.",
        "proof_status": "single-pair unbind exact"
        + (" (verified)" if proof.get("single_pair_unbind_exact") else " -- NOT VERIFIED"),
        "link": "../track_b_holographic_binding/submission_artifacts/dashboard.html",
    }


def main() -> int:
    a = summarize_track_a()
    b = summarize_track_b()

    body = f"""<title>Kronecker Embedding V2</title>
<style>
{PALETTE_CSS}
body {{ margin: 0; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.page {{ max-width: 820px; margin: 0 auto; padding: 48px 20px 60px 20px; }}
h1 {{ font-size: 1.8rem; margin: 0 0 6px 0; }}
.subtitle {{ color: var(--text-secondary); margin: 0 0 36px 0; max-width: 60ch; }}
.card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 24px 26px; margin: 0 0 18px 0; text-decoration: none; display: block; color: inherit; }}
.card:hover {{ border-color: var(--series-1); }}
.card h2 {{ font-size: 1.15rem; margin: 0 0 6px 0; }}
.card p {{ color: var(--text-secondary); margin: 0 0 10px 0; }}
.status {{ font-size: 0.82rem; color: var(--good); font-weight: 600; }}
.footer {{ margin-top: 30px; font-size: 0.82rem; color: var(--text-muted); }}
.footer a {{ color: var(--text-secondary); }}
</style>
<div class="viz-root page">
<h1>Kronecker Embedding V2</h1>
<p class="subtitle">Two of the instructor's five extension problems, proven separately, each with a
standalone analytic proof (no model required) and a small trained transformer testing whether the
construction helps in practice.</p>

<a class="card" href="{a['link']}">
<h2>{a['title']}</h2>
<p>{a['tagline']}</p>
<div class="status">{a['proof_status']}</div>
</a>

<a class="card" href="{b['link']}">
<h2>{b['title']}</h2>
<p>{b['tagline']}</p>
<div class="status">{b['proof_status']}</div>
</a>

<div class="footer">
<a href="../README.md">README</a> ·
<a href="../ARCHITECTURE.md">Architecture &amp; design decisions</a> ·
<a href="../CITATIONS.md">Citations</a> ·
<a href="evidence.json">Evidence (raw JSON)</a>
</div>
</div>
"""

    out = ROOT / "submission_artifacts" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
