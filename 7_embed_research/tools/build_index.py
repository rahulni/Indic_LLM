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
from tools.chart_kit import (  # noqa: E402
    PALETTE_CSS,
    REPO_EVIDENCE_URL,
    REPO_TREE_URL,
    THEME_BOOT_JS,
    THEME_TOGGLE_HTML,
    THEME_TOGGLE_JS,
)

TRACK_A = ROOT / "track_a_numeral_crt"
TRACK_B = ROOT / "track_b_holographic_binding"


def evidence_label() -> str:
    """Counts PASS rows off the artifact rather than hardcoding a total, so the
    landing page cannot keep advertising a pass rate the evidence no longer shows.
    """
    path = ROOT / "submission_artifacts" / "evidence.json"
    if not path.exists():  # build_index.py is runnable standalone, before run_all.py
        return "Evidence (raw JSON)"
    rows = load_json(path)
    n_pass = sum(1 for r in rows if r["result"] == "PASS")
    return f"Evidence ({n_pass}/{len(rows)} PASS)"


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
<script>{THEME_BOOT_JS}</script>
<style>
{PALETTE_CSS}
body {{ margin: 0; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.62; -webkit-font-smoothing: antialiased; }}
.page {{ max-width: 820px; margin: 0 auto; padding: 64px 20px 64px 20px; }}
h1 {{ font-size: 2.4rem; line-height: 1.12; letter-spacing: -0.025em; margin: 0 0 12px 0; }}
.subtitle {{ color: var(--text-secondary); margin: 0 0 40px 0; max-width: 62ch; font-size: 1.02rem; }}
.card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 16px;
  padding: 26px 28px; margin: 0 0 16px 0; text-decoration: none; display: block; color: inherit;
  box-shadow: var(--shadow); transition: border-color 0.12s, transform 0.12s; }}
.card:hover {{ border-color: var(--border-strong); transform: translateY(-2px); }}
.card h2 {{ font-size: 1.2rem; letter-spacing: -0.01em; margin: 0 0 7px 0; }}
.card p {{ color: var(--text-secondary); margin: 0 0 14px 0; }}
.status {{ font-size: 0.8rem; color: var(--good); font-weight: 600;
  display: inline-flex; align-items: center; gap: 7px; }}
.status::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: currentColor; flex: none; }}
.footer {{ margin-top: 36px; padding-top: 18px; border-top: 1px solid var(--border);
  font-size: 0.82rem; color: var(--text-muted); }}
.footer a {{ color: var(--text-secondary); }}
.footer-source {{ margin-top: 8px; }}
.theme-toggle {{ position: fixed; top: 16px; right: 16px; z-index: 20;
  width: 38px; height: 38px; border-radius: 10px; cursor: pointer;
  background: var(--surface-1); border: 1px solid var(--border-strong); color: var(--text-secondary);
  font-size: 15px; line-height: 1; display: flex; align-items: center; justify-content: center;
  box-shadow: var(--shadow); }}
.theme-toggle:hover {{ color: var(--text-primary); border-color: var(--text-muted); }}
.theme-toggle .icon-dark {{ display: inline; }}
.theme-toggle .icon-light {{ display: none; }}
:root[data-theme="light"] .theme-toggle .icon-dark {{ display: none; }}
:root[data-theme="light"] .theme-toggle .icon-light {{ display: inline; }}
@media (max-width: 620px) {{
  .page {{ padding: 44px 18px 52px 18px; }}
  h1 {{ font-size: 1.8rem; }}
  .theme-toggle {{ top: 10px; right: 10px; }}
}}
</style>
{THEME_TOGGLE_HTML}
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
<a href="readme.html">README</a> ·
<a href="../ARCHITECTURE.md">Architecture &amp; design decisions</a> ·
<a href="../CITATIONS.md">Citations</a>
<div class="footer-source">Source: <a href="{REPO_TREE_URL}">Code + README</a> ·
<a href="{REPO_EVIDENCE_URL}">{evidence_label()}</a></div>
</div>
</div>
<script>{THEME_TOGGLE_JS}</script>
"""

    out = ROOT / "submission_artifacts" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
