# -*- coding: utf-8 -*-
"""Assemble the GitHub Pages site: a landing page plus both dashboards.

    python tools/build_pages.py [output_dir]

GitHub serves a repository's HTML as source, not as a page, so a link to
``submission_artifacts/dashboard.html`` shows markup rather than the dashboard.
Pages is the fix: the same self-contained files, served over HTTP where a browser
renders them.

Nothing is regenerated here. The dashboards are copied verbatim from the
committed artifacts, so what is published is byte-identical to what is in the
repository -- publishing must not become a second place where numbers are
computed.
"""
from __future__ import annotations

import html
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tdes.hashing import read_json, write_text          # noqa: E402

# (source artifact dir, published filename, label, whether it is the default run)
RUNS = [
    ("submission_artifacts", "dashboard.html", "Default run", True),
    ("submission_artifacts_stdlib", "dashboard-stdlib.html", "Fallback run", False),
]


def esc(x) -> str:
    return html.escape(str(x))


def _summary(art: str) -> dict | None:
    """Read back the few figures the landing page states, from the same files the
    dashboard reads. Returns None when a run was never produced."""
    try:
        ev = read_json(os.path.join(art, "evidence.json"))
    except Exception:
        return None
    rep = read_json(os.path.join(art, "reports.json"))
    meta = read_json(os.path.join(art, "run_meta.json"))
    perf = read_json(os.path.join(art, "performance.json"))
    model = rep.get("model", {})
    s = ev.get("summary", {})
    return {
        "passed": s.get("passed"), "total": s.get("total"),
        "all_passed": bool(s.get("all_passed")),
        "backend": model.get("backend", "stdlib"),
        "arch": (f"{model['n_layers']}-layer transformer, "
                 f"{model['d_model']} dim, {model['n_heads']} heads"
                 if model.get("backend") == "torch"
                 else f"neural n-gram, context {model.get('k', '?')}"),
        "params": model.get("parameters_count"),
        # "NVIDIA GeForce RTX 3070 Laptop GPU" wraps to two lines in the card and
        # the vendor prefix carries no information the reader needs.
        "hardware": (model.get("gpu_name") or model.get("device", "cpu"))
                    .replace("NVIDIA GeForce ", ""),
        "vocab": rep.get("tokenizer", {}).get("vocab_size"),
        "seconds": meta.get("wall_clock_seconds"),
        "tok_per_s": perf.get("rates", {}).get(
            "useful_loss_bearing_tokens_per_second"),
        "profile": meta.get("profile"),
    }


def build(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    cards, written = [], []

    for art_name, published, label, is_default in RUNS:
        art = os.path.join(ROOT, art_name)
        src = os.path.join(art, "dashboard.html")
        info = _summary(art)
        if not (os.path.isfile(src) and info):
            continue
        shutil.copyfile(src, os.path.join(out_dir, published))
        written.append(published)
        cards.append(f"""
    <a class="card{' primary' if is_default else ''}" href="{published}">
      <div class="row">
        <h2>{esc(label)}</h2>
        <span class="verdict {'ok' if info['all_passed'] else 'bad'}">
          {esc(info['passed'])}/{esc(info['total'])}</span>
      </div>
      <p class="arch">{esc(info['arch'])}</p>
      <dl>
        <div><dt>Parameters</dt><dd>{info['params']:,}</dd></div>
        <div><dt>Vocabulary</dt><dd>{info['vocab']:,}</dd></div>
        <div><dt>Ran on</dt><dd>{esc(info['hardware'])}</dd></div>
        <div><dt>Useful tokens/sec</dt><dd>{info['tok_per_s']:,.0f}</dd></div>
        <div><dt>Wall clock</dt><dd>{info['seconds']:,.0f}s</dd></div>
        <div><dt>Profile</dt><dd><code>{esc(info['profile'])}</code></dd></div>
      </dl>
      <span class="go">Open the dashboard &rarr;</span>
    </a>""")

    if not cards:
        raise SystemExit("no dashboards found -- run run_demo.py first")

    write_text(os.path.join(out_dir, "index.html"),
               INDEX.replace("{{CARDS}}", "".join(cards)))
    # Tell Pages not to run the files through Jekyll, which would otherwise skip
    # anything it does not recognise and can mangle inline content.
    write_text(os.path.join(out_dir, ".nojekyll"), "")
    return ["index.html", ".nojekyll"] + written


INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training Data Execution System</title>
<style>
:root{--bg:#f7f8fa;--fg:#14171c;--mut:#5f6875;--dim:#8b94a1;--card:#fff;
      --line:#e2e6ec;--ok:#1a7f37;--bad:#c02718;--accent:#3d6fa8;
      --shadow:0 1px 2px rgba(16,24,40,.05),0 10px 30px -14px rgba(16,24,40,.22)}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--fg:#e6e9ef;--mut:#98a2b1;
      --dim:#6f7987;--card:#171c23;--line:#252c36;--ok:#48b95e;--bad:#f2685c;
      --accent:#79aae0;--shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px -16px rgba(0,0,0,.75)}}
*{box-sizing:border-box}
body{margin:0;padding:0 20px 64px;background:var(--bg);color:var(--fg);
  font:15px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
header,main,footer{max-width:900px;margin:0 auto}
header{padding:56px 0 8px}
.eyebrow{margin:0 0 8px;color:var(--dim);font-size:.76rem;font-weight:600;
  letter-spacing:.11em;text-transform:uppercase}
h1{font-size:2.2rem;line-height:1.1;margin:0 0 12px;letter-spacing:-.026em;font-weight:680}
.lede{color:var(--mut);font-size:1.02rem;max-width:60ch;margin:0 0 8px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px;
  margin-top:26px}
.card{display:block;text-decoration:none;color:inherit;background:var(--card);
  border:1px solid var(--line);border-radius:14px;padding:20px 22px;
  box-shadow:var(--shadow);transition:transform .13s,border-color .13s}
.card:hover{transform:translateY(-2px);border-color:var(--accent)}
.card.primary{border-color:color-mix(in srgb,var(--accent) 45%,var(--line))}
.row{display:flex;align-items:center;justify-content:space-between;gap:12px}
h2{font-size:1.12rem;margin:0;letter-spacing:-.015em}
.verdict{font-size:.82rem;font-weight:700;padding:3px 10px;border-radius:20px;white-space:nowrap}
.verdict.ok{color:var(--ok);background:color-mix(in srgb,var(--ok) 13%,transparent)}
.verdict.bad{color:var(--bad);background:color-mix(in srgb,var(--bad) 13%,transparent)}
.arch{color:var(--mut);font-size:.89rem;margin:6px 0 14px}
dl{margin:0 0 16px;display:grid;grid-template-columns:1fr 1fr;gap:7px 18px}
dl div{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dotted var(--line);
  padding-bottom:4px}
dt{color:var(--dim);font-size:.78rem}
dd{margin:0;font-size:.85rem;font-variant-numeric:tabular-nums;font-weight:600}
code{font:.86em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:color-mix(in srgb,var(--fg) 8%,transparent);padding:1px 5px;border-radius:4px}
.go{color:var(--accent);font-size:.87rem;font-weight:600}
footer{padding-top:34px;color:var(--mut);font-size:.87rem}
footer a{color:var(--accent)}
@media (max-width:560px){h1{font-size:1.7rem}dl{grid-template-columns:1fr}}
</style></head><body>
<header>
  <p class="eyebrow">Training data, accounted for</p>
  <h1>Training Data Execution System</h1>
  <p class="lede">A training data pipeline that can prove what it did: immutable
  shards, a double-entry ledger, and a crash that resumes on the exact next batch.
  Two model backends drive the identical data path.</p>
</header>
<main>{{CARDS}}</main>
<footer>
  <p>These pages are the dashboards committed in the repository, served verbatim
  so a browser renders them. Every figure on them is read back out of that run's
  artifacts.</p>
  <p><a href="https://github.com/rahulni/Indic_LLM/tree/main/6_build_dataset">Source, artifacts and README on GitHub</a></p>
</footer>
</body></html>
"""


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "site")
    files = build(out)
    total = sum(os.path.getsize(os.path.join(out, f)) for f in files)
    print(f"wrote {len(files)} files to {out} ({total / 1e6:.1f} MB)")
    for f in files:
        print(f"  {f}")
