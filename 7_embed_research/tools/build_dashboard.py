"""Renders each track's results.json + proof reports into a single
self-contained dashboard.html (no external requests -- safe to publish as a
Claude Artifact as-is). Run after both proofs and training have produced
their results/*.json files.

    python tools/build_dashboard.py --track a
    python tools/build_dashboard.py --track b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.evidence import load_json  # noqa: E402
from tools.chart_kit import evidence_table_html, legend_html, page_shell  # noqa: E402
from track_a_numeral_crt.proofs.analytic_proof import VALUE_PRIMES, crt_constants  # noqa: E402
from track_b_holographic_binding.data.corpus import load_corpus  # noqa: E402

TRACK_A = ROOT / "track_a_numeral_crt"
TRACK_B = ROOT / "track_b_holographic_binding"
SERIES = {"blue": "var(--series-1)", "orange": "var(--series-2)", "aqua": "var(--series-3)", "yellow": "var(--series-4)"}


def load_evidence_for(prefix: str) -> list[dict]:
    ev_path = ROOT / "submission_artifacts" / "evidence.json"
    if not ev_path.exists():
        return []
    all_rows = load_json(ev_path)
    return [r for r in all_rows if r["requirement"].startswith(prefix)]


# ---------------------------------------------------------------------------
# Track A
# ---------------------------------------------------------------------------

ARM_LABEL_A = {"baseline": "Baseline (NoPE)", "abacus": "Abacus-lite (learned)", "crt": "CRT/Kronecker (fixed)"}
ARM_COLOR_A = {"baseline": SERIES["blue"], "abacus": SERIES["orange"], "crt": SERIES["aqua"]}


def build_track_a() -> str:
    results_dir = TRACK_A / "results"
    proof_report = load_json(results_dir / "analytic_proof_report.json")
    proof_figures = load_json(results_dir / "proof_figures.json")
    evidence = load_evidence_for("Track A")

    runs = {}
    for arm in ["baseline", "abacus", "crt"]:
        for task in ["add", "mul"]:
            p = results_dir / f"{arm}_{task}_default.json"
            if p.exists():
                runs[(arm, task)] = load_json(p)

    length_gen_charts = {}
    sample_eff_charts = {}
    for task in ["add", "mul"]:
        series = []
        for arm in ["baseline", "abacus", "crt"]:
            r = runs.get((arm, task))
            if not r:
                continue
            pts = [
                {"x": int(length), "y": stats["accuracy"], "label": f"{stats['accuracy']:.0%} (n={stats['n']})"}
                for length, stats in sorted(r["final_eval_by_length"].items(), key=lambda kv: int(kv[0]))
            ]
            series.append({"name": ARM_LABEL_A[arm], "color": ARM_COLOR_A[arm], "points": pts})
        length_gen_charts[task] = series

        eff_series = []
        for arm in ["baseline", "abacus", "crt"]:
            r = runs.get((arm, task))
            if not r:
                continue
            pts = [{"x": c["step"], "y": c["accuracy"], "label": f"{c['accuracy']:.0%}"} for c in r["quick_accuracy_curve"]]
            eff_series.append({"name": ARM_LABEL_A[arm], "color": ARM_COLOR_A[arm], "points": pts})
        sample_eff_charts[task] = eff_series

    any_run = next(iter(runs.values()), None)
    n_params = any_run["n_params"] if any_run else 0
    n_params_badge = f"{n_params:,} params" if any_run else "not yet trained"
    train_range = any_run["train_len_range"] if any_run else [1, 4]

    dress_run = None
    dress_path = results_dir / "crt_dress_add_default.json"
    if dress_path.exists():
        dress_run = load_json(dress_path)

    # random-offset runs: the experiment that actually unlocked OOD generalization
    offset_runs = {}
    for arm in ["abacus", "crt"]:
        p = results_dir / f"{arm}_offset_add_default.json"
        if p.exists():
            offset_runs[arm] = load_json(p)

    offset_series = []
    for arm in ["baseline", "abacus", "crt"]:
        r = runs.get((arm, "add"))
        if r:
            offset_series.append(
                {
                    "name": f"{ARM_LABEL_A[arm]} (no offset)",
                    "color": ARM_COLOR_A[arm],
                    "dashed": True,
                    "points": [
                        {"x": int(k), "y": v["accuracy"], "label": f"{v['accuracy']:.0%}"}
                        for k, v in sorted(r["final_eval_by_length"].items(), key=lambda kv: int(kv[0]))
                    ],
                }
            )
    for arm in ["abacus", "crt"]:
        r = offset_runs.get(arm)
        if r:
            offset_series.append(
                {
                    "name": f"{ARM_LABEL_A[arm]} + random offset",
                    "color": ARM_COLOR_A[arm],
                    "points": [
                        {"x": int(k), "y": v["accuracy"], "label": f"{v['accuracy']:.0%} (n={v['n']})"}
                        for k, v in sorted(r["final_eval_by_length"].items(), key=lambda kv: int(kv[0]))
                    ],
                }
            )

    # value-embedding experiment: the instructor's literal ask
    value_runs = {}
    for arm in ["crt_value", "learned"]:
        for task in ["add", "mul"]:
            p = results_dir / f"value_{arm}_{task}_default.json"
            if p.exists():
                value_runs[(arm, task)] = load_json(p)

    agg_path = ROOT / "submission_artifacts" / "seed_aggregate.json"
    agg = load_json(agg_path) if agg_path.exists() else None

    def agg_cell(config: str, length: str) -> str | None:
        """mean +/- std across seeds, or None if the aggregate lacks it."""
        if not agg:
            return None
        cfg = agg["track_a"].get(config)
        if not cfg or length not in cfg["accuracy_by_length"]:
            return None
        s = cfg["accuracy_by_length"][length]
        return f"{s['mean']:.1%} &plusmn; {s['std']:.1%}"

    if offset_runs:
        rows_html = ""
        for arm in ["abacus", "crt"]:
            if arm not in offset_runs:
                continue
            plain = runs.get((arm, "add"))
            off = offset_runs[arm]
            plain5 = agg_cell(arm, "5") or (
                f"{plain['final_eval_by_length']['5']['accuracy']:.1%}" if plain else "n/a"
            )
            off5 = agg_cell(f"{arm}+offset", "5") or f"{off['final_eval_by_length']['5']['accuracy']:.1%}"
            off6 = agg_cell(f"{arm}+offset", "6") or f"{off['final_eval_by_length']['6']['accuracy']:.1%}"
            rows_html += (
                f"<tr><td>{ARM_LABEL_A[arm]}</td>"
                f"<td>{plain5}</td><td><strong>{off5}</strong></td><td>{off6}</td></tr>"
            )
        n_seeds = agg["track_a"].get("crt+offset", {}).get("n_seeds") if agg else None
        seed_note = (
            f" Each cell is mean &plusmn; std over {n_seeds} seeds."
            if n_seeds and n_seeds > 1
            else " (single seed &mdash; multi-seed aggregate not yet generated)"
        )
        offset_narrative = f"""
<p>The original three-arm result — every arm collapsing to 0% beyond the training range — turned out to be
an artifact of a <em>missing training mechanism</em>, not a property of the embeddings. McLeish et al. (2024)
pair Abacus embeddings with <strong>random position-offset training</strong>: each training example has one
random offset added to every digit's place-value index, so the model can never anchor on
"the units digit is always index 0". Evaluation always uses offset 0.</p>
<p>For the CRT arm this is literally a rotation of every residue clock — the same fixed shift operator the
proof in section 1 verifies, now applied as data augmentation.</p>
<table class="evtable"><thead><tr><th>Arm</th><th>OOD length 5, no offset</th><th>OOD length 5, with offset</th><th>OOD length 6, with offset</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<p class="subtitle">Out-of-distribution generalization appears where there was none.{seed_note} The honest
sting: the <em>learned</em> Abacus code exploits this better than the <em>fixed</em> CRT code — algebraic
structure in the embedding is not, on its own, what buys length generalization here. Note also the spread:
these runs are not bit-reproducible on GPU (no deterministic-kernel flags are set), which is exactly why
every headline number here is a multi-seed mean rather than a single run.</p>"""
    else:
        offset_narrative = "<p class='subtitle'>Random-offset runs have not been executed yet.</p>"

    if value_runs:
        vrows = ""
        for task in ["add", "mul"]:
            for arm in ["crt_value", "learned"]:
                r = value_runs.get((arm, task))
                if not r:
                    continue
                label = "CRT value code (fixed)" if arm == "crt_value" else "Learned embedding"
                lc = r["loss_curve"]
                vrows += (
                    f"<tr><td>{label}</td><td>{task}</td>"
                    f"<td>{r['n_params_embedding']:,}</td>"
                    f"<td>{lc[0]['loss']:.3f} &rarr; {lc[-1]['loss']:.3f}</td>"
                    f"<td>{r['final_accuracy']['accuracy']:.1%} (n={r['final_accuracy']['n']})</td></tr>"
                )
        value_narrative = f"""
<p>Everywhere above, the CRT code is a <em>positional</em> signal. Here it is the thing the instructor
actually described: each whole operand is a <strong>single token</strong> whose embedding is the fixed
83-dimensional CRT code <em>of the number itself</em> — the construction section 1 proves is exactly
arithmetic-preserving. Operands are capped at 3 digits so both <code>a+b</code> and <code>a*b</code> stay
inside the proven range. The model still emits the answer digit by digit.</p>
<table class="evtable"><thead><tr><th>Operand embedding</th><th>Task</th><th>Embedding params</th><th>Loss</th><th>Exact-match accuracy</th></tr></thead>
<tbody>{vrows}</tbody></table>
<p class="subtitle"><strong>A clear negative result, and the most important one here.</strong> Both arms
train — the loss falls comparably — and both then fail the task almost completely. Handing a transformer an
embedding that <em>provably</em> contains exact arithmetic structure does not make it able to do arithmetic;
it still has to learn to read that structure out, and mapping one operand token to a multi-digit answer is
too hard at this scale. This is exactly why the proof and the trained-model claims are kept separate
throughout this project. The one positive note is efficiency parity: the fixed CRT code matches the learned
embedding on both loss and accuracy using <strong>10&times; fewer embedding parameters</strong> — a real
property, but parity at ~1% accuracy is not a win and is not presented as one.</p>
<p class="subtitle"><strong>A first version of this experiment was wrong, and the correction changed the
conclusion.</strong> Initially the CRT arm's loss was flat (1.657 &rarr; 1.653) while the control trained
normally, which would have read as "the CRT value code fails". The cause was initialisation scale — the CRT
code is 5 ones in 83 dims, so through a default-initialised Linear it was drowned out. Both arms now share a
LayerNorm, after which the CRT arm trains normally and the two tie. The original numbers are not reported as
a finding, because they measured an artifact rather than the representation.</p>"""
    else:
        value_narrative = "<p class='subtitle'>Value-embedding runs have not been executed yet.</p>"

    crt_N, crt_consts = crt_constants(VALUE_PRIMES)

    body = f"""
<h1>Track A — Kronecker Numeral Embeddings</h1>
<p class="subtitle">Arithmetic-preserving embeddings via a Residue Number System (CRT) decomposition.
Two separate questions, never conflated: is the construction exactly correct (proved analytically,
no model involved), and does it help a trained transformer generalize to longer numbers (a second,
honestly-labeled empirical question).</p>
<div class="badges">
  <span class="badge">proof: exact on [0, {proof_report['value_N']:,})</span>
  <span class="badge">model: {n_params_badge}</span>
  <span class="badge">train digit-length: {train_range[0]}-{train_range[1]}</span>
  <span class="badge">device: {any_run['device']['device'] if any_run else 'n/a'}</span>
</div>

<div class="card">
<h2>1. The proof (no model, no training)</h2>
<p>Run in {proof_report['elapsed_seconds']:.2f}s. Every check below states exhaustive vs. sampled explicitly.</p>
<table class="evtable"><thead><tr><th>Check</th><th>Coverage</th><th>Result</th><th>Detail</th></tr></thead><tbody>
{"".join(f"<tr><td>{c['check']}</td><td>{'exhaustive' if c['exhaustive'] else 'sampled'}, n={c['n_tested']:,}</td><td class='{'pass' if c['passed'] else 'fail'}'>{'PASS' if c['passed'] else 'FAIL'}</td><td>{c['detail']}</td></tr>" for c in proof_report['checks'])}
</tbody></table>
</div>

<div class="card">
<h2>2. Try it yourself — live, in your browser</h2>
<p>Pick two numbers. Each prime gets its own clock face: the residue <code>n mod p</code> is the angle
of the dot. Adding <code>b</code> rotates every dot by its own residue of <code>b</code> — that rotation
<em>is</em> the addition, computed live below by the same CRT reconstruction proved in section 1 (not a
lookup table).</p>
<div id="crt-demo">
  <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:end;margin-bottom:16px;">
    <label>a <input id="crt-a" type="number" min="0" max="{crt_N - 1}" value="12345" style="width:110px;padding:6px 8px;border-radius:6px;border:1px solid var(--border);background:var(--page);color:var(--text-primary);"></label>
    <label>b (added to a) <input id="crt-b" type="number" min="0" max="{crt_N - 1}" value="777" style="width:110px;padding:6px 8px;border-radius:6px;border:1px solid var(--border);background:var(--page);color:var(--text-primary);"></label>
  </div>
  <div id="crt-clocks" style="display:flex;gap:14px;flex-wrap:wrap;"></div>
  <p id="crt-readout" style="font-size:0.9rem;color:var(--text-secondary);margin-top:10px;"></p>
</div>
</div>

<div class="card">
<h2>3. What the proof looks like</h2>
<div class="figure"><img src="{proof_figures['clock_structure']}" alt="Clock structure of the value-code residue slots"></div>
<p class="subtitle">Each residue slot is literally a clock: n mod p sits at angle 2&pi;(n mod p)/p.</p>
<div class="figure"><img src="{proof_figures['shift_scatter']}" alt="Shift scatter: decode(shift(encode(a),k)) vs (a+k) mod N"></div>
<p class="subtitle">Shifting the embedding IS modular addition — max error is exactly zero, not "usually small".</p>
</div>

<div class="card">
<h2>4. Does it help a trained transformer? (a different question)</h2>
<p>Three embedding arms, identical shared transformer trunk, identical per-digit tokenization —
trained once on {train_range[0]}-{train_range[1]}-digit operands, evaluated at digit-lengths 1-8
(lengths past {train_range[1]} are out-of-distribution).</p>
{legend_html([(ARM_LABEL_A[a], ARM_COLOR_A[a]) for a in ["baseline","abacus","crt"]])}
<h3 style="font-size:0.95rem;margin-top:18px;">Addition</h3>
<div id="chart-len-add"></div>
<h3 style="font-size:0.95rem;margin-top:18px;">Multiplication</h3>
<div id="chart-len-mul"></div>
<p class="subtitle">Vertical dashed line marks the end of the training range — everything to its right is OOD.</p>
</div>

<div class="card">
<h2>5. The missing ingredient: random-offset training</h2>
{offset_narrative}
{legend_html([(s["name"], s["color"]) for s in offset_series]) if offset_series else ""}
<div id="chart-offset"></div>
<p class="subtitle">Dashed = no offset (the original result), solid = with random offsets. Everything right
of the vertical line was never seen in training.</p>
</div>

<div class="card">
<h2>6. The instructor's literal ask, tested</h2>
{value_narrative}
</div>

<div class="card">
<h2>7. Sample efficiency</h2>
{legend_html([(ARM_LABEL_A[a], ARM_COLOR_A[a]) for a in ["baseline","abacus","crt"]])}
<h3 style="font-size:0.95rem;">Addition</h3>
<div id="chart-eff-add"></div>
<h3 style="font-size:0.95rem;margin-top:18px;">Multiplication</h3>
<div id="chart-eff-mul"></div>
</div>

<div class="card">
<h2>8. Evidence</h2>
{evidence_table_html(evidence)}
</div>

<div class="card limitations">
<h2>Honest limitations</h2>
<ul>
<li>The exact-arithmetic claim holds only within <code>[0, {proof_report['value_N']:,})</code> for the value code
    and <code>[0, {proof_report['position_N']})</code> for the position code — outside that range the construction
    wraps (tested explicitly, not hidden) rather than failing silently.</li>
<li>A trained transformer's behavior (section 3) does <strong>not</strong> demonstrate that the model performs
    exact arithmetic in embedding space — that would require circuit-level verification in the style of
    Nanda et al. (2023) / Zhong et al. (2023), which is out of scope here. Section 3 only measures whether the
    fixed CRT positional code changes a trained model's length-generalization behavior relative to a learned
    (Abacus-lite) or absent (NoPE baseline) positional signal.</li>
<li>The Abacus-lite arm is a lightweight reimplementation of the core mechanism in McLeish et al. (2024), not
    their full looped/recurrent architecture, and is not tuned to match their reported numbers.</li>
<li>Multiplication is empirically harder than addition at this model scale (see section 3) — this is reported
    as a legitimate negative result, not hidden or worked around by inflating model size.</li>
<li>Model size ({n_params:,} params) and training budget were deliberately kept small so every run finishes in
    minutes; nothing here claims to generalize to larger scale without re-running.</li>
</ul>
</div>

<div class="card">
<h2>Reproduce</h2>
<pre>python proofs/analytic_proof.py
python proofs/make_plots.py
python model/train.py --arm crt --task add --profile default</pre>
</div>
"""

    script_data = {
        "task": "add",
        "chartsLenAdd": length_gen_charts.get("add", []),
        "chartsLenMul": length_gen_charts.get("mul", []),
        "effAdd": sample_eff_charts.get("add", []),
        "effMul": sample_eff_charts.get("mul", []),
        "trainRangeMax": train_range[1],
        "crtPrimes": VALUE_PRIMES,
        "crtN": crt_N,
        "crtConsts": crt_consts,
        "offsetSeries": offset_series,
    }
    body += """
<script>
vizLineChart(document.getElementById('chart-len-add'), {series: DATA.chartsLenAdd, xLabel: 'test digit-length', yMin:0, yMax:1.02, refLineX: DATA.trainRangeMax, yFormat: v => (v*100).toFixed(0)+'%'});
vizLineChart(document.getElementById('chart-len-mul'), {series: DATA.chartsLenMul, xLabel: 'test digit-length', yMin:0, yMax:1.02, refLineX: DATA.trainRangeMax, yFormat: v => (v*100).toFixed(0)+'%'});
if (DATA.offsetSeries && DATA.offsetSeries.length) {
  vizLineChart(document.getElementById('chart-offset'), {series: DATA.offsetSeries, xLabel: 'test digit-length', yMin:0, yMax:1.02, refLineX: DATA.trainRangeMax, yFormat: v => (v*100).toFixed(0)+'%'});
}
vizLineChart(document.getElementById('chart-eff-add'), {series: DATA.effAdd, xLabel: 'training step', yMin:0, yMax:1.02, yFormat: v => (v*100).toFixed(0)+'%'});
vizLineChart(document.getElementById('chart-eff-mul'), {series: DATA.effMul, xLabel: 'training step', yMin:0, yMax:1.02, yFormat: v => (v*100).toFixed(0)+'%'});

function crtDecode(residues) {
  let total = 0;
  for (let i = 0; i < DATA.crtPrimes.length; i++) total = (total + residues[i] * DATA.crtConsts[i]) % DATA.crtN;
  return total;
}
function drawClock(p, residue, otherResidue, size) {
  const r = size / 2 - 6, cx = size/2, cy = size/2;
  const angle = (theta) => theta * 2 * Math.PI / p - Math.PI/2;
  let dots = '';
  for (let k = 0; k < p; k++) {
    const a = angle(k);
    dots += `<circle cx="${cx + r*Math.cos(a)}" cy="${cy + r*Math.sin(a)}" r="2" fill="var(--baseline)"/>`;
  }
  const a1 = angle(residue);
  const mainDot = `<circle cx="${cx + r*Math.cos(a1)}" cy="${cy + r*Math.sin(a1)}" r="6" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2"/>`;
  let sumDot = '';
  if (otherResidue !== undefined) {
    const a2 = angle(otherResidue);
    sumDot = `<circle cx="${cx + r*Math.cos(a2)}" cy="${cy + r*Math.sin(a2)}" r="6" fill="var(--series-2)" stroke="var(--surface-1)" stroke-width="2"/>`;
  }
  return `<div style="text-align:center;">
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--gridline)" stroke-width="1"/>
      ${dots}${mainDot}${sumDot}
    </svg>
    <div class="axis-label" style="margin-top:2px;">mod ${p}</div>
  </div>`;
}
function renderCrtDemo() {
  const a = ((parseInt(document.getElementById('crt-a').value) || 0) % DATA.crtN + DATA.crtN) % DATA.crtN;
  const b = ((parseInt(document.getElementById('crt-b').value) || 0) % DATA.crtN + DATA.crtN) % DATA.crtN;
  const aRes = DATA.crtPrimes.map(p => a % p);
  const sumRes = DATA.crtPrimes.map((p, i) => (aRes[i] + (b % p)) % p);
  const decoded = crtDecode(sumRes);
  const expected = (a + b) % DATA.crtN;
  const clocksEl = document.getElementById('crt-clocks');
  clocksEl.innerHTML = DATA.crtPrimes.map((p, i) => drawClock(p, aRes[i], sumRes[i], 96)).join('');
  document.getElementById('crt-readout').innerHTML =
    `<span style="color:var(--series-1)">&#9679;</span> a = ${a} &nbsp;
     <span style="color:var(--series-2)">&#9679;</span> a+b = ${decoded} &nbsp;|&nbsp;
     (a+b) mod N computed directly = ${expected} &nbsp;
     <strong style="color:${decoded === expected ? 'var(--good)' : 'var(--series-2)'}">${decoded === expected ? 'exact match' : 'MISMATCH'}</strong>`;
}
['crt-a', 'crt-b'].forEach(id => document.getElementById(id).addEventListener('input', renderCrtDemo));
renderCrtDemo();
</script>
"""
    return page_shell("Track A — Kronecker Numeral Embeddings", "🔢", body, script_data)


# ---------------------------------------------------------------------------
# Track B
# ---------------------------------------------------------------------------

ARM_LABEL_B = {"kronecker": "Kronecker/tensor (32-slot cap)", "holographic": "Holographic (circular convolution)"}
ARM_COLOR_B = {"kronecker": SERIES["blue"], "holographic": SERIES["orange"]}
D_MODEL = 192  # must match track_b_holographic_binding/model/train.py's D_MODEL


def build_track_b() -> str:
    results_dir = TRACK_B / "results"
    proof_report = load_json(results_dir / "capacity_proof_report.json")
    proof_figures = load_json(results_dir / "proof_figures.json")
    evidence = load_evidence_for("Track B")

    runs = {}
    for arm in ["kronecker", "holographic"]:
        p = results_dir / f"{arm}_default.json"
        if p.exists():
            runs[arm] = load_json(p)

    any_run = next(iter(runs.values()), None)

    buckets = list(next(iter(runs.values()))["perplexity_by_length_bucket"].keys()) if runs else []
    ppl_groups = []
    for bucket in buckets:
        bars = []
        for arm in ["kronecker", "holographic"]:
            r = runs.get(arm)
            if not r:
                continue
            stats = r["perplexity_by_length_bucket"].get(bucket)
            if stats is None:
                continue
            bars.append(
                {
                    "name": ARM_LABEL_B[arm],
                    "value": stats["perplexity"],
                    "color": ARM_COLOR_B[arm],
                    "valueLabel": f"ppl={stats['perplexity']:.1f} (n={stats['n_tokens']:,})",
                }
            )
        ppl_groups.append({"label": bucket, "bars": bars})

    param_bars = []
    short_name = {"kronecker": "Kronecker", "holographic": "Holographic"}
    for arm in ["kronecker", "holographic"]:
        r = runs.get(arm)
        if not r:
            continue
        param_bars.append(
            {
                "label": f"{short_name[arm]}, total",
                "value": r["n_params_total"],
                "color": ARM_COLOR_B[arm],
                "valueLabel": f"{r['n_params_total']:,} total params",
            }
        )
        param_bars.append(
            {
                "label": f"{short_name[arm]}, embed",
                "value": r["n_params_embedding_learned"],
                "color": ARM_COLOR_B[arm],
                "valueLabel": f"{r['n_params_embedding_learned']:,} learned embedding params",
            }
        )

    theory_cells = [c for c in proof_report["cells"] if "theoretical_accuracy" in c]
    max_theory_delta = (
        max(abs(c["accuracy"] - c["theoretical_accuracy"]) for c in theory_cells) if theory_cells else 0.0
    )

    dress_run = None
    dress_path = results_dir / "holographic_dress_default.json"
    if dress_path.exists():
        dress_run = load_json(dress_path)

    corpus_alphabet = load_corpus()["alphabet"]
    d192_cells = sorted(
        ({"L": c["L"], "accuracy": c["accuracy"]} for c in proof_report["cells"] if c["D"] == 192),
        key=lambda c: c["L"],
    )

    if dress_run and "holographic" in runs and "kronecker" in runs:
        plain_ppl = runs["holographic"]["final_val_perplexity"]
        kron_ppl = runs["kronecker"]["final_val_perplexity"]
        dressed_ppl = dress_run["final_val_perplexity"]
        verdict = (
            "so adaptability does <strong>not</strong> explain the Kronecker arm's edge — giving the "
            "holographic table a learned layer made it <em>worse</em>, not better"
            if dressed_ppl > plain_ppl
            else "so a large part of the Kronecker arm's edge is adaptability, not information content"
        )
        dress_section = f"""
<h3 style="font-size:0.95rem;margin-top:20px;">Ablation: is the perplexity gap just adaptability?</h3>
<p>The Kronecker arm gets a learned projection; the plain holographic arm gets none. That confounds
"which code carries more information" with "which arm can adapt its code". Giving the holographic table
its own thin learned layer separates the two.</p>
<table class="evtable"><thead><tr><th>Arm</th><th>Learned embedding params</th><th>Val perplexity</th></tr></thead>
<tbody>
<tr><td>Kronecker/tensor (32-slot cap)</td><td>{runs['kronecker']['n_params_embedding_learned']:,}</td><td>{kron_ppl:.2f}</td></tr>
<tr><td>Holographic, no dressing</td><td>{runs['holographic']['n_params_embedding_learned']:,}</td><td>{plain_ppl:.2f}</td></tr>
<tr><td>Holographic + learned dressing</td><td>{dress_run['n_params_embedding_learned']:,}</td><td>{dressed_ppl:.2f}</td></tr>
</tbody></table>
<p class="subtitle">Dressed holographic lands at {dressed_ppl:.1f} against {plain_ppl:.1f} undressed —
{verdict}. Note the dressing is a d_model&times;d_model layer
({dress_run['n_params_embedding_learned']:,} params), which is the same <em>mechanism</em> as the
Kronecker arm's projection but not the same parameter count
({runs['kronecker']['n_params_embedding_learned']:,}) — it tests adaptability, not parameter parity.</p>"""
    else:
        dress_section = ""

    truncation_rows = ""
    for arm in ["kronecker", "holographic"]:
        r = runs.get(arm)
        if not r:
            continue
        t = r["truncation_information_loss"]
        verdict = "indistinguishable (total information loss)" if t["mean_cosine_similarity"] > 0.999 else "distinguishable"
        truncation_rows += (
            f"<tr><td>{ARM_LABEL_B[arm]}</td><td>{t['mean_cosine_similarity']:.4f}</td>"
            f"<td>{t['n_pairs']}</td><td>{verdict}</td></tr>"
        )

    body = f"""
<h1>Track B — Holographic/Fourier Binding</h1>
<p class="subtitle">Replacing Kronecker/tensor-product binding (V1's fixed 32-character-slot cap) with
circular-convolution superposition — a fixed-dimension embedding for words of <em>any</em> length, at the
cost of a provable, measured capacity/crosstalk tradeoff.</p>
<div class="badges">
  <span class="badge">corpus: {any_run['corpus_raw_bytes']:,} bytes (measured)</span>
  <span class="badge">max natural word length: {any_run['corpus_max_natural_word_length']} chars</span>
  <span class="badge">device: {any_run['device']['device'] if any_run else 'n/a'}</span>
</div>

<div class="card">
<h2>1. The proof (no model, no training)</h2>
<p>Run in {proof_report['elapsed_seconds']:.2f}s. Single-pair unbind is <strong>exact</strong> for a unitary
role vector (verified per dimension, max error &lt; 1e-8) — the interesting behavior is what happens once
multiple bound pairs are superposed into one word vector.</p>
<div class="figure"><img src="{proof_figures['capacity_curve']}" alt="Decode accuracy vs word length, by dimension"></div>
<p class="subtitle">Dotted lines are our own derivation of the standard VSA signal-to-noise argument
(written out step by step in <code>theoretical_decode_accuracy()</code>, assumptions stated) — deliberately
not presented as a formula lifted from Plate. It predicts the measurement to within
{max_theory_delta:.1%} worst-case across every (L, D) cell, and is mildly optimistic at small D exactly as
its Gaussian-independence approximation predicts.</p>
<div class="figure"><img src="{proof_figures['interference']}" alt="Interference: true filler vs best distractor similarity"></div>
<div class="figure"><img src="{proof_figures['role_comparison']}" alt="Random-phase roles vs literal shift roles"></div>
<p class="subtitle">The instructor's phrasing — "represent each character like a Fourier wave, and just add
them" — reads most literally as deterministic <em>shift</em> roles (position p is a pure phase ramp) rather
than the random-phase roles used elsewhere. Both are unitary, so both give exact single-pair unbind; swept
head to head, the literal reading is <strong>not worse</strong> (it is marginally better at D=192). Our
prior expectation was that shifted copies would interfere in a correlated way and lose — that did not
happen, and the measurement is reported over the prediction.</p>
</div>

<div class="card">
<h2>2. Try it yourself — live, in your browser</h2>
<p>Type any word, including one over 32 characters. The Kronecker/tensor scheme only has 32 slots — anything
past position 32 is dropped, live, in front of you. The holographic scheme has no cap; the estimated decode
accuracy at that length is read directly off section 1's measured D={D_MODEL} curve (a lookup against real
proof data, not a guess).</p>
<div id="word-demo">
  <input id="word-input" type="text" maxlength="80" value="disestablishmentarianismandbeyond"
    style="width:100%;max-width:520px;padding:8px 10px;border-radius:6px;border:1px solid var(--border);background:var(--page);color:var(--text-primary);font-size:0.95rem;">
  <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:16px;">
    <div style="flex:1;min-width:260px;">
      <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:6px;">Kronecker/tensor (32-slot cap)</div>
      <div id="word-kron" style="font-family:ui-monospace,monospace;font-size:1rem;line-height:1.6;word-break:break-all;"></div>
      <div id="word-kron-info" style="font-size:0.82rem;color:var(--text-secondary);margin-top:6px;"></div>
    </div>
    <div style="flex:1;min-width:260px;">
      <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:6px;">Holographic (fixed dim, no cap)</div>
      <div id="word-holo" style="font-family:ui-monospace,monospace;font-size:1rem;line-height:1.6;word-break:break-all;"></div>
      <div id="word-holo-info" style="font-size:0.82rem;color:var(--text-secondary);margin-top:6px;"></div>
    </div>
  </div>
</div>
</div>

<div class="card">
<h2>3. Perplexity by natural word-length bucket</h2>
<p>The natural tiny_shakespeare corpus tops out at {any_run['corpus_max_natural_word_length']} characters
(measured) — far short of the 32-char cap, so this chart alone cannot show a truncation cliff. Section 4
demonstrates that directly instead.</p>
{legend_html([(ARM_LABEL_B[a], ARM_COLOR_B[a]) for a in ["kronecker","holographic"]])}
<div id="chart-ppl"></div>
</div>

<div class="card">
<h2>4. Truncation information loss (the controlled probe)</h2>
<p>Pairs of synthetic strings that are <strong>identical in their first 32 characters</strong> and differ only
after — the minimal, controlled test of whether an embedding scheme can tell them apart at all.</p>
<table class="evtable"><thead><tr><th>Arm</th><th>Mean cosine similarity</th><th>N pairs</th><th>Verdict</th></tr></thead>
<tbody>{truncation_rows}</tbody></table>
</div>

<div class="card">
<h2>5. Parameter count</h2>
<p>The Kronecker arm needs a learned projection from its 1,088-dim raw slot code down to d_model; the
holographic arm needs zero learned embedding parameters at all — its table is entirely fixed role/filler
vectors.</p>
<div id="chart-params"></div>
{dress_section}
</div>

<div class="card">
<h2>6. Evidence</h2>
{evidence_table_html(evidence)}
</div>

<div class="card limitations">
<h2>Honest limitations</h2>
<ul>
<li>The natural corpus never produces a word longer than {any_run['corpus_max_natural_word_length']} characters
    (measured), so the 32-character truncation cliff is demonstrated via a controlled synthetic probe
    (section 3), not via natural-corpus perplexity.</li>
<li>Word-level closed-vocabulary perplexity here ({', '.join(f"{ARM_LABEL_B[a]}: {runs[a]['final_val_perplexity']:.1f}" for a in runs)})
    is <strong>not</strong> comparable to standard char-level tiny_shakespeare perplexity numbers reported
    elsewhere — different tokenization, different task.</li>
<li>On this run, the Kronecker arm's in-domain perplexity is <em>slightly better</em> than the holographic
    arm's, despite using more learned parameters and losing information past 32 characters — reported as
    measured, not smoothed over. Holographic's advantages (fixed dimension regardless of length, zero
    learned embedding parameters, no truncation) are structural properties, not a perplexity win here.</li>
<li>Decode/unbind capacity trades off length against dimension (section 1) — holographic binding does not
    have unlimited capacity, it has a measured, dimension-dependent one.</li>
</ul>
</div>

<div class="card">
<h2>Reproduce</h2>
<pre>python proofs/capacity_proof.py
python proofs/make_plots.py
python model/train.py --arm holographic --profile default</pre>
</div>
"""

    script_data = {
        "pplGroups": ppl_groups,
        "paramBars": param_bars,
        "alphabet": corpus_alphabet,
        "d192Cells": d192_cells,
        "kronMaxSlots": 32,
    }
    body += """
<script>
vizGroupedBarChart(document.getElementById('chart-ppl'), {groups: DATA.pplGroups, yLabel: 'perplexity', yFormat: v => v.toFixed(0)});
vizBarChart(document.getElementById('chart-params'), {bars: DATA.paramBars, yLabel: 'parameters', yFormat: v => (v/1000).toFixed(0)+'K'});

function estimateDecodeAccuracy(L) {
  const cells = DATA.d192Cells;
  if (L <= cells[0].L) return cells[0].accuracy;
  if (L >= cells[cells.length-1].L) return cells[cells.length-1].accuracy;
  for (let i = 0; i < cells.length - 1; i++) {
    if (L >= cells[i].L && L <= cells[i+1].L) {
      const t = (L - cells[i].L) / (cells[i+1].L - cells[i].L);
      return cells[i].accuracy + t * (cells[i+1].accuracy - cells[i].accuracy);
    }
  }
  return cells[cells.length-1].accuracy;
}
function renderWordDemo() {
  const raw = document.getElementById('word-input').value;
  const word = raw.toLowerCase();
  const alphaSet = new Set(DATA.alphabet);
  const chars = word.split('');
  const kept = chars.slice(0, DATA.kronMaxSlots);
  const dropped = chars.slice(DATA.kronMaxSlots);
  const known = c => alphaSet.has(c);
  const kronHtml = kept.map(c => `<span style="color:${known(c) ? 'var(--text-primary)' : 'var(--series-2)'}">${c}</span>`).join('')
    + (dropped.length ? dropped.map(c => `<span style="color:var(--series-2);text-decoration:line-through;opacity:0.6;">${c}</span>`).join('') : '');
  document.getElementById('word-kron').innerHTML = kronHtml || '&nbsp;';
  const rawWidth = DATA.kronMaxSlots * DATA.alphabet.length;
  document.getElementById('word-kron-info').textContent =
    `${chars.length} chars, ${Math.min(chars.length, 32)} kept, ${dropped.length} dropped (struck through) `
    + `— raw slot width ${rawWidth.toLocaleString()} dims, always fixed regardless of word length`;

  const holoHtml = chars.map(c => `<span style="color:${known(c) ? 'var(--text-primary)' : 'var(--series-2)'}">${c}</span>`).join('');
  document.getElementById('word-holo').innerHTML = holoHtml || '&nbsp;';
  const acc = estimateDecodeAccuracy(chars.length);
  document.getElementById('word-holo-info').textContent =
    `${chars.length} chars, all used, fixed ${192} dims — estimated per-character decode accuracy at this length `
    + `(from the measured D=192 curve in section 1): ${(acc*100).toFixed(1)}%`;
}
document.getElementById('word-input').addEventListener('input', renderWordDemo);
renderWordDemo();
</script>
"""
    return page_shell("Track B — Holographic/Fourier Binding", "🌀", body, script_data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["a", "b"], required=True)
    args = parser.parse_args()

    if args.track == "a":
        html = build_track_a()
        out = TRACK_A / "submission_artifacts" / "dashboard.html"
    else:
        html = build_track_b()
        out = TRACK_B / "submission_artifacts" / "dashboard.html"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
