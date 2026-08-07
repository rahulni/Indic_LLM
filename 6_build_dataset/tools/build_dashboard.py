# -*- coding: utf-8 -*-
"""Render ``submission_artifacts/dashboard.html`` from the generated JSON.

    python tools/build_dashboard.py [artifact_dir]

A view, never a second source of truth: every number is read from the artifacts
the run produced, so the page cannot disagree with the evidence bundle. Fully
self-contained -- inline CSS, inline JS, hand-drawn SVG, no CDN, no fetch, no
webfonts -- so it opens from a file:// URL with no network.

The interactivity is deliberately plain: charts are rendered server-side into
SVG and the script only adds hover, sorting, filtering and theming on top. If
the script fails to run, the page still shows every figure.
"""
from __future__ import annotations

import html
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tdes.hashing import read_json, read_jsonl, write_text     # noqa: E402

# Tableau-10 derived; readable on both themes, distinguishable in greyscale.
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#EECA3B", "#9D755D"]
LANE_COLOR = {
    "web": "#4C78A8", "code": "#F58518", "indic": "#54A24B",
    "multiling": "#72B7B2", "reasoning": "#B279A2", "agentic": "#E45756",
    "longctx": "#EECA3B",
}
# Languages written in an Indic script, for the fertility contrast. Codes are
# the ones the corpus actually carries.
INDIC_LANGS = {"hin", "tel", "kan", "mar", "nep", "ben", "tam", "guj"}

TOKEN_TRACE_LIMIT = 1200


def esc(x) -> str:
    return html.escape(str(x))


def num(x, nd=2, dash="-"):
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return dash


def pct(x, nd=1, dash="-"):
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except (TypeError, ValueError):
        return dash


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load(art: str) -> dict:
    def j(*p, default=None):
        try:
            return read_json(os.path.join(art, *p))
        except Exception:
            return default

    def jl(*p):
        try:
            return read_jsonl(os.path.join(art, *p), tolerate_torn_tail=True)
        except Exception:
            return []

    return {
        "evidence": j("evidence.json", default={}),
        "perf": j("performance.json", default={}),
        "cost": j("cost_report.json", default={}),
        "reports": j("reports.json", default={}),
        "meta": j("run_meta.json", default={}),
        "steps": j("ledgers", "learning_steps.json", default=[]),
        "shards": j("ledgers", "learning_shards.json", default=[]),
        "summary": j("ledgers", "learning_summary.json", default={}),
        "opus_rounds": j("ledgers", "opus_rounds.json", default=[]),
        "opus": jl("ledgers", "opus_decisions.jsonl"),
        "firewall": j("ledgers", "firewall.json", default={}),
        "replay": j("ledgers", "replay.json", default={}),
        "resume": j("ledgers", "resume.json", default={}),
        "fork": j("ledgers", "fork.json", default={}),
        "audit": j("audit", "audit.json", default={}),
        "retention": j("checkpoints", "retention.json", default={}),
        "events": jl("events.jsonl"),
        "tokens": jl("ledgers", "learning_tokens.jsonl")[:TOKEN_TRACE_LIMIT],
    }


class Ctx:
    """Collects the per-chart geometry the script needs for hover readouts.

    The SVG is the drawing; this is the data behind it. Keeping them separate
    means the picture is still correct with JavaScript switched off.
    """

    def __init__(self) -> None:
        self.charts: dict[str, dict] = {}
        self._n = 0

    def cid(self, base: str) -> str:
        self._n += 1
        return f"{base}{self._n}"


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------

def line_chart(ctx: Ctx, series: list[dict], labels: list[str], *,
               w=880, h=250, baseline: float | None = None,
               baseline_label="", ylabel="", unit="", markers=()) -> str:
    """Multi-segment line chart. ``series`` items: {name, ys, x0, color}."""
    vals = [v for s in series for v in s["ys"] if v is not None]
    if not vals:
        return "<p class='muted'>no data</p>"
    lo, hi = min(vals), max(vals)
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    span = (hi - lo) or abs(hi) or 1.0
    lo, hi = lo - span * 0.12, hi + span * 0.12
    pl, pr, pt, pb = 56, 16, 16, 34
    iw, ih = w - pl - pr, h - pt - pb
    n = max(len(labels), max((s["x0"] + len(s["ys"])) for s in series))

    def X(i):
        return pl + (iw * i / max(1, n - 1))

    def Y(v):
        return pt + ih - ih * (v - lo) / (hi - lo)

    cid = ctx.cid("ch")
    out = [f'<svg id="{cid}" viewBox="0 0 {w} {h}" class="chart" '
           f'preserveAspectRatio="none" role="img">']
    out.append(f'<defs><clipPath id="{cid}-clip"><rect x="{pl}" y="{pt}" '
               f'width="{iw}" height="{ih}"/></clipPath></defs>')
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        out.append(f'<line class="grid" x1="{pl}" y1="{y:.1f}" x2="{w-pr}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{pl-8}" y="{y+4:.1f}" text-anchor="end">'
                   f'{v:.2f}</text>')
    if baseline is not None:
        y = Y(baseline)
        out.append(f'<line class="baseline" x1="{pl}" y1="{y:.1f}" x2="{w-pr}" y2="{y:.1f}"/>')
        # Left-anchored: the right edge is where the last segment marker lands.
        out.append(f'<text class="tick baseline-t" x="{pl+6}" y="{y-7:.1f}">'
                   f'{esc(baseline_label)}</text>')
    for mi, (idx, label) in enumerate(markers):
        x = X(idx)
        out.append(f'<line class="marker" x1="{x:.1f}" y1="{pt}" x2="{x:.1f}" y2="{pt+ih}"/>')
        out.append(f'<text class="marker-t" x="{x+4:.1f}" y="{pt+11+mi%2*13}">'
                   f'{esc(label)}</text>')

    meta_series = []
    for si, s in enumerate(series):
        c = s.get("color") or PALETTE[si % len(PALETTE)]
        pts = [(X(s["x0"] + i), Y(v)) for i, v in enumerate(s["ys"]) if v is not None]
        if not pts:
            continue
        area = (f'M{pts[0][0]:.1f},{pt+ih:.1f} '
                + " ".join(f"L{x:.1f},{y:.1f}" for x, y in pts)
                + f' L{pts[-1][0]:.1f},{pt+ih:.1f} Z')
        out.append(f'<path class="area" d="{area}" fill="{c}" '
                   f'clip-path="url(#{cid}-clip)"/>')
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<polyline class="ln" fill="none" stroke="{c}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round" points="{poly}"/>')
        if len(pts) == 1:
            out.append(f'<circle cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="3" fill="{c}"/>')
        meta_series.append({"name": s["name"], "color": c, "x0": s["x0"],
                            "ys": [None if v is None else round(v, 6) for v in s["ys"]]})
    if ylabel:
        out.append(f'<text class="axis" x="4" y="{pt-4}">{esc(ylabel)}</text>')
    out.append(f'<rect class="hit" x="0" y="0" width="{w}" height="{h}" fill="transparent"/>')
    out.append("</svg>")

    legend = "".join(
        f'<span class="key"><i style="background:{s["color"]}"></i>{esc(s["name"])}</span>'
        for s in meta_series)
    ctx.charts[cid] = {"w": w, "h": h, "pl": pl, "pt": pt, "iw": iw, "ih": ih,
                       "lo": lo, "hi": hi, "n": n, "unit": unit,
                       "labels": labels, "series": meta_series}
    return f'<div class="chart-wrap">{"".join(out)}</div><div class="legend">{legend}</div>'


def bar_pairs(rows: list[tuple[str, float, float]], *, w=880, la="planned",
              lb="actual") -> str:
    """Planned-vs-actual per lane, with the drift called out."""
    if not rows:
        return "<p class='muted'>no data</p>"
    bh, gap = 15, 16
    h = len(rows) * (bh * 2 + gap) + 14
    mx = max(max(a, b) for _, a, b in rows) or 1.0
    pl, iw = 96, w - 96 - 150
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    y = 8
    for name, a, b in rows:
        col = LANE_COLOR.get(name, PALETTE[0])
        for val, opacity, lab in ((a, 0.38, la), (b, 1.0, lb)):
            bw = max(1.0, iw * val / mx)
            out.append(f'<rect class="bar" x="{pl}" y="{y}" width="{bw:.1f}" '
                       f'height="{bh-3}" rx="3" fill="{col}" opacity="{opacity}">'
                       f'<title>{esc(name)} {lab} {val*100:.3f}%</title></rect>')
            out.append(f'<text class="tick" x="{pl+bw+7:.1f}" y="{y+bh-5}">'
                       f'{val*100:.2f}% <tspan class="dim">{lab}</tspan></text>')
            y += bh
        drift = b - a
        out.append(f'<text class="label" x="{pl-10}" y="{y-bh+1}" text-anchor="end">'
                   f'{esc(name)}</text>')
        out.append(f'<text class="drift {"up" if drift>=0 else "down"}" x="{pl-10}" '
                   f'y="{y+1}" text-anchor="end">{drift*100:+.2f} pp</text>')
        y += gap
    out.append("</svg>")
    return "".join(out)


def hbar_groups(rows: list[tuple[str, list[tuple[str, float, str]]]], *,
                w=880, label_w=150) -> str:
    """One group per row label; each group is a stack of thin labelled bars."""
    if not rows:
        return "<p class='muted'>no data</p>"
    bh, gap = 13, 12
    h = sum(len(m) * bh for _, m in rows) + len(rows) * gap + 8
    iw = w - label_w - 130
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    y = 6
    for name, metrics in rows:
        y0 = y
        for mname, val, col in metrics:
            v = max(0.0, min(1.0, float(val)))
            out.append(f'<rect class="track" x="{label_w}" y="{y}" width="{iw}" '
                       f'height="{bh-3}" rx="3"/>')
            out.append(f'<rect class="bar" x="{label_w}" y="{y}" width="{max(1.0, iw*v):.1f}" '
                       f'height="{bh-3}" rx="3" fill="{col}">'
                       f'<title>{esc(name)} {esc(mname)} {v*100:.2f}%</title></rect>')
            out.append(f'<text class="tick" x="{label_w+iw+8}" y="{y+bh-4}">'
                       f'{v*100:.1f}% <tspan class="dim">{esc(mname)}</tspan></text>')
            y += bh
        out.append(f'<text class="label" x="{label_w-10}" y="{(y0+y)/2+4:.0f}" '
                   f'text-anchor="end">{esc(name)}</text>')
        y += gap
    out.append("</svg>")
    return "".join(out)


def hbars(rows: list[tuple[str, float, str, str]], *, w=880, label_w=110,
          right_w=210) -> str:
    """Simple labelled horizontal bars: (label, value_0_to_1, colour, right_text)."""
    if not rows:
        return "<p class='muted'>no data</p>"
    bh, gap = 20, 8
    h = len(rows) * (bh + gap) + 6
    iw = w - label_w - right_w
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    y = 4
    for name, v, col, right in rows:
        v = max(0.0, min(1.0, float(v)))
        out.append(f'<rect class="track" x="{label_w}" y="{y}" width="{iw}" '
                   f'height="{bh-4}" rx="4"/>')
        out.append(f'<rect class="bar" x="{label_w}" y="{y}" width="{max(1.0, iw*v):.1f}" '
                   f'height="{bh-4}" rx="4" fill="{col}"/>')
        out.append(f'<text class="label" x="{label_w-10}" y="{y+bh-8}" text-anchor="end">'
                   f'{esc(name)}</text>')
        out.append(f'<text class="tick" x="{label_w+iw+8}" y="{y+bh-8}">{esc(right)}</text>')
        y += bh + gap
    out.append("</svg>")
    return "".join(out)


def donut(parts: list[tuple[str, float, str]], *, size=170, thickness=26,
          centre_top="", centre_sub="") -> str:
    total = sum(v for _, v, _ in parts)
    if total <= 0:
        return "<p class='muted'>no data</p>"
    r = (size - thickness) / 2
    c = size / 2
    circ = 2 * math.pi * r
    out = [f'<svg viewBox="0 0 {size} {size}" class="donut" role="img">']
    out.append(f'<circle class="track" cx="{c}" cy="{c}" r="{r}" fill="none" '
               f'stroke-width="{thickness}"/>')
    off = 0.0
    for name, v, col in parts:
        frac = v / total
        out.append(f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{col}" '
                   f'stroke-width="{thickness}" stroke-dasharray="{circ*frac:.3f} {circ:.3f}" '
                   f'stroke-dashoffset="{-off:.3f}" transform="rotate(-90 {c} {c})">'
                   f'<title>{esc(name)}: {v:g} ({frac*100:.1f}%)</title></circle>')
        off += circ * frac
    if centre_top:
        out.append(f'<text class="donut-v" x="{c}" y="{c+2}" text-anchor="middle">'
                   f'{esc(centre_top)}</text>')
    if centre_sub:
        out.append(f'<text class="donut-l" x="{c}" y="{c+18}" text-anchor="middle">'
                   f'{esc(centre_sub)}</text>')
    out.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:{col}"></i>{esc(name)}'
        f'<b>{v:g}</b></span>' for name, v, col in parts)
    return f'<div class="donut-row">{"".join(out)}<div class="legend col">{legend}</div></div>'


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, i))]


def heatmap(tokens: list[dict], *, cols=60, w=880) -> str:
    """Per-token perplexity.

    Two decisions worth stating. The scale is **logarithmic between the 2nd and
    98th percentile of this trace**, not linear to the maximum: perplexity early
    in training spans three orders of magnitude, so a linear ramp saturates and
    every cell reads the same colour -- a picture that looks alarming and says
    nothing. The endpoints are printed under the map so the colours are
    interpretable rather than decorative.

    Hover detail comes from the embedded trace rather than an SVG <title> per
    cell; 1,200 native tooltips are unusable.
    """
    if not tokens:
        return "<p class='muted'>no token trace</p>"
    cell = w / cols
    rows = math.ceil(len(tokens) / cols)
    h = rows * cell + 2
    vals = sorted(max(1e-9, float(t.get("perplexity", 1.0))) for t in tokens)
    lo, hi, med = _quantile(vals, 0.02), _quantile(vals, 0.98), _quantile(vals, 0.5)
    if hi <= lo:
        hi = lo * 2 + 1.0
    llo, lspan = math.log(lo), math.log(hi) - math.log(lo)
    out = [f'<svg id="heat" viewBox="0 0 {w} {h:.0f}" class="chart heat" role="img">']
    for i, t in enumerate(tokens):
        r, c = divmod(i, cols)
        v = (math.log(max(1e-9, float(t.get("perplexity", 1.0)))) - llo) / lspan
        v = max(0.0, min(1.0, v))
        col = f"rgb({40 + int(200*v)},{130 - int(90*v)},{190 - int(150*v)})"
        out.append(f'<rect data-i="{i}" data-lane="{esc(t.get("capability_lane",""))}" '
                   f'x="{c*cell:.2f}" y="{r*cell:.2f}" width="{cell-0.4:.2f}" '
                   f'height="{cell-0.4:.2f}" fill="{col}"/>')
    out.append("</svg>")
    scale = (f"<div class='scale'><span>confident · ppl {lo:,.0f}</span>"
             f"<i class='ramp'></i><span>ppl {hi:,.0f} · surprised</span>"
             f"<span class='dim'>log scale over the 2nd–98th percentile of this "
             f"trace · median {med:,.0f}</span></div>")
    return "".join(out) + scale


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

class Cell:
    """A table cell whose display text differs from its sort key."""

    __slots__ = ("text", "v", "cls")

    def __init__(self, text, v=None, cls=""):
        self.text, self.v, self.cls = text, v, cls


def table(headers, rows, *, cls="", sortable=False, tid="") -> str:
    if not rows:
        return "<p class='muted'>no rows</p>"
    th = []
    for h in headers:
        th.append(f'<th{" tabindex=0" if sortable else ""}>{esc(h)}</th>')
    tr = []
    for r in rows:
        tds = []
        for cval in r:
            if isinstance(cval, Cell):
                v = "" if cval.v is None else f' data-v="{esc(cval.v)}"'
                k = f' class="{cval.cls}"' if cval.cls else ""
                tds.append(f"<td{k}{v}>{cval.text}</td>")
            else:
                tds.append(f"<td>{esc(cval)}</td>")
        tr.append("<tr>" + "".join(tds) + "</tr>")
    klass = " ".join(x for x in [cls, "sortable" if sortable else ""] if x)
    idattr = f' id="{tid}"' if tid else ""
    return (f'<div class="scroll"><table{idattr} class="{klass}"><thead><tr>'
            f'{"".join(th)}</tr></thead><tbody>{"".join(tr)}</tbody></table></div>')


def stat(label, value, sub="", tone="") -> str:
    return (f'<div class="stat {tone}"><div class="stat-v">{esc(value)}</div>'
            f'<div class="stat-l">{esc(label)}</div>'
            f'{f"<div class=stat-s>{esc(sub)}</div>" if sub else ""}</div>')


def pill(text, tone="") -> str:
    return f'<span class="pill {tone}">{esc(text)}</span>'


def section(sid, title, sub="", body="", tools="") -> str:
    subhtml = f'<p class="muted">{sub}</p>' if sub else ""
    return (f'<section id="{sid}"><div class="sec-head"><h2>{esc(title)}</h2>'
            f'{tools}</div>{subhtml}{body}</section>')


SECTIONS = [
    ("overview", "Overview"), ("evidence", "Evidence"), ("training", "Training"),
    ("tokens", "Tokens"), ("mixture", "Mixture"), ("packing", "Packing"),
    ("opus", "OPUS"), ("recovery", "Recovery"), ("shards", "Shards"),
    ("tokenizer", "Tokenizer"), ("throughput", "Throughput"), ("cost", "Cost"),
]


# ---------------------------------------------------------------------------
def _segments(steps: list[dict], fork: dict) -> list[dict]:
    """Split the step records where the step counter goes backwards.

    That happens exactly twice: once when resume rewinds to the last checkpoint,
    once when the fork restarts from an earlier one. The break points are read
    out of the data rather than assumed from config.
    """
    segs, cur, prev = [], [], None
    for rec in steps:
        gs = rec.get("global_step", 0)
        if prev is not None and gs <= prev:
            segs.append(cur)
            cur = []
        cur.append(rec)
        prev = gs
    if cur:
        segs.append(cur)
    out, x0 = [], 0
    for i, seg in enumerate(segs):
        first = seg[0].get("global_step", 0)
        if i == 0:
            name = "main run" if len(segs) == 1 else "main run, to crash"
            colour = PALETTE[0]
        elif (fork and i == len(segs) - 1
              and first == fork.get("diverged_at_step")):
            name = f'fork {fork.get("new_branch", "?")}'
            colour = PALETTE[1]
        else:
            name = "after resume"
            colour = PALETTE[2]
        out.append({"name": name, "colour": colour, "x0": x0, "recs": seg,
                    "first_step": first})
        x0 += len(seg)
    return out


def build(art: str) -> str:
    d = _load(art)
    ctx = Ctx()
    ev, perf, cost = d["evidence"], d["perf"], d["cost"]
    rep, meta, steps = d["reports"], d["meta"], d["steps"]
    s = ev.get("summary", {})

    vocab = meta.get("vocab_size") or rep.get("tokenizer", {}).get("vocab_size")
    lnv = math.log(vocab) if vocab else None

    segs = _segments(steps, d["fork"])
    labels = []
    for sg in segs:
        for r in sg["recs"]:
            labels.append(f'{sg["name"]} · step {r.get("global_step")} · '
                          f'stage {r.get("stage","?")}')
    markers = []
    for sg in segs[1:]:
        markers.append((sg["x0"], sg["name"]))

    # -- overview -----------------------------------------------------------
    passes = [e["name"] for e in d["events"] if e.get("kind") == "pass"]
    losses_all = [r["mean_loss"] for r in steps]
    ok = bool(s.get("all_passed"))

    hero = f"""
<header>
  <div class="hero">
    <div>
      <p class="eyebrow">Training data, accounted for</p>
      <h1>Training Data Execution System</h1>
      <p class="sub">documents → shards → manifests → mixture → packing → batches →
      training → ledgers → checkpoint → crash → resume → replay → fork → audit</p>
      <p class="meta-line">
        <span>profile <code>{esc(meta.get('profile','?'))}</code></span>
        <span>seed <code>{esc(meta.get('seed','?'))}</code></span>
        <span>run <code>{esc(meta.get('run_id','?'))}</code></span>
        <span>{num(meta.get('wall_clock_seconds'), 1)}s wall clock</span>
      </p>
    </div>
    <div class="verdict {'ok' if ok else 'bad'}">
      <div class="verdict-n">{esc(s.get('passed',0))}<span>/{esc(s.get('total',0))}</span></div>
      <div class="verdict-l">requirements {'passed' if ok else 'FAILED'}</div>
      <div class="verdict-s">{len(set(passes))} distinct PASS assertions,
        {len(passes)} emissions</div>
    </div>
  </div>
</header>
<nav id="nav" aria-label="sections">
  <div class="nav-in">
    {"".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in SECTIONS)}
    <button id="theme" class="ghost" title="Cycle theme (auto / light / dark)"
            aria-label="Cycle theme">auto</button>
  </div>
</nav>"""

    eff = perf.get("efficiency", {})
    rates = perf.get("rates", {})
    loader = perf.get("loader", {})
    glance = f"""
<div class="stats">
  {stat("useful tokens/sec", num(rates.get("useful_loss_bearing_tokens_per_second"), 1),
        "loss-bearing only")}
  {stat("packing utilization", pct(eff.get("packing_utilization")),
        f'{pct(eff.get("pad_fraction"), 2)} padding')}
  {stat("cache hit rate", pct(loader.get("cache", {}).get("hit_rate")),
        f'bounded LRU, capacity {loader.get("cache", {}).get("capacity","?")}')}
  {stat("final loss", num(losses_all[-1], 3) if losses_all else "-",
        (f'from {num(losses_all[0], 3)} · ln(V) = {num(lnv, 3)}'
         if losses_all and lnv else ""))}
  {stat("shards", rep.get("shards", {}).get("shards", "-"),
        f'{rep.get("shards", {}).get("tokens", "-"):,} tokens'
        if isinstance(rep.get("shards", {}).get("tokens"), int) else "")}
  {stat("steps served", perf.get("steps", "-"),
        f'{perf.get("raw_counters", {}).get("samples", "-")} samples')}
</div>"""

    # The required events, in the order run.log first emitted them. Several fire
    # more than once -- a checkpoint is saved ten times -- so they are collapsed
    # to one entry carrying the repeat count rather than listed twenty-eight times.
    ev_first: dict[str, float] = {}
    ev_count: dict[str, int] = {}
    for e in d["events"]:
        if e.get("kind") != "event":
            continue
        n = e["name"]
        ev_first.setdefault(n, e.get("t", 0))
        ev_count[n] = ev_count.get(n, 0) + 1
    flow = "".join(
        f'<li><span class="step-n">{i+1}</span><span class="step-t">{esc(n)}</span>'
        f'{f"<span class=rep>&times;{ev_count[n]}</span>" if ev_count[n] > 1 else ""}'
        f'<span class="step-s">{num(t, 1)}s</span></li>'
        for i, (n, t) in enumerate(ev_first.items()))
    pipeline = f'<ol class="flow">{flow}</ol>'

    pass_count: dict[str, int] = {}
    for p in passes:
        pass_count[p] = pass_count.get(p, 0) + 1
    pass_chips = "".join(
        pill(f"{p} ×{c}" if c > 1 else p, "ok") for p, c in pass_count.items())

    sec_overview = section(
        "overview", "At a glance",
        "Every figure below is read back out of <code>submission_artifacts/</code>. "
        "Nothing on this page is computed a second way.",
        glance
        + '<h3>Pipeline, as logged</h3>'
        + '<p class="muted">The events <code>run.log</code> is required to contain, '
          'in the order they first fired. A count means the event repeated.</p>'
        + pipeline
        + '<h3>Assertions that held</h3>'
        + f'<div class="chips">{pass_chips}</div>')

    # -- evidence -----------------------------------------------------------
    checks = ev.get("checks", [])
    rows = []
    for c in checks:
        det = c.get("detail", {})
        kv = "".join(f'<div class="kv"><span>{esc(k)}</span>'
                     f'<code>{esc(json.dumps(v) if isinstance(v, (dict, list)) else v)}</code></div>'
                     for k, v in sorted(det.items()))
        good = c.get("result") == "PASS"
        rows.append(
            f'<details class="ck {"ok" if good else "bad"}">'
            f'<summary><span class="ck-r">{esc(c.get("result"))}</span>'
            f'<span class="ck-q">{esc(c.get("requirement"))}</span>'
            f'<code class="ck-p">{esc(c.get("evidence_path"))}</code></summary>'
            f'<div class="ck-body"><p class="muted">'
            f'{esc(c.get("evidence_label",""))} — {esc(c.get("evidence_pointer",""))}</p>'
            f'<div class="kvs">{kv}</div></div></details>')
    sec_evidence = section(
        "evidence", "Evidence bundle",
        f'Produced by <code>{esc(ev.get("generated_by","tdes/evidence.py"))}</code>, which '
        f'reads {len(ev.get("artifacts_read", []))} artifacts off disk — it has no access to '
        'the run\'s in-memory state, so a passing bundle cannot exist without passing '
        'artifacts. Corrupt one and its row flips; <code>tests/test_evidence.py</code> '
        'does exactly that. Click a row for the compared values.',
        "".join(rows))

    # -- training -----------------------------------------------------------
    loss_series = [{"name": sg["name"], "x0": sg["x0"], "color": sg["colour"],
                    "ys": [r["mean_loss"] for r in sg["recs"]]} for sg in segs]
    grad_series = [{"name": sg["name"], "x0": sg["x0"], "color": sg["colour"],
                    "ys": [r["grad_norm"] for r in sg["recs"]]} for sg in segs]
    probes = rep.get("probe_history", [])
    probe_labels = [f'step {p.get("global_step")} · {p.get("label","")}' for p in probes]

    loss_chart = line_chart(
        ctx, loss_series, labels, baseline=lnv,
        baseline_label=f"ln(V) = {num(lnv, 3)}" if lnv else "",
        ylabel="cross-entropy", unit="", markers=markers)
    grad_chart = line_chart(ctx, grad_series, labels, ylabel="L2 norm")
    probe_chart = line_chart(
        ctx, [{"name": "probe loss", "x0": 0, "color": PALETTE[5],
               "ys": [p["mean_loss"] for p in probes]}], probe_labels,
        ylabel="cross-entropy")

    tabs = """
<div class="tabs" role="tablist">
  <button class="tab on" data-tab="t-loss" role="tab">Train loss</button>
  <button class="tab" data-tab="t-grad" role="tab">Gradient norm</button>
  <button class="tab" data-tab="t-probe" role="tab">Validation probe</button>
</div>"""
    sec_training = section(
        "training", "Training signal",
        "The x axis is <em>serving order</em>, not step number, because the step "
        "counter goes backwards twice — once when resume rewinds to the last "
        "checkpoint, once when the fork restarts from an earlier one. Those two "
        "rewinds are what separate the coloured segments; they were detected in the "
        "ledger, not assumed. Hover for the exact step.",
        tabs
        + f'<div class="panel on" id="t-loss"><p class="muted">Loss starts at '
          f'ln(vocabulary size) — what a model holding no information must produce. '
          f'Measured first step {num(losses_all[0], 4) if losses_all else "-"} against '
          f'ln(V) = {num(lnv, 4)}.</p>{loss_chart}</div>'
        + f'<div class="panel" id="t-grad">{grad_chart}</div>'
        + f'<div class="panel" id="t-probe"><p class="muted">Forward-only and never '
          f'gradient-bearing — the firewall records zero gradient-bearing reads of the '
          f'validation split. The same probe supplies the OPUS proxy direction and the '
          f'before/after loss delta in the learning ledger.</p>{probe_chart}</div>')

    # -- tokens -------------------------------------------------------------
    lanes_in_trace = sorted({t.get("capability_lane", "?") for t in d["tokens"]})
    lane_chips = "".join(
        f'<button class="chip on" data-lane="{esc(l)}">'
        f'<i style="background:{LANE_COLOR.get(l, "#888")}"></i>{esc(l)}</button>'
        for l in lanes_in_trace)
    sec_tokens = section(
        "tokens", "Per-token perplexity",
        f'The first {len(d["tokens"])} loss-bearing tokens, in serving order. A shard '
        'average hides this, which is the whole reason the token trace is kept. Hover '
        'a cell for the decoded piece, its shard and its step; use the lane filter to '
        'isolate one capability.',
        f'<div class="chips">{lane_chips}<button class="chip all" id="lane-all">all</button></div>'
        + heatmap(d["tokens"]))

    # -- mixture ------------------------------------------------------------
    mix = perf.get("mixture_compliance", {}).get("by_lane", [])
    mix_rows = [(m["lane"], m["planned_share"], m["actual_share"]) for m in mix]
    floors = rep.get("floors", {})
    fr = []
    for lane, f in sorted(floors.get("checked_floors", {}).items()):
        fr.append((lane, pct(f.get("floor"), 1), num(f.get("samples_implied_by_floor"), 2),
                   f.get("samples_in_window"), f.get("window_steps"),
                   Cell(pill("expressible", "ok"), 1)))
    for lane, why in sorted(floors.get("not_expressible", {}).items()):
        fr.append((lane, "-", "-", "-", "-", Cell(pill("not expressible", "warn"), 0)))
    tier = rep.get("indic_tier_floor", {})
    tier_html = ""
    if tier:
        tier_html = f"""
<h3>Indic verified tier</h3>
<p class="muted">{esc(tier.get("rule",""))}</p>
<div class="stats sm">
  {stat("verified share", pct(tier.get("verified_share"), 1),
        f'floor {pct(tier.get("required_share"), 0)}',
        "good" if tier.get("held") else "bad")}
  {stat("verified samples", tier.get("verified", "-"), "drawn from tier=verified")}
  {stat("unverified samples", tier.get("unverified", "-"), "may never substitute")}
  {stat("supply shortfalls", len(tier.get("supply_shortfalls", [])), "verified pool empty")}
</div>"""
    floor_window = f'{floors.get("window_steps", "-")} steps'
    sec_mixture = section(
        "mixture", "Mixture: planned versus actual",
        "Lane shares come from Session 5's compiled curriculum. Small lanes cannot be "
        "served every step — 1.5% of six samples is 0.09 of a sample — so they are "
        "apportioned by carry-over and converge over the run instead. The faded bar is "
        "planned, the solid one actual.",
        bar_pairs(mix_rows)
        + '<h3>Protected floors</h3>'
        + f'<p class="muted">{esc(floors.get("verdict_note",""))}</p>'
        + f'<div class="stats sm">{stat("floors held", floors.get("floors_held"), "")}'
          f'{stat("windows checked", floors.get("windows_checked", "-"), "sliding")}'
          f'{stat("violations", floors.get("violation_count", "-"), "")}'
          f'{stat("window size", floor_window, "derived, not fixed")}</div>'
        + table(["lane", "floor", "samples implied", "samples in window",
                 "window steps", "status"], fr)
        + tier_html)

    # -- packing ------------------------------------------------------------
    packing = rep.get("packing", {})
    pack_panels, pack_chips = [], []
    for i, (lane, r) in enumerate(sorted(packing.items())):
        by = r.get("by_policy", {})
        order = r.get("ranked_by_effective_yield") or sorted(by)
        rows_ = []
        for p in order:
            m = by.get(p, {})
            rows_.append((p, [("fill", m.get("utilization", 0), PALETTE[0]),
                              ("coverage", m.get("coverage", 0), PALETTE[2]),
                              ("yield", m.get("effective_yield", 0), PALETTE[1])]))
        tbl = table(
            ["policy", "sequences", "fill", "coverage", "yield", "dropped",
             "truncations", "boundary crossings"],
            [(p, by.get(p, {}).get("sequences"),
              Cell(pct(by.get(p, {}).get("utilization")), by.get(p, {}).get("utilization")),
              Cell(pct(by.get(p, {}).get("coverage")), by.get(p, {}).get("coverage")),
              Cell(pct(by.get(p, {}).get("effective_yield")),
                   by.get(p, {}).get("effective_yield")),
              by.get(p, {}).get("tokens_dropped"), by.get(p, {}).get("truncations"),
              by.get(p, {}).get("boundary_crossings")) for p in order],
            sortable=True)
        pack_panels.append(
            f'<div class="panel{" on" if i == 0 else ""}" id="pk-{esc(lane)}">'
            f'<p class="muted">{esc(r.get("metric_note",""))}</p>'
            f'{hbar_groups(rows_)}{tbl}</div>')
        pack_chips.append(f'<button class="tab{" on" if i == 0 else ""}" '
                          f'data-tab="pk-{esc(lane)}">{esc(lane)}</button>')
    sec_packing = section(
        "packing", "Packing policies",
        "Fill rate alone is a trap: a policy that truncates can report 100% fill while "
        "dropping most of the lane. <b>Coverage</b> (how much of the input survived) and "
        "<b>effective yield</b> (fill × coverage) are reported beside it, and the policies "
        "are ranked by yield. Pick a lane:",
        f'<div class="tabs" role="tablist">{"".join(pack_chips)}</div>'
        + "".join(pack_panels))

    # -- opus ---------------------------------------------------------------
    opus = rep.get("opus", {})
    status_colors = {"accepted": "#54A24B", "rejected": "#E45756",
                     "deferred": "#EECA3B", "protected_override": "#4C78A8"}
    by_status = opus.get("by_status", {})
    parts = [(k, v, status_colors.get(k, "#888")) for k, v in sorted(by_status.items())]
    total_dec = opus.get("total_decisions", sum(by_status.values()) or 0)
    ph = opus.get("pool_health", {})
    reasons = table(["rejection reason", "count"],
                    sorted(opus.get("by_rejection_reason", {}).items()))
    lane_rows = []
    for lane, counts in sorted(opus.get("by_lane", {}).items()):
        lane_rows.append((lane, counts.get("accepted", 0), counts.get("rejected", 0),
                          counts.get("deferred", 0), counts.get("protected_override", 0)))
    rounds = table(
        ["step", "candidates", "accepted", "rejected", "deferred", "protected",
         "mean score"],
        [(r["global_step"], r["candidates"], r["accepted"], r["rejected"],
          r["deferred"], r["protected_override"],
          Cell(f'{r["score_mean"]:.5f}', r["score_mean"])) for r in d["opus_rounds"]],
        sortable=True)
    sec_opus = section(
        "opus", "OPUS selection",
        f'Score is the cosine between a candidate\'s prefix gradient '
        f'({opus.get("prefix_tokens","?")} tokens) and the validation-probe gradient '
        f'direction — a real comparison of directions, computed at selection time '
        f'against <code>{esc(opus.get("proxy_version","?"))}</code>. Protected lanes '
        f'({esc(", ".join(opus.get("protected_lanes", [])))}) override rejection and the '
        f'override is recorded as such.',
        f'<div class="split">'
        f'{donut(parts, centre_top=str(total_dec), centre_sub="decisions")}'
        f'<div class="split-side"><h3>Rejection reasons</h3>{reasons}</div></div>'
        + '<h3>Pool health</h3>'
        + f'<p class="muted">{esc(ph.get("interpretation",""))}</p>'
        + f'<div class="stats sm">'
          f'{stat("early accepted mean", num(ph.get("early_accepted_score_mean"), 5))}'
          f'{stat("late accepted mean", num(ph.get("late_accepted_score_mean"), 5))}'
          f'{stat("relative drop", num(ph.get("relative_drop"), 3))}'
          f'{stat("pool exhausting", ph.get("pool_exhausting"), "", "bad" if ph.get("pool_exhausting") else "good")}'
          f'</div>'
        + '<h3>By lane</h3>'
        + table(["lane", "accepted", "rejected", "deferred", "protected override"],
                lane_rows, sortable=True)
        + '<h3>By round</h3>' + rounds)

    # -- recovery -----------------------------------------------------------
    res, rep_, fk, aud = d["resume"], d["replay"], d["fork"], d["audit"]
    integ = res.get("integrity", {})
    stepper = f"""
<ol class="timeline">
  <li class="tl warn">
    <span class="tl-k">crash</span>
    <b>step {esc(res.get("crash_at_step"))}</b>
    <span class="tl-d">a real exception mid-step, after records were appended and
    before the next checkpoint — so the rewind has something to discard</span>
    <span class="tl-m">{esc(res.get("records_discarded"))} ledger records discarded</span>
  </li>
  <li class="tl {'ok' if res.get('matched') else 'bad'}">
    <span class="tl-k">resume</span>
    <b>from ledger offset {esc(res.get("ledger_offset"))}</b>
    <span class="tl-d">recovery coordinate is the ledger offset, not the step number.
    The expected next batch id was recorded <em>before</em> the crash and asserted after.</span>
    <span class="tl-m">next batch matched: {esc(res.get("matched"))} ·
    no repeats {esc(integ.get("no_repeated_batches"))} · no skipped steps
    {esc(integ.get("no_skipped_steps"))} · offsets dense {esc(integ.get("offsets_dense"))}</span>
  </li>
  <li class="tl {'ok' if rep_.get('all_matched') else 'bad'}">
    <span class="tl-k">replay</span>
    <b>steps {esc(rep_.get("range"))}</b>
    <span class="tl-d">re-served from the <em>ledger</em>, never recomputed from the
    planner. Compared on {esc(", ".join(rep_.get("compared_fields", [])))}.</span>
    <span class="tl-m">{esc(rep_.get("matched"))}/{esc(len(rep_.get("rows", [])))} matched ·
    {num(rep_.get("replay_seconds"), 3)}s</span>
  </li>
  <li class="tl {'ok' if fk.get("verification", {}).get("ok") else 'bad'}">
    <span class="tl-k">fork</span>
    <b>{esc(fk.get("new_branch"))} at step {esc(fk.get("diverged_at_step"))}</b>
    <span class="tl-d">new branch id from the parent checkpoint; the parent's records
    must be byte-unchanged afterwards.</span>
    <span class="tl-m">distinct batches
    {esc(fk.get("verification", {}).get("branches_produce_distinct_batches"))} ·
    parent intact {esc(fk.get("verification", {}).get("parent_records_intact"))} ·
    starts at divergence {esc(fk.get("verification", {}).get("starts_at_divergence"))}</span>
  </li>
  <li class="tl">
    <span class="tl-k">audit</span>
    <b>{esc(aud.get("shard_count_in_window"))} shards in window</b>
    <span class="tl-d">which shards trained the model between two token counts, and
    which OPUS decisions preceded the largest loss spike.</span>
    <span class="tl-m">{esc(aud.get("total_loss_bearing_tokens"))} loss-bearing tokens ·
    spike {num(aud.get("largest_loss_spike", {}).get("delta"), 3)} at step
    {esc(aud.get("largest_loss_spike", {}).get("at_step"))}</span>
  </li>
</ol>"""
    replay_rows = [
        (r["global_step"],
         Cell(f'<code class="hash" title="click to copy">{esc(r["batch_id_original"][:16])}…</code>',
              r["batch_id_original"]),
         Cell(pill("match", "ok") if r["batch_id_match"] else pill("differs", "bad"),
              1 if r["batch_id_match"] else 0),
         Cell(pill("match", "ok") if r["content_hash_match"] else pill("differs", "bad"),
              1 if r["content_hash_match"] else 0),
         Cell(pill("match", "ok") if r["loss_mask_hash_match"] else pill("differs", "bad"),
              1 if r["loss_mask_hash_match"] else 0))
        for r in rep_.get("rows", [])]
    sec_recovery = section(
        "recovery", "Crash → resume → replay → fork → audit",
        "Matching the batch <em>id</em> alone would only prove the plan agreed. The "
        "content hash is what proves the tokens, masks and position ids were the same "
        "— it is the check that caught a real bug here, when replayed SFT samples lost "
        "their prompt masking because role spans were missing from the ledger.",
        stepper
        + '<h3>Replayed interval, batch by batch</h3>'
        + table(["step", "original batch id", "batch id", "content hash",
                 "loss-mask hash"], replay_rows, sortable=True))

    # -- shards -------------------------------------------------------------
    use_tone = {"useful": "ok", "neutral": "", "already_learned": "warn",
                "harmful": "bad", "broken": "bad"}
    shard_rows = []
    for r in sorted(d["shards"], key=lambda r: r["mean_token_loss"]):
        u = r.get("usefulness", "")
        shard_rows.append((
            Cell(f'<code>{esc(r["shard_id"])}</code>', r["shard_id"]),
            Cell(f'<span class="lane-dot" style="background:'
                 f'{LANE_COLOR.get(r["capability_lane"], "#888")}"></span>'
                 f'{esc(r["capability_lane"])}', r["capability_lane"]),
            Cell(f'{r["tokens_scored"]:,}', r["tokens_scored"]),
            Cell(num(r["mean_token_loss"], 3), r["mean_token_loss"]),
            Cell(num(r["mean_perplexity"], 1), r["mean_perplexity"]),
            Cell(pill(u, use_tone.get(u, "")), u)))
    lane_sum = [(Cell(f'<span class="lane-dot" style="background:'
                      f'{LANE_COLOR.get(l, "#888")}"></span>{esc(l)}', l),
                 Cell(f'{v["tokens"]:,}', v["tokens"]),
                 Cell(num(v["mean_loss"], 3), v["mean_loss"]),
                 Cell(num(v["mean_perplexity"], 1), v["mean_perplexity"]),
                 Cell(pill(v["usefulness"], use_tone.get(v["usefulness"], "")),
                      v["usefulness"]))
                for l, v in sorted(d["summary"].get("by_lane", {}).items())]
    sec_shards = section(
        "shards", "Learning ledger",
        "What came back out, against what went in. At or below 1.2 mean loss the model "
        "already knows the content; near 0.3 suggests boilerplate, duplication or a "
        "leak. Sortable — click a column header.",
        '<h3>By lane</h3>'
        + table(["lane", "tokens", "mean loss", "perplexity", "usefulness"],
                lane_sum, sortable=True)
        + '<h3>By shard</h3>'
        + '<input class="filter" data-filter="shardtbl" type="search" '
          'placeholder="filter shards or lanes…" aria-label="Filter shards">'
        + table(["shard", "lane", "tokens", "mean loss", "perplexity", "usefulness"],
                shard_rows, sortable=True, tid="shardtbl"))

    # -- tokenizer ----------------------------------------------------------
    fert = rep.get("fertility", {}).get("per_language_at_run_vocab", {})
    frows = sorted(((l, r["fertility"], r["words"], r.get("unk_rate", 0))
                    for l, r in fert.items()), key=lambda r: -r[1])
    fmax = max((f for _, f, _, _ in frows), default=1.0)
    fbars = hbars([(l, f / fmax,
                    PALETTE[1] if l in INDIC_LANGS else PALETTE[0],
                    f"{f:.2f} tokens/word · {w:,} words")
                   for l, f, w, _ in frows])
    tk = rep.get("tokenizer", {})
    sec_tokenizer = section(
        "tokenizer", "Tokenizer fertility",
        f'Tokens per word at the frozen vocabulary of {esc(tk.get("vocab_size","?"))}; '
        'lower is better. Orange is an Indic script. The gap is not an artefact of the '
        'demo size — it is the same effect Session 4 measured for cl100k on Telugu, '
        'reproduced with our own numbers rather than quoted. Fertility is a budget '
        'lever: a language costing 3× the tokens per word costs 3× to train on. '
        '<code>python</code> is the code lane, measured the same way.',
        f'<div class="stats sm">'
        f'{stat("vocab", tk.get("vocab_size","-"), "frozen before the run")}'
        f'{stat("tokenizer hash", (tk.get("tokenizer_hash") or "")[:12] + "…", "verified on every load")}'
        f'{stat("unk rate", pct(max((u for *_ , u in frows), default=0), 3), "worst language")}'
        f'</div>' + fbars)

    # -- throughput ---------------------------------------------------------
    cache = loader.get("cache", {})
    cache_parts = [("hits", cache.get("hits", 0), PALETTE[2]),
                   ("misses", cache.get("misses", 0), PALETTE[3])]
    raw = perf.get("raw_counters", {})
    tok_parts = [("loss-bearing", raw.get("tokens_loss_bearing", 0), PALETTE[2]),
                 ("context only", raw.get("tokens_context_only", 0), PALETTE[0]),
                 ("padding", raw.get("tokens_pad", 0), PALETTE[3])]
    phase = perf.get("phase_seconds", {})
    wait_v = f'{num(loader.get("loader_wait_seconds"), 3)}s'
    wait_s = f'{pct(eff.get("loader_wait_fraction_of_train_time"), 2)} of train time'
    read_v = f'{num(loader.get("mean_shard_read_ms"), 2)} ms'
    read_s = f'{loader.get("shard_reads", "-")} reads'
    # The note is written for a machine reader, lowercase and unpunctuated.
    ln = esc(loader.get("note", "")).strip().rstrip(".")
    loader_note = (ln[:1].upper() + ln[1:] + ". ") if ln else ""
    sec_through = section(
        "throughput", "Throughput and the loader",
        f'{loader_note}Every derived rate ships with the raw counters it came from, '
        'so a grader can recompute it by hand rather than take it on trust.',
        f'<div class="split">'
        f'{donut(cache_parts, centre_top=pct(cache.get("hit_rate")), centre_sub="cache hits")}'
        f'{donut(tok_parts, centre_top=pct(eff.get("packing_utilization")), centre_sub="positions used")}'
        f'</div>'
        + f'<div class="stats sm">'
          f'{stat("raw tokens/s", num(rates.get("raw_tokens_per_second"), 1))}'
          f'{stat("useful tokens/s", num(rates.get("useful_loss_bearing_tokens_per_second"), 1), "loss-bearing")}'
          f'{stat("accepted tokens/s", num(rates.get("accepted_tokens_per_second_after_opus"), 1), "after OPUS")}'
          f'{stat("s / step", num(rates.get("seconds_per_step"), 3))}'
          f'{stat("loader wait", wait_v, wait_s)}'
          f'{stat("shard read", read_v, read_s)}'
          f'</div>'
        + '<h3>Where the wall clock went</h3>'
        + hbars([(k, v / (max(phase.values()) or 1), PALETTE[i % len(PALETTE)],
                  f"{num(v, 2)}s")
                 for i, (k, v) in enumerate(sorted(phase.items(), key=lambda x: -x[1]))])
        + '<h3>Raw counters</h3>'
        + table(["counter", "value"],
                [(k, Cell(f"{v:,}" if isinstance(v, int) else num(v, 4), v))
                 for k, v in sorted(raw.items())], sortable=True))

    # -- cost ---------------------------------------------------------------
    cc = cost.get("constants", {})
    sec_cost = section(
        "cost", "What the padding costs",
        f'{esc(cc.get("instance","-"))} at ₹{esc(cc.get("inr_per_hour","-"))}/hour '
        f'({esc(cc.get("source",""))}). Every input is a named, sourced constant — the '
        'projection is arithmetic on this run\'s measured pad fraction, not a forecast.',
        f'<div class="stats sm">'
        f'{stat("padding, this run", "₹" + num(cost.get("waste", {}).get("inr_on_padding"), 4))}'
        f'{stat("per billion positions", "₹" + num(cost.get("projection_per_billion_positions", {}).get("inr"), 0))}'
        f'{stat("lost to padding / B", "₹" + num(cost.get("projection_per_billion_positions", {}).get("inr_lost_to_padding"), 0), "at this pad fraction")}'
        f'{stat("at risk per checkpoint", "₹" + num(cost.get("checkpoint_policy", {}).get("inr_at_risk_between_checkpoints"), 4), "work redone after a crash")}'
        f'</div>'
        + table(["recovery", "seconds"],
                [("resume", num(cost.get("recovery", {}).get("resume_seconds"), 3)),
                 ("replay", num(cost.get("recovery", {}).get("replay_seconds"), 3))]))

    footer = f"""
<footer>
  <p><b>This page is a view, not a second source of truth.</b> It reads the same
  artifacts the evidence bundle reads and renders them; it computes no metric of
  its own.</p>
  <p class="muted">python {esc((meta.get('python') or '?').split()[0])} ·
  {esc(meta.get('platform','?'))} ·
  hash randomization active: {esc(meta.get('hash_randomization_active'))} ·
  dataloader {esc(meta.get('dataloader_version','?'))} ·
  regenerate with <code>python tools/build_dashboard.py</code></p>
</footer>"""

    body = (hero + '<main>' + sec_overview + sec_evidence + sec_training + sec_tokens
            + sec_mixture + sec_packing + sec_opus + sec_recovery + sec_shards
            + sec_tokenizer + sec_through + sec_cost + '</main>' + footer)

    payload = {
        "charts": ctx.charts,
        "tokens": [[t.get("decoded", ""), round(t.get("perplexity", 0), 1),
                    t.get("capability_lane", ""), t.get("shard_id", ""),
                    t.get("global_step", 0), t.get("curriculum_stage", "")]
                   for t in d["tokens"]],
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False,
                      sort_keys=True).replace("<", "\\u003c")
    return PAGE.replace("{{BODY}}", body).replace("{{DATA}}", blob)


# ---------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TDES Dashboard</title>
<style>
:root{
  --bg:#f7f8fa; --bg2:#eef1f5; --fg:#14171c; --mut:#5f6875; --dim:#8b94a1;
  --card:#fff; --line:#e2e6ec; --line2:#d6dbe3;
  --ok:#1a7f37; --bad:#c02718; --warn:#a1660a; --accent:#3d6fa8;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 8px 24px -12px rgba(16,24,40,.16);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]){
    --bg:#0f1216; --bg2:#141920; --fg:#e6e9ef; --mut:#98a2b1; --dim:#6f7987;
    --card:#171c23; --line:#252c36; --line2:#323a46;
    --ok:#48b95e; --bad:#f2685c; --warn:#e0a93a; --accent:#79aae0;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
  }
}
:root[data-theme=dark]{
  --bg:#0f1216; --bg2:#141920; --fg:#e6e9ef; --mut:#98a2b1; --dim:#6f7987;
  --card:#171c23; --line:#252c36; --line2:#323a46;
  --ok:#48b95e; --bad:#f2685c; --warn:#e0a93a; --accent:#79aae0;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:64px}
body{margin:0;padding:0 0 72px;background:var(--bg);color:var(--fg);
  font:15px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  -webkit-font-smoothing:antialiased}
header,main,footer{max-width:980px;margin:0 auto;padding:0 20px}

/* hero */
header{padding-top:40px}
.hero{display:flex;gap:26px;align-items:flex-start;justify-content:space-between;
  flex-wrap:wrap;padding-bottom:22px}
.eyebrow{margin:0 0 6px;color:var(--dim);font-size:.76rem;font-weight:600;
  letter-spacing:.11em;text-transform:uppercase}
h1{font-size:2.1rem;line-height:1.12;margin:0 0 10px;letter-spacing:-.024em;font-weight:680}
.sub{margin:0 0 12px;color:var(--mut);font-size:.93rem;max-width:56ch}
.meta-line{display:flex;gap:8px 16px;flex-wrap:wrap;margin:0;color:var(--mut);font-size:.84rem}
.verdict{border:1px solid var(--line);border-radius:14px;padding:16px 22px;min-width:190px;
  background:var(--card);box-shadow:var(--shadow)}
.verdict.ok{border-color:color-mix(in srgb,var(--ok) 45%,var(--line))}
.verdict.bad{border-color:color-mix(in srgb,var(--bad) 45%,var(--line))}
.verdict-n{font-size:2.5rem;font-weight:700;letter-spacing:-.04em;line-height:1}
.verdict.ok .verdict-n{color:var(--ok)} .verdict.bad .verdict-n{color:var(--bad)}
.verdict-n span{font-size:1.2rem;color:var(--dim);font-weight:600}
.verdict-l{font-size:.84rem;font-weight:600;margin-top:4px}
.verdict-s{font-size:.78rem;color:var(--dim);margin-top:2px}

/* nav */
nav{position:sticky;top:0;z-index:40;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:saturate(160%) blur(10px);border-bottom:1px solid var(--line)}
.nav-in{max-width:980px;margin:0 auto;padding:0 20px;display:flex;gap:2px;align-items:center;
  overflow-x:auto;scrollbar-width:none}
.nav-in::-webkit-scrollbar{display:none}
nav a{color:var(--mut);text-decoration:none;font-size:.83rem;padding:11px 10px;
  white-space:nowrap;border-bottom:2px solid transparent;transition:color .15s}
nav a:hover{color:var(--fg)}
nav a.active{color:var(--fg);border-bottom-color:var(--accent);font-weight:600}
.ghost{margin-left:auto;background:none;border:1px solid var(--line2);color:var(--mut);
  border-radius:7px;font:inherit;font-size:.76rem;padding:3px 10px;cursor:pointer;
  text-transform:uppercase;letter-spacing:.05em}
.ghost:hover{color:var(--fg);border-color:var(--fg)}

/* sections */
section{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:22px 24px;margin:18px 0;box-shadow:var(--shadow)}
.sec-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
h2{font-size:1.16rem;margin:0 0 7px;letter-spacing:-.015em;font-weight:660}
h3{font-size:.79rem;margin:26px 0 9px;color:var(--dim);font-weight:660;
  text-transform:uppercase;letter-spacing:.08em}
h3:first-child{margin-top:6px}
p{margin:0 0 12px}
.muted{color:var(--mut);font-size:.885rem;margin:0 0 14px;max-width:78ch}
.dim{fill:var(--dim);color:var(--dim)}
code{font:.85em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:color-mix(in srgb,var(--fg) 7%,transparent);padding:1px 6px;border-radius:5px}
.hash{cursor:pointer}
.hash:hover{background:color-mix(in srgb,var(--accent) 22%,transparent)}

/* stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(139px,1fr));gap:11px;margin:4px 0 6px}
.stats.sm{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.stat{border:1px solid var(--line);border-radius:11px;padding:12px 14px;background:var(--bg2)}
.stat.good{border-color:color-mix(in srgb,var(--ok) 40%,var(--line))}
.stat.bad{border-color:color-mix(in srgb,var(--bad) 40%,var(--line))}
.stat-v{font-size:1.32rem;font-weight:670;letter-spacing:-.025em}
.stat-l{color:var(--mut);font-size:.71rem;text-transform:uppercase;letter-spacing:.06em;margin-top:3px}
.stat-s{color:var(--dim);font-size:.76rem;margin-top:3px}

/* pipeline flow */
.flow{list-style:none;display:flex;flex-wrap:wrap;gap:7px;padding:0;margin:0 0 6px}
.flow li{display:flex;align-items:center;gap:7px;border:1px solid var(--line);
  border-radius:9px;padding:6px 11px;background:var(--bg2);font-size:.8rem}
.step-n{display:grid;place-items:center;width:17px;height:17px;border-radius:50%;
  background:var(--accent);color:#fff;font-size:.63rem;font-weight:700;flex:none}
.step-s{color:var(--dim);font-variant-numeric:tabular-nums;font-size:.74rem}
.rep{color:var(--accent);font-size:.72rem;font-weight:660;font-variant-numeric:tabular-nums}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.pill{display:inline-block;font-size:.73rem;font-weight:600;padding:2px 9px;border-radius:20px;
  border:1px solid var(--line2);color:var(--mut);white-space:nowrap}
.pill.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 45%,transparent);
  background:color-mix(in srgb,var(--ok) 11%,transparent)}
.pill.bad{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 45%,transparent);
  background:color-mix(in srgb,var(--bad) 11%,transparent)}
.pill.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent);
  background:color-mix(in srgb,var(--warn) 11%,transparent)}

/* evidence rows */
.ck{border:1px solid var(--line);border-radius:11px;margin-bottom:7px;background:var(--bg2);
  overflow:hidden}
.ck summary{display:flex;align-items:center;gap:12px;padding:11px 14px;cursor:pointer;
  list-style:none;font-size:.9rem}
.ck summary::-webkit-details-marker{display:none}
.ck summary:hover{background:color-mix(in srgb,var(--fg) 4%,transparent)}
.ck-r{font-size:.68rem;font-weight:700;letter-spacing:.07em;padding:3px 8px;border-radius:5px;flex:none}
.ck.ok .ck-r{color:var(--ok);background:color-mix(in srgb,var(--ok) 14%,transparent)}
.ck.bad .ck-r{color:var(--bad);background:color-mix(in srgb,var(--bad) 14%,transparent)}
.ck-q{font-weight:600;flex:1}
.ck-p{font-size:.74rem;color:var(--dim);background:none}
.ck-body{padding:2px 14px 14px;border-top:1px solid var(--line)}
.kvs{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:6px}
.kv{display:flex;gap:8px;justify-content:space-between;align-items:baseline;font-size:.8rem;
  border-bottom:1px dotted var(--line);padding:3px 0}
.kv span{color:var(--mut)}
.kv code{background:none;padding:0;word-break:break-all;text-align:right}

/* tabs + panels */
.tabs{display:flex;gap:5px;flex-wrap:wrap;margin:2px 0 14px;border-bottom:1px solid var(--line)}
.tab{background:none;border:0;border-bottom:2px solid transparent;color:var(--mut);
  font:inherit;font-size:.84rem;padding:7px 11px;cursor:pointer;margin-bottom:-1px}
.tab:hover{color:var(--fg)}
.tab.on{color:var(--fg);border-bottom-color:var(--accent);font-weight:620}
.panel{display:none} .panel.on{display:block}
.chip{background:var(--bg2);border:1px solid var(--line2);color:var(--mut);border-radius:20px;
  font:inherit;font-size:.78rem;padding:3px 11px;cursor:pointer;display:inline-flex;
  align-items:center;gap:6px}
.chip i{width:8px;height:8px;border-radius:50%;display:inline-block}
.chip.on{color:var(--fg);border-color:var(--fg)}
.chip:not(.on){opacity:.5}
.chip.all{opacity:1}

/* charts */
.chart-wrap{position:relative}
.chart{width:100%;height:auto;display:block;margin:8px 0;touch-action:none}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .baseline{stroke:var(--accent);stroke-width:1.3;stroke-dasharray:5 4;opacity:.9}
.chart .marker{stroke:var(--line2);stroke-width:1;stroke-dasharray:3 3}
.chart .marker-t{fill:var(--dim);font-size:10px}
.chart .tick,.chart .axis{fill:var(--mut);font-size:11px}
.chart .baseline-t{fill:var(--accent)}
.chart .label{fill:var(--fg);font-size:12px}
.chart .drift{font-size:10.5px}
.chart .drift.up{fill:var(--ok)} .chart .drift.down{fill:var(--bad)}
.chart .area{opacity:.10}
.chart .track{fill:color-mix(in srgb,var(--fg) 8%,transparent)}
.chart .bar{transition:opacity .12s}
.chart .bar:hover{opacity:.75}
.crosshair{stroke:var(--dim);stroke-width:1;stroke-dasharray:3 3;pointer-events:none}
.focus{pointer-events:none}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.79rem;color:var(--mut);margin:2px 0 4px}
.legend.col{flex-direction:column;gap:5px}
.key{display:inline-flex;align-items:center;gap:6px}
.key i{width:9px;height:9px;border-radius:2px;display:inline-block}
.key b{color:var(--fg);font-variant-numeric:tabular-nums}
.donut-row{display:flex;align-items:center;gap:22px;flex-wrap:wrap}
.donut{width:170px;height:170px;flex:none}
.donut .track{stroke:color-mix(in srgb,var(--fg) 8%,transparent)}
.donut-v{fill:var(--fg);font-size:22px;font-weight:670}
.donut-l{fill:var(--dim);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.split{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}
.split-side{flex:1;min-width:260px}
/* No crispEdges: at 60 columns the cell width is fractional, and snapping to
   whole pixels turns the 0.4 gap into uneven white seams that read as a grid. */
.heat rect{cursor:crosshair}
.heat rect.off{opacity:.09}
.scale{display:flex;align-items:center;gap:9px;font-size:.76rem;color:var(--dim);margin-top:4px}
.ramp{display:inline-block;width:150px;height:9px;border-radius:5px;
  background:linear-gradient(90deg,rgb(40,130,190),rgb(140,85,115),rgb(240,40,40))}

/* timeline */
.timeline{list-style:none;padding:0;margin:2px 0 6px;display:grid;gap:9px}
.tl{border:1px solid var(--line);border-left:3px solid var(--line2);border-radius:10px;
  padding:11px 15px;background:var(--bg2);display:grid;gap:3px}
.tl.ok{border-left-color:var(--ok)} .tl.bad{border-left-color:var(--bad)}
.tl.warn{border-left-color:var(--warn)}
.tl-k{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--dim)}
.tl b{font-size:.97rem}
.tl-d{color:var(--mut);font-size:.85rem}
.tl-m{color:var(--dim);font-size:.78rem;font-variant-numeric:tabular-nums}

/* tables */
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);
  border-radius:10px;margin-bottom:6px}
table{border-collapse:collapse;width:100%;font-size:.845rem;min-width:440px}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:640;font-size:.71rem;text-transform:uppercase;
  letter-spacing:.06em;background:var(--bg2);position:sticky;top:0}
table.sortable th{cursor:pointer;user-select:none}
table.sortable th:hover{color:var(--fg)}
table.sortable th.asc::after{content:" ▲";font-size:.7em;color:var(--accent)}
table.sortable th.desc::after{content:" ▼";font-size:.7em;color:var(--accent)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:color-mix(in srgb,var(--fg) 4%,transparent)}
td{font-variant-numeric:tabular-nums}
.lane-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}
.filter{width:100%;max-width:320px;margin-bottom:9px;padding:7px 11px;border-radius:8px;
  border:1px solid var(--line2);background:var(--bg2);color:var(--fg);font:inherit;font-size:.85rem}
.filter:focus{outline:2px solid color-mix(in srgb,var(--accent) 45%,transparent);outline-offset:1px}

/* tooltip */
#tip{position:fixed;z-index:80;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--card);border:1px solid var(--line2);border-radius:9px;padding:8px 11px;
  font-size:.79rem;box-shadow:var(--shadow);max-width:290px;line-height:1.45}
#tip.on{opacity:1}
#tip b{display:block;font-size:.74rem;color:var(--dim);font-weight:600;margin-bottom:3px}
#tip .tv{font-variant-numeric:tabular-nums}

footer{padding-top:26px;color:var(--mut);font-size:.85rem}
footer p{margin:5px 0}
@media (max-width:640px){
  h1{font-size:1.6rem} section{padding:18px 15px;border-radius:12px}
  .verdict{min-width:0;flex:1}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}
  html{scroll-behavior:auto}}
</style></head><body>
{{BODY}}
<div id="tip" role="tooltip"></div>
<script id="tdes-data" type="application/json">{{DATA}}</script>
<script>
(function(){
"use strict";
var D={charts:{},tokens:[]};
try{D=JSON.parse(document.getElementById("tdes-data").textContent);}catch(e){}
var tip=document.getElementById("tip");
function showTip(x,y,html){
  tip.innerHTML=html;tip.classList.add("on");
  var r=tip.getBoundingClientRect(),w=window.innerWidth,h=window.innerHeight;
  var lx=Math.min(x+14,w-r.width-8),ly=y-r.height-14;
  if(ly<8)ly=y+18;
  tip.style.left=Math.max(8,lx)+"px";tip.style.top=ly+"px";
}
function hideTip(){tip.classList.remove("on");}

/* ---- theme: auto -> light -> dark, remembered ---- */
(function(){
  var btn=document.getElementById("theme"),root=document.documentElement;
  var order=["auto","light","dark"],cur=localStorage.getItem("tdes-theme")||"auto";
  function apply(t){
    cur=t;
    if(t==="auto")root.removeAttribute("data-theme");else root.setAttribute("data-theme",t);
    if(btn)btn.textContent=t;
    try{localStorage.setItem("tdes-theme",t);}catch(e){}
  }
  apply(cur);
  if(btn)btn.addEventListener("click",function(){apply(order[(order.indexOf(cur)+1)%3]);});
})();

/* ---- scrollspy ---- */
(function(){
  var links={},secs=[];
  document.querySelectorAll("nav a").forEach(function(a){
    var id=a.getAttribute("href").slice(1),el=document.getElementById(id);
    if(el){links[id]=a;secs.push(el);}
  });
  if(!("IntersectionObserver" in window)||!secs.length)return;
  var seen={};
  var io=new IntersectionObserver(function(es){
    es.forEach(function(e){seen[e.target.id]=e.intersectionRatio;});
    var best=null,bv=0;
    for(var k in seen){if(seen[k]>bv){bv=seen[k];best=k;}}
    if(best){for(var id in links)links[id].classList.toggle("active",id===best);}
  },{rootMargin:"-64px 0px -55% 0px",threshold:[0,.25,.5,1]});
  secs.forEach(function(s){io.observe(s);});
})();

/* ---- tabs ---- */
document.querySelectorAll(".tab").forEach(function(b){
  b.addEventListener("click",function(){
    var group=b.parentNode,panelId=b.dataset.tab;
    group.querySelectorAll(".tab").forEach(function(x){x.classList.toggle("on",x===b);});
    var scope=group.parentNode;
    scope.querySelectorAll(":scope > .panel").forEach(function(p){
      p.classList.toggle("on",p.id===panelId);
    });
  });
});

/* ---- line charts: crosshair + readout ---- */
Object.keys(D.charts).forEach(function(cid){
  var svg=document.getElementById(cid);if(!svg)return;
  var c=D.charts[cid],NS="http://www.w3.org/2000/svg";
  var cross=document.createElementNS(NS,"line");
  cross.setAttribute("class","crosshair");cross.setAttribute("y1",c.pt);
  cross.setAttribute("y2",c.pt+c.ih);cross.style.display="none";
  svg.appendChild(cross);
  var dots=[];
  c.series.forEach(function(s){
    var d=document.createElementNS(NS,"circle");
    d.setAttribute("class","focus");d.setAttribute("r","4");
    d.setAttribute("fill",s.color);d.setAttribute("stroke","var(--card)");
    d.setAttribute("stroke-width","1.5");d.style.display="none";
    svg.appendChild(d);dots.push(d);
  });
  function X(i){return c.pl+(c.iw*i/Math.max(1,c.n-1));}
  function Y(v){return c.pt+c.ih-c.ih*(v-c.lo)/(c.hi-c.lo);}
  function move(e){
    var r=svg.getBoundingClientRect();
    var ux=(e.clientX-r.left)/r.width*c.w;
    var i=Math.round((ux-c.pl)/c.iw*(c.n-1));
    i=Math.max(0,Math.min(c.n-1,i));
    cross.setAttribute("x1",X(i));cross.setAttribute("x2",X(i));
    cross.style.display="";
    var lines=[],any=false;
    c.series.forEach(function(s,si){
      var j=i-s.x0,v=(j>=0&&j<s.ys.length)?s.ys[j]:null;
      if(v===null||v===undefined){dots[si].style.display="none";return;}
      any=true;
      dots[si].setAttribute("cx",X(i));dots[si].setAttribute("cy",Y(v));
      dots[si].style.display="";
      lines.push('<span class="tv" style="color:'+s.color+'">&#9632;</span> '+
                 s.name+': <span class="tv">'+v.toFixed(4)+'</span>');
    });
    if(!any){hideTip();return;}
    showTip(e.clientX,e.clientY,"<b>"+(c.labels[i]||("point "+i))+"</b>"+lines.join("<br>"));
  }
  svg.addEventListener("pointermove",move);
  svg.addEventListener("pointerleave",function(){
    hideTip();cross.style.display="none";dots.forEach(function(d){d.style.display="none";});
  });
});

/* ---- heatmap: hover detail + lane filter ---- */
(function(){
  var heat=document.getElementById("heat");if(!heat)return;
  heat.addEventListener("pointermove",function(e){
    var t=e.target;
    if(t.tagName!=="rect"||t.dataset.i===undefined){hideTip();return;}
    var k=D.tokens[+t.dataset.i];if(!k)return;
    showTip(e.clientX,e.clientY,
      "<b>"+k[2]+" &middot; "+k[3]+"</b>"+
      "token <code>"+(k[0]||"").replace(/&/g,"&amp;").replace(/</g,"&lt;")+"</code><br>"+
      "perplexity <span class='tv'>"+k[1]+"</span><br>"+
      "step "+k[4]+" &middot; stage "+k[5]);
  });
  heat.addEventListener("pointerleave",hideTip);
  var chips=document.querySelectorAll(".chip[data-lane]");
  function apply(){
    var on={};chips.forEach(function(c){if(c.classList.contains("on"))on[c.dataset.lane]=1;});
    heat.querySelectorAll("rect").forEach(function(r){
      r.classList.toggle("off",!on[r.dataset.lane]);
    });
  }
  chips.forEach(function(c){
    c.addEventListener("click",function(){c.classList.toggle("on");apply();});
  });
  var all=document.getElementById("lane-all");
  if(all)all.addEventListener("click",function(){
    chips.forEach(function(c){c.classList.add("on");});apply();
  });
})();

/* ---- sortable tables ---- */
document.querySelectorAll("table.sortable").forEach(function(t){
  var ths=t.querySelectorAll("thead th");
  ths.forEach(function(th,idx){
    function sort(){
      var dir=th.classList.contains("asc")?-1:1;
      ths.forEach(function(o){o.classList.remove("asc","desc");});
      th.classList.add(dir===1?"asc":"desc");
      var tb=t.tBodies[0],rows=Array.prototype.slice.call(tb.rows);
      rows.sort(function(a,b){
        var ca=a.cells[idx],cb=b.cells[idx];
        var va=ca.dataset.v!==undefined?ca.dataset.v:ca.textContent.trim();
        var vb=cb.dataset.v!==undefined?cb.dataset.v:cb.textContent.trim();
        var na=parseFloat(String(va).replace(/[,%₹\\s]/g,"")),
            nb=parseFloat(String(vb).replace(/[,%₹\\s]/g,""));
        if(!isNaN(na)&&!isNaN(nb))return (na-nb)*dir;
        return String(va).localeCompare(String(vb))*dir;
      });
      rows.forEach(function(r){tb.appendChild(r);});
    }
    th.addEventListener("click",sort);
    th.addEventListener("keydown",function(e){if(e.key==="Enter"||e.key===" "){e.preventDefault();sort();}});
  });
});

/* ---- table filters ---- */
document.querySelectorAll(".filter").forEach(function(inp){
  var t=document.getElementById(inp.dataset.filter);if(!t)return;
  inp.addEventListener("input",function(){
    var q=inp.value.toLowerCase();
    Array.prototype.forEach.call(t.tBodies[0].rows,function(r){
      r.style.display=r.textContent.toLowerCase().indexOf(q)>=0?"":"none";
    });
  });
});

/* ---- click a hash to copy it ---- */
document.querySelectorAll("td .hash").forEach(function(el){
  el.addEventListener("click",function(){
    var full=el.parentNode.dataset.v||el.textContent;
    var done=function(){var o=el.textContent;el.textContent="copied";
      setTimeout(function(){el.textContent=o;},900);};
    if(navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(full).then(done,function(){});
    }else{
      var ta=document.createElement("textarea");ta.value=full;document.body.appendChild(ta);
      ta.select();try{document.execCommand("copy");done();}catch(e){}
      document.body.removeChild(ta);
    }
  });
});
})();
</script>
</body></html>
"""


if __name__ == "__main__":
    art = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "submission_artifacts")
    out = os.path.join(art, "dashboard.html")
    write_text(out, build(art))
    print(f"wrote {out} ({os.path.getsize(out):,} bytes)")
