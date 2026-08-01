# -*- coding: utf-8 -*-
"""
charts.py - inline SVG figures built from the same audit data as the plan.

Every figure is generated from mixture_results.json, so a number cannot appear
in a chart that disagrees with the table beside it.

Form choices follow the data's job rather than habit:
  - Lane shares and epochs are MAGNITUDE -> one hue, ranked bars. Nine lanes
    would need nine categorical hues, which no validated palette supports; a
    single hue sorted by value carries magnitude better anyway.
  - Verdicts are STATE -> the reserved status palette, always chip + glyph +
    label so the status never rides on hue alone.
  - Gate growth is a single headline -> a stat pair, not a chart.
"""

import html

# Marks: one hue, validated >= 3:1 on both surfaces (#F7F8FA / #12141A).
BAR = "var(--mark)"
BAR_ALT = "var(--mark-alt)"
STATUS_CLASS = {"SUPPLY-OK": "ok", "REPEAT": "warn", "GENERATE": "serious"}


def _esc(s):
    return html.escape(str(s), quote=True)


def _fmt_tokens(x):
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{x/1e9:.0f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    return f"{x:,.0f}"


def ranked_bars(rows, value_key, label_key, fmt, title, subtitle="",
                status_key=None, ref=None, ref_label="", axis_max=None):
    """Horizontal ranked bars. rows: list of dicts. One hue; magnitude by length.

    Direct labels on every bar, so the chart is readable without hover and
    without relying on colour for identity.

    `axis_max` clips the axis when one value would flatten every other bar into
    an unreadable sliver - the agentic lane at 22.7 epochs against a median near
    1.7 leaves seven lanes under 25px. A clipped bar gets a visible break marker
    and still carries its true value as a direct label, so nothing is hidden;
    the alternative (silently squashing the rest) hides more."""
    rows = sorted(rows, key=lambda r: -r[value_key])
    true_max = max([r[value_key] for r in rows] + ([ref] if ref else [0])) or 1
    vmax = axis_max or true_max
    rowh, gap, padl, padr = 26, 6, 132, 56
    h = len(rows) * (rowh + gap) + 8
    w = 640

    parts = [
        f'<figure class="fig">',
        f'<figcaption class="fig-cap"><span class="fig-title">{_esc(title)}</span>',
    ]
    if subtitle:
        parts.append(f'<span class="fig-sub">{_esc(subtitle)}</span>')
    parts.append("</figcaption>")
    parts.append(
        f'<div class="fig-body"><svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="{_esc(title)}" preserveAspectRatio="xMinYMin meet">')

    plot_w = w - padl - padr
    if ref:
        x = padl + plot_w * (ref / vmax)
        parts.append(
            f'<line class="ref" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h-14}"/>'
            f'<text class="ref-t" x="{x+4:.1f}" y="{h-4}">{_esc(ref_label)}</text>')

    for i, r in enumerate(rows):
        y = i * (rowh + gap)
        val = r[value_key]
        clipped = val > vmax * 1.001
        bw = max(2.0, plot_w * (min(val, vmax) / vmax))
        cls = ""
        if status_key and r.get(status_key) in STATUS_CLASS:
            cls = f' bar-{STATUS_CLASS[r[status_key]]}'
        parts.append(
            f'<text class="bar-label" x="{padl-10}" y="{y+rowh*0.68:.1f}" '
            f'text-anchor="end">{_esc(r[label_key])}</text>'
            # 4px rounded data-end, anchored to the baseline
            f'<rect class="bar{cls}" x="{padl}" y="{y+3}" width="{bw:.1f}" '
            f'height="{rowh-6}" rx="4"><title>{_esc(r[label_key])}: '
            f'{_esc(fmt(val))}</title></rect>')
        if clipped:
            # zig-zag break: says "this bar runs off the axis" without pretending
            bx = padl + bw
            parts.append(
                f'<path class="clip-mark" d="M{bx-7:.1f},{y+3} l7,{(rowh-6)/2:.1f} '
                f'l-7,{(rowh-6)/2:.1f}" />')
        # A clipped bar's label sits at the right edge of the plot, so it must
        # stay short or it runs past the viewBox. The break marker and the
        # caption carry the "off scale" meaning instead.
        parts.append(
            f'<text class="bar-val" x="{padl+bw+8:.1f}" y="{y+rowh*0.68:.1f}">'
            f"{_esc(fmt(val))}</text>")

    parts.append("</svg></div></figure>")
    return "\n".join(parts)


def phase_ladder(lanes, phases):
    """Small multiples: one sparkline per lane across the four phases.

    A 4x9 stacked bar would need nine categorical hues. Small multiples show
    what actually matters here - the SHAPE of each lane's trajectory - and need
    only one hue."""
    cells = []
    for ln in sorted(lanes, key=lambda l: -l["share"]):
        vals = ln["phases"]
        vmax = max(vals) or 1
        w, h = 132, 40
        bw = w / len(vals)
        bars = []
        for j, v in enumerate(vals):
            bh = (v / vmax) * (h - 12)
            bars.append(
                f'<rect x="{j*bw+2:.1f}" y="{h-bh-10:.1f}" width="{bw-4:.1f}" '
                f'height="{max(1.0,bh):.1f}" rx="2" class="spark"/>'
                f'<text class="spark-x" x="{j*bw+bw/2:.1f}" y="{h-1}" '
                f'text-anchor="middle">{_esc(phases[j])}</text>')
        trend = "rises" if vals[-1] > vals[0] else ("falls" if vals[-1] < vals[0] else "flat")
        cells.append(
            f'<div class="sm-cell"><div class="sm-head">'
            f'<span class="sm-name">{_esc(ln["label"])}</span>'
            f'<span class="sm-val">{ln["share"]:.2f}%</span></div>'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="'
            f'{_esc(ln["label"])} across phases: {trend} from {vals[0]:.0f}% to '
            f'{vals[-1]:.0f}%">{"".join(bars)}</svg></div>')
    return (f'<figure class="fig"><figcaption class="fig-cap">'
            f'<span class="fig-title">Curriculum shape, lane by lane</span>'
            f'<span class="fig-sub">Share of each phase’s tokens. '
            f'Scarce lanes are held small early and concentrated in the anneal.'
            f'</span></figcaption>'
            f'<div class="smallmult">{"".join(cells)}</div></figure>')


def funnel(stages, title, subtitle=""):
    """Cleaning funnel. One hue; survival read by length, loss labelled."""
    vmax = max(s["docs"] for s in stages) or 1
    rowh, gap, padl, padr = 24, 5, 120, 96
    h = len(stages) * (rowh + gap) + 4
    w = 640
    plot_w = w - padl - padr
    parts = [f'<figure class="fig"><figcaption class="fig-cap">'
             f'<span class="fig-title">{_esc(title)}</span>']
    if subtitle:
        parts.append(f'<span class="fig-sub">{_esc(subtitle)}</span>')
    parts.append(f'</figcaption><div class="fig-body">'
                 f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(title)}">')
    prev = None
    for i, s in enumerate(stages):
        y = i * (rowh + gap)
        bw = max(2.0, plot_w * (s["docs"] / vmax))
        lost = "" if prev is None else f"−{prev - s['docs']:,}"
        parts.append(
            f'<text class="bar-label" x="{padl-10}" y="{y+rowh*0.68:.1f}" '
            f'text-anchor="end">{_esc(s["name"])}</text>'
            f'<rect class="bar" x="{padl}" y="{y+3}" width="{bw:.1f}" '
            f'height="{rowh-6}" rx="4"><title>{_esc(s["name"])}: '
            f'{s["docs"]:,} docs</title></rect>'
            f'<text class="bar-val" x="{padl+bw+8:.1f}" y="{y+rowh*0.68:.1f}">'
            f'{s["docs"]:,}<tspan class="loss"> {lost}</tspan></text>')
        prev = s["docs"]
    parts.append("</svg></div></figure>")
    return "\n".join(parts)


def stat_row(stats):
    """Summary before detail. Not a chart - a headline number is the right form
    for a single value."""
    cells = []
    for s in stats:
        delta = ""
        if s.get("delta"):
            cells_cls = "up" if s["delta"].startswith("+") else "flat"
            delta = f'<span class="stat-delta {cells_cls}">{_esc(s["delta"])}</span>'
        cells.append(
            f'<div class="stat"><div class="stat-k">{_esc(s["k"])}</div>'
            f'<div class="stat-v">{_esc(s["v"])}{delta}</div>'
            f'<div class="stat-n">{_esc(s.get("n",""))}</div></div>')
    return f'<div class="statrow">{"".join(cells)}</div>'
