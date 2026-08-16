"""A small, dependency-free SVG chart toolkit shared by both tracks'
dashboards. Follows the project's dataviz conventions: 2px lines, >=8px
ringed markers, a 2px surface gap between bars, a legend for >=2 series,
hover tooltips (crosshair on lines, per-bar on bars), and a light/dark
palette driven entirely by CSS custom properties so the same markup works
standalone or published as a Claude Artifact.
"""
from __future__ import annotations

import json

PALETTE_CSS = """
:root {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --good:           #006300;
  --series-1:       #2a78d6;
  --series-2:       #eb6834;
  --series-3:       #1baf7a;
  --series-4:       #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --good:           #0ca30c;
    --series-1:       #3987e5;
    --series-2:       #d95926;
    --series-3:       #199e70;
    --series-4:       #c98500;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --page:           #0d0d0d;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #898781;
  --gridline:       #2c2c2a;
  --baseline:       #383835;
  --border:         rgba(255,255,255,0.10);
  --good:           #0ca30c;
  --series-1:       #3987e5;
  --series-2:       #d95926;
  --series-3:       #199e70;
  --series-4:       #c98500;
}
body { background: var(--page); color: var(--text-primary); margin: 0; }
"""

BASE_CSS = """
.viz-root { background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px 22px; margin: 0 0 20px 0; }
h1 { font-size: 1.6rem; margin: 0 0 4px 0; }
h2 { font-size: 1.1rem; margin: 0 0 12px 0; }
.subtitle { color: var(--text-secondary); margin: 0 0 18px 0; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 22px 0; }
.badge { font-size: 0.78rem; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border);
  color: var(--text-secondary); background: var(--page); }
table.evtable { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
table.evtable th, table.evtable td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--gridline); }
table.evtable th { color: var(--text-muted); font-weight: 600; font-size: 0.76rem; text-transform: uppercase;
  letter-spacing: 0.03em; }
.pass { color: var(--good); font-weight: 600; }
.fail { color: var(--series-2); font-weight: 600; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 6px 0 14px 0; font-size: 0.82rem; color: var(--text-secondary); }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-swatch { width: 14px; height: 2px; border-radius: 1px; }
.legend-swatch.dot { width: 8px; height: 8px; border-radius: 50%; }
.chart-wrap { position: relative; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 10px; font-size: 0.8rem; box-shadow: 0 4px 14px rgba(0,0,0,0.15);
  opacity: 0; transition: opacity 0.08s; z-index: 5; white-space: nowrap; }
.tooltip .t-value { font-weight: 700; color: var(--text-primary); }
.tooltip .t-label { color: var(--text-secondary); }
.crosshair { stroke: var(--baseline); stroke-width: 1; pointer-events: none; opacity: 0; }
.axis-label { fill: var(--text-muted); font-size: 11px; }
.grid-line { stroke: var(--gridline); stroke-width: 1; }
.bar-rect { cursor: pointer; }
.bar-rect:hover { filter: brightness(1.08); }
.limitations { border-left: 3px solid var(--series-2); padding: 4px 16px; background: var(--page); border-radius: 0 8px 8px 0; }
.limitations li { margin: 6px 0; }
.figure img { max-width: 100%; border-radius: 6px; border: 1px solid var(--border); }
code, pre { font-family: ui-monospace, "SF Mono", Consolas, monospace; }
pre { background: var(--page); padding: 12px 14px; border-radius: 8px; overflow-x: auto; font-size: 0.82rem;
  border: 1px solid var(--border); }
.page-footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
  font-size: 0.82rem; color: var(--text-muted); }
.page-footer a { color: var(--text-secondary); }
"""

# Shared by build_dashboard.py (via page_shell) and build_index.py. Defined once
# here because both generators emit it and a duplicated literal would drift.
REPO_TREE_URL = "https://github.com/rahulni/Indic_LLM/tree/main/7_embed_research"
REPO_EVIDENCE_URL = (
    "https://github.com/rahulni/Indic_LLM/blob/main/7_embed_research"
    "/submission_artifacts/evidence.json"
)

CHART_JS = """
function vizTooltip(root) {
  let tip = root.querySelector('.tooltip');
  if (!tip) { tip = document.createElement('div'); tip.className = 'tooltip'; root.appendChild(tip); }
  return {
    show(x, y, html) { tip.innerHTML = html; tip.style.left = (x + 14) + 'px'; tip.style.top = (y - 10) + 'px'; tip.style.opacity = 1; },
    hide() { tip.style.opacity = 0; }
  };
}

function vizLineChart(el, cfg) {
  // cfg: {series:[{name,color,points:[{x,y,label}]}], width,height,margin,xTicks,yTicks,yFormat,xLabel,yLabel,yMax}
  const W = cfg.width || 640, H = cfg.height || 320;
  const m = cfg.margin || {top: 16, right: 20, bottom: 34, left: 46};
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const allX = cfg.series.flatMap(s => s.points.map(p => p.x));
  const allY = cfg.series.flatMap(s => s.points.map(p => p.y));
  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMax = cfg.yMax !== undefined ? cfg.yMax : Math.max(...allY) * 1.08;
  const yMin = cfg.yMin !== undefined ? cfg.yMin : Math.min(0, Math.min(...allY));
  const sx = x => m.left + (xMax === xMin ? iw / 2 : (x - xMin) / (xMax - xMin) * iw);
  const sy = y => m.top + ih - (yMax === yMin ? 0 : (y - yMin) / (yMax - yMin) * ih);

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">`;
  const yTicks = cfg.yTicks || 4;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + (yMax - yMin) * i / yTicks;
    const y = sy(v);
    svg += `<line class="grid-line" x1="${m.left}" x2="${W - m.right}" y1="${y}" y2="${y}"/>`;
    svg += `<text class="axis-label" x="${m.left - 8}" y="${y + 4}" text-anchor="end">${cfg.yFormat ? cfg.yFormat(v) : v.toFixed(2)}</text>`;
  }
  const xTickVals = cfg.xTicks || allX.filter((v, i, a) => a.indexOf(v) === i).sort((a,b)=>a-b);
  xTickVals.forEach(v => {
    svg += `<text class="axis-label" x="${sx(v)}" y="${H - m.bottom + 18}" text-anchor="middle">${v}</text>`;
  });
  svg += `<line class="grid-line" x1="${m.left}" x2="${W - m.right}" y1="${H - m.bottom}" y2="${H - m.bottom}" stroke="var(--baseline)"/>`;

  if (cfg.refLineX !== undefined) {
    svg += `<line x1="${sx(cfg.refLineX)}" x2="${sx(cfg.refLineX)}" y1="${m.top}" y2="${H-m.bottom}" stroke="var(--text-muted)" stroke-dasharray="4,3" stroke-width="1"/>`;
  }

  cfg.series.forEach((s, si) => {
    const path = s.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x)} ${sy(p.y)}`).join(' ');
    const dash = s.dashed ? ' stroke-dasharray="5,4" opacity="0.75"' : '';
    svg += `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"${dash}/>`;
    s.points.forEach((p, pi) => {
      svg += `<circle class="pt" data-si="${si}" data-pi="${pi}" cx="${sx(p.x)}" cy="${sy(p.y)}" r="4" fill="${s.color}" stroke="var(--surface-1)" stroke-width="2"/>`;
    });
  });

  if (cfg.xLabel) svg += `<text class="axis-label" x="${m.left + iw/2}" y="${H - 2}" text-anchor="middle">${cfg.xLabel}</text>`;
  svg += `</svg>`;

  el.innerHTML = `<div class="chart-wrap">${svg}</div>`;
  const wrap = el.querySelector('.chart-wrap');
  const tooltip = vizTooltip(wrap);
  wrap.querySelectorAll('circle.pt').forEach(c => {
    c.addEventListener('pointerenter', (e) => {
      const si = +c.dataset.si, pi = +c.dataset.pi;
      const s = cfg.series[si], p = s.points[pi];
      const rect = wrap.getBoundingClientRect();
      tooltip.show(e.clientX - rect.left, e.clientY - rect.top,
        `<div class="t-label">${s.name}</div><div class="t-value">${p.label !== undefined ? p.label : p.y}</div>`);
      c.setAttribute('r', 6);
    });
    c.addEventListener('pointerleave', () => { tooltip.hide(); c.setAttribute('r', 4); });
  });
}

function vizBarChart(el, cfg) {
  // cfg: {bars:[{label,value,color,valueLabel}], width,height,margin,yFormat,yMax,yLabel}
  const W = cfg.width || 640, H = cfg.height || 300;
  const m = cfg.margin || {top: 16, right: 20, bottom: 44, left: 54};
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const yMax = cfg.yMax !== undefined ? cfg.yMax : Math.max(...cfg.bars.map(b => b.value)) * 1.15;
  const n = cfg.bars.length;
  const slot = iw / n;
  const barW = Math.min(48, slot * 0.55);
  const sy = v => m.top + ih - (v / yMax) * ih;

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">`;
  const yTicks = cfg.yTicks || 4;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMax * i / yTicks;
    const y = sy(v);
    svg += `<line class="grid-line" x1="${m.left}" x2="${W - m.right}" y1="${y}" y2="${y}"/>`;
    svg += `<text class="axis-label" x="${m.left - 8}" y="${y + 4}" text-anchor="end">${cfg.yFormat ? cfg.yFormat(v) : v}</text>`;
  }
  svg += `<line x1="${m.left}" x2="${W - m.right}" y1="${H - m.bottom}" y2="${H - m.bottom}" stroke="var(--baseline)" stroke-width="1"/>`;

  cfg.bars.forEach((b, i) => {
    const cx = m.left + slot * i + slot / 2;
    const x = cx - barW / 2;
    const y = sy(b.value);
    const h = (H - m.bottom) - y;
    svg += `<rect class="bar-rect" data-i="${i}" x="${x}" y="${y}" width="${barW}" height="${Math.max(h,1)}" rx="4" fill="${b.color}"/>`;
    svg += `<text class="axis-label" x="${cx}" y="${H - m.bottom + 18}" text-anchor="middle">${b.label}</text>`;
  });
  if (cfg.yLabel) svg += `<text class="axis-label" x="${m.left}" y="${m.top - 4}" text-anchor="start">${cfg.yLabel}</text>`;
  svg += `</svg>`;

  el.innerHTML = `<div class="chart-wrap">${svg}</div>`;
  const wrap = el.querySelector('.chart-wrap');
  const tooltip = vizTooltip(wrap);
  wrap.querySelectorAll('rect.bar-rect').forEach(r => {
    r.addEventListener('pointerenter', (e) => {
      const b = cfg.bars[+r.dataset.i];
      const rect = wrap.getBoundingClientRect();
      tooltip.show(e.clientX - rect.left, e.clientY - rect.top,
        `<div class="t-label">${b.label}</div><div class="t-value">${b.valueLabel !== undefined ? b.valueLabel : b.value}</div>`);
    });
    r.addEventListener('pointerleave', () => tooltip.hide());
  });
}

function vizGroupedBarChart(el, cfg) {
  // cfg: {groups:[{label, bars:[{name,value,color,valueLabel}]}], seriesNames:[...], width,height,yFormat,yMax,yLabel}
  const W = cfg.width || 640, H = cfg.height || 320;
  const m = cfg.margin || {top: 16, right: 20, bottom: 44, left: 60};
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const allVals = cfg.groups.flatMap(g => g.bars.map(b => b.value));
  const yMax = cfg.yMax !== undefined ? cfg.yMax : Math.max(...allVals) * 1.15;
  const nGroups = cfg.groups.length;
  const groupSlot = iw / nGroups;
  const nBars = cfg.groups[0].bars.length;
  const barW = Math.min(28, (groupSlot * 0.7) / nBars);
  const sy = v => m.top + ih - (yMax === 0 ? 0 : (v / yMax) * ih);

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">`;
  const yTicks = cfg.yTicks || 4;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMax * i / yTicks;
    const y = sy(v);
    svg += `<line class="grid-line" x1="${m.left}" x2="${W - m.right}" y1="${y}" y2="${y}"/>`;
    svg += `<text class="axis-label" x="${m.left - 8}" y="${y + 4}" text-anchor="end">${cfg.yFormat ? cfg.yFormat(v) : v}</text>`;
  }
  svg += `<line x1="${m.left}" x2="${W - m.right}" y1="${H - m.bottom}" y2="${H - m.bottom}" stroke="var(--baseline)" stroke-width="1"/>`;

  cfg.groups.forEach((g, gi) => {
    const groupStart = m.left + groupSlot * gi + (groupSlot - barW * nBars) / 2;
    g.bars.forEach((b, bi) => {
      const x = groupStart + bi * barW;
      const y = sy(b.value);
      const h = (H - m.bottom) - y;
      svg += `<rect class="bar-rect" data-g="${gi}" data-b="${bi}" x="${x}" y="${y}" width="${barW - 2}" height="${Math.max(h,1)}" rx="3" fill="${b.color}"/>`;
    });
    svg += `<text class="axis-label" x="${m.left + groupSlot * gi + groupSlot/2}" y="${H - m.bottom + 18}" text-anchor="middle">${g.label}</text>`;
  });
  if (cfg.yLabel) svg += `<text class="axis-label" x="${m.left}" y="${m.top - 4}" text-anchor="start">${cfg.yLabel}</text>`;
  svg += `</svg>`;

  el.innerHTML = `<div class="chart-wrap">${svg}</div>`;
  const wrap = el.querySelector('.chart-wrap');
  const tooltip = vizTooltip(wrap);
  wrap.querySelectorAll('rect.bar-rect').forEach(r => {
    r.addEventListener('pointerenter', (e) => {
      const g = cfg.groups[+r.dataset.g], b = g.bars[+r.dataset.b];
      const rect = wrap.getBoundingClientRect();
      tooltip.show(e.clientX - rect.left, e.clientY - rect.top,
        `<div class="t-label">${g.label} — ${b.name}</div><div class="t-value">${b.valueLabel !== undefined ? b.valueLabel : b.value}</div>`);
    });
    r.addEventListener('pointerleave', () => tooltip.hide());
  });
}
"""


def legend_html(items: list[tuple[str, str]], dot: bool = False) -> str:
    cls = "legend-swatch dot" if dot else "legend-swatch"
    spans = "".join(
        f'<div class="legend-item"><span class="{cls}" style="background:{color}"></span>{name}</div>'
        for name, color in items
    )
    return f'<div class="legend">{spans}</div>'


def evidence_table_html(evidence_rows: list[dict]) -> str:
    rows = ""
    for r in evidence_rows:
        cls = "pass" if r["result"] == "PASS" else "fail"
        rows += (
            f"<tr><td>{r['requirement']}</td><td class='{cls}'>{r['result']}</td>"
            f"<td><code>{r['file']}</code></td><td>{r['compared']}</td></tr>"
        )
    return (
        "<table class='evtable'><thead><tr><th>Requirement</th><th>Result</th>"
        f"<th>File</th><th>Compared</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def page_shell(title: str, favicon_note: str, body: str, script_data: dict) -> str:
    data_json = json.dumps(script_data)
    # DATA and the chart-drawing functions must be defined BEFORE the body's
    # own trailing <script> (which calls them) runs -- scripts execute in
    # document order, so this block goes first, ahead of the body markup.
    return f"""<title>{title}</title>
<style>
{PALETTE_CSS}
{BASE_CSS}
.page {{ max-width: 900px; margin: 0 auto; padding: 28px 20px 60px 20px; }}
</style>
<script>
const DATA = {data_json};
{CHART_JS}
</script>
<div class="viz-root page">
{body}
<div class="page-footer">
<a href="../../submission_artifacts/index.html">&larr; All tracks</a> &middot;
<a href="{REPO_TREE_URL}">Code + README on GitHub</a>
</div>
</div>
"""
