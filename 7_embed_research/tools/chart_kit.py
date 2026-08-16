"""A small, dependency-free SVG chart toolkit shared by both tracks'
dashboards. Follows the project's dataviz conventions: 2px lines, >=8px
ringed markers, a 2px surface gap between bars, a legend for >=2 series,
hover tooltips (crosshair on lines, per-bar on bars), and a light/dark
palette driven entirely by CSS custom properties so the same markup works
standalone or published as a Claude Artifact.
"""
from __future__ import annotations

import json

# Dark is the DEFAULT, unconditionally -- bare :root carries the black palette
# and the OS preference is deliberately not consulted, so the page is black on
# load regardless of the visitor's system setting. Light is reachable only by
# the toggle, which stamps [data-theme="light"] on <html>. Both series sets are
# validated against their own surface (see dataviz six checks); the light set's
# green and amber sit below 3:1 contrast, which is why every chart also ships a
# legend and the evidence tables repeat the numbers in text.
PALETTE_CSS = """
:root {
  color-scheme: dark;
  --surface-1:      #141413;
  --surface-2:      #1c1c1a;
  --page:           #0a0a0a;
  --text-primary:   #f5f5f3;
  --text-secondary: #b8b7b0;
  --text-muted:     #8a8880;
  --gridline:       #262624;
  --baseline:       #3a3a37;
  --border:         rgba(255,255,255,0.09);
  --border-strong:  rgba(255,255,255,0.16);
  --good:           #3fbf5f;
  --series-1:       #3987e5;
  --series-2:       #d95926;
  --series-3:       #199e70;
  --series-4:       #c98500;
  --shadow:         0 1px 2px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.28);
}
:root[data-theme="light"] {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --surface-2:      #f4f4f1;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --border-strong:  rgba(11,11,11,0.18);
  --good:           #006300;
  --series-1:       #2a78d6;
  --series-2:       #eb6834;
  --series-3:       #1baf7a;
  --series-4:       #eda100;
  --shadow:         0 1px 2px rgba(11,11,11,0.05), 0 8px 24px rgba(11,11,11,0.06);
}
body { background: var(--page); color: var(--text-primary); margin: 0; }
"""

BASE_CSS = """
.viz-root { background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.62; -webkit-font-smoothing: antialiased; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 14px;
  padding: 26px 28px; margin: 0 0 22px 0; box-shadow: var(--shadow); }
h1 { font-size: 2rem; line-height: 1.2; letter-spacing: -0.02em; margin: 0 0 8px 0; }
/* Headings keep their hand-written "1." / "2." numbers: the prose cross-refers
   to them ("proved in section 1"), so a CSS counter could silently drift out of
   step with the text it is being cited by. */
h2 { font-size: 1.18rem; line-height: 1.3; letter-spacing: -0.01em; margin: 0 0 14px 0; }
h3 { font-size: 0.96rem; margin: 22px 0 8px 0; color: var(--text-primary); }
.subtitle { color: var(--text-secondary); margin: 0 0 18px 0; max-width: 72ch; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 26px 0; }
.badge { font-size: 0.78rem; padding: 5px 11px; border-radius: 999px; border: 1px solid var(--border);
  color: var(--text-secondary); background: var(--surface-1); }
/* Headline numbers. The dataviz form heuristic: a single value with no
   comparison across categories is a stat tile, not a chart. */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px; margin: 0 0 22px 0; }
.stat-tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 14px;
  padding: 20px 22px; box-shadow: var(--shadow); }
.stat-value { font-size: 1.85rem; line-height: 1.1; font-weight: 700; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; }
.stat-value.compare { font-size: 1.32rem; line-height: 1.25; }
.stat-vs { font-size: 0.78em; font-weight: 500; color: var(--text-muted);
  margin: 0 0.42em; letter-spacing: 0; }
.stat-label { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--text-muted); font-weight: 600; margin-bottom: 9px; }
.stat-note { font-size: 0.82rem; color: var(--text-secondary); margin-top: 7px; }
.stat-tile .accent-1 { color: var(--series-1); }
.stat-tile .accent-2 { color: var(--series-2); }
.stat-tile .accent-good { color: var(--good); }
table.evtable { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
table.evtable th, table.evtable td { text-align: left; padding: 9px 11px; border-bottom: 1px solid var(--gridline); }
table.evtable td { font-variant-numeric: tabular-nums; }
table.evtable tbody tr:last-child td { border-bottom: none; }
table.evtable tbody tr:hover td { background: var(--surface-2); }
table.evtable th { color: var(--text-muted); font-weight: 600; font-size: 0.76rem; text-transform: uppercase;
  letter-spacing: 0.03em; }
.table-scroll { overflow-x: auto; }
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
.limitations { border-left: 3px solid var(--series-2); padding: 4px 18px; background: var(--surface-2);
  border-radius: 0 10px 10px 0; }
.limitations li { margin: 8px 0; }
.figure { margin: 18px 0; }
.figure img { max-width: 100%; border-radius: 10px; border: 1px solid var(--border); display: block; }
/* A PNG's ink is baked in, so each figure ships twice and CSS picks the copy
   matching the active theme. Dark is default, hence .fig-light starts hidden. */
.fig-light { display: none; }
.fig-dark { display: block; }
:root[data-theme="light"] .fig-light { display: block; }
:root[data-theme="light"] .fig-dark { display: none; }
code, pre { font-family: ui-monospace, "SF Mono", Consolas, monospace; }
code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }
pre { background: var(--surface-2); padding: 14px 16px; border-radius: 10px; overflow-x: auto; font-size: 0.82rem;
  border: 1px solid var(--border); }
pre code { background: none; padding: 0; }
a { color: var(--series-1); text-decoration-color: var(--border-strong); text-underline-offset: 2px; }
a:hover { text-decoration-color: currentColor; }
.theme-toggle { position: fixed; top: 16px; right: 16px; z-index: 20;
  width: 38px; height: 38px; border-radius: 10px; cursor: pointer;
  background: var(--surface-1); border: 1px solid var(--border-strong); color: var(--text-secondary);
  font-size: 15px; line-height: 1; display: flex; align-items: center; justify-content: center;
  box-shadow: var(--shadow); }
.theme-toggle:hover { color: var(--text-primary); border-color: var(--text-muted); }
.theme-toggle .icon-dark { display: inline; }
.theme-toggle .icon-light { display: none; }
:root[data-theme="light"] .theme-toggle .icon-dark { display: none; }
:root[data-theme="light"] .theme-toggle .icon-light { display: inline; }
.page-footer { margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--border);
  font-size: 0.82rem; color: var(--text-muted); }
.page-footer a { color: var(--text-secondary); }
@media (max-width: 620px) {
  .viz-root { font-size: 14px; }
  .card { padding: 20px 17px; border-radius: 12px; }
  h1 { font-size: 1.55rem; }
  .theme-toggle { top: 10px; right: 10px; }
}
"""

# Shared by build_dashboard.py (via page_shell) and build_index.py. Defined once
# here because both generators emit it and a duplicated literal would drift.
REPO_TREE_URL = "https://github.com/rahulni/Indic_LLM/tree/main/7_embed_research"
REPO_EVIDENCE_URL = (
    "https://github.com/rahulni/Indic_LLM/blob/main/7_embed_research"
    "/submission_artifacts/evidence.json"
)

# Applied before first paint so a light-mode visitor never sees a black flash.
# Absent a stored choice this does nothing, which leaves the CSS default (dark).
THEME_BOOT_JS = """
(function () {
  try {
    var t = localStorage.getItem('kv2-theme');
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  } catch (e) {}
})();
"""

THEME_TOGGLE_HTML = """
<button class="theme-toggle" id="theme-toggle" type="button"
  aria-label="Switch between dark and light theme" title="Switch theme">
<span class="icon-dark">&#9788;</span><span class="icon-light">&#9789;</span>
</button>
"""

THEME_TOGGLE_JS = """
document.getElementById('theme-toggle').addEventListener('click', function () {
  var root = document.documentElement;
  var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  root.setAttribute('data-theme', next);
  try { localStorage.setItem('kv2-theme', next); } catch (e) {}
});
"""


def stat_tiles_html(tiles: list[dict]) -> str:
    """A row of headline numbers. Each tile is {label, note?, accent?} plus
    either `value` (one number) or `values` (a pair, rendered as "a vs b").

    The helper owns the "vs" markup so a two-value tile also gets the smaller
    type size that keeps both numbers on one line -- leaving that to the caller
    meant every comparison had to remember to shrink itself, and one that forgot
    wrapped raggedly mid-number.

    Used instead of a chart wherever the figure is a single number with no
    across-category comparison to make -- plotting one value as a lone bar is
    the classic form error.
    """
    cells = ""
    for t in tiles:
        accent = f" {t['accent']}" if t.get("accent") else ""
        note = f'<div class="stat-note">{t["note"]}</div>' if t.get("note") else ""
        if t.get("values"):
            left, right = t["values"]
            inner = f'{left}<span class="stat-vs">vs</span>{right}'
            size = " compare"
        else:
            inner = t["value"]
            size = ""
        cells += (
            '<div class="stat-tile">'
            f'<div class="stat-label">{t["label"]}</div>'
            f'<div class="stat-value{size}{accent}">{inner}</div>'
            f"{note}</div>"
        )
    return f'<div class="stat-grid">{cells}</div>'


def figure_html(uris, alt: str) -> str:
    """Emits a figure as its light and dark renderings, letting CSS show one.

    Accepts the plain string an older proof_figures.json holds, so a dashboard
    can still be built from figure data generated before render_themed existed.
    """
    if isinstance(uris, str):
        return f'<div class="figure"><img src="{uris}" alt="{alt}"></div>'
    return (
        '<div class="figure">'
        f'<img class="fig-dark" src="{uris["dark"]}" alt="{alt}">'
        f'<img class="fig-light" src="{uris["light"]}" alt="{alt}">'
        "</div>"
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
<script>{THEME_BOOT_JS}</script>
<style>
{PALETTE_CSS}
{BASE_CSS}
.page {{ max-width: 900px; margin: 0 auto; padding: 40px 20px 64px 20px; }}
</style>
<script>
const DATA = {data_json};
{CHART_JS}
</script>
{THEME_TOGGLE_HTML}
<div class="viz-root page">
{body}
<div class="page-footer">
<a href="../../submission_artifacts/index.html">&larr; All tracks</a> &middot;
<a href="{REPO_TREE_URL}">Code + README on GitHub</a>
</div>
</div>
<script>{THEME_TOGGLE_JS}</script>
"""
