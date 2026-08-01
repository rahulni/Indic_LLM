# -*- coding: utf-8 -*-
"""
build_site.py - renders MIXTURE_PLAN.md and README.md into one navigable page.

The site is a VIEW of the generated Markdown, not a second copy of it. It reads
the .md files build_plan.py already emits and the JSON audit.run() already
produced, so a number cannot appear here that disagrees with the document.

    python build_site.py        # writes ../site.html
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charts          # noqa: E402
import md2html         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "site.html")
OUT_FRAG = os.path.join(ROOT, "site.fragment.html")


def _load(name):
    with io.open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def _json(name):
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


THEME_TOGGLE = """
<button id="themebtn" class="themebtn" type="button" aria-label="Switch colour theme">
  <span aria-hidden="true">◐</span><span class="themebtn-t">Theme</span>
</button>
<script>
(function () {
  var r = document.documentElement, b = document.getElementById('themebtn');
  var KEY = 'drishtikon-theme';
  try { var v = localStorage.getItem(KEY); if (v) r.setAttribute('data-theme', v); } catch (e) {}
  b.addEventListener('click', function () {
    var cur = r.getAttribute('data-theme');
    if (!cur) {
      cur = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    var next = cur === 'dark' ? 'light' : 'dark';
    r.setAttribute('data-theme', next);
    try { localStorage.setItem(KEY, next); } catch (e) {}
  });
})();
</script>
"""

CSS = r"""
/* ---------------------------------------------------------------------------
   Tokens. Neutrals carry a slight indigo bias so they read as chosen rather
   than inherited. Page chrome is deliberately achromatic: colour is spent on
   data marks and status only.
   ------------------------------------------------------------------------ */
:root {
  color-scheme: light;
  --ground:#F7F8FA; --surface:#FFFFFF; --surface-2:#EFF1F6; --sunken:#E7EAF1;
  --ink:#151823; --ink-2:#454B5E; --ink-3:#6C7385; --rule:#DCE0EA; --rule-2:#C6CBD9;
  --accent:#3B4CC0;                 /* structural only: links, active nav      */
  --mark:#2a78d6; --mark-alt:#eb6834;
  --ok:#0ca30c; --warn:#fab219; --serious:#ec835a; --critical:#d03b3b; --info:#2a78d6;
  --chip-ink:#151823;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --measure: 74ch;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --ground:#12141A; --surface:#171A22; --surface-2:#1E2230; --sunken:#0E1016;
    --ink:#E8EAF0; --ink-2:#AAB1C2; --ink-3:#7C8496; --rule:#272C3A; --rule-2:#39404F;
    --accent:#8B96EE;
    --mark:#3987e5; --mark-alt:#d95926;
    --chip-ink:#E8EAF0;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:#12141A; --surface:#171A22; --surface-2:#1E2230; --sunken:#0E1016;
  --ink:#E8EAF0; --ink-2:#AAB1C2; --ink-3:#7C8496; --rule:#272C3A; --rule-2:#39404F;
  --accent:#8B96EE;
  --mark:#3987e5; --mark-alt:#d95926;
  --chip-ink:#E8EAF0;
}
:root[data-theme="light"] {
  color-scheme: light;
  --ground:#F7F8FA; --surface:#FFFFFF; --surface-2:#EFF1F6; --sunken:#E7EAF1;
  --ink:#151823; --ink-2:#454B5E; --ink-3:#6C7385; --rule:#DCE0EA; --rule-2:#C6CBD9;
  --accent:#3B4CC0; --mark:#2a78d6; --mark-alt:#eb6834; --chip-ink:#151823;
}

* { box-sizing:border-box; }
body { margin:0; background:var(--ground); color:var(--ink);
       font-family:var(--sans); font-size:15.5px; line-height:1.62;
       -webkit-font-smoothing:antialiased; }
::selection { background:color-mix(in srgb, var(--accent) 26%, transparent); }

/* --- shell --------------------------------------------------------------- */
.wrap { display:grid; grid-template-columns:280px minmax(0,1fr); gap:0;
        max-width:1360px; margin:0 auto; }
.rail { position:sticky; top:0; height:100vh; overflow-y:auto;
        border-right:1px solid var(--rule); padding:22px 18px 40px;
        background:var(--ground); }
.main { min-width:0; padding:30px 44px 120px; }

.brand { font-family:var(--mono); font-size:12px; letter-spacing:.09em;
         text-transform:uppercase; color:var(--ink-3); margin-bottom:4px; }
.brand-t { font-family:var(--mono); font-weight:650; font-size:17px;
           letter-spacing:-.02em; line-height:1.25; margin-bottom:16px;
           text-wrap:balance; }

.docsw { display:flex; gap:4px; background:var(--surface-2); padding:3px;
         border-radius:8px; margin-bottom:16px; }
.docsw button { flex:1; font:inherit; font-family:var(--mono); font-size:12px;
                padding:7px 8px; border:0; border-radius:6px; cursor:pointer;
                background:transparent; color:var(--ink-2); letter-spacing:.02em; }
.docsw button[aria-selected="true"] { background:var(--surface); color:var(--ink);
                box-shadow:0 1px 2px rgba(0,0,0,.10); font-weight:640; }

.search { width:100%; font:inherit; font-family:var(--mono); font-size:12.5px;
          padding:8px 10px; border:1px solid var(--rule-2); border-radius:7px;
          background:var(--surface); color:var(--ink); margin-bottom:6px; }
.search::placeholder { color:var(--ink-3); }
.hits { font-family:var(--mono); font-size:11px; color:var(--ink-3);
        min-height:16px; margin-bottom:10px; }

nav.toc { font-family:var(--mono); font-size:12.5px; }
nav.toc a { display:block; padding:3.5px 8px 3.5px 10px; color:var(--ink-2);
            text-decoration:none; border-left:2px solid transparent;
            border-radius:0 5px 5px 0; }
nav.toc a:hover { background:var(--surface-2); color:var(--ink); }
nav.toc a.lvl3 { padding-left:24px; font-size:11.5px; color:var(--ink-3); }
nav.toc a.active { border-left-color:var(--accent); color:var(--accent);
                   background:var(--surface-2); font-weight:640; }
nav.toc a:focus-visible, .docsw button:focus-visible, .search:focus-visible,
a:focus-visible, summary:focus-visible {
  outline:2px solid var(--accent); outline-offset:2px; }

/* --- type ---------------------------------------------------------------- */
.main h1,.main h2,.main h3,.main h4 { font-family:var(--mono); letter-spacing:-.02em;
  text-wrap:balance; scroll-margin-top:18px; }
.h1 { font-size:31px; font-weight:660; line-height:1.2; margin:0 0 14px; }
.h2 { font-size:20px; font-weight:640; margin:44px 0 12px; padding-top:16px;
      border-top:1px solid var(--rule); }
.h3 { font-size:16px; font-weight:640; margin:30px 0 9px; color:var(--ink); }
.h4 { font-size:14px; font-weight:640; margin:22px 0 7px; color:var(--ink-2); }
.main p, .main li { max-width:var(--measure); }
.main p { margin:0 0 13px; }
.main ul,.main ol { margin:0 0 14px; padding-left:22px; }
.main li { margin:3px 0; }
.anchor { text-decoration:none; color:var(--ink-3); opacity:0; margin-left:-16px;
          padding-right:6px; font-weight:400; }
h2:hover .anchor, h3:hover .anchor, h4:hover .anchor { opacity:.6; }
a { color:var(--accent); text-underline-offset:2px; }
.xref { font-family:var(--mono); font-size:.94em; text-decoration:none;
        border-bottom:1px dotted currentColor; }
strong { font-weight:660; color:var(--ink); }
code { font-family:var(--mono); font-size:.885em; background:var(--surface-2);
       padding:1.5px 5px; border-radius:4px; color:var(--ink); }
pre.code { font-family:var(--mono); font-size:12.5px; line-height:1.55;
  background:var(--sunken); border:1px solid var(--rule); border-radius:9px;
  padding:13px 15px; overflow-x:auto; margin:0 0 16px; }
pre.code code { background:none; padding:0; }
blockquote { margin:0 0 16px; padding:11px 16px; background:var(--surface-2);
  border-left:3px solid var(--accent); border-radius:0 8px 8px 0;
  color:var(--ink); max-width:var(--measure); }
blockquote p { margin:0; }

/* --- tables -------------------------------------------------------------- */
.tablewrap { overflow-x:auto; margin:0 0 20px; border:1px solid var(--rule);
             border-radius:10px; background:var(--surface); }
.tablewrap:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
table { border-collapse:collapse; width:100%; font-size:13px;
        font-variant-numeric:tabular-nums; }
th { font-family:var(--mono); font-size:11px; letter-spacing:.05em;
     text-transform:uppercase; text-align:left; color:var(--ink-3);
     padding:10px 13px; border-bottom:1px solid var(--rule-2);
     background:var(--surface-2); white-space:nowrap; position:sticky; top:0; }
td { padding:9px 13px; border-bottom:1px solid var(--rule); vertical-align:top;
     color:var(--ink-2); }
tbody tr:last-child td { border-bottom:0; }
tbody tr:hover td { background:var(--surface-2); }
td code { font-size:.9em; }

/* --- status chips: form + glyph + label, never colour alone -------------- */
.chip { display:inline-flex; align-items:center; gap:5px; font-family:var(--mono);
  font-size:10.5px; font-weight:640; letter-spacing:.04em; padding:2.5px 8px;
  border-radius:999px; white-space:nowrap; color:var(--chip-ink);
  border:1px solid; }
.chip-glyph { font-size:9px; line-height:1; }
.chip-ok       { border-color:var(--ok);       background:color-mix(in srgb,var(--ok) 15%,transparent); }
.chip-warn     { border-color:var(--warn);     background:color-mix(in srgb,var(--warn) 20%,transparent); }
.chip-serious  { border-color:var(--serious);  background:color-mix(in srgb,var(--serious) 20%,transparent); }
.chip-critical { border-color:var(--critical); background:color-mix(in srgb,var(--critical) 17%,transparent); }
.chip-info     { border-color:var(--info);     background:color-mix(in srgb,var(--info) 15%,transparent); }

/* --- figures ------------------------------------------------------------- */
.fig { margin:0 0 24px; padding:16px 18px 12px; background:var(--surface);
       border:1px solid var(--rule); border-radius:11px; }
.fig-cap { display:flex; flex-direction:column; gap:3px; margin-bottom:12px; }
.fig-title { font-family:var(--mono); font-size:13px; font-weight:650;
             letter-spacing:-.01em; }
.fig-sub { font-size:12.5px; color:var(--ink-3); max-width:var(--measure); }
.fig-body { overflow-x:auto; }
.fig svg { display:block; width:100%; min-width:520px; height:auto;
           font-family:var(--mono); font-variant-numeric:tabular-nums; }
.bar { fill:var(--mark); }
.bar-ok { fill:var(--ok); } .bar-warn { fill:var(--warn); }
.bar-serious { fill:var(--serious); }
.bar-label { font-size:11px; fill:var(--ink-2); }
.bar-val { font-size:11px; fill:var(--ink); font-weight:600; }
.loss { fill:var(--ink-3); font-weight:400; }
.ref { stroke:var(--ink-3); stroke-width:1; stroke-dasharray:3 3; }
.ref-t { font-size:10px; fill:var(--ink-3); }
.clip-mark { fill:none; stroke:var(--ground); stroke-width:2.5; }
.spark { fill:var(--mark); }
.spark-x { font-size:8px; fill:var(--ink-3); }
.smallmult { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
             gap:14px; }
.sm-cell { border:1px solid var(--rule); border-radius:8px; padding:9px 10px;
           background:var(--ground); }
.sm-head { display:flex; justify-content:space-between; align-items:baseline;
           gap:6px; margin-bottom:3px; }
.sm-name { font-size:11px; color:var(--ink-2); overflow:hidden;
           text-overflow:ellipsis; white-space:nowrap; }
.sm-val { font-family:var(--mono); font-size:11px; font-weight:650; }
.sm-cell svg { min-width:0; }

/* --- stat row ------------------------------------------------------------ */
.statrow { display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
           gap:10px; margin:0 0 26px; }
.stat { background:var(--surface); border:1px solid var(--rule);
        border-radius:10px; padding:13px 15px; }
.stat-k { font-family:var(--mono); font-size:10.5px; letter-spacing:.06em;
          text-transform:uppercase; color:var(--ink-3); margin-bottom:5px; }
.stat-v { font-family:var(--mono); font-size:21px; font-weight:660;
          letter-spacing:-.025em; font-variant-numeric:tabular-nums;
          display:flex; align-items:baseline; gap:7px; }
.stat-delta { font-size:11.5px; font-weight:640; }
.stat-delta.up { color:var(--ok); }
.stat-n { font-size:11.5px; color:var(--ink-3); margin-top:3px; line-height:1.4; }


.themebtn { position:fixed; top:14px; right:16px; z-index:50; display:flex;
  align-items:center; gap:7px; font-family:var(--mono); font-size:11.5px;
  letter-spacing:.05em; text-transform:uppercase; color:var(--ink-2);
  background:var(--surface); border:1px solid var(--rule-2); border-radius:999px;
  padding:7px 13px; cursor:pointer; }
.themebtn:hover { color:var(--ink); border-color:var(--ink-3); }
.themebtn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
@media (max-width:940px) { .themebtn-t { display:none; } }

/* --- alerts, details, mermaid ------------------------------------------- */
.alert { display:flex; flex-direction:column; gap:5px; margin:0 0 18px;
  padding:12px 15px; border-radius:9px; border:1px solid; border-left-width:3px;
  background:var(--surface); max-width:var(--measure); }
.alert-h { display:flex; align-items:center; gap:7px; font-family:var(--mono);
  font-size:11px; font-weight:660; letter-spacing:.07em; text-transform:uppercase; }
.alert-g { display:inline-grid; place-items:center; width:15px; height:15px;
  border-radius:50%; font-size:9px; color:var(--ground); }
.alert-note      { border-color:var(--info); }
.alert-note .alert-h, .alert-note .alert-g   { color:var(--info); }
.alert-note .alert-g   { background:var(--info); color:var(--surface); }
.alert-important { border-color:var(--accent); }
.alert-important .alert-h { color:var(--accent); }
.alert-important .alert-g { background:var(--accent); color:var(--surface); }
.alert-warning   { border-color:var(--warn); }
.alert-warning .alert-h { color:var(--warn); }
.alert-warning .alert-g { background:var(--warn); color:#151823; }
.alert-caution   { border-color:var(--critical); }
.alert-caution .alert-h { color:var(--critical); }
.alert-caution .alert-g { background:var(--critical); color:var(--surface); }
.alert-tip       { border-color:var(--ok); }
.alert-tip .alert-h { color:var(--ok); }
.alert-tip .alert-g { background:var(--ok); color:var(--surface); }

details { margin:0 0 20px; border:1px solid var(--rule); border-radius:10px;
  background:var(--surface); padding:0 15px; max-width:var(--measure); }
details[open] { padding-bottom:10px; }
summary { cursor:pointer; padding:11px 0; font-family:var(--mono); font-size:12px;
  letter-spacing:.04em; text-transform:uppercase; color:var(--ink-2); }
details ul { margin:4px 0 0; padding-left:18px; font-size:13.5px; }
details li { margin:2px 0; }

pre.mermaid { background:var(--surface); border:1px solid var(--rule);
  border-radius:10px; padding:16px; margin:0 0 20px; overflow-x:auto;
  text-align:center; }

td.ta-right, th.ta-right { text-align:right; }
td.ta-center, th.ta-center { text-align:center; }

/* --- search highlight ---------------------------------------------------- */
mark.hit { background:color-mix(in srgb,var(--warn) 55%,transparent);
           color:var(--ink); border-radius:2px; padding:0 1px; }
.dimmed { display:none; }

.doc[hidden] { display:none; }
.foot { margin-top:56px; padding-top:18px; border-top:1px solid var(--rule);
        font-family:var(--mono); font-size:11.5px; color:var(--ink-3);
        max-width:var(--measure); }

@media (max-width:940px) {
  .wrap { grid-template-columns:1fr; }
  .rail { position:static; height:auto; border-right:0;
          border-bottom:1px solid var(--rule); }
  .main { padding:22px 20px 80px; }
  nav.toc { max-height:210px; overflow-y:auto; }
}
@media (prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; } }
html { scroll-behavior:smooth; }
"""

JS = r"""
(function () {
  var docs = Array.prototype.slice.call(document.querySelectorAll('.doc'));
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.docsw button'));
  var search = document.getElementById('q');
  var hits = document.getElementById('hits');

  function activeDoc() { return docs.find(function (d) { return !d.hidden; }); }

  function showDoc(id) {
    docs.forEach(function (d) { d.hidden = (d.dataset.doc !== id); });
    tabs.forEach(function (t) {
      t.setAttribute('aria-selected', String(t.dataset.doc === id));
    });
    document.querySelectorAll('nav.toc').forEach(function (n) {
      n.hidden = (n.dataset.doc !== id);
    });
    if (search.value) run(search.value);
    window.scrollTo(0, 0);
  }
  tabs.forEach(function (t) {
    t.addEventListener('click', function () { showDoc(t.dataset.doc); });
  });

  /* --- search: filter sections, highlight matches ----------------------- */
  function clear(root) {
    root.querySelectorAll('mark.hit').forEach(function (m) {
      var p = m.parentNode;
      p.replaceChild(document.createTextNode(m.textContent), m);
      p.normalize();
    });
    root.querySelectorAll('.dimmed').forEach(function (e) {
      e.classList.remove('dimmed');
    });
  }

  function walk(node, re, count) {
    if (node.nodeType === 3) {
      var t = node.nodeValue, m = t.match(re);
      if (!m) return count;
      var frag = document.createDocumentFragment(), last = 0, r = new RegExp(re.source, 'gi'), x;
      while ((x = r.exec(t)) !== null) {
        frag.appendChild(document.createTextNode(t.slice(last, x.index)));
        var mk = document.createElement('mark');
        mk.className = 'hit'; mk.textContent = x[0];
        frag.appendChild(mk); last = x.index + x[0].length; count.n++;
      }
      frag.appendChild(document.createTextNode(t.slice(last)));
      node.parentNode.replaceChild(frag, node);
      return count;
    }
    if (node.nodeType === 1 && !/^(SCRIPT|STYLE|MARK)$/.test(node.tagName)) {
      Array.prototype.slice.call(node.childNodes).forEach(function (c) {
        walk(c, re, count);
      });
    }
    return count;
  }

  function run(q) {
    var doc = activeDoc();
    if (!doc) return;
    clear(doc);
    q = q.trim();
    if (q.length < 2) { hits.textContent = ''; return; }
    var re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
    var count = walk(doc, re, { n: 0 });
    hits.textContent = count.n ? count.n + ' match' + (count.n === 1 ? '' : 'es')
                               : 'no matches';
  }
  var t;
  search.addEventListener('input', function () {
    clearTimeout(t); t = setTimeout(function () { run(search.value); }, 140);
  });
  search.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') { search.value = ''; run(''); search.blur(); }
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== search) {
      e.preventDefault(); search.focus();
    }
  });

  /* --- scroll-spy -------------------------------------------------------- */
  var spy = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) return;
      var id = en.target.id;
      document.querySelectorAll('nav.toc a').forEach(function (a) {
        a.classList.toggle('active', a.getAttribute('href') === '#' + id);
      });
    });
  }, { rootMargin: '0px 0px -76% 0px', threshold: 0 });
  document.querySelectorAll('.main h2[id], .main h3[id]').forEach(function (h) {
    spy.observe(h);
  });
})();
"""


def build():
    plan_md = _load("MIXTURE_PLAN.md")
    readme_md = _load("README.md")
    res = _json("mixture_results.json")

    plan_html, plan_toc = md2html.render(plan_md)
    readme_html, readme_toc = md2html.render(readme_md)

    # ---- figures, from the same audit data ------------------------------
    figs = {}
    if res:
        audited = res["audited"]
        budget = res["budget"]
        gate = None
        try:
            import spec
            gate = spec.GATING
        except Exception:
            pass

        figs["stats"] = charts.stat_row([
            dict(k="Token budget", v=f"{budget/1e12:.2f}T",
                 n="parameter, not assumption — swept 2.4T→14T"),
            dict(k="Data gate", v=f"{gate['measured_clean_tokens']/1e6:.1f}M" if gate else "—",
                 delta=f"+{gate['added_this_session']/1e6:.1f}M" if gate else None,
                 n="cleaned tokens, measured" + (f" · {gate['growth_multiple']:.2f}× session 4" if gate else "")),
            dict(k="Capability lanes", v=str(len(audited)),
                 n="every share defended and benchmark-tied"),
            dict(k="Protected floor", v=f"{res['floor_total']:.1f}%",
                 n="of every window, before the selector runs"),
            dict(k="Anneal reserve", v=f"{res['anneal']['pct_of_budget']:.0f}%",
                 n="verified tiers only, supply-audited"),
            dict(k="Proxy screen", v=f"${res['cost']['total_usd']:,.0f}",
                 n=f"{res['cost']['pct_of_full']:.2f}% of the full run"),
        ])

        figs["mixture"] = charts.ranked_bars(
            [dict(label=a["label"], share=a["share"], verdict=a["verdict"])
             for a in audited],
            "share", "label", lambda v: f"{v:.2f}%",
            "Budget share by capability lane",
            "Derived as the token-weighted average of the phase schedule — an "
            "output of the curriculum, not an independent input.",
            status_key="verdict")

        figs["epochs"] = charts.ranked_bars(
            [dict(label=a["label"], epochs=a["epochs"], verdict=a["verdict"])
             for a in audited],
            "epochs", "label", lambda v: f"{v:.2f}×",
            "Epochs of real data each lane needs",
            "Against candidate tokens, after the selector’s 2× draw. The dashed "
            "line is the 4-epoch repetition ceiling; bars are coloured by verdict.",
            status_key="verdict", ref=4.0, ref_label="4-epoch ceiling", axis_max=8.0)

        figs["phases"] = charts.phase_ladder(
            [dict(label=a["label"], share=a["share"], phases=a["phases"])
             for a in audited],
            [p["id"] for p in res["phases"]])

    # cleaning funnel from the session-5 run
    ver = _json(os.path.join("cleaning", "results_verified.json"))
    if ver:
        fn = ver.get("result", {}).get("survival_funnel")
        if fn:
            figs["funnel"] = charts.funnel(
                [dict(name=s["name"], docs=s["docs"]) for s in fn],
                "Cleaning Sangraha Verified, stage by stage",
                "14,596 drawn → 11,554 kept. The 16.6% quality-filter loss is a "
                "length bias in the stopword rule, diagnosed in §16.1.")

    def inject(html_str, marker, key):
        if key in figs and marker in html_str:
            return html_str.replace(marker, figs[key] + marker, 1)
        return html_str

    plan_html = inject(plan_html, '<h2 id="sec-4"', "mixture")
    plan_html = inject(plan_html, '<h3 id="sec-5.3"', "epochs")
    plan_html = inject(plan_html, '<h2 id="sec-3"', "phases")
    plan_html = inject(plan_html, '<h2 id="sec-2"', "funnel")

    def toc_html(toc, doc_id, hidden):
        items = []
        for lvl, hid, title in toc:
            if lvl == 1:
                continue
            items.append(f'<a class="lvl{lvl}" href="#{hid}">{title}</a>')
        return (f'<nav class="toc" data-doc="{doc_id}"'
                f'{" hidden" if hidden else ""}>{"".join(items)}</nav>')

    stats = figs.get("stats", "")
    page = f"""<div class="wrap">
<aside class="rail">
  <div class="brand">ERA V5 · Session 5</div>
  <div class="brand-t">Drishtikon-40B<br>Mixture &amp; Curriculum</div>
  <div class="docsw" role="tablist" aria-label="Document">
    <button role="tab" data-doc="plan" aria-selected="true">Specification</button>
    <button role="tab" data-doc="readme" aria-selected="false">Repo guide</button>
  </div>
  <input id="q" class="search" type="search" placeholder="Search  ( / )"
         aria-label="Search this document" autocomplete="off">
  <div id="hits" class="hits" role="status" aria-live="polite"></div>
  {toc_html(plan_toc, "plan", False)}
  {toc_html(readme_toc, "readme", True)}
</aside>
<main class="main">
  <article class="doc" data-doc="plan">
    {stats}
    {plan_html}
    <div class="foot">Generated by plan/build_site.py from MIXTURE_PLAN.md and
    mixture_results.json. Every figure is drawn from the same audit output as the
    tables beside it, so a chart cannot disagree with the document.</div>
  </article>
  <article class="doc" data-doc="readme" hidden>
    {readme_html}
    <div class="foot">Generated by plan/build_site.py from README.md.</div>
  </article>
</main>
</div>"""

    body = f"<style>{CSS}</style>\n{page}\n<script>{JS}</script>\n"

    # Two outputs, because the two destinations want different things.
    #
    #  site.html          a COMPLETE document: doctype, <head>, <title>, and a
    #                     theme toggle. This is what GitHub Pages, raw.githack
    #                     and a local double-click serve. Without a doctype a
    #                     browser renders it in quirks mode.
    #  site.fragment.html body-only, for publishing as an Artifact - that
    #                     pipeline supplies its own doctype/head/body wrapper,
    #                     so a full document there would nest inside another.
    standalone = (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Drishtikon-40B — V5 Mixture &amp; Curriculum</title>\n"
        '<meta name="description" content="The V5 data mixture-and-curriculum '
        'specification for Drishtikon-40B: budget shares, supply audit, '
        'protected floor, anneal reserve, and the proxy experiment.">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"</head>\n<body>\n{THEME_TOGGLE}\n{body}</body>\n</html>\n")

    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(standalone)
    with io.open(OUT_FRAG, "w", encoding="utf-8") as f:
        f.write(body)
    out = standalone
    print(f"wrote {os.path.normpath(OUT)}  ({len(out):,} chars, standalone)")
    print(f"wrote {os.path.normpath(OUT_FRAG)}  ({len(body):,} chars, for Artifact)")
    print(f"  figures: {', '.join(sorted(figs)) or 'none'}")
    print(f"  sections: plan {len(plan_toc)}, readme {len(readme_toc)}")
    return out


if __name__ == "__main__":
    build()
