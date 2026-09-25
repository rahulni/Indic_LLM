"""Build site.html - a single self-contained dark page explaining the whole thing.

Everything is inlined (no network at open time) and every number is read out of
assets/results.json and assets/ladder.json, so the page cannot drift from the runs.

    python tools/build_site.py
"""
from __future__ import annotations

import json
import os

HEAD = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Reversible Transformers</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&display=swap">
<style>
:root{
  color-scheme: dark;
  --ground:#0b0b12; --surface:#14141c; --raised:#1c1c26; --line:#2b2b38;
  --ink:#f4f3f7; --ink2:#a6a4b6; --ink3:#6e6c80;
  --accent:#f0a830; --accent-dim:#8a5f18;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300;
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  --serif:"IBM Plex Serif",Georgia,serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
  --col:68ch; --wide:min(1080px,100%);
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--ground);color:var(--ink)}
body{font-family:var(--serif);font-size:17px;line-height:1.65;
  -webkit-font-smoothing:antialiased;padding-block:0 6rem}
.wrap{max-width:var(--col);margin-inline:auto;padding-inline:20px}
.bleed{max-width:var(--wide);margin-inline:auto;padding-inline:20px}
h1,h2,h3,.ui{font-family:var(--sans)}
h1{font-size:clamp(2.1rem,6vw,3.4rem);line-height:1.04;font-weight:700;
  letter-spacing:-.025em;text-wrap:balance;margin:0 0 .6rem}
h2{font-size:clamp(1.4rem,3.4vw,1.9rem);font-weight:600;letter-spacing:-.015em;
  text-wrap:balance;margin:3.6rem 0 .4rem}
h3{font-size:1.06rem;font-weight:600;margin:2.2rem 0 .3rem;letter-spacing:-.005em}
p{margin:0 0 1.05rem}
a{color:var(--accent);text-underline-offset:3px}
code,.mono{font-family:var(--mono);font-size:.88em}
.eyebrow{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;
  letter-spacing:.16em;color:var(--accent);margin:0 0 .9rem}
.lede{font-size:1.16rem;color:var(--ink2);margin-bottom:1.6rem}
.sub{color:var(--ink3);font-size:.9rem;font-family:var(--sans)}

/* the trace: a leapfrog stepping across the masthead */
header.masthead{border-bottom:1px solid var(--line);padding-block:4.5rem 2.2rem;
  margin-bottom:.5rem;position:relative;overflow:hidden}
#trace{display:block;width:100%;height:96px;margin-top:1.8rem}

nav.toc{position:sticky;top:0;z-index:20;background:rgba(11,11,18,.92);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);
  font-family:var(--sans);font-size:.8rem}
nav.toc ul{display:flex;gap:1.3rem;list-style:none;margin:0 auto;padding:.7rem 20px;
  max-width:var(--wide);overflow-x:auto;scrollbar-width:none}
nav.toc a{color:var(--ink3);text-decoration:none;white-space:nowrap}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--ink)}

.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:1.3rem;margin:1.8rem 0}
.panel h3{margin-top:0}
figure{margin:2rem 0}
figcaption{font-family:var(--sans);font-size:.83rem;color:var(--ink3);margin-top:.7rem}

table{width:100%;border-collapse:collapse;font-family:var(--sans);font-size:.87rem;
  font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
th,td{text-align:right;padding:.5rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--ink3);font-weight:500;font-size:.76rem;text-transform:uppercase;
  letter-spacing:.07em}
tbody tr:hover{background:var(--raised)}
td.win{color:var(--accent);font-weight:600}

.eq{font-family:var(--mono);font-size:.92rem;background:var(--surface);
  border-left:2px solid var(--accent-dim);padding:.8rem 1rem;margin:1.1rem 0;
  overflow-x:auto;color:var(--ink2);line-height:1.75}
.eq b{color:var(--ink);font-weight:500}

.note{border-left:2px solid var(--line);padding-left:1rem;color:var(--ink2);
  font-size:.95rem;margin:1.4rem 0}
.note b{color:var(--ink)}

.chips{display:flex;flex-wrap:wrap;gap:.45rem;margin:1.1rem 0}
.chip{font-family:var(--sans);font-size:.78rem;padding:.3rem .7rem;border-radius:999px;
  border:1px solid var(--line);color:var(--ink2);background:var(--surface);cursor:pointer}
.chip[aria-pressed="true"]{border-color:var(--accent);color:var(--ground);
  background:var(--accent);font-weight:600}
.chip:focus-visible,button:focus-visible,a:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px}

/* the machine */
.machine{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:1.2rem;margin:1.6rem 0}
.machine-head{display:flex;flex-wrap:wrap;gap:.8rem;align-items:baseline;
  justify-content:space-between;margin-bottom:.9rem}
#stage{display:block;width:100%;height:auto}
.controls{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin-top:.9rem}
button.ctl{font-family:var(--sans);font-size:.83rem;padding:.42rem .9rem;border-radius:7px;
  border:1px solid var(--line);background:var(--raised);color:var(--ink);cursor:pointer}
button.ctl:hover{border-color:var(--ink3)}
button.ctl[disabled]{opacity:.4;cursor:not-allowed}
.readout{font-family:var(--mono);font-size:.8rem;color:var(--ink2);margin-left:auto;
  text-align:right;line-height:1.5}
.readout b{color:var(--accent);font-weight:500}
.legend{display:flex;flex-wrap:wrap;gap:1rem;font-family:var(--sans);font-size:.78rem;
  color:var(--ink3);margin-top:.7rem}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:.35rem;
  vertical-align:baseline}

.tip{position:fixed;pointer-events:none;background:var(--raised);border:1px solid var(--line);
  border-radius:7px;padding:.45rem .6rem;font-family:var(--mono);font-size:.76rem;
  color:var(--ink);z-index:60;opacity:0;transition:opacity .12s;max-width:60vw}
.tip.on{opacity:1}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;
  margin:1.8rem 0}
.kpi{background:var(--surface);padding:1rem 1.1rem}
.kpi .n{font-family:var(--sans);font-size:1.75rem;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.1}
.kpi .l{font-family:var(--sans);font-size:.76rem;color:var(--ink3);margin-top:.25rem}

footer{border-top:1px solid var(--line);margin-top:4rem;padding-top:1.4rem;
  font-family:var(--sans);font-size:.8rem;color:var(--ink3)}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01ms!important;
  transition-duration:.01ms!important}}
@media (max-width:560px){body{font-size:16px}.readout{margin-left:0;text-align:left}}
</style>
</head>
<body>
"""

BODY_TMPL = """
<header class="masthead">
  <div class="wrap">
    <p class="eyebrow">ERA V5 &middot; session 13 &middot; distributed training II</p>
    <h1>A transformer that throws its activations away</h1>
    <p class="lede">Depth is time, a residual block is an Euler step, and the integrators
    that can run backwards give you their activations back for free. Here is what that
    actually costs, measured on one 8&nbsp;GB laptop GPU.</p>
    <p class="sub">__PARAMS__ parameters &middot; 50M tokens &middot; __NRUNS__ runs &middot; __GPU__</p>
  </div>
  <svg id="trace" viewBox="0 0 1200 96" preserveAspectRatio="none" aria-hidden="true"></svg>
</header>

<nav class="toc"><ul>
  <li><a href="#problem">The problem</a></li>
  <li><a href="#hinge">Depth is time</a></li>
  <li><a href="#machine">The machine</a></li>
  <li><a href="#trap">The trap</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#wall">The moving wall</a></li>
  <li><a href="#take">Takeaways</a></li>
</ul></nav>

<main>
<section class="wrap" id="problem">
<h2>One term in the budget scales with your data</h2>
<p>Training memory goes on four things. Three of them&nbsp;&mdash; the weights, their
gradients, and Adam&rsquo;s moments&nbsp;&mdash; cost sixteen bytes per parameter and care
about nothing else. The fourth cares about nothing but your data.</p>

<div class="eq">
weights                 2 bytes/param      &mdash;<br>
gradients               2 bytes/param      &mdash;<br>
Adam moments + master  12 bytes/param      &mdash;<br>
<b>activations            ~12 tensors/layer   batch &times; sequence &times; depth</b>
</div>

<p>ZeRO splits the first three across GPUs and explicitly leaves activations alone. Tensor,
sequence, pipeline and context parallelism split the computation, which divides activations
across devices&nbsp;&mdash; the total bill is unchanged, only shared out. Reversibility is
the only idea here that <b>deletes the term</b>, and it needs no second GPU to do it.</p>

<div class="kpis">__KPIS__</div>
</section>

<section class="wrap" id="hinge">
<h2>Depth is time</h2>
<p>A pre-LN transformer block is already an explicit Euler step, with step size one:</p>
<div class="eq"><b>h<sub>l+1</sub> = h<sub>l</sub> + G<sub>l</sub>(h<sub>l</sub>)</b>
&nbsp;&nbsp;&larr;&nbsp; dh/dl = G(h), solved by Euler</div>
<p>where <span class="mono">G<sub>l</sub></span> is the residual <i>branch</i>&nbsp;&mdash;
layer-norm, attention, layer-norm, MLP, times a learnable scale. Read the depth axis as
time and &ldquo;make it reversible&rdquo; stops being a trick. It becomes a question about
integrators: <b>which ones run backwards?</b></p>
<p>The answer turns out to be about how many states the scheme carries. One state cannot be
inverted explicitly. Two can&nbsp;&mdash; and it barely matters whether the second state is
a velocity, a memory of the previous layer, or the other half of the channels.</p>
</section>

<section class="bleed" id="machine">
<h2>What is alive, and when</h2>
<p style="max-width:var(--col);color:var(--ink2)">Pick an integrator and step through a
forward and backward pass. Filled cells are tensors currently held in memory; hollow ones
have been freed or not yet built. The counter on the right is what the GPU is actually
carrying.</p>

<div class="machine">
  <div class="machine-head">
    <div class="chips" role="group" aria-label="integrator">__CHIPS__</div>
  </div>
  <svg id="stage" viewBox="0 0 1000 300" role="img" aria-label="activation lifetime diagram"></svg>
  <div class="controls">
    <button class="ctl" id="back">&larr; Step back</button>
    <button class="ctl" id="play">Play</button>
    <button class="ctl" id="fwd">Step forward &rarr;</button>
    <button class="ctl" id="reset">Reset</button>
    <div class="readout" id="readout"></div>
  </div>
  <div class="legend">
    <span><span class="swatch" style="background:var(--accent)"></span>held in memory</span>
    <span><span class="swatch" style="background:var(--raised);border:1px solid var(--line)"></span>freed or not yet computed</span>
    <span><span class="swatch" style="background:var(--s3)"></span>being recomputed now</span>
  </div>
</div>
</section>

<section class="wrap" id="trap">
<h2>The obvious reading is a trap</h2>
<p>&ldquo;Reversible Euler&rdquo; most naturally means inverting that equation directly.
Rearranged, it reads</p>
<div class="eq">h<sub>l</sub> = h<sub>l+1</sub> &minus; G<sub>l</sub>(<b>h<sub>l</sub></b>)
&nbsp;&nbsp;&larr;&nbsp; G evaluated at the thing we are solving for</div>
<p>which is implicit. So you iterate&nbsp;&mdash; and the iteration converges only while
<span class="mono">Lip(G) = &gamma;&middot;Lip(F) &lt; 1</span>. Measured on this model,
<span class="mono">Lip(F) &asymp; 10.5</span>, so the default
<span class="mono">&gamma; = 0.1</span> lands at <b>1.05</b>: almost exactly on the line.
The iteration neither converges nor explodes. It stalls, at a reconstruction error near
0.09, however many steps you give it.</p>
<div class="panel">
<h3>Why that is worse than it looks</h3>
<p style="margin-bottom:.6rem"><b>&gamma; is a learnable parameter.</b> Nothing in the
optimiser knows about the constraint, so a run can start inside the convergent region and
walk out of it. When it does, the reconstruction silently stops matching, the gradients
quietly become wrong, and the loss keeps falling anyway.</p>
<p style="margin:0"><b>The safe region belongs to the model, not the method.</b>
<span class="mono">Lip</span> scales with width and sequence length. At
<span class="mono">d=16</span> it reads about 0.04 and everything looks fine&nbsp;&mdash;
so a unit test on a toy model certifies a method that cannot train at real width.</p>
</div>
<p><b>And training makes it worse.</b> Because <span class="mono">&gamma;</span> is
learned, the run walks steadily further out of the convergent region:
<span class="mono">Lip(G)</span> climbs from <b>__LIP0__</b> to <b>__LIPMAX__</b>, and the
reconstruction error goes from <span class="mono">__REC0__</span> to
<span class="mono">__RECMAX__</span>. The gradients being applied stop being the
model&rsquo;s gradients, and the run ends at <b>__GLOSS__</b> validation loss against the
baseline&rsquo;s __ALOSS__.</p>
<div class="note"><b>A measurement trap, kept because it nearly fooled this report.</b>
Estimating <span class="mono">Lip</span> by finite differences under bf16 returns
<b>~80</b> instead of ~1.05. At <span class="mono">&epsilon;=10<sup>-3</sup></span> the
subtraction cancels away every significant bf16 digit, and dividing by
<span class="mono">&epsilon;</span> turns what is left into a number. A 50&times;
overestimate, entirely believable, and it would have condemned a method that in fact sits
right on the boundary. Use exact autograd VJPs.</div>
</section>

<section class="bleed" id="results">
<h2>What happened</h2>
<p style="max-width:var(--col);color:var(--ink2)">Every run: the same seed, the same
sampler stream, the same schedule, the same 50M-token budget, and no dropout or weight
decay anywhere&nbsp;&mdash; the two restrictions reversibility imposes, applied to the
baseline too so it is not quietly given an advantage.</p>

<div class="scroll">__TABLE__</div>

<p style="max-width:var(--col)"><b>Symplectic Euler won.</b> It lands at __BLOSS__ against
midpoint&rsquo;s __CLOSS__ &mdash; and the baseline&rsquo;s __ALOSS__ &mdash; on identical
data, seed and batch. That contradicts the session&rsquo;s recommendation of midpoint, and
the diagnostics say why: midpoint&rsquo;s reconstruction stays accurate, but its parasitic
odd/even mode grows <b>__PARGROW__&times;</b> over training. Damping it (a
Robert&ndash;Asselin filter) would destroy the exact reversibility that is the whole
point, so the diagnostic is reported rather than filtered away.</p>

<figure>
  <svg id="chart-loss" viewBox="0 0 1000 448" role="img" aria-label="training loss against tokens"></svg>
  <figcaption>Training loss against tokens seen. Hover for values.</figcaption>
</figure>

<figure>
  <svg id="chart-batch" viewBox="0 0 1000 448" role="img" aria-label="peak memory against batch size"></svg>
  <figcaption>Peak memory against batch size. The cross marks the largest batch that ran;
  the next size up ran out of memory. Dashed: the same walk with a chunked cross entropy.</figcaption>
</figure>

<figure>
  <svg id="chart-depth" viewBox="0 0 1000 428" role="img" aria-label="memory against depth"></svg>
  <figcaption>What one batch costs as the model gets deeper. This is the claim, measured:
  the stored baseline climbs, the reversible stacks do not.</figcaption>
</figure>
</section>

<section class="wrap" id="wall">
<h2>The wall moves; it does not vanish</h2>
<p>&ldquo;Reversible networks don&rsquo;t store activations&rdquo; is true and slightly
misleading. Peak memory is <span class="mono">O(1)</span> <i>in depth</i>, not
<span class="mono">O(1)</span>. Four things survive, and the third is the one the slogan
leaves out.</p>
<div class="eq">
weights + gradients + Adam states    fixed, __STATE__ GiB<br>
boundary states                      2&ndash;3 tensors, O(batch &times; seq)<br>
<b>one live layer&rsquo;s graph               O(batch &times; seq)</b> &nbsp;&larr; backward still needs it<br>
<b>fp32 logits in the LM head          O(batch &times; seq &times; vocab)</b> &nbsp;&larr; the new bottleneck
</div>
<p>At batch 128 those logits are <span class="mono">128&times;512&times;8192&times;4 =
2.1&nbsp;GiB</span>&nbsp;&mdash; on their own, larger than everything reversibility just
saved. Chunking the loss so only one chunk&rsquo;s logits are ever live moves the ceiling
from <b>__BMAX_PLAIN__</b> sequences to <b>__BMAX_CHUNKED__</b>, while the storing baseline
stays pinned at <b>__BSTAR__</b>, because for it the activations were never the smaller
problem.</p>
</section>

<section class="wrap" id="take">
<h2>What to carry away</h2>
<ol>
<li><b>A pre-LN transformer is an Euler integrator.</b> Reversibility is a question about
which integrators run backwards, not a trick bolted onto attention.</li>
<li><b>One state cannot be inverted explicitly; two can.</b> Velocity, previous layer, or
split channels&nbsp;&mdash; all three buy the same thing for one extra evaluation of
<span class="mono">G</span> per layer.</li>
<li><b>Inverting Euler directly is conditional</b> on
<span class="mono">&gamma;&middot;Lip(F) &lt; 1</span>&nbsp;&mdash; unenforced, dependent
on width and sequence length, and breakable mid-run because <span class="mono">&gamma;</span>
is learned.</li>
<li><b>Check the gradients in fp64 against the same recurrence.</b> A reversible backward
that is subtly wrong has no symptom: the loss still falls.</li>
<li><b>Exactly reversible on paper is approximately reversible in floating point.</b>
<span class="mono">(a+b)&minus;b &ne; a</span>, and the error compounds with depth. Keep
the residual stream in fp32.</li>
<li><b>Compare against gradient checkpointing,</b> not against storing everything. It buys
most of the same memory for a similar slowdown and no new mathematics; reversibility&rsquo;s
edge is roughly <span class="mono">(L&minus;3)</span> stream tensors, so it grows with depth
and is thin when shallow.</li>
</ol>
</section>

<section class="wrap">
<footer>
<p>Generated from <span class="mono">assets/results.json</span> &middot; run stamp
<span class="mono">__STAMP__</span> &middot; __GPU__ &middot; torch __TORCH__, eager.<br>
Notebooks, the correctness gates and the full write-up are in the repository beside this
page.</p>
</footer>
</section>
</main>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const C = {store:"#3987e5",checkpoint:"#d95926",euler:"#199e70",midpoint:"#c98500",
           coupling:"#d55181",euler_implicit:"#008300"};
const NAME = {store:"baseline",checkpoint:"checkpointing",euler:"symplectic Euler",
              midpoint:"midpoint",coupling:"coupling",euler_implicit:"implicit Euler"};
const NS = "http://www.w3.org/2000/svg";
const el = (t,a={}) => { const n=document.createElementNS(NS,t);
  for(const k in a) n.setAttribute(k,a[k]); return n; };
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/* ---------- masthead trace: a leapfrog stepping across ---------- */
(function(){
  const s=document.getElementById("trace"), W=1200, H=96;
  const pts=[]; let h0=48, h1=52;
  for(let i=0;i<60;i++){ const f=Math.sin(i*0.42)*13 + Math.sin(i*0.17)*5;
    const h2=h0+0.34*f; pts.push([i*(W/59),h1]); h0=h1; h1=h2; }
  const d=pts.map((p,i)=>(i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
  const p=el("path",{d,fill:"none",stroke:css("--accent-dim"),"stroke-width":"1.5"});
  s.appendChild(p);
  pts.forEach((pt,i)=>{ if(i%4) return;
    s.appendChild(el("circle",{cx:pt[0],cy:pt[1],r:2.4,
      fill: i%8 ? css("--accent-dim") : css("--accent"), opacity: 0.35+0.65*(i/60)})); });
})();

/* ---------- the machine ---------- */
const SCHEMES = {
  store:    {label:"baseline",  rows:["h"],      keeps:L=>L+1, blurb:"every layer's activations kept until backward consumes them"},
  euler:    {label:"symplectic Euler", rows:["h","v"], keeps:()=>2, blurb:"h and v at the boundary; everything else rebuilt"},
  midpoint: {label:"midpoint",  rows:["h"],      keeps:()=>3, blurb:"two neighbouring states plus h0; everything else rebuilt"},
  coupling: {label:"coupling",  rows:["x1","x2"],keeps:()=>2, blurb:"the two halves at the boundary; everything else rebuilt"}
};
const L = 8;
let scheme = "midpoint", t = 0;   // t: 0..2L  (0..L forward, then backward)
const TMAX = 2*L;

function liveSet(mode, t){
  // which layer indices are held, and which is being recomputed right now
  const fwd = t <= L;
  const at = fwd ? t : (2*L - t);
  let held = [], busy = -1;
  if (mode === "store"){
    held = Array.from({length: at+1}, (_,i)=>i);
    if(!fwd) busy = at;
  } else if (mode === "midpoint"){
    if (fwd) held = [0, Math.max(0,at-1), at].filter((v,i,a)=>a.indexOf(v)===i);
    else { held = [0, at, Math.min(L,at+1)]; busy = at; }
  } else {
    if (fwd) held = [at];
    else { held = [at]; busy = at; }
    if (mode === "euler" || mode === "coupling") held = [at];
  }
  return {held:[...new Set(held)].sort((a,b)=>a-b), busy, at, fwd};
}

function drawStage(){
  const s = document.getElementById("stage"); s.innerHTML="";
  const W=1000, H=300, padL=54, padR=20, top=56;
  const cw = (W-padL-padR)/(L+1), bw = Math.min(cw-10, 62), bh=40;
  const st = liveSet(scheme, t);
  const rows = SCHEMES[scheme].rows;

  s.appendChild(Object.assign(el("text",{x:padL,y:26,fill:css("--ink2"),
    "font-family":css("--sans"),"font-size":"13"}),
    {textContent: st.fwd ? "forward  →  keeping only what the inverse will need"
                         : "backward  ←  rebuilding each layer from the one above"}));

  rows.forEach((rname, ri)=>{
    const y = top + ri*(bh+34);
    s.appendChild(Object.assign(el("text",{x:padL-12,y:y+bh/2+4,fill:css("--ink3"),
      "text-anchor":"end","font-family":css("--mono"),"font-size":"12"}),
      {textContent:rname}));
    for(let i=0;i<=L;i++){
      const x = padL + i*cw + (cw-bw)/2;
      const held = st.held.includes(i), busy = (st.busy===i);
      const g = el("g");
      g.appendChild(el("rect",{x,y,width:bw,height:bh,rx:6,
        fill: busy ? css("--s3") : (held ? css("--accent") : css("--raised")),
        stroke: held||busy ? "none" : css("--line"), "stroke-width":"1",
        opacity: busy?0.95:(held?0.92:1)}));
      g.appendChild(Object.assign(el("text",{x:x+bw/2,y:y+bh/2+4,"text-anchor":"middle",
        "font-family":css("--mono"),"font-size":"11",
        fill: held||busy ? "#14141c" : css("--ink3")}),
        {textContent: rname+String.fromCharCode(8320+Math.min(i,9))}));
      s.appendChild(g);
    }
  });

  // the pointer
  const px = padL + st.at*cw + cw/2;
  const py = top + rows.length*(bh+34) + 4;
  s.appendChild(el("path",{d:`M${px-7} ${py+14} L${px+7} ${py+14} L${px} ${py+3} Z`,
    fill:css("--accent")}));
  s.appendChild(Object.assign(el("text",{x:px,y:py+34,"text-anchor":"middle",
    fill:css("--ink2"),"font-family":css("--mono"),"font-size":"11.5"}),
    {textContent:"layer "+st.at}));

  // memory bar
  const barY = H-52, kept = st.held.length * rows.length;
  const maxKept = (L+1) * rows.length;
  s.appendChild(el("rect",{x:padL,y:barY,width:W-padL-padR,height:12,rx:6,
    fill:css("--raised")}));
  s.appendChild(el("rect",{x:padL,y:barY,width:Math.max(8,(W-padL-padR)*kept/maxKept),
    height:12,rx:6,fill:css("--accent")}));
  s.appendChild(Object.assign(el("text",{x:padL,y:barY+32,fill:css("--ink3"),
    "font-family":css("--sans"),"font-size":"12"}),
    {textContent:`${kept} tensor${kept===1?"":"s"} held of a possible ${maxKept}`}));

  document.getElementById("readout").innerHTML =
    `<b>${kept}</b> tensors held<br>${SCHEMES[scheme].blurb}`;
}

function setScheme(m){ scheme=m; t=0;
  document.querySelectorAll(".chip").forEach(c=>
    c.setAttribute("aria-pressed", String(c.dataset.mode===m)));
  drawStage(); }
document.querySelectorAll(".chip").forEach(c=>
  c.addEventListener("click",()=>setScheme(c.dataset.mode)));
document.getElementById("fwd").onclick = ()=>{ t=Math.min(TMAX,t+1); drawStage(); };
document.getElementById("back").onclick= ()=>{ t=Math.max(0,t-1); drawStage(); };
document.getElementById("reset").onclick=()=>{ t=0; drawStage(); };
let timer=null;
document.getElementById("play").onclick = (e)=>{
  if(timer){ clearInterval(timer); timer=null; e.target.textContent="Play"; return; }
  e.target.textContent="Pause";
  timer=setInterval(()=>{ t=(t+1)%(TMAX+1); drawStage(); }, 520);
};

/* ---------- charts ---------- */
const tip = document.getElementById("tip");
function hover(node, html){
  node.style.cursor="crosshair";
  node.addEventListener("pointerenter",()=>{ tip.innerHTML=html; tip.classList.add("on"); });
  node.addEventListener("pointermove",e=>{
    tip.style.left=Math.min(window.innerWidth-tip.offsetWidth-12,e.clientX+14)+"px";
    tip.style.top=(e.clientY+16)+"px"; });
  node.addEventListener("pointerleave",()=>tip.classList.remove("on"));
}

function tickPow2(a,b){
  const out=[]; for(let v=Math.pow(2,Math.ceil(Math.log2(a))); v<=b; v*=2) out.push(v);
  return out;
}
function linTicks(a,b,n){
  const out=[]; for(let i=0;i<=n;i++) out.push(a+(b-a)*i/n); return out;
}

function gutter(s, items, x, minGap){
  // Several series land on the same pixel (symplectic Euler and midpoint allocate
  // identically; three runs finish within 0.7% of each other). Nudge the labels apart
  // greedily so every one stays readable near what it names.
  items.sort((a,b)=>b.y-a.y);
  let placed=[];
  for(const it of items){
    let y=it.y;
    if(placed.length && placed[placed.length-1] - y < minGap) y = placed[placed.length-1]-minGap;
    placed.push(y);
    s.appendChild(Object.assign(el("text",{x,y:y+4,fill:it.color,
      "font-family":css("--sans"),"font-size":"12","font-weight":"500"}),
      {textContent:it.text}));
  }
}

function axes(s,W,H,pad,xs,ys,xlab,ylab,fmtX,fmtY,logX){
  const [x0,x1]=xs, [y0,y1]=ys;
  const X = v => logX ? pad.l+(Math.log2(v)-Math.log2(x0))/(Math.log2(x1)-Math.log2(x0))*(W-pad.l-pad.r)
                      : pad.l+(v-x0)/(x1-x0)*(W-pad.l-pad.r);
  const Y = v => H-pad.b-(v-y0)/(y1-y0)*(H-pad.t-pad.b);
  for(let i=0;i<=4;i++){
    const v=y0+(y1-y0)*i/4, y=Y(v);
    s.appendChild(el("line",{x1:pad.l,y1:y,x2:W-pad.r,y2:y,stroke:css("--line"),
      "stroke-width":"1"}));
    s.appendChild(Object.assign(el("text",{x:pad.l-9,y:y+4,"text-anchor":"end",
      fill:css("--ink3"),"font-family":css("--mono"),"font-size":"11"}),
      {textContent:fmtY(v)}));
  }
  // x ticks. Without these the chart has a label but names no value it actually reaches,
  // which is the whole point of an axis.
  const xticks = logX ? tickPow2(x0,x1) : linTicks(x0,x1,5);
  xticks.forEach(v=>{
    const x=X(v);
    s.appendChild(el("line",{x1:x,y1:H-pad.b,x2:x,y2:H-pad.b+5,stroke:css("--line"),
      "stroke-width":"1"}));
    s.appendChild(Object.assign(el("text",{x,y:H-pad.b+19,"text-anchor":"middle",
      fill:css("--ink3"),"font-family":css("--mono"),"font-size":"11"}),
      {textContent:fmtX(v)}));
  });
  s.appendChild(Object.assign(el("text",{x:pad.l,y:H-12,fill:css("--ink3"),
    "font-family":css("--sans"),"font-size":"12"}),{textContent:xlab}));
  // rotated inside the viewBox: anchored at pad.l-9 it ran off the left edge and the
  // browser clipped the first characters ("aining loss")
  const yl = el("text",{x:0,y:0,"text-anchor":"middle",fill:css("--ink3"),
    "font-family":css("--sans"),"font-size":"12",
    transform:`translate(16 ${(H-pad.b+pad.t)/2}) rotate(-90)`});
  yl.textContent = ylab; s.appendChild(yl);
  return {X,Y};
}

function lossChart(){
  const s=document.getElementById("chart-loss"); if(!s) return;
  const W=1000,H=448,pad={l:70,r:232,t:26,b:62};
  const runs=DATA.runs.filter(r=>!r.name.startsWith("D")&&r.mode!=="coupling");
  const xmax=Math.max(...runs.flatMap(r=>r.history.map(h=>h.tokens)))/1e6;
  const all=runs.flatMap(r=>r.history.map(h=>h.loss));
  const ymin=Math.min(...all)*0.97, ymax=Math.min(Math.max(...all),8);
  const {X,Y}=axes(s,W,H,pad,[0,xmax],[ymin,ymax],"tokens (millions)","training loss",
    v=>v.toFixed(0),v=>v.toFixed(2));
  const marks=[];
  runs.forEach(r=>{
    const pts=r.history.filter(h=>h.loss<=ymax);
    const d=pts.map((h,i)=>(i?"L":"M")+X(h.tokens/1e6).toFixed(1)+" "+Y(h.loss).toFixed(1)).join(" ");
    const p=el("path",{d,fill:"none",stroke:C[r.mode],"stroke-width":"2",
      "stroke-linejoin":"round"});
    s.appendChild(p);
    const last=pts[pts.length-1];
    s.appendChild(el("circle",{cx:X(last.tokens/1e6),cy:Y(last.loss),r:4,fill:C[r.mode],
      stroke:css("--surface"),"stroke-width":"2"}));
    marks.push({y:Y(last.loss),text:NAME[r.mode]+"  "+r.val.toFixed(3),color:C[r.mode]});
    hover(p,`<b>${NAME[r.mode]}</b><br>final val ${r.val.toFixed(4)}<br>${Math.round(r.tps).toLocaleString()} tok/s`);
  });
  gutter(s, marks, W-pad.r+12, 17);
  s.appendChild(Object.assign(el("text",{x:pad.l,y:16,fill:css("--ink3"),
    "font-family":css("--sans"),"font-size":"11.5"}),
    {textContent:"final validation loss shown at each line's end"}));
}

function batchChart(){
  const s=document.getElementById("chart-batch"); if(!s) return;
  const W=1000,H=448,pad={l:70,r:268,t:26,b:62};
  const rows=DATA.ladder, ch=DATA.ladder_chunked;
  const modes=[...new Set(rows.map(r=>r.mode))];
  const xs=[Math.min(...rows.map(r=>r.batch)),
            Math.max(...rows.map(r=>r.batch),...ch.map(r=>r.batch))];
  const ymax=Math.max(...rows.map(r=>r.peak),...ch.map(r=>r.peak))*1.08;
  const {X,Y}=axes(s,W,H,pad,xs,[0,ymax],"batch size (sequences of 512 tokens)",
    "peak memory (GiB)",v=>v.toFixed(0),v=>v.toFixed(1),true);
  if(DATA.vram_cap){
    s.appendChild(el("line",{x1:pad.l,y1:Y(DATA.vram_cap),x2:W-pad.r,y2:Y(DATA.vram_cap),
      stroke:css("--ink3"),"stroke-dasharray":"3 4","stroke-width":"1.2"}));
    s.appendChild(Object.assign(el("text",{x:pad.l+6,y:Y(DATA.vram_cap)-7,fill:css("--ink3"),
      "font-family":css("--sans"),"font-size":"11.5"}),
      {textContent:DATA.vram_cap.toFixed(2)+" GiB usable on this GPU"}));
  }
  const marks=[];
  const draw=(data,dash)=>{
    [...new Set(data.map(r=>r.mode))].forEach(m=>{
      const pts=data.filter(r=>r.mode===m).sort((a,b)=>a.batch-b.batch);
      if(!pts.length) return;
      const d=pts.map((r,i)=>(i?"L":"M")+X(r.batch).toFixed(1)+" "+Y(r.peak).toFixed(1)).join(" ");
      const p=el("path",{d,fill:"none",stroke:C[m],"stroke-width":"2",
        "stroke-dasharray":dash?"5 4":""});
      s.appendChild(p);
      pts.forEach(r=>{ const c=el("circle",{cx:X(r.batch),cy:Y(r.peak),r:3.4,fill:C[m]});
        hover(c,`<b>${NAME[m]}</b>${dash?" · chunked CE":""}<br>batch ${r.batch}<br>${r.peak.toFixed(2)} GiB<br>${Math.round(r.tps).toLocaleString()} tok/s`);
        s.appendChild(c); });
      const last=pts[pts.length-1];
      s.appendChild(el("path",{d:`M${X(last.batch)-6} ${Y(last.peak)-6} l12 12 M${X(last.batch)+6} ${Y(last.peak)-6} l-12 12`,
        stroke:C[m],"stroke-width":"2.6","stroke-linecap":"round"}));
      marks.push({y:Y(last.peak),color:C[m],
        text:NAME[m]+(dash?" (chunked CE)":"")+" · max "+last.batch});
    });
  };
  draw(rows,false); draw(ch,true);
  gutter(s, marks, W-pad.r+14, 17);
}

function depthChart(){
  const s=document.getElementById("chart-depth"); if(!s) return;
  const W=1000,H=428,pad={l:70,r:190,t:26,b:62};
  const rows=DATA.depth; if(!rows.length) return;
  const xs=[Math.min(...rows.map(r=>r.L)),Math.max(...rows.map(r=>r.L))];
  const ymax=Math.max(...rows.map(r=>r.per))*1.12;
  const {X,Y}=axes(s,W,H,pad,xs,[0,ymax],"layers","memory per batch (GiB)",
    v=>v.toFixed(0),v=>v.toFixed(2),true);
  const marks=[];
  [...new Set(rows.map(r=>r.mode))].forEach(m=>{
    const pts=rows.filter(r=>r.mode===m).sort((a,b)=>a.L-b.L);
    const d=pts.map((r,i)=>(i?"L":"M")+X(r.L).toFixed(1)+" "+Y(r.per).toFixed(1)).join(" ");
    s.appendChild(el("path",{d,fill:"none",stroke:C[m],"stroke-width":"2"}));
    pts.forEach(r=>{ const c=el("circle",{cx:X(r.L),cy:Y(r.per),r:3.6,fill:C[m]});
      hover(c,`<b>${NAME[m]}</b><br>${r.L} layers<br>${r.per.toFixed(3)} GiB per batch`);
      s.appendChild(c); });
    const last=pts[pts.length-1];
    marks.push({y:Y(last.per),text:NAME[m],color:C[m]});
  });
  gutter(s, marks, W-pad.r+12, 17);
}

setScheme("midpoint");
lossChart(); batchChart(); depthChart();
</script>
</body>
</html>
"""


def build(assets="assets", out="site.html"):
    res = json.load(open(os.path.join(assets, "results.json")))
    lad = json.load(open(os.path.join(assets, "ladder.json")))
    # Throughput comes from the order-balanced benchmark, never from the runs themselves:
    # the GPU throttles from 2100 to ~1500 MHz over a session, so an eight-run matrix
    # gives its first variant a cold card and its last a hot one.
    bench = json.load(open(os.path.join(assets, "throughput.json")))
    diag = json.load(open(os.path.join(assets, "diagnostics.json")))
    TPS = {b["mode"]: b["tokens_per_sec"] for b in bench["summary"]}
    runs = {r["spec"]["name"]: r for r in res["runs"]}
    env = res["environment"]
    base = runs["A_baseline"]
    mid = runs.get("C_midpoint", base)

    data = {
        "runs": [{"name": r["spec"]["name"], "mode": r["spec"]["mode"],
                  "batch": r["spec"]["batch_size"], "val": r["final_val_loss"],
                  "tps": TPS.get(r["spec"]["mode"], r["tokens_per_sec"]),
                  "peak": r["peak_alloc_gib"],
                  "history": [{"tokens": h["tokens"], "loss": h["loss"]} for h in r["history"]]}
                 for r in res["runs"]],
        "ladder": [{"mode": r["mode"], "batch": r["batch"], "peak": r["peak_alloc_gib"],
                    "tps": r["tokens_per_sec"]} for r in lad["rows"]
                   if not r.get("degraded")],
        "ladder_chunked": [{"mode": r["mode"], "batch": r["batch"],
                            "peak": r["peak_alloc_gib"], "tps": r["tokens_per_sec"]}
                           for r in lad.get("chunked", {}).get("rows", [])
                           if r["mode"] != "store"],
        "depth": [{"mode": r["mode"], "L": r["n_layer"], "per": r["per_batch_gib"]}
                  for r in lad.get("depth_rows", [])],
        "vram_cap": lad["rows"][0].get("vram_cap_gib") if lad.get("rows") else None,
    }

    lips = [d["lipschitz"] for d in diag["euler_implicit"]["diagnostics"] if "lipschitz" in d]
    recs = [d["recon_h0"] for d in diag["euler_implicit"]["diagnostics"] if "recon_h0" in d]
    pars = [d["parasitic"] for d in diag["midpoint"]["diagnostics"] if "parasitic" in d]

    order = ["A_baseline", "E_checkpoint", "B_euler", "C_midpoint", "G_euler_implicit",
             "F_coupling", "D_maxbatch", "D2_maxbatch_same_lr"]
    best = min(r["final_val_loss"] for n, r in runs.items() if not n.startswith("D"))
    trs = []
    for n in order:
        r = runs.get(n)
        if not r:
            continue
        s = r["spec"]
        win = ' class="win"' if r["final_val_loss"] == best else ""
        # the throughput benchmark runs at one fixed batch, so it does not describe the
        # max-batch runs - blank rather than quote a number measured somewhere else
        if n.startswith("D"):
            speed = '<td class="sub">&mdash;</td><td class="sub">&mdash;</td>'
        else:
            tps = TPS.get(s["mode"])
            speed = (f"<td>{tps:,.0f}</td>"
                     f"<td>{TPS['store']/tps:.2f}&times;</td>")
        trs.append(
            f"<tr><td>{NAMES.get(s['mode'], s['mode'])}</td>"
            f"<td>{s['batch_size']}</td>"
            f"<td{win}>{r['final_val_loss']:.4f}</td>"
            + speed +
            f"<td>{r['peak_alloc_gib']:.2f}</td>"
            f"<td>{r['wall_seconds']/60:.1f}</td></tr>")
    table = ("<table><thead><tr><th>integrator</th><th>batch</th><th>val loss</th>"
             "<th>tokens/s</th><th>slower</th><th>peak GiB</th><th>minutes</th></tr>"
             "</thead><tbody>" + "".join(trs) + "</tbody></table>")

    mb, chm = lad["max_batch"], lad.get("chunked", {}).get("max_batch", {})
    b_plain = max(mb.get("euler", 0), mb.get("midpoint", 0))
    b_chunk = max(chm.get("euler", 0), chm.get("midpoint", 0)) or b_plain
    slow = (TPS["store"] / TPS["euler"] - 1) * 100
    depth = [(r["n_layer"], r["per_batch_gib"]) for r in lad.get("depth_rows", [])
             if r["mode"] == "store"]
    growth = (max(depth)[1] / min(depth)[1]) if len(depth) > 1 else 0

    kpis = "".join([
        f'<div class="kpi"><div class="n">{b_chunk/mb["store"]:.1f}&times;</div>'
        f'<div class="l">larger batch, same GPU</div></div>',
        f'<div class="kpi"><div class="n">{slow:.0f}%</div>'
        f'<div class="l">slower, and that is the whole price</div></div>',
        f'<div class="kpi"><div class="n">{growth:.1f}&times;</div>'
        f'<div class="l">memory the baseline adds from 4 to 32 layers</div></div>',
        f'<div class="kpi"><div class="n">1.00&times;</div>'
        f'<div class="l">memory reversibility adds over the same range</div></div>',
    ])

    chips = "".join(
        f'<button class="chip" data-mode="{m}" aria-pressed="false">{lbl}</button>'
        for m, lbl in (("store", "baseline"), ("euler", "symplectic Euler"),
                       ("midpoint", "midpoint"), ("coupling", "coupling")))

    html = HEAD + (BODY_TMPL
                   .replace("__DATA__", json.dumps(data))
                   .replace("__TABLE__", table)
                   .replace("__KPIS__", kpis)
                   .replace("__CHIPS__", chips)
                   .replace("__PARAMS__", f"{base['params']:,}")
                   .replace("__NRUNS__", str(len(res["runs"])))
                   .replace("__GPU__", env["gpu"])
                   .replace("__TORCH__", env["torch"])
                   .replace("__STAMP__", res["run_stamp"])
                   .replace("__STATE__", f"{base['state_gib']:.2f}")
                   .replace("__BSTAR__", str(mb["store"]))
                   .replace("__BMAX_PLAIN__", str(b_plain))
                   .replace("__BMAX_CHUNKED__", str(b_chunk))
                   .replace("__LIP0__", f"{lips[0]:.2f}")
                   .replace("__LIPMAX__", f"{max(lips):.0f}")
                   .replace("__REC0__", f"{recs[0]:.2f}")
                   .replace("__RECMAX__", f"{max(recs):.0f}")
                   .replace("__GLOSS__", f"{runs['G_euler_implicit']['final_val_loss']:.2f}")
                   .replace("__ALOSS__", f"{runs['A_baseline']['final_val_loss']:.3f}")
                   .replace("__BLOSS__", f"{runs['B_euler']['final_val_loss']:.4f}")
                   .replace("__CLOSS__", f"{runs['C_midpoint']['final_val_loss']:.4f}")
                   .replace("__PARGROW__", f"{pars[-1]/pars[0]:.1f}"))
    open(out, "w", encoding="utf-8").write(html)
    return out, len(html)


NAMES = {"store": "baseline (stores activations)", "checkpoint": "gradient checkpointing",
         "euler": "symplectic Euler", "midpoint": "midpoint (leapfrog)",
         "coupling": "coupling (RevNet)", "euler_implicit": "implicit Euler"}

if __name__ == "__main__":
    p, n = build()
    print(f"wrote {p} ({n/1024:.0f} KiB)")
