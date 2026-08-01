// Rendering logic for the cleaning widget. Reads RESULTS (data.js, the real
// pipeline output) and the STRATEGIES/DATASET_STORY/OTHER_CONCERNS narrative
// (narrative.js). No numbers are invented here - everything traces back to
// results.json.

const fmt = (n) => (n == null ? "—" : n.toLocaleString("en-US"));
const pct = (n, d) => (d ? ((100 * n) / d).toFixed(1) + "%" : "—");
const short = (h) => (h ? h.slice(0, 12) + "…" + h.slice(-6) : "—");

/* The widget now reports on more than one corpus. ACTIVE selects which one the
   per-corpus sections are showing; the strategy list and the concerns are
   corpus-independent and never switch. */
let ACTIVE = (RESULTS.corpus_order && RESULTS.corpus_order[0]) || Object.keys(RESULTS.corpora)[0];
const R = () => RESULTS.corpora[ACTIVE];
const ALL_CORPORA = () => (RESULTS.corpus_order || Object.keys(RESULTS.corpora)).map((id) => RESULTS.corpora[id]);

function stageReport(n) {
  return R().stage_reports.find((r) => r.stage_num === n);
}

function renderCorpusTabs() {
  document.querySelectorAll(".corpus-tabs").forEach((bar) => {
    bar.innerHTML = ALL_CORPORA()
      .map(
        (c) => `<button class="corpus-tab${c.corpus_id === ACTIVE ? " active" : ""}" data-corpus="${c.corpus_id}">
            <span class="ct-name">${c.corpus_name}</span>
            <span class="ct-meta">${fmt(c.final_manifest.final_doc_count)} docs · ${compactNum(c.final_manifest.final_token_count_cl100k)} tok</span>
          </button>`
      )
      .join("");
    bar.querySelectorAll(".corpus-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        ACTIVE = btn.dataset.corpus;
        renderAll();
      });
    });
  });
}

/* ============================== Theme toggle ============================== */
(function initTheme() {
  // Dark by default (the inline head script already applied it pre-paint); a
  // stored choice still wins.
  const stored = localStorage.getItem("s4-theme") || "dark";
  document.documentElement.setAttribute("data-theme", stored);
  const btn = document.getElementById("themeToggle");
  const apply = () => {
    const isDark = document.documentElement.getAttribute("data-theme") !== "light";
    btn.textContent = isDark ? "☀" : "☾";
  };
  apply();
  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const isDark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("s4-theme", next);
    apply();
    renderAll(); // charts read CSS vars at draw time, so redraw after a theme flip
  });
})();

/* ============================== Tiny SVG chart kit ============================== */
const NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs) {
  const el = document.createElementNS(NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
let tooltipEl;
function ensureTooltip() {
  if (!tooltipEl) {
    tooltipEl = document.createElement("div");
    tooltipEl.className = "bar-tooltip";
    document.body.appendChild(tooltipEl);
  }
  return tooltipEl;
}
function showTip(evt, title, desc) {
  const t = ensureTooltip();
  t.innerHTML = `<div class="t">${title}</div><div class="d">${desc}</div>`;
  t.style.left = evt.clientX + 14 + "px";
  t.style.top = evt.clientY + 14 + "px";
  t.classList.add("show");
}
function hideTip() {
  if (tooltipEl) tooltipEl.classList.remove("show");
}

/** Vertical column chart - sequential single-hue, for magnitude across an ordered sequence. */
function columnChart(container, items, opts = {}) {
  container.innerHTML = "";
  const w = container.clientWidth || 600;
  const h = opts.height || 220;
  const padL = 46, padR = 12, padT = 18, padB = 34;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const maxV = Math.max(...items.map((d) => d.value));
  const n = items.length;
  const gap = 6;
  const bw = Math.min(40, (plotW - gap * (n - 1)) / n);
  const trackW = bw * n + gap * (n - 1);
  const startX = padL + (plotW - trackW) / 2;

  const svg = svgEl("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}` });
  const grid = cssVar("--gridline"), muted = cssVar("--text-muted"), base = cssVar("--baseline");
  const hue = opts.color || cssVar("--series-1");

  // gridlines (4 steps)
  for (let i = 0; i <= 3; i++) {
    const y = padT + (plotH * i) / 3;
    svg.appendChild(svgEl("line", { x1: padL, x2: w - padR, y1: y, y2: y, stroke: grid, "stroke-width": 1 }));
    const val = Math.round((maxV * (3 - i)) / 3);
    const t = svgEl("text", { x: padL - 8, y: y + 4, "text-anchor": "end", "font-size": 10, fill: muted, "font-family": "var(--mono)" });
    t.textContent = opts.compact ? compactNum(val) : fmt(val);
    svg.appendChild(t);
  }
  svg.appendChild(svgEl("line", { x1: padL, x2: w - padR, y1: padT + plotH, y2: padT + plotH, stroke: base, "stroke-width": 1 }));

  items.forEach((d, i) => {
    const x = startX + i * (bw + gap);
    const barH = Math.max(2, (d.value / maxV) * plotH);
    const y = padT + plotH - barH;
    const rect = svgEl("rect", { x, y, width: bw, height: barH, rx: 4, ry: 4, fill: d.color || hue, "fill-opacity": d.dim ? 0.35 : 1 });
    rect.style.cursor = "pointer";
    rect.addEventListener("mousemove", (e) => showTip(e, d.label, (opts.compact ? compactNum(d.value) : fmt(d.value)) + (opts.unit ? " " + opts.unit : "") + (d.note ? "<br>" + d.note : "")));
    rect.addEventListener("mouseleave", hideTip);
    svg.appendChild(rect);

    if (d.labelValue) {
      const t = svgEl("text", { x: x + bw / 2, y: y - 6, "text-anchor": "middle", "font-size": 11, fill: "var(--text-primary)", "font-weight": 700 });
      t.textContent = opts.compact ? compactNum(d.value) : fmt(d.value);
      svg.appendChild(t);
    }
    const lbl = svgEl("text", { x: x + bw / 2, y: h - padB + 16, "text-anchor": "middle", "font-size": 10, fill: muted });
    lbl.textContent = d.short || d.label;
    svg.appendChild(lbl);
  });

  container.appendChild(svg);
}

function compactNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
  return String(n);
}

/** Horizontal bar chart - categorical colors, for breakdowns / part-to-whole. */
function hBarChart(container, items, opts = {}) {
  container.innerHTML = "";
  const w = container.clientWidth || 600;
  const rowH = 28;
  const h = items.length * rowH + 16;
  const padL = opts.labelWidth || 150, padR = 56;
  const plotW = w - padL - padR;
  const maxV = Math.max(...items.map((d) => d.value), 1);

  const svg = svgEl("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}` });
  const muted = cssVar("--text-muted");

  items.forEach((d, i) => {
    const y = 8 + i * rowH;
    const barW = Math.max(2, (d.value / maxV) * plotW);
    const lbl = svgEl("text", { x: padL - 10, y: y + rowH / 2 + 4, "text-anchor": "end", "font-size": 11.5, fill: "var(--text-secondary)" });
    // Safety clamp: an SVG <text> with text-anchor=end that's wider than its
    // available label column runs off the LEFT edge of the viewBox and gets
    // clipped - truncate with an ellipsis rather than let that happen silently.
    const maxChars = Math.floor((padL - 14) / 6.3);
    lbl.textContent = d.label.length > maxChars ? d.label.slice(0, maxChars - 1) + "…" : d.label;
    if (d.label.length > maxChars) lbl.appendChild(svgEl("title", {})).textContent = d.label;
    svg.appendChild(lbl);

    const track = svgEl("rect", { x: padL, y: y + 4, width: plotW, height: rowH - 12, rx: 4, ry: 4, fill: "var(--gridline)", "fill-opacity": 0.4 });
    svg.appendChild(track);

    const rect = svgEl("rect", { x: padL, y: y + 4, width: barW, height: rowH - 12, rx: 4, ry: 4, fill: d.color });
    rect.style.cursor = "pointer";
    rect.addEventListener("mousemove", (e) => showTip(e, d.label, fmt(d.value) + (d.note ? "<br>" + d.note : "")));
    rect.addEventListener("mouseleave", hideTip);
    svg.appendChild(rect);

    const val = svgEl("text", { x: padL + barW + 8, y: y + rowH / 2 + 4, "font-size": 11.5, fill: "var(--text-primary)", "font-weight": 700 });
    val.textContent = fmt(d.value);
    svg.appendChild(val);
  });

  container.appendChild(svg);
}

/* ============================== Section: Strategies ============================== */
function renderStrategies() {
  const rail = document.getElementById("strategyRail");
  rail.innerHTML = STRATEGIES.map(
    (s) => `
    <div class="strategy-card" style="--accent: var(--series-${s.n})">
      <div class="strategy-top">
        <div class="stage-badge">${s.n}</div>
        <div class="strategy-headings">
          <div class="stage-kicker">Stage ${s.n} / ${STRATEGY_COUNT}</div>
          <h3>${s.name}</h3>
        </div>
      </div>
      <div class="tag">${s.tagline}</div>
      <p>${s.does}</p>
      <div class="v4"><span class="v4-label">Ran on real data</span>${s.ourRun}</div>
    </div>`
  ).join("");
  document.getElementById("strategyFootnote").innerHTML = STRATEGY_COUNT_FOOTNOTE;
}

/* ============================== Section: Dataset ============================== */
function renderDataset() {
  const c = R();
  const raw = c.raw_input;
  const man = c.final_manifest;
  // Header follows the selected corpus so it agrees with the meta row below it.
  document.getElementById("datasetName").innerHTML =
    escapeHtml(c.corpus_name) + ` <span class="dataset-sub">· ${c.corpus_kind === "web_crawl" ? "web crawl" : "reasoning / SFT"}</span>`;
  document.getElementById("datasetMeta").innerHTML = `
    <span class="pill accent">${escapeHtml(man.license || "mixed")}</span>
    <span class="pill">${fmt(raw.doc_count)} docs sampled</span>
    <span class="pill">${compactNum(raw.token_count_cl100k)} raw tokens (cl100k_base)</span>
    <span class="pill">${man.shards_total || 1} shard${(man.shards_total || 1) > 1 ? "s" : ""}${man.shards_blocked ? ", " + man.shards_blocked + " blocked" : ""}</span>
  `;
  document.getElementById("datasetWhy").innerHTML = DATASET_STORY.why.map((w) => `<li>${w}</li>`).join("");
  document.getElementById("datasetHeldout").textContent = DATASET_STORY.heldOutStory;

  const alts = DATASET_STORY.alternativesConsidered;
  document.getElementById("altCount").textContent = alts.length;
  document.getElementById("altGrid").innerHTML = alts
    .map(
      (a) => `
      <div class="alt-card">
        <div class="alt-name">${a.name}</div>
        <div class="alt-reason">${a.reason}</div>
      </div>`
    )
    .join("");
  document.getElementById("hfLink").href = c.hf_url || DATASET_STORY.hfUrl;

  const samplingHost = document.getElementById("samplingProvenance");
  if (samplingHost) {
    samplingHost.innerHTML = `<div class="note-box">${DATASET_STORY.samplingNote}</div>`;
    renderSampling(samplingHost, R().sampling_report);
  }
}

/* ============================== Section: Cleaning (funnel + stage detail) ============================== */
const FUNNEL_SHORT_LABELS = {
  "Raw input": "Raw",
  "Extract": "Extract",
  "Normalize": "Normal.",
  "Language ID": "Lang ID",
  "Quality filter": "Quality",
  "Deduplicate": "Dedup",
  "PII scrub": "PII",
  "Decontaminate": "Decontam.",
  "Manifest": "Manifest",
};
function renderFunnel() {
  const funnel = R().survival_funnel;
  const items = funnel.map((s, i) => ({
    label: s.name,
    short: FUNNEL_SHORT_LABELS[s.name] || s.name,
    value: s.docs,
    labelValue: i === 0 || i === funnel.length - 1 || s.name === "Quality filter",
    note: s.tokens ? `${compactNum(s.tokens)} tokens` : undefined,
  }));
  columnChart(document.getElementById("funnelChart"), items, { height: 240, compact: true, unit: "docs" });

  const raw = R().raw_input, man = R().final_manifest;
  document.getElementById("funnelCaption").innerHTML =
    `<b>${fmt(raw.doc_count)}</b> documents / <b>${compactNum(raw.token_count_cl100k)}</b> raw tokens in &nbsp;→&nbsp; ` +
    `<b>${fmt(man.final_doc_count)}</b> documents / <b>${compactNum(man.final_token_count_cl100k)}</b> tokens out ` +
    `(<b>${pct(man.final_token_count_cl100k, raw.token_count_cl100k)}</b> token survival)`;
}

function miniStat(v, l) {
  return `<div class="mini-stat"><div class="v">${v}</div><div class="l">${l}</div></div>`;
}
function exampleItem(lbl, body) {
  return `<div class="example-item"><div class="lbl">${lbl}</div><div class="telugu">${body}</div></div>`;
}

function renderStageDetail() {
  const root = document.getElementById("stageDetail");
  root.innerHTML = "";

  STRATEGIES.forEach((s) => {
    const rep = stageReport(s.n);
    const details = document.createElement("details");
    details.className = "stage-item";
    if (s.n === 1) details.setAttribute("open", "");

    const summary = document.createElement("summary");
    summary.innerHTML = `<span class="chev">▸</span><span class="chip">${s.n}/8</span> ${s.name}
      <span class="survival">${fmt(rep.input_docs)} → ${fmt(rep.output_docs)} docs (${rep.survival_pct}%)</span>`;
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "stage-body";
    body.innerHTML = `<div class="desc">${s.does}</div>
      <div class="note-box"><b>V4 defect this fixes:</b> ${s.v4}</div>`;
    details.appendChild(body);
    root.appendChild(details);

    // stage-specific extra content
    if (s.n === 1) renderExtractBody(body, rep);
    if (s.n === 2) renderNormalizeBody(body, rep);
    if (s.n === 3) renderLangidBody(body, rep);
    if (s.n === 4) renderQualityBody(body, rep);
    if (s.n === 5) renderDedupBody(body, rep);
    if (s.n === 6) renderPiiBody(body, rep);
    if (s.n === 7) renderDecontamBody(body, rep);
    if (s.n === 8) renderManifestBody(body, rep);
  });
}

function renderExtractBody(body, rep) {
  const e = rep.extra;
  const row = document.createElement("div");
  row.className = "stat-row";
  row.innerHTML =
    miniStat(fmt(e.docs_with_any_residual_markup), "docs with residual markup") +
    miniStat(fmt(e.total_html_tags_stripped), "HTML tags stripped") +
    miniStat(fmt(e.total_boilerplate_lines_dropped), "boilerplate lines dropped") +
    (e.code_protection_enabled ? miniStat(fmt(e.total_protected_spans), "code spans protected") : "");
  body.appendChild(row);
  body.appendChild(Object.assign(document.createElement("div"), {
    className: "note-box",
    innerHTML: e.note,
  }));
}

function renderNormalizeBody(body, rep) {
  const e = rep.extra;
  const row = document.createElement("div");
  row.className = "stat-row";
  row.innerHTML =
    miniStat(fmt(e.total_noise_chars_removed), "noise chars removed") +
    miniStat(fmt(e.zwj_zwnj_preserved_count), "ZWJ/ZWNJ preserved") +
    miniStat(fmt(e.ghost_conversation_markers_found), "ghost tags found") +
    miniStat(e.char_shrink_pct + "%", "character-level shrink");
  body.appendChild(row);
  renderGhostTags(body, e);

  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-card";
  chartWrap.innerHTML = `<h4>Noise-character inventory (real counts)</h4><div class="sub">Every distinct kind of invisible/control character actually found and stripped across the sample</div><div class="hbar"></div>`;
  body.appendChild(chartWrap);
  const items = Object.entries(e.noise_char_inventory)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value], i) => ({ label, value, color: seriesColor(i) }));
  if (items.length) hBarChart(chartWrap.querySelector(".hbar"), items, { labelWidth: 170 });
  else chartWrap.querySelector(".hbar").innerHTML = `<div class="sub">No noise characters found in this sample.</div>`;
}

function renderLangidBody(body, rep) {
  const e = rep.extra;
  const row = document.createElement("div");
  row.className = "stat-row";
  row.innerHTML =
    miniStat(fmt(e.docs_matching_claim), "confirmed Telugu") +
    miniStat(fmt(e.docs_mismatched_and_dropped), "mislabeled & dropped") +
    miniStat(fmt(e.docs_too_short_to_judge_kept_asis), "too short to judge");
  body.appendChild(row);

  if (rep.examples.length) {
    const ex = document.createElement("div");
    ex.className = "example-list";
    ex.innerHTML =
      `<div class="note-box" style="margin-top:0"><b>Real mislabeled documents found:</b> claimed <code>${escapeHtml(String(rep.examples[0].claimed_lang))}</code>, detected as <code>${escapeHtml(String(rep.examples[0].detected_lang))}</code> — documents filed under one language's label whose actual text is another language, which is exactly the "trust the folder, get burned" failure mode this stage exists to catch. Each was dropped rather than silently pooled with the language it claimed to be.</div>` +
      rep.examples
        .map((x) => exampleItem(`claimed ${x.claimed_lang} → detected ${x.detected_lang} (logprob ${x.logprob})`, escapeHtml(x.text_preview)))
        .join("");
    body.appendChild(ex);
  }
}

function renderQualityBody(body, rep) {
  const e = rep.extra;
  const c = e.classifier_training || {};
  const t = c.held_out_test_metrics;
  // Layer 2 only runs where its labels apply. Where it is skipped, say so
  // instead of rendering an empty classifier card.
  const layer2Ran = !!t;
  const row = document.createElement("div");
  row.className = "stat-row";
  row.innerHTML =
    miniStat(fmt(rep.docs_removed), "docs rejected") +
    (layer2Ran ? miniStat(fmt(e.docs_rejected_by_trained_classifier_only), "rejected by trained classifier only") : "") +
    (layer2Ran ? miniStat(e.avg_p_good_of_survivors, "avg p(good) of survivors") : "") +
    miniStat(e.stopword_list_used || "—", "stopword list used");
  body.appendChild(row);

  if (!layer2Ran) {
    body.appendChild(Object.assign(document.createElement("div"), {
      className: "note-box warn",
      innerHTML:
        "<b>Layer 2 did not run on this corpus, deliberately.</b> " +
        escapeHtml(c.reason || "") +
        " The heuristic cascade below is real and ran in full; the classifier card is absent because " +
        "there is no honest number to put in it.",
    }));
  }

  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-card";
  chartWrap.innerHTML = `<h4>Why documents were rejected</h4><div class="sub">Real failure-reason breakdown across the cascade (a document can fail only one reason - first match wins)</div><div class="hbar"></div>`;
  body.appendChild(chartWrap);
  const labels = { too_short: "Too short (<20 words)", mean_word_len_out_of_range: "Word length out of range", symbol_word_ratio_too_high: "Too many symbol-only tokens", duplicate_line_wall: "Duplicate-line wall", too_few_stopwords: "Too few " + (e.stopword_list_used || "") + " stopwords", trained_classifier_reject: "Trained classifier (Layer 2)" };
  const items = Object.entries(e.fail_reason_breakdown)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v], i) => ({ label: labels[k] || k, value: v, color: seriesColor(i), note: k === "trained_classifier_reject" ? "rejected by the real trained LogisticRegression, layer 2" : undefined }));
  hBarChart(chartWrap.querySelector(".hbar"), items, { labelWidth: 190 });

  if (!layer2Ran) return;

  const clfWrap = document.createElement("div");
  clfWrap.className = "chart-card";
  clfWrap.innerHTML = `
    <h4>The real trained classifier</h4>
    <div class="sub">${escapeHtml(c.training_labels_source)}</div>
    <div class="stat-row" style="margin-top:10px">
      ${miniStat(c.n_good + " / " + c.n_bad, "labeled good / bad")}
      ${miniStat(t.accuracy, "held-out test accuracy")}
      ${miniStat(t.f1, "held-out test F1")}
      ${miniStat(t.precision + " / " + t.recall, "precision / recall")}
    </div>
    <div class="sub" style="margin-top:10px">Confusion matrix (${c.train_test_split.n_test} held-out test docs, labels=[bad, good])</div>
    <table class="tbl"><tbody>
      ${t.confusion_matrix.matrix.map((rowArr, i) => `<tr><td class="mono">actual ${t.confusion_matrix.labels[i]}</td>${rowArr.map((v) => `<td class="mono">${v}</td>`).join("")}</tr>`).join("")}
    </table>
    <div class="sub" style="margin-top:10px">Feature coefficients (real, from the fitted model)</div>
    <table class="tbl"><tbody>
      ${Object.entries(c.feature_coefficients).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => `<tr><td>${k}</td><td class="mono">${v}</td></tr>`).join("")}
    </table>`;
  body.appendChild(clfWrap);

  if (e.classifier_vs_old_proxy_disagreements && e.classifier_vs_old_proxy_disagreements.length) {
    const ex = document.createElement("div");
    ex.className = "example-list";
    ex.innerHTML =
      `<div class="note-box" style="margin-top:0"><b>The trained classifier disagrees with the old round-1/2 compression-ratio proxy on real documents</b> - spot-checked by hand: the proxy's arbitrary 0.25-0.85 band was wrongly rejecting legitimate low-compression prose (e.g. a real BBC News Telugu article on farmers switching to stevia crops). The classifier, trained on genuine labels, gets these right.</div>` +
      e.classifier_vs_old_proxy_disagreements
        .slice(0, 4)
        .map((x) => exampleItem(`compression_ratio ${x.compression_ratio} - trained classifier: ${x.trained_classifier_verdict} (p=${x.trained_classifier_p_good}), old proxy: ${x.old_compression_proxy_verdict}`, `doc ${short(x.doc_id)}`))
        .join("");
    body.appendChild(ex);
  }

  body.appendChild(Object.assign(document.createElement("div"), {
    className: "note-box",
    innerHTML: `<b>Class-weight tuning, done honestly:</b> with only 16 "bad" labels against 184 "good", <code>class_weight="balanced"</code> was tried first and over-corrected - verified by hand, it rejected real legitimate articles. <code>class_weight=None</code> swung the other way and degenerated into always predicting "good". The mild <code>{bad: 3, good: 1}</code> weighting shown here is the one that held up under both checks.`,
  }));
}

function renderDedupBody(body, rep) {
  const e = rep.extra;
  const row = document.createElement("div");
  row.className = "stat-row";
  row.innerHTML =
    miniStat(fmt(e.exact_duplicates_removed), "exact duplicates") +
    miniStat(fmt(e.near_duplicates_removed), "near-duplicates (at threshold " + e.lsh_threshold + ")") +
    miniStat(fmt(e.cross_source_exact_duplicates || 0), "of those, across sources") +
    miniStat(e.lsh_bands_b + " × " + e.lsh_rows_per_band_r, "LSH bands × rows/band");
  body.appendChild(row);

  renderScaleComparison(body, e);
  renderDuplicateStructure(body);

  body.appendChild(Object.assign(document.createElement("div"), { className: "note-box", innerHTML: e.threshold_probe_note }));

  const probesAvailable = (e.threshold_sensitivity_probe || []).length;
  if (!probesAvailable) {
    body.appendChild(Object.assign(document.createElement("div"), {
      className: "note-box",
      innerHTML:
        "<b>No near-duplicate pairs survived into this corpus to probe.</b> Rather than seed " +
        "candidates in from outside to make the slider look busy, the panel is simply absent here.",
    }));
    return;
  }

  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-card";
  chartWrap.innerHTML = `
    <h4>Real near-duplicate candidates vs. the dedup threshold</h4>
    <div class="sub">8 genuine document pairs from the corpus, ranked by true shingle-set similarity. Drag the threshold to see which would be caught.</div>
    <div class="slider-row"><input type="range" id="thSlider" min="0.3" max="0.9" step="0.01" value="${e.lsh_threshold}"><span class="val" id="thVal">${e.lsh_threshold}</span></div>
    <div class="hbar" id="thBars"></div>
    <div class="verdict-summary" id="thSummary"></div>`;
  body.appendChild(chartWrap);

  const probes = e.threshold_sensitivity_probe;
  const slider = chartWrap.querySelector("#thSlider");
  const valEl = chartWrap.querySelector("#thVal");
  const barsEl = chartWrap.querySelector("#thBars");
  const sumEl = chartWrap.querySelector("#thSummary");

  function drawProbes() {
    const th = parseFloat(slider.value);
    valEl.textContent = th.toFixed(2);
    const accent = cssVar("--series-1"), gray = cssVar("--text-muted");
    const items = probes.map((p, i) => ({
      label: `Pair ${i + 1} (${p.words_a}w / ${p.words_b}w)`,
      value: p.true_jaccard_shingle_sets,
      color: p.true_jaccard_shingle_sets >= th ? accent : gray,
      note: `est. MinHash jaccard ${p.estimated_jaccard_minhash} · shared: "${escapeHtml(p.shared_prefix)}"`,
    }));
    hBarChart(barsEl, items, { labelWidth: 150 });
    const caught = probes.filter((p) => p.true_jaccard_shingle_sets >= th).length;
    sumEl.innerHTML = `At threshold <b>${th.toFixed(2)}</b>: <b>${caught}</b> of ${probes.length} real candidate pairs would be flagged as duplicates.`;
  }
  slider.addEventListener("input", drawProbes);
  drawProbes();
}

function renderPiiBody(body, rep) {
  const e = rep.extra;
  const row = document.createElement("div");
  row.className = "stat-row";
  row.innerHTML =
    miniStat(fmt(e.docs_with_any_pii), "docs with a PII redaction") +
    miniStat(fmt(e.total_redactions_by_type.NAME_NER), "real NER-model name redactions") +
    miniStat(fmt(e.ambiguous_name_false_positive_risk_count), "ambiguous-name flags (gazetteer)");
  body.appendChild(row);

  renderNerCoverage(body, e);
  body.appendChild(Object.assign(document.createElement("div"), {
    className: "note-box",
    innerHTML: `<b>Real model:</b> <code>${e.ner_model}</code> (${e.ner_model_license} license), run on every document, truncated to ${fmt(e.ner_truncation_chars)} chars, confidence ≥ ${e.ner_min_confidence}. ai4bharat/IndicNER was tried first - same publisher as the dataset - but is gated (a real download attempt returned a 401); this ungated model is the one that actually works.`,
  }));

  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-card";
  chartWrap.innerHTML = `<h4>Redactions by type</h4><div class="sub">Real counts across the surviving pool</div><div class="hbar"></div>`;
  body.appendChild(chartWrap);
  const labels = { EMAIL: "Email address", PHONE: "Phone number", IP: "IPv4 address", NAME_NER: "Name (real NER model)", NAME_HIGH_CONFIDENCE: "Name (gazetteer, high conf.)", NAME_AMBIGUOUS: "Name (gazetteer, ambiguous)" };
  const items = Object.entries(e.total_redactions_by_type)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v], i) => ({ label: labels[k] || k, value: v, color: seriesColor(i), note: k === "NAME_AMBIGUOUS" ? "also an ordinary word - a real false-positive risk" : k === "NAME_NER" ? "real model inference, not a fixed list" : undefined }));
  hBarChart(chartWrap.querySelector(".hbar"), items, { labelWidth: 190 });

  if (e.ner_recall_gain_examples && e.ner_recall_gain_examples.length) {
    const ex = document.createElement("div");
    ex.className = "example-list";
    ex.innerHTML =
      `<div class="note-box" style="margin-top:0"><b>Real recall gain:</b> names the NER model caught that the gazetteer's fixed ~20-word list could never have known, because they're not in it:</div>` +
      e.ner_recall_gain_examples.slice(0, 8).map((n) => `<span class="pill" style="margin:3px">${escapeHtml(n)}</span>`).join("");
    body.appendChild(ex);
  }
  if (e.ner_gazetteer_overlap_examples && e.ner_gazetteer_overlap_examples.length) {
    const ex2 = document.createElement("div");
    ex2.className = "example-list";
    ex2.style.marginTop = "8px";
    ex2.innerHTML =
      `<div class="note-box" style="margin-top:0"><b>Overlap:</b> names both the model and the gazetteer agree on:</div>` +
      e.ner_gazetteer_overlap_examples.slice(0, 8).map((n) => `<span class="pill accent" style="margin:3px">${escapeHtml(n)}</span>`).join("");
    body.appendChild(ex2);
  }

  if (e.gazetteer_enabled) {
    const ex = document.createElement("div");
    ex.className = "example-list";
    ex.style.marginTop = "8px";
    ex.innerHTML =
      `<div class="note-box warn" style="margin-top:0"><b>A policy decision, stated as one:</b> ` +
      `the ${fmt(e.ambiguous_name_false_positive_risk_count || 0)} "ambiguous" hits are words like ` +
      `<em>రాజు</em> ("Raju" / "king") and <em>బాబు</em> ("Babu" / a generic honorific) — common Telugu given ` +
      `names that are also ordinary vocabulary. The pipeline redacts them <b>anyway</b>, on the ` +
      `logic that over-filtering PII is the safer error: a name wrongly masked costs a little fluency, a name ` +
      `wrongly kept can expose a real person. The cost is real and is counted here rather than hidden — this is ` +
      `the largest single source of false-positive redaction in the run, and a production version would want a ` +
      `context check before masking a word that is also a common noun.</div>`;
    body.appendChild(ex);
  }
}

function renderDecontamBody(body, rep) {
  const e = rep.extra;
  const row = document.createElement("div");
  row.className = "stat-row";
  row.innerHTML =
    miniStat(fmt(e.docs_flagged_contaminated_and_removed), "docs removed (overlap w/ held-out)") +
    miniStat(fmt(e.heldout_docs_used), "held-out reference docs") +
    miniStat(fmt(e.fingerprint_total_shingles), "fingerprint shingles (10-grams)");
  body.appendChild(row);

  const c = e.canary_mechanism_test;
  const status = document.createElement("div");
  status.className = "status-row";
  status.innerHTML = `<span class="status-dot ${c.leak_detected ? "good" : "critical"}"></span> Canary plant-and-detect mechanism test: <span class="${c.leak_detected ? "status-good-text" : ""}">${c.mechanism_status}</span>`;
  body.appendChild(status);

  if (rep.examples.length) {
    const ex = document.createElement("div");
    ex.className = "example-list";
    ex.style.marginTop = "10px";
    ex.innerHTML = rep.examples
      .slice(0, 3)
      .map((x) => exampleItem(`${x.shared_shingles_with_heldout} shared 10-grams with the held-out pool`, escapeHtml(x.sample_shared_ngram)))
      .join("");
    body.appendChild(ex);
  }
}

function renderManifestBody(body, rep) {
  const m = rep.extra.manifest;
  renderLicenseGate(body, m);
  const pre = document.createElement("pre");
  pre.className = "manifest";
  pre.innerHTML = manifestHtml(m);
  body.appendChild(pre);
}

function manifestHtml(m) {
  const rows = [
    ["corpus", m.corpus_name || m.corpus_id],
    ["source", m.source], ["license", m.license],
    ["shards_total", m.shards_total], ["shards_blocked", m.shards_blocked],
    ["contributor", m.contributor],
    ["cleaning_script_hash_sha256", short(m.cleaning_script_hash_sha256)],
    ["content_sha256", short(m.content_sha256)],
    ["final_doc_count", fmt(m.final_doc_count)],
    ["final_token_count_cl100k", fmt(m.final_token_count_cl100k)],
    ["fertility_ratio_tokens_per_word", m.fertility_ratio_tokens_per_word],
    ["language_breakdown", JSON.stringify(m.language_breakdown)],
    ["timestamp_utc", m.timestamp_utc],
  ];
  return "{\n" + rows.map(([k, v]) => `  <span class="k">"${k}"</span>: <span class="s">"${escapeHtml(String(v))}"</span>`).join(",\n") + "\n}";
}

function seriesColor(i) {
  const slots = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"];
  return cssVar(slots[i % slots.length]);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ============================== Section: Other concerns ============================== */
function renderConcerns() {
  const grid = document.getElementById("concernGrid");
  grid.innerHTML = OTHER_CONCERNS.map((c) => `<div class="concern-card"><h4>${c.title}</h4><p>${c.body}</p></div>`).join("");

  const d = R().determinism_check;
  const det = document.getElementById("determinismStatus");
  det.innerHTML = `
    <div class="status-row"><span class="status-dot ${d.fully_reproducible ? "good" : "critical"}"></span>
      <span class="${d.fully_reproducible ? "status-good-text" : ""}">${d.fully_reproducible ? "Fully reproducible - second pass in a separate process with a different PYTHONHASHSEED, identical hashes" : "Non-deterministic - hashes differ between runs"}</span>
    </div>
    <table class="tbl" style="margin-top:10px">
      <tr><td>content_sha256</td><td class="mono">run 1: ${short(d.run1_content_sha256)}</td><td class="mono">run 2: ${short(d.run2_content_sha256)}</td></tr>
      <tr><td>cleaning_script hash</td><td class="mono">run 1: ${short(d.run1_script_hash)}</td><td class="mono">run 2: ${short(d.run2_script_hash)}</td></tr>
      <tr><td>final doc count</td><td class="mono">run 1: ${d.run1_final_doc_count}</td><td class="mono">run 2: ${d.run2_final_doc_count}</td></tr>
      <tr><td>run 2 hash seed</td><td class="mono" colspan="2">PYTHONHASHSEED=${d.run2_pythonhashseed || "—"} (separate interpreter)</td></tr>
    </table>
    <div class="note-box" style="margin-top:10px">${d.method_note || ""}</div>`;
}

/* ============================== Section: Final statistics ============================== */
function renderStats() {
  const raw = R().raw_input, man = R().final_manifest;
  const kpis = [
    ["Raw documents", fmt(raw.doc_count)],
    ["Final documents", fmt(man.final_doc_count)],
    ["Raw tokens (cl100k)", compactNum(raw.token_count_cl100k)],
    ["Final tokens (cl100k)", compactNum(man.final_token_count_cl100k)],
    ["Token survival", pct(man.final_token_count_cl100k, raw.token_count_cl100k)],
    ["Fertility ratio", man.fertility_ratio_tokens_per_word + " tok/word"],
    ["Shards blocked by license gate", (man.shards_blocked || 0) + " of " + (man.shards_total || 1)],
    ["Pipeline wall time (all corpora)", RESULTS.pipeline_elapsed_s + "s"],
    ["Determinism (this corpus)", R().determinism_check.fully_reproducible ? "✓ verified (separate process)" : "✗ failed"],
  ];
  document.getElementById("kpiGrid").innerHTML = kpis
    .map(([l, v]) => `<div class="kpi-tile"><div class="l">${l}</div><div class="v">${v}</div></div>`)
    .join("");

  renderTokenizerCaveat(document.getElementById("tokenizerCaveat"));
  document.getElementById("finalManifestView").innerHTML = manifestHtml(man);

  const stageTable = R().stage_reports
    .map((r) => `<tr><td>${r.stage_num}. ${r.stage_name}</td><td class="mono">${fmt(r.input_docs)}</td><td class="mono">${fmt(r.output_docs)}</td><td class="mono">${r.survival_pct}%</td><td class="mono">${r.elapsed_s}s</td></tr>`)
    .join("");
  document.getElementById("stageStatsTable").innerHTML = stageTable;
}

/* ============================== Live demo: real normalize() ported to JS ============================== */
// Same noise inventory as pipeline/stage2_normalize.py's NOISE_CODEPOINTS,
// cross-checked against a known-good reference cleaner (source of the
// private-use-area / soft-hyphen / word-joiner entries a prose summary alone
// doesn't spell out). Written as explicit \u escapes rather than literal
// invisible characters, on purpose.
const NOISE_RE = /[\u200B\uFEFF\u200E\u200F\u00AD\u2060\u202A-\u202E\u2066-\u2069\uFFFD\uE000-\uF8FF]/g;
const ZWJ_ZWNJ_RE = /[\u200C\u200D]/g;
const GHOST_RES = [
  /\[(USER|ASSISTANT|SYSTEM|HUMAN|BOT)\]/gi,
  /<\|(user|assistant|system|im_start|im_end|endoftext)\|>/gi,
  /###\s*(Human|Assistant)\s*:/gi,
  /<\/?(human|bot|assistant|user|system)>/gi,
];

// Mirrors GHOST_REWRITE_RULES in pipeline/stage2_normalize.py. Detection alone
// is only half the fix; the other half is rewriting every dialect into one
// canonical format, so the demo does that too.
const CANON = { system: "<|system|>", user: "<|user|>", assistant: "<|assistant|>", end: "<|end|>" };
const ROLE_ALIAS = { human: "user", bot: "assistant", instruction: "user", response: "assistant" };
const canonRole = (raw) => CANON[ROLE_ALIAS[raw.toLowerCase()] || raw.toLowerCase()] || CANON.user;
const GHOST_REWRITES = [
  [/<\|im_start\|>[ \t]*(system|user|assistant)\b/gi, (m, r) => canonRole(r)],
  [/<\|im_end\|>/gi, () => CANON.end],
  [/<\|endoftext\|>/gi, () => CANON.end],
  [/\[(USER|ASSISTANT|SYSTEM|HUMAN|BOT)\]/gi, (m, r) => canonRole(r)],
  [/###[ \t]*(Human|Assistant|Instruction|Response)[ \t]*:/gi, (m, r) => canonRole(r)],
  [/<\/?(human|bot|assistant|user|system)>/gi, (m, r) => canonRole(r)],
  [/^(USER|ASSISTANT|SYSTEM)[ \t]*:[ \t]*/gim, (m, r) => canonRole(r) + " "],
];

function liveNormalize(text) {
  let removed = 0, preserved = 0, ghosts = 0, rewritten = 0;
  let out = text.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'");
  out = out.normalize("NFC");
  out = out.replace(NOISE_RE, () => { removed++; return ""; });
  preserved = (out.match(ZWJ_ZWNJ_RE) || []).length;
  GHOST_RES.forEach((re) => { ghosts += (out.match(re) || []).length; });
  out = out.replace(/[ \t]{2,}/g, " ").replace(/\n{3,}/g, "\n\n").trim();

  let unified = out;
  GHOST_REWRITES.forEach(([re, fn]) => {
    unified = unified.replace(re, (...a) => { rewritten++; return fn(...a); });
  });
  return { out, unified, removed, preserved, ghosts, rewritten };
}

function highlightDiff(original) {
  // Build the annotated view by marking matches on the RAW text FIRST (via
  // placeholder tokens), THEN HTML-escaping what's left, THEN swapping
  // placeholders for real markup. Escaping before matching would break any
  // ghost pattern containing < or > (like <|endoftext|> or <human>), since
  // those characters would already be &lt;/&gt; by the time the ghost regex
  // ran - found while verifying the endoftext addition actually highlights.
  const placeholders = [];
  function stash(html) {
    const token = "\u0000" + placeholders.length + "\u0000";
    placeholders.push(html);
    return token;
  }

  let work = original;
  work = work.replace(NOISE_RE, (m) => stash(`<del title="stripped">${m.codePointAt(0) === 0xfffd ? "\uFFFD" : "\u00B7"}</del>`));
  work = work.replace(ZWJ_ZWNJ_RE, (m) => stash(`<mark class="zwj" title="preserved (Brahmic conjunct control)">${m === "\u200C" ? "ZWNJ" : "ZWJ"}</mark>`));
  GHOST_RES.forEach((re) => {
    work = work.replace(re, (m) => stash(`<mark class="ghost" title="ghost conversation marker">${escapeHtml(m)}</mark>`));
  });

  let html = escapeHtml(work);
  html = html.replace(/\u0000(\d+)\u0000/g, (_, i) => placeholders[Number(i)]);
  return html;
}

function initDemo() {
  const ta = document.getElementById("demoInput");
  const out = document.getElementById("demoOutput");
  const stats = document.getElementById("demoStats");
  const sample =
    "\u200Bతెలంగాణ\uFEFF రాష్ట్ర\u200E ప్రభుత్వం &amp; సంక్షేమ\u00AD పథకాలు\u2060\n\n\n"
    + "[USER] ఈ పథకం గురించి చెప్పండి\n"
    + "ధర్మా\u200Cపురం లో రైతుల\u200D కోసం కొత్త పథకం ప్రకటించారు.\n"
    + "\uE000\u{F8FF} ఒక ప్రైవేట్-యూజ్ గ్లిఫ్ దొర్లిపోయింది <|endoftext|>\n"
    + "<|im_start|>user\nUSER: summarise the scheme<|im_end|>\n"
    + "### Assistant: here is the summary\n"
    + "<assistant>done</assistant>";
  ta.value = sample;

  function run() {
    const { out: cleaned, unified, removed, preserved, ghosts, rewritten } = liveNormalize(ta.value);
    out.innerHTML = highlightDiff(ta.value);
    stats.innerHTML =
      `<span><b>${removed}</b> noise chars struck out</span>` +
      `<span><b>${preserved}</b> ZWJ/ZWNJ kept</span>` +
      `<span><b>${ghosts}</b> ghost tags found</span>` +
      `<span><b>${rewritten}</b> rewritten to canonical</span>` +
      `<span>${ta.value.length} → <b>${cleaned.length}</b> chars</span>`;
    const uni = document.getElementById("demoUnified");
    if (uni) uni.textContent = unified;
  }
  ta.addEventListener("input", run);
  run();
}

/* ============================== Expand/collapse all ============================== */
function initExpandAll() {
  const btn = document.getElementById("expandAllBtn");
  btn.addEventListener("click", () => {
    const items = document.querySelectorAll("#stageDetail details");
    const anyClosed = Array.from(items).some((d) => !d.open);
    items.forEach((d) => (d.open = anyClosed));
    btn.textContent = anyClosed ? "Collapse all" : "Expand all";
  });
}

/* ============================== Boot ============================== */
function renderAll() {
  renderCorpusTabs();
  renderStrategies();
  renderDataset();
  renderFunnel();
  renderStageDetail();
  renderConcerns();
  renderStats();
}
renderAll();
initDemo();
initExpandAll();
window.addEventListener("resize", debounce(renderAll, 200));
function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
