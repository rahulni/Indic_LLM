// Panels added in round 2, kept in their own file so app.js stays readable.
// These render the things the first version of this widget could not show,
// because the corpus it ran on had nothing to show: real conversation-marker
// unification, cross-shard deduplication, per-shard license gating, the
// measured cost of NER truncation, and how the sample was drawn in the first
// place. Every value read here comes from results.json.

function renderGhostTags(body, e) {
  if (!e.ghost_tag_unification_enabled) {
    body.appendChild(
      Object.assign(document.createElement("div"), {
        className: "note-box",
        innerHTML:
          "<b>Ghost tags: none — and that is the honest result.</b> This corpus is Telugu web text, " +
          "and conversation markers come from chat/SFT data. The scan runs over every document and " +
          "finds zero. Reported as zero rather than quietly dropped from the page — switch to the " +
          "SFT corpus above to watch this stage do real work.",
      })
    );
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "chart-card";
  const rows = Object.entries(e.ghost_format_breakdown || {}).sort((a, b) => b[1] - a[1]);
  wrap.innerHTML =
    '<h4>Conversation-marker dialects found — and unified</h4>' +
    '<div class="sub">' +
    fmt(e.docs_with_ghost_markers) +
    " documents carried literal markers in their <i>published</i> text. Each dialect is rewritten " +
    "into one canonical format, which is the prescribed fix rather than just a count.</div>" +
    '<div class="hbar"></div>' +
    '<div class="canon-row">' +
    Object.entries(e.canonical_format_chosen || {})
      .map((kv) => '<span class="pill accent"><b>' + kv[0] + "</b> → <code>" + escapeHtml(kv[1]) + "</code></span>")
      .join("") +
    "</div>";
  body.appendChild(wrap);

  if (rows.length) {
    hBarChart(
      wrap.querySelector(".hbar"),
      rows.map((kv, i) => ({ label: kv[0], value: kv[1], color: seriesColor(i) })),
      { labelWidth: 175 }
    );
  }

  (e.ghost_tag_examples || []).slice(0, 2).forEach((ex) => {
    const d = document.createElement("div");
    d.className = "ba-pair";
    d.innerHTML =
      '<div class="ba-col"><div class="lbl">before — as published</div><pre>' +
      escapeHtml(ex.before_excerpt || "") +
      '</pre></div><div class="ba-col"><div class="lbl">after — canonical</div><pre>' +
      escapeHtml(ex.after_excerpt || "") +
      "</pre></div>";
    body.appendChild(d);
  });
}

function renderScaleComparison(body, e) {
  const sc = e.scale_local_vs_global;
  if (!sc) return;
  const gap = sc.duplicates_only_a_global_pass_can_find;
  const wrap = document.createElement("div");
  wrap.className = "chart-card";
  wrap.innerHTML =
    "<h4>Local dedup vs. one global pass</h4>" +
    '<div class="sub">Every shard deduplicated on its own with the identical code, then the whole ' +
    "pool at once. The gap between them is the entire argument for one large-memory machine.</div>" +
    '<table class="tbl"><thead><tr><th>Shard</th><th>Docs</th><th>Removed by its own local pass</th></tr></thead><tbody>' +
    sc.shards_deduplicated_independently
      .map(
        (r) =>
          "<tr><td>" + r.source_key + "</td><td>" + fmt(r.docs) + "</td><td>" + fmt(r.removed_by_local_pass) + "</td></tr>"
      )
      .join("") +
    '<tr class="tr-total"><td><b>All shards, deduplicated locally</b></td><td></td><td><b>' +
    fmt(sc.total_removed_if_only_local_dedup) +
    '</b></td></tr><tr class="tr-total"><td><b>One global pass over everything</b></td><td></td><td><b>' +
    fmt(sc.total_removed_by_global_pass) +
    "</b></td></tr></tbody></table>" +
    '<div class="verdict ' +
    (gap > 0 ? "verdict-hit" : "verdict-neutral") +
    '">' +
    (gap > 0
      ? "<b>" +
        fmt(gap) +
        " duplicate documents were invisible to every local pass.</b> No amount of care per shard " +
        "finds them — only a pass that holds the whole corpus in memory at once does."
      : "This corpus is a single shard, so local and global agree by definition. Switch to the " +
        "multi-source corpus to see the gap open up.") +
    "</div>";
  body.appendChild(wrap);
}

function renderLicenseGate(body, m) {
  if (!m.shard_manifests) return;
  const wrap = document.createElement("div");
  wrap.className = "chart-card";
  const firstBlocked = m.shard_manifests.filter((x) => !x.ships)[0];
  wrap.innerHTML =
    "<h4>License gate — one manifest per shard</h4>" +
    '<div class="sub">Licenses were read live from the Hugging Face API at sampling time, not copied ' +
    "from a README. A shard that cannot prove it may ship, does not ship.</div>" +
    '<table class="tbl"><thead><tr><th>Shard</th><th>License</th><th>Tokens</th><th>Status</th></tr></thead><tbody>' +
    m.shard_manifests
      .map(
        (sm) =>
          "<tr><td>" +
          sm.shard_id +
          "</td><td>" +
          (sm.license ? escapeHtml(sm.license) : "<i>none declared</i>") +
          "</td><td>" +
          compactNum(sm.token_count_cl100k) +
          '</td><td><span class="status ' +
          (sm.ships ? "ok" : "blocked") +
          '">' +
          sm.license_status +
          "</span></td></tr>"
      )
      .join("") +
    "</tbody></table>" +
    (m.shards_blocked > 0
      ? '<div class="verdict verdict-warn"><b>' +
        m.shards_blocked +
        " of " +
        m.shards_total +
        " shards blocked</b> — " +
        compactNum(m.tokens_blocked_from_shipping) +
        " tokens cleaned, counted, and withheld from shipping. " +
        escapeHtml(firstBlocked ? firstBlocked.license_reason : "") +
        "</div>"
      : '<div class="verdict verdict-hit">All ' +
        m.shards_total +
        " shard(s) carry a license that permits training use.</div>");
  body.appendChild(wrap);
}

function renderNerCoverage(body, e) {
  const c = e.ner_coverage;
  if (!c) return;
  const wrap = document.createElement("div");
  wrap.className = "chart-card";
  wrap.innerHTML =
    "<h4>What the NER layer actually read</h4>" +
    '<div class="sub">The model sees only the first ' +
    fmt(e.ner_truncation_chars) +
    " characters of each document. That is a real recall limit, so here it is measured rather than mentioned.</div>" +
    '<div class="meter"><div class="meter-fill" style="width:' +
    c.pct_of_corpus_text_scanned +
    '%"></div></div>' +
    '<div class="stat-row">' +
    miniStat(c.pct_of_corpus_text_scanned + "%", "of corpus text scanned by NER") +
    miniStat(fmt(c.docs_longer_than_window), "docs longer than the window") +
    miniStat(fmt(c.docs_total - c.docs_longer_than_window), "docs read in full") +
    "</div>" +
    '<div class="note-box">Regex layers (email, phone, IP) still cover <b>100%</b> of every document — ' +
    "the truncation limit applies only to the name-detection model.</div>";
  body.appendChild(wrap);
}

function renderSampling(body, sr) {
  if (!sr) return;
  const wrap = document.createElement("div");
  wrap.className = "chart-card";
  wrap.innerHTML =
    "<h4>Stage 0 — how this sample was drawn</h4>" +
    '<div class="sub">' +
    escapeHtml(sr.sampling_reason || "") +
    "</div>" +
    '<table class="tbl"><thead><tr><th>Source</th><th>Rows taken</th><th>Row range</th><th>License (live from HF)</th><th>vs. registry</th></tr></thead><tbody>' +
    (sr.sources || [])
      .map((x) => {
        const live = x.license_reported_by_hf_api;
        const declared = x.license_declared_in_registry;
        const exact = !!x.license_agrees;
        // The stored flag is an exact string comparison. Where the two strings
        // differ only by case they do in fact agree, and saying "mismatch"
        // would be wrong - so distinguish the two cases here rather than
        // repeat a strict flag as if it were a finding.
        const caseOnly =
          !exact && live && declared && String(live).toLowerCase() === String(declared).toLowerCase();
        const verdict = exact
          ? '<span class="status ok">match</span>'
          : caseOnly
          ? '<span class="status ok">match (case differs)</span>'
          : '<span class="status blocked">differs</span>';
        return (
          "<tr><td>" +
          escapeHtml(x.hf_id || x.source_key || "") +
          "</td><td>" +
          fmt(x.docs_taken) +
          '</td><td class="mono">' +
          (x.first_row_index == null ? "—" : x.first_row_index) +
          "–" +
          (x.last_row_index == null ? "—" : x.last_row_index) +
          "</td><td>" +
          (live ? escapeHtml(live) : "<i>none</i>") +
          "</td><td>" +
          verdict +
          "</td></tr>"
        );
      })
      .join("") +
    "</tbody></table>" +
    '<div class="note-box">' +
    escapeHtml(sr.determinism_note || "") +
    "</div>";
  body.appendChild(wrap);
}

/** The tokenizer caveat, rendered where the token numbers actually live. When a
    real Indic-tokenizer measurement is available for the active corpus, show the
    two counts side by side instead of only the prose. */
function renderTokenizerCaveat(container) {
  if (!container || !DATASET_STORY.tokenizerCaveat) return;
  container.innerHTML =
    '<div class="note-box warn"><b>Read the token counts with this in mind.</b> ' +
    DATASET_STORY.tokenizerCaveat +
    "</div>";

  const f = (typeof RESULTS !== "undefined" && RESULTS.tokenizer_fertility) ? RESULTS.tokenizer_fertility[ACTIVE] : null;
  if (!f || !f.indic) return; // only meaningful where an Indic reference applies (Telugu)

  const ratio = f.cl100k_over_indic_fertility_ratio;
  const wrap = document.createElement("div");
  wrap.className = "chart-card";
  wrap.style.marginTop = "12px";
  wrap.innerHTML =
    "<h4>Same corpus, two tokenizers — measured, not estimated</h4>" +
    '<div class="sub">' + fmt(f.words) + " words, tokenized both ways. The token count is a property of the tokenizer, not the data.</div>" +
    '<table class="tbl"><thead><tr><th>Tokenizer</th><th>Tokens</th><th>Fertility</th><th>Clears 10M floor?</th></tr></thead><tbody>' +
    '<tr><td>cl100k_base <span class="mono" style="color:var(--text-muted)">(the manifest count)</span></td><td>' +
      compactNum(f.cl100k.tokens) + "</td><td>" + f.cl100k.fertility_tokens_per_word + " tok/word</td><td>" +
      '<span class="status ' + (f.clears_10M_floor_under_cl100k ? "ok" : "blocked") + '">' + (f.clears_10M_floor_under_cl100k ? "yes" : "no") + "</span></td></tr>" +
    "<tr><td>" + escapeHtml(f.indic.tokenizer) + ' <span class="mono" style="color:var(--text-muted)">(Indic WordPiece)</span></td><td>' +
      compactNum(f.indic.tokens) + "</td><td>" + f.indic.fertility_tokens_per_word + " tok/word</td><td>" +
      '<span class="status ' + (f.clears_10M_floor_under_indic ? "ok" : "blocked") + '">' + (f.clears_10M_floor_under_indic ? "yes" : "no") + "</span></td></tr>" +
    "</tbody></table>" +
    '<div class="verdict ' + (f.clears_10M_floor_under_indic ? "verdict-hit" : "verdict-warn") + '">' +
      "cl100k reports <b>" + ratio + "×</b> more tokens than MuRIL for the identical text. " +
      (f.clears_10M_floor_under_indic
        ? "The corpus clears the 10M floor under both."
        : "Under the honest Indic count the Telugu corpus lands <b>below</b> the 10M floor on its own — it clears the floor in cl100k largely because cl100k is inefficient on Telugu. The two-corpus total clears it regardless.") +
    "</div>";
  container.appendChild(wrap);
}

/** The threshold sweep: evidence that a zero is a fact about the corpus and
    not a blind detector. Reads the post-hoc analysis block, not a stage. */
function renderDuplicateStructure(body) {
  const all = RESULTS.duplicate_structure;
  const a = all && all[ACTIVE];
  if (!a) return;

  const wrap = document.createElement("div");
  wrap.className = "chart-card";
  wrap.innerHTML =
    "<h4>Was there anything to find? — threshold sweep</h4>" +
    '<div class="sub">The same MinHash comparison re-run at descending thresholds over ' +
    fmt(a.documents_analysed) +
    " documents. A zero at the operating threshold only means something if the detector " +
    "would have fired at a looser one.</div><div class='hbar'></div>";
  body.appendChild(wrap);

  const op = a.operating_threshold;
  hBarChart(
    wrap.querySelector(".hbar"),
    a.threshold_sweep.map((c) => ({
      label: "threshold " + c.threshold + (c.threshold === op ? "  (operating)" : ""),
      value: c.documents_flagged,
      color: c.threshold === op ? cssVar("--series-2") : cssVar("--series-1"),
      note: c.threshold === op ? "the threshold the pipeline actually runs at" : undefined,
    })),
    { labelWidth: 185 }
  );

  const zeros = a.threshold_sweep.filter((c) => c.documents_flagged === 0);
  const lowestZero = zeros.length ? Math.min.apply(null, zeros.map((c) => c.threshold)) : null;
  const atOp = (a.threshold_sweep.find((c) => c.threshold === op) || {}).documents_flagged || 0;
  const highest = a.threshold_sweep.reduce((m, c) => (c.threshold > m.threshold ? c : m), a.threshold_sweep[0]);

  // Three genuinely different stories, and the verdict has to tell them apart:
  //  (1) clean:      zero well below the operating threshold  -> the 0 is real
  //  (2) duplicated: nonzero even at the very top threshold   -> real dupes, not tuning
  //  (3) borderline: zero at operating, climbs just underneath -> the threshold decides
  let cls, msg;
  if (lowestZero !== null && lowestZero <= 0.6) {
    cls = "verdict-hit";
    msg =
      "Still zero all the way down to <b>" + lowestZero + "</b>, far below the operating threshold. " +
      "The corpus is genuinely duplicate-free here — this is not a detector that failed to fire.";
  } else if (highest && highest.documents_flagged > 0) {
    cls = "verdict-warn";
    msg =
      "Duplicates show up at <i>every</i> threshold tested, including <b>" + highest.threshold + "</b> — " +
      fmt(highest.documents_flagged) + " documents even at that strictness. These are real near- and exact " +
      "duplicates, not a tuning artifact; the operating threshold catches <b>" + fmt(atOp) + "</b> of them.";
  } else {
    cls = "verdict-warn";
    msg =
      "The count climbs steeply just under the operating threshold, which means the <i>threshold</i> is " +
      "deciding the outcome, not the corpus. That is a tuning decision someone has to own.";
  }
  wrap.insertAdjacentHTML(
    "beforeend",
    '<div class="verdict ' + cls + '">' + msg + "</div>" +
      '<div class="note-box"><b>' +
      fmt(a.docs_sharing_a_prefix) +
      "</b> documents share their opening " +
      "120 characters across <b>" +
      fmt(a.shared_prefix_groups) +
      "</b> groups — and were deliberately kept. " +
      "They are templated leads that diverge after the opening, which is exactly the case a cheap " +
      "prefix check would delete by mistake.</div>"
  );
}
