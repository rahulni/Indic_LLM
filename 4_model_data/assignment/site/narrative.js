// Authored narrative content. Numbers referenced here live in data.js
// (RESULTS) - this file is prose and structure only.

const STRATEGY_COUNT = 8;

const STRATEGIES = [
  {
    n: 1,
    id: "extract",
    name: "Extract",
    tagline: "Turn raw HTML into clean prose",
    does: "Pull the article body out of a crawled page and drop navigation, cookie banners, and footers around it.",
    v4: "The prior run's extraction was naive HTML stripping that keeps navigation and legal boilerplate as if it were content.",
    ourRun:
      "Two corpora, two different jobs. Sangraha's unverified tier ships already-extracted plain text, so there this is a verification pass that strips residual markup and drops nothing. The SFT corpus contains fenced code blocks, and code legitimately contains angle-bracket sequences the markup stripper would destroy - so code spans and conversation markers are masked before the strip and restored byte-for-byte after. Same stage, different rules, because the corpus decides which rules are safe.",
  },
  {
    n: 2,
    id: "normalize",
    name: "Normalize",
    tagline: "One canonical character-level form - and the ghost-tag fix",
    does: "NFC unicode normalization, strip control/zero-width/BOM/bidi/private-use characters, unescape HTML entities, collapse whitespace - while explicitly preserving ZWNJ/ZWJ, the two invisible characters that carry real meaning in Brahmic scripts. Folded into this stage: the 'ghost-tag trap' - literal conversation-format markers baked into text as ordinary characters.",
    v4: "The prior run had no clean_text() in any of its 6 ingestion scripts, producing 46 garbage vocab tokens - including private-use-area characters - plus ghost [USER] tags baked straight into the pretraining text.",
    ourRun:
      "Real NFC normalization and noise-character stripping over both corpora, cross-checked against a known-good reference implementation rather than prose alone. On the SFT corpus the ghost-tag stage finally has real work: those datasets ship pre-flattened text with literal ChatML markers already in the published text field, and this stage does not merely count them - it rewrites every dialect into one canonical format, which is the prescribed fix. Whitespace collapse is suppressed inside code blocks, because Python indentation is semantics.",
  },
  {
    n: 3,
    id: "langid",
    name: "Language ID",
    tagline: "Detect the language at runtime - don't trust the folder",
    does: "Run a real language identifier on every document and compare it against the label the source claimed, translating between ISO 639-1 and ISO 639-3 codes explicitly rather than assuming they line up.",
    v4: "The prior run trusted the folder name (verified/asm/) with no runtime detection, and Telugu was coded te where the pipeline expected tel, so whole pools were silently misrouted.",
    ourRun:
      "py3langid run on every document in both corpora, mapped through an explicit 639-1→639-3 table - the same code families that te/tel bug lives in. Documents whose detected language contradicts the claimed label are dropped, not silently kept.",
  },
  {
    n: 4,
    id: "quality",
    name: "Quality filter",
    tagline: "Two layers: heuristics, then a real trained classifier",
    does: "A Gopher/C4-style heuristic cascade (word length, symbol ratio, duplicate lines, stopword presence) followed by a trained classifier scoring educational value.",
    v4: "The prior run's selector used an English-heavy proxy that systematically under-valued Indic text, forcing an always-on bypass just to stop it discarding good Indic data.",
    ourRun:
      "Layer 1 runs with a script-appropriate stopword list - Telugu for the web corpus, English for the SFT corpus - rather than one English ruleset applied to everything, which is the exact filter-bias failure this guards against. Layer 2 is a real scikit-learn LogisticRegression trained on 200 documents Claude actually read and labeled, and it runs only on the Telugu corpus: those labels say nothing about English reasoning traces, so on the SFT corpus layer 2 is skipped and reported as skipped rather than run for show.",
  },
  {
    n: 5,
    id: "dedup",
    name: "Deduplicate",
    tagline: "Global MinHash + LSH, not per-shard",
    does: "Shingle every document, summarize each shingle set with a MinHash signature, and use LSH banding to catch near-duplicates without an all-pairs comparison - run once, globally, over the whole surviving pool.",
    v4: "Sangraha, our own Indic web crawl, had ZERO dedup at any level - wasted compute and a real memorization risk once training reaches scale.",
    ourRun:
      "The stage now proves the central claim about global dedup instead of describing it. Every source shard is first deduplicated in isolation, exactly as one person working alone on their own shard would do it, using the identical code. Then the global pass runs over the merged pool, and the gap between the two is the duplication no local pass could ever see. On the Telugu corpus that gap is zero - and a separate threshold sweep down to 0.3 is reported alongside it, because a zero at the operating threshold means nothing unless you can show the detector was capable of finding something. On the SFT corpus the gap is large, because two of its four published sources overlap byte-for-byte.",
  },
  {
    n: 6,
    id: "pii",
    name: "PII scrub",
    tagline: "Regex for the structured stuff, a real NER model for names",
    does: "Regex catches emails, phone numbers, and IPv4 addresses. A real NER model catches personal names, which follow no fixed pattern.",
    v4: "Dolma ran regex PII scrubbing; the Indic pipeline had none, so identifiers passed straight through into the training set.",
    ourRun:
      "Which model runs is a property of the corpus, not a constant: kuppuluri/telugu_bertu_ner for Telugu, dslim/bert-base-NER for English. ai4bharat/IndicNER was the obvious first choice for the Indic side - same publisher as the dataset - but it is gated and a real download attempt returned 401. The curated Telugu name gazetteer runs alongside the model on Telugu and is switched off entirely on English, where it would produce a meaningless zero rather than a measurement. The NER layer reads only the first 1,000 characters of a document, and the widget reports exactly what share of corpus text that leaves unscanned.",
  },
  {
    n: 7,
    id: "decontaminate",
    name: "Decontaminate",
    tagline: "Fingerprint the held-out set, scan for leaks, plant a canary",
    does: "Fingerprint a held-out reference pool at the 10-word-shingle level, scan every training document for overlap, remove any hit, then separately verify the leak-detection mechanism with a real plant-and-detect canary-string test.",
    v4: "The prior run kept a Golden Proxy of the test splits that was never trained on, so decontamination could be verified rather than assumed.",
    ourRun:
      "Each corpus gets a held-out pool that is disjoint by construction: for Telugu, a slice of Sangraha's own verified tier; for the SFT corpus, rows past the end of the training slice. The sampler additionally drops any held-out document whose text also appears in the training pool, so the disjointness is enforced rather than assumed.",
  },
  {
    n: 8,
    id: "manifest",
    name: "Manifest",
    tagline: "Provenance that makes the corpus auditable - and blocks what can't ship",
    does: "Emit one JSON record per shard: source, license, contributor, the cleaning script's own hash, a content hash computed from the cleaned text, a real token count, and the language breakdown. No shard ships without one.",
    v4: "The prior run copy-pasted dataset sizes and generated non-deterministic IDs; its token estimate of words×1.3 was wrong for Indic by 2-10x.",
    ourRun:
      "One manifest per source shard, not one for the whole corpus - which is what makes the license gate meaningful, since it can block one source while the others ship. It does exactly that here: of the four SFT sources, one declares no license at all and two are copyleft, so three of four are marked BLOCKED and their tokens are counted but not shipped. The whole pipeline re-runs in a separate process with a different hash seed and the content hash is diffed.",
  },
];

const STRATEGY_COUNT_FOOTNOTE =
  "The reference pipeline map enumerates \"eight stages\" and names them Extract → Normalize → Language ID → Quality filter → " +
  "Deduplicate → PII scrub → Decontaminate → Manifest, and the prose recap and the source material agree (\"stage one to stage " +
  "eight is one script\"). The \"ghost-tag trap\" gets its own dedicated section and its own interactive widget, but the canonical " +
  "stage list folds that fix into Normalize rather than counting it separately, since both are early character/format-level fixes " +
  "that run before quality filtering. We count 8 to match the authoritative stage list, and say plainly that a finer-grained count " +
  "would land on 9. The source is explicit that eight is a floor, not a ceiling: \"this is a minimal set... this is the minimum " +
  "that you have to do.\"";

const DATASET_STORY = {
  name: "Two corpora, chosen so all eight strategies have real work to do",
  hfUrl: "https://huggingface.co/datasets/ai4bharat/sangraha",
  license: "CC-BY-4.0 (Telugu) · mixed, see manifest (SFT)",
  tier: "sangraha unverified/tel + a 4-source reasoning/SFT mix",
  why: [
    "The Telugu web crawl is Sangraha's unverified tier - the exact corpus the reference pipeline map names as having had ZERO dedup in the prior (V4) run, so running real MinHash/LSH over it is a direct callback rather than a generic exercise.",
    "Telugu is the canonical worked example for two separate stages: the te/tel language-code bug in Language ID, and the Indic quality-filter-bias case in Quality filter. One language lets both worked examples be checked against real data.",
    "The second corpus exists because the first one cannot exercise everything. A Telugu web crawl contains no conversation markers, so the ghost-tag stage had nothing to find; and a single-source corpus cannot demonstrate cross-source deduplication at all. The SFT mix fixes both, with real published data rather than manufactured examples.",
    "The SFT sources come from the same Hugging Face profile the reference example link points at (\"there's fable traces, there's a reasoning corpus, spend some time on this\"). Their -sft variants ship pre-flattened text with literal ChatML markers already in the published text field, which is what makes the ghost tags genuine rather than something our own ingestion script introduced.",
  ],
  notReused:
    "The reference example link (lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled) is a model page, not a dataset; its underlying training data (lordx64/reasoning-distill-claude-opus-4-7-max) is ~8,124 rows of already-pristine reasoning traces. It was explicitly flagged as not to be reused - an example of how to find a dataset, not the answer. The sibling -sft datasets used here are from the same publisher but are not that dataset.",
  alternativesConsidered: [
    {
      name: "Sangraha unverified — Assamese",
      reason:
        "Lets you hunt for the other canonical worked example, an Assamese-labelled document that is really Bengali. A strong pick, set aside only because Telugu covers two worked examples instead of one.",
    },
    {
      name: "wikimedia/wikipedia (te config)",
      reason:
        "Ungated, real, and a known reference point (\"the whole of Wikipedia is 5.23 billion words\"). Kept as the documented fallback in case Sangraha access changed; not needed, since Sangraha was ungated and worked cleanly.",
    },
    {
      name: "MegaScience / open-r1 reasoning sets",
      reason:
        "Inspected as candidates for the second corpus and rejected on evidence: both store conversation structure as separate JSON columns rather than as flattened text, so ghost tags would have had to be manufactured by our own ingestion script. The lordx64 -sft variants ship the markers already baked into the published text, which is the difference between demonstrating a defect and creating one.",
    },
    {
      name: "A larger single Telugu draw, and no second corpus",
      reason:
        "Simpler, and it was the first round's approach. Rejected because three of the eight strategies - Extract, ghost tags, and cross-source dedup - then have literally nothing to act on, and a widget reporting three zeroes is a worse answer to \"what was cleaned\" than a second corpus costs.",
    },
  ],
  heldOutStory:
    "Each corpus needs a held-out reference pool that must never appear in training. For Telugu we draw from a disjoint slice of Sangraha's own verified tier - the role the classic \"Golden Proxy\" pattern plays, and a pool sharing enough real-world sources with the unverified tier that any overlap found is a genuine signal. For the SFT corpus we take rows past the end of the training slice. In both cases the sampler then removes any held-out document whose text also occurs in the training pool, so disjointness is enforced in code rather than argued for in prose.",
  samplingNote:
    "How the sample is drawn is itself a cleaning decision, and the first round got it wrong. A uniform random ~8% draw of a 150k-document shard keeps both halves of any given duplicate pair with probability p² - about 0.6% - so that sample could not have revealed duplicates whether or not they were there. Both corpora are now drawn as contiguous slices, and the draw itself is a hashed pipeline stage (stage0_sample.py) recording offsets, row indices, and each source's license as reported live by the Hugging Face API - so the provenance chain starts at the download rather than at a frozen file with a forgotten recipe. Worth saying plainly: this fix removed a measurement artifact, but it did not change the Telugu result. That slice really does contain no duplicates at any defensible threshold, and the sweep reported in the dedup stage is how we know the difference.",
  tokenizerCaveat:
    "Every token count on this page is cl100k_base, and that choice flatters the Telugu corpus — so it is measured against a real Indic tokenizer rather than hand-waved. cl100k has almost no Telugu merges and encodes Telugu near the byte level, giving 13.27 tokens/word; the identical text under google/muril-base-cased, a WordPiece tokenizer built for Indian scripts, is 2.30 tokens/word. So the Telugu corpus is 46.2M cl100k tokens but only 7.99M MuRIL tokens — a 5.8× gap, and under the honest Indic count it falls just below the 10M-token floor for Telugu alone. It clears the floor in cl100k, and the two corpora together clear it on any tokenizer, but the plain fact is that the token count is a property of the tokenizer, not the data. That is exactly why fertility is reported rather than a token total left to stand on its own.",
};

const OTHER_CONCERNS = [
  {
    title: "The sampling fix was necessary, and it was not sufficient — here is what actually happened",
    body: "The first round drew ~8% of a 150k-document shard uniformly at random and then reported 1 near-duplicate, explaining it as a property of the corpus. That explanation was unsafe: a uniform draw keeps both halves of a duplicate pair with probability p², about 0.6%, so the sample could not have shown duplicates whether or not they existed. Both corpora are now drawn as contiguous slices, which removes that artifact. The honest result is that it changed nothing for Telugu — the redrawn slice still contains zero duplicates at the operating threshold. So rather than assert the corpus is clean, the pipeline sweeps the threshold down and reports the curve: zero at 0.75, zero at 0.6, and only a handful of pairs by 0.5. That flat curve is the evidence that the zero is a fact about the corpus rather than a blind detector. What actually made deduplication measurable was not the sampling fix but the second corpus, where two published sources genuinely overlap. Both facts are reported, including the one that did not go the way it was predicted to.",
  },
  {
    title: "The ghost-tag rewriter can over-reach, and here is exactly where",
    body: "The dialect breakdown counts each rule separately on purpose, because two of them have known rough edges and hiding them in a total would be dishonest. First, the rule that catches a bare <code>USER:</code> at the start of a line: it exists because that pattern genuinely occurs nested inside already-marked ChatML blocks in one source, but a line of quoted transcript beginning <code>USER:</code> would be rewritten too — it is the single largest over-reach risk and would need a tighter guard at scale. Second, the XML-style rule maps a closing tag like <code>&lt;/assistant&gt;</code> to the role marker rather than to the end marker, which is not quite right; it fired only a couple of times across the whole SFT corpus, so its real-world impact here is negligible, but it is a real bug and it is named rather than smoothed over. Both are visible in the per-dialect counts in the Normalize stage.",
  },
  {
    title: "Documents that start identically are not duplicates",
    body: "The same analysis counts documents sharing their first 120 characters, and finds groups the dedup pass deliberately kept. These are templated news leads that diverge after the opening — exactly the case where a cheap prefix check would delete legitimate, distinct articles. Reporting the two measures side by side is the point: the prefix signal looks like duplication and is not, which is why the pipeline uses shingled MinHash similarity over whole documents rather than the cheaper thing.",
  },
  {
    title: "License gating, with teeth",
    body: "Three of the four SFT sources fail the gate: one declares no license at all, and two are AGPL-3.0. Unknown provenance is a blocking condition, and copyleft on training data is a legal decision no pipeline should make silently, so all three are marked BLOCKED - their tokens are counted in the statistics and excluded from what ships. This is why the manifest is per-shard rather than per-corpus: one bad source should cost you that source, not the whole run. The licenses were read live from the Hugging Face API at sampling time, not copied from a README.",
  },
  {
    title: "License strings need normalising before they gate anything",
    body: "The sampler records each source's license as our registry declares it and as the Hugging Face API reports it live, then compares them. For Sangraha those are <code>CC-BY-4.0</code> and <code>cc-by-4.0</code> — the same license, and an exact string comparison calls it a mismatch. The gate itself already lowercases before deciding, so nothing was blocked wrongly, but the informational flag is strict and the widget labels that row \"match (case differs)\" rather than repeating a strict comparison as if it were a finding. It is a small thing that points at a real one: a provenance field is only as good as its normalisation, and \"the strings differ\" is not the same claim as \"the licenses differ\".",
  },
  {
    title: "Determinism, proven in a way that can actually fail",
    body: "The first round re-ran the pipeline twice inside a single process and diffed the content hash. That check was weaker than it looked: both passes shared one PYTHONHASHSEED, so any accidental dependence on set or dict iteration order would have reproduced identically and passed while proving nothing. The second pass now runs in a genuinely separate interpreter with a different, randomly chosen hash seed.",
  },
  {
    title: "The widget-access limitation, checked rather than assumed",
    body: "The source material warns that some strategies live inside interactive widgets, which a plain-text read cannot see. Checked directly: of the 10 widget files behind the reference material, only 3 - the pipeline map, the live text-cleaner, and the ghost-tag demo - contain real saved content. The other 7 are 149-byte stubs where the lazy-loaded iframe never rendered before the page was saved, so that content genuinely is not recoverable from these files. The 3 that exist were read directly rather than through prose about them; the private-use-area fix and the named ghost-tag formats come from there.",
  },
  {
    title: "What the NER layer does not see",
    body: "The name-detection model reads only the first 1,000 characters of each document, which bounds runtime but is a real recall limit. Rather than mention it and move on, the pipeline measures it: the widget reports what share of total corpus text was actually scanned and how many documents run past the window. Regex layers still cover the full document, so structured identifiers - emails, phones, IPs - are unaffected.",
  },
  {
    title: "A stage that refuses to run rather than pretend",
    body: "The trained quality classifier is fit on 200 hand-labelled Telugu web documents. Those labels say nothing about English reasoning traces, so on the SFT corpus layer 2 does not run and is reported as SKIPPED with its reason, rather than being applied to produce a number that would look like a measurement and would not be one. Labelling a genuine sample of that corpus is the honest way to enable it, and that work is not done.",
  },
  {
    title: "Training data the classifier never sees",
    body: "The 200 labelled documents live in a frozen pool drawn separately from the shard, deliberately kept out of the pool the stage filters. The classifier is therefore never fit on a document it later scores, so the corpus-level rejection rate is a genuine out-of-sample number rather than a memorised one.",
  },
  {
    title: "Why the quality filter drops less here than a raw-web funnel would",
    body: "A canonical cleaning funnel on raw Common Crawl shows the corpus falling to about 44% by the quality stage. Ours drops far less, and the reason is the input, not a broken filter: that funnel starts from raw Common Crawl HTML, where boilerplate, SEO walls, and navigation chrome dominate. Sangraha's unverified tier has already been through AI4Bharat's own extraction, and the SFT corpus is frontier-model output. Starting from cleaner input means less to remove - which is a fact about the corpora, and is reported as such rather than tuned until the numbers match a target.",
  },
  {
    title: "Per-snapshot dedup caution",
    body: "A known caution: naive global dedup across multiple Common Crawl snapshots throws away legitimate re-reporting of the same event across years - an article about India written in 2004, 2014, and 2024 are not duplicates. Our Telugu sample is a single crawl snapshot, so the risk does not apply here, but it is why the dedup stage is scoped to within-snapshot content similarity rather than blind cross-time removal, and it is a documented consideration for scaling to a multi-year corpus.",
  },
  {
    title: "The 'marketing vs. verified' honesty check",
    body: "Indic dataset headline token counts routinely collapse once synthetic and machine-translated padding is stripped. In that spirit we drew from Sangraha's unverified tier and called it that, rather than quietly starting from the already-curated verified tier and letting the widget imply the cleaning stages did more work than they did.",
  },
  {
    title: "Not overstating how prominent this corpus is",
    body: "An earlier version of this page claimed this corpus was \"discussed at length\" in the sourcing research it came from. Checking the source, it appears once, in a list of Indic sources alongside IndicCorp, Culture-X, and FineWeb - Wikipedia gets considerably more airtime. The dataset choice stands on its own merits, which are the prior-run zero-dedup callback and the two Telugu worked examples. The overstatement is corrected here rather than left in place.",
  },
];
