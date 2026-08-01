# Plan — ERA V5 Session 4 assignment

The assignment asks five questions and implies a sixth. This is the plan for
answering all of them with real code and real numbers, plus the plan for the
second round of work that closed the gaps a review found in round one.

## The assignment, restated

1. How many strategies are listed in Session 4, and what are they?
2. Find a 10–100M token dataset (not the example link), ideally one that
   connects back to Session 3.
3. Apply the cleanups. Say what was cleaned, why, and how.
4. Any other strategy or concern cleaned up?
5. Final statistics.
6. (Implied) Ship it as a widget with a real UI.

## Round 1 — the approach

**Q1 — count the strategies.** Do not guess from the transcript alone. The
lesson ships its own pipeline-map widget; read it directly. It says "eight
stages" and names them. Answer 8, cite the source, and disclose that a
finer-grained count lands on 9 if the ghost-tag trap is split out of Normalize.

**Q2 — pick a dataset.** Constraints: 10–100M tokens, not the example link
(the instructor said explicitly it is an example of *how to find* a dataset,
not the answer), and preferably a Session 3 dataset. AI4Bharat Sangraha's
`unverified/tel` tier fits: it is named in Session 3's list of Indic sources,
it is the corpus the course's own audit says had zero deduplication, and
Telugu is the lesson's worked example for two separate stages (the `te`/`tel`
language-code bug and Indic quality-filter bias).

**Q3 — apply the cleanups.** Write all 8 stages as real Python. No simulation:
real MinHash/LSH via `datasketch`, a real trained classifier via `scikit-learn`,
a real NER model via `transformers`, real language ID via `py3langid`. Each
stage writes a survivors file and a JSON report. The reports are the only
source of truth the widget is allowed to read.

**Q4/Q5 — concerns and statistics.** Run the pipeline twice and diff the
content hash to prove determinism rather than assert it. Report the survival
funnel honestly, including stages that removed nothing.

**Deliverable.** A static, self-contained site generated from `results.json`.

## Round 2 — what the review found, and the plan to fix it

An external review of round 1 confirmed the strategy count, the honesty of the
disclosures, and that the pipeline was genuinely real. It raised seven issues.
All are addressed here except the Netlify deploy, which was descoped.

### The substantive problem: three of eight strategies had no work to do

Round 1 used a **uniform random ~8% sample** of one 150k-document Sangraha
shard. That choice destroys duplicate pairs by construction — if you sample 8%
of documents independently, the odds of drawing *both* halves of a specific
duplicate pair are about 0.6%. So deduplication, the session's headline topic,
found 1 near-duplicate in 11,461 documents. Ghost tags found 0, because a
Telugu web crawl contains no conversation markers. Extract dropped 0, because
Sangraha's unverified tier ships pre-extracted plain text.

Three of the eight strategies were therefore demonstrated on paper only.

**Fix 1 — change how the Telugu sample is drawn.** Take a *contiguous* slice of
the shard instead of a uniform random draw, so any duplicate pairs that exist in
the shard survive into the sample together rather than being split apart by the
draw.

> **Outcome (after running):** this fixed the *method* but did not change the
> *result* — the contiguous Telugu slice still contains zero duplicates at the
> operating threshold (and down to 0.6 in the reported sweep). That prediction
> was wrong, and it is reported as wrong: the widget shows the threshold sweep as
> evidence the zero is real, and what actually made deduplication *measurable*
> was Fix 2's second corpus, where two published sources overlap and the global
> pass removes 656 duplicates that per-shard passes (15 total) never see.

**Fix 2 — add a second corpus, chosen so the dark stages light up.** The
assignment's example link points at the `lordx64` Hugging Face profile, and the
instructor said to spend time browsing it. Its SFT variants ship
**pre-flattened text with literal `<|im_start|>` / `<|im_end|>` ChatML markers
already in the stored text** — genuine ghost tags in a published corpus,
nothing manufactured. Verified before committing to the design:

- `fable-sft-combined-v2` and `agentic-distill-fable-5-sft` share **100 of 100**
  sampled rows byte-identically — the "combined" set contains the other. This
  is a real cross-source exact-duplicate case, which is exactly the lesson's
  global-deduplication scenario: two shards each clean on their own, still
  duplicated against each other.
- 6–7 literal conversation markers per row across the reasoning sets.
- Licenses genuinely differ: Apache-2.0, AGPL-3.0 (×2), and **one with no
  declared license at all** — which the lesson's manifest gating rule says must
  block the shard.

So corpus B is a four-source reasoning/SFT mix from that profile, contiguous
slices, ~10–15M tokens. It exercises ghost tags, cross-source dedup, license
gating, and the extract stage, none of which corpus A could reach.

### The correctness and provenance fixes

**Fix 3 — the sampler becomes a hashed stage 0.** Round 1's raw sample was
frozen with no script recording how it was drawn. For a session about
reproducibility and script hashes, the provenance chain had a hole at step
zero. `stage0_sample.py` now performs the draw with a recorded seed, offsets,
and shard identifiers, and its hash goes into the manifest with the others.

**Fix 4 — the determinism check runs in a fresh subprocess.** Round 1 ran both
passes inside one process, so both shared a `PYTHONHASHSEED`. Any latent
set-iteration-order dependence would have passed spuriously. Run 2 now spawns a
subprocess with a *different* randomized hash seed, which is a real test.

**Fix 5 — per-shard manifests and license gating.** The lesson asks for a
manifest per shard carrying the source, license, and every cleaning script's
hash, and for unknown-license shards to be blocked. Round 1 emitted one global
manifest. Now each source shard gets its own, with a `blocked` status when the
license is missing or unsafe.

**Fix 6 — stage 6 measures what it misses, and batches.** NER truncates at
1,000 characters; round 1 disclosed that but never quantified it. It now
reports the share of corpus text left unscanned. Inference is batched, because
the pipeline now runs four times (two corpora × two determinism passes)
instead of twice. Corpus B routes to an English NER model rather than the
Telugu one.

**Fix 7 — the widget stops overstating and starts caveating.**
- The token count is `cl100k_base`, whose Telugu fertility is 13.2 tokens/word.
  The same corpus under an Indic-appropriate tokenizer would be roughly 7–9M
  tokens, *below* the assignment's floor. Say so.
- "Session 3 discussed Sangraha at length" is false — it appears once, in a
  list. Soften to what the transcript actually supports.
- The quality filter dropped 8% where the lesson's own funnel drops to 44%.
  That is because Sangraha's unverified tier is not raw Common Crawl HTML.
  Explain it rather than leave the discrepancy hanging.

## Order of work

1. `corpora.py` registry; stages become corpus-aware.
2. `stage0_sample.py`; re-draw Telugu contiguous, download corpus B.
3. Stage 6 batching + English NER routing + truncation coverage.
4. Stage 8 per-shard manifests + license gating.
5. `run_all.py`: loop corpora, subprocess determinism run.
6. Execute. Regenerate `results.json`.
7. Rebuild `data.js`, correct `narrative.js`, extend `app.js` to two corpora.

## What stays honest

Extract remains a verification pass for corpus A — Sangraha's unverified tier
genuinely has no HTML to strip, and pretending otherwise would be theatre.
Corpus B does give it real markup to handle. Where a stage finds nothing, the
widget says it found nothing.
