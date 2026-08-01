# Session 5 — Data Mixture & Curriculum

![budget](https://img.shields.io/badge/budget-3.00T%20tokens-3B4CC0?style=flat-square&labelColor=1E2230)
![data gate](https://img.shields.io/badge/data%20gate-93.0M%20cleaned-0ca30c?style=flat-square&labelColor=1E2230)
![lanes](https://img.shields.io/badge/lanes-9-2a78d6?style=flat-square&labelColor=1E2230)
![invariants](https://img.shields.io/badge/invariants-all%20passing-0ca30c?style=flat-square&labelColor=1E2230)
![proxy](https://img.shields.io/badge/proxy-3%20arms%20%C3%97%202%20seeds%20run-0ca30c?style=flat-square&labelColor=1E2230)
![deps](https://img.shields.io/badge/dependencies-stdlib%20only-6C7385?style=flat-square&labelColor=1E2230)

The deliverable is **[`MIXTURE_PLAN.md`](MIXTURE_PLAN.md)** — the V5 mixture-and-curriculum
specification for Drishtikon-40B.

It is generated, not written. Every number in it comes out of the three files in
[`plan/`](plan/), and the build refuses to emit the document if the arithmetic
does not close.

> [!TIP]
> **Read the plan as a rendered page** →
> [rahulni.github.io/Indic_LLM/5_data_mixture_curriculum/site.html](https://rahulni.github.io/Indic_LLM/5_data_mixture_curriculum/site.html)
>
> Sticky contents, full-text search, cross-links, and figures drawn from the same
> audit data. GitHub always shows `.html` as source rather than rendering it, so
> [`site.html`](site.html) needs a host — GitHub Pages above, a local
> double-click, or
> [raw.githack.com](https://raw.githack.com/rahulni/Indic_LLM/main/5_data_mixture_curriculum/site.html)
> with no setup at all.

<!-- SCORECARD:START -->

## The plan at a glance

Each row is a decision the plan makes, the number it lands on, and the section that defends it.

| Decision | Where it lands | Defended in |
|---|---|---|
| **Budget share per capability lane** | **9 lanes**, summing to 100.00% of 3.00T — general 52.2% · code 23.6% · reasoning 7.5% · indic 5.0%, … | [§4](MIXTURE_PLAN.md#4-the-mixture) |
| **Indic tier split** — verified / unverified / translated / synthetic | **88.9% / 11.1% / 0.0% / 0.0%** of a 211.8B lane. 213.5B of real Indic tokens are available and **declined** | [§7](MIXTURE_PLAN.md#7-the-indic-slot-split-four-ways) |
| **Agentic, reasoning, long-context**, against named inventory datasets | 2.00% / 7.47% / 2.85%, each with a per-dataset table (tokens, provenance tag, dedup keep, epoch cap) | [§8](MIXTURE_PLAN.md#8-agentic-reasoning-long-context--named-against-the-inventory) |
| **Protected always-on floor** the selector may not cross | **7.00%** of every 1000-step window (indic 3.00% · longctx 1.50% · reasoning 1.00%, …), asserted below each lane's share | [§9](MIXTURE_PLAN.md#9-protected-always-on-floor) |
| **Anneal reserve** held back for the cooldown | **3% = 90.0B**, verified tiers only, and supply-audited against its own eligibility rules | [§10](MIXTURE_PLAN.md#10-the-anneal-reserve-and-schedule-mechanics) |
| **Difficulty bands**, one worked example each | 4 bands — D1 (MBPP task 2), D2 (GSM8K train, item 1), D3 (SWE-bench Verified, django__django-11099), D4 (Terminal-Bench / SWE-bench Live-Pro class) | [§11.1](MIXTURE_PLAN.md#111-difficulty-bands) |
| **Reasoning-length bands**, one worked example each | 4 bands — `<think:low>` ≤128 tok, `<think:medium>` ≤1,024 tok, `<think:high>` ≤8,192 tok, `<think:ultra>` ≤65,536 tok, each with a worked trace and answer | [§11.2](MIXTURE_PLAN.md#112-reasoning-length-bands) |
| **Benchmark accountability** per lane | Benchmark column on every lane — SWE-bench, τ-bench, BFCL, LiveCodeBench, AIME, MILU, FLORES, RULER, MMLU-Pro | [§4](MIXTURE_PLAN.md#4-the-mixture) |
| **Supply sizing** — where a share needs repeating or generating | `SUPPLY-OK` / `REPEAT` (4) / `GENERATE` (2) per lane, with the manufactured share costed in dollars and wall-clock | [§5.3](MIXTURE_PLAN.md#53-supply-against-demand), [§6](MIXTURE_PLAN.md#6-manufacturing-plan) |
| **Proxy design** at 1B / 3B, and the metric that refutes the mixture | 7 arms at 1B × 20.0B + a 3B confirmation, **$6,320** (0.42% of the run), with 9 metrics and **pre-registered** decision rules | [§13](MIXTURE_PLAN.md#13-the-proxy-experiment) |
| **Proxy execution** | ⚠️ **Partial.** The 1B/3B screen was not run — no GPUs. An 11M-param micro-proxy was: 3 arms × 2 seeds, both verdicts INCONCLUSIVE, and the reason is the finding — **both pre-registered rules were underpowered** | [§13.7](MIXTURE_PLAN.md#137-the-micro-proxy-what-was-actually-run) |
| **Data gate**, cleaning aimed at the starved tier | **93,019,466 cleaned tokens** (1.73× session 4). The +39.2M added is Sangraha **Verified** — the tier the audit ranks first | [§2](MIXTURE_PLAN.md#2-data-gating-status), [§16](MIXTURE_PLAN.md#16-where-the-cleaning-goes-next) |

### The three weakest numbers

1. **Agentic is 82% data that does not exist.** 4.0B of real trajectories against 90.0B of demand — **23 epochs** of everything ever published. Costed at §6, with a fallback if the harness slips. Stated, not hidden.
2. **14T is infeasible and the plan says so.** The budget is swept 2.4T→14T; at 14T the build *refuses*, and it fails on the **web** lane — not agentic, not Indic. See [§3.1](MIXTURE_PLAN.md#31-budget-sweep).
3. **The Indic conclusion is conditional.** AI4Bharat documents no tokenizer for its 251.3B count; four sources checked. At our own measured cl100k fertility the whole plan becomes infeasible. [§14](MIXTURE_PLAN.md#14-sensitivity) swings it and says so.

<!-- SCORECARD:END -->

<details open>
<summary><b>Contents</b></summary>

- [What the assignment asked, and where it is answered](#what-the-assignment-asked-and-where-it-is-answered)
- [Invariants enforced](#invariants-enforced)
- [Budget](#budget)
- [Review round 2](#review-round-2) · [3](#review-round-3) · [4](#review-round-4)
- [Review round 5 — decontamination, measured](#review-round-5--decontamination-measured)
- [Review round 6 — a proxy was actually run](#review-round-6--a-proxy-was-actually-run)
- [Review round 7 — the priority-1 cleaning job was run](#review-round-7--the-priority-1-cleaning-job-was-run)
- [Not yet done](#not-yet-done)

</details>

```
plan/spec.py         declarative: budget, phases, lanes, inventory, floor,
                     bands, LR schedule, sequence ladder, manufacturing plan,
                     proxy arms, sensitivity bounds. No prose, no math.
plan/audit.py        the arithmetic + every invariant the plan asserts about
                     itself. Raises rather than emitting a number that is wrong.
plan/build_plan.py   renders MIXTURE_PLAN.md and mixture_results.json.

proxy/arms.py        generates + validates the 7 proxy arm configs from spec
proxy/run_proxy.py   the screen itself; --dry-run needs no GPU

decon/measure.py     real benchmark-contamination measurement (stdlib only)
decon_results.json   its output - measured, not asserted
```

Reproduce:

```bash
cd plan
python audit.py                    # invariants + per-lane verdict
python build_plan.py               # writes ../MIXTURE_PLAN.md

cd ../proxy
python arms.py                     # print + validate all arm configs
python run_proxy.py --dry-run      # full control flow, no GPU

cd ../decon
python measure.py                  # fetch benchmarks + scan corpora (~40s)
```

No dependencies beyond the standard library. (Actually training needs torch +
transformers; see [`proxy/README.md`](proxy/README.md).)

## Invariants enforced

The build fails, rather than producing a plausible-looking document, if:

- phase weights do not sum to 1.0
- any phase's lane shares do not sum to 100
- whole-run lane shares do not sum to 100, or lane tokens do not sum to the budget
- a protected floor exceeds its own lane's whole-run share (an unreachable floor)
- the Indic four-tier split does not sum to the Indic lane
- Indic real supply exceeds lane demand (a tier cap set too high)
- difficulty bands in any phase do not sum to 100, or length bands do not sum to 100
- **a lane needs manufactured tokens but has no costed entry in `MANUFACTURING`**
- **the anneal reserve cannot be stocked under its own eligibility rules at one epoch**
- **proxy epochs do not equal full-run epochs lane by lane** (corpus scaling broken)

The last three are the point of the exercise. They make it structurally
impossible to hand a lane a share it has no data for without saying how the
tokens get made and what that costs, to declare a premium cooldown you cannot
fill, or to ship a repetition experiment that never repeats anything.

Each one has already caught a real error in this plan — see §5.1, §10.1 and
§13.1, which document what failed and what changed.

## Budget

The budget is a **parameter**, not an assumption. Session 3's design memo locked
14T for a 40B dense model; session 5, asked directly, answered "between 2.4 to 4
trillion tokens" for what V5 actually trains on. `audit.sweep()` runs the same
mixture across all four and reports which survive:

| Budget | Feasible | `GENERATE` lanes | Manufacturing |
|---:|---|---:|---:|
| 2.4T | ✅ | 2 | $13,100 |
| **3.0T** (primary) | ✅ | 2 | $28,105 |
| 4.0T | ✅ | 3 | $53,136 |
| 14.0T | ❌ build refused | — | — |

> [!CAUTION]
> 14T fails on the **web** lane, not agentic or Indic — there is not enough
> high-quality English web in existence to feed it once counts are converted to
> one tokenizer and doubled for the selector's discard rate.

```bash
python -c "import audit; print(audit.run(4e12)['audited'])"   # any scenario
```

## Review round 2

After a first draft passed its own invariants, an adversarial review found three
defects that the invariants did not cover. All three are now fixed and enforced:

| Fix | Defect | Consequence |
|---|---|---|
| **Units** (§5.1) | Token counts from four different tokenizers were summed, then an Indic-efficiency credit was claimed on top — double-counting | Web moved 0.97 → 1.16 epochs (`SUPPLY-OK` → `REPEAT`); code onto its cap at 3.96; Indic manufactured 17.9% → 39.6% |
| **Anneal supply** (§10.1) | Eligibility rules were declared but never checked against supply | 9.5B-token shortfall found; Phase D recomposed |
| **Proxy corpus scaling** (§13.1) | 1B/20B tokens left pools at full size, so arm A3 would have run at 0.006 epochs and could never observe repetition | Pools subsampled by the budget factor; proxy epochs now equal full-run epochs by construction |

## Review round 3

A pass against the session transcript rather than against the plan's own logic
found a fourth defect and one wrong input:

| Fix | Defect | Consequence |
|---|---|---|
| **Selector accounting** (§5.2) | `keep_fraction` appeared only in prose. Discarded batches are thrown away and never re-offered, so supply must cover *candidate* tokens | Per-lane multiplier of 1.40×–2.00×; the plan trains 3.00T and must hold 5.79T of corpus |
| **Budget sweep** (§3.1) | The plan asserted 14T from a session-3 design memo; session 5 named 2.4–4T for V5 | 14T now provably infeasible; 3T primary. Indic flipped from 39.6% manufactured to **0%**, running on verified text alone with the entire translated tier unused |

## Review round 4

| Fix | What changed |
|---|---|
| **Muennighoff decay** (§5.4) | The 4-epoch cliff replaced with the paper's fitted curve `D' = U + U·R*·(1−exp(−R/R*))`, R*=15.387. The cap yields **3.726 effective epochs, not 4.0**; 20.2B tokens of nominal budget buy nothing |
| **LR schedule** (§10.2) | Warmup → cosine to 10% of peak → linear to zero across phase D. The anneal is now a learning-rate event, not only a data event |
| **Sequence ladder** (§10.3) | 4K→8K→16K→32K→64K, length-homogeneous batches, 4,096 floor, tokens-per-batch held at 2,097,152 (asserted) |
| **Band crossfade** (§10.4) | 15% linear interpolation at every phase boundary, grad-norm target 0.2 / trip 0.5, extend-the-fade rather than cut-LR on trip |
| **Arms A5, A6** (§13.2) | A5 disables the selector — the 2×-corpus cost had no arm testing it. A6 ablates the crossfade |
| **Decontamination** (§13.5) | 12-gram + MinHash against 16 named suites, at corpus build *and* reserve build. Reported as **not yet measured** |
| **Sensitivity** (§14) | Four assumptions swung to their bounds with the whole plan re-audited at each |

### What the sensitivity analysis found

**2 of 4 assumptions swing a conclusion.**

- ⚠️ **Sangraha's tokenizer** — AI4Bharat publishes 251.3B tokens and documents
  no tokenizer. Checked the HF dataset card, the arXiv abstract, the full paper
  PDF, and the IndicLLMSuite GitHub README: **none state it.** At our own
  measured cl100k fertility the entire plan becomes *infeasible*. §7 is
  conditional until this is closed.
- ⚠️ **Tokens per court judgment** — flips civic from `REPEAT` to `GENERATE` at
  the low bound.
- ✅ **Web dedup haircut** — holds down to 0.40. Risk retired by the budget fix.
- ✅ **Agentic trajectory length** — stays 70–87% manufactured across a 9× swing.
  The conclusion is robust even though the number is a guess.

## Review round 5 — decontamination, measured

`decon/measure.py` pulls real test items through the HF datasets-server and
scans the corpora we hold. Stdlib only; n-grams are over whitespace tokens, so
they are comparable across scripts.

| Corpus | Docs | 12-gram hits | Rate | Detection floor |
|---|---:|---|---:|---:|
| Sangraha unverified / Telugu | 11,607 | none | 0.0000% | 0.0258% |
| Reasoning / SFT mix | 9,553 | mbpp: 2 | 0.0209% | 0.0314% |

80,880 distinct 12-grams across gsm8k, mmlu, humaneval, mbpp, belebele_tel, arc.
The Telugu zero is reported as a **bound**, not a result — the detector fires on
the code corpus, so it works; 11.6k documents simply cannot resolve below
~0.026%. Both code hits are the same binary-search idiom, which the plan's
strip-the-document policy would remove but which is not really contamination.

Scope: 2 of 9 lanes, 6 of 16 suites. FLORES-200 is gated (Belebele substitutes,
its passages come from FLORES); the SWE-bench family, Terminal-Bench and τ-bench
are not scannable this way — and those are the suites that matter most for our
targets.

### The first run of this measurement was wrong

It reported 0.560% on Telugu and 0.105% on reasoning. Both were normaliser
artefacts. Stripping punctuation with `[^\w\s]` fails because Python's `\w` does
not match Unicode combining marks — Telugu vowel signs and viramas were treated
as punctuation, **shattering every word into individual consonants**, and twelve
bare consonants collide by chance. The same stripping turned
`for i in range(i+1, n):` into `for i in range i 1 n`, so the code "matches" were
generic loop idioms. This is the exact class of bug session 4 guarded against
with Brahmic joiner preservation, reintroduced by one regex.

Keeping punctuation dropped the rates to 0.0000% and 0.0209%. A number from a
broken detector looks identical to one from a working detector, which is why the
matched n-grams are printed and inspected rather than just the rate.

### A bug the proxy harness caught

LR warmup was specified as 15B absolute tokens — 0.5% of the 3T run, but **75%
of a 20B proxy arm**. Every arm would have been compared while still warming up.
Now a fraction. Found by running `run_proxy.py --dry-run`, which is the argument
for building the harness before the plan is final.

## Review round 6 — a proxy was actually run

The §13 screen (7 arms at 1B) still has not been run; it needs GPUs we don't
have. What *was* run is a deliberately narrower experiment on one RTX 3070
Laptop: **11.0M-parameter byte-level model, 25M byte-tokens per arm, 3 arms ×
2 seeds, ~94 min total**, targeting only the two claims that are
training-dynamics rather than capability claims.

```bash
cd microproxy
python prepare.py     # byte arrays from the session-4 corpora
python run_all.py     # 3 arms x 2 seeds
python analyse.py     # -> ../microproxy_results.json
```

| Arm | Indic by phase | Crossfade | Spike rate (transition) | Peak grad-norm | Final bpb indic | Final bpb reasoning |
|---|---:|---:|---:|---:|---:|---:|
| A0 baseline | 10/30/60/80 | 15% | 0.0028 | 0.786 | 1.4245 | 3.1213 |
| A2 no floor | 0/37/68/86 | 15% | 0.0028 | 1.038 | **1.4102** | 3.1564 |
| A6 sharp | 10/30/60/80 | 0% | 0.0046 | 0.966 | 1.4355 | 3.1145 |

A2's total indic share is matched to A0's (33.3% vs 33.0%), so it isolates
*ordering*, not budget.

> [!IMPORTANT]
> **Both verdicts: INCONCLUSIVE** — not because the mixture is fine, but because
> both pre-registered decision rules turned out to be untestable at a scale we
> could afford. That is the run's most useful output, and a finding about the
> plan itself.

In detail, because
**both pre-registered decision rules turned out to be untestable at a scale we
could afford** — which is the run's most useful output and a finding about the
plan itself:

- **Spike-count rule had no power.** 8 spike events total across both arms and
  both seeds. The 1.67× ratio is a ratio of small integers.
- **Grad-norm limb did not transfer.** The rule's "peak > 0.5" threshold was
  written against the full run's 0.2 target; A0's peak *outside* any transition
  is already 2.131, so it would have fired unconditionally and manufactured a
  KEEP verdict.
- **bpb rule was swamped by seed noise.** Within-arm seed spread is 0.0174 bpb;
  the between-arm effect is 0.0143. Both seeds agree on sign (A2 ahead on
  indic), but 2-of-2 sign agreement is p=0.5.

The rules were calibrated for 1B params on 20B tokens and neither survives the
drop to 11M. §13.4 needs a minimum-detectable-effect column before those rules
are trusted at 1B either — the same failure would then cost $6,320 instead of an
afternoon.

A post-hoc continuous metric (per-step excess loss in robust-sigma units) is
used for the crossfade verdict and **labelled as post-hoc**, with the
pre-registered number still reported. Changing the metric after seeing data is
how results get manufactured; the justification is that the pre-registered one
recorded 8 events, not that its answer was unwelcome.

## Review round 7 — the priority-1 cleaning job was run

§16 ranks Sangraha **Verified** as the top cleaning target: session 4 cleaned the
*unverified* tier, which at 3T supplies 11.1% of the Indic lane and is barred
from the anneal, while Verified supplies 88.9% and is the only tier the cooldown
accepts. That job has now been run rather than just written down.

```bash
cd cleaning
python run_verified.py --check-only   # draw + prove disjointness
python run_verified.py                # full 8 stages + determinism re-run
```

| | Session 4 | + this session | Cumulative |
|---|---:|---:|---:|
| Cleaned tokens | 53,781,200 | **+39,238,266** | **93,019,466** |
| Documents | 18,478 | +11,554 | 30,032 |

**1.73× gate**, and the added tokens are the only ones in the corpus the anneal
will accept.

### Contamination guard — proven, not assumed

`telugu_web`'s held-out Golden Proxy is drawn from `verified/tel` **rows 0–299**.
A contiguous-from-head Verified training slice would have trained directly on it.
The registry sets `skip_rows=300` and `run_verified.py` asserts the result:

```
frozen held-out doc_ids: 300
new training doc_ids:    14,596
lowest training row:     300
overlap:                 0        -> DISJOINT
```

### What the run turned up (§16.1)

Verified loses **16.58%** at the quality filter against unverified's 8.01% —
twice the cull on the better tier. I chased two wrong explanations before
measuring it:

1. ~~Classifier transfer failure~~ — the trained classifier rejects 1.8× more,
   but the layer-1 `too_few_stopwords` heuristic does ~60% of removals in *both*
   corpora.
2. ~~OCR/ASR content~~ — unsupportable. Sangraha's parquet carries no sub-source
   labels, so OCR fragments and concise articles are indistinguishable here.

The measured cause is a **fixed-window length bias**: the rule wants ≥2 stopword
hits in the first 250 words, and Verified documents are 18% shorter (257 vs 313
mean words). Predicted rejection 10.28% vs observed 9.90% — the scan explains the
funnel. Fix is a length-normalised threshold before this tier is cleaned at scale.

A second methodological note: stage 7 removed 5.00% of Verified vs 0.42% of
unverified. That is **not** evidence Verified is dirtier — it is an artifact of
drawing the held-out set adjacent to the training slice in the same shard, where
session 4's came from a different tier. The filter correctly removed the overlap.
The lesson is about held-out design, and it applies to §13's proxy too.

Corroboration worth noting: Verified fertility measured **13.292 tok/word**
against unverified's 13.268 — independently confirming the cl100k Telugu figure
§5.1 depends on.

### Session 4 is untouched

`telugu_verified` is deliberately **not** in `corpora.ORDER`, and the runner calls
`process_corpus()` rather than `main()`, so `results.json` is never rewritten. All
eight original session-4 artifacts were hashed before and verified byte-identical
after. The two new `raw_sample/telugu_verified_*.jsonl` files are additive — they
live in session 4's folder only because `corpora.py` hardcodes `RAW_DIR`.

## Not yet done

- **The full §13 screen has not been trained** — 7 arms at 1B needs real GPUs.
- **Priority-2 cleaning is not done.** The reasoning lane is 35% manufactured
  with 3 of 4 sources failing the licence gate. Fixing the licences comes before
  scaling it.
- **The length-bias fix is diagnosed, not applied.** §16.1 recommends a
  length-normalised stopword threshold; the 39.2M tokens above were cleaned with
  the biased one, so ~10% of eligible Verified documents were discarded for being
  concise. Re-running with the fix would recover them.
- **Decontamination covers 2 of 9 lanes and 6 of 16 suites** — real within that
  scope, silent outside it.
- **The micro-proxy tests 2 lanes, not 9.** A Telugu→English shift is the
  sharpest possible transition; a crossfade effect absent here could still
  appear with nine lanes and gentler, more frequent boundaries.

## Inputs it depends on

| From | What |
|---|---|
| `3_40b_model/plan-notes.md` | 14T token budget, 40B dense, the session-3 mix this plan revises |
| `4_model_data/assignment/results.json` | the 53,781,200 measured cleaned tokens behind the data gate |
| `session_5_transcipt.md` | loss-masking policy, selector pathology, reasoning-band framing |

Supply figures are tagged `MEASURED` / `PUBLISHED` / `DERIVED` / `ESTIMATED` in
`spec.py`, and the tag travels into the document, so which numbers are
measurements and which are assumptions stays visible.

## Not done yet

The proxy experiment in §13 is specified and costed ($5,814, 0.082% of the full
run) but **has not been run**. Until it is, the plan is a hypothesis, which is
what §1 says it is.
