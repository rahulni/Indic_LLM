# Attention, in the order it actually happened

**Live site:** https://rahulni.github.io/Indic_LLM/8_attention/
**Repository:** https://github.com/rahulni/Indic_LLM/tree/main/8_attention

A web app explaining every attention mechanism from 2014 to 2026, ordered by the date
each one actually launched, with honest trade-offs for each.

Vanilla attention was never wrong. It was expensive. Everything after it is somebody
looking at that bill and trying to pay less of it — and every one of them gives something
up to do it. Laid out chronologically you can watch the field change its mind: first it
wants exactness, then memory, then length, then memory again.

**52 mechanisms. 50 dates verified against the arXiv API. 2 that have no paper at all,
and say so.**

---

## Where the dates came from

This is the part that is easiest to get wrong and easiest to check, so it is automated
rather than promised.

### The rule

Each mechanism is dated by its **arXiv v1 submission date** — the `published` field in
the arXiv API, which is when version 1 went public. That is used as the launch date
because it is the first moment the work existed publicly, it is machine-checkable, and it
is stable: later revisions change the content but never the v1 timestamp. Conference
publication dates run months later and would compress the timeline misleadingly.

### The check

`tools/verify_dates.py` reads the same `mechanisms.json` the site renders, batches the
arXiv API, and for every paper-backed entry asserts that **both the date and the paper
title** match what arXiv returns. The title check exists because a correct date attached
to the wrong paper is still wrong — it is what catches a transposed identifier.

```
$ python tools/verify_dates.py
arXiv dates verified against the API : 50
non-paper dates with written evidence: 2
total mechanisms                     : 52

OK - every date is backed by a primary source.
```

It exits non-zero on any mismatch and runs in CI on every push, so a date that stops
matching its source turns the build red rather than sitting there quietly. The full table
is in [CITATIONS.md](CITATIONS.md), generated from the same file.

### A warning about arXiv identifiers

The identifier prefix is **not** a reliable date. Identifiers are assigned around the
announcement cycle, not on submission:

- YaRN is `2309.00071` but was submitted **31 August 2023**, not September.
- Titans is `2501.00663` but was submitted **31 December 2024**, not 2025.

Reading the month off the identifier is the most common way these dates go wrong.

### The two that have no paper

Not every mechanism has a preprint, and pretending otherwise would be the exact failure
this project is guarding against. These are exempt from the API check but must carry a
written `date_evidence` string naming the dated artifact, or `verify_dates.py` fails them
too:

| Mechanism | Date | Source |
|---|---|---|
| **NTK-aware scaled RoPE** | ~28 Jun 2023 | A [Reddit post by u/bloc97 in r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), corroborated by [huggingface/text-generation-inference issue #512](https://github.com/huggingface/text-generation-inference/issues/512). There is **no paper**; it was later folded into YaRN as a baseline. |
| **DeepSeek Sparse Attention (DSA)** | 29 Sep 2025 | A product release, announced in the [DeepSeek API docs](https://api-docs.deepseek.com/news/news250929/) with day-0 vLLM support. The formal write-up came later in the DeepSeek-V3.2 report (arXiv 2512.02556). |

---

## Two things worth flagging

### 1. `DroPE` and `DRoPE` are different papers

This is the trap this assignment is most likely to spring, and it is worth stating plainly:

| | **DroPE** | **DRoPE** |
|---|---|---|
| arXiv | 2512.12167 | 2503.15029 |
| v1 date | 13 Dec 2025 | 20 Mar 2025 |
| Authors | Gelberg, Eguchi, Akiba, Cetin (Sakana AI) | different group |
| Field | LLM context extension | autonomous-driving agent trajectories |
| What it does | Train with RoPE, drop it, recalibrate briefly, extend context zero-shot | Encodes agent heading angles in a rotary embedding |

The names differ only in capitalisation and the papers are nine months and one research
field apart. An agent asked to "explain DroPE" will very plausibly fetch the driving paper
and write a fluent, entirely wrong section.

The mechanism described in Session 8 — train with RoPE, then remove it for a large length
extension — is unambiguously the Sakana one.

### 2. GSA was taught but is not on the required list

The session covered **GSA** alongside GQA. That is **Gated Slot Attention**
([arXiv 2409.07146](https://arxiv.org/abs/2409.07146), v1 11 Sep 2024, NeurIPS 2024) — a
linear-attention method with bounded memory slots and adaptive forgetting. It is *not* a
mis-hearing of GQA: the two were enumerated adjacently, and they are different categories
of thing entirely (GQA shares KV heads in softmax attention; GSA is a linear-attention
method).

It appears nowhere in the assignment's required list, so checking only that list would
have silently dropped a mechanism that was actually taught. That is why
`check_completeness.py` validates against two lists rather than one.

More of these are collected on the site's **Fact-check** section, including the
NSA-vs-DSA distinction, the Mistral sliding-window attribution, and MQA's four-year gap
between being published and being adopted.

---

## Coverage

All 18 mechanisms named in the assignment are covered, plus 34 more.

**Required:** standard attention · learned absolute positions · sinusoidal · RoPE ·
ALiBi · MQA · GQA · sliding window · attention sinks · NTK-aware scaling · YaRN · linear
attention · delta rule · Gated DeltaNet · MLA · sparse and top-k attention · NSA/DSA ·
DroPE

**Added, because the story does not work without them:** Bahdanau and Luong attention
(2014–15, where attention is invented as a patch) · Shaw relative positions ·
Transformer-XL · Longformer · BigBird · Linformer · Performer · FlashAttention 1/2/3 ·
Positional Interpolation · NoPE · PagedAttention · Ring Attention · RWKV · Mamba /
Mamba-2 · GLA · **GSA** · Infini-attention · DeltaNet parallelisation · Differential
Transformer · Titans · Lightning Attention · MoBA · Kimi Linear / KDA

**2026 frontier**, all web-verified rather than recalled: HySparse (Feb 2026) · Kwai
Summary Attention (Apr 2026) · BFLA (May 2026) · FlashMorph (Jun 2026)

Each entry is labelled with its category, because not everything on the list is an
attention mechanism — sinusoidal encodings are a positional scheme, YaRN is a RoPE
rescaling trick, PagedAttention is a serving optimisation. Saying so accurately is part of
understanding them.

---

## The interactive parts compute real numbers

Nothing on the page is a pre-rendered animation. Every figure comes from
[`app/src/lib/attention.ts`](app/src/lib/attention.ts) running in the browser.

- **The attention matrix** runs real scaled dot-product attention over a seven-token
  sentence. Turn the `÷√d` scaling **off** and watch mean row entropy collapse — that is
  why the scaling exists. Turn the **learned Q/K projections off** and the diagonal
  lights up, because every vector's best match becomes itself. That is why queries and
  keys are separate networks.
- **The sparse-pattern grid** re-masks one shared matrix live across all six patterns,
  reporting what fraction of cells each actually computes.
- **The KV-cache calculator** computes real byte counts for MHA / GQA / MQA / MLA and
  prints the arithmetic so you can check it. Push the context slider to 1M.
- **The RoPE dial** verifies relative-position invariance at four absolute positions and
  shows the residual spread is floating-point noise.
- **The state visualiser** runs the session's own 40 → 55 → 95 example, comparing additive
  updates against the delta rule over a sequence of writes.

---

## Verify it yourself

```bash
cd 8_attention

# Every date checked against the live arXiv API (date AND title)
python tools/verify_dates.py

# All 18 required mechanisms, plus everything taught in the session,
# each with non-empty pros AND cons
python tools/check_completeness.py

# Regenerate the citation table from the same source file
python tools/build_citations.py

cd app
npm ci
npm run smoke     # renders the whole app in Node, asserts the maths
npm run build
npm run preview
```

`npm run smoke` is not a formality. It server-renders every section and asserts that
softmax rows sum to 1, that causality holds (nothing attends to its own future), that
unscaled attention is measurably more peaked than scaled, and that the KV-cache ordering
MHA > GQA > MQA > MLA actually holds.

All four commands run in CI on every push, before the site is deployed.

---

## Structure

```
8_attention/
  README.md                  this file
  CITATIONS.md               generated evidence table - do not hand-edit
  tools/
    verify_dates.py          checks every date against the arXiv API
    check_completeness.py    checks coverage + that every mechanism lists costs
    build_citations.py       regenerates CITATIONS.md
  app/
    src/data/mechanisms.json single source of truth - 52 entries
    src/data/eras.json       the narrative arc
    src/lib/attention.ts     real softmax / RoPE / KV / mask arithmetic
    src/viz/                 six computed visualisations
    src/components/          timeline, cards, fact-check
```

`mechanisms.json` is the single source of truth. The site renders it, the date-checker
validates it, the completeness test asserts against it, and `CITATIONS.md` is generated
from it. The page and its citations physically cannot disagree.

---

## Notes

The session transcript is not in this repository. It is course material rather than this
project's work, the repo is public, and it contains classroom discussion — so it stays
local, consistent with the existing `.gitignore` policy for transcripts. The conclusions
drawn from it (dates, trade-offs, the fact-check section) are original writing and are
published here.

## Deployment

GitHub Pages on this repository serves the committed branch contents directly, so the
site is simply the files committed at `8_attention/index.html` and `8_attention/assets/`
— the same pattern as the earlier submissions. Pushing to `main` publishes it; there is
no deploy step.

To republish after a change:

```bash
cd app && npm run build
cd .. && python tools/publish_site.py
```

[`.github/workflows/attention-checks.yml`](../.github/workflows/attention-checks.yml)
runs the date verification, the coverage check and the smoke test on every push, and
fails if the committed bundle no longer matches a fresh build of the source.
