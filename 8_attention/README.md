# Attention, in the order it actually happened

[![checks](https://github.com/rahulni/Indic_LLM/actions/workflows/attention-checks.yml/badge.svg)](https://github.com/rahulni/Indic_LLM/actions/workflows/attention-checks.yml)

An interactive reference covering **58 attention mechanisms from 2014 to 2026**, ordered by
the date each one actually launched, with honest trade-offs for every entry.

**→ [Read it here](https://rahulni.github.io/Indic_LLM/8_attention/)**

Vanilla attention was never wrong. It was expensive. Everything after it is somebody
looking at that bill and trying to pay less of it — and each one gives something up to do
it. Laid out chronologically you can watch the field change its mind: first it wants
exactness, then memory, then length, then memory again.

---

## Contents

- [What this is](#what-this-is)
- [How the dates are verified](#how-the-dates-are-verified)
- [Coverage](#coverage)
- [Further reading](#further-reading)
- [Naming traps worth knowing](#naming-traps-worth-knowing)
- [Project structure](#project-structure)
- [Development](#development)
- [Provenance and scope](#provenance-and-scope)

---

## What this is

Most explanations of attention are either a tutorial on the 2017 paper or a survey listing
thirty mechanisms by family. This is neither. It is a **timeline**, and the ordering is the
argument: each mechanism is presented as an answer to a problem that existed at that
moment, so you can watch the field change direction rather than just enumerate where it
ended up.

Three things it tries to do differently:

- **Every mechanism states what it costs.** A technique written up with only advantages has
  not been understood. `tools/check_completeness.py` fails the build on an empty cost list,
  so this is enforced rather than intended.
- **Every number is computed, not drawn.** The attention matrix runs a real softmax; the
  KV-cache figures are arithmetic you can check; the animations are driven by
  [`app/src/lib/attention.ts`](app/src/lib/attention.ts) in the browser.
- **Every date is checked against its primary source**, automatically, on every push.

It reads at three levels — *new to this*, *I build with these*, *I read the papers* — which
change what is shown and how much, not merely which paragraph renders. Trade-offs and dates
stay visible at every level.

---

## How the dates are verified

Dates are the part of a project like this that is easiest to get wrong and easiest to
check, so they are automated rather than promised.

### The rule

Each mechanism is dated by its **arXiv v1 submission date** — the `published` field from
the arXiv API, which is when version 1 went public. That is used as the launch date because
it is the first moment the work existed publicly, it is machine-checkable, and it is
stable: later revisions change the content but never the v1 timestamp. Conference dates run
months later and would compress the timeline misleadingly.

### The check

[`tools/verify_dates.py`](tools/verify_dates.py) reads the same `mechanisms.json` the site
renders, batches the arXiv API, and asserts that **both the date and the paper title**
match. The title check exists because a correct date attached to the wrong paper is still
wrong — it is what catches a transposed identifier.

```
$ python tools/verify_dates.py
arXiv dates verified against the API : 56
non-paper dates with written evidence: 2
total mechanisms                     : 58

OK - every date is backed by a primary source.
```

It exits non-zero on any mismatch and runs in CI on every push. The full table is in
[CITATIONS.md](CITATIONS.md), generated from the same file.

### arXiv identifiers are not dates

The identifier prefix reflects the announcement cycle, not the submission:

- YaRN is `2309.00071` but was submitted **31 August 2023**, not September.
- Titans is `2501.00663` but was submitted **31 December 2024**, not 2025.

Reading the month off the identifier is the most common way these dates go wrong.

### The two entries with no paper

Not everything has a preprint, and pretending otherwise would defeat the point. These are
exempt from the API check but must carry written evidence naming the dated artifact, or
`verify_dates.py` fails them too:

| Mechanism | Date | Source |
|---|---|---|
| **NTK-aware scaled RoPE** | ~28 Jun 2023 | A [Reddit post by u/bloc97 in r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), corroborated by [huggingface/text-generation-inference issue #512](https://github.com/huggingface/text-generation-inference/issues/512). There is **no paper**; it was later folded into YaRN as a baseline. |
| **DeepSeek Sparse Attention** | 29 Sep 2025 | A product release, announced in the [DeepSeek API docs](https://api-docs.deepseek.com/news/news250929/) with day-0 vLLM support. The formal write-up came later in the DeepSeek-V3.2 report (arXiv 2512.02556). |

---

## Coverage

58 mechanisms across nine categories — because not everything here is an attention
mechanism. Sinusoidal encodings are a positional scheme, YaRN is a RoPE rescaling trick,
PagedAttention is a serving optimisation. Labelling them accurately is part of
understanding them.

**The mechanism and its positions** — scaled dot-product · multi-head · learned absolute ·
sinusoidal · Shaw relative · RoPE · ALiBi · NoPE · DroPE

**KV cache and compression** — MQA · GQA · MLA · TPA · Linformer · Infini-attention ·
Titans · Kwai Summary Attention

**Sparsity** — Sparse Transformers · Longformer · BigBird · top-k · sliding window · NSA ·
MoBA · DSA · BFLA

**Linear and recurrent** — linear attention · Performer · the delta rule · DeltaNet · GLA ·
GSA · Gated DeltaNet · RWKV · RetNet · Mamba · Mamba-2 · xLSTM · Log-Linear Attention

**Context extension** — Transformer-XL · Positional Interpolation · NTK-aware · YaRN ·
attention sinks

**Systems** — FlashAttention 1/2/3 · PagedAttention · Ring Attention

**Hybrids** — Lightning Attention · Kimi Linear · HySparse · FlashMorph

**Quality rather than cost** — Differential Transformer · Forgetting Transformer ·
Multi-Token Attention

Plus the origin story: Bahdanau (2014) and Luong (2015), where attention is invented as a
patch for a bottleneck rather than as an architecture.

---

## Further reading

The site has a [Further reading](https://rahulni.github.io/Indic_LLM/8_attention/#reading)
section grouped by what you want next, and most mechanism cards link to a canonical
implementation under *Go deeper*. The short version:

| If you want | Start with |
|---|---|
| Intuition before maths | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [Lilian Weng on attention](https://lilianweng.github.io/posts/2018-06-24-attention/) |
| The paper as runnable code | [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) |
| To run the modern linear family | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) |
| To serve these at scale | [vLLM](https://docs.vllm.ai/en/latest/) · [HF attention interface](https://huggingface.co/docs/transformers/en/attention_interface) |
| The whole field surveyed | [*Efficient Attention Mechanisms for LLMs*](https://arxiv.org/abs/2507.19595) |
| To follow what actually ships | [Raschka's architecture comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) |

Each resource on the site says what it is **not** good for as well as what it is, because a
recommendation without its limits costs you an afternoon before you find out it was the
wrong one.

Cards also cross-link along **lineage** rather than similarity, so walking MQA → GQA → MLA →
TPA reads as one problem attacked four ways rather than as four unrelated entries. All 80
URLs in the project are re-checked by [`tools/check_links.py`](tools/check_links.py).

---

## Naming traps worth knowing

These are the places where a confident sentence is wrong, and where an AI assistant asked
about them is likely to produce fluent nonsense. The site's **Fact-check** section covers
more.

### `DroPE` and `DRoPE` are different papers

|  | **DroPE** | **DRoPE** |
|---|---|---|
| arXiv | 2512.12167 | 2503.15029 |
| v1 date | 13 Dec 2025 | 19 Mar 2025 |
| Field | LLM context extension | autonomous-driving agent trajectories |
| What it does | Train with RoPE, drop it, recalibrate, extend context zero-shot | Encodes agent heading angles in a rotary embedding |

One letter of casing apart, nine months apart, unrelated fields. Ask for "DroPE" and there
is a real chance you get the driving paper described with total confidence.

### `GSA` is not `GQA`

**Gated Slot Attention** ([2409.07146](https://arxiv.org/abs/2409.07146)) is a
linear-attention method with bounded memory slots and adaptive forgetting.
**Grouped-Query Attention** ([2305.13245](https://arxiv.org/abs/2305.13245)) is a
KV-head-sharing scheme for ordinary softmax attention. Different categories entirely.

### `NSA` is not `DSA`

Both are DeepSeek sparse attention, seven months apart, and they work differently. NSA
([2502.11089](https://arxiv.org/abs/2502.11089)) selects coarse **blocks** using three
branches. DSA (released 29 Sep 2025) selects individual **tokens** using a lightning
indexer feeding sparse MLA.

### And two dates

*Attention Is All You Need* is **12 June 2017**, not 2018 — BERT is what made Transformers
inescapable in 2018, but the mechanism predates it by over a year. MQA is **November 2019**
and GQA is **May 2023**; they are taught together, but they are three and a half years
apart, and that gap is one of the more interesting things on the timeline.

---

## Project structure

```
8_attention/
├── README.md                     this file
├── ANSWERS.md                    write-up: what the timeline shows
├── CITATIONS.md                  generated evidence table — do not hand-edit
├── index.html + assets/          the published site (served by GitHub Pages)
├── tools/
│   ├── verify_dates.py           every date and title, against the arXiv API
│   ├── check_completeness.py     coverage, and that every entry states its costs
│   ├── check_links.py            every URL still resolves
│   ├── build_citations.py        regenerates CITATIONS.md
│   └── publish_site.py           promotes the build to the served location
└── app/
    └── src/
        ├── data/mechanisms.json  single source of truth — 58 entries
        ├── data/eras.json        the narrative arc
        ├── data/formulas.ts      equations bound to the visuals
        ├── data/resources.ts     curated further reading
        ├── lib/attention.ts      real softmax / RoPE / KV / mask arithmetic
        ├── viz/                  diagrams and animated scenes
        └── components/           timeline, cards, on-ramp, fact-check
```

`mechanisms.json` is the single source of truth. The site renders it, the date checker
validates it, the completeness test asserts against it, and `CITATIONS.md` is generated
from it — so the page and its citations cannot disagree.

---

## Development

Requires Python 3.12+ and Node 20+.

```bash
# Verification — stdlib only, no npm needed
python tools/verify_dates.py         # every date + title against the live arXiv API
python tools/check_completeness.py   # coverage, and non-empty costs on every entry
python tools/check_links.py          # every URL still resolves
python tools/build_citations.py      # regenerate CITATIONS.md

# The app
cd app
npm ci
npm run dev        # http://localhost:5173/
npm run smoke      # server-renders every level and asserts the maths
npm run build

# Publish
cd .. && python tools/publish_site.py
```

`npm run smoke` is not a formality. It renders all three reading levels in Node and asserts
that softmax rows sum to 1, that causality holds, that unscaled attention is measurably
more peaked than scaled, that MHA > GQA > MQA > MLA, and that all 171 cross-links resolve
to real mechanisms.

All of these run in CI on every push, before the site is published.

### How publishing works

GitHub Pages on this repository serves committed branch contents directly, so the site is
simply the files at `8_attention/index.html` and `8_attention/assets/`. Pushing to `main`
publishes it; there is no deploy step. Rebuild with `npm run build`, then
`python tools/publish_site.py`, which records a fingerprint of the content files in
`build-info.json`. CI fails if that fingerprint no longer matches the data in the
repository — which catches "edited the data, forgot to republish" without demanding
byte-identical builds across platforms.

---

## Provenance and scope

Built for a session on modern attention variants in an LLM engineering course, which is why
the write-up in [ANSWERS.md](ANSWERS.md) is addressed to an instructor. The rest is meant
to stand on its own.

**Scope.** This covers attention and its immediate neighbours — positions, KV cache,
sparsity, linear and recurrent alternatives, and the systems work that changed which of
them are practical. It does not cover tokenisation, MoE routing, optimisers, or training
recipes except where they bear directly on an attention decision.

**Editorial rules.** No mechanism ships without stated costs. No date ships without a
primary source. No animation shows an invented metric — the visualisations show structure,
and where a consequence is qualitative the caption says so in words. Where an equation is
not verbatim from its paper it is labelled *essential form*, and says so.

**Course material.** The session transcript is not in this repository. It is the course's
content rather than this project's, the repository is public, and it records classroom
discussion. Conclusions drawn from it are original writing and are published here.
