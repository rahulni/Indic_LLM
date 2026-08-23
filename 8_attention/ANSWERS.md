# Submission answers

---

## Question 1 — the links

**Live app:** https://rahulni.github.io/Indic_LLM/8_attention/

**GitHub repo:** https://github.com/rahulni/Indic_LLM/tree/main/8_attention

58 mechanisms, 2014 to 2026, ordered by launch date. 56 dates verified against the arXiv
API by `tools/verify_dates.py`, which checks both the submission date and the paper title
and fails the build on any mismatch. The remaining 2 have no paper at all and carry
written evidence instead. Sources and method are in [README.md](README.md); the full
table is in [CITATIONS.md](CITATIONS.md).

---

## Question 2 — what the timeline shows that a list could not

I built the list first, grouped by family, the way I would normally organise notes. Then I
sorted it by date and a lot of it read differently. Six things I could not see before.

### 1. Some ideas arrive years before they matter

MQA is **6 November 2019**. GQA is **22 May 2023**.

Grouped by family these sit next to each other under "KV-cache tricks" and look like they
came from the same moment. Sorted by date there is a three-and-a-half year gap, and the
gap is the interesting part.

MQA was not ignored because it was wrong. It was ignored because in 2019 hardly anyone was
serving long contexts to lots of users at once, so it solved a problem nobody had yet.
When models finally got real traffic, a gentler version of the same idea became standard
within months.

That reframed the whole exercise for me: **a mechanism shows up when its bill shows up,
not when someone thinks of it.**

### 2. The empty years are informative

2015 to 2018 is almost bare. 2023 is a wall.

I had assumed research output roughly tracks how many ideas are available. It does not —
it tracks when something starts hurting. 2023 is crowded because two bills arrived
together: KV cache (GQA, MQA's revival, PagedAttention) and context length (Positional
Interpolation, NTK-aware scaling, YaRN, attention sinks, sliding windows in production).

Both are *serving* problems, not research problems. They cluster in the year models
stopped being demos and started having users. Grouped by family, those eight or nine
entries scatter across five sections and the pattern disappears completely.

### 3. You can watch one idea end another

The approximation wave — Linformer (8 Jun 2020), Performer (30 Sep 2020) — essentially
stops after FlashAttention (27 May 2022).

Nobody refuted those methods. They just stopped being worth it, because FlashAttention
showed exact attention could be faster in wall-clock terms if you kept it in fast memory.
The field had been optimising FLOPs when the real constraint was memory bandwidth.

A list cannot say "this one ended that one." Two dates next to each other can.

### 4. Linear attention leaves and comes back different

Linear attention appears in **June 2020**, goes quiet, and returns from **2024** onward —
but never in its original form. It comes back as one layer type among several: MiniMax-01
runs seven linear layers per full one, Kimi Linear runs three, HySparse keeps five full
layers out of forty-nine.

I did not expect that shape. Every serious attempt to *replace* attention has ended up
putting it back as a minority. You only notice the round trip if the entries are in time
order.

### 5. Having the idea and being able to use it are different dates too

The delta rule is **February 2021**. It was clearly better than plain linear attention and
then went nowhere for three years, because its read-before-write step could not be
parallelised. What changed in **June 2024** was not the idea — it was an algorithm for
computing it in chunks.

One mechanism, two dates that matter. Grouped by family they collapse into one entry and
the three-year delay becomes invisible.

### 6. Your arc holds up

You said we would watch the field change its mind — exactness, then memory, then length,
then memory again. Reading it off the dates:

| | when | what |
|---|---|---|
| exactness | 2022 | FlashAttention beats the approximations by being exact and IO-aware |
| memory | mid-2023 | GQA becomes the default almost overnight |
| length | mid–late 2023 | PI, NTK-aware, YaRN, attention sinks, sliding windows |
| memory again | 2024 | MLA compresses the cache; DeltaNet and Gated DeltaNet make the state trainable |

It checks out. The one thing I would add is a fifth beat: **2025–26 stops looking for a
single answer** and settles on a schedule of several mechanisms across depth. That is why
the open question now is a ratio — how many cheap layers per expensive one — rather than
which mechanism wins.

### What this let me predict, and one thing it did not

Reading the trend, I expected the survivors to be trainable sparsity, compressed caches
and hybrid layer schedules. Checking what 2026 actually shipped: **MLA** in Kimi K2.5,
GLM-5 and Ling 2.5, **Gated DeltaNet** through Qwen3.5, **DSA** across several mid-2026
releases. That part held.

The part I got wrong is more useful. **xLSTM** (7 May 2024) published a scaling curve that
Pareto-dominates Transformers on loss per unit of compute — and nobody shipped it. Being
better on the number everyone says they care about was not enough. Kernels, tooling and
inertia are apparently part of the mechanism too, and nothing in the dates would have told
me that.

---

## Question 2 bonus — a mechanism not covered in the session

### Log-Linear Attention

| | |
|---|---|
| **arXiv** | [2506.04761](https://arxiv.org/abs/2506.04761) |
| **v1 date** | **5 June 2025** |
| **Venue** | ICLR 2026 |
| **Source for the date** | The arXiv API `published` field, checked automatically by `tools/verify_dates.py`, which also verifies the returned title is *"Log-Linear Attention"* so a transposed identifier cannot pass |

**Why it is worth a place.** Every efficient-attention method on the required list picks
one of two positions: a fixed-size state (linear attention, DeltaNet, Mamba) or full
quadratic attention. The fixed state is the ceiling of the entire linear family — those
models are RNNs underneath, and a constant-size hidden state hard-limits how much context
can be represented.

Log-linear attention replaces that fixed state with a **logarithmically growing set of
hidden states**: O(n log n) compute, logarithmic memory, and it still keeps the
matmul-heavy parallel form that makes linear attention trainable on real hardware. It is a
framework rather than one model — the paper instantiates it on both Mamba-2 and Gated
DeltaNet.

It is on the site with its costs stated, the main one being that it **gives up the exact
thing linear attention was for**. Flat memory was the selling point; logarithmic growth is
still growth, and at extreme context a genuinely fixed state wins.

### Why I am not claiming GSA for this

The session covered **GSA — Gated Slot Attention**
([arXiv 2409.07146](https://arxiv.org/abs/2409.07146), v1 11 September 2024, NeurIPS 2024),
which appears nowhere in the assignment's written list. It is a linear-attention method
with bounded memory slots and adaptive forgetting, and it is a genuinely different thing
from GQA — one letter apart, different category entirely.

But it does not qualify for this bonus, because you did cover it — you said "GQA GSA" in
the session. It is missing from the written list, which is a smaller claim. I nearly
submitted it as a find before rereading the question.

It did change how the site is checked, though: `tools/check_completeness.py` now validates
against **two** lists — the assignment's 18, and the mechanisms actually taught in the
session. Checking only the written list would have silently dropped GSA.

---

## Two dates worth a second look

You asked to be told if we caught anything, so — both offered as things to check rather
than as claims.

**The Transformer.** The session put it at "2018 and 17". *Attention Is All You Need* is
arXiv v1 **12 June 2017**, NeurIPS December 2017; there is no 2018 version. Easy to slip
on, since BERT (October 2018) is when Transformers became unavoidable — but it moves the
timeline's starting point by a year, which matters more here than it normally would.

**DroPE has a near-twin.** This one caught me out while building, and I suspect it would
catch anyone:

| | **DroPE** | **DRoPE** |
|---|---|---|
| arXiv | 2512.12167 | 2503.15029 |
| v1 | 13 Dec 2025 | 19 Mar 2025 |
| Field | LLM context extension | autonomous-driving agent trajectories |

One letter of casing apart, nine months apart, unrelated fields. The mechanism described
in the session — train with RoPE, remove it, extend — is unambiguously the first one. Ask
an agent to "explain DroPE" and there is a real chance it writes up the driving paper with
complete confidence. Worth warning the cohort about. A few more of these are on the site's
**Fact-check** section.
