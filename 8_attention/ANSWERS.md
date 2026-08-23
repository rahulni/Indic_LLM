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

Six things became visible only after sorting by date. None of them survive being written
as a grouped list, which is the point.

### 1. "Launched" and "started mattering" are different dates, sometimes by years

**MQA is 6 November 2019. GQA is 22 May 2023.**

In any list these two sit adjacent under "KV-cache tricks" and read as siblings from the
same moment. On a time axis there is a **three-and-a-half year hole** between them, and
the hole is the interesting part. MQA was not ignored because it was wrong. It was ignored
because in 2019 almost nobody was serving long contexts to many concurrent users, so it
solved a problem that did not yet hurt. The moment models met real traffic, a gentler
version of the same idea became the default within months.

The general lesson, which only a timeline can teach: **a mechanism arrives when the bill
it pays arrives, not when the idea is available.**

### 2. The empty stretches say as much as the crowded ones

**2015 to 2018 is nearly barren. 2023 is a wall.**

The field's output does not track how many ideas were available — it tracks when something
started hurting. 2023 is dense because two bills came due at once: the KV cache (GQA,
MQA's revival, PagedAttention) and context length (Positional Interpolation, NTK-aware
scaling, YaRN, attention sinks, sliding windows in production). Both of those are *serving*
problems. They appear the year models stopped being research artifacts and started having
users.

A list sorted by family would scatter those across five different sections and destroy the
single most obvious pattern in the data.

### 3. You can see one idea kill another

**The approximation wave — Linformer (Jun 2020), Performer (Sep 2020) — stops almost
exactly when FlashAttention lands (27 May 2022).**

Those methods were not refuted. They were made pointless: FlashAttention showed that exact
attention, tiled to stay in fast memory, beat the approximations in wall-clock terms. The
field had been optimising the wrong quantity — FLOPs instead of memory bandwidth.

A list cannot express "this superseded that." Adjacency in time can, and it is visible here
as a cluster that simply stops.

### 4. Linear attention makes a round trip, and comes back changed

Linear attention appears in **June 2020**, is largely abandoned, and returns from **2024**
onward — but never in the form it left. It comes back as **one layer type among several**:
MiniMax-01 at seven linear layers per full one, Kimi Linear at three, HySparse keeping five
full layers out of forty-nine.

The round trip is invisible without time. And its shape is the actual finding: every
attempt to *replace* attention has ended up **reintroducing it as a minority**.

### 5. An idea's date and its usable date are also different

The **delta rule is February 2021**. It was clearly better than plain linear attention and
went almost nowhere for three years, because its read-before-write recurrence could not be
parallelised. The thing that changed in **June 2024** was not the idea but an algorithm for
computing it chunkwise.

Two dates, one mechanism — and on a family-grouped list they collapse into a single entry
that hides why the gap existed.

### 6. Your arc checks out, and it is legible in the dates

You said we would watch the field change its mind: exactness, then memory, then length,
then memory again. Reading it off the timeline:

| | when | what |
|---|---|---|
| **exactness** | 2022 | FlashAttention beats the approximations by being exact and IO-aware |
| **memory** | mid-2023 | GQA becomes the default almost overnight |
| **length** | mid-late 2023 | PI, NTK-aware, YaRN, attention sinks, sliding windows |
| **memory again** | 2024 | MLA compresses the cache; DeltaNet and Gated DeltaNet make the state trainable |

It holds. What I would add is a fifth beat: **2025–26 stops looking for one answer at all**
and settles on a *schedule* of several mechanisms across depth — which is why the open
question in 2026 is a ratio, not a mechanism.

### What that lets you guess — and one thing it does not

Checked against what 2026 actually shipped, the convergence held: **MLA** in Kimi K2.5,
GLM-5 and Ling 2.5; **Gated DeltaNet** through Qwen3.5; **DSA** across several mid-2026
releases.

But one thing on this timeline the trends do not explain. **xLSTM** (7 May 2024) published
a scaling curve that Pareto-dominates Transformers on loss per unit compute — and **nobody
shipped it.** Being better on the metric everyone claims to optimise turned out not to be
sufficient. Ecosystem, kernels and inertia are apparently part of the mechanism too, and no
amount of staring at the dates would have told me that.

---

## Question 2 bonus — a mechanism not covered in the session

### Log-Linear Attention

| | |
|---|---|
| **arXiv** | [2506.04761](https://arxiv.org/abs/2506.04761) |
| **v1 date** | **5 June 2025** |
| **Venue** | ICLR 2026 |
| **Source for the date** | The arXiv API `published` field, checked automatically by `tools/verify_dates.py`, which also verifies the returned title is *"Log-Linear Attention"* so a transposed identifier cannot pass |

**Why it matters, in one paragraph.** Every efficient-attention method on the required
list picks one of two positions: a fixed-size state (linear attention, DeltaNet, Mamba) or
full quadratic attention. The fixed state is the ceiling of the entire linear family —
those models are RNNs at heart, and a constant-size hidden state is a hard limit on what
context can be represented. Log-linear attention replaces that fixed state with a
**logarithmically growing set of hidden states**, giving O(n log n) compute and logarithmic
memory, and it still admits the matmul-rich parallel form that makes linear attention
trainable on real hardware. It is a framework rather than a single model — the paper
instantiates it on both Mamba-2 and Gated DeltaNet.

It is on the site with honest costs, the main one being that **it gives up the very thing
linear attention was for**: flat memory. Logarithmic growth is still growth, and at extreme
context a genuinely fixed state wins.

### A correction I should make about GSA

I want to be precise here rather than claim more than I earned.

The session covered **GSA — Gated Slot Attention** ([arXiv 2409.07146](https://arxiv.org/abs/2409.07146),
v1 11 September 2024, NeurIPS 2024), which appears **nowhere in the assignment's written
list**. It is a linear-attention method with bounded memory slots and adaptive forgetting,
and it is genuinely different from GQA — one letter apart, completely different category.

But it does **not** qualify for this bonus, because you *did* cover it — you said "GQA GSA"
in the session. It is missing from the written list, which is a smaller and different
claim. Presenting it as something you had not covered would be an overclaim, and the whole
point of this project was not making those.

It did change how the site is verified: `tools/check_completeness.py` validates against
**two** lists — the assignment's 18, and the mechanisms actually taught in the session.
Checking only the written list would have silently dropped GSA.

---

## One thing I think is wrong in the session

Offered because you asked to be told.

The Transformer was attributed to **"Vaswani … 2018 and 17."** *Attention Is All You Need*
is arXiv v1 **12 June 2017**, published at NeurIPS in December 2017. There is no 2018
version. The confusion is understandable — BERT (October 2018) is when Transformers became
unavoidable — but it shifts the timeline's zero point by a year.

There is also a naming trap in the required list worth flagging, because an AI agent walks
straight into it. **DroPE** and **DRoPE** are two different papers:

| | **DroPE** | **DRoPE** |
|---|---|---|
| arXiv | 2512.12167 | 2503.15029 |
| v1 | 13 Dec 2025 | 20 Mar 2025 |
| Field | LLM context extension | autonomous-driving agent trajectories |

They differ by one letter of casing, are nine months apart, and are in unrelated fields.
The mechanism described in the session — train with RoPE, remove it, extend — is
unambiguously the first. Ask an agent for "DroPE" and there is a real chance it writes up
the driving paper instead. More of these are on the site's **Fact-check** section.
