# Kronecker Embedding V2

Two of the instructor's five extension problems, chosen deliberately and proven **separately**
("each are separate, don't try and mix them" — the instructor's own instruction, and the
approach confirmed with the person this was built for). Each track ships a standalone,
model-free analytic proof and a small trained transformer that asks a second, honestly
separate question: does the construction actually help a model in practice?

**Live dashboards** (GitHub Pages):
[**Track A**](https://rahulni.github.io/Indic_LLM/7_embed_research/track_a_numeral_crt/submission_artifacts/dashboard.html) ·
[**Track B**](https://rahulni.github.io/Indic_LLM/7_embed_research/track_b_holographic_binding/submission_artifacts/dashboard.html) ·
[Landing page](https://rahulni.github.io/Indic_LLM/7_embed_research/submission_artifacts/index.html)

Both dashboards are single self-contained HTML files with the data baked in — they also open
straight from disk with no server:
[Track A](track_a_numeral_crt/submission_artifacts/dashboard.html) ·
[Track B](track_b_holographic_binding/submission_artifacts/dashboard.html) ·
[index](submission_artifacts/index.html)

**Docs:** [**Plain-English explainer**](EXPLAIN_LIKE_IM_10.md) (no maths needed — start here if
you want the ideas without the notation) · [Architecture & design decisions](ARCHITECTURE.md) ·
[Citations](CITATIONS.md) · [Track A README](track_a_numeral_crt/README.md) ·
[Track B README](track_b_holographic_binding/README.md)

---

## What "Kronecker Embedding V1" is

**Shravan (2026), [*Kronecker Embeddings: Byte-Level Structured Token Representations for
Parameter-Efficient Language Models*](https://arxiv.org/abs/2605.29459)** ([code](https://github.com/theschoolofai/kronecker-embeddings)).
Equation 1, §3.2:

```
kappa(b) = (1/sqrt(L)) * SUM_{p=1..L}  c_{b_p} (x) p_p
```

`c_{b_p}` is a one-hot over the **byte** value (`d_c = 256`), `p_p` a one-hot over byte
position (`d_p = 32`), so `D = 8192`, followed by one learned `Linear(D, d_model)` — replacing
the `|V| x d_model` embedding table entirely. Tokens longer than 32 **bytes** are truncated;
the codec is deterministic and not invertible.

> [!NOTE]
> This project was built **before** that paper and repository were available to me. I verified
> that "kronecker" appeared in zero files and zero commits of this repo's prior history, and
> reconstructed the scheme from the assignment prose, labelling it a reconstruction. Audited
> against Eq. 1 afterwards, the reconstruction's structure was right but had two real
> deviations — a missing `1/√L` normaliser and a character rather than byte alphabet. Both are
> fixed, all affected numbers re-measured, and the episode is recorded in
> [`CITATIONS.md`](CITATIONS.md) and [Track B's README](track_b_holographic_binding/README.md)
> rather than smoothed over.

Two of the instructor's five stated extension problems are addressed here:

- **Track A** (the instructor's problem 1 — arithmetic-preserving embeddings): *"What if
  embeddings can store mathematical structure... when we do 9+9, the mathematical meaning
  part is itself 18."* Solved by encoding an integer via its residues modulo a set of coprime
  primes — the Kronecker/CRT decomposition of a finite abelian group.
  **Scope note:** the assignment says to append these dimensions to V1's existing code. Track A
  instead builds a self-contained digit-arithmetic experiment, so it proves the arithmetic
  claim but never literally concatenates onto V1's 8,192-dim byte codec. That is a deliberate
  scoping decision, not an oversight — the proof and the ablation are both cleaner in
  isolation — but it does mean Track A is not a drop-in patch to V1.
- **Track B** (the instructor's problem 4 — a real Fourier alternative to Kronecker): *"Why
  can't I represent each character like a Fourier wave, and just add them to make a word?"*
  Solved by replacing tensor-product binding with circular-convolution superposition, which
  removes the 32-character cap entirely at the cost of a measured, provable capacity tradeoff.

Both tracks reuse the same non-negotiable structure: **the exact/provable claim is verified
first, analytically, with no model involved — the trained transformer answers a second,
separate, honestly-labeled empirical question, and the two are never conflated.**

## Where to start reading

**In 5 minutes:** open the two dashboards linked above. Each opens with its proof table, then
an interactive widget you can type into, then the trained-model results and the limitations.
That is the whole story, in order, with every number rendered from the result files.

**In 30 minutes, to understand what was actually built**, read in this order:

| # | Read | Why |
|---|---|---|
| 1 | this README | the two problems, the headline results, the limitations |
| 2 | [`track_a_numeral_crt/proofs/analytic_proof.py`](track_a_numeral_crt/proofs/analytic_proof.py) | **the heart of Track A.** The module docstring states exactly what is proven and how; the `check_*` functions are the proof. No model, no torch — pure numpy, runs in 2s |
| 3 | [`track_b_holographic_binding/proofs/capacity_proof.py`](track_b_holographic_binding/proofs/capacity_proof.py) | **the heart of Track B.** Same shape: docstring states the claim, `theoretical_decode_accuracy()` derives the bound with assumptions written out, `sweep_cell()` measures it |
| 4 | [`ARCHITECTURE.md`](ARCHITECTURE.md) | **why** every non-obvious choice was made, and 13 judgment calls stated explicitly rather than buried — including the two mistakes that were caught and corrected |
| 5 | each track's README ([A](track_a_numeral_crt/README.md), [B](track_b_holographic_binding/README.md)) | the claim stated precisely, all results in tables, per-track limitations |
| 6 | [`CITATIONS.md`](CITATIONS.md) | every citation, with what was verified vs. what was not — including one paper title I got wrong and corrected |

**If you only care about one thing per track:**

- **Track A** — the proof is exact and unconditional; the trained model is a *negative*
  result. Read `analytic_proof.py`, then Track A README's "The missing ingredient" and
  "The instructor's literal ask, tested" sections.
- **Track B** — binding is exactly invertible for one pair and degrades predictably for many.
  Read `capacity_proof.py`, then Track B README's truncation-probe table.

**Map of the code**, if you want to trace a claim to the line that produces it:

| To understand… | Read |
|---|---|
| how a number becomes an embedding | `analytic_proof.py` → `encode_onehot`, `crt_decode_array` |
| why `+` and `×` become shifts | `analytic_proof.py` → `check_additive_per_modulus`, `mod_mul_via_log_array` |
| how a word becomes one vector | `capacity_proof.py` → `random_unitary_vector`, `sweep_cell` |
| the three Track A embedding arms | `track_a_numeral_crt/model/embeddings.py` (~120 lines, one class each) |
| the two Track B embedding arms | `track_b_holographic_binding/model/embeddings.py` |
| the shared transformer (identical across every arm) | `common/transformer.py` |
| the instructor's literal "embedding(9) knows it is 9" | `track_a_numeral_crt/model/value_train.py` |
| how nothing here can fake a PASS | `common/evidence.py` + `tools/run_all.py` |

## Quick start

```bash
pip install -r requirements.txt              # numpy + matplotlib; the proofs need nothing else
python track_a_numeral_crt/proofs/analytic_proof.py           # < 3s, exhaustive checks
python track_b_holographic_binding/proofs/capacity_proof.py   # < 20s

# the trained-model runs are opt-in and need torch
pip install -r requirements-torch.txt
python track_a_numeral_crt/model/train.py --arm crt --task add --profile fast     # smoke test, seconds
python track_a_numeral_crt/model/train.py --arm crt --task add --profile default  # real numbers, ~2 min

# the experiment that changed Track A's headline, and the instructor's literal ask
python track_a_numeral_crt/model/train.py --arm abacus --task add --profile default --random-offset 24
python track_a_numeral_crt/model/value_train.py --arm crt_value --task add --profile default

python tools/run_all.py    # proofs -> seed aggregate -> evidence.json -> both dashboards; non-zero exit on any FAIL
python -m unittest tests.test_common \
  track_a_numeral_crt.proofs.test_analytic_proof \
  track_b_holographic_binding.proofs.test_capacity_proof     # 30 tests, no GPU needed
```

## Evidence

28 of 28 requirements pass. Every row is graded by re-reading an artifact file on disk, never
from in-memory state — see [`submission_artifacts/evidence.json`](submission_artifacts/evidence.json)
for the full machine-readable list, or each track's dashboard for the same table rendered.

| Track | Proof (no model) | Trained-model runs |
|---|---|---|
| A — Kronecker Numeral Embeddings | 7/7 checks — all four ring operations exact on `[0, 1,062,347)`, run in ~2s | 6 arms × task + 5 offset/ablation + 4 value-embedding runs |
| B — Holographic Binding | 4/4 checks — unbind exact, derived bound predicts measurement, literal shift roles swept | 2 arms + dressing ablation |

Headline results are 3-seed means; one evidence row exists purely to assert that
(`Headline results are reported over multiple seeds, not a single run`), because these GPU
runs are not bit-reproducible.

## Headline results

**Track A.** The proof is unconditional: **all four ring operations** — addition, subtraction,
multiplication and division-by-units — are exactly recoverable from the embedding via fixed
shift operators, zero error, verified exhaustively wherever exhaustive was feasible. The proof
is equally explicit about the limits: division is undefined for non-units, and residue codes
provably destroy *ordering* (you cannot compare magnitudes without reconstructing).

The trained-model story took two rounds, and both are reported:

1. Initially **neither structured arm** generalized past the 1–4 digit training range (0.0% at
   test length 5), while the no-positional-signal NoPE baseline managed 35.7% ± 9.0%.
2. A reviewer pass found the cause: McLeish et al. pair Abacus embeddings with **random
   position-offset training**, which had not been implemented. Adding it produced real OOD
   generalization — **Abacus-lite 0% → 93.8% ± 7.2%** and **CRT 0% → 34.2% ± 19.1%** at test
   length 5 (3 seeds, n=200).

The honest sting is in that gap: the *learned* code exploits the mechanism far better than the
*fixed* CRT code — and CRT+offset (34.2%) only draws level with using no positional signal at
all (35.7%). On this evidence the CRT positional code does not earn its place.

The experiment closest to the instructor's literal ask — each operand a single token carrying
the fixed CRT code *of its value* — **fails almost completely (≤1% exact match)** for the CRT
code *and* for a learned control, which tie exactly. Handing a model provably-exact arithmetic
structure does not make it able to do arithmetic; it still has to learn to read that structure
out. The one real positive is efficiency parity: the fixed code matches the learned one on
both loss and accuracy with **10× fewer embedding parameters**. See
[Track A's README](track_a_numeral_crt/README.md).

**Track B.** Single-pair unbind is error-free, and decode accuracy degrades with length
exactly as a **derived** SNR bound predicts — that bound is our own write-up of the standard
VSA argument (assumptions stated in the docstring, validated against measurement by a unit
test), deliberately *not* attributed to Plate since we could not verify a specific equation
from the primary text. The instructor's literal phrasing ("each character a Fourier wave")
maps to deterministic *shift* roles; swept head-to-head against random-phase roles, the
literal reading is **not worse** — contradicting our own prior expectation, reported over the
prediction.

The controlled truncation probe is unambiguous: **the Kronecker arm cannot distinguish two
words differing only after byte 32 (cosine similarity 1.0000, exactly)**, while the holographic
arm can (0.7895) — with *zero* learned embedding parameters against Kronecker's 1,573,056.

But on the metric the corpus actually measures, **Kronecker wins clearly**: 155.7 ± 1.9 vs
190.0 ± 7.0 perplexity, a 34-point gap far outside the seed spread. That gap *doubled* when a
bug in our own baseline was fixed — the first version omitted Eq. 1's `1/√L` normaliser and
reported Kronecker at 172.4, i.e. the bug had been flattering this project's own alternative.
An ablation rules out "it's just the learned projection": giving the holographic arm one made
it *worse* (213.1). See [Track B's README](track_b_holographic_binding/README.md).

## Honest limitations

> [!WARNING]
> - **Track A's exact-arithmetic claim holds only within a bounded range** — `[0, 1,062,347)`
>   for the value code, `[0, 42)` for the position code. Outside that range the construction
>   wraps modulo N (demonstrated deliberately in the proof, not hidden).
> - **A trained transformer's behavior does not demonstrate it performs exact arithmetic in
>   embedding space.** That would require circuit-level mechanistic verification (in the style
>   of Nanda et al. 2023 or Zhong et al. 2023) — explicitly out of scope here.
> - **Neither trained-model experiment used a large model or a long training run.** Every run
>   finishes in minutes on a single consumer GPU by design, so nothing here claims to
>   generalize to larger scale without re-running.
> - **Even with random-offset training, generalization collapses by test length 6-7.** The
>   gain is real but bounded; McLeish et al.'s larger looped/recurrent architecture is not
>   reimplemented here, and their far more extreme length ranges are not approached.
> - **Track B's capacity bound is our own derivation, not a formula taken from Plate.** We
>   could not verify a specific numbered equation in the primary text, so rather than
>   attribute one we derived the standard SNR argument ourselves, wrote every assumption into
>   the docstring, and validated it against our own measurements with a test. See
>   [`CITATIONS.md`](CITATIONS.md) for exactly what was and wasn't verified.
> - **Track B's word-level perplexity numbers are not comparable** to standard char-level
>   tiny_shakespeare benchmarks reported elsewhere — different tokenization, different task.
> - **These GPU runs are not bit-reproducible.** No deterministic-kernel flags are set, so two
>   identical commands with the same seed gave 91.5% and 85.5%. Every headline number is a
>   3-seed mean; treat the *ordering* of the arms as the finding, not the exact percentages.
> - **One reported result was wrong before it was right.** The first value-embedding run made
>   the CRT arm look broken; the cause was initialisation scale, not the representation. It
>   was fixed and re-run, and the artifact is documented rather than deleted (Track A README).
>   Assume other unforced errors of that kind are possible in work at this scale.
> - This is coursework-scale research (a few million parameters, minutes of compute), not a
>   claim of state-of-the-art performance on either task.

## Repository layout

Every `.py` file below is small and single-purpose; the two `proofs/` files are the ones worth
reading first.

```
7_embed_research/
  README.md  ARCHITECTURE.md  CITATIONS.md
  requirements.txt              numpy + matplotlib (proofs only, no GPU)
  requirements-torch.txt        torch, opt-in, for the trained-model runs

  common/                       shared and track-agnostic
    transformer.py                the decoder-only trunk EVERY arm shares (embeddings pluggable)
    evidence.py                   PASS/FAIL harness; grades only from files on disk
    device.py  seeding.py  plotting.py

  track_a_numeral_crt/
    proofs/analytic_proof.py      ** the proof: 4 ring ops exact on [0, 1062347) **
    proofs/test_analytic_proof.py  + the RNS ordering limitation, as a test
    proofs/make_plots.py          clock-structure and shift-scatter figures
    data/generate_arithmetic.py   digit tokenizer + random-offset augmentation
    model/embeddings.py           the 3 arms: NoPE / Abacus-lite / CRT
    model/train.py                --arm --task --profile --random-offset --seed
    model/value_train.py          ** the instructor's literal ask: operand = 1 token **
    model/evaluate.py             greedy decode + length-generalization eval
    results/                      every run's JSON (committed); *.pt weights are not
    submission_artifacts/         dashboard.html + run logs

  track_b_holographic_binding/
    proofs/capacity_proof.py      ** the proof: exact unbind + derived capacity bound **
    proofs/test_capacity_proof.py  asserts the bound actually predicts measurement
    proofs/make_plots.py          capacity curve, interference, role comparison
    data/corpus.py                tiny_shakespeare + synthetic long-word probe set
    model/embeddings.py           the 2 arms: Kronecker 32-slot / holographic
    model/train.py  evaluate.py   word-level LM + the truncation probe
    results/  submission_artifacts/

  submission_artifacts/          index.html, evidence.json, seed_aggregate.json
  tools/
    run_all.py                    proofs -> aggregate -> evidence -> dashboards
    aggregate_seeds.py            multi-seed mean +/- std
    build_dashboard.py  build_index.py  chart_kit.py
  tests/test_common.py           trunk, evidence harness, random-offset invariants
```
