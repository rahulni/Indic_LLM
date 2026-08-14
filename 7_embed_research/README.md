# Kronecker Embedding V2

Two of the instructor's five extension problems, chosen deliberately and proven **separately**
("each are separate, don't try and mix them" — the instructor's own instruction, and the
approach confirmed with the person this was built for). Each track ships a standalone,
model-free analytic proof and a small trained transformer that asks a second, honestly
separate question: does the construction actually help a model in practice?

**Live dashboards:** [Track A](https://claude.ai/code/artifact/7b2f859c-7f7b-4d1e-9643-7883c0faeac8) ·
[Track B](https://claude.ai/code/artifact/439c4917-c4c5-489c-b279-2e7fb3dc826f)
(also viewable locally, no server needed: [Track A](track_a_numeral_crt/submission_artifacts/dashboard.html) ·
[Track B](track_b_holographic_binding/submission_artifacts/dashboard.html) ·
[Landing page](submission_artifacts/index.html))

**Docs:** [Architecture & design decisions](ARCHITECTURE.md) · [Citations](CITATIONS.md) ·
[Track A README](track_a_numeral_crt/README.md) · [Track B README](track_b_holographic_binding/README.md)

---

## What "Kronecker Embedding V1" was

Described only verbally in the assignment (no code for it exists anywhere in this repository
— checked exhaustively against the full git history before writing a line of this project):
every word, regardless of length, is encoded into a fixed **32 character-position slots**.
Short words waste slots; words longer than 32 characters are silently truncated. Two of the
instructor's five stated extension problems are addressed here:

- **Track A** (the instructor's problem 1 — arithmetic-preserving embeddings): *"What if
  embeddings can store mathematical structure... when we do 9+9, the mathematical meaning
  part is itself 18."* Solved by appending new dimensions beyond the 32 character slots that
  encode an integer via its residues modulo a set of coprime primes — the Kronecker/CRT
  decomposition of a finite abelian group.
- **Track B** (the instructor's problem 4 — a real Fourier alternative to Kronecker): *"Why
  can't I represent each character like a Fourier wave, and just add them to make a word?"*
  Solved by replacing tensor-product binding with circular-convolution superposition, which
  removes the 32-character cap entirely at the cost of a measured, provable capacity tradeoff.

Both tracks reuse the same non-negotiable structure: **the exact/provable claim is verified
first, analytically, with no model involved — the trained transformer answers a second,
separate, honestly-labeled empirical question, and the two are never conflated.**

## Quick start

```bash
pip install -r requirements.txt              # numpy, matplotlib -- the proofs need nothing else
python track_a_numeral_crt/proofs/analytic_proof.py     # < 2 seconds, exhaustive checks
python track_b_holographic_binding/proofs/capacity_proof.py   # < 20 seconds

pip install -r requirements-torch.txt         # opt-in, only needed for the trained-model runs
python track_a_numeral_crt/model/train.py --arm crt --task add --profile fast      # smoke test, seconds
python track_a_numeral_crt/model/train.py --arm crt --task add --profile default   # real numbers, ~3 min

python tools/run_all.py                       # proofs -> evidence.json -> both dashboards, exits non-zero on any FAIL
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
*fixed* CRT code — and the CRT positional code with offsets only draws level with using no
positional signal at all. On this evidence it does not earn its place. And the experiment closest to the instructor's literal ask — each operand a
single token carrying the fixed CRT code *of its value* — **fails almost completely (≤1%
exact match)** for the CRT code *and* for a learned control, which tie exactly. Handing a
model provably-exact arithmetic structure does not make it able to do arithmetic. The one real
positive is efficiency parity: the fixed code matches the learned one on both loss and
accuracy with **10× fewer embedding parameters**. See
[Track A's README](track_a_numeral_crt/README.md).

**Track B.** Single-pair unbind is error-free, and decode accuracy degrades with length
exactly as a **derived** SNR bound predicts — that bound is our own write-up of the standard
VSA argument (assumptions stated in the docstring, validated against measurement by a unit
test), deliberately *not* attributed to Plate since we could not verify a specific equation
from the primary text. The instructor's literal phrasing ("each character a Fourier wave")
maps to deterministic *shift* roles; swept head-to-head against random-phase roles, the
literal reading is **not worse** — contradicting our own prior expectation, reported over the
prediction.

The controlled truncation probe is unambiguous: **the Kronecker/tensor arm cannot distinguish
two words differing only after character 32 (cosine similarity 1.0000, exactly)**, while the
holographic arm can (0.7895) — with *zero* learned embedding parameters against the Kronecker
arm's 209,088. On raw perplexity the Kronecker arm is ahead (172.4 ± 3.1 vs 190.0 ± 7.0 over
3 seeds — a real gap, larger than the seed spread); an ablation shows this is **not** explained
by its learned projection, since giving the holographic arm one made it *worse* (213.1). See
[Track B's README](track_b_holographic_binding/README.md).

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
> - **Track B's holographic capacity bound is stated qualitatively, sourced from a secondary
>   citation check, not a page-and-equation reference pulled directly from Plate (1995/2003).**
>   See [`CITATIONS.md`](CITATIONS.md) for exactly what was and wasn't verified.
> - **Track B's word-level perplexity numbers are not comparable** to standard char-level
>   tiny_shakespeare benchmarks reported elsewhere — different tokenization, different task.
> - This is coursework-scale research (a few million parameters, minutes of compute), not a
>   claim of state-of-the-art performance on either task.

## Repository layout

```
7_embed_research/
  README.md  ARCHITECTURE.md  CITATIONS.md
  requirements.txt  requirements-torch.txt
  common/                        shared trunk, device/seeding/evidence/plotting helpers
  track_a_numeral_crt/           Track A: proof, data generator, model, results, dashboard
  track_b_holographic_binding/   Track B: proof, corpus prep, model, results, dashboard
  submission_artifacts/          top-level landing page + combined evidence.json
  tools/                         run_all.py, build_dashboard.py, build_index.py, chart_kit.py
  tests/                         shared-module smoke tests
```
