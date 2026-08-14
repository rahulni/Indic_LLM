# Track A — Kronecker Numeral Embeddings

[← top-level README](../README.md) · [Architecture](../ARCHITECTURE.md) · [Citations](../CITATIONS.md) ·
[**Live dashboard**](https://rahulni.github.io/Indic_LLM/7_embed_research/track_a_numeral_crt/submission_artifacts/dashboard.html) ·
[local copy](submission_artifacts/dashboard.html)

## The problem

From the instructor's assignment: *"What if embeddings can store mathematical structure as
well. Say 9... somehow it has stored the meaning of 9 (in absolute math terms), such that
when we actually do 9+9, the mathematical meaning part of the embeddings is itself 18! When
we do 9\*9... it becomes 81!"* — with the constraint that this should **append** new
dimensions to the existing 32 character slots, not replace them.

## The claim, stated precisely

For integers `n` in `[0, N)`, represent `n` by its residues modulo a set of pairwise-coprime
primes `p_1, ..., p_k` (`N = p_1 * ... * p_k`) — the Kronecker/CRT decomposition
`Z/NZ ≅ Z/p_1Z × ... × Z/p_kZ`. Because this is a *ring* isomorphism, not just a group one:

> For all `a, b` with `a + b < N`: `decode(shift(encode(a), residues(b))) = encode(a + b)`,
> exactly, where `shift` is a fixed per-slot cyclic permutation (no learned parameters).
>
> For all `a, b` with `a * b < N`: the same holds for multiplication, via a fixed per-slot
> discrete-log re-indexing (valid because every modulus here is prime, so its multiplicative
> group is cyclic).

Two modulus sets are used, for different purposes — see [ARCHITECTURE.md](../ARCHITECTURE.md#judgment-calls-explicit-not-silently-resolved) point 5:

| Purpose | Primes | N | Width |
|---|---|---|---|
| Digit **position** code (reused as the transformer's positional embedding) | `[2, 3, 7]` | 42 | 12 |
| Numeral **value** code (the exhaustive exact-arithmetic proof) | `[11, 13, 17, 19, 23]` | 1,062,347 | 83 |

## Method summary

**The proof** (`proofs/analytic_proof.py`) encodes each residue as a one-hot vector,
concatenates across primes, and verifies the shift/decode identity above — exhaustively where
computationally feasible, sampled (with the sample size always stated) where not. It runs in
under 2 seconds, needs no GPU, and needs nothing but numpy.

**The trained-model experiment** is a second, separate question: does a *fixed* CRT
positional code change a small transformer's ability to generalize arithmetic to longer
operands than it was trained on, compared to a *learned* positional code (a lightweight
reimplementation of Abacus embeddings, McLeish et al. 2024) or *no* positional code at all
(NoPE — see [ARCHITECTURE.md](../ARCHITECTURE.md), point 1, for why this is a fair baseline
and not a strawman)? All three arms share an identical transformer trunk and identical
per-digit tokenization — the embedding is the only thing that differs.

## Evidence

From the last run (`results/analytic_proof_report.json`, `results/*_default.json`):

| Check | Coverage | Result |
|---|---|---|
| CRT bijection (position code, N=42) | exhaustive, n=42 | PASS |
| CRT bijection (value code, N=1,062,347) | exhaustive, n=1,062,347 | PASS |
| Additive homomorphism per modulus | exhaustive, n=62 and n=1,469 | PASS |
| Full-pipeline addition | exhaustive n=5,311,735 + sampled n=1,000,000 | PASS |
| Multiplicative homomorphism via discrete log (incl. zero-case) | exhaustive, n=62 and n=1,469 | PASS |
| Full-pipeline multiplication + explicit wraparound demo | exhaustive n=1,062,961 | PASS |
| Full-pipeline subtraction + explicit underflow demo | exhaustive n=5,313,204 | PASS |
| Division by units via negated discrete log (+ both failure modes) | exhaustive n=1,386 + sampled n=100,000 | PASS |

### "Can we describe whole mathematics?"

All four ring operations are exact within `[0, N)`, each realized by the *same* fixed shift
machinery: addition rotates each residue clock forward, subtraction rotates it back,
multiplication rotates in discrete-log space, division rotates back in log space. What the
proof also shows — and states rather than hides — is where this stops:

- **Division is only exact for units.** If `gcd(b, N) > 1` (e.g. `b = 11`, one of the moduli)
  then `b` has no inverse and division is undefined. If `b` is a unit but does not divide `a`,
  the result is the well-defined ring element `a·b⁻¹ mod N` — *not* integer division
  (`7·2⁻¹ mod N = 531,177`, not `3`). Both cases are asserted in the proof, not glossed.
- **Order is destroyed.** Residue codes preserve the ring structure but *not* magnitude
  comparison: you cannot tell whether `a < b` from the residues without a full CRT
  reconstruction. This is the classical, well-known RNS limitation, and it is the honest
  counterweight to the four homomorphism results — it is asserted as its own unit test
  (`test_magnitude_comparison_is_not_preserved`).

**Trained-model results** (2.68-2.71M params, trained on 1-4 digit operands, ~2 min/run on an
RTX 3070 Laptop GPU). Addition is reported as **mean ± std over 3 seeds**, `n=200` per test
length; multiplication is single-seed (`n=50`) and labelled as such.

Addition, exact-match accuracy by test digit-length (lengths >4 were never trained on):

| Arm | 4 (ID) | 5 (OOD) | 6 (OOD) | 7 (OOD) |
|---|---|---|---|---|
| Baseline (NoPE) | 96.0% ± 3.1% | **35.7% ± 9.0%** | 2.2% ± 2.3% | 0.0% |
| Abacus-lite (learned) | 100.0% ± 0.0% | 0.0% ± 0.0% | 0.0% | 0.0% |
| CRT/Kronecker (fixed) | 98.7% ± 2.3% | 0.0% ± 0.0% | 0.0% | 0.0% |

Multiplication, single seed, `n=50` (harder for every arm, as prior work reports):

| Arm | 1 | 2 | 3 | 4 (ID) | 5+ (OOD) |
|---|---|---|---|---|---|
| Baseline (NoPE) | 100% | 36% | 4% | 0% | 0% |
| Abacus-lite (learned) | 100% | 92% | 8% | 0% | 0% |
| CRT/Kronecker (fixed) | 100% | 74% | 4% | 0% | 0% |
| CRT + learned dressing | 100% | 78% | 4% | 0% | 0% |

> [!NOTE]
> **The NoPE baseline is the only arm that generalizes at all here** (35.7% at length 5, where
> both structured arms score exactly 0%). That is not an accident of this setup — it is the
> central finding of Kazemnejad et al. (2023), who show NoPE causal transformers out-generalize
> APE/ALiBi/RoPE on reasoning tasks. Adding *any* explicit place-value signal, learned or fixed,
> made length generalization worse until the training mechanism below was added.

**Reading this honestly:** every arm reaches ~100% in-distribution addition accuracy, and
every arm — including the CRT arm this track is built around — collapses to 0% by test
length 6. Multiplication is harder for every arm even in-distribution, consistent with prior
literature.

### The missing ingredient: random-offset training

That all-arms-fail result turned out to be an artifact of a **missing training mechanism**,
not a property of the embeddings. McLeish et al. (2024) pair Abacus embeddings with
**random position-offset training**: each training example gets one random offset added to
every digit's place-value index, so the model can never anchor on "the units digit is always
index 0". Evaluation always uses offset 0. For the CRT arm this is literally a rotation of
every residue clock — the same fixed shift operator the proof verifies, used as augmentation.

Run with `--random-offset 24`, mean ± std over 3 seeds, `n=200` per length:

| Arm | len 5, no offset | len 5, **with offset** | len 6, with offset | len 7, with offset |
|---|---|---|---|---|
| Abacus-lite (learned) | 0.0% ± 0.0% | **93.8% ± 7.2%** | 35.7% ± 15.8% | 7.7% ± 8.8% |
| CRT/Kronecker (fixed) | 0.0% ± 0.0% | **34.2% ± 19.1%** | 0.7% ± 0.8% | 0.0% |

Out-of-distribution generalization appears where there was none, and it now extends two
lengths beyond the training range rather than zero.

**The honest sting: the *learned* Abacus code exploits the mechanism far better than the
*fixed* CRT code** (93.8% vs 34.2% at length 5). Algebraic structure in the embedding is not,
by itself, what buys length generalization — the training mechanism is, and a learnable code
makes better use of it. Note also that the fixed CRT code with offsets (34.2%) only draws
level with the *no-embedding-at-all* NoPE baseline (35.7%): on this evidence the CRT
positional code does not earn its place.

Multiplication remains unsolved even with offsets (33.5% at length 2, 0% by length 4).

> [!WARNING]
> **The spreads here are large** (±19.1% for CRT+offset, ±15.8% for Abacus+offset at length 6)
> and these GPU runs are **not bit-reproducible** — no deterministic-kernel flags are set, so
> two identical commands with the same seed produced 91.5% and 85.5%. Every number above is a
> 3-seed mean for that reason. Treat the ordering of the arms as the finding, not the exact
> percentages.

### The instructor's literal ask, tested

Everywhere above the CRT code is a *positional* signal. This experiment
(`model/value_train.py`) is the thing actually described in the assignment: each whole
operand is a **single token** whose embedding is the fixed 83-dim CRT code *of the number
itself*. Operands are capped at 3 digits so both `a+b` and `a*b` stay inside the proven range.

| Operand embedding | Task | Embedding params | Final loss | Exact-match (n=200) |
|---|---|---|---|---|
| CRT value code (fixed) | add | **19,392** | 1.658 → 1.049 | 0.5% |
| Learned embedding | add | 195,264 | 1.567 → 1.028 | 0.5% |
| CRT value code (fixed) | mul | **19,392** | 1.995 → 1.297 | 1.0% |
| Learned embedding | mul | 195,264 | 1.854 → 1.172 | 1.0% |

**A clear negative result, and the most important one in this track.** Both arms train (loss
falls comparably) and both then fail the task almost completely. Handing a transformer an
embedding that *provably* contains exact arithmetic structure does **not** make it able to do
arithmetic — it still has to learn to *read* that structure out, and mapping one operand token
to a multi-digit answer is evidently too hard at this scale and budget. This is precisely why
the analytic proof and the trained-model claims are kept rigorously separate throughout this
project.

The one positive note is efficiency parity: the fixed CRT code matches the learned embedding's
accuracy *and* its loss trajectory using **10× fewer embedding parameters** (19,392 vs
195,264), and unlike the learned table it would extend to unseen operands for free. That is a
real property — but parity at ~1% accuracy is not a win, and it is not presented as one.

> [!NOTE]
> **A first version of this experiment was wrong, and the correction changed the conclusion.**
> Initially the CRT arm's loss was essentially flat (1.657 → 1.653) while the learned arm
> trained normally, which would have read as "the CRT value code fails". The real cause was
> initialisation scale: the CRT code is a sparse binary vector (5 ones in 83 dims), so through
> a default-initialised `Linear` its output sits far below `nn.Embedding`'s and was drowned out.
> Both arms now end in the same `LayerNorm`, after which the CRT arm trains normally and the
> two arms tie. The original numbers are not reported as a finding, because they measured an
> initialisation artifact rather than the representation.

## Reproduce

```bash
python proofs/analytic_proof.py           # exact-arithmetic proof, < 2s
python proofs/make_plots.py               # clock-structure + shift-scatter figures
python -m unittest proofs.test_analytic_proof

pip install -r ../requirements-torch.txt
python model/train.py --arm crt --task add --profile fast      # smoke test, seconds
python model/train.py --arm crt --task add --profile default   # real numbers, ~3 min, needs a GPU-capable machine to be fast (falls back to CPU otherwise)
```

## Honest limitations

- The exact-arithmetic claim holds only within `[0, 1,062,347)` — outside that range the
  construction wraps modulo N (demonstrated deliberately, not hidden: see the wraparound
  check in the evidence table).
- **The trained transformer does not demonstrate exact arithmetic in embedding space.** That
  is a much stronger claim than anything tested here, and would require circuit-level
  mechanistic verification (Nanda et al. 2023, Zhong et al. 2023) — explicitly out of scope.
- **Without random-offset training, neither structured arm achieved any OOD generalization**
  (0.0% at length 5), while the no-positional-signal NoPE baseline reached 35.7%. With offsets
  both structured arms generalize, and the *learned* code beats the *fixed* CRT code by a wide
  margin (93.8% vs 34.2%). The honest reading: the training mechanism, not the algebraic
  structure, is what unlocks length generalization here — and the CRT positional code never
  convincingly beats doing nothing.
- **Generalization still collapses by length 7-8 even with offsets.** This is a much smaller
  model and budget than McLeish et al.'s, and their looped/recurrent architecture is not
  reimplemented here.
- **Run-to-run variance is large and the runs are not bit-reproducible on GPU** (±19% std on
  one headline cell; identical commands differing by 6 points). All headline numbers are
  3-seed means; single runs from this codebase should not be quoted as precise.
- **The value-embedding experiment fails almost completely for both arms** (≤1% exact match).
  Exact arithmetic structure in an embedding does not transfer to a trained model's behavior
  for free.
- Abacus-lite is a lightweight reimplementation of the core mechanism only, not tuned to
  reproduce McLeish et al.'s reported numbers, and not their full architecture.
- Model size (~2.7M params) and training budget (3,000 steps) were kept deliberately small so
  every run finishes in minutes.

## What this is not

See [ARCHITECTURE.md](../ARCHITECTURE.md#what-this-is-not).
