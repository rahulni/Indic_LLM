# Track B — Holographic/Fourier Binding

[← top-level README](../README.md) · [Architecture](../ARCHITECTURE.md) · [Citations](../CITATIONS.md) ·
[**Live dashboard**](https://rahulni.github.io/Indic_LLM/7_embed_research/track_b_holographic_binding/submission_artifacts/dashboard.html) ·
[local copy](submission_artifacts/dashboard.html)

## The problem

From the instructor's assignment: *"What is a REAL Fourier alternative of Kronecker? Why
can't I represent each character like a Fourier wave, and just add them to make a word!!"* —
and separately, the complaint that V1's Kronecker/tensor-product scheme forces a fixed
32-character-slot cap on every word, wasting space on short words and truncating long ones.

## The claim, stated precisely

Binding two symbols via **circular convolution** is, by the Convolution Theorem, exactly
elementwise multiplication of their DFT spectra:

> `bind(role, filler) = ifft(fft(role) * fft(filler)).real`

Building each position's **role** vector so its DFT has unit magnitude at every frequency
(random phase — literally a point on a circle per frequency, i.e. a Fourier wave) makes
single-pair binding **exactly** invertible — multiplying by the conjugate spectrum divides out
a magnitude-1 factor:

> `unbind(bind(role, filler), role) == filler`, exactly, error < 1e-8 (floating point only).

A word is composed by **summing** every character's role-bound filler into one
**fixed-dimension** vector, with no length cap:

> `word_vec = sum_i bind(role(i), filler(char_i))`

The cost: unbinding position `i` from a word of length `L` now also picks up interference from
the other `L-1` superposed pairs. `proofs/capacity_proof.py` measures exactly how decode
accuracy degrades with `L` and improves with embedding dimension `D` — a real, provable
capacity/dimension tradeoff, not an unlimited-capacity claim.

## Method summary

**The proof** generates random unitary role vectors and random Gaussian filler vectors,
verifies single-pair unbind is exact, then sweeps decode accuracy across word lengths
`L ∈ {1..40}` (deliberately crossing the 32-character cap) and dimensions
`D ∈ {64,128,192,256,512,1024}` — `D=192` is included specifically because it is the trained
model's actual `d_model`. Runs in under 20 seconds, pure numpy.

**The trained-model experiment** is a word-level decoder-only language model (closed
3,000-word vocabulary, standard learned softmax output, identical across arms) whose *input*
word embedding is computed one of two ways:

- **Kronecker/tensor (arm `kronecker`)** — V1's actual mechanism: a fixed one-hot code per
  character position, capped at 32 slots, concatenated (1,088-dim raw), then a **learned**
  linear projection down to `d_model`. Words over 32 characters are truncated.
- **Holographic (arm `holographic`)** — the construction above, fixed `d_model`-dimensional
  output, **zero** learned embedding parameters, no length cap.

Trained on tiny_shakespeare (Karpathy's `char-rnn` dataset); a synthetic long-word supplement
is used only for the length-related proofs/probes, kept separate from natural-corpus numbers
(see [ARCHITECTURE.md](../ARCHITECTURE.md), point 7).

## Evidence

From the last run (`results/capacity_proof_report.json`, `results/*_default.json`):

- Corpus, measured (not remembered): **1,115,394 bytes**, 252,268 word tokens, 12,638 unique
  words, 3,000-word closed vocabulary, 34-character alphabet, **max natural word length: 15
  characters**.
- Single-pair unbind: **exact** for every dimension tested (max error < 1e-8).
- Decode accuracy at the model's actual dimension (D=192): 100% at L≤4, 90.6% at L=16, 64.2%
  at L=32, 55.5% at L=40.

### The capacity curve has a theory to match against

The headline plot overlays a **derived** capacity bound on the measurements. It is our own
write-up of the standard VSA signal-to-noise argument — spelled out step by step in
`theoretical_decode_accuracy()`, with every assumption stated (unitary roles, iid Gaussian
fillers, interference terms treated as independent Gaussians). It is deliberately **not**
presented as "Plate's formula", because we did not read a specific numbered equation out of
Plate; see [CITATIONS.md](../CITATIONS.md) for that distinction in full.

It is validated the only honest way available here — against our own measurements, by a unit
test that asserts it predicts empirical decode accuracy and is never pessimistic. At D=192 it
tracks measurement within ~2 points across the whole length sweep, and is mildly optimistic at
small D exactly as its independence approximation predicts.

### The instructor's literal phrasing, tested head to head

"Represent each character like a Fourier wave, and just add them" reads most literally as
deterministic **shift** roles — role for position *p* is the unit impulse at *p*, whose DFT is
a pure phase ramp — rather than the random-phase roles used elsewhere. Both are unitary, so
both give exact single-pair unbind. Swept against each other at D=192:

| L | random-phase roles | shift roles (literal reading) |
|---|---|---|
| 8 | 99.6% | 99.7% |
| 16 | 90.6% | 91.5% |
| 32 | 64.2% | 65.5% |
| 40 | 55.5% | 56.0% |

**The literal reading is not worse** — marginally better, in fact. Our prior expectation was
that shifted copies of similar fillers would interfere in a correlated way and lose; that did
not happen, and the measurement is reported over the prediction.

Mean ± std over 3 seeds:

| Arm | Total params | Learned embedding params | Val perplexity | Truncation-pair cosine similarity |
|---|---|---|---|---|
| Kronecker/tensor (32-slot cap) | 3,460,728 | 209,088 | **172.4 ± 3.1** | **1.0000** (indistinguishable) |
| Holographic (circular convolution) | 3,251,640 | **0** | 190.0 ± 7.0 | 0.7895 (distinguishable) |

The truncation-pair test is the sharpest result in this track: 100 pairs of synthetic strings
sharing an identical 32-character prefix and differing only after it. The Kronecker arm's
mean cosine similarity across all 100 pairs is **exactly 1.0000** — it is *structurally
incapable* of telling them apart, not just empirically bad at it. The holographic arm can
(0.7858), using strictly fewer parameters.

**Reading the perplexity number honestly:** the Kronecker arm is ahead on in-domain perplexity
(172.4 ± 3.1 vs 190.0 ± 7.0), despite more learned parameters and the information loss above.
The gap is larger than the seed spread, so it is a real effect, not noise — reported as
measured. The natural corpus's words never get long enough (max 15 chars) to exercise
holographic binding's actual structural advantage — see Honest limitations.

### Is that gap just adaptability? (ablation: no)

The comparison above is confounded: the Kronecker arm gets a learned projection (from its
1,088-dim raw slot code), the plain holographic arm gets none. So "which code carries more
information" is tangled with "which arm can adapt its code". Giving the holographic table its
own thin learned layer separates them:

| Arm | Learned embedding params | Val perplexity |
|---|---|---|
| Kronecker/tensor (32-slot cap) | 209,088 | 172.4 ± 3.1 (3 seeds) |
| Holographic, no dressing | 0 | 190.0 ± 7.0 (3 seeds) |
| Holographic + learned dressing | 37,056 | **213.1** (single seed) |

Dressing made it **worse**, not better. So the Kronecker arm's edge is *not* explained by
having a learned projection — adding one to the holographic arm hurt it at this budget. Note
the dressing is a `d_model × d_model` layer, the same *mechanism* as the Kronecker projection
but not the same parameter count: this ablation tests adaptability, not parameter parity.

## Reproduce

```bash
# the proof -- no GPU, no torch
python proofs/capacity_proof.py     # exact-unbind check + L x D sweep + shift-role sweep, < 20s
python proofs/make_plots.py         # capacity curve (with the derived bound overlaid),
                                    # interference, and the role-construction comparison
python -m unittest proofs.test_capacity_proof   # 7 tests, incl. "the bound predicts measurement"

pip install -r ../requirements-torch.txt        # everything below needs torch

python model/train.py --arm holographic --profile fast       # smoke test, seconds
python model/train.py --arm holographic --profile default    # real numbers, ~1-2 min
python model/train.py --arm kronecker   --profile default    # the 32-slot baseline

# the ablation that rules out "it's just adaptability"
python model/train.py --arm holographic --dress --profile default

# reproduce the 3-seed perplexity table
for s in 0 1 2; do python model/train.py --arm kronecker --profile default --seed $s; done
python ../tools/aggregate_seeds.py
```

The corpus downloads and caches itself on first run (`data/.cache/`, gitignored). Every run
writes JSON into `results/`, which is the only source the READMEs, `evidence.json` and the
dashboard read from.

## Honest limitations

- **The natural corpus never produces a word longer than 15 characters** (measured) — the
  32-character truncation cliff is demonstrated via the controlled synthetic probe above, not
  via natural-corpus perplexity, which cannot show it at all.
- **Word-level closed-vocabulary perplexity is not comparable** to standard char-level
  tiny_shakespeare perplexity numbers reported elsewhere — different tokenization, different
  task, different vocabulary.
- **Holographic binding's perplexity is not better** on this run — its advantages (fixed
  dimension regardless of length, zero learned embedding parameters, no truncation) are
  structural properties demonstrated directly, not a perplexity win here.
- **Decode/unbind capacity is bounded, not unlimited** — it trades off against word length and
  improves with dimension, exactly as measured in `proofs/capacity_proof.py`. This is not "a
  free replacement for Kronecker," it is a different, measured tradeoff.
- The exact closed-form interference/capacity formula is cited qualitatively (SNR ~ 1/L,
  improving with D) from a secondary source, not pulled as a specific numbered equation
  directly from Plate (1995/2003) — see [CITATIONS.md](../CITATIONS.md) for exactly what was
  verified and what wasn't.

## What this is not

See [ARCHITECTURE.md](../ARCHITECTURE.md#what-this-is-not).
