# Architecture and design decisions

Hand-written. Numbers live in [README.md](README.md) and each track's own README, all of
which read only from `results/*.json` produced by the code in this repo — this file holds
the reasoning, which is not derivable from a run.

---

## The one idea, per track

**Track A.** Kronecker's own theorem on finite abelian groups says every such group is a
direct product of cyclic groups. For integers mod N, that decomposition is the Chinese
Remainder Theorem: pick pairwise-coprime primes `p_1...p_k` with `N = p_1 * ... * p_k`, and
`Z/NZ ≅ Z/p_1Z × ... × Z/p_kZ` is not just a group isomorphism but a **ring** isomorphism —
both `+` and `×` become independent, per-slot operations. This is the same Residue Number
System used in RSA and homomorphic-encryption hardware for exactly this reason: no carries,
fully parallel modular arithmetic. Embedding a number as the concatenation of one-hot residue
codes per prime turns "add k" into a fixed cyclic shift per slot (diagonalized by the
per-prime DFT — the literal Fourier connection), and turns "multiply by k" into the same
shift, applied in discrete-log space (valid because each `p_i` is prime, so `(Z/p_iZ)*` is
cyclic — the classical Zech-logarithm trick). Both operations are exact, closed-form, and
require no learned parameters.

**Track B.** The scheme being extended is Eq. 1 of Shravan (2026), arXiv:2605.29459:
`kappa(b) = (1/sqrt(L)) * SUM_p c_{b_p} (x) p_p` — a Kronecker product of a one-hot byte
(`d_c=256`) with a one-hot position (`d_p=32`), summed over positions, giving `D=8192`. That
is Smolensky-style tensor-product binding: because the position vectors are *orthogonal*, the
bound pairs never interfere, which is exactly why the dimension has to be `d_c x d_p` and why
tokens past 32 bytes have nowhere to go and are truncated. Track B asks what happens if you
give up that orthogonality on purpose.

Binding two symbols via circular convolution is, by the Convolution Theorem,
exactly elementwise multiplication of their DFT spectra. Building a role vector whose DFT has
unit magnitude at every frequency (random phase — literally points on a circle per frequency
component) makes single-pair binding exactly invertible: multiplying by the conjugate spectrum
divides out a magnitude-1 factor with zero error. Summing several such bound pairs into one
fixed-dimension vector — rather than concatenating them, which is what a literal Kronecker/
tensor product does, and which is why V1 needs a growing (or capped) number of slots — removes
the length cap entirely. The cost is interference: unbinding one pair now also picks up
cross-talk from every other superposed pair, with a measured, dimension-dependent capacity.

## Repository layout

```
7_embed_research/
  README.md  ARCHITECTURE.md  CITATIONS.md
  requirements.txt              numpy, matplotlib — everything the proofs need
  requirements-torch.txt        torch, opt-in, only for the trained-model runs
  .gitignore                    excludes *.pt weight blobs; submission_artifacts/ is NOT excluded

  common/
    device.py        cuda/mps/cpu autodetect, param counting
    transformer.py    shared decoder-only trunk (pre-norm, causal self-attn) — pluggable input embeddings
    seeding.py        seeds python/numpy/torch together
    evidence.py        PASS/FAIL harness, grades from on-disk artifacts only
    plotting.py         matplotlib -> PNG + base64 data URI, for inlining into self-contained HTML

  track_a_numeral_crt/
    proofs/analytic_proof.py + test_analytic_proof.py     the exact-arithmetic proof (4 ring ops)
    proofs/make_plots.py                                    clock-structure + shift-scatter figures
    data/generate_arithmetic.py                              digit tokenizer + random-offset augmentation
    model/embeddings.py                                       three positional-embedding arms
    model/train.py + evaluate.py                              trainer + length-generalization eval
    model/value_train.py                                       the instructor's literal ask: operand = 1 token
    results/  submission_artifacts/dashboard.html

  track_b_holographic_binding/
    proofs/capacity_proof.py + test_capacity_proof.py        exact unbind, derived bound, shift-role sweep
    proofs/make_plots.py                                      capacity curve, interference, role comparison
    data/corpus.py                                             tiny_shakespeare fetch + word vocab + synthetic long words
    model/embeddings.py                                        two word-embedding arms (+ optional dressing)
    model/train.py + evaluate.py                               word-level LM trainer + truncation probe
    results/  submission_artifacts/dashboard.html

  submission_artifacts/     index.html, evidence.json, seed_aggregate.json
  tools/run_all.py, aggregate_seeds.py, build_dashboard.py, build_index.py, chart_kit.py
  tests/test_common.py      trunk, evidence harness, random-offset invariants
```

`common/` holds only genuinely shared, track-agnostic code (the transformer trunk, device
detection, seeding, plotting helpers, the evidence harness). The embedding classes are
deliberately **not** shared — they are the thing each track is comparing, so each track owns
its own `model/embeddings.py`.

## Judgment calls (explicit, not silently resolved)

1. **Track A's baseline arm uses no positional embedding at all (NoPE), not a standard
   absolute positional embedding.** This was a live decision during implementation: a
   transformer with literally zero position signal sounds like a strawman, but it is in fact
   a real, citable, competitive baseline — Kazemnejad et al. (2023, NeurIPS) show NoPE
   causal transformers *outperform* APE/ALiBi/RoPE on length generalization for exactly this
   kind of reasoning task, because the causal attention mask alone carries some order
   information. Using it as the baseline makes the three-arm comparison a fair test of
   *which extra positional signal, if any, helps* — not a test against a deliberately
   crippled strawman.

2. **The Abacus arm reuses the same 12-dimensional CRT code's *width*, not its content, as a
   *learned* embedding table**, so that arms 2 and 3 differ only in fixed-vs-learned, not also
   in dimensionality. This keeps the ablation to one variable at a time.

3. **Track A's "does embedding(9)+embedding(9) become embedding(18)" claim is proven only in
   `proofs/analytic_proof.py`, never by the trained transformer.** The trained model answers a
   different, honestly-separate question: does a *fixed, algebraically structured* digit
   *positional* code change a small transformer's length-generalization behavior, compared to
   a learned positional code or none at all. Conflating these two would be the single easiest
   way to overclaim, so the README is explicit about the separation everywhere it comes up.

4. **Track A's first empirical result was a negative one, and the follow-up explains why.**
   In the first round no embedding arm generalized past the training digit-length range. The
   honest response was to report it plainly rather than inflate the model until the number
   improved. A reviewer pass then identified the actual gap: McLeish et al. pair Abacus
   embeddings with **random position-offset training**, which had not been implemented, so the
   experiment was testing the embeddings without the mechanism that makes them work. Adding it
   (`--random-offset`) produced real OOD generalization (0% → 91.5% for Abacus-lite at length
   5, 0% → 41.5% for CRT). Both rounds are reported: the negative result was not deleted once
   a better one existed, because the sequence is the finding — *the mechanism, not the
   algebraic structure, is what unlocks length generalization here*.

10. **The offset is drawn per example and applied uniformly to every digit in it.** A per-digit
    offset would destroy the alignment between a's units digit, b's units digit and the
    answer's units digit — which is the entire thing the model has to learn. Pinned down by
    `test_offset_is_uniform_within_an_example`.

11. **Track A's value-embedding experiment is the instructor's literal ask, and it fails —
    after a correction that changed what "fails" means.** `model/value_train.py` gives each
    whole operand a single token embedded with the fixed CRT *value* code — "embedding(9)
    knows it is 9". The first run showed the CRT arm's loss essentially flat (1.657 → 1.653)
    against a normally-training control, which would have supported a much stronger claim:
    that the CRT code is unusable as a value embedding. That was false. The CRT code is a
    sparse binary vector (5 ones in 83 dims), so through a default-initialised `Linear` its
    output scale sits far below `nn.Embedding`'s and the operand signal was simply drowned
    out — an initialisation artifact, not a property of the representation. Both arms now end
    in the same `LayerNorm`; the CRT arm then trains normally (1.658 → 1.049) and the two arms
    **tie exactly** at ≤1% exact match, with the CRT code using 10× fewer embedding
    parameters. The corrected result is the one reported, the artifact is documented in the
    README rather than quietly deleted, and the episode is itself the sharpest statement of
    this project's central methodological point: a proof about a representation is not a claim
    about what a model trained on that representation will do — and a negative result is only
    worth reporting once you have ruled out that you caused it.

13. **GPU runs here are not bit-reproducible, and that is stated rather than papered over.**
    Every run seeds Python, NumPy and Torch (`common/seeding.py`), but no
    `torch.use_deterministic_algorithms(True)` / cuDNN-deterministic flags are set, so
    non-deterministic CUDA kernels make repeated runs of the *same command with the same seed*
    differ. This is not hypothetical: two identical `abacus --random-offset 24 --seed 0`
    invocations produced 91.5% and 85.5% at test length 5. The response was not to hide the
    discrepancy or pick the better number, but to (a) report every headline result as
    **mean ± std over 3 seeds** rather than a single run, and (b) say so here. Forcing full
    determinism was considered and rejected for now: it would slow training materially and
    would create a false impression of precision that a 3-seed spread communicates better.

14. **The baseline was reconstructed before the source existed, then corrected against it.**
    This project was built without access to the V1 paper or code — verified, not assumed:
    "kronecker" appeared in zero files and zero commits of this repository's prior history. The
    baseline was reconstructed from the assignment prose and labelled as a reconstruction.
    When the real sources arrived, auditing against Eq. 1 showed the *structure* was right
    (one-hot ⊗ one-hot summed over positions is a coordinate permutation of block placement,
    and the following `Linear` absorbs permutations) but two things were wrong: the `1/sqrt(L)`
    normaliser was missing, and the alphabet was 34 corpus characters rather than 256 bytes.
    Both are fixed and every affected number re-measured.

    The instructive part is the **direction** of the error. The missing normaliser
    *weakened* the baseline — correcting it moved Kronecker's perplexity from 174.6 down to
    ~153, i.e. the bug had been flattering this project's own contribution. A bug that makes
    your comparison look better than it should is the kind you are least likely to go looking
    for, which is the argument for implementing a baseline from its specification rather than
    from your own reading of what it probably does.

12. **Track B's theoretical curve is our own derivation, labeled as such.** The plan called for
    overlaying "the" capacity bound from Plate. We could not verify a specific numbered
    equation from the primary text, so rather than attribute a formula we had not read, we
    derived the standard SNR argument ourselves, wrote every assumption into the docstring, and
    validated it against our own measurements with a unit test. `CITATIONS.md` states exactly
    what was and was not verified. Attributing a plausible-looking equation to Plate would have
    been the easier path and a false one.

5. **Two separate CRT modulus sets in Track A** — `[2,3,7]` (N=42, width 12) for digit
   *positions*, `[11,13,17,19,23]` (N=1,062,347, width 83) for the exhaustive *value*
   arithmetic proof. They serve different range/dimension requirements; unifying them would
   either break the "cover positions 0-31" requirement or make the exhaustive value proof too
   small to be interesting.

6. **Track B's "word-level language model with character-composed input embeddings"
   framing** was a live disambiguation of the instructor's "character-level language model"
   phrasing. Built here: a closed-vocabulary softmax **output** (standard, learned, identical
   across arms) fed by a character-composed **input** embedding (the thing under test).
   Full compositional-unbind *generation* (decoding "which word" by unbinding rather than a
   softmax) is a substantially harder generative problem and is out of scope, named
   explicitly rather than silently assumed away.

7. **Track B's synthetic long-word supplement is necessary, not decorative.** The natural
   tiny_shakespeare corpus's longest word is 15 characters (measured, not assumed) — nowhere
   near the 32-character cap the whole track is about. Without synthetic words the
   truncation-cliff claim would be untestable from the natural corpus alone; the synthetic
   probe (`evaluate.py`'s `make_truncation_pairs`) is the only place that claim is actually
   demonstrated, and it is kept clearly separate from natural-corpus perplexity numbers.

8. **D=192 was added to Track B's capacity-proof dimension sweep** specifically because it is
   the trained model's actual `d_model` — the interactive dashboard's decode-accuracy lookup
   reads real proof data at the model's real dimension, not an interpolated approximation
   from a neighboring sweep point.

9. **The interactive "try it yourself" widgets are baked-data, not live inference.** Track A's
   clock/shift widget computes CRT residues and reconstruction live in JavaScript (cheap
   modular arithmetic — safe and exact to run client-side); Track B's word-truncation widget
   shows the real 32-slot truncation live but reads its decode-accuracy estimate off the
   proof's precomputed D=192 sweep rather than reimplementing FFT-based binding in
   JavaScript. Neither widget calls back into Python or a server — both are fully
   self-contained, which is also what makes them safe to publish as standalone artifacts.

## What this is not

- Not a claim that either construction improves a production language model's downstream
  task performance — both trained-model experiments are small, fast, coursework-scale
  ablations meant to test a specific mechanism, not to compete with published benchmarks.
- Not a mechanistic-interpretability result. Nothing here inspects a trained model's internal
  circuits (attention patterns, learned Fourier features, etc.) — that is exactly the kind of
  claim Nanda et al. (2023) and Zhong et al. (2023) make about *emergent* structure, and this
  project's contrast is to hard-wire structure into the embedding instead, not to verify what
  a trained network does with it internally.
- Not a full reimplementation of McLeish et al. (2024)'s Abacus embeddings (their looped
  architecture and extreme length ranges are out of scope) or of Plate's full HRR formalism
  (no resonator-network-style iterative clean-up is implemented; decoding is plain
  nearest-neighbor cosine similarity).
- Not a claim that Track B's holographic binding is a strict improvement over Kronecker/
  tensor-product binding — it trades a hard length cap and learned-parameter cost for a
  measured capacity/dimension tradeoff and (on this run) a small perplexity regression. The
  README states both sides.
