# Citations

Every citation below was checked via live web search during this project
(not recalled from memory) before being used in a README or ARCHITECTURE.md
claim. Two corrections are recorded explicitly because they are exactly the
failure mode this file exists to catch.

## The work this project extends

- **Shravan, R. "Kronecker Embeddings: Byte-Level Structured Token
  Representations for Parameter-Efficient Language Models." arXiv:2605.29459v1,
  28 May 2026.** Code: <https://github.com/theschoolofai/kronecker-embeddings>

  This is "Kronecker Embedding V1" — the scheme the assignment asks to extend.
  Its Eq. 1 (§3.2) is:

  ```
  kappa(b) = (1/sqrt(L)) * SUM_{p=1..L}  c_{b_p} (x) p_p
  ```

  with `c_{b_p}` a one-hot over the byte value (`d_c = 256`), `p_p` a **one-hot**
  over byte position (`d_p = 32` in the production setting), `D = d_c * d_p =
  8192`, followed by a single learned `Linear(D, d_model)`. Tokens longer than
  `d_p` **bytes** are truncated (UTF-8-safe; the paper reports ≤0.18% of tokens
  affected at `d_p = 32`). The codec is deterministic and **not invertible**.
  Track B's `kronecker` arm implements exactly this, and is the baseline every
  Track B comparison is made against.

  > [!WARNING]
  > **This citation was added late, and that is itself worth recording.** The
  > whole project was built and a first version published *before* this paper
  > and repository were available to me. I verified beforehand that the string
  > "kronecker" appeared in **zero** files and **zero** commits in this
  > repository, and reconstructed the scheme from the assignment prose instead,
  > labelling it as a reconstruction. When the real sources arrived I audited
  > the implementation against Eq. 1. The reconstruction's *structure* turned
  > out to be correct — because both factors are one-hot, the sum of Kronecker
  > products is a coordinate permutation of the block layout I had built, and
  > the following `Linear` absorbs the permutation — but it had **two real
  > deviations**, both since fixed: the `1/sqrt(L)` normaliser was missing (so
  > code norm grew as `sqrt(L)`, 1.0 → 5.66 across L=1..32, instead of staying
  > at 1.0), and it used a 34-character corpus alphabet rather than the
  > 256-value byte alphabet (understating the arm's projection parameters by
  > ~7.5×). Both are documented in
  > `track_b_holographic_binding/README.md`. Earlier revisions of these docs
  > described the reconstruction as "V1's actual mechanism", which overstated
  > what I knew; that wording has been corrected.

## Track A — numeral embeddings, arithmetic, positional encoding

- McLeish, S., Kirchenbauer, J., Miller, D. Y., Singh, S., Bhatele, A.,
  Goldblum, M., Panda, A., Goldstein, T. **"Transformers Can Do Arithmetic
  with the Right Embeddings."** NeurIPS 2024. arXiv:2405.17399. (Source of
  the Abacus embedding idea that Track A's `abacus` arm lightly
  reimplements — digit-identity embedding plus a learned embedding of each
  digit's place-value index within its own operand. Track A does **not**
  reimplement their full looped/recurrent architecture or their extreme
  length ranges; this is stated explicitly in `track_a_numeral_crt/README.md`.)

- Golkar, S. et al. **"xVal: A Continuous Numerical Tokenization for
  Scientific Language Models."** arXiv:2310.02989 (2023). (Note: the paper's
  own arXiv title differs from the paraphrase "A Continuous Number Encoding
  for Large Language Models" that circulates informally — the arXiv listing
  itself uses "Tokenization," not "Encoding." Cited here as related work on
  encoding numeric structure directly into embeddings, not reimplemented.)

- Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. **"Progress
  measures for grokking via mechanistic interpretability."** ICLR 2023
  (Oral). arXiv:2301.05217. (Related work: shows a trained one-layer
  transformer *discovers* Fourier/circular structure for modular addition
  via grokking. Track A's contrast is to hard-wire that structure into the
  embedding from the start rather than rely on a network discovering it —
  Track A does not attempt circuit-level verification of what a trained
  model internally computes, which is what this paper does.)

- Zhong, Z., Liu, Z., Tegmark, M., Andreas, J. **"The Clock and the
  Pizza: Two Stories in Mechanistic Explanation of Neural Networks."**
  NeurIPS 2023. (Related work, same reasoning as above — cited, not
  reimplemented or claimed to be reproduced.)

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V.
  **"Grokking: Generalization Beyond Overfitting on Small Algorithmic
  Datasets."** arXiv:2201.02177 (2022). **This is an arXiv preprint with no
  conference venue** — do not attribute it to a conference.

- Kantamneni, S., Tegmark, M. **"Language Models Use Trigonometry to Do
  Addition."** arXiv:2502.00873 (Feb 2025).
  **Correction, kept here on purpose:** an earlier working note in this
  project cited this paper under the title "Language Models Encode Numbers
  Using a Generalized Helix" — that title does not exist. "Generalized
  helix" is a phrase from the paper's *content* (it describes numbers as
  represented on a generalized-helix manifold, manipulated via a "Clock"
  algorithm), not its title. The correct title is above. This is left in
  as a record of exactly the kind of error the project's "do not
  hallucinate" constraint is meant to catch, and how it was caught (a live
  search, not a second guess).

- Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P., Reddy, S. **"The
  Impact of Positional Encoding on Length Generalization in
  Transformers."** NeurIPS 2023. arXiv:2305.19466. (Justifies Track A's
  `baseline` arm as a real, citable choice rather than an strawman: this
  paper's central empirical finding is that transformers with **no**
  positional encoding at all (NoPE) — order information coming only from
  the causal attention mask — generalize to longer sequences on
  reasoning/arithmetic tasks *better* than APE, ALiBi, or RoPE. Track A's
  baseline arm is exactly this NoPE configuration, not an under-powered
  strawman.)

- Garner, H. **"The Residue Number System."** IRE Transactions on
  Electronic Computers, EC-8(2):140-147, 1959. (Classical source for
  Residue Number Systems / CRT-based parallel modular arithmetic — the
  mechanism Track A's `crt` arm is built on. RNS is also the basis of
  RNS-CKKS-style homomorphic encryption implementations, cited here only
  as context for why CRT decomposition is a real, load-bearing engineering
  technique and not a novel trick invented for this project.)

- Schönhage, A., Strassen, V. **"Schnelle Multiplikation großer Zahlen."**
  Computing, 7:281-292, 1971. (Classical source for Number-Theoretic-
  Transform-based large-integer multiplication — cited as the standard
  reference for the Fourier/NTT view of multiplication-as-convolution that
  motivates, but is not literally reimplemented by, Track A's discrete-log
  "multiplication reduces to addition" construction.)

## Track B — holographic binding, Vector Symbolic Architectures

- Plate, T. A. **"Holographic Reduced Representations."** IEEE
  Transactions on Neural Networks, 6(3):623-641, 1995. Also: Plate, T. A.
  **Holographic Reduced Representation: Distributed Representation for
  Cognitive Structures.** CSLI Publications, 2003. (Primary source for
  binding-via-circular-convolution and its capacity/crosstalk behavior —
  the mechanism Track B's `holographic` arm implements directly.
  **Verification status:** a live search corroborated the qualitative
  claim used in `track_b_holographic_binding/README.md` — per-component
  signal power scales as 1/D while per-component interference (crosstalk)
  power scales as m/D for m superposed bound pairs, giving a
  signal-to-noise ratio of order 1/m independent of D on its own, so
  capacity trades off length against dimension exactly as
  `proofs/capacity_proof.py` measures empirically — but this was
  confirmed via a secondary aggregator source, not a page-and-equation
  citation pulled directly from the 1995/2003 primary text.
  **Consequently, the theoretical curve plotted in this repo is our OWN
  derivation of that standard SNR argument, written out step by step in
  `theoretical_decode_accuracy()` in
  `track_b_holographic_binding/proofs/capacity_proof.py`, with every
  assumption stated in the docstring — it is deliberately NOT presented as
  "Plate's formula", because we did not read a specific numbered equation
  out of Plate. It is validated the only way that is honest here: against
  our own measurements, by a unit test
  (`test_theoretical_bound_tracks_measurement`) that asserts it predicts
  the empirical decode accuracy within 8 points and is never pessimistic.
  It tracks measurement within ~2 points at D=192 and is mildly optimistic
  at small D, exactly as its stated Gaussian-independence approximation
  predicts.**)

- Smolensky, P. **"Tensor Product Variable Binding and the Representation
  of Symbolic Structures in Connectionist Systems."** Artificial
  Intelligence, 46(1-2):159-216, 1990. (The tensor-product binding
  mechanism that Track B's `kronecker` arm is a concrete instance of —
  cited as the origin of the binding scheme circular convolution is
  contrasted against, not reimplemented beyond what V1's own 32-slot
  mechanism already is.)

- Kanerva, P. **"Hyperdimensional Computing: An Introduction to Computing
  in Distributed Representation with High-Dimensional Random Vectors."**
  Cognitive Computation, 1:139-159, 2009. (General Vector Symbolic
  Architecture / Hyperdimensional Computing background — cited as context
  for the binding/superposition framework, not a specific technique
  reimplemented here.)

- Frady, E. P., Kent, S. J., Olshausen, B. A., Sommer, F. T. **"Resonator
  Networks, 1: An Efficient Solution for Factoring High-Dimensional,
  Distributed Representations of Data Structures."** Neural Computation,
  32(12):2311-2331, 2020 (plus companion Part 2, same venue). (Cited as
  further reading on decoding/factoring superposed VSA representations —
  not reimplemented; Track B's decode step is plain nearest-neighbor
  cosine similarity, not a resonator network.)

## Corpus

- Karpathy, A. **tiny_shakespeare** (`char-rnn` dataset). Public domain
  text (Shakespeare) redistributed for language-model demos. Track B
  downloads this directly (see `data/corpus.py`) and reports its
  **measured** byte count and word statistics at run time rather than a
  remembered figure — see `track_b_holographic_binding/README.md`'s
  Evidence section for the exact numbers from the last run.
