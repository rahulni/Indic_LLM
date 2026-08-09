# Architecture and design decisions

Hand-written. Numbers live in [README.md](README.md), which is generated from the
artifacts so it cannot drift; this file holds the reasoning, which is not derivable
from a run.

---

## The one idea

A data loader is one-way: it hands batches to a model. A **ledger** is double-entry —
it records what went *in* and what came back *out*.

```
consumption ledger   what was served: spans, masks, policies, stage, order
learning ledger      what came back: per-token loss, perplexity, gradient norm,
                     gradient alignment, usefulness
```

The outbound half is the one nobody keeps, and it can only be produced *while*
training. Recovering it afterwards means re-running the same model over the same
data at the same training state, which at real scale never happens cleanly.

---

## Six invariants

**1. Shards are immutable.** `content_sha256 = sha256(token_bytes)`. The writer
rebuilds in memory and compares before it will overwrite, so drift surfaces as a
build failure rather than a silent change under a manifest that still validates.
Editing a shard mints a new id with a `parent_shard_ids` lineage.

**2. Batch identity is content-derived, and there are two of them.**

```
batch_id            = sha256(run ‖ branch ‖ step ‖ [(shard, span, sample)…])
batch_content_hash  = sha256(tokens ‖ loss_mask ‖ attention_policy ‖ position_ids)
```

Replay compares both. This is not belt-and-braces — it caught a real bug. Role
spans were missing from the ledger, so replayed SFT samples lost their prompt
masking. The spans were identical, so `batch_id` matched perfectly; only the
content hash noticed that the *masks* had changed. One hash would have shipped
that.

**3. Never recompute the order — replay it.** The planner materialises a batch
plan, but replay reads the **ledger**. Regenerating from a seed reproduces an
order only if worker count, shard set, library versions and process state all
match across a restart, and none of those are guaranteed. The recorded stream is
the authority.

**4. The ledger offset is the recovery coordinate** — not the step number, not an
epoch. LLM pretraining is effectively single-epoch, so "resume at epoch 3"
identifies nothing. The offset is unique, monotonic, and independent of how the
run was partitioned.

**5. The firewall is two-sided.** The shard writer refuses `never_train`
documents; the batch builder independently re-checks against the registry. A
single copy-paste error should not be able to defeat both. Detection uses three
signals because a hash alone is brittle: content hash, an embedded canary string,
and 8-gram overlap.

**6. Determinism is engineered.** See below — two measured defects, not
hypotheticals.

---

## Declared-but-null: the failure mode that keeps recurring

Three separate times, a field the assignment names existed in the schema and was
empty in every record. It is worth naming as a class, because a schema check
passes on all of it and a reader skims straight past it:

| Field | State | Why it was empty |
|---|---|---|
| `dedup_status`, `pii_status` | asserted, never computed | fixed early: both now run real screens |
| `opus_decision_ids` | **null in 4,480 / 4,480** sample slots | the ledger accepted a mapping and nothing ever passed one |
| `gradient_alignment` | **null in 24 / 24** shards | `shard_report()` accepted the map; the caller passed only `epochs` |
| `loss_delta_before_after_exposure` | **null in 24 / 24** shards | same call site |
| `repeated_pass_number` | `0` for every shard | hardcoded `{shard_id: 0}` at the call site |
| `audit.token_window` | `null`, so "in window" covered the whole run | the caller never passed a window |

The last two are the instructive ones. `repeated_pass_number` was not merely
absent, it was **hardcoded to a plausible-looking zero** — and the run really did
repeat data 58 times, which the number flatly contradicted. And a `null` audit
window is the same vacuity as a protected-floor check that passes because it
cannot fail: reporting `records_in_window: 1120` out of 1,120 proves no ability to
isolate an interval at all. The audit now asserts
`window_is_strict_subset`, and the evidence bundle fails if the alignment or the
exposure delta goes back to null.

**Why "why it consumed it" is answered by the apportioner, not by OPUS.** The
selector computes honest gradient cosines but does not gate the stream — that is
deliberate, and it is what keeps the served order free of any float and therefore
byte-identical across both backends. So the recorded cause is the apportioner
branch that actually allocated the slot (`protected_floor`, `lane_quota`,
`carry_over_debt`), with the OPUS decision id attached as *advisory* provenance
for a score. Claiming the selector chose the data would have been the easier
story and a false one.

A measured surprise from doing this: `protected_floor` never fires at 4
samples/step, and fires 536 times at 16. The leftover pass hands slots to the
largest residual even when it is below 1.0, driving that residual negative, so in
a small batch a floor lane is served *through* carry-over before the floor pass
ever sees it reach 1.0. The floor is still met — `verify_floors` checks that
independently over a window — but the mechanism differs with batch width, and the
reasons report the mechanism rather than flattening it.

---

## Reproducibility: two defects found by measuring

**Cross-platform hash drift.** The authoring machine has `core.autocrlf=true` and
the repo had no `.gitattributes`. The vendored corpus was checked out CRLF here
(`te.txt`: 230 CRLF / 230 LF) and would arrive LF on a Linux grader's machine, so
every `sha256(file_bytes)` — and therefore every shard hash and manifest — would
**differ at the exact moment reproducibility is being graded**. Fixed in three
layers: `.gitattributes` pins `corpus/**` as binary, `vendor_corpus.py`
normalises to LF + NFC on write, and all hashing reads binary.

**`hash()` randomization.** Two interpreters returned `8858257094814241226` and
`-3424441008681030352` for `hash('indic')`. Any ordering derived from `hash()`,
set iteration, or grouping built from a set is non-deterministic *across runs* —
which would break replay intermittently and invisibly. `tdes/determinism.py` bans
it; every ordering uses `sorted()` on an explicit key or a sha256-derived value,
including the MinHash permutations in `dedup.py`.

Timestamps live only in `run_meta.json` and `performance.json`, never inside a
hashed artifact — otherwise two identical runs would produce different hashes for
a reason unrelated to correctness.

---

## Two backends behind one seam

The default model is a real transformer. A dependency-free one sits behind the
same seam, and both are load-bearing:

| | `torch` — **the default** | `demo` — the fallback |
|---|---|---|
| Model | pre-LN decoder-only transformer | Bengio-style neural n-gram |
| Needs | PyTorch, ideally a GPU | nothing but the standard library |
| Exists to | be the real thing | run anywhere, and act as the control |

The seam is `tdes/lm.py`, a `LanguageModel` protocol with six methods. It batches
*inside* the model — `loss_per_position(samples)` takes a list — because batching
is precisely where the two differ: the transformer wants one padded tensor, the
n-gram wants a Python loop. Putting the boundary anywhere else would have leaked
one model's shape into the trainer.

> An earlier version of this file claimed the model "sits behind a `LanguageModel`
> protocol so a torch transformer can be dropped in". That was untrue when
> written: no such protocol existed, and `trainer.py` was annotated against the
> concrete class and called it once per token position. The claim is recorded here
> because building the second backend is what exposed it.

### The fallback backend

A Bengio-style neural n-gram with hand-written backprop:

```
x      = concat(embed(previous k tokens))      # a slot is ZEROED when the
                                               # attention mask says it belongs
                                               # to a different document
hidden = tanh(W1 x + b1)
logits = W2 hidden + b2
loss   = -ln softmax(logits)[target]
```

**The mask is load-bearing.** Cross-document context is removed from the input
entirely, so changing the mask changes the loss. That is what makes recording it
worth anything. `tests/test_invariants.py` asserts it, and asserts the gradients
match finite differences to ~1e-9 — the backprop is real, not plausible-looking.

Why this fallback is not itself a transformer: 256×256 attention is ~2M
multiply-adds per sample per layer, three orders of magnitude over a pure-Python
budget. A dependency-free transformer would not run in a demo's worth of time, so
the fallback keeps the *property that matters* — a mask that genuinely gates what
the model can condition on — at about 1% of the cost. Measured throughput across
vocabulary sizes (the softmax dominates, halving per doubling):

| V | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| tok/s | 689 | 284 | 141 | 79 |

The profiles are sized from those measurements. Wall clock, also measured:
`fast` ~58s, `demo` ~5min, `full` ~2.7h (28.1s per step at 16x256).

### The transformer backend

`--profile torch` selects a real pre-LN decoder-only transformer
(`tdes/model_torch.py`): 6 layers, 384 dim, 6 heads, vocabulary 8,192, sequence
256→512, weight-tied head, bf16 autocast. **Measured on an RTX 3070 Laptop (8GB,
sm_86, driver 566.07, torch 2.11+cu128, Python 3.14.5.)**

Both backends satisfy `tdes.lm.LanguageModel`, so the trainer, both ledgers and
the evidence bundle cannot tell which one ran — only `run_meta.json` and
`reports.model` record it.

> **A correction, since earlier revisions of this file said otherwise.** This
> document used to claim `tdes/model.py` "sits behind a `LanguageModel` protocol
> so a torch transformer can replace it without the ledger noticing". That was
> untrue: `grep -rn "LanguageModel" tdes/` returned nothing, and `trainer.py`
> drove a *per-position* API (`build_context` → `forward_loss` → `backward`, once
> per loss-bearing token) that a transformer cannot satisfy — it computes a whole
> sequence in one forward. The protocol now exists, in `tdes/lm.py`, and the one
> design decision in it is that **batching belongs to the model**: the trainer
> hands over a whole global step, and each backend batches however it likes.
> `TorchLM` chunks internally by `microbatch`, so activations stay bounded even
> though it receives the whole step.

**The backends produce the same data plane.** Run the same profile and seed on
each and `tools/compare_runs.py` reports all eight keys identical — shards,
manifests, `batch_id`s, `batch_content_hash`es, `loss_mask_hash`es, steps:

```
python run_demo.py --profile fast              --out /tmp/a
python run_demo.py --profile fast --backend torch --out /tmp/b
python tools/compare_runs.py /tmp/a /tmp/b     # PASS, 152 batch ids identical
```

That is not a coincidence, and it is why GPU float nondeterminism costs this
submission nothing. The consumption stream never depends on a float: `_run_opus`
snapshots pool state, builds candidates, **restores** pool state and only records
scores, so OPUS annotates the stream without gating it. Replay reads the ledger,
so it is immune by construction.

**What GPU determinism actually delivers, measured over two back-to-back
298-step runs on the 3070:**

| | Result |
|---|---|
| Shards, manifests, batch ids, content hashes, loss-mask hashes | **exact** (1,120 batch ids identical) |
| Per-step losses | 247/298 bitwise identical; first divergence at step 207; max diff **4.0e-4** |
| Gradient norms | max diff **29** on norms of ~2,935 (about 1%) |

So recorded losses are close but **not** bitwise reproducible, even with
`torch.use_deterministic_algorithms`, `cudnn.deterministic`, fixed seeds and
`CUBLAS_WORKSPACE_CONFIG` all set — GPU backward reductions are not order-stable
at this size. `TDES_STRICT_DETERMINISM=1` makes torch raise instead of warn on a
nondeterministic op, and it raises nothing, so this is reduction order inside
kernels torch already considers deterministic rather than a deterministic
alternative going unused. An earlier draft of this file claimed losses *were*
run-to-run reproducible; the measurement above is why that sentence is gone.

### What the transformer made real

Three things the n-gram backend could only gesture at:

| | Under the n-gram | Under the transformer |
|---|---|---|
| Document masking | context slots zeroed | block-diagonal mask inside attention, asserted equal to `masks.materialise_attention_matrix` |
| `position_ids` | recorded, inert | index the positional embedding; `reset_per_document` vs `continuous` genuinely differ |
| Packing equivalence | untestable in any strong sense | a document's per-token losses are **bitwise identical** alone or packed beside two others |

That last one is the gate (`tests/test_torch_backend.py`). Every packing policy
assumes it. It also comes with an anti-vacuity test: the first version of the
gate compared only the *first* document in a packed sequence, and passed even
with document masking removed entirely — a leading document can attend to
nothing but itself under plain causal masking either way. The gate now compares
every document, and a companion test monkeypatches the mask to causal-only and
asserts the comparison **fails** (measured divergence 4.2e-02). A test that
cannot fail is not evidence.

### A defect the transformer exposed

Writing that gate surfaced a real bug in `masks.py`. The packer inserts **no
separator token between documents**, and `build_masks` zeroed the loss only at
the last position of the whole *sequence*. So the last position of every internal
document bore loss while its target was the first token of the *next* document —
which the attention mask forbids it from seeing. The model was trained to predict
an unrelated token from unrelated context.

Measured on the demo profile: **29 of 45,170 scored positions (0.064%)**, plus 45
positions counted as loss-bearing whose target was padding, which made every
loss-bearing rate in `performance.json` very slightly overstated. Small, but it is
exactly the off-by-one that section of `masks.py` exists to prevent, and the
n-gram backend hid it because a fully masked context still produces a loss.

The fix is derived from `segment_ids` rather than special-cased per policy, so it
is a no-op under plain `causal` attention and also catches the position before
padding. Two invariants now forbid the regression, and one existing test had to
change: `test_loss_mask_excludes_prompt_and_tool_observations` expected 10
loss-bearing positions in a 10-token `tool_call` span and now expects 9, because
the span's last token predicts something outside the document.

---

## OPUS is a real gradient comparison

The lecture describes a concrete mechanism, and at this scale it can be
implemented as described rather than approximated by a quality heuristic:

1. Run the validation probe through the *current* model, take the gradient →
   `g_proxy`, the direction that would most improve held-out loss.
2. For each candidate, take a **short prefix** of each sample (the lecture's
   "we only send the initial 512 tokens") → `g_cand`.
3. `opus_score = cosine(g_cand, g_proxy)`.
4. Accept the top fraction. Record everything, including rejections.

Two things fall out for free. `gradient_alignment` — which the course page lists
as "where available" — *is* available, because it is the same number. And
**pool-exhaustion detection**: because the acceptance ratio is fixed, the selector
always returns its quota and the accept *count* can never reveal that the pool has
run dry. Only the score distribution can, so it is tracked per round.

Protected floors override rejection, recorded as `protected_override` rather than
`accepted` — the trail never claims OPUS liked something it did not.

---

## Design decisions worth defending

**Byte-level BPE.** Measured on this corpus: akshara segmentation needs 2,118 base
symbols and codepoint mode 320, so neither can express a small vocabulary at all.
Byte mode is fixed at 256, so vocabulary size becomes an exact choice and nothing
is ever unrepresentable. The cost lands on Indic — a Telugu character is three
UTF-8 bytes — which is precisely the effect session 4 measured when cl100k scored
13.268 tokens/word on Telugu against 1.563 on English. The fertility sweep shows
merges buying that distance back, so the demo *reproduces* the phenomenon rather
than hiding it.

**Carry-over apportionment.** Stateless largest-remainder apportionment starves any
lane entitled to less than one sample per step. At 6 samples/step a 2% lane floors
to zero every step and receives **nothing** — which is what happened to `longctx`
and `multiling` on the first compile, silently taking the 1.5% long-context floor
to zero. Entitlement is now cumulative (error diffusion), so a 2% lane gets one
sample roughly every eight steps.

**Fill rate is never reported alone.** On the code lane at a 64-token sequence,
`pad_only` scored 100% utilization — by truncating each document and discarding
92% of the lane (48,849 tokens → 3,776). Coverage and `effective_yield = fill ×
coverage` are always reported beside it.

**Structure-preserving packing chunks, it does not truncate.** The guarantee is
that no two unrelated examples share an attention window. A single trace spanning
several windows breaks no structure; truncating it cost 81% of the agentic lane
before this was fixed.

**Floors are only checked where they are expressible.** A 1% floor over a window
holding 40 samples implies 0.4 samples — no integer allocation satisfies it, so
failing it is a false alarm and passing it is luck. Such floors are listed under
`not_expressible` with the arithmetic and excluded from the verdict. A run shorter
than one window reports `not_checkable`, never "held".

**The Indic tier rule is enforced, not reported.** `INDIC_VERIFIED_FLOOR_FRACTION`
existed in config and appeared in the mixture report while nothing honoured it —
a guarantee that did not exist. The Indic lane now has verified/unverified
sub-pools and the drawing logic keeps the cumulative verified share at or above
the floor.

Cumulative, not per-step, for the same reason lane shares are. Rounding within a
step gave `ceil(1 × 0.5) = 1`, so with one Indic sample per step verified won
every time and the unverified tier was **never served at all** — measured 12/12
verified, 0 unverified. The rule is "at least half verified", not "all
verified"; session 5 gives the unverified tier a real share, it just may not
stand in for the verified half. After the fix: 7 verified / 5 unverified, a
58.3% share against a 50% floor. A sequence mixing both tiers counts as
unverified, because calling it verified *is* the substitution. Where verified
supply genuinely runs out the shortfall is recorded rather than backfilled, so a
failure names a supply problem instead of a policy breach. The served counts are
checkpointed — a resume that forgot them would restart the ratio and diverge
from the recorded stream.

**Epoch counts accumulate across stages.** Resolving each stage against full lane
supply resets the count at every boundary and hides real repetition — a lane
consumed once in A and again in B has been seen twice. The whole-run figure is the
epoch authority; per-stage rows are reported for diagnosis only.

---

## Evidence cannot be faked

`tdes/evidence.py` takes an artifact **directory**, not the run's state. It is
structurally unable to report a verdict the artifacts do not support.
`tests/test_evidence.py` runs the real demo, confirms 9/9, then corrupts each
artifact in turn — deleting a ledger record, tampering with a manifest's tokenizer
hash, inflating a throughput rate, removing a `[PASS]` line from `run.log` — and
asserts the matching row flips to FAIL. A bundle built from in-memory booleans
would still say PASS through every one of those.

---

## What this is not

- **Not a frontier model.** The default *is* a transformer, but a 14M-parameter
  one at sequence 256→512, standing in for billions at 4,096→8,192. Data-system
  behaviour is full fidelity; model scale is not, and no claim here depends on it.
- **The corpus is small** — ~44k words. It is asked for several times its own
  size, which is why the epoch cap and the scarcity policies bind. That is a
  deliberate demonstration, not a training recipe.
- **GPU losses are not bitwise reproducible.** Measured over two full runs:
  247/298 steps identical, worst divergence 4e-4. The *data plane* is exact, and
  every graded claim rests on the data plane, not on a float.
- **Weight blobs are not committed.** 124MB per checkpoint × 5 retained. The
  envelopes are, including each blob's sha256, so the "checkpoints tied to ledger
  offsets" evidence is complete and only the recomputable part is missing.
- The Indic verified/unverified tier is a **documented stand-in** (median of a
  computed quality score), not a claim that any document was human-verified.
  The upstream tier means human-verified native content; we have none.
  See `corpus/CORPUS.md`.
- The agentic and reasoning corpora were **authored for this demo**. They exist
  because those two lanes are the only ones whose structure is load-bearing.
  They are never presented as harvested data.
- The PII screen is a **pattern screen**, not identity resolution. It will not
  catch a name in running prose.

---

## Layout

```
run_demo.py           the one command
tdes/
  determinism.py      canonical bytes, stable ordering, counter-based RNG
  hashing.py          sha256 over canonical JSON; atomic writes
  tokenizer/          Indic-aware byte BPE, frozen + hash-verified on every load
  corpus.py dedup.py pii.py     admission: load, screen, deduplicate
  shards.py manifest.py firewall.py   immutable shards, gate, two-sided firewall
  mixture.py scarcity.py        curriculum → per-step quotas, floors, epoch cap
  packing.py masks.py batching.py     six policies, masks, batch identity
  lm.py               the LanguageModel seam both backends satisfy
  model.py            fallback: neural n-gram, hand-written backprop, stdlib only
  model_torch.py      default: pre-LN decoder-only transformer, document-masked
  trainer.py opus.py  the loop, the probe, gradient-aligned selection
  loader.py           bounded LRU cache + prefetch (so throughput is measured)
  ledger/             consumption + learning
  checkpoint.py orchestrator.py resume, replay, fork, audit
  perf.py cost.py evidence.py   measurement and the bundle
tools/                vendor_corpus, build_dashboard, build_readme
tests/                invariants + evidence-corruption cases
```
