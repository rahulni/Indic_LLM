<h1 align="center">Loss Functions and Output Heads</h1>

<p align="center">
  <em>Making the four lines between a model's output and its scalar loss<br>
  correct, observable, and hard to get silently wrong.</em>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/rahulni/Indic_LLM/blob/main/9_loss_func_op_head/loss_and_heads.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="https://nbviewer.org/github/rahulni/Indic_LLM/blob/main/9_loss_func_op_head/loss_and_heads.ipynb"><img alt="Render in nbviewer" src="https://img.shields.io/badge/render-nbviewer-f37726?logo=jupyter&logoColor=white"></a>
  <a href="loss_and_heads.ipynb"><img alt="View on GitHub" src="https://img.shields.io/badge/view-on%20GitHub-181717?logo=github"></a>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.11.0-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Device" src="https://img.shields.io/badge/run-2000%20steps%20on%20cuda-4c8bf5">
</p>

---

## Open it

**[▶ Run in Colab](https://colab.research.google.com/github/rahulni/Indic_LLM/blob/main/9_loss_func_op_head/loss_and_heads.ipynb)** · **[Render in nbviewer](https://nbviewer.org/github/rahulni/Indic_LLM/blob/main/9_loss_func_op_head/loss_and_heads.ipynb)** · **[Read it here on GitHub](loss_and_heads.ipynb)**

The notebook is committed **with all outputs and both figures embedded**, so every table
and chart below is visible without executing anything. It installs its own dependencies
and downloads its own data, so the Colab link runs unedited on a free T4.

---

## The four lines

```python
hidden = model(tokens)
logits = output_head(hidden)
loss   = cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                       tokens[:, 1:].reshape(-1))
```

Every serious bug that lives here shares one property: **it does not raise an exception**.
A target shift in the wrong direction produces a beautiful loss curve and a worthless
model. So each section prints evidence, and most of them assert on it.

### At a glance

| | |
|---|---|
| 🔍 **Shift verified in strings** | 255/255 positions, rendered as bytes so BPE fragments stay legible |
| 🎯 **Untrained perplexity** | 51,578 against a vocabulary of 50,257 — asserted, not eyeballed |
| 🧮 **Tied vs untied head** | differs by exactly `V × D` = 12,865,792 parameters |
| 💾 **Chunked cross-entropy** | 3,100 MB → 401 MB, **7.72× less**, gradients bit-for-bit equal |
| 🧠 **Second head (t+2)** | 4.7829 vs 5.5608 — the gap opens early and never closes |
| ⚠️ **Wrong shift** | trains to **0.0019** against a correct **3.5256**, and looks better doing it |

---

## Part 1 — the seven checks

<sub>Run: **full**, 2000 steps, `cuda`, torch `2.11.0+cu128`.</sub>

| # | measurement | value |
|---|---|---|
| **1** | flattened loss matrix | `(1020, 50257)` vs `(1020,)` |
| **2** | shift positions verified | 255/255 |
| **3** | contributing tokens, pad counted -> masked | 1,020 -> 629 tokens; loss 10.6149 -> 10.8571 |
| **4** | loss, boundary kept -> masked | 4.0239 -> 4.0081 (boundary positions average 8.0422, 2.01x the rest) |
| **5** | init loss / perplexity | 10.8508 / 51,578  (`ln(50257)` = 10.8249, V = 50,257) |
| **6** | tied vs untied parameters | 17,656,576 vs 30,522,368  (difference 12,865,792 = V x D) |
| **7** | peak memory, ordinary -> chunked | 3,100 MB -> 401 MB attributable to the loss (**7.72x** less, 1.06x the time) |

Each number is produced by the notebook cell above it and written into a `RESULTS` dict,
which the final cell renders as this exact table — so the write-up cannot drift from the
run that produced it.

### The shift, verified in strings

```
 pos  input token           target token        
--------------------------------------------------
   0  b' gr'                b'ac'               
   1  b'ac'                 b'eless'            
   2  b'eless'              b' be'              
   3  b' be'                b' to'              
   4  b' to'                b' be'
```

<details>
<summary><b>Why bytes and not <code>decode()</code></b></summary>

<br>

Rendered with `enc.decode_single_token_bytes(i)`, **not** `enc.decode([i])`. GPT-2 uses
byte-level BPE, so many token ids are fragments of a UTF-8 sequence and decode to the
replacement character. On an exercise whose whole point is *print the strings*, having
them render as boxes would defeat it. `repr()` also keeps leading spaces visible instead
of losing them to column alignment.

</details>

<details>
<summary><b>Why an untrained model must sit near the vocabulary size</b></summary>

<br>

An untrained model spreads probability evenly over the vocabulary, so `loss = ln(V)` and
`perplexity ≈ V`. The expected value sits slightly *above* `ln(V)`: cross-entropy is
`logsumexp(z) - z_target`, and for `z ~ N(0, σ²)` the target term averages to zero while
`E[logsumexp(z)] ≈ ln(V) + σ²/2`. With std-0.02 init and `D = 256`, `σ ≈ 0.32`, predicting
`10.825 + 0.051 = 10.876`. The measured 10.8508 lands just under, which is Jensen's
inequality.

The notebook asserts on this. A fresh model that starts anywhere else is leaking the
answer, and there is no point training it.

</details>

<details>
<summary><b>Chunked cross-entropy, and the trap inside it</b></summary>

<br>

Chunking only the *forward* pass saves nothing: every chunk's logits stay alive in the
autograd graph until `backward()` runs. The chunked version calls `backward()` inside the
loop so each chunk is freed as it goes, then pushes the accumulated gradient into the
trunk in one step.

```python
h_det = hidden.detach().requires_grad_(True)   # cut the graph
for sl in chunks:
    loss_c = F.cross_entropy(head(h_det[sl]), y[sl], reduction="sum")
    (loss_c / n_valid).backward()              # this chunk's logits freed here
hidden.backward(h_det.grad)                    # reattach, one shot
```

The notebook asserts the chunked loss **and both gradient sets** match the naive path to
`1e-4`. A memory win that changes the answer is not a win.

Two measurement details. Memory is the delta over what was already resident, because
`max_memory_allocated()` is process-wide and would otherwise be diluted by the model
already in VRAM. And the batch is sized so the ordinary path fits: at `B = 8` on an 8 GB
card it spills into shared system memory over PCIe, and the resulting "chunking is 20×
faster" measures Windows' memory manager rather than cross-entropy.

</details>

---

## Part 2 — a second output head

One trunk, two heads. Head 1 predicts `t+1`; head 2 reads the *same* hidden state and
predicts `t+2`.

| # | measurement | value |
|---|---|---|
| **8** | Part 2 - two heads (val) | head1 4.7829, head2 5.5608, sum 10.3437  (gap 0.7779) |

<p align="center">
  <img src="assets/two_heads.png" alt="Two output heads: t+1 and t+2 from the same hidden state" width="720">
</p>

**What happens over training.** Head 2's loss sits above head 1's, the gap opens within the
first few hundred steps, and then **it holds** rather than closing. Both curves descend
together, so head 2 is learning perfectly well — it is solving a strictly harder problem.

The reason is information, not capacity. Predicting `t+2` means marginalising over the
token at `t+1` you have not seen. For a stationary source,

$$H(X_{t+2} \mid X_{1..t}) \;\ge\; H(X_{t+2} \mid X_{1..t+1}) \;=\; H(X_{t+1} \mid X_{1..t})$$

by conditioning-reduces-entropy and then stationarity. The ordering is a property of the
data, not of the model, so no amount of extra capacity in head 2 can close it.

> **One confound worth removing.** With weight tying, head 1 *is* the embedding matrix, so
> it collects gradient from the input path as well as the output path, while head 2 does
> not. That makes the measured gap partly an artifact of the asymmetry. The control with
> both heads independent moves the gap from **0.7779** to **0.8179** — it survives,
> and slightly widens, so the ordering is not a tying artifact.

Training feeds one token at a time either way — the extra head changes the loss, never the
input. The payoff is at inference: if head 2 is often right you can accept two tokens per
forward pass, checking the speculation against the next step and discarding it on a miss.
That is multi-token prediction, and it is what makes speculative decoding work.

---

## Going further

Three of the checks above are necessary but not sufficient. These pin down where.

| # | measurement | value |
|---|---|---|
| **3b** | pad counted vs masked, after training | reported 2.5855 vs 5.2508; on real tokens 5.1296 vs 5.0973; predicts pad 0.0% vs 0.0% |
| **4b** | doc B loss, cross-document attention on -> off | 4.1568 -> 4.1413 (+0.0155 from attention alone) |
| **4c** | same, with a random-token neighbour | 5.5784 -> 4.1413 (+1.4371) |
| **8b** | gap, tied head 1 vs both untied | 0.7779 vs 0.8179 - the ordering is not an artifact of weight tying |

<details open>
<summary><b>Masking the boundary does not isolate the documents</b></summary>

<br>

Masking fixes the **loss**. It does nothing about **attention**: every token in document B
can still read all of document A. Real packing needs a block-diagonal attention mask too.

```
packed, causal mask only (B can read A)        4.1568
packed, block-diagonal mask (B cannot)         4.1413
as its own sequence, positions from 0          4.0549
-----------------------------------------------------
cost of letting B attend across the join      +0.0155
```

Between two Shakespeare excerpts the cost is **+0.0155** — real but small, because
document A is more of the same corpus and supplies harmless stylistic context. Replace it
with uniform random token ids and the cost jumps to **+1.4371**. The mechanism is real;
its magnitude depends entirely on how unrelated the neighbour is, which is the argument for
masking by default rather than measuring case by case.

**The check that the mask works:** under the block-diagonal mask document B scores
**4.1413 in both experiments**, identical to four decimals, because it genuinely cannot see
what precedes it. Document A is asserted bit-identical under both masks.

The third row is lower than either packed row for a separate reason: scored as its own
sequence, document B occupies positions 0–126, where the model is better calibrated, rather
than 128–254. That is the other half of real packing — production setups reset position ids
per document as well as masking attention. This comparison holds positions fixed and changes
only attention, so the +0.0155 measures one thing.

</details>

<details>
<summary><b>Padding, demonstrated rather than asserted</b></summary>

<br>

The mechanical check is that the contributing-token count changes. But the *reason* it
matters only appears under training, so two identical models were trained, one counting pad
and one masking it.

The pad-counting model **reports 2.5855** against the other's 5.2508 — less than
half — and is the worse model on the only objective that matters: **5.1296** against
**5.0973** on real tokens. A number you would be pleased to see in a training log,
from the worse of the two models.

Both predict pad at ~0% of real-token positions at this scale. The degeneration shows up in
the loss long before it shows up in the argmax, which is precisely why the loss is not
trustworthy on its own.

</details>

---

## Appendix — the beautiful wrong loss curve

> A demonstration rather than a measurement. The warning is that *a target shift in the
> incorrect direction can produce a beautiful loss curve*, and a warning like that is worth
> answering with evidence rather than agreement.

Three models. Same data, same seed, same initialisation. Only the target slice differs.

<p align="center">
  <img src="assets/wrong_shift.png" alt="The same model trained under three target shifts" width="720">
</p>

| variant | slicing | asks the model for | final loss |
|---|---|---|---|
| **correct** | `logits[:, :-1]` vs `tokens[:, 1:]` | the next token | **3.5256** |
| no shift | `logits` vs `tokens` | the token it is looking at | 0.0019 |
| reversed | `logits[:, 1:]` vs `tokens[:, :-1]` | the token it just saw | 0.0656 |

The two runs reaching a *lower* loss are the two that are wrong. **No shift** asks the model
to emit the token it already holds — the residual stream carries the embedding straight to
the head, so it is the identity function with extra steps. **Reversed** asks for the token
one step back, which a single attention head can do.

Neither raises an exception. Neither is visible on a loss curve. The only artefact that
catches it in one glance:

```
no shift (t)   final loss 0.0019
  input             target            
  ----------------------------------
  b' gr'            b' gr'               <-- identical
  b'ac'             b'ac'                <-- identical
  b'eless'          b'eless'             <-- identical
  b' be'            b' be'               <-- identical
  b' to'            b' to'               <-- identical
  b' be'            b' be'               <-- identical
```

---

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook loss_and_heads.ipynb
```

`QUICK_RUN = True` (the default) runs 300 steps — enough to read the notebook in a couple of
minutes. Set it to `False` for the 2000-step numbers above.

> A quick run writes its figures and `results.json` to `assets/quick/`, so opening the
> notebook to look around cannot overwrite the committed full-run artifacts in `assets/`.

<details>
<summary><b>What is in here, and two caveats about the model</b></summary>

<br>

```
loss_and_heads.ipynb       the whole thing, top to bottom, outputs included
nb_source.py              the same code as a runnable script; the notebook is built from it
tools/build_notebook.py   nb_source.py         ->  .ipynb
tools/build_readme.py     assets/results.json  ->  this file
assets/                   results.json and the two figures
```

Every section has the same four beats: **why this check exists** and the failure it catches,
the code, the printed evidence, then **what you should be seeing** so a reader can tell a
pass from a fail.

Section 11 rebuilds the block with the components most current models use — **RMSNorm,
pre-norm, SwiGLU** — and puts it through the same gate, showing that swapping block
internals changes nothing the loss harness cares about. It comes last on purpose: you do not
debug a loss harness and a new architecture at the same time.

Two things to keep in mind when reading the curves. With `n_embd = 256` against
`V = 50257`, only ~4.7M of this model's ~17.7M parameters are actual transformer — the rest
is two large lookup tables. That is deliberate, since it is what makes the tied/untied and
chunked-loss measurements worth taking, but it is not a well-proportioned model. And
TinyShakespeare is ~338k tokens against 17.7M parameters, so validation loss is tracked and
plotted alongside training loss throughout.

Numbers here are from an RTX 3070 Laptop (8 GB), torch `2.11.0+cu128`.

</details>
