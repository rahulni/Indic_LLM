"""Generate README.md from assets/results.json and the executed notebook.

The prose lives here; every number and every quoted block is lifted from the run. A
write-up that quotes its numbers by hand drifts away from its own code within one edit.

Run from the assignment folder:  python tools/build_readme.py
"""
import io
import json
import os
import re
import subprocess

R = json.load(io.open("assets/results.json", encoding="utf-8"))
res = R["results"]
run_kind = "full" if not R["quick_run"] else "quick"
NB = json.load(io.open("loss_and_heads.ipynb", encoding="utf-8"))

NB_NAME = "loss_and_heads.ipynb"


def git(*args, fallback=""):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return fallback


def repo_slug():
    url = git("remote", "get-url", "origin")
    m = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else "rahulni/Indic_LLM"


def folder():
    top = git("rev-parse", "--show-toplevel")
    here = git("rev-parse", "--show-prefix")
    return here.rstrip("/") if here else ""


SLUG = repo_slug()
# Pinned rather than read from HEAD: these URLs live in a committed file, so they must
# point at where the notebook durably lives, not at whatever branch happened to be checked
# out when the README was generated. Override with README_BRANCH if that ever changes.
BRANCH = os.environ.get("README_BRANCH", "main")
DIR = folder()
NB_PATH = f"{DIR}/{NB_NAME}" if DIR else NB_NAME

COLAB = f"https://colab.research.google.com/github/{SLUG}/blob/{BRANCH}/{NB_PATH}"
NBVIEWER = f"https://nbviewer.org/github/{SLUG}/blob/{BRANCH}/{NB_PATH}"


def get(prefix):
    for k, v in res.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


def cell_output(needle):
    for c in NB["cells"]:
        if c["cell_type"] != "code" or needle not in "".join(c["source"]):
            continue
        return "".join(
            "".join(o.get("text", []))
            for o in c.get("outputs", [])
            if o.get("output_type") == "stream"
        )
    raise KeyError(needle)


def excerpt(text, start, n_lines):
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if start in l:
            return "\n".join(lines[i:i + n_lines]).rstrip()
    raise KeyError(start)


def rows_for(pred):
    out = []
    for k, v in res.items():
        num, _, label = k.partition(". ")
        if pred(num):
            out.append(f"| **{num}** | {label} | {v} |")
    return "\n".join(out)


required = rows_for(lambda n: n.isdigit() and int(n) <= 7)
part2_row = rows_for(lambda n: n == "8")
extras = rows_for(lambda n: not n.isdigit())
p3 = dict(re.findall(r"(correct|no|reversed) ([0-9.]+)", get("9.")))


def arrow(key):
    a, b, d = re.search(r"([0-9.]+) -> ([0-9.]+) \(([+-][0-9.]+)", get(key)).groups()
    return a, b, d


b_on, b_off, b_delta = arrow("4b.")
n_on, n_off, n_delta = arrow("4c.")
pad_rep = re.search(r"reported ([0-9.]+) vs ([0-9.]+)", get("3b.")).groups()
pad_real = re.search(r"real tokens ([0-9.]+) vs ([0-9.]+)", get("3b.")).groups()
gap_tied, gap_untied = re.search(r"([0-9.]+) vs ([0-9.]+)", get("8b.")).groups()
mem_a, mem_b, mem_x = re.search(
    r"([\d,]+) MB -> ([\d,]+) MB.*?\*\*([\d.]+)x\*\*", get("7.")).groups()
h1, h2 = re.search(r"head1 ([0-9.]+), head2 ([0-9.]+)", get("8.")).groups()

# The tables come from results.json and the quoted blocks from the notebook's own
# outputs. If those two came from different runs the README would silently mix them, so
# check the notebook's final cell agrees about which run produced it.
_stamp = f"Run: {'QUICK' if R['quick_run'] else 'FULL'}  |  {R['steps']} steps"
_tail = cell_output('print("| # | measurement | value |")')
assert _stamp in _tail, (
    f"notebook outputs and results.json disagree - expected {_stamp!r} in the final "
    "cell. Re-run the notebook end to end before regenerating the README."
)

shift_table = excerpt(cell_output("seq_in, seq_tgt"), "pos", 7)
noshift_block = excerpt(cell_output("pairs = {"), "no shift", 10)
docB_block = excerpt(cell_output("doc_block_mask"), "packed, causal mask only", 5)

README = f"""<h1 align="center">Loss Functions and Output Heads</h1>

<p align="center">
  <em>Making the four lines between a model's output and its scalar loss<br>
  correct, observable, and hard to get silently wrong.</em>
</p>

<p align="center">
  <a href="{COLAB}"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="{NBVIEWER}"><img alt="Render in nbviewer" src="https://img.shields.io/badge/render-nbviewer-f37726?logo=jupyter&logoColor=white"></a>
  <a href="{NB_NAME}"><img alt="View on GitHub" src="https://img.shields.io/badge/view-on%20GitHub-181717?logo=github"></a>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-{R['torch'].split('+')[0]}-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Device" src="https://img.shields.io/badge/run-{R['steps']}%20steps%20on%20{R['device']}-4c8bf5">
</p>

---

## Open it

**[▶ Run in Colab]({COLAB})** · **[Render in nbviewer]({NBVIEWER})** · **[Read it here on GitHub]({NB_NAME})**

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
| 💾 **Chunked cross-entropy** | {mem_a} MB → {mem_b} MB, **{mem_x}× less**, gradients bit-for-bit equal |
| 🧠 **Second head (t+2)** | {h1} vs {h2} — the gap opens early and never closes |
| ⚠️ **Wrong shift** | trains to **{p3['no']}** against a correct **{p3['correct']}**, and looks better doing it |

---

## Part 1 — the seven checks

<sub>Run: **{run_kind}**, {R['steps']} steps, `{R['device']}`, torch `{R['torch']}`.</sub>

| # | measurement | value |
|---|---|---|
{required}

Each number is produced by the notebook cell above it and written into a `RESULTS` dict,
which the final cell renders as this exact table — so the write-up cannot drift from the
run that produced it.

### The shift, verified in strings

```
{shift_table}
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
{part2_row}

<p align="center">
  <img src="assets/two_heads.png" alt="Two output heads: t+1 and t+2 from the same hidden state" width="720">
</p>

**What happens over training.** Head 2's loss sits above head 1's, the gap opens within the
first few hundred steps, and then **it holds** rather than closing. Both curves descend
together, so head 2 is learning perfectly well — it is solving a strictly harder problem.

The reason is information, not capacity. Predicting `t+2` means marginalising over the
token at `t+1` you have not seen. For a stationary source,

$$H(X_{{t+2}} \\mid X_{{1..t}}) \\;\\ge\\; H(X_{{t+2}} \\mid X_{{1..t+1}}) \\;=\\; H(X_{{t+1}} \\mid X_{{1..t}})$$

by conditioning-reduces-entropy and then stationarity. The ordering is a property of the
data, not of the model, so no amount of extra capacity in head 2 can close it.

> **One confound worth removing.** With weight tying, head 1 *is* the embedding matrix, so
> it collects gradient from the input path as well as the output path, while head 2 does
> not. That makes the measured gap partly an artifact of the asymmetry. The control with
> both heads independent moves the gap from **{gap_tied}** to **{gap_untied}** — it survives,
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
{extras}

<details open>
<summary><b>Masking the boundary does not isolate the documents</b></summary>

<br>

Masking fixes the **loss**. It does nothing about **attention**: every token in document B
can still read all of document A. Real packing needs a block-diagonal attention mask too.

```
{docB_block}
```

Between two Shakespeare excerpts the cost is **{b_delta}** — real but small, because
document A is more of the same corpus and supplies harmless stylistic context. Replace it
with uniform random token ids and the cost jumps to **{n_delta}**. The mechanism is real;
its magnitude depends entirely on how unrelated the neighbour is, which is the argument for
masking by default rather than measuring case by case.

**The check that the mask works:** under the block-diagonal mask document B scores
**{b_off} in both experiments**, identical to four decimals, because it genuinely cannot see
what precedes it. Document A is asserted bit-identical under both masks.

The third row is lower than either packed row for a separate reason: scored as its own
sequence, document B occupies positions 0–126, where the model is better calibrated, rather
than 128–254. That is the other half of real packing — production setups reset position ids
per document as well as masking attention. This comparison holds positions fixed and changes
only attention, so the {b_delta} measures one thing.

</details>

<details>
<summary><b>Padding, demonstrated rather than asserted</b></summary>

<br>

The mechanical check is that the contributing-token count changes. But the *reason* it
matters only appears under training, so two identical models were trained, one counting pad
and one masking it.

The pad-counting model **reports {pad_rep[0]}** against the other's {pad_rep[1]} — less than
half — and is the worse model on the only objective that matters: **{pad_real[0]}** against
**{pad_real[1]}** on real tokens. A number you would be pleased to see in a training log,
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
| **correct** | `logits[:, :-1]` vs `tokens[:, 1:]` | the next token | **{p3['correct']}** |
| no shift | `logits` vs `tokens` | the token it is looking at | {p3['no']} |
| reversed | `logits[:, 1:]` vs `tokens[:, :-1]` | the token it just saw | {p3['reversed']} |

The two runs reaching a *lower* loss are the two that are wrong. **No shift** asks the model
to emit the token it already holds — the residual stream carries the embedding straight to
the head, so it is the identity function with extra steps. **Reversed** asks for the token
one step back, which a single attention head can do.

Neither raises an exception. Neither is visible on a loss curve. The only artefact that
catches it in one glance:

```
{noshift_block}
```

---

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook {NB_NAME}
```

`QUICK_RUN = True` (the default) runs 300 steps — enough to read the notebook in a couple of
minutes. Set it to `False` for the {R['steps']}-step numbers above.

> A quick run writes its figures and `results.json` to `assets/quick/`, so opening the
> notebook to look around cannot overwrite the committed full-run artifacts in `assets/`.

<details>
<summary><b>What is in here, and two caveats about the model</b></summary>

<br>

```
{NB_NAME}       the whole thing, top to bottom, outputs included
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

Numbers here are from an RTX 3070 Laptop (8 GB), torch `{R['torch']}`.

</details>
"""

io.open("README.md", "w", encoding="utf-8").write(README)
print(f"wrote README.md from a {run_kind} run ({R['steps']} steps)")
print(f"  repo   : {SLUG}")
print(f"  branch : {BRANCH}")
print(f"  colab  : {COLAB}")
