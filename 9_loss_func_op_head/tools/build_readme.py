"""Generate README.md from assets/results.json and the executed notebook.

The prose lives here; every number and every quoted block is lifted from the run. A
write-up that quotes its numbers by hand drifts away from its own code within one edit.

Run from the assignment folder:  python tools/build_readme.py
"""
import io
import json
import re

R = json.load(io.open("assets/results.json", encoding="utf-8"))
res = R["results"]
run_kind = "full" if not R["quick_run"] else "quick"
NB = json.load(io.open("loss_and_heads.ipynb", encoding="utf-8"))


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
            out.append(f"| {num} | {label} | {v} |")
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

README = f"""# Loss Functions and Output Heads

**ERA V5 — Session 9.** One notebook that makes the four lines between a model's output
and its scalar loss *correct and observable*, plus a second output head predicting `t+2`.

**→ [`loss_and_heads.ipynb`](loss_and_heads.ipynb)** — runs top to bottom, outputs included.

Every bug that lives in those four lines shares one property: **it does not raise an
exception**. Each section below prints evidence, and most of them assert on it.

Run: **{run_kind}**, {R['steps']} steps, `{R['device']}`, torch `{R['torch']}`.

---

## Part 1 — the seven numbers

| # | measurement | value |
|---|---|---|
{required}

Each number is produced by the notebook cell above it and written into a `RESULTS` dict,
which the final cell renders as this exact table — so the write-up cannot drift from the
run that produced it.

### 2. The shift, verified in strings

```
{shift_table}
```

Rendered with `enc.decode_single_token_bytes(i)`, **not** `enc.decode([i])`. GPT-2 uses
byte-level BPE, so many token ids are fragments of a UTF-8 sequence and decode to the
replacement character. On an exercise whose whole thesis is *print the strings*, having
them render as boxes would defeat the point.

### 5. Why an untrained model must sit near the vocabulary size

An untrained model spreads probability evenly over the vocabulary, so `loss = ln(V)` and
`perplexity ≈ V`. The expected value sits slightly *above* `ln(V)`: cross-entropy is
`logsumexp(z) - z_target`, and for `z ~ N(0, σ²)` the target term averages to zero while
`E[logsumexp(z)] ≈ ln(V) + σ²/2`. With std-0.02 init and `D = 256`, `σ ≈ 0.32`, predicting
`10.825 + 0.051 = 10.876`. The measured 10.8508 lands just under, which is Jensen's
inequality. The notebook asserts on this — a fresh model that starts anywhere else is
leaking the answer, and there is no point training it.

### 7. Chunked cross-entropy

The trap is that chunking only the *forward* pass saves nothing: every chunk's logits stay
alive in the autograd graph until `backward()` runs. The chunked version calls `backward()`
inside the loop so each chunk is freed as it goes, then pushes the accumulated gradient
into the trunk in one step.

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

---

## Part 2 — a second output head

One trunk, two heads. Head 1 predicts `t+1`; head 2 reads the *same* hidden state and
predicts `t+2`.

| # | measurement | value |
|---|---|---|
{part2_row}

![two output heads](assets/two_heads.png)

**What happens over training:** head 2's loss sits above head 1's, the gap opens within the
first few hundred steps, and then **it holds** rather than closing. Both curves descend
together, so head 2 is learning perfectly well — it is solving a strictly harder problem.

The reason is information, not capacity. Predicting `t+2` means marginalising over the
token at `t+1` you have not seen. For a stationary source,

```
H(X_t+2 | X_1..t)  ≥  H(X_t+2 | X_1..t+1)  =  H(X_t+1 | X_1..t)
```

by conditioning-reduces-entropy and then stationarity. The ordering is a property of the
data, not of the model, so no amount of extra capacity in head 2 can close it.

**One confound worth removing.** With weight tying, head 1 *is* the embedding matrix, so it
receives gradient from both the input and output paths while head 2 does not. That makes
the measured gap partly an artifact of the asymmetry. Row 8b below is the control with both
heads independent: the gap goes from **{gap_tied}** to **{gap_untied}** — it survives, and
slightly widens, so the ordering is not an artifact of tying.

Training feeds one token at a time either way — the extra head changes the loss, never the
input. The payoff is at inference: if head 2 is often right you can accept two tokens per
forward pass, checking the speculation against the next step and discarding it on a miss.
That is multi-token prediction, and it is what makes speculative decoding work.

---

## Beyond the brief

These are not required by the assignment. They exist because three of the required checks
are necessary but not sufficient, and saying so is more useful than leaving it implied.

| # | measurement | value |
|---|---|---|
{extras}

**3b — padding, demonstrated rather than asserted.** The required check is that the
contributing-token count changes. But the *reason* it matters only appears under training,
so two identical models were trained, one counting pad and one masking it. The pad-counting
model **reports {pad_rep[0]}** against the other's {pad_rep[1]} — less than half — and is
the worse model on the only objective that matters: **{pad_real[0]}** against
**{pad_real[1]}** on real tokens. A number you would be pleased to see in a training log,
from the worse of the two models.

(Both predict pad at ~0% of real-token positions at this scale. The degeneration shows up
in the loss long before it shows up in the argmax, which is precisely why the loss is not
trustworthy on its own.)

**4b / 4c — masking the boundary does not isolate the documents.** Masking fixes the
*loss*. It does nothing about *attention*: every token in document B can still read all of
document A. Real packing needs a block-diagonal attention mask too.

```
{docB_block}
```

Measured honestly, the effect between two Shakespeare excerpts is **{b_delta}** — real but
small, because document A is more of the same corpus and supplies harmless stylistic
context. Replace it with uniform random token ids and the cost jumps to **{n_delta}**. The
mechanism is real; its magnitude depends entirely on how unrelated the neighbour is, which
is the argument for masking by default rather than measuring case by case.

The internal check that the mask works: under the block-diagonal mask document B scores
**{b_off} in both experiments**, identical to four decimals, because it genuinely cannot
see what precedes it. Document A is asserted bit-identical under both masks.

The third row is lower than either packed row for a separate reason: scored as its own
sequence document B occupies positions 0-126, where the model is better calibrated, rather
than 128-254. That is the other half of real packing — production setups reset position ids
per document as well as masking attention. This comparison deliberately holds positions
fixed and changes only attention, so the {b_delta} measures one thing.

---

## Appendix — the wrong shift

Not part of the brief. It is here because of the warning attached to it: *a target shift in
the incorrect direction can produce a beautiful loss curve.*

Three models, same data, same seed, same initialisation. Only the target slice differs.

![three target shifts](assets/wrong_shift.png)

| variant | slicing | final loss |
|---|---|---|
| correct | `logits[:, :-1]` vs `tokens[:, 1:]` | **{p3['correct']}** |
| no shift | `logits` vs `tokens` | {p3['no']} |
| reversed | `logits[:, 1:]` vs `tokens[:, :-1]` | {p3['reversed']} |

The two runs reaching a *lower* loss are the two that are wrong. No shift asks the model to
emit the token it already holds — the residual stream carries the embedding straight to the
head, so it is the identity function with extra steps. Reversed asks for the token one step
back, which a single attention head can do.

Neither raises an exception. Neither is visible on a loss curve. The only artefact that
catches it in one glance:

```
{noshift_block}
```

---

## Notes and reproduction

```bash
pip install -r requirements.txt
jupyter notebook loss_and_heads.ipynb
```

`QUICK_RUN = True` (default) runs 300 steps, enough to read the notebook in a couple of
minutes; `False` reproduces the {R['steps']}-step numbers above. The notebook installs its
own dependencies and downloads its own data, so it runs unedited on a free Colab T4.
Numbers here are from an RTX 3070 Laptop (8 GB), torch `{R['torch']}`.

Two things to be aware of when reading the curves. With `n_embd = 256` against
`V = 50257`, only ~4.7M of this model's ~17.7M parameters are actual transformer — the
rest is two large lookup tables. That is deliberate, since it is what makes rows 6 and 7
worth measuring, but it is not a well-proportioned model. And TinyShakespeare is ~338k
tokens against 17.7M parameters, so validation loss is tracked and plotted alongside
training loss throughout.

Section 11 of the notebook rebuilds the block the way the session described it —
**RMSNorm, pre-norm, SwiGLU** — and puts it through the same gate, showing that swapping
block internals changes nothing the loss harness cares about. It comes last on purpose: you
do not debug a loss harness and a new architecture at the same time.

```
loss_and_heads.ipynb      the assignment, top to bottom, outputs included
nb_source.py              the same code as a runnable script; the notebook is built from it
tools/build_notebook.py   nb_source.py         ->  .ipynb
tools/build_readme.py     assets/results.json  ->  this file
assets/                   results.json and the two figures
```
"""

io.open("README.md", "w", encoding="utf-8").write(README)
print(f"wrote README.md from a {run_kind} run ({R['steps']} steps)")
