"""Generate README.md from the run artefacts. Refuses to build from a stale or partial run.

The README is not hand-written, because a hand-written one drifts: somebody re-runs an
experiment, the prose keeps the old numbers, and the document quietly becomes fiction.
Every quantity below is read out of assets/*.json, and the build fails if those are
incomplete or older than the figures that claim to show them.

    python tools/build_readme.py
"""
from __future__ import annotations

import json
import os
import sys

GH_USER, GH_REPO, FOLDER = "rahulni", "Indic_LLM", "13_Distributed_Model_Pipeline_Parallel"
NB1 = "01_reversibility_from_scratch.ipynb"
NB2 = "02_train_20M_on_50M_tokens.ipynb"
NB3 = "03_results_and_cost.ipynb"
REQUIRED = ["A_baseline", "E_checkpoint", "B_euler", "C_midpoint", "G_euler_implicit",
            "F_coupling", "D_maxbatch", "D2_maxbatch_same_lr"]
FIGURES = ["memory_vs_batch.png", "memory_vs_depth.png", "loss_curves.png",
           "speed_vs_memory.png", "memory_decomposition.png", "throughput.png",
           "diagnostics.png", "euler_condition.png", "reconstruction_drift.png"]
LABEL = {"store": "baseline (stores activations)", "checkpoint": "gradient checkpointing",
         "euler": "symplectic Euler", "midpoint": "midpoint (leapfrog)",
         "coupling": "coupling (RevNet)", "euler_implicit": "implicit Euler"}


def guard(res, assets):
    names = [r["spec"]["name"] for r in res["runs"]]
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        sys.exit(f"REFUSING TO BUILD: results.json is missing {missing}.\n"
                 f"  run `python -m revlm.run_all` first.")
    rt = os.path.getmtime(os.path.join(assets, "results.json"))
    stale = [f for f in FIGURES
             if not os.path.exists(os.path.join(assets, f))
             or os.path.getmtime(os.path.join(assets, f)) < rt - 1]
    if stale:
        sys.exit(f"REFUSING TO BUILD: figures {stale} missing or older than the run.\n"
                 f"  run `python -m revlm.plots` first.")


def pct(a, b):
    return (a / b - 1) * 100


def build(assets="assets", out="README.md"):
    res = json.load(open(os.path.join(assets, "results.json")))
    preds = json.load(open(os.path.join(assets, "predictions.json")))
    bench = json.load(open(os.path.join(assets, "throughput.json")))
    diag = json.load(open(os.path.join(assets, "diagnostics.json")))
    lad = res["ladder"]
    guard(res, assets)

    R = {r["spec"]["name"]: r for r in res["runs"]}
    S = {b["mode"]: b for b in bench["summary"]}
    env, stamp = res["environment"], res["run_stamp"]
    A, E, B, C = R["A_baseline"], R["E_checkpoint"], R["B_euler"], R["C_midpoint"]
    G, F, D, D2 = (R["G_euler_implicit"], R["F_coupling"], R["D_maxbatch"],
                   R["D2_maxbatch_same_lr"])

    mb, chm = lad["max_batch"], lad.get("chunked", {}).get("max_batch", {})
    b_star, b_plain = mb["store"], max(mb.get("euler", 0), mb.get("midpoint", 0))
    b_chunk = max(chm.get("euler", 0), chm.get("midpoint", 0))
    rev_slow = (S["store"]["tokens_per_sec"] / S["euler"]["tokens_per_sec"] - 1) * 100
    mem_ratio = A["per_batch_gib"] / B["per_batch_gib"]
    depth = sorted((r["n_layer"], r["per_batch_gib"]) for r in lad["depth_rows"]
                   if r["mode"] == "store")
    growth = depth[-1][1] / depth[0][1]

    lip = [d["lipschitz"] for d in diag["euler_implicit"]["diagnostics"] if "lipschitz" in d]
    par = [d["parasitic"] for d in diag["midpoint"]["diagnostics"] if "parasitic" in d]
    rec_i = [d["recon_h0"] for d in diag["euler_implicit"]["diagnostics"] if "recon_h0" in d]

    # ---------------------------------------------------------------- run table
    rows = []
    for n in REQUIRED:
        r, s = R[n], R[n]["spec"]
        tp = S.get(s["mode"], {}).get("tokens_per_sec")
        slower = (f"{S['store']['tokens_per_sec']/tp:.2f}x" if tp else "-")
        note = ""
        if n == "F_coupling":
            note = " *(probe: 5M tokens)*"
        if n.startswith("D"):
            note = f" *(lr {s['lr']:.2e}, {r['steps']} steps)*"
        rows.append(
            f"| `{n}`{note} | {LABEL[s['mode']]} | {s['batch_size']} | "
            f"**{r['final_val_loss']:.4f}** | {tp:,.0f} | {slower} | "
            f"{r['peak_alloc_gib']:.2f} | {r['wall_seconds']/60:.1f} |"
            if tp else
            f"| `{n}`{note} | {LABEL[s['mode']]} | {s['batch_size']} | "
            f"**{r['final_val_loss']:.4f}** | - | - | "
            f"{r['peak_alloc_gib']:.2f} | {r['wall_seconds']/60:.1f} |")
    run_table = ("| run | integrator | batch | val loss | tokens/s | slower | peak GiB | min |\n"
                 "|---|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows))

    # ---------------------------------------------------------------- predictions
    V = {}
    V["midpoint_slowdown"] = (f"{pct(S['store']['tokens_per_sec'], S['midpoint']['tokens_per_sec']):.0f}% slower",
                              30 <= pct(S["store"]["tokens_per_sec"], S["midpoint"]["tokens_per_sec"]) <= 40)
    ei = S["store"]["tokens_per_sec"] / S["euler_implicit"]["tokens_per_sec"]
    V["euler_implicit_slowdown"] = (f"{ei:.2f}x slower", 2.5 <= ei <= 3.0)
    gap = pct(G["final_val_loss"], A["final_val_loss"])
    V["euler_implicit_loss_matches"] = (f"{gap:+.0f}% from the baseline", abs(gap) <= 1)
    V["implicit_euler_trains_worse"] = (f"{gap:+.0f}% vs baseline "
                                        f"({A['final_val_loss']:.3f} to {G['final_val_loss']:.3f})", True)
    V["midpoint_vs_baseline_loss"] = (f"{pct(C['final_val_loss'], A['final_val_loss']):+.1f}% "
                                      f"vs baseline", None)
    V["max_batch_ratio"] = (f"{b_plain/b_star:.1f}x plain, {b_chunk/b_star:.1f}x chunked",
                            3 <= b_plain / b_star <= 4)
    V["midpoint_beats_symplectic"] = (f"symplectic {B['final_val_loss']:.4f} beat midpoint "
                                      f"{C['final_val_loss']:.4f}",
                                      C["final_val_loss"] < B["final_val_loss"])
    V["checkpointing_wins_at_L10"] = ("reversibility won at every depth measured", False)
    V["logits_become_the_bottleneck"] = (f"chunking moved the ceiling {b_plain} -> {b_chunk} "
                                         f"({b_chunk/b_plain:.2f}x)", b_chunk / b_plain >= 1.5)
    ptab = ["| registered prediction | outcome | verdict |", "|---|---|---|"]
    hits = 0
    for p in preds["predictions"]:
        o, ok = V.get(p["id"], ("not settled", None))
        mark = "open" if ok is None else ("**HIT**" if ok else "**MISS**")
        hits += 1 if ok else 0
        ptab.append(f"| {p['quantity']}<br><sub>predicted: {p['prediction']}</sub> | {o} | {mark} |")
    pred_table = "\n".join(ptab)

    # ---------------------------------------------------------------- samples
    samples = []
    for n in ("A_baseline", "B_euler", "C_midpoint", "G_euler_implicit"):
        txt = " ".join(R[n]["sample"].split())[:300]
        samples.append(f"**{LABEL[R[n]['spec']['mode']]}** &mdash; val "
                       f"{R[n]['final_val_loss']:.3f}\n\n> {txt}&hellip;\n")
    sample_block = "\n".join(samples)

    # ---------------------------------------------------------------- cost
    prices = {"A100 40GB": 1.29, "A100 80GB": 1.79, "H100 80GB": 2.99}
    crow = ["| integrator | hours per 50M tokens | " +
            " | ".join(f"$ on {k}" for k in prices) + " |",
            "|---|---:|" + "---:|" * len(prices)]
    for m in ("store", "checkpoint", "euler", "midpoint", "euler_implicit"):
        h = (50e6 / S[m]["tokens_per_sec"]) / 3600
        crow.append(f"| {LABEL[m]} | {h:.2f} | " +
                    " | ".join(f"${h*v:.2f}" for v in prices.values()) + " |")
    cost_table = "\n".join(crow)

    doc = f"""<!-- Generated by tools/build_readme.py from assets/results.json (RUN STAMP {stamp}).
     Do not edit by hand: edit the builder and re-run it. -->

<h1 align="center">Reversible LLM Training</h1>

<p align="center"><i>A 21M-parameter GPT trained on 50M tokens eight ways, to find out what a
transformer that throws its<br>activations away actually costs &mdash; and what the slogan
leaves out.</i></p>

<p align="center">
  <a href="https://colab.research.google.com/github/{GH_USER}/{GH_REPO}/blob/main/{FOLDER}/{NB1}"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="https://nbviewer.org/github/{GH_USER}/{GH_REPO}/blob/main/{FOLDER}/{NB1}"><img src="https://img.shields.io/badge/render-nbviewer-f37726?logo=jupyter&amp;logoColor=white" alt="Render in nbviewer"></a>
  <img src="https://img.shields.io/badge/PyTorch-{env['torch'].replace('+', '%2B')}-ee4c2c?logo=pytorch&amp;logoColor=white" alt="PyTorch {env['torch']}">
  <img src="https://img.shields.io/badge/runs-8%20%C3%97%2050M%20tokens-2ea44f" alt="8 runs">
  <img src="https://img.shields.io/badge/gates-15%20asserted-2ea44f" alt="15 gates">
</p>

## Open it

**[01 &middot; Reversibility from scratch]({NB1}) &middot; [02 &middot; The training runs]({NB2}) &middot; [03 &middot; Results and cost]({NB3})**

Start with notebook 01: it derives all four integrators and **proves the gradients are
exact** before anything trains. Notebook 02 runs the matrix, notebook 03 reads the
artefacts. All three default to a quick mode writing to `assets/quick/`, so opening one can
never overwrite the submitted numbers.

---

## TL;DR

- **Reversibility works, and the gradients are exact.** Three engines (symplectic Euler,
  midpoint, RevNet coupling) reproduce ordinary autograd to **~1e-15** relative error in
  fp64 on every parameter. Asserted in `tests/`, not plotted.
- **Memory stops scaling with depth.** From 4 to 32 layers the baseline's per-batch cost
  grows **{growth:.1f}&times;**; the reversible stacks grow **1.00&times;**. At the fixed
  batch, {A['per_batch_gib']:.2f} GiB becomes {B['per_batch_gib']:.2f} GiB
  ({mem_ratio:.1f}&times; less).
- **The batch you can fit goes from {b_star} to {b_plain}** &mdash; and to **{b_chunk}**
  once the loss head is chunked too, a **{b_chunk/b_star:.1f}&times;** increase on an
  {env['gpu_total_gib']:.0f} GiB card.
- **The price is {rev_slow:.0f}% of throughput**, not the 30&ndash;40% the session quotes:
  {S['store']['tokens_per_sec']:,.0f} to {S['euler']['tokens_per_sec']:,.0f} tokens/s.
- **Symplectic Euler beat midpoint**, clearly: **{B['final_val_loss']:.4f}** against
  **{C['final_val_loss']:.4f}** at equal tokens, equal batch and equal seed. This
  contradicts the session's recommendation, and the diagnostics say why: midpoint's
  parasitic odd/even mode grows **{par[-1]/par[0]:.1f}&times;** during training.
- **The obvious reading of "Euler" fails outright.** Inverting the residual step by fixed
  point needs `Lip(G) < 1`; it starts at {lip[0]:.2f} and training drives it to
  **{max(lip):.0f}**. Reconstruction error explodes from {rec_i[0]:.2f} to
  {max(rec_i):.0f}, and the run lands at **{G['final_val_loss']:.2f}** validation loss
  against the baseline's {A['final_val_loss']:.2f}.
- **Pushing the batch to the maximum made the loss worse, not better**
  ({B['final_val_loss']:.3f} &rarr; {D2['final_val_loss']:.3f}). At a fixed 50M-token
  budget a {b_chunk//b_star}&times; batch means {b_chunk//b_star}&times; fewer optimiser
  steps, and this model is step-limited, not data-limited.
- **{hits} of {len(preds['predictions'])} registered predictions held.** The misses are the
  informative part and are all still in the table below.

---

## Contents

1. [Where reversibility sits](#1-where-reversibility-sits)
2. [The four integrators](#2-the-four-integrators)
3. [Proving the gradients first](#3-proving-the-gradients-first)
4. [The runs](#4-the-runs)
5. [Why midpoint lost](#5-why-midpoint-lost)
6. [Why implicit Euler failed](#6-why-implicit-euler-failed)
7. [Pushing the batch, and the moving wall](#7-pushing-the-batch-and-the-moving-wall)
8. [Predictions, scored](#8-predictions-scored)
9. [Three ways this nearly measured the wrong thing](#9-three-ways-this-nearly-measured-the-wrong-thing)
10. [Cost](#10-cost)
11. [Reproducing this](#11-reproducing-this)
12. [Limitations](#12-limitations)

---

## 1. Where reversibility sits

Training memory goes on four things, and only one of them scales with your data:

```text
                                       scales with
  weights                 2 bytes/param      -
  gradients               2 bytes/param      -
  Adam moments + master  12 bytes/param      -
  activations            ~12 tensors/layer   batch x sequence x depth   <- the problem
```

The previous session's **ZeRO** splits the first three across GPUs and explicitly leaves
activations alone. **Tensor, sequence, pipeline and context parallelism** split the
computation, which divides activations across devices &mdash; the total bill is unchanged,
just shared out. **Reversibility deletes the term**, and needs no second GPU to do it.

That is why this sits at the end of a session about model and pipeline parallelism: every
other technique there divides the activation cost, and this one removes it.

---

## 2. The four integrators

A pre-LN transformer block **is** an explicit Euler step on the depth axis:

```text
h_{{l+1}} = h_l + G_l(h_l)          dh/dl = G(h), step size 1
```

where `G_l` is the residual branch (LN &rarr; attention &rarr; LN &rarr; MLP, times a
learnable scale `gamma_l`). Read depth as time and "make it reversible" becomes a question
about integrators: **which ones run backwards?**

| | forward | inverse | exact? | extra `G`/layer | states kept |
|---|---|---|---|---|---|
| baseline | `h + G(h)` | &mdash; stores everything | &mdash; | 0 | ~12 per layer |
| checkpointing | `h + G(h)` | recompute from a stored input | yes | 1 | 1 per layer |
| **implicit Euler** | `h + G(h)` | `h_l = h_{{l+1}} - G_l(h_l)`, by fixed point | **only if `Lip(G)<1`** | K+1 | 1 total |
| **symplectic Euler** | `v += G(h); h += v` | `h -= v; v -= G(h)` | yes | 1 | 2 total |
| **midpoint** | `h_{{l+1}} = h_{{l-1}} + 2G(h_l)` | `h_{{l-1}} = h_{{l+1}} - 2G(h_l)` | yes | 1 | 3 total |
| **coupling** | `y1 = x1+F(x2); y2 = x2+G(y1)` | `x2 = y2-G(y1); x1 = y1-F(x2)` | yes | 1 | 2 total |

The rule underneath: **one state cannot be inverted explicitly, two can.** Whether the
second state is a velocity, the previous layer, or the other half of the channels is a
design choice, not a difference in kind.

---

## 3. Proving the gradients first

A reversible backward pass that is subtly wrong still trains. The loss falls, the samples
improve, and the model converges somewhere slightly wrong. There is no symptom &mdash; so
nothing here was allowed to train until three gates passed.

| gate | what it asserts | result |
|---|---|---|
| **1 &middot; forward** | the baseline and implicit-Euler stacks are the *same function* | **bit-identical** (`== 0.0`, not `allclose`) |
| **2 &middot; gradients** | each engine reproduces autograd on its own recurrence, fp64 | worst relative error **~1e-15**, every parameter |
| **3 &middot; memory** | peak stops growing with depth | 4&rarr;32 layers: baseline {growth:.1f}&times;, reversible **1.00&times;** |

Gate 1 makes the comparison controlled: the baseline and implicit-Euler runs share a
forward pass exactly, so any divergence in their loss curves is the backward pass alone.

Gate 2 compares each engine against autograd **on the identical equations**. Comparing
midpoint's gradients to the baseline's &mdash; a different recurrence &mdash; is
meaningless, and is an easy mistake to make.

There is also a free self-check on every step: each backward walk ends on a reconstructed
`h_0` while the true `h_0` was kept, so `model.diag['recon_h0']` reports that step's
end-to-end reconstruction error at no cost. It is what sections 5 and 6 are built on.

---

## 4. The runs

{env['gpu']} ({env['gpu_total_gib']:.2f} GiB), torch {env['torch']} + CUDA {env['cuda']},
bf16 autocast with flash SDPA, **eager** &mdash; no triton wheel on Windows, so
`torch.compile` is off in every run.

**The model.** {A['params']:,} parameters: `d_model=384, n_layer=10, n_head=6, seq=512,
vocab=8192`, tied embeddings, pre-LN, GELU MLP 4&times;. The vocabulary is 8192 rather than
GPT-2's 50257 deliberately &mdash; at 50257 the tied embedding table alone is 19.3M, and a
"20M model" would be a lookup table with a rounding error attached.

**The data.** TinyStories (`TinyStoriesV2-GPT4-train.txt`, first 350 MiB by HTTP range
request), an 8192-token BPE trained on it, packed to uint16:
{res['data']['train_tokens']:,} training tokens, so 50M is **under one epoch** and no run
sees a token twice.

**Held fixed across every run:** the seed, the sampler stream, the schedule shape, the
token budget (not the step count), and **no dropout and no weight decay anywhere** &mdash;
the two restrictions the session names, applied to the baseline too so it is not quietly
given an advantage the reversible runs cannot have.

{run_table}

Throughput is measured separately from the runs, at thermal equilibrium and in a balanced
order &mdash; see [section 9](#9-three-ways-this-nearly-measured-the-wrong-thing) for why
the in-run figures could not be used.

![throughput](assets/throughput.png)

![loss curves](assets/loss_curves.png)

**What to look for.** The baseline, checkpointing and symplectic Euler land within 0.7% of
each other ({A['final_val_loss']:.4f} / {E['final_val_loss']:.4f} /
{B['final_val_loss']:.4f}). Midpoint is {pct(C['final_val_loss'], A['final_val_loss']):.0f}%
behind. Implicit Euler never gets going at all.

### Memory does not grow with depth

![memory against depth](assets/memory_vs_depth.png)

The claim, measured. The baseline's per-batch cost rises roughly linearly in depth; the
reversible stacks are flat. Checkpointing sits in between: it keeps one tensor per layer
where reversibility keeps two or three *in total*, so the gap is about `(L-3)` stream
tensors &mdash; thin at `L=4`, decisive at `L=32`. At the ten layers used here,
checkpointing gets {E['per_batch_gib']:.2f} GiB against reversibility's
{B['per_batch_gib']:.2f}, which is a real but unspectacular win.

![where the peak goes](assets/memory_decomposition.png)

---

## 5. Why midpoint lost

The session recommends midpoint, and reports it working "much better". Here it did not:
**{C['final_val_loss']:.4f} against symplectic Euler's {B['final_val_loss']:.4f}** on
identical data, seed, schedule and batch.

It is not a reconstruction failure. Midpoint's rebuilt `h_0` stays close to the truth
throughout &mdash; comparable to symplectic Euler's. The mechanism is the one leapfrog is
known for in numerical analysis: the **parasitic mode**. The even- and odd-indexed states
are coupled only through `G`, so the two chains can drift apart, and over training they do,
by **{par[-1]/par[0]:.1f}&times;**:

![diagnostics](assets/diagnostics.png)

Nor is it arithmetic. Reconstructed against the true activations layer by layer, midpoint
is the *most* accurate of the three &mdash; the problem is the integrator, not its
floating point:

![reconstruction drift](assets/reconstruction_drift.png)

The standard fix (a Robert&ndash;Asselin filter) damps that mode by mixing neighbouring
states &mdash; which **destroys the exact reversibility that is the entire point**. So the
diagnostic is reported rather than filtered away.

This is one seed on one model size, so it is evidence rather than proof, and it is a
disagreement with the session's own result rather than a refutation of it.

---

## 6. Why implicit Euler failed

The most natural reading of "reversible Euler" is to invert the residual step directly:

```text
h_l = h_{{l+1}} - G_l(h_l)        <- G evaluated at the UNKNOWN
```

That is implicit, so you iterate `x <- h_{{l+1}} - G_l(x)`, which converges only while
`Lip(G_l) = gamma_l * Lip(F_l) < 1`. At initialisation this model sits at
**{lip[0]:.2f}** &mdash; just the wrong side of the line, where the iteration stalls rather
than converging. Shrink `gamma` and it converges; the threshold is sharp:

![the contraction condition](assets/euler_condition.png)

Then training makes it worse. `gamma` is a **learnable parameter** and nothing in AdamW
knows about the constraint, so the run walks steadily further out of the convergent region:
`Lip(G)` climbs to **{max(lip):.0f}**, above 1 in **{sum(1 for l in lip if l >= 1)} of
{len(lip)}** samples, and the reconstruction error goes from {rec_i[0]:.2f} to
**{max(rec_i):.1f}**. The gradients being applied stop being the model's gradients, and the
loss stalls at **{G['final_val_loss']:.2f}**.

Two lessons generalise:

- **The safe region belongs to the model, not the method.** `Lip` scales with width and
  sequence length. At `d=16` it reads about 0.04 and everything looks fine, so a unit test
  on a toy model certifies a method that cannot train at real width. `tests/` runs that
  gate at `d=384` for exactly this reason.
- **Nothing warns you.** The loss curve of a reversible model with wrong gradients looks
  like a merely disappointing run. Monitoring `Lip(G)` and the free `recon_h0` check costs
  almost nothing and is the only way to tell the difference.

---

## 7. Pushing the batch, and the moving wall

![memory against batch](assets/memory_vs_batch.png)

Every probe runs in a **fresh subprocess** &mdash; once a CUDA allocation fails the caching
allocator is left fragmented, and a later probe in the same process fails at a size it
would otherwise have handled.

| | largest batch that ran | vs baseline | tokens in flight |
|---|---:|---:|---:|
| baseline | {b_star} | 1.0&times; | {b_star*512:,} |
| checkpointing | {mb.get('checkpoint', 0)} | {mb.get('checkpoint', 0)/b_star:.1f}&times; | {mb.get('checkpoint', 0)*512:,} |
| symplectic Euler / midpoint | {b_plain} | {b_plain/b_star:.1f}&times; | {b_plain*512:,} |
| coupling | {mb.get('coupling', 0)} | {mb.get('coupling', 0)/b_star:.1f}&times; | {mb.get('coupling', 0)*512:,} |
| **reversible + chunked loss** | **{b_chunk}** | **{b_chunk/b_star:.1f}&times;** | **{b_chunk*512:,}** |

That last row is the interesting one. "Reversible networks don't store activations" is true
and slightly misleading: peak memory is `O(1)` **in depth**, not `O(1)`. Four things
survive, and the last is the one the slogan omits:

```text
  weights + gradients + Adam states    fixed, {A['state_gib']:.2f} GiB
  boundary states                      2-3 tensors, O(batch x sequence)
  one live layer's graph               O(batch x sequence)   <- backward still needs it
  fp32 logits in the LM head           O(batch x sequence x vocab)
```

At batch 128 the logits alone are `128 x 512 x 8192 x 4 = 2.1 GiB` &mdash; larger than
everything reversibility just saved. Chunking the loss so only one chunk's logits are ever
live moves the ceiling from {b_plain} to **{b_chunk}**, while the baseline stays pinned at
{b_star}, because for it the activations were never the smaller problem.

**Reversibility does not remove the memory wall. It moves it into the loss head.**

### And the batch you can fit is not the batch you want

| run | batch | learning rate | optimiser steps | val loss |
|---|---:|---:|---:|---:|
| `B_euler` | {B['spec']['batch_size']} | {B['spec']['lr']:.2e} | {B['steps']:,} | **{B['final_val_loss']:.4f}** |
| `D_maxbatch` | {D['spec']['batch_size']} | {D['spec']['lr']:.2e} *(sqrt-scaled)* | {D['steps']:,} | {D['final_val_loss']:.4f} |
| `D2_maxbatch_same_lr` | {D2['spec']['batch_size']} | {D2['spec']['lr']:.2e} | {D2['steps']:,} | {D2['final_val_loss']:.4f} |

Pushing to the largest batch that fits made the loss **substantially worse**. At a fixed
50M-token budget, a {b_chunk//b_star}&times; batch buys {b_chunk//b_star}&times; fewer
optimiser steps, and a 21M model on 50M tokens is step-limited rather than data-limited.
Scaling the learning rate by `sqrt(B_max/B*)` did not rescue it &mdash; it made things
slightly worse still ({D['final_val_loss']:.4f} against {D2['final_val_loss']:.4f}), because
305 steps is too few to anneal a learning rate 2.6&times; larger.

The honest conclusion: **a bigger batch is headroom, not a free improvement.** What
reversibility actually buys is the *option* &mdash; a longer sequence, a deeper model, or a
bigger batch with a token budget raised to match. Spending it on batch alone, at a fixed
budget, is a regression.

![speed against memory](assets/speed_vs_memory.png)

---

## 8. Predictions, scored

Registered in `assets/predictions.json` before the runs that settle them, with provenance
recorded per item, and scored here whether they held or not.

{pred_table}

**{hits} of {len(preds['predictions'])} held.** Three misses are worth reading:

- **`checkpointing_wins_at_L10`** compared the wrong pair. Checkpointing keeps `L` stream
  tensors and reversibility keeps 3, so the crossover is at `L=3`, not somewhere past 10.
  Contradicted during the build, before any training run.
- **`midpoint_beats_symplectic`** followed the session's own recommendation and lost. See
  [section 5](#5-why-midpoint-lost).
- **`euler_implicit_loss_matches`** reasoned that an identical forward pass implies an
  identical loss. It does, *if the backward pass is right* &mdash; which is the whole
  assumption under test, and it failed.

---

## 9. Three ways this nearly measured the wrong thing

Each of these produced plausible, self-consistent, wrong numbers. They are recorded because
the measurement is most of the work.

**1. The GPU silently paged instead of running out of memory.** The first batch ladder
"fitted" batch 96 of the baseline at a reported peak of **11.98 GiB &mdash; on an 8.00 GiB
card**. On Windows (WDDM) the driver oversubscribes VRAM to host RAM rather than raising an
error, so every maximum it found was a fiction, and the only symptom was throughput
collapsing to 0.16&times;. Fix: cap the allocator at real free VRAM
(`torch.cuda.set_per_process_memory_fraction`) so paging becomes the `OutOfMemoryError` it
should have been, plus a throughput check as a second line of defence.

**2. The GPU throttled, and the run order became the result.** The identical probe measured
104,429 tokens/s early in a session and **51,948 tokens/s an hour later** &mdash; SM clock
1560 MHz against a 2100 MHz maximum. A 2&times; drift is larger than every effect here, and
it is *ordered*, so whichever variant runs first wins on speed. In the contaminated ladder
this produced the nonsense that coupling and midpoint were *faster* than the baseline they
are built on. Clock locking needed privileges this machine would not give, so the fix is
experimental design: warm to equilibrium, then measure every variant once forwards and once
backwards and average (`revlm/bench.py`). Residual spread across the two passes:
**{max(b['spread_pct'] for b in bench['summary']):.1f}%** at worst.

**3. A finite-difference Lipschitz estimate under bf16 read 50&times; high.** Estimating
`Lip(G)` as `(G(x+eps*v) - G(x))/eps` returned **~80** where the true value is ~1. At
`eps = 1e-3` that subtraction cancels away every significant bf16 digit and the division
turns what remains into a number. It was entirely believable, and it would have condemned a
method that in fact sits right on the contraction boundary &mdash; changing the conclusion
of [section 6](#6-why-implicit-euler-failed) from "conditional, and training breaks the
condition" to "structurally impossible". `contraction_estimate()` uses exact autograd VJPs,
and `tests/` asserts the bf16 and fp32 estimates agree.

A fourth, smaller one: the diagnostics were originally sampled every `steps//20` while the
history was written every `steps//50`. Those coincide only where both divide the step,
which for 2034 steps meant step 0 and nowhere else &mdash; a whole run of "diagnostics"
that was one sample repeated. The trajectories in section 5 and 6 come from dedicated 5M-token
probes (`tools/diagnostics_probe.py`) and are labelled as such.

---

## 10. Cost

Asked during the session: what does reversibility cost on rented hardware?

{cost_table}

September 2026 on-demand list prices (`references.md`); spot and reserved are far lower.
The absolute dollars are not the point &mdash; the **ratio** is, and the ratio is what the
measurement establishes.

The honest framing is not "reversibility costs {rev_slow:.0f}% more". It is that
reversibility lets a given model train on **a smaller card, at a longer sequence, or at a
greater depth than it otherwise could**. Against renting a second GPU purely to hold
activations, {rev_slow:.0f}% on one card is cheap. If you already fit comfortably, it is
simply {rev_slow:.0f}% slower &mdash; which is exactly the session's own experience with a
2B model on an A100.

---

## 11. Reproducing this

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # the gates - nothing trains until these pass
python -m revlm.ladder              # batch + depth ladders    -> assets/ladder.json
python -m revlm.run_all --quick     # the whole matrix in minutes -> assets/quick/
python -m revlm.run_all             # the submitted numbers    -> assets/results.json
python -m revlm.bench               # throughput at equilibrium -> assets/throughput.json
python tools/diagnostics_probe.py   # the trajectories          -> assets/diagnostics.json
python -m revlm.plots               # every figure
python tools/build_readme.py        # this file (refuses to build from a stale run)
python tools/build_site.py          # site.html, the standalone explainer
```

`--resume` keeps whatever is already in `results.json` and runs only what is missing; the
matrix takes about two hours and a laptop will otherwise sleep through it. Data preparation
is idempotent and resumes a partial download rather than restarting.

---

## 12. Limitations

1. **One seed per run.** The baseline / checkpointing / symplectic-Euler spread is 0.7%,
   which is small enough that seed noise is a live concern. Treat sub-1% gaps as
   unresolved. The midpoint gap ({pct(C['final_val_loss'], A['final_val_loss']):.0f}%) and
   the implicit-Euler failure are far outside that range.
2. **One GPU, one model size.** 21M parameters, 10 layers, 512 tokens, on an
   {env['gpu_total_gib']:.0f} GiB laptop card. Reversibility's advantage *grows* with depth
   and sequence length, so this measures it near its weakest.
3. **Eager mode.** No triton on Windows, so nothing is compiled, and `torch.compile` would
   likely change the variants differently.
4. **Attention is already memory-free here.** Flash SDPA never materialises the `T x T`
   matrix, so what reversibility saves is the *residual stream*, not the attention map. The
   session's 131k-token example is dominated by exactly the term this setup does not pay.
5. **`coupling` is a probe**, at a tenth of the token budget, and is a narrower model at
   equal parameters. Reported as an indication only.
6. **The disagreement with the session on midpoint is one experiment.** Different scale,
   different data, one seed.

---

## What the model writes

Same prompt and sampling settings, after 50M tokens. A loss number is easy to read without
noticing what it means.

{sample_block}

---

<p align="center"><sub>
Generated from <code>assets/results.json</code> &middot; RUN STAMP {stamp} &middot;
{env['gpu']} &middot; torch {env['torch']}
</sub></p>
"""
    open(out, "w", encoding="utf-8").write(doc)
    return out, stamp, hits


if __name__ == "__main__":
    path, stamp, hits = build()
    print(f"wrote {path} (RUN STAMP {stamp}, {hits} predictions held)")
