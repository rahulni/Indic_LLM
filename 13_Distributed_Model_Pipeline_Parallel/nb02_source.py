# %% [markdown]
# # Training a 21M GPT on 50M tokens, six ways
#
# **The assignment, run.** Fix the largest batch the baseline can handle. Train again with
# reversibility at that same batch, testing more than one variant. Then push the batch as
# far as it will go. Report final loss, tokens/second, peak memory — and anything else the
# runs turn up.
#
# This notebook is the harness. The derivations and the correctness proofs are in
# [`01_reversibility_from_scratch.ipynb`](01_reversibility_from_scratch.ipynb) and are
# worth reading first — nothing here is meaningful unless the gradients are right, and
# that is established there rather than assumed here.
#
# ### The matrix
#
# ```text
#   A  baseline, activations stored        B*       the assignment's fixed batch
#   E  baseline + gradient checkpointing   B*       the honest rival
#   B  symplectic Euler                    B*       assignment variant 1
#   C  midpoint / leapfrog                 B*       assignment variant 2
#   G  implicit Euler (fixed point)        B*       the obvious reading of "Euler"
#   F  RevNet coupling                     B*       the "etc" variant, a short probe
#   D  whichever of B/C won                B_max    "push the batch size"
#   D2 the same, at the unscaled LR        B_max    so the confound is visible
# ```
#
# **E is the run most reports leave out.** Reversibility's real competitor is not "store
# everything" — it is plain gradient checkpointing, which already buys most of the memory
# back for a similar slowdown and no maths at all. Without E on the table, reversibility
# looks far better than it is.
#
# ### Two things held fixed, deliberately
#
# **A token budget, not a step budget.** A run at batch 192 and a run at batch 48 both see
# exactly 50M tokens. Comparing by steps would hand the large-batch run 4x the data.
#
# **No dropout and no weight decay in *any* run.** Reversibility forbids both (dropout
# would need the backward pass to replay the same random mask; the session names weight
# decay as the other restriction). Applying them to the baseline only would quietly give it
# an advantage the reversible runs are not allowed to have.

# %%
QUICK_RUN = True   # a few minutes, writes to assets/quick/. Set False to reproduce the README.

import os, sys, json, math

if not os.path.exists("revlm"):
    for parent in (".", "..", "../.."):
        if os.path.exists(os.path.join(parent, "revlm")):
            sys.path.insert(0, os.path.abspath(parent)); break
    else:
        print("revlm/ not found - upload the package folder next to this notebook")

import torch
from revlm import data as D, ladder as L, plots
from revlm.train import RunSpec, train_one, environment, save
from revlm.model import GPT, GPTConfig

plots.style()
OUT = "assets/quick" if QUICK_RUN else "assets"
TOKENS = 2_000_000 if QUICK_RUN else 50_000_000
os.makedirs(OUT, exist_ok=True)
print(json.dumps(environment(), indent=2))
print(f"\nwriting to {OUT}/  ·  token budget {TOKENS:,}")

# %% [markdown]
# ## 1. The data
#
# TinyStories, first 350 MiB by HTTP range request, an 8192-token BPE trained on it, packed
# to a flat `uint16` stream.
#
# Both choices are load-bearing. **TinyStories** because a 21M model trained on 50M tokens
# of general web text produces fluent-sounding noise, while on TinyStories it produces
# actual stories — so the variants can be compared on what they *write*, not only on a
# loss number. **8192 tokens** because at GPT-2's 50257 the tied embedding table alone is
# 19.3M parameters, and a "20M model" would have no transformer left inside it.

# %%
meta = D.prepare("data")
print(f"\n  {meta['train_tokens']:,} training tokens")
print(f"  50M tokens is {50e6/meta['train_tokens']*100:.0f}% of one epoch - nothing repeats")

# %% [markdown]
# ## 2. Finding B* and B_max
#
# `B*` is the largest batch the **baseline** survives — the assignment's "fix the batch size
# you can run". `B_max` is the largest a reversible stack survives.
#
# Each probe runs in a **fresh subprocess**. That is not fastidiousness: once a CUDA
# allocation fails, the caching allocator is left fragmented, and a later probe in the same
# process can fail at a size it would otherwise have handled. An in-process ladder reports a
# maximum batch that is too low — and reports it consistently enough to look trustworthy.

# %%
if QUICK_RUN:
    B_STAR, B_MAX = 8, 32
    ladder_data = {"note": "quick mode skips the ladder"}
    print(f"quick mode: B* = {B_STAR}, B_max = {B_MAX} (fixed, not measured)")
else:
    if not os.path.exists("assets/ladder.json"):
        L.main(["revlm.ladder"])
    ladder_data = json.load(open("assets/ladder.json"))
    mb = ladder_data["max_batch"]
    B_STAR = mb["store"]
    B_MAX = max(mb.get("midpoint", 0), mb.get("euler", 0))
    print(f"B* = {B_STAR} (largest the baseline survives)")
    print(f"B_max = {B_MAX} ({B_MAX/B_STAR:.1f}x more sequences in flight)")
    for mode, b in mb.items():
        print(f"    {mode:<16} {b:4d}  ({b*512:,} tokens per step)")

# %% [markdown]
# ## 3. The runs
#
# Roughly two hours at the full budget. Each run streams its results to disk as it
# finishes, so a crash keeps whatever completed.

# %%
matrix = [
    RunSpec(name="A_baseline",       mode="store",          batch_size=B_STAR, tokens=TOKENS),
    RunSpec(name="E_checkpoint",     mode="checkpoint",     batch_size=B_STAR, tokens=TOKENS),
    RunSpec(name="B_euler",          mode="euler",          batch_size=B_STAR, tokens=TOKENS),
    RunSpec(name="C_midpoint",       mode="midpoint",       batch_size=B_STAR, tokens=TOKENS),
    RunSpec(name="G_euler_implicit", mode="euler_implicit", batch_size=B_STAR, tokens=TOKENS,
            euler_iters=4),
    RunSpec(name="F_coupling",       mode="coupling",       batch_size=B_STAR,
            tokens=max(1, TOKENS // 10)),
]

results = []
payload = {"run_stamp": "", "environment": environment(), "tokens_budget": TOKENS,
           "b_star": B_STAR, "b_max": B_MAX, "data": dict(meta),
           "ladder": ladder_data, "runs": results}

for spec in matrix:
    results.append(train_one(spec, meta, OUT))
    save(payload, OUT)

# %% [markdown]
# ## 4. Push the batch
#
# Whichever of symplectic Euler and midpoint reached the lower validation loss gets run
# again at `B_max`.
#
# **The confound, declared.** At a fixed token budget, a 4x larger batch means 4x fewer
# optimiser steps, and fewer steps at the same learning rate means a worse loss — for
# reasons that have nothing to do with reversibility. So this runs twice: once with the
# learning rate scaled by `sqrt(B_max/B*)`, once unchanged. The gap between those two *is*
# the batch-size effect, and it can be read off rather than argued about.

# %%
scored = {r["spec"]["name"]: r["final_val_loss"] for r in results
          if r["spec"]["name"] in ("B_euler", "C_midpoint")}
winner = min(scored, key=scored.get)
mode = "euler" if winner == "B_euler" else "midpoint"
scale = math.sqrt(B_MAX / B_STAR)
print(f"winner on validation loss: {winner} ({scored[winner]:.4f}) -> pushing to {B_MAX}")

for spec in [
    RunSpec(name="D_maxbatch", mode=mode, batch_size=B_MAX, tokens=TOKENS,
            lr=RunSpec.lr * scale, ce_chunks=4),
    RunSpec(name="D2_maxbatch_same_lr", mode=mode, batch_size=B_MAX, tokens=TOKENS,
            ce_chunks=4),
]:
    results.append(train_one(spec, meta, OUT))
    save(payload, OUT)

import datetime
payload["run_stamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
payload["winner"] = winner
save(payload, OUT)
print(f"\nRUN STAMP {payload['run_stamp']}")

# %% [markdown]
# ## 5. The table

# %%
base = results[0]
print(f"{'run':<22}{'batch':>6}{'val loss':>10}{'tok/s':>9}{'vs base':>9}{'peak GiB':>10}{'min':>7}")
for r in results:
    s = r["spec"]
    print(f"{s['name']:<22}{s['batch_size']:>6}{r['final_val_loss']:>10.4f}"
          f"{r['tokens_per_sec']:>9,.0f}{base['tokens_per_sec']/r['tokens_per_sec']:>8.2f}x"
          f"{r['peak_alloc_gib']:>10.2f}{r['wall_seconds']/60:>7.1f}")

# %% [markdown]
# ## 6. Did the reconstruction hold up?
#
# Every reversible step ends its backward walk on a rebuilt `h_0` while the true `h_0` was
# kept, so each run carries its own end-to-end reconstruction error — sampled through
# training, not just checked once at the start.
#
# For the implicit-Euler run there is a second number to watch: `Lip(G)`. Its fixed-point
# inverse only converges while that stays below 1, and `gamma` is learnable, so the run can
# walk out of the safe region on its own.

# %%
for r in results:
    diags = r["diagnostics"]
    if not diags:
        continue
    name = r["spec"]["name"]
    worst = max(d.get("recon_h0", 0) for d in diags)
    line = f"  {name:<22} worst reconstruction error {worst:.3e}"
    lips = [d["lipschitz"] for d in diags if "lipschitz" in d]
    if lips:
        line += f"   Lip(G) {min(lips):.2f} -> {max(lips):.2f}"
        line += "  CONTRACTION LOST" if max(lips) >= 1 else "  (stayed below 1)"
    par = [d["parasitic"] for d in diags if "parasitic" in d]
    if par:
        line += f"   parasitic mode {par[0]:.2e} -> {par[-1]:.2e}"
    print(line)

# %% [markdown]
# ## 7. What they write
#
# The same prompt and sampling settings for every variant. A loss number is easy to read
# without noticing what it means; this is harder to fool yourself about.

# %%
for r in results:
    if r["spec"]["name"].startswith(("D2", "F_")):
        continue
    print(f"--- {r['spec']['name']}  (val {r['final_val_loss']:.4f}) " + "-" * 24)
    print(" ".join(r["sample"].split())[:400], "...\n")

# %% [markdown]
# ## 8. Figures

# %%
if not QUICK_RUN:
    for p in plots.all_figures(OUT):
        print("wrote", p)
    from IPython.display import Image, display
    for f in ("loss_curves.png", "memory_vs_batch.png", "memory_decomposition.png"):
        display(Image(os.path.join(OUT, f)))

# %% [markdown]
# Analysis, the scored predictions and the cost comparison are in
# [`03_results_and_cost.ipynb`](03_results_and_cost.ipynb), which reads the artefacts this
# notebook just wrote.
