# %% [markdown]
r"""
## §5 The update-to-weight ratio, layer by layer, and when warmup stops shaping it

**Intuition.** A weight's absolute update means little on its own. A step of $10^{-3}$ is huge for a
weight of size $10^{-2}$ and negligible for one of size 10. The scale-free quantity is
$$r = \frac{\lVert \Delta W\rVert}{\lVert W\rVert},$$
the fraction of itself a tensor moves in one step. A healthy run keeps it near $10^{-3}$. (Some
write it upside down, "weight-to-update"; the information is the same.)

**Why warmup exists, in these terms.** Adam moves every weight by about $\eta\cdot\rho$:
- $\rho \approx 1$ when the gradients agree from step to step.
- $\rho$ is much smaller when they are noise.

At initialization almost every weight is wrong in the same direction, so gradients agree and
$\rho \approx 1$. This is exactly when weights are smallest relative to their updates. Warmup ramps
$\eta$ up while $\rho$ falls.

**What "warmup stops changing the ratio" means here (rule, §0).**
- Compare a run warmed up over $W$ steps with one that is not, at the same peak learning rate.
- After warmup the learning rate is held constant, so any later difference comes from the warmup
  alone.
- For each tensor, warmup has stopped changing the ratio at the first step after which the 9-step
  rolling median stays within 10% of the no-warmup run's.

Warmup changes the ratio from step 1 by construction, because the learning rate itself is smaller.
So the informative number is where its influence **ends**.

**Runs.**
- Warmup $W \in \{0, 6, 30, 100\}$ steps, then a constant peak learning rate, base model.
- Every tensor's ratio logged on every step.
- A **RAdam** run with no warmup at all. RAdam replaces warmup with a correction for the noise of
  $v$ in early steps.
"""

# %%
WARMUPS = qk([0, 6, 30, 100], [0, 2, 6, 20])
ratio_runs = {w: run(base_cfg(LR_BASE, kind="constant", warmup=w, log_every=1, detail_every=qk(10, 5),
                              eval_at=[STOP, TOTAL], sample_at=[])) for w in WARMUPS}
radam = run(base_cfg(LR_BASE, kind="constant", warmup=0, opt="radam", log_every=1, eval_at=[STOP, TOTAL], sample_at=[]))

names = ratio_runs[0]["tensors"]
MATS = [i for i, n in enumerate(names) if n == "emb" or (n.startswith("b") and n.split(".")[-1] in ("q", "kv", "o", "up", "down"))]


def rolling_median(a, k=RULES["warmup_stop"]["window"]):
    pad = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k, axis=0), axis=-1)


def ratio_matrix(res, key="total"):
    r = np.array(res["ratios"][key], dtype=float)                       # (steps, tensors)
    return r


rm = {w: rolling_median(ratio_matrix(ratio_runs[w])) for w in WARMUPS}
n_steps = min(len(v) for v in rm.values())
WSTOP = {}
for w in WARMUPS[1:]:
    rel = rm[w][:n_steps] / rm[0][:n_steps] - 1
    WSTOP[w] = {p: [stop_step(rel[:, i], p) for i in MATS] for p in RULES["warmup_stop"]["p"]}
    final = np.median(rm[w][-50:], axis=0)
    plateau = [stop_step(rm[w][:, i] / final[i] - 1, 0.10) for i in MATS]
    peak_gap = [int(np.argmax(np.abs(np.log(rm[w][:n_steps, i] / rm[0][:n_steps, i])))) + 1 for i in MATS]
    s10 = np.array(WSTOP[w][0.10])
    fmt = lambda v: f"{int(v)}" if v <= n_steps else f"never (>{n_steps})"
    print(f"warmup {w:>3}: warmup stops changing the ratio at step {fmt(np.median(s10))} (median over {len(MATS)} matrices; "
          f"range {fmt(s10.min())}–{fmt(s10.max())}).  At 5%: {fmt(np.median(WSTOP[w][0.05]))}; at 20%: {fmt(np.median(WSTOP[w][0.20]))}.  "
          f"The ratio itself settles by step {fmt(np.median(plateau))}; warmup's largest effect is at step {int(np.median(peak_gap))}")
    WSTOP[w]["plateau"], WSTOP[w]["peak_gap"] = plateau, peak_gap

# Exploratory, not pre-registered: when does the warmup *length* stop mattering? Compare warmed-up runs with each
# other (a shorter warmup against the next longer one) using the same ±10% rolling-median rule.
PAIRWISE = {}
for a_, b_ in zip(WARMUPS[1:], WARMUPS[2:]):
    rel = rm[a_][:n_steps] / rm[b_][:n_steps] - 1
    PAIRWISE[f"{a_} vs {b_}"] = int(np.median([stop_step(rel[:, i], 0.10) for i in MATS]))
print("exploratory (not pre-registered): warmup length stops mattering at step  "
      + ",  ".join(f"{k}: {v if v <= n_steps else 'never'}" for k, v in PAIRWISE.items()) + "  (median over matrices, ±10%)")

mx = {w: float(np.nanmax(ratio_matrix(ratio_runs[w])[:, MATS])) for w in WARMUPS}
first = {w: float(np.median(ratio_matrix(ratio_runs[w])[0, MATS])) for w in WARMUPS}
print("\nlargest ratio any matrix sees in the run: " + ", ".join(f"warmup {w}: {mx[w]:.2e}" for w in WARMUPS))
print("median ratio at step 1:                  " + ", ".join(f"warmup {w}: {first[w]:.2e}" for w in WARMUPS))
observe("warmup_effect_ends_in_run", all(np.median(WSTOP[w][0.10]) <= n_steps for w in WARMUPS[1:]),
        f"expected warmup's effect on the update/weight ratio (against no warmup) to end within the {n_steps}-step run, "
        f"for every warmup length")
observe("warmup_lowers_max_ratio", mx[WARM] < mx[0], f"warmup over {WARM} steps lowers the largest ratio from {mx[0]:.2e} to {mx[WARM]:.2e}")
observe("warmup_stop_after_warmup", all(np.median(WSTOP[w][0.10]) >= w for w in WARMUPS[1:]),
        "for every warmup length, the ratio stops being affected only after the warmup itself has ended")

# %%
w_show = WARM
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
mat = np.log10(ratio_matrix(ratio_runs[w_show])[:, MATS].T)
vlo, vhi = np.nanpercentile(mat, [2, 98])
im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=vlo, vmax=vhi, interpolation="nearest",
               extent=(0.5, mat.shape[1] + 0.5, len(MATS) - 0.5, -0.5))
ax.set_yticks(range(len(MATS)), [names[i] for i in MATS], fontsize=6.5)
ax.grid(False)
ax.axvline(w_show, color=INK, lw=0.8, ls="--")
ax.set(xlabel="step", title=f"log10 update/weight ratio, every matrix (warmup {w_show})")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(colors=MUTED)
ax = axs[1]
med = {w: np.median(ratio_matrix(ratio_runs[w])[:, MATS], axis=1) for w in WARMUPS}
for w, col in zip(WARMUPS, (C["red"], C["yellow"], C["blue"], C["aqua"])):
    ax.plot(np.arange(1, len(med[w]) + 1), med[w], color=col, lw=1.2)
    label_end(ax, len(med[w]), med[w][-1], f"warmup {w}", col)
rmed = np.median(ratio_matrix(radam)[:, MATS], axis=1)
ax.plot(np.arange(1, len(rmed) + 1), rmed, "--", color=C["violet"], lw=1.1); label_end(ax, len(rmed), rmed[-1], "RAdam, no warmup", C["violet"], dy=-9)
ax.axhline(LR_BASE * 0.1, color=MUTED, lw=0.8, ls=":")
ax.text(3, LR_BASE * 0.1 * 1.15, "decay floor ηλ", color=MUTED, fontsize=8)
ax.axhline(1e-3, color=MUTED, lw=0.6)
ax.set(yscale="log", xscale="log", xlabel="step", xlim=(1, TOTAL * 3.5), title="median over matrices")
fig.tight_layout()
finish(fig, "update_ratio_warmup")

# %% [markdown]
r"""
**Taking the ratio apart.** Three different things push the measured ratio around:
1. **The decay part.** AdamW shrinks every matrix by $\eta\lambda$ per step regardless of the
   gradient. That is a floor under the ratio, shown dotted above, and at short horizons it is not
   negligible.
2. **Adam's "fraction of η".** The gradient part of the update is $\eta\cdot\hat m/(\sqrt{\hat v}+\epsilon)$.
   Its RMS divided by $\eta$ is the $\rho$ above. It should start near 1 and fall as the gradients
   decorrelate.
3. **The weights themselves.** $\lVert W\rVert$ grows or shrinks as training proceeds.

Two more views from the same runs:
- The **spectral** ratio $\lVert\Delta W\rVert_2/\lVert W\rVert_2$ is how much the update can change
  the layer's output along its most-amplified direction. This is the quantity muP and Muon reason
  about.
- The **share of updates smaller than half a bf16 spacing** is the share that would be rounded away
  if the weights were stored in bf16. That is why training keeps an fp32 master copy.
"""

# %%
adam_frac = {w: np.median(ratio_matrix(ratio_runs[w], "adam_frac")[:, MATS], axis=1) for w in (0, WARM)}
grad_part = {w: np.median(ratio_matrix(ratio_runs[w], "grad")[:, MATS], axis=1) for w in (0, WARM)}
det = ratio_runs[WARM]["detail"]
dsteps = np.array(det["steps"])
spec = np.array(det["spectral"], dtype=float)
frob_ids = [names.index(n) for n in ("b0.q", f"b{BASE_ARCH['n_layer'] // 2}.up", f"b{BASE_ARCH['n_layer'] - 1}.down", "emb")]
frob = ratio_matrix(ratio_runs[WARM])[dsteps - 1][:, frob_ids]
lost = np.array(det["bf16_lost"], dtype=float)[:, MATS]
lost_floor = np.array(det["bf16_lost_floor"], dtype=float)[:, MATS]
rows = np.array([[np.nan if x is None else x for x in r] for r in det["emb_rows"]], dtype=float)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
ax = axs[0]
for w, col in ((0, C["red"]), (WARM, C["blue"])):
    ax.plot(np.arange(1, len(adam_frac[w]) + 1), adam_frac[w], color=col); label_end(ax, len(adam_frac[w]), adam_frac[w][-1], f"warmup {w}", col)
ax.set(xscale="log", xlabel="step", ylabel="RMS(m̂/√v̂)", title="Adam's step as a fraction of η", xlim=(1, TOTAL * 3), ylim=(0, 1.1))
ax = axs[1]
for j, (nm, col) in enumerate(zip(("b0.q", "mid up", "last down", "emb"), SERIES)):
    ax.plot(dsteps, spec[:, j] / frob[:, j], "-o", ms=2.5, color=col); label_end(ax, dsteps[-1], spec[-1, j] / frob[-1, j], nm, col)
ax.axhline(1, color=MUTED, lw=0.7)
ax.set(xlabel="step", ylabel="spectral / Frobenius ratio", title="Updates concentrate in a few directions", xlim=(0, TOTAL * 1.35))
ax = axs[2]
for arr, col, lab in ((lost, C["orange"], "at the peak lr"), (lost_floor, C["violet"], "at a 10x smaller lr")):
    ax.plot(dsteps, np.median(arr, 1) * 100, "-o", ms=2.5, color=col)
    ax.fill_between(dsteps, np.percentile(arr, 10, 1) * 100, np.percentile(arr, 90, 1) * 100, color=col, alpha=0.2, lw=0)
    label_end(ax, dsteps[-1], np.median(arr[-1]) * 100, lab, col)
ax.set(xlabel="step", ylabel="% of elements", ylim=(0, 100), xlim=(0, TOTAL * 1.6), title="Updates bf16 storage would round away")
fig.tight_layout()
finish(fig, "update_ratio_anatomy")

decay_share = float(np.median(1 - grad_part[WARM][-50:] / med[WARM][-50:]))
print(f"Adam's step as a fraction of η: step 1 {adam_frac[0][0]:.2f} (no warmup), last 50 steps {np.median(adam_frac[WARM][-50:]):.2f}")
print(f"the decay floor ηλ = {LR_BASE * 0.1:.1e}; over the last 50 steps the median matrix ratio is {np.median(med[WARM][-50:]):.2e}")
print(f"spectral ratio / Frobenius ratio, last logged step: " + ", ".join(f"{n} {v:.1f}x" for n, v in zip(("b0.q", "mid up", "last down", "emb"), spec[-1] / frob[-1])))
print(f"share of matrix elements whose update is below half a bf16 spacing: {np.median(lost[-5:]) * 100:.0f}% at the peak lr, "
      f"{np.median(lost_floor[-5:]) * 100:.0f}% if the lr were 10x smaller, as at the end of a decay (median, last logged steps)")
labels = ["100 most frequent", "rank 100–1k", "rank 1k–8k", "rank ≥ 8k (rare/unseen)", "in this batch", "not in this batch"]
print("embedding rows, median update/weight ratio over the run:  " + ";  ".join(f"{l} {np.nanmedian(rows[:, i]):.1e}" for i, l in enumerate(labels)))
observe("adam_fraction_falls", adam_frac[0][0] > 2 * np.median(adam_frac[WARM][-50:]),
        f"Adam's step falls from {adam_frac[0][0]:.2f} η at step 1 to {np.median(adam_frac[WARM][-50:]):.2f} η once gradients decorrelate")
observe("bf16_would_lose_updates_late", np.median(lost_floor[-5:]) > 0.5,
        "at a decayed learning rate most updates are smaller than half a bf16 spacing: without an fp32 copy they would vanish")
observe("rare_rows_move_tied", np.nanmedian(rows[:, 3]) > 0.1 * np.nanmedian(rows[:, 0]),
        "with a tied head, rare-token rows still move every step (the output softmax gives every row a gradient)")
print(f"\nRAdam with no warmup: final validation loss {final_loss(radam):.4f}; "
      + ", ".join(f"AdamW warmup {w}: {final_loss(ratio_runs[w]):.4f}" for w in WARMUPS) + "  (one seed each)")

if MODE != "learn":
    ANS["warmup"] = _clean(dict(
        warm=WARM, n_steps=n_steps, pairwise=PAIRWISE,
        stop={str(w): {f"{p:.0%}": int(np.median(v[p])) for p in RULES["warmup_stop"]["p"]} for w, v in WSTOP.items()},
        stop_range={str(w): [int(min(v[0.10])), int(max(v[0.10]))] for w, v in WSTOP.items()},
        plateau={str(w): int(np.median(v["plateau"])) for w, v in WSTOP.items()},
        peak_gap={str(w): int(np.median(v["peak_gap"])) for w, v in WSTOP.items()},
        max_ratio={str(w): v for w, v in mx.items()}, first_ratio={str(w): v for w, v in first.items()},
        adam_frac_first=adam_frac[0][0], adam_frac_late=float(np.median(adam_frac[WARM][-50:])),
        decay_floor=LR_BASE * 0.1, late_ratio=float(np.median(med[WARM][-50:])), decay_share=decay_share,
        spectral_over_frob=(spec[-1] / frob[-1]).tolist(), bf16_lost=float(np.median(lost[-5:])),
        bf16_lost_floor=float(np.median(lost_floor[-5:])),
        emb_rows={l: float(np.nanmedian(rows[:, i])) for i, l in enumerate(labels)},
        val={str(w): final_loss(r) for w, r in ratio_runs.items()}, radam_val=final_loss(radam)))
    save_results()

# %% [markdown]
r"""
**What the measurements say.**

- **Skipping warmup is not transient.** Without warmup, the first steps move some matrices by up to
  20% of their size in one step. That run never catches up: its validation loss ends about 0.65
  higher, and its update/weight ratios stay well below those of the warmed-up runs to the end. By the
  rule fixed in §0, warmup's effect on the ratio **never ends** within 300 steps.
- **The length of warmup stops mattering much sooner.** Warmed-up runs of different lengths agree with
  each other from about step 140–150 (exploratory readout above). Their ratios settle near
  $9\times10^{-3}$ by about step 160, whatever the warmup length.
- **RAdam with no warmup recovers most, but not all, of the damage.**

> **Carry forward:** log $\lVert\Delta W\rVert/\lVert W\rVert$ for every matrix from step one.
> Warmup's job is the early steps where Adam moves every weight by nearly its full $\eta$. Skipping
> it leaves a lasting mark, while its exact length washes out within about 150 steps here.
"""
