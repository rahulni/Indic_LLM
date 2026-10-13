# %% [markdown]
r"""
## §4 Bias correction in a real model

§2 answered "when does bias correction stop mattering?" for one weight, in closed form. A model
has 31.5M weights, an ε that competes with tiny gradients, and warmup scaling down exactly the
early steps where the correction acts. So the question is asked again here, of the loss.

**Setup.**
- The base model at its tuned learning rate, cosine schedule, 300 steps.
- `HandAdamW` with bias correction on and off.
- Crossed with $\beta_2 \in \{0.95, 0.999\}$ and warmup $\in \{0, 30\}$ steps.
- All eight runs share seed 0, so they start from the same weights and see the same batches.

**Rule (§0).** The difference stops mattering at the first step after which the 20-step-EMA loss
gap between "off" and "on" stays inside ±2 seed standard deviations, as measured in §3.

**Counterfactual.** In every "on" run, the optimizer also computes what the uncorrected update would
have been at that exact step, from the same state. The ratio of the two, per tensor, can be set
against §2's closed form $f(t)$. A tensor whose ratio falls below $f(t)$ is one where ε is
competing with $\sqrt{v}$.
"""

# %%
BC_RUNS = {}
for b2 in (0.95, 0.999):
    for w in (0, WARM):
        for bc in (True, False):
            BC_RUNS[(b2, w, bc)] = run(base_cfg(LR_BASE, warmup=w, opt="hand", betas=(0.9, b2), bias_correction=bc,
                                                track=bc, log_every=0))

band = RULES["bias_correction_stop"]["band_sd"] * SEED_SD
MODEL_STOP, VAL_TABLE = {}, {}
print(f"seed band: ±{band:.4f} (2 s.d. of the smoothed training loss, from §3)\n")
print(f"{'β2':>6} {'warmup':>7} {'val on':>8} {'val off':>8} {'off - on':>9}   loss gap stays inside the seed band from step")
for b2 in (0.95, 0.999):
    for w in (0, WARM):
        on_r, off_r = BC_RUNS[(b2, w, True)], BC_RUNS[(b2, w, False)]
        n = min(len(on_r["hist"]["loss"]), len(off_r["hist"]["loss"]))
        gap = ema_curve(off_r["hist"]["loss"][:n], 20) - ema_curve(on_r["hist"]["loss"][:n], 20)
        stop = stop_step(np.where(np.isfinite(gap), gap, np.inf), band) if n == TOTAL else None
        stop = stop if (stop is not None and stop <= n) else None
        MODEL_STOP[(b2, w)] = stop
        v_on, v_off = final_loss(on_r), final_loss(off_r)
        VAL_TABLE[(b2, w)] = (v_on, v_off)
        div = f" (off diverged at {off_r['diverged_at']})" if off_r["diverged_at"] else ""
        print(f"{b2:>6} {w:>7} {v_on:8.4f} {v_off:8.4f} {v_off - v_on:+9.4f}   "
              f"{stop if stop else 'never within ' + str(TOTAL)}{div}")

val_sd = ANS.get("base", {}).get("val_sd") or float(np.std([final_loss(cos_seeds[s]) for s in SEEDS], ddof=1))
d = {k: v[1] - v[0] for k, v in VAL_TABLE.items()}
observe("bc_off_hurts_0999_no_warmup", d[(0.999, 0)] > 2 * val_sd,
        f"β2=0.999 without warmup: switching bias correction off costs {d[(0.999, 0)]:+.3f} in final loss (seed s.d. {val_sd:.3f})")
observe("warmup_hides_bc_0999", abs(d[(0.999, WARM)]) < abs(d[(0.999, 0)]),
        f"expected warmup to shrink that cost (β2=0.999); measured {d[(0.999, 0)]:+.3f} without warmup and "
        f"{d[(0.999, WARM)]:+.3f} with it")
observe("bc_negligible_095", max(abs(d[(0.95, 0)]), abs(d[(0.95, WARM)])) < 2 * val_sd,
        f"expected switching bias correction off to be harmless at β2=0.95 (within 2 seed s.d. = {2 * val_sd:.3f}); "
        f"measured {d[(0.95, 0)]:+.3f} without warmup and {d[(0.95, WARM)]:+.3f} with it")

# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
for row, b2 in enumerate((0.95, 0.999)):
    ax = axs[row, 0]
    for w, ls in ((0, "-"), (WARM, ":")):
        for bc, col in ((True, C["blue"]), (False, C["orange"])):
            ys = BC_RUNS[(b2, w, bc)]["hist"]["loss"][:20]
            ax.plot(np.arange(1, len(ys) + 1), ys, ls, color=col, lw=1.3)
    ax.set(title=f"β2={b2}: first 20 steps (solid: no warmup, dotted: warmup {WARM})", ylabel="training loss")
    ax = axs[row, 1]
    for w, col in ((0, C["orange"]), (WARM, C["aqua"])):
        on_r, off_r = BC_RUNS[(b2, w, True)], BC_RUNS[(b2, w, False)]
        n = min(len(on_r["hist"]["loss"]), len(off_r["hist"]["loss"]))
        gap = ema_curve(off_r["hist"]["loss"][:n], 20) - ema_curve(on_r["hist"]["loss"][:n], 20)
        ax.plot(np.arange(1, n + 1), gap, color=col)
        lab = f"warmup {w}: inside from step {MODEL_STOP[(b2, w)]}" if MODEL_STOP[(b2, w)] else f"warmup {w}: never inside"
        label_end(ax, n, gap[-1], lab, col)
    ax.axhspan(-band, band, color=MUTED, alpha=0.18, lw=0)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set(title=f"β2={b2}: loss(off) − loss(on), 20-step EMA; grey = seed band", xlim=(0, TOTAL * 1.6))
axs[1, 0].set_xlabel("step"); axs[1, 1].set_xlabel("step")
fig.tight_layout()
finish(fig, "bias_correction_model")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), sharey=False)
CF_DEV = {}
for ax, b2 in zip(axs, (0.95, 0.999)):
    r = BC_RUNS[(b2, 0, True)]["counterfactual"]
    st = np.array(r["steps"]); ratio = np.array(r["ratio"], dtype=float)
    closed = (1 - 0.9 ** st) / np.sqrt(1 - b2 ** st)
    ax.fill_between(st, np.nanpercentile(ratio, 10, 1), np.nanpercentile(ratio, 90, 1), color=C["blue"], alpha=0.25, lw=0)
    ax.plot(st, np.nanmedian(ratio, 1), color=C["blue"]); label_end(ax, st[-1], np.nanmedian(ratio[-1]), "measured (median, 10–90%)", C["blue"])
    ax.plot(st, closed, "--", color=C["orange"]); label_end(ax, st[-1], closed[-1], "closed form f(t)", C["orange"], dy=-10)
    ax.set(xscale="log", xlabel="step", title=f"β2={b2}: uncorrected / corrected update, same state", xlim=(1, TOTAL * 6))
    CF_DEV[b2] = float(np.nanmax(np.abs(np.nanmedian(ratio, 1) / closed - 1)))
axs[0].set_ylabel("update-norm ratio")
fig.tight_layout()
finish(fig, "bias_correction_counterfactual")
for b2 in (0.95, 0.999):
    print(f"β2={b2}: the median tensor's counterfactual ratio is within {CF_DEV[b2]:.1%} of the closed form at every logged step")
observe("counterfactual_matches_closed_form", max(CF_DEV.values()) < 0.05,
        "in the real model the uncorrected/corrected update ratio follows §2's closed form (median tensor within 5%)")

if MODE != "learn":
    ANS["bias_correction"] = _clean(dict(
        closed_form_stop={str(b2): {f"{p:.0%}": n for p, n in v.items()} for b2, v in BC_STOP.items()},
        first_step=float(f_closed[0]), peak=float(f_closed.max()), peak_t=int(t20[f_closed.argmax()]), at20=float(f_closed[-1]),
        excess_displacement_0999=EXCESS_0999,
        model_stop={f"b2={b2},warmup={w}": s for (b2, w), s in MODEL_STOP.items()},
        val={f"b2={b2},warmup={w}": dict(on=v[0], off=v[1]) for (b2, w), v in VAL_TABLE.items()},
        counterfactual_max_dev=CF_DEV, seed_band=band))
    save_results()

# %% [markdown]
r"""
**What the measurements say.** The closed form from §2 predicts the model's behaviour, and two of
the expectations recorded above did not hold.

- **$\beta_2 = 0.999$.** Switching correction off is not an early-steps problem. The uncorrected
  multiplier is still about 2× at step 300, so the whole run trains at 2–6.6× its tuned learning rate.
  The loss never comes back inside the seed band. Warmup does not hide it: the multiplier peaks
  around step 12 and stays above 2 long after a 30-step warmup has ended.
- **$\beta_2 = 0.95$.** The correction acts only over the first ~20 steps, and there it *raises* the
  step. Uncorrected, the first step is $0.45\,\eta$.
  - Without warmup, the uncorrected run is the *better* one: its own small early steps work as a
    built-in warmup.
  - With warmup, the two runs are indistinguishable from step 16.

> **Carry forward:** how long bias correction matters is set by $1/(1-\beta_2)$, and in a short run
> that can be the whole run. With $\beta_2 = 0.999$ a missing correction quietly multiplies the
> learning rate for thousands of steps. With $\beta_2 = 0.95$ its only visible job is shaping the
> first ~20 steps, which warmup also does.
"""
