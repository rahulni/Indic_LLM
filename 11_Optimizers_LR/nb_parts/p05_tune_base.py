# %% [markdown]
r"""
### Tuning the base model's learning rate

Sections §4–§6 and §M need one well-chosen peak learning rate for the base model. It is chosen the
same way every comparison in this notebook is tuned:
- a factor-2 grid, scored by validation loss at the final step
- extended at an edge whenever the best point sits on that edge, so the minimum is bracketed
- two more seeds at the winner, which measure how much the loss moves from seed to seed
- one repeat of seed 0, which measures how much the GPU's non-deterministic kernels move it on
  their own

That repeat is the **noise floor**: a smaller difference than this cannot mean anything.
"""

# %%
GRID_BASE = qk([5e-4, 1e-3, 2e-3, 4e-3, 8e-3], [1e-3, 4e-3, 1.6e-2])
SEEDS = qk([0, 1, 2], [0, 1])

cos_grid = lr_sweep(lambda lr: base_cfg(lr), GRID_BASE, final_loss, max_ext=3)
finite = {lr: final_loss(r) for lr, r in cos_grid.items() if math.isfinite(final_loss(r))}
LR_BASE = min(finite, key=finite.get)
cos_seeds = {s: run(base_cfg(LR_BASE, seed=s)) for s in SEEDS}
cos_repeat = run(base_cfg(LR_BASE, seed=0), tag="repeat")

print(f"{'peak lr':>9} {'val@' + str(STOP):>9} {'val@' + str(TOTAL):>9}")
for lr, r in cos_grid.items():
    note = f"diverged at step {r['diverged_at']}" if r["diverged_at"] else ""
    print(f"{lr:9.2e} {final_loss(r, STOP):9.4f} {final_loss(r, TOTAL):9.4f}  {note}{'  ← best' if lr == LR_BASE else ''}")
check(LR_BASE not in (min(cos_grid), max(cos_grid)), f"the best peak learning rate ({LR_BASE:.0e}) is bracketed by the grid")

loss0 = cos_seeds[0]["hist"]["loss"][0]
check(abs(loss0 - math.log(V_BASE)) < 0.3, f"untrained loss {loss0:.2f} ≈ ln(49,152) = {math.log(V_BASE):.2f}: the initial predictions are close to uniform")

smooth = np.array([ema_curve(cos_seeds[s]["hist"]["loss"], 20) for s in SEEDS])
SEED_SD = float(np.median(smooth[:, WARM * 2:].std(0, ddof=1)))
vals = np.array([final_loss(cos_seeds[s]) for s in SEEDS])
nondet_val = abs(final_loss(cos_repeat) - final_loss(cos_seeds[0]))
nondet_train = float(np.max(np.abs(ema_curve(cos_repeat["hist"]["loss"], 20) - smooth[0])))
print(f"\nseed-to-seed: validation loss {vals.mean():.4f} ± {vals.std(ddof=1):.4f} (s.d., {len(SEEDS)} seeds); "
      f"smoothed training loss s.d. {SEED_SD:.4f} per step (median)")
print(f"same seed, run twice: validation loss differs by {nondet_val:.5f}; smoothed training loss by at most {nondet_train:.4f}")
observe("nondeterminism_below_seed_noise", nondet_val < vals.std(ddof=1),
        "re-running the same seed moves the validation loss less than changing the seed does")
if MODE != "learn":
    ANS["base"] = _clean(dict(lr=LR_BASE, grid={f"{lr:.3g}": final_loss(r) for lr, r in cos_grid.items()},
                              val_mean=vals.mean(), val_sd=vals.std(ddof=1), seed_sd_train=SEED_SD,
                              nondet_val=nondet_val, nondet_train=nondet_train, loss0=loss0, n_params=N_BASE))
    save_results()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
lrs = list(cos_grid)
for step, col in ((STOP, C["yellow"]), (TOTAL, C["blue"])):
    a1.plot(lrs, [final_loss(cos_grid[lr], step) for lr in lrs], "-o", ms=4, color=col)
    label_end(a1, lrs[-1], final_loss(cos_grid[lrs[-1]], step), f"step {step}", col)
a1.plot([LR_BASE] * len(SEEDS), vals, "o", ms=4, mfc="none", color=C["blue"])
a1.set(xscale="log", xlabel="peak learning rate", ylabel="validation loss",
       title=f"Base model, cosine: best peak lr {LR_BASE:.0e}")
steps = np.arange(1, len(smooth[0]) + 1)
for i, s in enumerate(SEEDS):
    a2.plot(steps, smooth[i], color=SERIES[i], lw=1.1)
a2.plot(steps, ema_curve(cos_repeat["hist"]["loss"], 20), color=MUTED, lw=0.8, ls="--")
a2.set(ylim=(smooth[:, -1].min() - 0.3, smooth[:, -1].min() + 2.5), xlabel="step", ylabel="training loss (20-step EMA)",
       title=f"{len(SEEDS)} seeds at the best lr, and a same-seed repeat (dashed)")
fig.tight_layout()
finish(fig, "base_lr_tuning")
print("\nsample from the tuned model at step", TOTAL, ":\n ", cos_seeds[0]["samples"].get(str(TOTAL), "(not stored)"))

# %% [markdown]
r"""
> **Carry forward:** the base model's peak learning rate is tuned and bracketed, and the seed noise
> and the GPU noise floor are measured. Later sections compare against these numbers instead of
> eyeballing curves.
"""
