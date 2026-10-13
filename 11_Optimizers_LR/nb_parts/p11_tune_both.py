# %% [markdown]
r"""
## §8 Tune both sides before accepting a comparison

Every comparison in this notebook gave each side its own tuning, by the same rule. This section uses
the grids already measured to show what happens when that discipline slips: the challenger is
tuned, and the baseline is not. No new training is needed.

**How to read the plots.**
- **Left:** the baseline's *learning rate* is off. The x-axis is how far it sits from the baseline's
  own best, in octaves (factors of 2). At $x = 0$ the comparison is fair, and everywhere else the
  apparent advantage is partly a tuning artifact.
- **Right:** the baseline's *own knob* is off. This happened in this very notebook. Cosine's floor was
  first tried only at 0, 10 and 20%, and the best sat at the edge. Extending the grid by the same edge
  rule used for learning rates moved cosine's best floor all the way to 100% (no decay), and
  WSD's advantage shrank accordingly.
"""

# %%
def apparent(challenger_best_loss, baseline_grid, baseline_best_lr):
    xs, ys = [], []
    for lr, r in baseline_grid.items():
        L = final_loss(r)
        if math.isfinite(L):
            xs.append(math.log2(lr / baseline_best_lr))
            ys.append(L - challenger_best_loss)                         # > 0: the challenger looks better
    return np.array(xs), np.array(ys)


PAIRS = {   # learning-rate grids; both schedules at their default knob (cosine floor 10%, WSD decay 20%)
    "WSD vs cosine (default knobs)": (final_loss(wsd_grid[LR_WSD]), cos_grid, LR_BASE, C["orange"]),
    "Muon vs AdamW": (final_loss(muon_grid[LR_MUON]), cos_grid, LR_BASE, C["magenta"]),
    "cosine vs WSD (default knobs)": (final_loss(cos_grid[LR_BASE]), wsd_grid, LR_WSD, C["blue"]),
}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 3.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
TB = {}
for lab, (best, grid, blr, col) in PAIRS.items():
    xs, ys = apparent(best, grid, blr)
    ax.plot(xs, ys, "-o", ms=4, color=col); label_end(ax, xs[-1], ys[-1], lab, col)
    TB[lab] = dict(fair=float(ys[xs == 0][0]) if (xs == 0).any() else float("nan"),
                   worst=float(ys.max()), worst_at_oct=float(xs[ys.argmax()]))
ax.axvline(0, color=MUTED, lw=0.8); ax.axhline(0, color=MUTED, lw=0.8)
ax.set(xlabel="baseline learning rate, octaves from the baseline's own best", ylabel="apparent advantage (loss)",
       xlim=(ax.get_xlim()[0], ax.get_xlim()[1] + 2.6), title="Mis-tuning the baseline's learning rate (seed 0)")
# The baseline's own knob matters as much: tuned WSD against cosine at every floor that was tried.
wsd_best = var_loss[WSD_KEY]
fls = sorted(cos_floor)
adv = [cos_floor[fl] - wsd_best for fl in fls]
bx.plot(fls, adv, "-o", ms=4, color=C["orange"])
best_fl = min(cos_floor, key=cos_floor.get)
bx.plot(best_fl, cos_floor[best_fl] - wsd_best, "o", ms=8, mfc="none", color=INK)
label_end(bx, best_fl, cos_floor[best_fl] - wsd_best, "tuned floor", INK, dy=-10)
bx.plot(0.1, cos_floor[0.1] - wsd_best, "o", ms=8, mfc="none", color=C["yellow"])
label_end(bx, 0.1, cos_floor[0.1] - wsd_best, "the usual 10% default", C["yellow"], dy=10)
bx.axhline(0, color=MUTED, lw=0.8)
bx.set(xlabel="cosine floor (fraction of peak lr)", ylabel="tuned WSD's apparent advantage", xlim=(-0.03, max(fls) + 0.3),
       title=f"Mis-tuning the baseline's own knob, at step {TOTAL}")
fig.tight_layout()
finish(fig, "tune_both_sides")
TB["WSD (tuned) vs cosine floor"] = dict(fair=float(cos_floor[best_fl] - wsd_best), worst=float(max(adv)),
                                         at_default_floor=float(cos_floor[0.1] - wsd_best), tuned_floor=best_fl)
for lab, v in TB.items():
    extra = f"    at cosine's 10% default floor: {v['at_default_floor']:+.4f}" if "at_default_floor" in v else \
        f"    worst mis-tuned baseline ({v['worst_at_oct']:+.0f} octaves): {v['worst']:+.4f}"
    print(f"{lab:<30} fair comparison: {v['fair']:+.4f}{extra}")
print(f"the step-{STOP} keep decision itself: with cosine's floor at its default the WSD margin would have been larger; "
      f"tuning the floor shrank it to {ANS.get('schedules', {}).get('diff', float('nan')):+.4f} (see §6)")

print("\ntuning budget per side, and what was held fixed:")
PARITY = [
    ("cosine vs WSD", f"{BUDGET['cosine']} vs {BUDGET['WSD']} trials", "lr grid + own knob, edge-extended (floor / decay length)", "warmup, β, weight decay, batch, data, seeds"),
    ("Muon vs AdamW", f"{len(muon_grid)} vs {len(cos_grid)} lr points, {len(SEEDS)} seeds each", "Muon's lr; AdamW's lr", "momentum 0.95 / β (0.9, 0.95), decay 0.1, schedule"),
    ("SP vs muP (width)", f"{sum(len(SP[w]['points']) for w in WIDTHS)} vs {sum(len(MUP[w]['points']) for w in WIDTHS)} runs", "lr grid per width", "depth, batch, steps, schedule"),
]
for row in PARITY:
    print(f"  {row[0]:<18} {row[1]:<34} tuned: {row[2]:<48} fixed: {row[3]}")
print(f"\nwidth {HOLD}: carrying width {WIDTHS[0]}'s best lr over unchanged costs {final_loss(sp_naive) - hy:+.4f} under SP "
      f"and {final_loss(mup_naive) - hy:+.4f} under muP, relative to tuning SP at width {HOLD}")
if MODE != "learn":
    ANS["tune_both"] = _clean(dict(pairs=TB, parity=PARITY, transfer_cost=dict(sp=final_loss(sp_naive) - hy, mup=final_loss(mup_naive) - hy)))
    save_results()

# %% [markdown]
r"""
> **Carry forward:** before believing "X beats Y", ask how many trials each side got and where each
> side's learning rate sat relative to its own best. A baseline even one octave off its best can
> produce an advantage as large as most published optimizer gains.
"""
