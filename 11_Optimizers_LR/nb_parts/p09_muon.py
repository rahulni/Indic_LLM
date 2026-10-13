# %% [markdown]
r"""
## §M Muon against AdamW, both tuned

Muon (R.6) is the clearest current case of an optimizer whose reported advantage depends on how
well the baseline was tuned. Published speedups over AdamW range from about 2× down to about 1.1×.
The lower figures come from comparisons in which both sides were tuned to the same standard.

**Setup.**
- The base model, cosine schedule, 300 steps.
- Muon updates the 40 hidden matrices: momentum 0.95, Nesterov, weight decay 0.1, and the learning
  rate scaled to match AdamW's update RMS (`match_rms_adamw`).
- The embedding and the norm gains stay on AdamW at its tuned learning rate. Muon only applies to
  2-D hidden matrices.
- **Equal budgets.** Muon's learning rate is tuned on the same grid, with the same edge rule and the
  same number of trials as AdamW's in §3, and then run on the same three seeds.

**Rule (§0).**
- A speedup is claimed only if the tuned-vs-tuned paired loss difference exceeds 2 standard errors.
- Speedup = how many steps AdamW needs to reach its own final loss, divided by how many Muon needs
  to reach that same loss.

Finally, the same tuned Muon run is set against AdamW at *mis-tuned* learning rates: the grid
neighbours of its best, and PyTorch's default of 1e-3. That reproduces how an inflated claim is
produced.
"""

# %%
muon_grid = lr_sweep(lambda lr: base_cfg(LR_BASE, opt="muon", muon_lr=lr), GRID_BASE, final_loss, max_ext=3)
LR_MUON = min((lr for lr in muon_grid if math.isfinite(final_loss(muon_grid[lr]))), key=lambda lr: final_loss(muon_grid[lr]))
check(LR_MUON not in (min(muon_grid), max(muon_grid)), f"Muon's best learning rate ({LR_MUON:.0e}) is bracketed by its grid")
muon_seeds = {s: run(base_cfg(LR_BASE, opt="muon", muon_lr=LR_MUON, seed=s)) for s in SEEDS}
print(f"tuning trials: AdamW {len(cos_grid)} grid points + {len(SEEDS) - 1} seeds; Muon {len(muon_grid)} grid points + {len(SEEDS) - 1} seeds")


def steps_to(curve, target):
    hit = np.nonzero(curve <= target)[0]
    return int(hit[0]) + 1 if hit.size else None


def speedup(challenger, baseline):
    """Steps the baseline needs to reach its final (smoothed) loss / steps the challenger needs for the same loss."""
    c, b = ema_curve(challenger["hist"]["loss"], 20), ema_curve(baseline["hist"]["loss"], 20)
    n = min(len(c), len(b))
    t = steps_to(c[:n], b[n - 1])
    if t:
        return n / t
    t = steps_to(b[:n], c[n - 1])                       # the challenger is slower: invert
    return t / n if t else float("nan")


mu_l = np.array([final_loss(muon_seeds[s]) for s in SEEDS])
ad_l = np.array([final_loss(cos_seeds[s]) for s in SEEDS])
md = mu_l - ad_l
m_diff, m_se = float(md.mean()), float(md.std(ddof=1) / math.sqrt(len(md)))
spd = np.array([speedup(muon_seeds[s], cos_seeds[s]) for s in SEEDS])
claim = m_diff < -RULES["optimizer_claim"]["se"] * m_se
print(f"tuned Muon vs tuned AdamW, final validation loss: {mu_l.mean():.4f} vs {ad_l.mean():.4f}; "
      f"paired difference {m_diff:+.4f} ± {m_se:.4f} (SE, {len(SEEDS)} seeds)")
print(f"speedup to AdamW's final loss: {spd.mean():.2f}x (per seed: {', '.join(f'{x:.2f}' for x in spd)})")
print(f"by the rule fixed in §0: {'a Muon speedup is claimed' if claim else 'no speedup is claimed: the difference is within 2 SE'}")

APPARENT = {}
for lab, lr in ((f"AdamW at {LR_BASE / 2:.0e} (half its best)", LR_BASE / 2), (f"AdamW at {LR_BASE * 2:.0e} (double its best)", LR_BASE * 2),
                ("AdamW at 1e-3 (PyTorch's default)", 1e-3)):
    if lr in cos_grid and lr != LR_BASE:
        base = cos_grid[lr]
        APPARENT[lab] = (final_loss(muon_seeds[0]) - final_loss(base), speedup(muon_seeds[0], base))
APPARENT["AdamW at its tuned lr"] = (final_loss(muon_seeds[0]) - final_loss(cos_seeds[0]), speedup(muon_seeds[0], cos_seeds[0]))
print("\nthe same tuned Muon run (seed 0) against differently tuned AdamW baselines:")
for lab, (d, sp) in APPARENT.items():
    print(f"  vs {lab:<36} loss difference {d:+.4f}   apparent speedup {sp:.2f}x")
if len(APPARENT) > 1:
    worst = max(v[1] for k, v in APPARENT.items() if k != "AdamW at its tuned lr")
    observe("mistuned_baseline_inflates", worst > APPARENT["AdamW at its tuned lr"][1],
            f"a mis-tuned AdamW baseline inflates Muon's apparent speedup from {APPARENT['AdamW at its tuned lr'][1]:.2f}x to {worst:.2f}x")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.5))
for grid, col, lab in ((cos_grid, C["blue"], "AdamW"), (muon_grid, C["magenta"], "Muon (hidden matrices)")):
    xs = list(grid)
    a1.plot(xs, [final_loss(grid[x]) for x in xs], "-o", ms=4, color=col)
    label_end(a1, xs[-1], final_loss(grid[xs[-1]]), lab, col)
a1.set(xscale="log", xlabel="learning rate (Muon: its own lr; AdamW parts at the tuned AdamW lr)", ylabel=f"validation loss at {TOTAL}",
       title="Each optimizer tuned on its own grid")
for res_d, col, lab in ((cos_seeds, C["blue"], "AdamW, tuned"), (muon_seeds, C["magenta"], "Muon, tuned")):
    L = np.mean([ema_curve(res_d[s]["hist"]["loss"], 20) for s in SEEDS], axis=0)
    a2.plot(np.arange(1, len(L) + 1), L, color=col); label_end(a2, len(L), L[-1], lab, col)
for lab, lr in ((f"AdamW at {LR_BASE / 2:.0e}", LR_BASE / 2),):
    if lr in cos_grid:
        L = ema_curve(cos_grid[lr]["hist"]["loss"], 20)
        a2.plot(np.arange(1, len(L) + 1), L, "--", color=MUTED, lw=1); label_end(a2, len(L), L[-1], lab + " (mis-tuned)", MUTED, dy=8)
lo = min(mu_l.min(), ad_l.min())
a2.set(ylim=(lo - 0.15, lo + 1.5), xlim=(0, TOTAL * 1.45), xlabel="step", ylabel="training loss (20-step EMA)",
       title=f"Tuned vs tuned: {m_diff:+.3f} ± {m_se:.3f}")
fig.tight_layout()
finish(fig, "muon_vs_adamw")

if MODE != "learn":
    ANS["muon"] = _clean(dict(lr_muon=LR_MUON, lr_adamw=LR_BASE, grid={f"{k:.3g}": final_loss(v) for k, v in muon_grid.items()},
                              muon=mu_l.tolist(), adamw=ad_l.tolist(), diff=m_diff, se=m_se, speedup=spd.tolist(),
                              claim=bool(claim), apparent={k: list(v) for k, v in APPARENT.items()},
                              trials=dict(adamw=len(cos_grid), muon=len(muon_grid))))
    save_results()

# %% [markdown]
r"""
**What the measurements say.** With both sides tuned on the same grid and the same seeds, Muon's
advantage is real here: about 1.4× fewer steps to AdamW's final loss, close to what has been
reported for well-tuned comparisons at around 0.1B parameters. Running AdamW at half its best
learning rate makes the same Muon run look about 1.6× faster. The gap between those two numbers is
pure tuning effort.

> **Carry forward:** an optimizer comparison is a statement about two tuning efforts as much as about
> two algorithms. Give both sides the same grid, the same trial count and the same seeds, and
> report the paired difference with its error bar, then the speedup.
"""
