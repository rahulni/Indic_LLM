# %% [markdown]
r"""
## §6 Cosine vs WSD: planned for 300 steps, stopped at 200

**The situation.** Two runs of the same model are planned for 300 steps, one with a cosine schedule
and one with WSD. Both are stopped at step 200, because the budget changed, the cluster was
needed, or a better idea came along. Which of the two models do you keep?

**Intuition.**
- At step 200 the cosine run is about halfway down its decay. Its learning rate is already low, so
  its weights are relatively settled, and its loss is probably the lower of the two.
- The WSD run is still at its peak learning rate. Its loss is higher because the weights are still
  bouncing, but that state has a property cosine lacks: it is not committed to any length.
- Decaying a WSD checkpoint for a short while turns it into a finished model.
- Not decaying it lets training continue as if nothing happened.
- Cosine cannot do either. Its run length was fixed before the first step.

**Doing the comparison properly.**
1. **Tune both sides by the same rule.** Each schedule gets its own learning-rate grid (same points,
   same edge rule), then its own knob: the floor for cosine, the decay length for WSD (plus one 1−√
   shape trial). Each knob grid is extended by the same edge rule, so neither side's best setting
   is left sitting at the edge of what was tried.
2. **Three paired seeds** per arm. The same seed means the same initial weights and the same
   batches, so arms are compared by paired differences.
3. **Compute-matched alternatives.** Finished models that cost ≤ 200 steps are compared with each
   other, and those costing ≤ 240 with each other:
   - cosine planned for 200
   - a 40-step cooldown from WSD's step-160 checkpoint
   - an average of WSD's weights during its flat phase ("constant learning rate + weight EMA")
   - cosine planned for 240, against WSD's step-200 checkpoint plus a 40-step cooldown
4. **The keep decision follows the rule written in §0**, not the plot.
"""

# %%
wsd_grid = lr_sweep(lambda lr: base_cfg(lr, kind="wsd", decay_frac=0.2), GRID_BASE, final_loss, max_ext=3)
LR_WSD = min((lr for lr in wsd_grid if math.isfinite(final_loss(wsd_grid[lr]))), key=lambda lr: final_loss(wsd_grid[lr]))
check(LR_WSD not in (min(wsd_grid), max(wsd_grid)), f"WSD's best peak learning rate ({LR_WSD:.0e}) is bracketed by its grid")

# Each schedule's own knob gets the same edge rule as the learning rate: if the best value sits at the edge of
# what was tried, try the next value out, until it is bracketed or the knob's feasible range ends.
def cos_cfg(fl):
    return base_cfg(LR_BASE) if fl == 0.1 else base_cfg(LR_BASE, floor=fl)       # floor 10% is the default


def wsd_cfg(df, shape="linear"):
    return base_cfg(LR_WSD, kind="wsd", decay_frac=df, **({} if shape == "linear" else dict(shape=shape)))


def knob_sweep(make, values, step_out, lo_lim, hi_lim, max_ext=8):
    """Extend at an edge until the best value is bracketed or the knob's feasible range ends."""
    res = {v: final_loss(run(make(v))) for v in values}
    for _ in range(max_ext):
        best = min(res, key=res.get)
        if best == max(res) and step_out(best, +1) <= hi_lim + 1e-9:
            new = step_out(best, +1)
        elif best == min(res) and step_out(best, -1) >= lo_lim - 1e-9:
            new = step_out(best, -1)
        else:
            break
        res[new] = final_loss(run(make(new)))
    return dict(sorted(res.items()))


# Feasible ranges: a cosine floor of 1.0 means no decay at all; WSD's decay length runs from 5% to 60% of the run.
LIMITS = {"cosine floor": (0.0, 1.0), "WSD decay length": (0.05, 0.6)}
cos_floor = knob_sweep(cos_cfg, [0.0, 0.1, 0.2], lambda v, d: round(v + 0.1 * d, 2), *LIMITS["cosine floor"])
wsd_decay = knob_sweep(wsd_cfg, [0.2, 0.3], lambda v, d: round(v * 2, 3) if d > 0 else round(v / 2, 3), *LIMITS["WSD decay length"])
for lab, sw in (("cosine floor", cos_floor), ("WSD decay length", wsd_decay)):
    b = min(sw, key=sw.get)
    at_limit = b in LIMITS[lab]
    observe(f"knob_bracketed_{lab.split()[0].lower()}", b not in (min(sw), max(sw)) or at_limit,
            f"the {lab} is bracketed: its best value ({b:g}) is inside the values tried ({min(sw):g}–{max(sw):g})"
            + (" or at the end of its feasible range" if at_limit else ""))
cos_variants = {f"cosine, floor {fl:.0%}": cos_cfg(fl) for fl in cos_floor}
wsd_variants = {f"WSD, last {df:.0%} linear": wsd_cfg(df) for df in wsd_decay}
wsd_variants["WSD, last 20% 1−√"] = wsd_cfg(0.2, "sqrt")
var_loss = {k: final_loss(run(c)) for k, c in {**cos_variants, **wsd_variants}.items()}
COS_KEY = min(cos_variants, key=var_loss.get)
WSD_KEY = min(wsd_variants, key=var_loss.get)
print(f"{'':<24}{'val@' + str(TOTAL):>9}")
for k, v in var_loss.items():
    print(f"{k:<24}{v:9.4f}{'  ← chosen' if k in (COS_KEY, WSD_KEY) else ''}")
BUDGET = {"cosine": len(cos_grid) + len(cos_variants) - 1, "WSD": len(wsd_grid) + len(wsd_variants) - 1}
print(f"tuning trials: cosine {BUDGET['cosine']}, WSD {BUDGET['WSD']} (learning-rate grid plus its own knob, "
      f"each extended at an edge by the same rule)")
print(f"peak learning rates: cosine {LR_BASE:.0e}, WSD {LR_WSD:.0e}")

# %%
COS_CFG, WSD_CFG = cos_variants[COS_KEY], wsd_variants[WSD_KEY]
EMA_DECAYS = [0.9, 0.95, 0.98]


def wsd_arms(seed):
    cfg = dict(WSD_CFG, seed=seed, ckpt_at=[BRANCH, STOP], ema=EMA_DECAYS)
    main, ck = train(cfg)
    shape = WSD_CFG["sched"].get("shape", "linear")
    out = dict(main=main)
    for name, start, total in (("cool_from_branch", BRANCH, STOP), ("cool_from_stop", STOP, STOP + COOL)):
        if start not in ck:                                             # the main run diverged before this step
            continue
        bcfg = dict(WSD_CFG, seed=seed, steps=total, resume=ck[start], eval_at=[total], sample_at=[total], log_every=0,
                    sched=dict(kind="wsd", warmup=WARM, decay_frac=COOL / total, shape=shape))
        out[name] = train(bcfg)[0]
    return out


arms = {}
for s in SEEDS:
    w = experiment(f"wsd_arms|{cfg_name(dict(WSD_CFG, seed=s))}", lambda s=s: wsd_arms(s))
    arms[s] = dict(cos300=run(dict(COS_CFG, seed=s)),
                   cos200=run(dict(COS_CFG, seed=s, steps=STOP)),
                   cos240=run(dict(COS_CFG, seed=s, steps=STOP + COOL)),
                   wsd300=w["main"], wsd_cool160=w.get("cool_from_branch"), wsd_cool200=w.get("cool_from_stop"))


def arm_loss(arm, step):
    vals = []
    for s in SEEDS:
        r = arms[s][arm]
        vals.append(final_loss(r, step) if r else float("inf"))
    return np.array(vals)


best_ema = min(EMA_DECAYS, key=lambda d: np.mean([arms[s]["wsd300"]["ema_evals"].get(str(d), {}).get(str(STOP), np.inf) for s in SEEDS]))
ema200 = np.array([arms[s]["wsd300"]["ema_evals"][str(best_ema)][str(STOP)] for s in SEEDS])
TABLE = {  # name: (compute in steps, per-seed losses)
    f"cosine-{TOTAL}, stopped at {STOP}": (STOP, arm_loss("cos300", STOP)),
    f"WSD-{TOTAL}, stopped at {STOP}": (STOP, arm_loss("wsd300", STOP)),
    f"cosine planned for {STOP}": (STOP, arm_loss("cos200", STOP)),
    f"WSD checkpoint {BRANCH} + {COOL}-step cooldown": (STOP, arm_loss("wsd_cool160", STOP)),
    f"WSD flat phase, weight EMA {best_ema} (constant lr + EMA)": (STOP, ema200),
    f"cosine-{TOTAL}, continued to {STOP + COOL}": (STOP + COOL, arm_loss("cos300", STOP + COOL)),
    f"WSD checkpoint {STOP} + {COOL}-step cooldown": (STOP + COOL, arm_loss("wsd_cool200", STOP + COOL)),
    f"cosine planned for {STOP + COOL}": (STOP + COOL, arm_loss("cos240", STOP + COOL)),
    f"cosine-{TOTAL}, completed": (TOTAL, arm_loss("cos300", TOTAL)),
    f"WSD-{TOTAL}, completed": (TOTAL, arm_loss("wsd300", TOTAL)),
}
ref = TABLE[f"cosine-{TOTAL}, stopped at {STOP}"][1]
print(f"{'':<52}{'compute':>8}{'val loss':>10}{'± s.d.':>8}{'vs cosine@' + str(STOP):>14}{'paired SE':>10}")
for k, (c, v) in TABLE.items():
    d = v - ref
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    print(f"{k:<52}{c:>8}{v.mean():10.4f}{v.std(ddof=1):8.4f}{d.mean():+14.4f}{se:10.4f}")

# %% [markdown]
r"""
**The state each model is in at step 200.** A loss is one number, so here is the rest of the picture:
- where each learning rate sits
- how hard the gradients are pushing, and whether clipping is firing
- how far each step moves the weights
- what each model writes
"""

# %%
def state_at(res, step, peak):
    h = res["hist"]
    i = step - 1
    rat = res["ratios"]
    j = rat["steps"].index(step) if step in rat["steps"] else None
    mats = [k for k, n in enumerate(res["tensors"]) if n == "emb" or n.split(".")[-1] in ("q", "kv", "o", "up", "down")]
    return dict(val=final_loss(res, step), train=float(np.mean(h["loss"][step - 10:step])),
                lr=h["lr"][i] * peak, gnorm=float(np.mean(h["gnorm"][i - 9:i + 1])),
                clipped=float(np.mean(h["clipped"][max(0, i - 49):i + 1])),
                ratio=float(np.median(np.array(rat["total"][j])[mats])) if j is not None else float("nan"))


STATE = {}
for arm, lab, peak in (("cos300", f"cosine-{TOTAL} at step {STOP}", LR_BASE), ("wsd300", f"WSD-{TOTAL} at step {STOP}", LR_WSD)):
    rows = [state_at(arms[s][arm], STOP, peak) for s in SEEDS]
    STATE[lab] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
print(f"{'':<26}{'val loss':>9}{'train loss':>11}{'lr now':>9}{'grad norm':>10}{'clipped':>9}{'update/weight':>14}")
for lab, st in STATE.items():
    print(f"{lab:<26}{st['val']:9.4f}{st['train']:11.4f}{st['lr']:9.1e}{st['gnorm']:10.3f}{st['clipped']:9.0%}{st['ratio']:14.2e}")
for arm, lab in (("cos300", f"cosine at {STOP}"), ("wsd300", f"WSD at {STOP}"), ("wsd_cool160", f"WSD {BRANCH} + cooldown")):
    smp = (arms[SEEDS[0]][arm] or {}).get("samples", {}).get(str(STOP))
    print(f"\n{lab}:  {smp}")

# %%
# The keep decision, by the rule fixed in §0
def paired(a, b):
    """Mean and standard error over seeds of the paired difference a - b."""
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def batch_losses(arm, step):
    """Per-eval-batch losses averaged over seeds, for an eval-sampling bootstrap."""
    return np.mean([np.array(arms[s][arm]["evals"][str(step)]["batches"]) for s in SEEDS], axis=0)


opt_cos = {f"cosine-{TOTAL} at {STOP}": ("cos300", STOP), f"cosine-{TOTAL} continued to {STOP + COOL}": ("cos300", STOP + COOL)}
opt_wsd = {f"WSD at {STOP}": ("wsd300", STOP), f"WSD {STOP} + {COOL}-step cooldown": ("wsd_cool200", STOP + COOL)}
best_c = min(opt_cos, key=lambda k: arm_loss(*opt_cos[k]).mean())
best_w = min(opt_wsd, key=lambda k: arm_loss(*opt_wsd[k]).mean())
diff, se = paired(arm_loss(*opt_wsd[best_w]), arm_loss(*opt_cos[best_c]))
bd = batch_losses(*opt_wsd[best_w]) - batch_losses(*opt_cos[best_c])
rng_b = np.random.default_rng(0)
se_eval = float(np.std([rng_b.choice(bd, len(bd)).mean() for _ in range(2000)]))
tie = abs(diff) <= RULES["keep_at_200"]["tie_se"] * se
KEEP = "WSD" if (tie or diff < 0) else "cosine"
print(f"best option within +20% compute: cosine → {best_c} ({arm_loss(*opt_cos[best_c]).mean():.4f});  "
      f"WSD → {best_w} ({arm_loss(*opt_wsd[best_w]).mean():.4f})")
print(f"paired difference WSD − cosine: {diff:+.4f} ± {se:.4f} (SE over seeds; eval-sampling SE {se_eval:.4f})"
      f"  →  {'within 2 SE, a tie, so ' if tie else ''}keep the {KEEP} run")
raw, raw_se = paired(arm_loss("wsd300", STOP), arm_loss("cos300", STOP))
print(f"as stopped, at step {STOP}: WSD − cosine = {raw:+.4f} ± {raw_se:.4f}")
observe("cosine_lower_at_stop", raw > 0, f"as stopped at step {STOP}, the cosine model has the lower loss")   # raw = WSD − cosine
d1, s1 = paired(arm_loss("wsd_cool200", STOP + COOL), arm_loss("cos300", STOP + COOL))
observe("wsd_cooldown_beats_cosine", d1 < 0,
        f"a {COOL}-step cooldown from WSD's step-{STOP} checkpoint vs cosine-{TOTAL} at the same compute: {d1:+.4f} ± {s1:.4f}")
d2, s2 = paired(arm_loss("wsd_cool200", STOP + COOL), arm_loss("cos240", STOP + COOL))
observe("wsd_cooldown_vs_cos240", d2 <= 2 * s2,
        f"…and vs a cosine run planned for {STOP + COOL} from the start: {d2:+.4f} ± {s2:.4f} (within 2 SE or better)")
d3, s3 = paired(ema200, arm_loss("cos200", STOP))
observe("ema_matches_cosine200", d3 <= 2 * s3,
        f"averaging WSD's flat-phase weights (EMA {best_ema}) vs a cosine run planned for {STOP}: {d3:+.4f} ± {s3:.4f} (within 2 SE or better)")

# How the verdict moved as cosine's own knob was tuned. Every floor below has three seeds already measured.
VERDICT_BY_FLOOR = {}
for fl in sorted(cos_floor):
    runs = {s: R.get(cfg_name(dict(cos_cfg(fl), seed=s))) for s in SEEDS}
    if not all(runs.values()):
        continue
    c200 = np.array([final_loss(runs[s], STOP) for s in SEEDS])
    c240 = np.array([final_loss(runs[s], STOP + COOL) for s in SEEDS])
    opt = c240 if c240.mean() < c200.mean() else c200
    dd, ee = paired(arm_loss(*opt_wsd[best_w]), opt)
    VERDICT_BY_FLOOR[fl] = dict(diff=dd, se=ee, keep="WSD" if (dd < 0 or abs(dd) <= RULES["keep_at_200"]["tie_se"] * ee) else "cosine",
                                tie=abs(dd) <= RULES["keep_at_200"]["tie_se"] * ee)
print(f"\nthe keep decision as cosine's floor is tuned (WSD's best option − cosine's best option, 3 paired seeds):")
for fl, v in VERDICT_BY_FLOOR.items():
    print(f"  cosine floor {fl:>4.0%}: {v['diff']:+.4f} ± {v['se']:.4f}  →  keep {v['keep']}{' (tie)' if v['tie'] else ''}")

# %%
fig, axs = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
ax = axs[0]
for arm, col, lab in (("cos300", C["blue"], "cosine"), ("wsd300", C["orange"], "WSD")):
    L = np.mean([ema_curve(arms[s][arm]["hist"]["loss"], 20) for s in SEEDS], axis=0)
    ax.plot(np.arange(1, len(L) + 1), L, color=col); label_end(ax, len(L), L[-1], lab, col)
for arm, col in (("wsd_cool160", C["yellow"]), ("wsd_cool200", C["aqua"])):
    if all(arms[s][arm] for s in SEEDS):
        L = np.mean([ema_curve(arms[s][arm]["hist"]["loss"], 20) for s in SEEDS], axis=0)
        x0 = BRANCH if arm == "wsd_cool160" else STOP
        ax.plot(np.arange(x0 + 1, x0 + len(L) + 1), L, color=col)
        label_end(ax, x0 + len(L), L[-1], f"cooldown from {x0}", col, dy=-8)
ax.axvline(STOP, color=MUTED, lw=0.8, ls="--")
lo = min(np.min(arm_loss("cos300", TOTAL)), np.min(arm_loss("wsd300", TOTAL)))
ax.set(ylim=(lo - 0.15, lo + 1.2), ylabel="training loss (20-step EMA)", xlim=(0, TOTAL * 1.3),
       title=f"Same model, two schedules: stopped at {STOP} (dashed), branches decayed from WSD checkpoints")
ax = axs[1]
for res, col, peak in ((arms[SEEDS[0]]["cos300"], C["blue"], LR_BASE), (arms[SEEDS[0]]["wsd300"], C["orange"], LR_WSD)):
    ax.plot(np.arange(1, len(res["hist"]["lr"]) + 1), np.array(res["hist"]["lr"]) * peak, color=col)
ax.axvline(STOP, color=MUTED, lw=0.8, ls="--")
ax.set(xlabel="step", ylabel="learning rate")
fig.tight_layout()
finish(fig, "cosine_vs_wsd")

fig, ax = plt.subplots(figsize=(8.6, 3.6))
keys = list(TABLE)
for i, k in enumerate(keys):
    c, v = TABLE[k]
    col = C["blue"] if "cosine" in k else (C["violet"] if "EMA" in k else C["orange"])
    ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), fmt="o", color=col, ms=5, capsize=2)
    ax.plot(v, [i] * len(v), ".", color=col, alpha=0.5)
ax.set_yticks(range(len(keys)), [f"{k}  [{TABLE[k][0]}]" for k in keys], fontsize=8)
ax.invert_yaxis()
ax.set(xlabel="validation loss (mean ± s.d., dots = seeds)", title="Every finished model, with its compute in steps")
finish(fig, "cosine_vs_wsd_arms")

if MODE != "learn":
    ANS["schedules"] = _clean(dict(
        lr_cos=LR_BASE, lr_wsd=LR_WSD, cos_variant=COS_KEY, wsd_variant=WSD_KEY, budget=BUDGET, variants=var_loss,
        table={k: dict(compute=c, mean=float(v.mean()), sd=float(v.std(ddof=1)), seeds=v.tolist()) for k, (c, v) in TABLE.items()},
        state=STATE, raw_diff=raw, raw_se=raw_se, cooldown_vs_cos300=[d1, s1], cooldown_vs_cos240=[d2, s2], ema_vs_cos200=[d3, s3],
        best_cos=best_c, best_wsd=best_w, diff=diff, se=se, se_eval=se_eval, tie=bool(tie), keep=KEEP,
        best_ema=best_ema, by_floor={f"{k:g}": v for k, v in VERDICT_BY_FLOOR.items()}, cos_floor_grid={f"{k:g}": v for k, v in cos_floor.items()},
        samples={a: (arms[SEEDS[0]][a] or {}).get("samples", {}).get(str(STOP)) for a in ("cos300", "wsd300", "wsd_cool160")}))
    save_results()

# %% [markdown]
r"""
**What the measurements say.** Keep the WSD run.

- **Tuned on its own knob, cosine stopped decaying.** Every step up in cosine's floor improved the
  loss at 300, from 0% to 100%. Its best floor is 100%, which means no decay at all. So in this regime
  the tuned "cosine" run is a constant learning rate, the same run as WSD's flat phase. At step 200
  the two models are therefore *identical*, as the state table shows, and the as-stopped comparison
  is an exact tie.
- **What separates them is WSD's ability to decay late.** A 40-step cooldown from WSD's step-200
  checkpoint beats simply continuing for the same compute, by a small but clear margin over 3 paired
  seeds. At the planned 300 steps WSD's final 10% decay wins as well.
- **Averaging WSD's flat-phase weights (EMA)** gives the best model of all at 200 steps of compute.
- **Every decay that starts early hurts here** (0.16 tokens per parameter, still far from
  convergence). Cosine's floor wanted to go to 100%, and WSD's best decay was only the last 10%.
  Longer runs are where earlier decay is expected to pay off, and they are not tested here.
- **Procedure, measured.** The first pass tried cosine floors only up to 20%, and the best sat at that
  edge. Extending the floor grid by the same edge rule used for learning rates, until it hit the
  floor's physical limit, shrank WSD's margin roughly threefold. The verdict never flipped (table
  above). This is the "tune both sides" principle at work inside this notebook.

> **Carry forward:** compare schedules by what you can *do* with the model when the run is cut
> short, compute-matched and tuned on both sides. Here, tuning the baseline properly turned "cosine
> vs WSD" into "constant vs decay-at-the-end". WSD's real advantage is the option to decay late, from
> any checkpoint.
"""
