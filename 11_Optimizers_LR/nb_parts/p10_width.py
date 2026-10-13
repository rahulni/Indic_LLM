# %% [markdown]
r"""
## §7 How the best learning rate moves with width, and what muP changes

**The question.** Sweeping the learning rate on a small model is cheap, and on a large one it is
not. Can the value found at width 256 be used at width 4,096?

**Intuition.** Under the standard parameterization (SP):
- Every hidden matrix is initialized at $1/\sqrt{\text{fan-in}}$.
- Adam moves each of its entries by about $\eta$.
- A wider matrix sums more of those entries into each output. Its output therefore changes more per
  step, roughly in proportion to width, unless $\eta$ shrinks to compensate.

So the best $\eta$ should fall roughly as $1/\text{width}$. muP rescales the per-layer learning rates
and the output multiplier so that this dependence cancels, and the best $\eta$ stays put.

**Protocol.**
- The width family from §3 (4 layers, head size 64) at widths 256, 512, 1,024.
- Everything except width held fixed: depth, the 16,384-token batch, 300 steps, warmup 30 + cosine.
- Coarse factor-2 grid, 1.25e-4 … 1.6e-2, one seed, extended at an edge if needed.
- Then refined with √2-spaced points around each width's best, and three seeds on the five points
  there.
- **Minimum (rule, §0):** the vertex of a parabola in $\log_2\eta$ through those five points, with a
  90% bootstrap interval over seeds.

**Extrapolation, then a test of it.**
- Fit $\log_2\eta^* = a + b\log_2(\text{width})$ and predict width 2,048.
- Then measure width 2,048 around the prediction. The prediction error at 2,048 calibrates the
  confidence in the 4,096 answer.
"""

# %%
WIDTHS, HOLD = qk([256, 512, 1024], [64, 128, 256]), qk(2048, 512)
CF = qk(2.0, 4.0)                                                    # coarse grid factor
COARSE = [qk(1.25e-4, 5e-4) * CF ** k for k in range(qk(8, 4))]
SWEEP_SEEDS = qk([0, 1, 2], [0, 1])


def width_cfg(width, lr, seed=0, mup=False):
    mup = mup and width != BASE_WIDTH_MUP                            # at the reference width muP *is* SP: reuse those runs
    return dict(family="width", width=width, mup=mup, seed=seed, steps=TOTAL, sched=dict(kind="cosine", warmup=WARM),
                opt=dict(kind="adamw", lr=lr, betas=(0.9, 0.95), wd=0.1), clip=1.0,
                eval_at=[qk(150, 30), TOTAL], log_every=qk(25, 10), diag=True)


def lat(lr):
    """Snap to the √(factor) lattice the grids live on, so equal learning rates get identical names."""
    k = round(2 * math.log(lr / COARSE[0], CF))
    return COARSE[0] * CF ** (k / 2)


def sweep_width(width, mup=False, refine=True):
    coarse = lr_sweep(lambda lr: width_cfg(width, lat(lr), mup=mup), COARSE, final_loss, factor=CF)
    pts = {lat(lr): r for lr, r in coarse.items()}
    fin = {lr: final_loss(r) for lr, r in pts.items() if math.isfinite(final_loss(r))}
    a = min(fin, key=fin.get)
    if not refine:
        return dict(points={lr: [final_loss(r)] for lr, r in pts.items()}, runs=pts, center=a)
    five = [lat(a * CF ** (k / 2)) for k in (-2, -1, 0, 1, 2)]
    losses = {}
    for lr in sorted(set(pts) | set(five)):
        seeds = SWEEP_SEEDS if lr in five else [0]
        losses[lr] = [final_loss(run(width_cfg(width, lr, seed=s, mup=mup))) for s in seeds]
    return dict(points=losses, runs={lr: run(width_cfg(width, lr, mup=mup)) for lr in losses}, center=a, five=five)


def vertex(lrs, losses_per_lr, rng=None):
    """Parabola vertex in log2(lr); with rng, resample seeds at every point (bootstrap)."""
    xs, ys = [], []
    for lr, ls in zip(lrs, losses_per_lr):
        ls = [l for l in ls if math.isfinite(l)]
        if not ls:
            continue
        xs.append(math.log2(lr))
        ys.append(np.mean(rng.choice(ls, len(ls))) if rng is not None else np.mean(ls))
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 3:                                                  # too few finite points for a parabola: use the grid argmin
        i = int(np.argmin(ys))
        return float(xs[i]), float(ys[i]), np.array([0.0, 0.0, ys[i]])
    c = np.polyfit(xs, ys, 2)
    x = -c[1] / (2 * c[0]) if c[0] > 0 else xs[np.argmin(ys)]
    x = float(np.clip(x, xs.min(), xs.max()))
    return x, float(np.polyval(c, x)), c


def analyse(sw, boot=RULES["lr_minimum"]["boot"]):
    five = sw.get("five") or sorted(sw["points"], key=lambda lr: abs(math.log(lr / sw["center"])))[:3]
    five = sorted(five)
    L = [sw["points"][lr] for lr in five]
    x, y, c = vertex(five, L)
    rng = np.random.default_rng(0)
    bx = np.array([vertex(five, L, rng)[0] for _ in range(boot)])
    lo, hi = np.percentile(bx, [5, 95])
    cost2 = float(max(np.polyval(c, x + 1), np.polyval(c, x - 1)) - y)          # loss cost of a 2x error either way
    return dict(lr=2 ** x, lo=2 ** lo, hi=2 ** hi, loss=y, boot=bx, cost2x=cost2, five=five)


SP = {w: sweep_width(w) for w in WIDTHS}
SPA = {w: analyse(SP[w]) for w in WIDTHS}
print(f"{'width':>6} {'params':>8} {'best lr (vertex)':>17} {'90% interval':>21} {'loss there':>11} {'cost of 2x error':>17}")
for w in WIDTHS:
    a = SPA[w]
    print(f"{w:>6} {WIDTH_PARAMS[w] / 1e6:7.1f}M {a['lr']:17.2e}   [{a['lo']:.2e}, {a['hi']:.2e}] {a['loss']:11.4f} {a['cost2x']:+17.4f}")
for w in WIDTHS:
    lrs = sorted(SP[w]["points"])
    check(SP[w]["center"] not in (lrs[0], lrs[-1]), f"width {w}: the best learning rate is bracketed by the grid")
observe("best_loss_falls_with_width", all(SPA[a]["loss"] > SPA[b]["loss"] for a, b in zip(WIDTHS, WIDTHS[1:])),
        "the best achievable loss falls as width grows (wider is better at its own best learning rate)")

# %%
X = np.log2(WIDTHS)
Y = np.log2([SPA[w]["lr"] for w in WIDTHS])
slope, icpt = np.polyfit(X, Y, 1)
B = np.stack([SPA[w]["boot"] for w in WIDTHS], 1)                   # bootstrap vertices, paired by index
bfit = np.array([np.polyfit(X, b, 1) for b in B])
pred = lambda width, fit=(slope, icpt): 2 ** (fit[0] * math.log2(width) + fit[1])
pred_hold = pred(HOLD)
pred_hold_ci = 2 ** np.percentile(bfit[:, 0] * math.log2(HOLD) + bfit[:, 1], [5, 95])
print(f"fit: log2(best lr) = {slope:.2f} · log2(width) + {icpt:.2f}   (slope 90% interval "
      f"[{np.percentile(bfit[:, 0], 5):.2f}, {np.percentile(bfit[:, 0], 95):.2f}]; the inverse-width rule says −1)")
print(f"prediction for width {HOLD}: {pred_hold:.2e}  (90% interval {pred_hold_ci[0]:.2e} … {pred_hold_ci[1]:.2e})")

# Held-out width: five √2-spaced points centred on the prediction, extended once at an edge if needed.
hold_pts = [lat(pred_hold * CF ** (k / 2)) for k in (-2, -1, 0, 1, 2)]
HOLD_RUNS = {lr: run(width_cfg(HOLD, lr)) for lr in hold_pts}
for _ in range(4):
    fin = {lr: final_loss(r) for lr, r in HOLD_RUNS.items() if math.isfinite(final_loss(r))}
    best = min(fin, key=fin.get)
    lo_, hi_ = min(HOLD_RUNS), max(HOLD_RUNS)
    if best not in (lo_, hi_):
        break
    new = lat(best / CF ** 0.5) if best == lo_ else lat(best * CF ** 0.5)
    HOLD_RUNS[new] = run(width_cfg(HOLD, new))
HOLD_RUNS = dict(sorted(HOLD_RUNS.items()))
fin = {lr: final_loss(r) for lr, r in HOLD_RUNS.items() if math.isfinite(final_loss(r))}
hb = min(fin, key=fin.get)
near = sorted(HOLD_RUNS, key=lambda lr: abs(math.log(lr / hb)))[:5]
hx, hy, _ = vertex(sorted(near), [[final_loss(HOLD_RUNS[lr])] for lr in sorted(near)])
hold_lr = 2 ** hx
err_oct = math.log2(hold_lr / pred_hold)
print(f"measured at width {HOLD}: best lr {hold_lr:.2e} (one seed)  →  the prediction was off by {err_oct:+.2f} octaves")
HOLD_BRACKETED = hb not in (min(HOLD_RUNS), max(HOLD_RUNS))
observe("hold_bracketed", HOLD_BRACKETED, f"width {HOLD}: the best learning rate is bracketed by the points measured around the prediction")

X4, Y4 = np.append(X, math.log2(HOLD)), np.append(Y, hx)
slope4, icpt4 = np.polyfit(X4, Y4, 1)
b4 = np.array([np.polyfit(X4, np.append(b, hx), 1) for b in B])
TARGET = 4096
ans_lr = pred(TARGET, (slope4, icpt4))
ans_ci = 2 ** np.percentile(b4[:, 0] * math.log2(TARGET) + b4[:, 1], [5, 95])
ci_oct = math.log2(ans_ci[1] / ans_ci[0])
cr = RULES["confidence_4096"]
CONF = "high" if (abs(err_oct) < cr["high_err_oct"] and ci_oct < cr["high_ci_oct"]) else ("medium" if abs(err_oct) < cr["medium_err_oct"] else "low")
if not HOLD_BRACKETED:
    CONF = "low"                                                     # an unbracketed check gives only a lower bound on the error
naive = SPA[WIDTHS[0]]["lr"] * WIDTHS[0] / TARGET
print(f"\nrefit on {len(X4)} widths: slope {slope4:.2f}.  Width {TARGET}: use {ans_lr:.2e} "
      f"(90% interval {ans_ci[0]:.2e} … {ans_ci[1]:.2e}, {ci_oct:.2f} octaves wide).  Confidence by the rule in §0: {CONF.upper()}")
print(f"for comparison, scaling width {WIDTHS[0]}'s best by 1/width gives {naive:.2e}; carrying it over unchanged gives {SPA[WIDTHS[0]]['lr']:.2e}")
# The bootstrap interval only resamples seeds; it says nothing about whether a straight line in log-log is the right
# model. The held-out width measures that directly: one octave of extrapolation missed by err_oct. 4,096 is again one
# octave beyond the data, so use that miss as the honest error band.
band = ans_lr * 2.0 ** (np.array([-1, 1]) * max(abs(err_oct), ci_oct / 2))
print(f"honest band: the same procedure missed by {abs(err_oct):.2f} octaves one octave out at {HOLD}, so expect the best lr at "
      f"{TARGET} within {band[0]:.2e} … {band[1]:.2e}; the bootstrap interval above reflects seed noise only")

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.35, 1]))
cols = dict(zip(WIDTHS + [HOLD], (C["aqua"], C["blue"], C["violet"], C["orange"])))
for w in WIDTHS:
    lrs = sorted(SP[w]["points"])
    m = [np.mean([l for l in SP[w]["points"][lr] if math.isfinite(l)] or [np.nan]) for lr in lrs]
    a1.plot(lrs, m, "-o", ms=3.5, color=cols[w])
    for lr in lrs:
        ls = SP[w]["points"][lr]
        if len(ls) > 1:
            a1.plot([lr] * len(ls), ls, ".", color=cols[w], alpha=0.45, ms=4)
    a = SPA[w]
    a1.plot(a["lr"], a["loss"], "*", ms=13, color=cols[w], mec=INK, mew=0.6)
    a1.plot([a["lo"], a["hi"]], [a["loss"] - 0.03] * 2, color=cols[w], lw=2.2)
    label_end(a1, lrs[-1], m[-1], f"width {w}", cols[w])
hl = list(HOLD_RUNS)
a1.plot(hl, [final_loss(HOLD_RUNS[lr]) for lr in hl], "-o", ms=3.5, color=cols[HOLD])
a1.plot(hold_lr, hy, "*", ms=13, color=cols[HOLD], mec=INK, mew=0.6)
label_end(a1, hl[-1], final_loss(HOLD_RUNS[hl[-1]]), f"width {HOLD} (held out)", cols[HOLD])
a1.axvline(pred_hold, color=cols[HOLD], lw=0.8, ls=":")
finite_all = [l for w in WIDTHS for ls in SP[w]["points"].values() for l in ls if math.isfinite(l)]
a1.set(xscale="log", ylim=(min(finite_all) - 0.1, min(finite_all) + 1.6), xlabel="peak learning rate",
       ylabel=f"validation loss at {TOTAL}", title="Loss vs learning rate; ★ = parabola vertex, bar = 90% interval")
ws = np.array(WIDTHS + [HOLD, TARGET])
a2.loglog(WIDTHS, [SPA[w]["lr"] for w in WIDTHS], "o", color=C["blue"], ms=6)
for w in WIDTHS:
    a2.plot([w, w], [SPA[w]["lo"], SPA[w]["hi"]], color=C["blue"], lw=1.6)
a2.plot(HOLD, hold_lr, "s", color=cols[HOLD], ms=6); label_end(a2, HOLD, hold_lr, f"{HOLD}: measured", cols[HOLD])
a2.plot(HOLD, pred_hold, "x", color=cols[HOLD], ms=7); label_end(a2, HOLD, pred_hold, "predicted", cols[HOLD], dy=-9)
xx = np.geomspace(WIDTHS[0] * 0.8, TARGET * 1.2, 50)
a2.plot(xx, [pred(x, (slope4, icpt4)) for x in xx], color=MUTED, lw=1)
a2.plot([TARGET, TARGET], ans_ci, color=C["red"], lw=2); a2.plot(TARGET, ans_lr, "D", color=C["red"], ms=6)
a2.annotate(f"{TARGET}: {ans_lr:.1e}\n({CONF} confidence)", (TARGET, ans_lr), xytext=(-10, -6), textcoords="offset points",
             color=C["red"], fontsize=8.5, ha="right", va="top")
a2.set(xlabel="width", ylabel="best peak learning rate", title=f"Best lr ∝ width^{slope4:.2f} (SP)")
a2.minorticks_off()
a2.set_xticks([int(w) for w in ws], [f"{int(w):,}" for w in ws])
yt = [y for y in (2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2) if a2.get_ylim()[0] <= y <= a2.get_ylim()[1]]
a2.set_yticks(yt, [f"{y:.0e}" for y in yt])
fig.tight_layout()
finish(fig, "lr_vs_width")

# %% [markdown]
r"""
**Why the right-hand side of each curve is a cliff.** Every sweep run ends with two diagnostics:
- the largest attention logit $q\cdot k/\sqrt{d}$ in any layer and head
- the RMS of the output logits

Both grow with the learning rate, and they grow faster in wider models. That growth is the failure
that QK-norm, logit soft-capping and MuonClip exist to prevent.

The same runs also log the update/weight ratio of the input embedding, the hidden matrices and the
untied output head separately. That shows whether one learning rate suits all three.
"""

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for w in WIDTHS:
    lrs = [lr for lr in sorted(SP[w]["runs"]) if SP[w]["runs"][lr] and SP[w]["runs"][lr].get("diag")]
    for ax, key in zip(axs, ("max_attn_logit", "logit_rms")):
        ys = [SP[w]["runs"][lr]["diag"][key] for lr in lrs]
        ax.plot(lrs, ys, "-o", ms=3, color=cols[w]); label_end(ax, lrs[-1], ys[-1], f"width {w}", cols[w])
axs[0].set(xscale="log", yscale="log", xlabel="peak learning rate", title="largest attention logit after training")
axs[1].set(xscale="log", yscale="log", xlabel="peak learning rate", title="output-logit RMS after training")
fig.tight_layout()
finish(fig, "lr_cliff_diagnostics")

HEADBODY = {}
for w in WIDTHS:
    r = SP[w]["runs"][lat(SPA[w]["lr"])] if lat(SPA[w]["lr"]) in SP[w]["runs"] else SP[w]["runs"][SP[w]["center"]]
    names_w = r["tensors"]
    tot = np.array(r["ratios"]["total"], dtype=float)
    late = tot[len(tot) // 2:]
    body = [i for i, n in enumerate(names_w) if n.split(".")[-1] in ("q", "kv", "o", "up", "down")]
    HEADBODY[w] = dict(emb=float(np.median(late[:, names_w.index("emb")])), body=float(np.median(late[:, body])),
                       head=float(np.median(late[:, names_w.index("head")])))
    print(f"width {w:>5} at its best lr, second half of the run: update/weight  embedding {HEADBODY[w]['emb']:.1e}   "
          f"hidden matrices {HEADBODY[w]['body']:.1e}   output head {HEADBODY[w]['head']:.1e}")

# %% [markdown]
r"""
### What muP changes

**Coordinate check first.** A muP implementation is trusted only after this check passes:
- Train each width for 10 steps at the *same* learning rate.
- Measure the RMS of the output logits and of the last block's output.

Under SP the logits blow up with width, because the output layer's updates add up coherently over
more inputs. Under a correct muP they stay roughly flat.
"""

# %%
def coord_check(width, mup, lr, steps=10):
    seed_all(0)
    cfg = width_cfg(width, lr, mup=mup)
    model = make_model(cfg).to(DEV)
    opt = make_opts(model, cfg["opt"])[0]
    for g in opt.param_groups:
        g["lr"] = g["peak"]
    xb, _ = eval_batches("width", 16)[0]
    rows = []
    for s in range(steps + 1):
        col = []
        with torch.no_grad(), autocast():
            model(xb[:8], collect=col)
        rows.append(dict(col))
        if s == steps:
            break
        win = REMAP[WIN["width"].batch(0, s, 16)].to(DEV)
        with autocast():
            loss = model(win[:, :-1], win[:, 1:])
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    del model, opt
    free_gpu()
    return rows


CC_LR = lat(SPA[WIDTHS[0]]["lr"])
CC = experiment("coord_check", lambda: {f"{'mup' if m else 'sp'}-{w}": coord_check(w, m, CC_LR)
                                        for m in (False, True) for w in WIDTHS + [HOLD]})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
growth = {}
for ax, m in zip(axs, ("sp", "mup")):
    for step, col in zip((0, 1, 5, 10), (MUTED, C["yellow"], C["orange"], C["red"])):
        ys = [CC[f"{m}-{w}"][step]["logits"] for w in WIDTHS + [HOLD]]
        ax.plot(WIDTHS + [HOLD], ys, "-o", ms=3, color=col); label_end(ax, HOLD, ys[-1], f"step {step}", col)
    growth[m] = CC[f"{m}-{HOLD}"][10]["logits"] / CC[f"{m}-{WIDTHS[0]}"][10]["logits"]
    ax.set(xscale="log", yscale="log", xlabel="width", xlim=(WIDTHS[0] * 0.8, HOLD * 2.2),
           title=f"{'standard' if m == 'sp' else 'muP'}: logit RMS, same lr {CC_LR:.1e}")
axs[0].set_ylabel("output-logit RMS")
fig.tight_layout()
finish(fig, "mup_coord_check")
print(f"logit RMS after 10 steps grows {growth['sp']:.1f}x from width {WIDTHS[0]} to {HOLD} under SP, {growth['mup']:.1f}x under muP")
observe("coord_check_mup_flatter", growth["mup"] < growth["sp"], "muP keeps the output logits flatter across width than SP does")

# %% [markdown]
r"""
**The muP sweep.** The same coarse grid at widths 512 and 1,024 (width 256 is shared: there the two
parameterizations are the same model). If muP is working, the three minima line up.
"""

# %%
MUP = {w: sweep_width(w, mup=True, refine=False) for w in WIDTHS}
MUPA = {w: analyse(MUP[w], boot=1) for w in WIDTHS}


def on_coarse(lr):
    k = math.log(lr / COARSE[0], CF)
    return abs(k - round(k)) < 1e-6


# the same estimator for both parameterizations: seed 0, coarse grid only, 3-point vertex
SPC = {w: analyse(dict(points={lr: [final_loss(r)] for lr, r in SP[w]["runs"].items() if on_coarse(lr)}, center=SP[w]["center"]), boot=1)
       for w in WIDTHS}
for w in WIDTHS:
    print(f"muP width {w:>5}: best lr {MUPA[w]['lr']:.2e}   loss {MUPA[w]['loss']:.4f}   (SP, same estimator: {SPC[w]['lr']:.2e}, {SPC[w]['loss']:.4f})")
mup_spread = math.log2(max(MUPA[w]["lr"] for w in WIDTHS) / min(MUPA[w]["lr"] for w in WIDTHS))
sp_spread = math.log2(max(SPC[w]["lr"] for w in WIDTHS) / min(SPC[w]["lr"] for w in WIDTHS))
print(f"spread of the best learning rate across widths {WIDTHS[0]}–{WIDTHS[-1]} (seed 0, coarse grid, same estimator): "
      f"SP {sp_spread:.2f} octaves, muP {mup_spread:.2f} octaves")
observe("mup_aligns_minima", mup_spread < sp_spread, f"muP narrows the spread of best learning rates ({sp_spread:.2f} → {mup_spread:.2f} octaves)")

# Transfer test at the held-out width: the width-256 optimum, used unchanged, under SP and under muP.
T_LR = lat(SPA[WIDTHS[0]]["lr"])
sp_naive = run(width_cfg(HOLD, T_LR))
mup_naive = run(width_cfg(HOLD, T_LR, mup=True))
print(f"\nwidth {HOLD} at width {WIDTHS[0]}'s best lr ({T_LR:.1e}), no retuning:  SP {final_loss(sp_naive):.4f}   "
      f"muP {final_loss(mup_naive):.4f}   (SP tuned at {HOLD}: {hy:.4f})")
observe("mup_transfer_beats_naive_sp", final_loss(mup_naive) < final_loss(sp_naive),
        f"carrying width {WIDTHS[0]}'s learning rate to width {HOLD} works better under muP than under SP")

fig, ax = plt.subplots(figsize=(7.4, 3.4))
for w in WIDTHS:
    lrs = sorted(MUP[w]["points"])
    ys = [MUP[w]["points"][lr][0] for lr in lrs]
    ax.plot(lrs, ys, "-o", ms=3.5, color=cols[w]); label_end(ax, lrs[-1], ys[-1], f"width {w}", cols[w])
    ax.plot(MUPA[w]["lr"], MUPA[w]["loss"], "*", ms=12, color=cols[w], mec=INK, mew=0.6)
fin_m = [l for w in WIDTHS for ls in MUP[w]["points"].values() for l in ls if math.isfinite(l)]
ax.set(xscale="log", ylim=(min(fin_m) - 0.1, min(fin_m) + 1.6), xlabel="peak learning rate (base value; hidden layers get ×256/width)",
       ylabel=f"validation loss at {TOTAL}", title=f"muP: minima spread {mup_spread:.2f} octaves (SP: {sp_spread:.2f})")
finish(fig, "mup_sweep")

if MODE != "learn":
    ANS["width"] = _clean(dict(
        sp={str(w): dict(lr=SPA[w]["lr"], lo=SPA[w]["lo"], hi=SPA[w]["hi"], loss=SPA[w]["loss"], cost2x=SPA[w]["cost2x"],
                         n_params=WIDTH_PARAMS[w], points={f"{lr:.4g}": v for lr, v in SP[w]["points"].items()}) for w in WIDTHS},
        slope=slope, slope_ci=np.percentile(bfit[:, 0], [5, 95]).tolist(), pred_hold=pred_hold, pred_hold_ci=pred_hold_ci.tolist(),
        hold=dict(width=HOLD, lr=hold_lr, loss=hy, err_oct=err_oct, points={f"{lr:.4g}": final_loss(r) for lr, r in HOLD_RUNS.items()}),
        slope4=slope4, answer=dict(width=TARGET, lr=ans_lr, ci=ans_ci.tolist(), ci_oct=ci_oct, confidence=CONF, band=band.tolist()),
        naive_inverse_width=naive, headbody=HEADBODY, coord_growth=growth,
        mup={str(w): dict(lr=MUPA[w]["lr"], loss=MUPA[w]["loss"]) for w in WIDTHS}, mup_spread=mup_spread, sp_spread=sp_spread,
        transfer=dict(lr=T_LR, sp=final_loss(sp_naive), mup=final_loss(mup_naive), sp_tuned=hy)))
    save_results()

# %% [markdown]
r"""
**What the measurements say.**

- **Under SP the best learning rate fell slightly faster than 1/width.** The fitted slope is about
  −1.1. A fit on three widths predicted width 2,048 to within 0.1 octave (about 7%), which is why
  the width-4,096 answer is given with a band of that size. The band is wider than the bootstrap
  interval, which only sees seed noise.
- **Under muP the three minima moved by only about a quarter of an octave**, against more than two
  octaves under SP.
- **Carrying width 256's best learning rate straight to width 2,048** cost more than a full nat of
  loss under SP, and only a few hundredths under muP.
- **One learning rate does not suit every layer under SP.** At each width's best learning rate the
  input embedding (initialized at scale 1) moved 15–40× less, relative to its size, than the hidden
  matrices. muP gives it its own constant rate for this reason.

> **Carry forward:** under the standard parameterization the best learning rate moves with width,
> so a small-model sweep has to be extrapolated, and the extrapolation has to be tested at a held-out
> width before it is trusted. muP's whole value is that it removes that step: it does not make the
> model better, it makes the small model's measurement valid for the large one.
"""
