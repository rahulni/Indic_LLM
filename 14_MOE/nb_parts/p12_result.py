from . import md, code, mathbox

CELLS = [
md(r'''
---
## K · The result

> The goal: after converting, the model must **keep training**, and its **loss must keep falling**.

One timeline from the first token to T3: the dense model, the two conversions, and the dense
control that never converted. The zoom panels show the training loss step by step through each
conversion.
'''),
code(r'''
def cat_runs(*names):
    xs, ys = [], []
    for n in names:
        L = R["runs"][n]
        xs.append(tokens_axis(L, np.arange(1, len(L["loss"]) + 1)))
        ys.append(np.asarray(L["loss"]))
    return np.concatenate(xs), np.concatenate(ys)

def val_series(names, start=None):
    xs, ys = ([start[0]], [start[1]]) if start else ([], [])
    for n in names:
        L = R["runs"][n]
        xs += list(tokens_axis(L, L["eval_step"])); ys += list(L["val"])
    return np.asarray(xs), np.asarray(ys)

if have("runs", "moe32") and have("runs", "control_b"):
    P = R["points"]
    t1, t2 = T1_TOK / 1e6, T2_TOK / 1e6
    fig = plt.figure(figsize=(12, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    xd, yd = val_series(["dense"])
    xc, yc = val_series(["control_a", "control_b"], (t1, P["dense_T1"]))
    xm8, ym8 = val_series(["moe8"], (t1, P["moe8_T1"]))
    xm32, ym32 = val_series(["moe32"], (t2, P["moe32_T2"]))
    ax.plot(xd, yd, color=DENSE_C, marker="o", ms=2.5, label="dense (Stage 1)")
    ax.plot(xc, yc, color=DENSE_C, ls="--", lw=1.6, label="dense control (never converted)")
    ax.plot(xm8, ym8, color=MOE_C, marker="o", ms=2.5, label="MoE path: MoE-8, then MoE-32")
    ax.plot(xm32, ym32, color=MOE_C, marker="o", ms=2.5)
    ax.plot([t2, t2], [ym8[-1], ym32[0]], color=MOE_C, lw=1, ls=":")
    for t, lab in ((t1, "T1: dense -> MoE-8 (partition)"), (t2, "T2: MoE-8 -> MoE-32 (clone + drop)")):
        ax.axvline(t, color=MUTED, lw=1)
        ax.text(t, 0.98, "  " + lab, transform=ax.get_xaxis_transform(), color=INK2, va="top", fontsize=8.5)
    lo = min(yc.min(), ym32.min())
    ax.set_ylim(lo - 0.05, lo + 0.9)
    ax.set_xlim(t1 * 0.45, xm32[-1] * 1.13)
    ax.set(xlabel="tokens seen (M)", ylabel="validation loss",
           title="Convert, keep training, keep dropping: the MoE path vs the dense control")
    up, dn = (("MoE-32", xm32[-1], ym32[-1]), ("dense control", xc[-1], yc[-1]))[:: 1 if ym32[-1] >= yc[-1] else -1]
    for (lab, x_, y_), dy in ((up, 14), (dn, -14)):
        ax.annotate(f"{lab} {y_:.3f}", (x_, y_), xytext=(10, dy), textcoords="offset points", color=INK2, va="center",
                    fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
    W, B = 150, 100
    for j, (pre, post_m, post_c, t, lab) in enumerate((("dense", "moe8", "control_a", t1, "T1"), ("moe8", "moe32", "control_b", t2, "T2"))):
        a = fig.add_subplot(gs[1, j])
        Lp = R["runs"][pre]
        xp = tokens_axis(Lp, np.arange(1, len(Lp["loss"]) + 1))[-B:]
        a.plot(xp, smooth(Lp["loss"], 10)[-B:], color=MOE_C if pre == "moe8" else DENSE_C, lw=1.6)
        if pre == "moe8":
            Lc = R["runs"]["control_a"]
            a.plot(tokens_axis(Lc, np.arange(1, len(Lc["loss"]) + 1))[-B:], smooth(Lc["loss"], 10)[-B:], color=DENSE_C, ls="--", lw=1.4)
        for n, c, ls in ((post_m, MOE_C, "-"), (post_c, DENSE_C, "--")):
            L = R["runs"][n]
            w_ = min(W, len(L["loss"]))
            a.plot(tokens_axis(L, np.arange(1, w_ + 1)), smooth(L["loss"][:w_], 10), color=c, ls=ls, lw=1.6,
                   label=("MoE path" if c == MOE_C else "dense control") if j == 0 else None)
        a.axvline(t, color=MUTED, lw=1)
        a.set(title=f"Zoom: training loss through {lab} (10-step mean)", xlabel="tokens seen (M)", ylabel="train loss" if j == 0 else None)
        if j == 0:
            a.legend()
    savefig(fig, "result_timeline")
'''),
md(r'''
### Validation loss at every checkpoint

"Right after" means zero training steps after the surgery: it is the price paid at the moment of
conversion.
'''),
code(r'''
if have("points", "moe32_T3") and have("params"):
    P, Pa = R["points"], R["params"]
    rows = [
        ("dense, end of Stage 1", "T1", P["dense_T1"], Pa["dense"], Pa["dense_active"]),
        ("MoE-8, right after conversion", "T1", P["moe8_T1"], Pa["moe8"], Pa["moe8_active"]),
        ("MoE-8, end of Stage 2", "T2", P["moe8_T2"], Pa["moe8"], Pa["moe8_active"]),
        ("dense control", "T2", P["control_T2"], Pa["dense"], Pa["dense_active"]),
        ("MoE-32, right after growth", "T2", P["moe32_T2"], Pa["moe32"], Pa["moe32_active"]),
        ("MoE-32, end of Stage 3", "T3", P["moe32_T3"], Pa["moe32"], Pa["moe32_active"]),
        ("dense control", "T3", P["control_T3"], Pa["dense"], Pa["dense_active"]),
    ]
    tok = {"T1": T1_TOK, "T2": T2_TOK, "T3": T2_TOK + S3 * TOK_STEP}
    lines = ["| model | at | tokens | val loss | total params | active params | training FLOPs/token (6 x active) |", "|---|---|---|---|---|---|---|"]
    for name, at, v, tot, act in rows:
        lines.append(f"| {name} | {at} | {tok[at] / 1e6:.0f}M | {v:.4f} | {tot / 1e6:.1f}M | {act / 1e6:.1f}M | {6 * act / 1e6:.0f}M |")
    display(Markdown("\n".join(lines)))
'''),
md(r'''
### Is the MoE actually better than the dense control? A paired test

Both final models are scored on the same 64 validation batches (2,048 sequences, 524K tokens). The
per-batch differences are paired, so batch difficulty cancels. A bootstrap over batches gives a 95%
interval for the mean difference. **This is one seed per model.** The interval covers validation
noise, not seed-to-seed variation.

🧠 **Intuition.**

* **Same exam.** To compare two students, give them the *same* exam and compare them question by
  question. Some questions are hard for everyone; comparing per question cancels that difficulty
  out, and what remains is the real difference between the students. Here a "question" is a
  validation batch.
* **What a nat is.** Loss in nats is $-\log$ of the probability the model gave the right next
  token. $e^{\text{loss}}$ is the **perplexity**: how many equally likely tokens the model is, in
  effect, choosing between. A gap of 0.02 nats means one model is about 2% more "perplexed" than
  the other.
'''),
mathbox("perplexity, and the paired bootstrap", r'''
**Nats and perplexity.** The model's loss and perplexity are

$$\ell = -\frac{1}{N}\sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = e^{\ell}, \qquad \frac{\mathrm{PPL}_A}{\mathrm{PPL}_B} = e^{\ell_A - \ell_B}.$$

At T3, $e^{1.569} = 4.80$ and $e^{1.549} = 4.71$. Their ratio is $e^{0.020} = 1.020$: the MoE is 2%
more perplexed.

**Paired differences.** For validation batch $b = 1, \dots, n$ (here $n = 64$), let
$d_b = \ell_b^{\text{MoE}} - \ell_b^{\text{dense}}$. Then

$$\bar d = \frac{1}{n}\sum_{b=1}^n d_b, \qquad \operatorname{Var}(\bar d) = \frac{\operatorname{Var}\ell^{\text{MoE}} + \operatorname{Var}\ell^{\text{dense}} - 2\operatorname{Cov}(\ell^{\text{MoE}}, \ell^{\text{dense}})}{n}.$$

Both models find the same batches hard, so the covariance is large and cancels most of the
variance. That is why a 0.02 difference can be measured precisely here.

**Bootstrap.**

1. Draw $n$ batch indices with replacement and recompute $\bar d^{\,*}$.
2. Repeat 10,000 times.
3. The 2.5th and 97.5th percentiles of $\bar d^{\,*}$ form the 95% interval.

If the interval excludes 0, the difference is larger than validation noise. It still says nothing
about seed-to-seed variation, which a second training seed would measure.
'''),
code(r'''
def paired_stats(a, b, seed=0):
    """Mean of per-batch differences a - b with a 95% bootstrap interval over batches."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    boots = d[np.random.default_rng(seed).integers(0, len(d), (10_000, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), wins=int((d < 0).sum()), n=len(d),
                a=float(a.mean()), b=float(b.mean()))

if COMPUTE and moe32 is not None and not have("final_pair"):
    a = evaluate(moe32, BUDGET["final_val_batches"], per_batch=True)
    b = evaluate(control, BUDGET["final_val_batches"], per_batch=True)
    R["final_pair"] = dict(moe32=a, control=b)
    save_results()
if have("final_pair"):
    a, b = np.asarray(R["final_pair"]["moe32"]), np.asarray(R["final_pair"]["control"])
    d = a - b
    st = paired_stats(a, b)
    lo, hi = st["lo"], st["hi"]
    R["paired"] = dict(mean=st["mean"], lo=lo, hi=hi, wins=st["wins"], n=st["n"],
                       moe32=float(a.mean()), control=float(b.mean()))
    print(f"MoE-32 {a.mean():.4f} vs dense control {b.mean():.4f}: difference {d.mean():+.4f} nats, "
          f"95% CI [{lo:+.4f}, {hi:+.4f}], MoE better on {(d < 0).sum()}/{len(d)} batches")
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.hist(d, bins=24, color=MOE_C, alpha=0.85, rwidth=0.9)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvspan(lo, hi, color=MOE_C, alpha=0.12, lw=0)
    ax.set(xlabel="per-batch loss difference, MoE-32 minus dense control (nats)", ylabel="batches",
           title="Paired comparison at T3 (shaded: 95% CI of the mean)")
    savefig(fig, "paired_T3")
'''),
md(r'''
### Why the MoE did not beat the dense control here

The core requirement held: after each conversion the model kept training, and its loss
fell well below where the dense model was at T1. The extra prediction, that MoE-32 would *beat* a
dense model given the same tokens and the same compute per token, **missed** in the submitted run.
MoE-32 finished about 0.02 nats behind, with a confidence interval that excludes zero. Reading the
timeline above, the gap comes from the two conversions, not from the experts learning slowly:

* **T1 cost about 0.05 nats** (partition keeps the dense output only on average). MoE-8 spent its
  8M tokens recovering, and ended Stage 2 about 0.01 behind the control.
* **T2 cost about 0.17 nats.** About 98% of tokens put all four picks inside one clone family (P5),
  and half of every clone's neurons were redrawn. Stage 3 had 12M tokens to recover and narrowed
  the gap to 0.02, but did not close it.
* Nothing here says MoE loses to dense in general. The published wins come with budgets of many
  times the dense model's tokens, and with experts much wider than 192. What this run does show is
  that **each conversion has a price in tokens**. The growth lab (Part J) points at cheaper ways to
  grow at this size: plain copies and staggered copies both ended below the drop + Gumbel recipe the
  main path used, which was registered before the run and is reported as run. **Part K.2** then
  runs both follow-ups to T3: copy growth recovers part of the gap, and not growing at all recovers
  most of it.

### What the models write

Same prompts, same sampling seed (temperature 0.8, top-40). The text is a qualitative check that
surgery did not break the model, not a measurement.
'''),
code(r'''
SAMPLE_ORDER = [("dense_T1", "dense at T1"), ("moe8_at_conversion", "MoE-8, 0 steps after conversion"), ("moe8_T2", "MoE-8 at T2"),
                ("moe32_at_growth", "MoE-32, 0 steps after growth"), ("moe32_T3", "MoE-32 at T3"), ("control_T3", "dense control at T3")]
for key, title in SAMPLE_ORDER:
    if have("samples", key):
        print(f"--- {title}")
        for s_ in R["samples"][key]:
            print("    " + s_.replace("\n", " "))
'''),
md(r'''
### Checklist
'''),
code(r'''
if have("points", "moe32_T3") and have("params"):
    P, Pa, D = R["points"], R["params"], R["runs"]["dense"]
    peak = max((R["runs"][n].get("peak_gib") or 0) for n in ("dense", "moe8", "moe32", "control_a", "control_b"))
    checks = [
        ("Trained a dense ('linear') model", D["val"][-1] < D["val"][0], f"val {D['val'][0]:.2f} -> {D['val'][-1]:.3f} over {S1 * TOK_STEP / 1e6:.0f}M tokens"),
        ("Converted it into an MoE (and grew that MoE)", Pa["moe32"] > Pa["dense"], f"{Pa['dense'] / 1e6:.1f}M -> {Pa['moe8'] / 1e6:.1f}M -> {Pa['moe32'] / 1e6:.1f}M total"),
        ("It continued to train after conversion", P["moe8_T2"] < P["moe8_T1"] and P["moe32_T3"] < P["moe32_T2"],
         f"MoE-8 {P['moe8_T1']:.3f} -> {P['moe8_T2']:.3f}; MoE-32 {P['moe32_T2']:.3f} -> {P['moe32_T3']:.3f}"),
        ("The loss dropped below where the dense model was", P["moe32_T3"] < P["dense_T1"], f"{P['dense_T1']:.3f} at T1 -> {P['moe32_T3']:.3f} at T3"),
        ("Bigger model at the same compute per token", Pa["moe32"] > 3 * Pa["dense"] and abs(Pa["moe32_active"] / Pa["dense_active"] - 1) < 0.01,
         f"{Pa['moe32'] / Pa['dense']:.1f}x the parameters, {Pa['moe32_active'] / Pa['dense_active']:.3f}x the active parameters"),
        ("Fits a laptop / Colab GPU", peak < 7.5, f"peak {peak:.2f} GiB allocated on {R['meta'].get('gpu', '?')}"),
    ]
    R["checklist"] = [dict(item=a, ok=bool(b), detail=c) for a, b, c in checks]
    save_results()
    display(Markdown("\n".join(f"- {'✅' if c['ok'] else '❌'} **{c['item']}**: {c['detail']}" for c in R["checklist"])))
'''),
]
