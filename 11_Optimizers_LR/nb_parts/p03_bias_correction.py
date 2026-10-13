# %% [markdown]
r"""
## §2 Bias correction off: the first twenty steps

**Intuition.** Both averages start at zero, but they recover at very different speeds. $m$ forgets
its zero start in about 10 steps ($\beta_1 = 0.9$). $v$ takes about 1,000 ($\beta_2 = 0.999$). Without
correction, $v$ reads far too low for far longer than $m$ does. Dividing by $\sqrt{v}$ then makes the
step too **large**, not too small.

**Math.** The corrections are plain scalars, so the uncorrected step divided by the corrected step
is the same for every gradient sequence ($\epsilon$ aside):
$$f(t) \;=\; \frac{m_t/\sqrt{v_t}}{\hat m_t/\sqrt{\hat v_t}} \;=\; \frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}}.$$
At $t=1$, $f = 0.1/\sqrt{0.001} = 3.16$. The first step without correction is 3.16 times too large.

"When does the difference stop mattering?" has a precise answer: the first step after which
$|f(t) - 1|$ stays below a threshold for good. The rule fixed in §0 uses 1%.
"""

# %%
rng = np.random.default_rng(0)
G20 = GRADS + list(np.round(0.5 + 0.05 * rng.standard_normal(15), 3))      # the five above, then 15 more around 0.5
on, off = adam_by_hand(G20, True), adam_by_hand(G20, False)
t20 = np.arange(1, 21)
f_closed = (1 - BETA1 ** t20) / np.sqrt(1 - BETA2 ** t20)
f_measured = np.array([b["step"] / a["step"] for a, b in zip(on, off)])
check(np.max(np.abs(f_measured / f_closed - 1)) < 1e-5, "measured step ratio equals (1-β1^t)/sqrt(1-β2^t) at every step (ε shifts it by < 1e-5)")

fig, axs = plt.subplots(2, 2, figsize=(9.6, 5.6), sharex=True)
panels = [("m", "m_hat", "first moment", "m (uncorrected)", "m̂ (corrected)"),
          ("v", "v_hat", "second moment", "v (uncorrected)", "v̂ (corrected)")]
for ax, (raw, hat, title, l1, l2) in zip(axs[0], panels):
    ax.plot(t20, [r[raw] for r in on], "-o", ms=3, color=C["orange"]); label_end(ax, 20, on[-1][raw], l1, C["orange"])
    ax.plot(t20, [r[hat] for r in on], "-o", ms=3, color=C["blue"]); label_end(ax, 20, on[-1][hat], l2, C["blue"])
    ax.set(title=title, xlim=(0, 27))
axs[0, 1].set_yscale("log")
ax = axs[1, 0]
ax.plot(t20, [-r["step"] / LR1 for r in off], "-o", ms=3, color=C["orange"]); label_end(ax, 20, -off[-1]["step"] / LR1, "without", C["orange"])
ax.plot(t20, [-r["step"] / LR1 for r in on], "-o", ms=3, color=C["blue"]); label_end(ax, 20, -on[-1]["step"] / LR1, "with", C["blue"])
ax.set(title="step size / η", xlabel="step", ylim=(0, 7.2))
ax = axs[1, 1]
ax.plot(t20, [r["w"] for r in off], "-o", ms=3, color=C["orange"]); label_end(ax, 20, off[-1]["w"], "without", C["orange"])
ax.plot(t20, [r["w"] for r in on], "-o", ms=3, color=C["blue"]); label_end(ax, 20, on[-1]["w"], "with", C["blue"])
ax.set(title="the weight", xlabel="step")
fig.suptitle("Adam with and without bias correction, first 20 steps (β1=0.9, β2=0.999, η=1e-3)", color=INK, x=0.01, ha="left")
fig.tight_layout()
finish(fig, "bias_correction_first20")
print(f"step/η without correction: t=1 {f_closed[0]:.2f}, peak {f_closed.max():.2f} at t={t20[f_closed.argmax()]}, t=20 {f_closed[-1]:.2f}")
print(f"after 20 steps the weight is at {off[-1]['w']:.5f} without correction vs {on[-1]['w']:.5f} with: "
      f"{(1 - off[-1]['w']) / (1 - on[-1]['w']):.1f}x as far from where it started")
check(round(f_closed[0], 2) == 3.16, "the first uncorrected step is 3.16 η")
check(f_closed[-1] > f_closed[0], "at step 20 the gap is wider than at step 1: it has not started closing")

# %% [markdown]
r"""
Within the first twenty steps the difference **grows**. It rises from 3.16× at step 1 to a peak near
step 12, because $m$ recovers in tens of steps while $v$ needs thousands. Twenty steps is far too
early to call. The same closed form, carried out to 100,000 steps for several $\beta_2$:
"""

# %%
T_LONG = np.arange(1, 200_001)


def multiplier(b2, b1=BETA1, t=T_LONG):
    return (1 - b1 ** t) / np.sqrt(1 - b2 ** t)


def stop_step(dev, p):
    """First step after which |dev| < p for every later step (dev indexed from t = 1)."""
    bad = np.nonzero(np.abs(dev) >= p)[0]
    return int(bad[-1]) + 2 if bad.size else 1


BC_STOP = {}
print(f"{'β2':>6} {'memory 1/(1-β2)':>16} {'t=1':>6} {'peak':>12}   steps until within  " + "  ".join(f"{p:.0%}" for p in RULES["bias_correction_stop"]["p"]))
for b2 in (0.95, 0.98, 0.99, 0.999):
    f = multiplier(b2)
    stops = {p: stop_step(f - 1, p) for p in RULES["bias_correction_stop"]["p"]}
    BC_STOP[b2] = stops
    print(f"{b2:>6} {1 / (1 - b2):16.0f} {f[0]:6.2f} {f.max():6.2f} @ t={f.argmax() + 1:<4}   " +
          "  ".join(f"{stops[p]:>5}" for p in RULES["bias_correction_stop"]["p"]))
m_part = stop_step(-(BETA1 ** T_LONG), 0.01)
v_part = stop_step(1 / np.sqrt(1 - 0.999 ** T_LONG) - 1, 0.01)
print(f"split for β2=0.999 at 1%: the m correction stops mattering after {m_part} steps, the v correction after {v_part}")
check(abs(BC_STOP[0.999][0.01] - v_part) <= 2 and v_part > 50 * m_part, "with β2=0.999 the v correction, not m, sets how long bias correction matters")

fig, ax = plt.subplots(figsize=(7.6, 3.4))
for b2, col in zip((0.95, 0.98, 0.99, 0.999), (C["aqua"], C["yellow"], C["violet"], C["orange"])):
    f = multiplier(b2)[:20000]
    ax.semilogx(T_LONG[:20000], f, color=col)
    s1 = BC_STOP[b2][0.01]
    ax.plot(s1, f[s1 - 1], "o", ms=4, color=col)
    label_end(ax, s1, f[s1 - 1], f"β2={b2}: within 1% from step {s1:,}", col, dx=5, dy=8)
ax.axhline(1, color=MUTED, lw=0.8)
ax.set(xlabel="step (log scale)", ylabel="uncorrected / corrected step", ylim=(0, 7),
       title="How long bias correction matters depends almost entirely on β2")
finish(fig, "bias_correction_long")

# %% [markdown]
r"""
**The per-step ratio converges, but the displacement does not.** With consistently signed gradients,
every uncorrected step overshoots by $(f(t)-1)\,\eta$, and those overshoots add up. The total is the
extra distance the weight has travelled by the time the ratio has converged. That distance is never
taken back.
"""

# %%
for b2 in (0.95, 0.999):
    excess = np.sum(multiplier(b2) - 1)
    print(f"β2={b2}: cumulative extra displacement {excess:+.0f} η (consistent gradients, upper bound)")
EXCESS_0999 = float(np.sum(multiplier(0.999) - 1))
check(EXCESS_0999 > 1000, "with β2=0.999, switching bias correction off adds over a thousand η of displacement")

# %% [markdown]
r"""
**Where ε starts to matter.** $\epsilon = 10^{-8}$ is added to $\sqrt{v}$. Uncorrected,
$\sqrt{v_1} = 0.0316\,|g|$ at the first step, so ε competes with gradients 30× larger than it does
with correction. This matters for weights whose gradients are tiny, as in many real layers.
"""

# %%
print(f"{'|g|':>8} {'step/η with correction':>23} {'without':>9}")
for g in (1e-2, 1e-4, 1e-6, 1e-7, 1e-8):
    with_bc = g / (g + EPS)
    without = 0.1 * g / (math.sqrt(0.001) * g + EPS)
    print(f"{g:8.0e} {with_bc:23.3f} {without:9.3f}")
on_1e7, off_1e7 = 1e-7 / (1e-7 + EPS), 0.1 * 1e-7 / (math.sqrt(0.001) * 1e-7 + EPS)
check(on_1e7 > 0.9 and off_1e7 < 1.0, "at |g| = 1e-7, ε already cancels the uncorrected 3.16x overshoot")

# %% [markdown]
r"""
### Two fixes that are often confused

*Bias correction* and *decoupled weight decay* (the "W" in AdamW) are separate fixes for separate
problems:

| fix | problem it solves | what switching it off does |
|---|---|---|
| bias correction ($\hat m$, $\hat v$) | both averages start at zero | early steps are up to 6.6× too large (β2 = 0.999) |
| decoupled decay (AdamW) | L2 decay gets divided by $\sqrt{\hat v}$ | how much a weight decays depends on its gradient history |

Below, the same weight ($w_0 = 1$, $\lambda = 0.1$, the 20 gradients above) is run four ways, each
fix toggled separately.
"""

# %%
def adam_variant(grads, bias_correction, decoupled, lam=0.1, b1=BETA1, b2=BETA2, lr=LR1):
    w, m, v, ws = 1.0, 0.0, 0.0, []
    for t, g in enumerate(grads, 1):
        if decoupled:
            w -= lr * lam * w                                   # AdamW: decay first, outside the ratio
        else:
            g = g + lam * w                                     # L2: decay joins the gradient
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = (m / (1 - b1 ** t), v / (1 - b2 ** t)) if bias_correction else (m, v)
        w -= lr * mh / (math.sqrt(vh) + EPS)
        ws.append(w)
    return ws


variants = {(bc, dec): adam_variant(G20, bc, dec) for bc in (True, False) for dec in (True, False)}
print(f"{'':<26}{'decoupled (AdamW)':>18}{'L2 (Adam)':>12}")
for bc in (True, False):
    print(f"{'bias correction ' + ('on' if bc else 'off'):<26}{1 - variants[(bc, True)][-1]:18.5f}{1 - variants[(bc, False)][-1]:12.5f}   ← distance moved in 20 steps")
d_bc = abs(variants[(False, True)][-1] - variants[(True, True)][-1])
d_dec = abs(variants[(True, False)][-1] - variants[(True, True)][-1])
print(f"switching bias correction off moves w by {d_bc:.2e}; switching decoupling off moves it by {d_dec:.2e}")
check(d_bc > 10 * d_dec, "over the first 20 steps bias correction matters far more than how decay is applied")

# %% [markdown]
r"""
> **Carry forward:** without bias correction, early Adam steps are up to 6.6× too large at
> $\beta_2=0.999$, and the gap *widens* over the first dozen steps. It is within 1% only after
> ~3,900 steps, or ~76 at $\beta_2 = 0.95$. The weights keep the extra displacement for good. §4
> measures what that does to a real model's loss.
"""
