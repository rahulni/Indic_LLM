# %% [markdown]
r"""
## §1 One weight, five gradients: Adam by hand

**Intuition.** Adam keeps two running averages per weight:
- The first, $m$, is the average gradient. It sets the **direction**.
- The second, $v$, is the average squared gradient. It sets the **scale**.

Their ratio is close to $\pm1$ whenever the gradients agree. So every weight moves by about $\eta$,
whatever its gradient's size. The gradient decides the direction and the learning rate decides the
distance.

**Math.** With $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, at step $t$:
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_t, &
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,\\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, &
\hat v_t &= \frac{v_t}{1-\beta_2^t},\\
\Delta w_t &= -\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\end{aligned}
$$

The hats are **bias correction**. Both averages start at zero, so early on they read too low. A
constant gradient $g$ gives $m_t = g(1-\beta_1^t)$ exactly, and dividing by $1-\beta_1^t$ undoes
that. The same holds for $v$. §2 switches the correction off.

The worked example: one weight at $w_0 = 1$, $\eta = 10^{-3}$, gradients 0.50, 0.40, 0.60, 0.45,
0.55. We compute it three ways:
1. **Exactly**, with 50-digit decimals. This is the reference.
2. **By hand** in ordinary floats, the five lines above.
3. **With `torch.optim.Adam`**, reading its internal state after every step.

The same check exists as a standalone script, `adam_by_hand.py`. It exits with code 1 on any
mismatch.
"""

# %%
GRADS = [0.50, 0.40, 0.60, 0.45, 0.55]
LR1, BETA1, BETA2, EPS, W0 = 1e-3, 0.9, 0.999, 1e-8, 1.0
FIELDS = ("m", "v", "m_hat", "v_hat", "step", "w")


def adam_exact(grads=GRADS):
    """The reference: every operation in 50-digit decimal arithmetic."""
    D = decimal.Decimal
    decimal.getcontext().prec = 50
    b1, b2, lr, eps = D("0.9"), D("0.999"), D("0.001"), D("1e-8")
    w, m, v, rows = D(1), D(0), D(0), []
    for t, g in enumerate(grads, 1):
        g = D(str(g))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        step = -lr * m_hat / (v_hat.sqrt() + eps)
        w += step
        rows.append(dict(m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=step, w=w))
    return rows


def adam_by_hand(grads=GRADS, bias_correction=True, b1=BETA1, b2=BETA2, lr=LR1, eps=EPS, w0=W0):
    """The five lines, in ordinary floats."""
    w, m, v, rows = w0, 0.0, 0.0, []
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t) if bias_correction else m
        v_hat = v / (1 - b2 ** t) if bias_correction else v
        step = -lr * m_hat / (math.sqrt(v_hat) + eps)
        w += step
        rows.append(dict(m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=step, w=w))
    return rows


ref = adam_exact()
hand = adam_by_hand()
print(f"{'t':>2} {'g':>5} {'m':>8} {'v':>9} {'m_hat':>7} {'v_hat':>7} {'step':>10} {'w':>9}")
for t, (g, r) in enumerate(zip(GRADS, hand), 1):
    print(f"{t:>2} {g:5.2f} {r['m']:8.4f} {r['v']:9.6f} {r['m_hat']:7.4f} {r['v_hat']:7.4f} {r['step']:10.6f} {r['w']:9.6f}")

# The standard worked table, to the digits it is usually printed with:
PRINTED = dict(m=[0.0500, 0.0850, 0.1365, 0.1678, 0.2061], v=[0.000250, 0.000410, 0.000769, 0.000971, 0.001273],
               m_hat=[0.5000, 0.4474, 0.5037, 0.4881, 0.5032], v_hat=[0.2500, 0.2050, 0.2567, 0.2431, 0.2550],
               step=[-0.001000, -0.000988, -0.000994, -0.000990, -0.000996],
               w=[0.999000, 0.998012, 0.997018, 0.996028, 0.995031])
DIGITS = dict(m=4, v=6, m_hat=4, v_hat=4, step=6, w=6)
check(all(round(r[f], DIGITS[f]) == PRINTED[f][i] for i, r in enumerate(hand) for f in FIELDS),
      "the hand computation reproduces the worked table to every printed digit")
worst = max(abs(r["step"] + LR1) for r in hand) / LR1
check(worst < 0.015, f"every step is within {worst:.1%} of η, although the gradients range from 0.40 to 0.60 (±20%)")

# %% [markdown]
r"""
**The same five steps in PyTorch.** `torch.optim.Adam` stores exactly two tensors per weight:
- `exp_avg`, which is $m$
- `exp_avg_sq`, which is $v$

plus the step count. These are the "two values per weight" counted in R.4.

It never stores $\hat m$ or $\hat v$. It folds both corrections into the step instead:
$$\Delta w = -\frac{\eta}{1-\beta_1^t}\cdot\frac{m}{\sqrt{v}/\sqrt{1-\beta_2^t}+\epsilon},$$
which is the textbook formula rearranged. So $\hat m$ and $\hat v$ are derived below from the stored
state, and the step is read as $w_\text{after} - w_\text{before}$.
"""

# %%
def adam_torch(grads=GRADS, dtype=torch.float64, device="cpu", **impl):
    p = torch.nn.Parameter(torch.tensor([W0], dtype=dtype, device=device))
    opt = torch.optim.Adam([p], lr=LR1, betas=(BETA1, BETA2), eps=EPS, **impl)
    rows = []
    for g in grads:
        before = p.detach().clone()
        p.grad = torch.tensor([g], dtype=dtype, device=device)
        opt.step()
        st = opt.state[p]
        t = int(st["step"])
        m, v = st["exp_avg"].item(), st["exp_avg_sq"].item()
        rows.append(dict(m=m, v=v, m_hat=m / (1 - BETA1 ** t), v_hat=v / (1 - BETA2 ** t),
                         step=(p.detach() - before).item(), w=p.item()))
    return rows


IMPLS = [("by hand, float64", lambda: adam_by_hand(), 1e-12),
         ("torch float64, for-loop", lambda: adam_torch(foreach=False), 1e-12),
         ("torch float64, foreach", lambda: adam_torch(foreach=True), 1e-12),
         ("torch float32, for-loop", lambda: adam_torch(dtype=torch.float32, foreach=False), 1e-6),
         ("torch float32, foreach", lambda: adam_torch(dtype=torch.float32, foreach=True), 1e-6)]
if DEV == "cuda":
    IMPLS += [("torch float64, fused CUDA", lambda: adam_torch(device="cuda", fused=True), 1e-12),
              ("torch float32, fused CUDA", lambda: adam_torch(dtype=torch.float32, device="cuda", fused=True), 1e-5)]

# Step by step, side by side: the hand computation and PyTorch's own numbers (float32, the default dtype).
torch_rows = adam_torch(dtype=torch.float32, foreach=False)
print(f"{'t':>2} {'':<8}" + "".join(f"{f:>13}" for f in FIELDS))
for t, (h, p) in enumerate(zip(hand, torch_rows), 1):
    print(f"{t:>2} {'by hand':<8}" + "".join(f"{h[f]:13.8f}" for f in FIELDS))
    print(f"{'':>2} {'PyTorch':<8}" + "".join(f"{p[f]:13.8f}" for f in FIELDS))
agree = min(int(-math.log10(max(abs(h[f] - p[f]) for h, p in zip(hand, torch_rows)) + 1e-30)) for f in FIELDS)
print(f"every quantity at every step agrees to at least {agree} decimal places (float32 PyTorch vs float64 by hand)\n")
check(agree >= 6, f"hand and PyTorch agree to {agree} decimal places on m, v, m̂, v̂, the step and w at every step")

diffs = {}
print("largest |difference| from the 50-digit reference over the five steps:")
print(f"{'implementation':<27}" + "".join(f"{f:>9}" for f in FIELDS) + "  tolerance")
for name, fn, tol in IMPLS:
    rows = fn()
    diffs[name] = {f: max(abs(float(r[f]) - float(e[f])) for r, e in zip(rows, ref)) for f in FIELDS}
    print(f"{name:<27}" + "".join(f"{diffs[name][f]:9.1e}" for f in FIELDS) + f"  {tol:.0e}")
for name, _, tol in IMPLS:
    check(max(diffs[name].values()) <= tol, f"{name}: every quantity within {tol:.0e} of the exact value")
if MODE != "learn":
    ANS["adam_by_hand"] = _clean(dict(table=[{k: float(v) for k, v in r.items()} for r in ref], grads=GRADS,
                                      torch_rows=torch_rows, agree_decimals=agree,
                                      diffs=diffs, tolerances={n: t for n, _, t in IMPLS}), sig=8)
    save_results()

# %% [markdown]
r"""
Every implementation agrees with the exact reference. float64 agrees to rounding error (~1e-16),
and the float32 loop and foreach kernels to ~1e-8. The fused CUDA kernel in float32 is the
exception: its $v$ differs by about 1e-5 *relative*, which shows up as ~3e-6 in $\hat v$, while its
step and weight still agree to 1e-8.

The cause is one subtraction. The fused kernel computes $1-\beta_2$ in float32. float32 cannot
represent 0.999 exactly, and subtracting it from 1 leaves an error of about 1.3e-5 relative. The
loop kernels compute $1-\beta_2$ in Python's float64 instead.

The step escapes the error because the kernel's own bias correction, $1-\beta_2^t$, uses the same
float32 $\beta_2$. Inside the kernel, $v/(1-\beta_2^t)$ is therefore still a properly normalised
average. The 3e-6 appears only in the $\hat v$ derived here from the stored $v$ with the exact
$\beta_2$.
"""

# %%
one_minus_b2_f32 = float(np.float32(1.0) - np.float32(0.999))
rel = abs(one_minus_b2_f32 - 0.001) / 0.001
print(f"1 - 0.999 in float32 = {one_minus_b2_f32:.10f}   relative error {rel:.2e}")
check(1e-5 < rel < 2e-5, "float32 computes 1 - 0.999 with a relative error of about 1.3e-5")
if DEV == "cuda":
    fused = adam_torch(dtype=torch.float32, device="cuda", fused=True)
    v_rel = max(abs(r["v"] - float(e["v"])) / float(e["v"]) for r, e in zip(fused, ref))
    observe("fused_fp32_v_error", abs(v_rel - rel) / rel < 0.2,
            f"the fused kernel's v is off by {v_rel:.2e} relative, the same size as float32's 1 - 0.999 error")

# %% [markdown]
r"""
**How large can one Adam step be?** "About $\eta$" is a typical value, not a bound. The paper that
introduced Adam gives the bound:
$$|\Delta w| \le \eta\cdot\max\!\Big(1,\; \frac{1-\beta_1}{\sqrt{1-\beta_2}}\Big).$$
With $\beta_2 = 0.999$ that is $3.16\,\eta$. With $\beta_2 = 0.95$ it is $\eta$.

The bound is reached in exactly one situation: a parameter whose gradient has been **zero for a
long time** suddenly receives one. Then $m$ jumps to $0.1g$ while $v$ is only $0.001g^2$. This is
the rare-token case. An embedding row for a word seen once in ten thousand batches takes the
largest step any weight in the model can take, at the moment it is least informed. After a
*short* silence the opposite happens: the step is smaller than $\eta$. The correction treats the
zeros as real gradients and divides by a $1-\beta^t$ that is still small.
"""

# %%
def step_after_silence(gap, b1=0.9, b2=0.999, g=1.0, eps=EPS):
    """Gradients are zero for `gap` steps, then one gradient g arrives (bias-corrected Adam)."""
    t = gap + 1
    m, v = (1 - b1) * g, (1 - b2) * g * g                    # zeros contributed nothing
    return (m / (1 - b1 ** t)) / (math.sqrt(v / (1 - b2 ** t)) + eps)


print(f"{'zero-gradient steps before':>27} {'step / η, β2=0.999':>19} {'step / η, β2=0.95':>18}")
for gap in (0, 10, 100, 1000, 10000):
    print(f"{gap:>27} {step_after_silence(gap):19.3f} {step_after_silence(gap, b2=0.95):18.3f}")
check(abs(step_after_silence(10**6) - 0.1 / math.sqrt(0.001)) < 1e-3, "after a long silence the step approaches 3.16 η (β2=0.999)")
check(step_after_silence(10**6, b2=0.95) <= 1.0, "with β2 = 0.95 even the rare-token step stays at or below η")

# %% [markdown]
r"""
**Weight decay: why AdamW exists.** L2 regularization adds $\tfrac12\lambda w^2$ to the loss, so
$\lambda w$ joins the gradient. Inside Adam that term gets divided by $\sqrt{\hat v}$ along with
everything else, so the decay a weight receives depends on its gradient history:
- Two weights at $w = 0.5$, $\lambda = 0.1$, $\eta = 10^{-3}$.
- One has $\sqrt{\hat v} = 1$, the other $\sqrt{\hat v} = 0.01$.
- L2 shrinks them by $5\times10^{-5}$ and $5\times10^{-3}$ per step.

AdamW applies the decay *after* the Adam step, where nothing divides it:
$$w \leftarrow w - \eta\frac{\hat m}{\sqrt{\hat v}+\epsilon} - \eta\lambda w.$$
Both weights then shrink by $\eta\lambda w = 5\times10^{-5}$.

A sharper way to see the L2 problem: with **no** loss gradient, L2-in-Adam turns the decay term
into a sign, so every weight shrinks by $\eta$ whatever its size. Decoupled decay shrinks each
weight in proportion to itself.
"""

# %%
lam = 0.1
for name, sq in (("A", 1.00), ("B", 0.01)):
    print(f"parameter {name}: sqrt(v_hat)={sq:.2f}   L2 shrink {LR1 * lam * 0.5 / sq:.1e}   decoupled shrink {LR1 * lam * 0.5:.1e}")
check(math.isclose(LR1 * lam * 0.5 / 0.01, 5e-3) and math.isclose(LR1 * lam * 0.5, 5e-5),
      "L2 decays the small-gradient weight 100x harder; decoupled decay treats both the same")


def one_decay_step(opt_cls, w):
    p = torch.nn.Parameter(torch.tensor([w], dtype=torch.float64))
    opt = opt_cls([p], lr=LR1, weight_decay=lam, foreach=False)
    p.grad = torch.zeros_like(p)                         # no loss gradient at all: only the decay acts
    opt.step()
    return w - p.item()


for w in (0.5, 0.05):
    print(f"w={w}:  Adam with L2 shrinks by {one_decay_step(torch.optim.Adam, w):.2e}   "
          f"AdamW shrinks by {one_decay_step(torch.optim.AdamW, w):.2e}")
check(all(abs(one_decay_step(torch.optim.Adam, w) - LR1) < 1e-6 for w in (0.5, 0.05)),
      "L2 inside Adam shrinks every weight by η regardless of its size")
check(all(abs(one_decay_step(torch.optim.AdamW, w) - LR1 * lam * w) < 1e-15 for w in (0.5, 0.05)),
      "AdamW shrinks each weight by exactly ηλw")

# %% [markdown]
r"""
> **Carry forward:** Adam moves every weight by about $\eta$, at most $3.16\,\eta$ with
> $\beta_2=0.999$, and it hits that maximum on rare tokens. PyTorch matches the hand computation to
> rounding error, except for a 1e-5 relative wobble in the fused float32 kernel's $v$. Decay
> belongs outside the Adam ratio.
"""
