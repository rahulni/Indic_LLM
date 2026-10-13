# %% [markdown]
r"""
## §R The rules before Adam

Adam combines four earlier ideas. Each fixes a specific failure of the one before it, and each fits
in a few lines of arithmetic. This section reproduces the standard worked numbers for each one,
on a CPU, so the later sections can take them as given.

### R.1 One learning rate, one curvature

Gradient descent moves a weight against its gradient by a fixed fraction:
$$w \leftarrow w - \eta\, g.$$
Take the loss $\mathcal{L} = (w-5)^2$, whose gradient is $2(w-5)$. Let $d = w - 5$ be the distance
still to travel. Substituting the gradient into the update gives
$$d_\text{new} = d\,(1 - 2\eta).$$
Every step multiplies the remaining distance by the same number $1-2\eta$:
- Its **sign** says whether the weight overshot the minimum.
- Its **size** says whether the distance shrank.

So convergence needs $|1 - 2\eta| < 1$. With a general curvature $c$ (gradient $c\,d$) the
multiplier is $1 - \eta c$. The safe learning rate is set by the curvature.
"""

# %%
def gd_path(eta, steps=5, w0=0.0, target=5.0, c=2.0):
    w, out = w0, []
    for _ in range(steps):
        w -= eta * c * (w - target)                     # gradient of (w - target)^2 is c * (w - target), c = 2
        out.append(w)
    return out


print(f"{'eta':>5} {'multiplier':>10}   w over five steps")
for eta in (0.01, 0.10, 0.90, 1.10):
    print(f"{eta:5.2f} {1 - 2 * eta:+10.2f}   " + ", ".join(f"{w:.2f}" for w in gd_path(eta)))
check([round(w, 2) for w in gd_path(0.01)] == [0.10, 0.20, 0.29, 0.39, 0.48], "eta=0.01 crawls: 0.10 … 0.48")
check([round(w, 2) for w in gd_path(0.10)] == [1.00, 1.80, 2.44, 2.95, 3.36], "eta=0.10 converges cleanly")
check([round(w, 2) for w in gd_path(0.90)] == [9.00, 1.80, 7.56, 2.95, 6.64], "eta=0.90 overshoots but converges")
check([round(w, 1) for w in gd_path(1.10)] == [11.0, -2.2, 13.6, -5.4, 17.4], "eta=1.10 diverges")

# %% [markdown]
r"""
A real loss has a different curvature along every direction, and one $\eta$ has to serve all of
them. Take $\mathcal{L} = \tfrac12(20u^2 + v^2)$, starting at $(1, 1)$. Each step multiplies $u$
by $(1-20\eta)$ and $v$ by $(1-\eta)$. An $\eta$ that suits $u$ leaves $v$ almost frozen. An $\eta$
large enough to move $v$ makes $u$ diverge.
"""

# %%
def gd_2d(eta, steps):
    u, v, path = 1.0, 1.0, [(1.0, 1.0)]
    for _ in range(steps):
        u, v = u * (1 - 20 * eta), v * (1 - eta)
        path.append((u, v))
    return np.array(path)


for eta in (0.01, 0.09, 0.11):
    u5, v5 = gd_2d(eta, 5)[-1]
    print(f"eta={eta:.2f}  u multiplier {1 - 20 * eta:+.2f}  v multiplier {1 - eta:.2f}  after 5 steps u={u5:+.3f} v={v5:.3f}")
check(round(gd_2d(0.01, 5)[-1, 1], 3) == 0.951, "eta=0.01: v has barely moved (0.951)")
check(round(gd_2d(0.09, 5)[-1, 1], 3) == 0.624, "eta=0.09: v still slow (0.624) while u overshoots")
check(round(gd_2d(0.11, 5)[-1, 0], 2) == -2.49, "eta=0.11: u has diverged to -2.49")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
uu, vv = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-0.3, 1.2, 200))
ax.contour(uu, vv, 0.5 * (20 * uu**2 + vv**2), levels=np.geomspace(0.02, 60, 14), colors=GRID, linewidths=0.7)
for eta, col in ((0.01, C["blue"]), (0.09, C["aqua"]), (0.11, C["red"])):
    p = gd_2d(eta, 12)
    ax.plot(p[:, 0], p[:, 1], "-o", ms=2.5, color=col, lw=1.1)
    label_end(ax, p[-1, 0], p[-1, 1], f"η={eta}", col)
ax.set(xlim=(-3, 3), ylim=(-0.3, 1.2), xlabel="u (curvature 20)", ylabel="v (curvature 1)",
       title="One learning rate cannot serve two curvatures (12 steps from (1, 1))")
finish(fig, "gd_two_curvatures")

# %% [markdown]
r"""
> **Carry forward:** a single learning rate shared by every weight breaks first, because the weights
> do not share a curvature.

### R.2 Momentum: average what is consistent

A weight whose gradient keeps its sign can safely move further than one whose gradient flips. An
exponential moving average stores one number and remembers about $1/(1-\beta_1)$ recent gradients:
$$m \leftarrow \beta_1 m + (1-\beta_1)\,g, \qquad \beta_1 = 0.9 \;\Rightarrow\; \text{about 10 steps of memory}.$$
Along a steep direction the sign flips every step, so the average cancels. Along a shallow direction
it accumulates.
"""

# %%
def ema(seq, beta=0.9):
    m, out = 0.0, []
    for g in seq:
        m = beta * m + (1 - beta) * g
        out.append(m)
    return out


steep, shallow = ema([1, -1, 1, -1, 1]), ema([0.2] * 5)
for t, (a, b) in enumerate(zip(steep, shallow), 1):
    print(f"step {t}: alternating ±1 → {a:+.3f}    constant 0.2 → {b:.3f}")
check([round(x, 3) for x in steep] == [0.100, -0.010, 0.091, -0.018, 0.084], "alternating gradients cancel")
check([round(x, 3) for x in shallow] == [0.020, 0.038, 0.054, 0.069, 0.082], "a constant gradient accumulates")
check(abs(steep[-1] - shallow[-1]) < 0.003, "after five steps the two averages are nearly equal despite a 5x gradient gap")

# %% [markdown]
r"""
> **Carry forward:** an exponential moving average cancels what alternates and accumulates what is
> consistent.

### R.3 A learning rate per parameter

A second average, this time of the squared gradient, estimates how large a parameter's gradients
usually are:
$$v \leftarrow \beta_2 v + (1-\beta_2)\,g^2, \qquad w \leftarrow w - \frac{\eta}{\sqrt{v}}\, g.$$
The quantity $\eta/\sqrt v$ is a learning rate that belongs to that parameter alone. Dividing by
$\sqrt v$ removes the gradient's magnitude and keeps only its sign and its consistency. A rare token's
embedding row, which gets a gradient once in a thousand batches, still takes a full-sized step
when the gradient arrives.
"""

# %%
def per_param(g, steps=200, b2=0.999, eta=1.0):
    v = 0.0
    for t in range(1, steps + 1):
        v = b2 * v + (1 - b2) * g * g
    v_hat = v / (1 - b2 ** steps)                               # bias-corrected, see §1
    return math.sqrt(v_hat), eta / math.sqrt(v_hat), eta * g / math.sqrt(v_hat)


for name, g in (("A", 1.0), ("B", 0.01)):
    s, lr_own, step = per_param(g)
    print(f"parameter {name}: gradient {g:.4f}  sqrt(v_hat) {s:.4f}  its own learning rate {lr_own:7.2f} η  step {step:.4f} η")
check(abs(per_param(1.0)[2] - per_param(0.01)[2]) < 1e-9, "gradients 100x apart, identical steps")
check(round(per_param(0.01)[1], 2) == 100.00, "the small-gradient parameter gets a 100x larger learning rate")

# %% [markdown]
r"""
### R.4 What the optimizer costs in memory

Mixed-precision training keeps five things per weight:
- the bf16 weight (2 bytes)
- its bf16 gradient (2)
- an fp32 master copy (4)
- Adam's $m$ (4)
- Adam's $v$ (4)

That is 16 bytes, and half of it is optimizer state. The fp32 master copy is there because many
updates are smaller than bf16's spacing between neighbouring numbers, especially once the learning
rate has decayed. §5 measures how many.
"""

# %%
rules = [("gradient descent", 8), ("with momentum", 12), ("AdamW", 16), ("8-bit AdamW", 10), ("Muon (keeps only m)", 12)]
gib = {name: 9e9 * b / 2**30 for name, b in rules}
for name, b in rules:
    print(f"{name:<22} {b:>2} bytes/weight   9B model: {gib[name]:6.1f} GiB")
check([round(gib[n], 1) for n, _ in rules[:4]] == [67.1, 100.6, 134.1, 83.8], "9B model: 67.1 / 100.6 / 134.1 / 83.8 GiB")

# %% [markdown]
r"""
### R.5 Batch size and the learning rate

The mean of $N$ independent samples has $1/\sqrt N$ of a single sample's noise. A larger batch
therefore gives a more accurate gradient, and a more accurate gradient can be trusted over a longer
step:
- For SGD the rule is linear: 4× the batch allows 4× the rate.
- For Adam it is the square root: 4× the batch allows 2× the rate, because Adam has already divided
  out the gradient's magnitude.

Accumulation steps and extra GPUs are the same change to the optimizer:
$$\text{global batch} = \text{micro-batch} \times \text{GPUs} \times \text{accumulation steps}.$$
Both rules stop working past the *critical batch size*, beyond which extra samples buy almost
nothing.
"""

# %%
rng = np.random.default_rng(0)
print(f"{'batch':>6} {'theory sqrt(8/N)':>17} {'measured':>9}")
for N in (8, 32, 56, 128, 512):
    sd = rng.standard_normal((20000, N)).mean(1).std() / rng.standard_normal((20000, 8)).mean(1).std()
    print(f"{N:>6} {math.sqrt(8 / N):17.3f} {sd:9.3f}")
check(abs(math.sqrt(8 / 56) - 0.378) < 5e-4, "a batch of 56 has 0.378x the noise of a batch of 8")
for change, mult in (("accumulation 1 → 4", 4), ("GPUs 8 → 32", 4), ("both together", 16)):
    print(f"{change:<20} global batch x{mult:<3} → Adam learning rate x{math.sqrt(mult):g}")
check(math.sqrt(16) == 4, "16x the global batch allows 4x the learning rate under Adam (square-root rule)")

# %% [markdown]
r"""
### R.6 Muon: orthogonalize the update

Adam scales each number in a weight matrix on its own. A matrix maps input directions to output
directions, though, and its momentum matrix is usually dominated by a few large singular values.
Muon replaces the momentum matrix with the nearest matrix whose singular values are all about 1, so
every direction moves by a similar amount. It does this cheaply, with five Newton–Schulz iterations
instead of an SVD.

Muon applies only to 2-D hidden matrices. Embeddings, norm gains and the output head stay on AdamW.
§M tunes both and compares them.
"""

# %%
def newton_schulz5(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315)):
    a, b, c = coeffs
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


torch.manual_seed(0)
U, _ = torch.linalg.qr(torch.randn(256, 64, dtype=torch.float64))
V, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
s = torch.logspace(0, math.log10(0.02), 64, dtype=torch.float64)          # singular values 1 … 0.02
G = U @ torch.diag(s) @ V.T
s_after = torch.linalg.svdvals(newton_schulz5(G))
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.semilogy(s.numpy(), "o", ms=3, color=C["orange"]); label_end(ax, 63, s[-1].item(), "momentum matrix", C["orange"])
ax.semilogy(s_after.numpy(), "o", ms=3, color=C["aqua"]); label_end(ax, 63, s_after[-1].item(), "after 5 Newton–Schulz steps", C["aqua"])
ax.set(xlabel="singular value index", ylabel="singular value", xlim=(-2, 95),
       title="Muon's update: singular values spanning 50x squeezed toward 1")
finish(fig, "muon_orthogonalize")
print(f"singular values before: {s.max():.3f} … {s.min():.3f} (ratio {s.max() / s.min():.0f}x)")
print(f"after Newton–Schulz:    {s_after.max():.3f} … {s_after.min():.3f} (ratio {s_after.max() / s_after.min():.2f}x)")
check(s_after.min() > 0.5 and s_after.max() < 1.5, "five Newton–Schulz steps bring every singular value into [0.5, 1.5]")

# %% [markdown]
r"""
> **Carry forward:** momentum fixes direction, a per-parameter scale fixes size, the batch sets how
> much the gradient can be trusted, and Muon treats a matrix as a matrix. Adam, next, is the first
> two together.
"""
