from . import md, code, mathbox

CELLS = [
md(r'''
---
## A · Refresher: the core ideas, with intuition, math, and checked numbers

> Each idea gets three things: a plain-language **🧠 intuition**, the **📐 math** (click to open),
> and a code cell that recomputes published numbers, where an `assert` fails if the arithmetic does
> not match. If you have five minutes, read this part.

### A.0 Notation used everywhere below

| symbol | meaning | in this notebook |
|---|---|---|
| $B,\ T,\ N=BT$ | sequences per batch, tokens per sequence, tokens per batch | 64 (2 micro-batches of 32), 256, 16,384 |
| $L,\ d$ | layers, hidden size | 8, 384 |
| $F$ | width (number of neurons) of the dense feed-forward block | 1,536 |
| $E,\ k$ | routed experts per layer, experts chosen per token | 8 → 32, 4 |
| $w,\ w_s$ | width of one routed expert, of the shared expert | 192, 768 |
| $c$ | how many experts contain a copy of each dense neuron | 2 (partition), $E$ (copy) |
| $s$ | route scale: multiplies the renormalised weights | 4 (partition), 1 (copy) |
| $x \in \mathbb{R}^{d}$ | one token's vector entering the feed-forward block | |
| $z = W_r x \in \mathbb{R}^{E}$ | router logits; $W_r$ is $E \times d$ | |
| $\sigma_i$ | score of expert $i$ (softmax, sigmoid or √softplus of $z$) | sigmoid |
| $\mathcal{T}(x)$ | the $k$ chosen experts | |
| $g_i$ | weight of chosen expert $i$ | |
| $b_i$ | balancing bias of expert $i$, used **only** to choose | |
| $\gamma,\ \alpha$ | bias update speed, auxiliary-loss weight | 0.001; 0.01 (Switch), 0.0001 (sequence) |
| $n_i,\ f_i,\ P_i$ | tokens sent to $i$, its share of all (token, expert) picks, its mean router probability | |
| $\eta,\ \ell$ | learning rate, cross-entropy loss in nats | 1.2e-3 peak |

### A.1 Why mixture-of-experts exists

```
STANDARD LAYER                              MIXTURE-OF-EXPERTS LAYER
token                                        token
  │                                            │
  ▼                                            ▼
┌──────────────────────────┐                 router ── scores 128 small networks
│ one feed-forward block   │                   │
│ 6,144 wide               │                   ▼
│ every weight is used     │                 picks 8 of them, each 768 wide
└──────────────────────────┘                   │
                                               ▼
                                             8 × 768 = 6,144 of width is used,
                                             120 small networks sit idle for this token
```

The feed-forward block holds most of a transformer's parameters, and every token uses all of
them. An MoE layer **stores** many small feed-forward networks (*experts*) and **runs** only the few
the router picks for each token. Parameters grow; work per token does not.

🧠 **Intuition.** Think of a hospital. A triage nurse (the router) sends each patient (token) to a
couple of the hospital's specialists (experts). The hospital pays every doctor's salary (memory
grows with *all* parameters), but each patient only takes the time of the few doctors they see
(compute grows with the *active* parameters). Another version: you do not use your whole
brain for every thought. The knowledge stays stored, and only the part you need switches on.
'''),
mathbox("compute follows the active parameters, memory follows the total", r'''
For training, the work per token and the memory are, to a good approximation:

$$\text{FLOPs per token} \;\approx\; 6\,N_{\text{active}}, \qquad \text{training memory} \;\approx\; 16\,N_{\text{total}}\ \text{bytes}.$$

* **6** = 2 (one multiply + one add per weight) × 3 (forward pass, backward pass for the
  activations, backward pass for the weights). This ignores the attention-score term, which grows
  with the sequence length.
* **16 bytes** = 4 (fp32 weight) + 4 (fp32 gradient) + 4 + 4 (Adam's two moments $m$, $v$).
  Activations come on top and scale with the batch.

**Worked example.**

| model | FLOPs per token | training state |
|---|---|---|
| reference MoE | $6 \times 3.35\text{B} = 20.1$ GFLOP | $16 \times 30.53\text{B} = 455$ GiB |
| a 30.2B dense model | 181 GFLOP | 450 GiB |
| our MoE-32 | $6 \times 18.98\text{M} = 114$ MFLOP | $16 \times 68.5\text{M} = 1.02$ GiB |
| our dense model | $6 \times 18.88\text{M} = 113$ MFLOP | 0.28 GiB |

The reference model buys the memory of a 30B model at one ninth of its compute. Our MoE-32 has the
same compute per token as our dense model and 3.6× its memory.
'''),
md(r'''
It creates three problems, and each later part of this notebook touches one of them:

| problem | what goes wrong | where in this notebook |
|---|---|---|
| **choosing** | a router must pick k of E experts per token | C (router), H (score functions) |
| **balancing** | the router falls in love with a few experts; the rest never learn | H (balancing lab), J (growth lab) |
| **placing** | experts live on different GPUs; tokens travel there and back | M (systems view) |
| **growing** | build the MoE from a dense model instead of from scratch | F (surgery), G, J |

### A.2 Terminology

| term | meaning | reference model (Qwen3-30B-A3B shape) | our final model |
|---|---|---|---|
| expert | one small SwiGLU feed-forward network | 128 per layer, width 768 | 32 per layer, width 192 |
| router (gate) | one matrix `d × E` scoring every expert for every token | 2048 × 128 | 384 × 32 |
| top-k | keep the k highest scores | k = 8 | k = 4 |
| shared expert | always on, never routed | none | one, width 768 |
| total params | every stored weight | 30.53B | ~68M |
| active params | weights one token touches | 3.35B | ~19M |
| load | share of tokens an expert receives; even = k / E | 6.25% | 12.5% |
| capacity | max tokens an expert may take per batch (old style) | — | dropless |
'''),
md(r'''
### A.3 Total vs active parameters: the calculator

One function counts any MoE transformer. It reproduces the published figures for the reference model
(Qwen3-30B-A3B: 48 layers, hidden 2,048, 32 query / 4 KV heads of 128, 128 experts of width 768,
top-8, vocabulary 151,936, untied embeddings) and the Mixtral 8x7B figures.

🧠 **Intuition.** An MoE layer is a dense layer whose feed-forward block has been cut into
pieces. Count the attention once, the router once, the shared expert once, and the experts *E*
times for storage but only *k* times for a token.
'''),
mathbox("the parameter count, per layer and per model", r'''
With $n_q$ query heads, $n_{kv}$ key/value heads of size $h$, and SwiGLU experts (three matrices
each: gate and up are $w \times d$, down is $d \times w$):

$$N^{\text{attn}} = d\,n_q h + 2\,d\,n_{kv} h + n_q h\,d, \qquad N^{\text{expert}} = 3\,d\,w, \qquad N^{\text{router}} = d\,E, \qquad N^{\text{shared}} = 3\,d\,w_s$$

$$N_{\text{total}} = L\big(N^{\text{attn}} + E\,N^{\text{expert}} + N^{\text{router}} + N^{\text{shared}}\big) + N^{\text{emb}}$$

$$N_{\text{active}} = L\big(N^{\text{attn}} + k\,N^{\text{expert}} + N^{\text{router}} + N^{\text{shared}}\big) + N^{\text{emb}}$$

The only difference between the two is $E$ versus $k$ in front of the expert term.

**Worked example (our MoE-32).**

| term | count |
|---|---|
| $N^{\text{attn}}$ | $384 \cdot 384 + 2 \cdot 384 \cdot 128 + 384 \cdot 384 = 393{,}216$ |
| $N^{\text{expert}}$ | $3 \cdot 384 \cdot 192 = 221{,}184$ |
| $N^{\text{shared}}$ | $3 \cdot 384 \cdot 768 = 884{,}736$ |
| $N^{\text{router}}$ | $384 \cdot 32 = 12{,}288$ |
| $N^{\text{emb}}$ (tied, counted once) | $4096 \cdot 384 = 1{,}572{,}864$ |

$$N_{\text{total}} = 8\,(393{,}216 + 32 \cdot 221{,}184 + 12{,}288 + 884{,}736) + 1{,}572{,}864 + \underbrace{6{,}528}_{\text{norms}} = 68{,}524{,}416$$

$$N_{\text{active}} = 8\,(393{,}216 + 4 \cdot 221{,}184 + 12{,}288 + 884{,}736) + 1{,}572{,}864 + 6{,}528 = 18{,}979{,}200$$

Both match `n_params(model)` exactly; a gate in Part M checks it.
'''),
code(r'''
def count_moe(L, d, vocab, n_q, n_kv, hd, E, k, w, shared=0, tied=False, dense_ffn=None):
    """Return (total, active) parameter counts, ignoring norms. dense_ffn: count a dense model instead."""
    attn = d * n_q * hd + 2 * d * n_kv * hd + n_q * hd * d
    emb = vocab * d * (1 if tied else 2)
    if dense_ffn:
        per = attn + 3 * d * dense_ffn
        return L * per + emb, L * per + emb
    expert, router, sh = 3 * d * w, d * E, 3 * d * shared
    total = L * (attn + E * expert + router + sh) + emb
    active = L * (attn + k * expert + router + sh) + emb
    return total, active

QWEN = dict(L=48, d=2048, vocab=151_936, n_q=32, n_kv=4, hd=128, E=128, k=8, w=768)
q_tot, q_act = count_moe(**QWEN)
attn_layer = 2048 * 32 * 128 * 2 + 2 * 2048 * 4 * 128
print(f"Qwen3-30B-A3B shape: total {q_tot / 1e9:.2f}B, active {q_act / 1e9:.2f}B, "
      f"attention {attn_layer / 1e6:.2f}M/layer, one expert {3 * 2048 * 768 / 1e6:.2f}M")
assert round(q_tot / 1e9, 2) == 30.53 and round(q_act / 1e9, 2) == 3.35
assert round(attn_layer / 1e6, 2) == 18.87

GiB = 2**30
print(f"training state at 16 bytes/param: MoE {q_tot * 16 / GiB:.0f} GiB vs a 30.2B dense model {30.2e9 * 16 / GiB:.0f} GiB")
print(f"work per token, 6 x active: MoE {6 * q_act / 1e9:.1f} GFLOP vs dense {6 * 30.2e9 / 1e9:.0f} GFLOP")
assert round(q_tot * 16 / GiB) == 455 and round(30.2e9 * 16 / GiB) == 450
assert round(6 * q_act / 1e9, 1) == 20.1 and round(6 * 30.2e9 / 1e9) == 181

kv = 2 * 4 * 128 * 2 * 48                       # K and V, 4 heads, 128 dims, 2 bytes, 48 layers
print(f"KV cache: {kv / 1024:.0f} KiB/token with GQA (4 KV heads), {kv * 8 / 1024:.0f} KiB with 32 heads; "
      f"{kv * 32768 / GiB:.0f} GiB vs {kv * 8 * 32768 / GiB:.0f} GiB for a 32K sequence")
assert kv / 1024 == 96 and kv * 32768 / GiB == 3

m_tot, m_act = count_moe(L=32, d=4096, vocab=32_000, n_q=32, n_kv=8, hd=128, E=8, k=2, w=14_336)
m_exp = 32 * 8 * 3 * 4096 * 14_336
print(f"Mixtral 8x7B: total {m_tot / 1e9:.1f}B (not 8 x 7 = 56B), active {m_act / 1e9:.1f}B, experts hold {m_exp / m_tot:.1%}")
assert round(m_tot / 1e9) == 47 and round(m_act / 1e9) == 13 and round(m_exp / m_tot, 2) == 0.97
'''),
md(r'''
**The E × k sweep.** Same layer, change how many experts it stores and how many each
token uses. The active size changes only with *k* (down a column it is constant), and the total
grows with *E* (across a row it is constant).
'''),
code(r'''
rows = ["| experts E \\ chosen k | " + " | ".join(f"k={k}" for k in (2, 4, 8, 16)) + " |", "|---|---|---|---|---|"]
for E in (32, 64, 128, 256):
    cells = []
    for k in (2, 4, 8, 16):
        t, a = count_moe(**{**QWEN, "E": E, "k": k})
        cells.append(f"{t / 1e9:.1f}B total · {a / 1e9:.2f}B active")
    rows.append(f"| **{E}** | " + " | ".join(cells) + " |")
display(Markdown("\n".join(rows)))
assert round(count_moe(**{**QWEN, "E": 256})[0] / 1e9) == 60          # "256 per layer, ~60B"
'''),
md(r'''
### A.4 The router, worked by hand

Eight experts, k = 2, one token whose router logits are `[2.0, 0.5, 1.2, −0.3, 0.9, 1.8, −1.0, 0.1]`.
Softmax makes the experts compete; sigmoid scores each one on its own. **Both pick the same two
experts** (0 and 5). They differ in how strongly the router's preference survives into the weights:
softmax keeps a 0.55/0.45 split, sigmoid flattens it to 0.507/0.493.

🧠 **Intuition.** The router is a panel of judges scoring every expert for this token. The
*choice* is decided by rank alone: who is in the top *k*. The *weights* decide how loudly each
chosen expert is heard.

* **Softmax is a ranked election.** The scores share a fixed budget of 1, so one point of logit gap
  always means the same ratio (×2.7), however large the logits are.
* **Sigmoid gives each expert an independent pass mark** between 0 and 1. Two confident passes, 0.88
  and 0.86, look almost the same, so after renormalising the router's preference is flattened:
  sigmoid all but erases the router's ranking among confident experts.
* **The balancing bias** is a thumb on the scale *for the choice only*. It can move an expert
  in or out of the top *k*, but the weights still come from the honest scores.
'''),
mathbox("scores, the choice, the weights, and why top-k has no gradient", r'''
**Scores.** With logits $z = W_r x$:

$$\text{softmax: } \sigma_i = \frac{e^{z_i}}{\sum_{j=1}^{E} e^{z_j}}, \qquad \text{sigmoid: } \sigma_i = \frac{1}{1 + e^{-z_i}}, \qquad \text{DeepSeek-V4: } \sigma_i = \sqrt{\operatorname{softplus}(z_i)}$$

**The choice and the weights.** The bias $b_i$ enters only the choice:

$$\mathcal{T}(x) = \operatorname{top}_k\big(\sigma_i + b_i\big), \qquad g_i = s\,\frac{\sigma_i}{\sum_{j \in \mathcal{T}} \sigma_j} \quad (i \in \mathcal{T}), \qquad y = \sum_{i \in \mathcal{T}} g_i\,E_i(x)$$

**Renormalised softmax keeps the gaps.** The softmax denominator over all $E$ experts cancels:

$$\frac{\sigma_i}{\sum_{j \in \mathcal{T}} \sigma_j} = \frac{e^{z_i}}{\sum_{j \in \mathcal{T}} e^{z_j}} \quad\Longrightarrow\quad \frac{g_a}{g_b} = e^{\,z_a - z_b}.$$

So the renormalised weights are just a softmax over the chosen logits, and only the logit *gap*
matters.

**Sigmoid flattens them.** For sigmoid, $g_a / g_b = \sigma(z_a)/\sigma(z_b)$. Since $\sigma \to 1$ for large
logits, this ratio goes to 1 as both logits grow, even when the gap stays fixed.

In the worked example the gap is $z_0 - z_5 = 0.2$:

| | ratio $g_0/g_5$ | weights |
|---|---|---|
| softmax | $e^{0.2} = 1.22$ | 0.55 / 0.45 |
| sigmoid | $0.881 / 0.858 = 1.027$ | 0.507 / 0.493 |

**Why top-k has no gradient.** $\mathcal{T}$ is piecewise constant in $z$: nudging a logit does not
change the chosen set, except exactly at a tie. So $\partial\mathcal{T}/\partial z = 0$ almost
everywhere, and the router learns only through the weights:

$$\frac{\partial \ell}{\partial z_j} = \sum_{i \in \mathcal{T}} \frac{\partial \ell}{\partial g_i}\,\frac{\partial g_i}{\partial z_j}, \qquad \frac{\partial \ell}{\partial g_i} = \Big\langle \frac{\partial \ell}{\partial y},\; E_i(x) \Big\rangle.$$

An expert whose output points downhill on the loss gets its weight, and with it its logit, pushed
up.

With $k = 1$ and renormalisation, $g \equiv s$ is a constant, so $\partial g / \partial z = 0$ and the router
never learns. That is why Switch Transformer used the *unnormalised* probability as its single
weight.
'''),
code(r'''
z = torch.tensor([2.0, 0.5, 1.2, -0.3, 0.9, 1.8, -1.0, 0.1])
for name, fn in (("softmax", lambda v: torch.softmax(v, 0)), ("sigmoid", torch.sigmoid)):
    s = fn(z)
    top = s.topk(2).indices
    w = s[top] / s[top].sum()
    print(f"{name:<8} scores {[round(v, 3) for v in s.tolist()]}  ->  experts {top.tolist()}  weights {[round(v, 3) for v in w.tolist()]}")
sm, sg = torch.softmax(z, 0), torch.sigmoid(z)
assert torch.equal(sm.topk(2).indices, sg.topk(2).indices)
assert [round(v, 2) for v in (sm[[0, 5]] / sm[[0, 5]].sum()).tolist()] == [0.55, 0.45]
assert [round(v, 3) for v in (sg[[0, 5]] / sg[[0, 5]].sum()).tolist()] == [0.507, 0.493]
assert torch.allclose(sm[[0, 5]] / sm[[0, 5]].sum(), torch.softmax(z[[0, 5]], 0))      # the identity above

# Loss-free balancing: a bias of -0.5 on expert 5 changes the CHOICE, not the weights.
bias = torch.zeros(8); bias[5] = -0.5
top_b = (sg + bias).topk(2).indices
w_b = sg[top_b] / sg[top_b].sum()
print(f"with bias -0.5 on expert 5: selection score {sg[5] + bias[5]:.3f} -> experts {top_b.tolist()}, "
      f"weights from the ORIGINAL scores {[round(v, 3) for v in w_b.tolist()]}")
assert sorted(top_b.tolist()) == [0, 2] and round((sg[5] + bias[5]).item(), 3) == 0.358

# Probabilistic top-k (the Lightning LM growth fix): sample k without replacement.
g = torch.Generator().manual_seed(0)
draws = torch.stack([(sg + -torch.log(-torch.log(torch.rand(8, generator=g)))).topk(2).indices for _ in range(2000)])
freq = torch.bincount(draws.flatten(), minlength=8) / 2000
print("Gumbel top-2, how often each expert is picked:", [f"{v:.0%}" for v in freq.tolist()])
'''),
md(r'''
**See the math: how much of the router's preference survives.** Two chosen experts *a* and *b*
with a logit gap of Δ. The plot shows the weight the leader gets. Softmax gives it
$1/(1+e^{-\Delta})$ wherever the logits sit. Sigmoid's split depends on *where* the logits are: the
higher they sit, the closer to 50/50, whatever the gap.
'''),
code(r'''
gap = torch.linspace(0, 3, 121)
fig, ax = plt.subplots(figsize=(7.5, 3.3))
ax.plot(gap, torch.sigmoid(gap), color=SERIES[4], label="softmax (any base logit)")
for base, color in ((0.0, SERIES[1]), (2.0, SERIES[3]), (4.0, SERIES[6])):
    sa, sb = torch.sigmoid(base + gap), torch.sigmoid(torch.tensor(base))
    ax.plot(gap, sa / (sa + sb), color=color, label=f"sigmoid, lower logit = {base:g}")
ax.axhline(0.5, color=MUTED, lw=1)
ax.set(xlabel="logit gap between the two chosen experts", ylabel="weight of the leader", ylim=(0.45, 1.0),
       title="Renormalised weight of the higher-scoring expert")
ax.legend(fontsize=8.5)
savefig(fig, "demo_softmax_vs_sigmoid")
# checks: softmax depends only on the gap; sigmoid at base 4 and gap 1 is nearly even
assert torch.allclose(torch.softmax(torch.tensor([5.0, 4.0]), 0)[0], torch.sigmoid(torch.tensor(1.0)))
assert abs(float(torch.sigmoid(torch.tensor(5.0)) / (torch.sigmoid(torch.tensor(5.0)) + torch.sigmoid(torch.tensor(4.0)))) - 0.5) < 0.005
'''),
md(r'''
### A.5 Why many small experts: the combinations

Fine-grained experts (DeepSeekMoE, 2024): split each expert into *m* smaller ones and choose *m*
times as many. Compute stays the same; the number of ways to assemble a token's network explodes.

🧠 **Intuition.** It is the same amount of LEGO cut into smaller bricks: the same plastic, far more
shapes. A few big experts must each hold very different knowledge; many small ones can be combined
per token.
'''),
mathbox("compute is unchanged, combinations explode", r'''
A token can use $\binom{E}{k}$ distinct sets of experts. Split every expert into $m$ pieces of width
$w/m$, and choose $mk$ of the $mE$ pieces:

$$\underbrace{mk \cdot 3d\frac{w}{m}}_{\text{active FFN params}} = k \cdot 3dw \quad\text{(unchanged)}, \qquad \binom{mE}{mk} \gg \binom{E}{k}.$$

**Worked example.** $\binom{16}{2} = 120$. Split by 4: $\binom{64}{8} = 4{,}426{,}165{,}368$.

Ours: Stage 2 has $\binom{8}{4} = 70$ sets, Stage 3 has $\binom{32}{4} = 35{,}960$.

Part F runs the exact version of this identity in reverse, as the "fine-grained split" gate:
cutting each expert into 4 and raising $k$ and $s$ by 4 gives identical logits.
'''),
code(r'''
for E, k, label in ((8, 2, "Mixtral, 8 choose 2"), (16, 2, "16 choose 2"), (64, 8, "split x4: 64 choose 8"),
                    (128, 8, "reference model, 128 choose 8"), (8, 4, "our Stage 2, 8 choose 4"), (32, 4, "our Stage 3, 32 choose 4")):
    print(f"{label:<32} {math.comb(E, k):>22,}")
assert math.comb(8, 2) == 28 and math.comb(16, 2) == 120
assert math.comb(128, 8) == 1_429_702_652_400 and math.comb(64, 8) == 4_426_165_368
'''),
md(r'''
### A.6 Capacity, auxiliary loss, MaxVio, and the cost of moving tokens

🧠 **Intuition.**

* **Capacity** is a restaurant where every waiter (expert) has a fixed number of tables. Guests who
  arrive after their waiter's tables are full are turned away: the token skips that expert and only
  the residual carries it forward.
* **The auxiliary loss** is a fine on the router: the busier an expert already is, the more it costs
  to send it more probability. It acts through the router's own gradient, so it competes with the
  language loss for the same weights, like a manager who must also keep an auditor happy.
* **MaxVio** is one number for "how overloaded is the busiest expert": 0 means perfectly even, and 1
  means the busiest has twice the average.
'''),
mathbox("capacity, the auxiliary loss and its minimum, MaxVio, and expert-parallel traffic", r'''
**Capacity and dropping.** With $N$ tokens, $k$ choices and $E$ experts, the average load is $Nk/E$.
The capacity and the fraction dropped are:

$$C = \Big\lceil \frac{Nk}{E}\cdot \mathrm{CF} \Big\rceil, \qquad \text{dropped} = \frac{1}{Nk}\sum_{i=1}^{E} \max(0,\; n_i - C).$$

Reference example: $8192 \cdot 8 / 128 = 512$, and $\times 1.25 = 640$ slots.

**The Switch auxiliary loss.** With $f_i$ the fraction of (token, expert) picks that go to expert $i$ and $P_i$
the mean router probability of $i$:

$$\mathcal{L}_{\text{aux}} = \alpha\,E \sum_{i=1}^{E} f_i P_i, \qquad f_i = \frac{1}{Nk}\sum_{t} \mathbb{1}[i \in \mathcal{T}_t], \qquad P_i = \frac{1}{N}\sum_t \tilde\sigma_{t,i}$$

Here $\tilde\sigma$ is the score normalised over all $E$ experts.

* $f$ is a count, so it has no gradient.
* $\partial \mathcal{L}_{\text{aux}} / \partial P_i = \alpha E f_i$: the busier expert $i$ is, the harder
  its probability is pushed down.

**Why balance is the minimum.** When $f$ tracks $P$, Cauchy–Schwarz gives

$$\sum_i P_i^2 \;\ge\; \frac{(\sum_i P_i)^2}{E} = \frac{1}{E} \quad\Longrightarrow\quad \mathcal{L}_{\text{aux}} \ge \alpha,$$

with equality only at uniform load. Fully collapsed ($f = P = $ one expert) gives $\alpha E$. With
$\alpha = 0.01$ and $E = 128$, that is 0.01 against 1.28.

**MaxVio**, with $n_i$ the tokens sent to expert $i$ and $\bar n$ their mean:

$$\mathrm{MaxVio} = \frac{\max_i n_i - \bar n}{\bar n}.$$

**Expert-parallel traffic** for one sequence of $T$ tokens over $L$ layers on $G$ GPUs:

$$\text{bytes} = \underbrace{2}_{\text{dispatch + combine}} \cdot \underbrace{k\,d \cdot 2}_{\text{bf16 copies}} \cdot T \cdot \underbrace{\big(1 - \tfrac{1}{G}\big)}_{\text{leave the GPU}} \cdot L \cdot \underbrace{2}_{\text{fwd + bwd}}$$

Reference model: $2 \cdot 32{,}768 \cdot 8192 \cdot \tfrac78 \cdot 48 \cdot 2 = 45.1$ GB per sequence.
'''),
code(r'''
cap = 8192 * 8 / 128
print(f"capacity: one 8,192-token sequence -> {cap:.0f} tokens per expert on average, x1.25 -> {cap * 1.25:.0f} slots")
assert cap == 512 and cap * 1.25 == 640

def switch_aux(f, P, alpha=0.01):
    return alpha * len(f) * float((f * P).sum())
even = torch.full((128,), 1 / 128)
one = torch.zeros(128); one[0] = 1.0
print(f"Switch aux loss (alpha 0.01, 128 experts): even {switch_aux(even, even):.2f}, collapsed {switch_aux(one, one):.2f}")
assert round(switch_aux(even, even), 2) == 0.01 and round(switch_aux(one, one), 2) == 1.28
rnd = torch.distributions.Dirichlet(torch.ones(128)).sample()
assert switch_aux(rnd, rnd) >= switch_aux(even, even) - 1e-9             # Cauchy-Schwarz: uniform is the minimum

def maxvio(load):
    load = torch.as_tensor(load, dtype=torch.float64)
    return float((load.max() - load.mean()) / load.mean())
print("MaxVio: even", maxvio([1, 1, 1, 1]), "| busiest has twice the average", maxvio([2, 1, 1, 0]))

bytes_tok = 8 * 2048 * 2                                    # k copies x d x 2 bytes
traffic = 2 * bytes_tok * 8192 * 7 / 8 * 48 * 2             # dispatch+combine, 7/8 leave the GPU, 48 layers, fwd+bwd
flops = 6 * 3.35e9 * 8192
print(f"expert-parallel traffic for one 8,192-token sequence: {traffic / 1e9:.1f} GB -> NVLink5 {traffic / 900e9:.3f}s, "
      f"NVLink4 {traffic / 450e9:.3f}s, network card {traffic / 50e9:.2f}s; compute {flops / 1e12:.1f} TFLOP = {flops / 2.25e15:.3f}s on a B200")
assert round(traffic / 1e9, 1) == 45.1 and round(flops / 2.25e15, 3) == 0.073
print(f"Lightning LM: chance a neuron is in none of 20 random half-experts = (1/2)^20 = {0.5 ** 20:.2e}  (one in a million)")
'''),
md(r'''
**Carry this forward:** total parameters set the memory, active parameters set the compute. The
router picks with `scores + bias` and weights with `scores`. Top-k itself has no gradient, so the
router learns only through the weights. Every published number used here can be recomputed from
the shapes, and here it was.
'''),
]
