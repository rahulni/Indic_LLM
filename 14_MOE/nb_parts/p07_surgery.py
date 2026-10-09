from . import md, code, mathbox

CELLS = [
md(r'''
---
## F · Conversion surgery: turning the dense block into experts

> Growing an MoE from a dense model, and three ways to seed the shared expert (crop, random, half
> fresh).

A SwiGLU block is a **sum over its neurons**. Neuron *j* computes
`w_down[:, j] · silu(w_gate[j] · x) · (w_up[j] · x)`, and the block's output is the sum of all *F*
of these. Any partition of the neurons into groups gives "experts" whose outputs **add back up to the
dense block exactly**. Everything in this section follows from that one fact.

```
dense SwiGLU, F = 1536 neurons          our Stage 2 (partition)
w_gate [1536, 384] ─┐                    shared expert  : 768 neurons, copied exactly   [768, 384]
w_up   [1536, 384] ─┼─ permute ───────►  2 shuffled copies of the other 768,
w_down [384, 1536] ─┘                     each cut into 4 slices of 192 -> 8 experts  [8, 192, 384]
                                         router (new) [8, 384]  ·  top-4  ·  route_scale = 4
active width = 768 shared + 4 x 192 routed = 1536 = the dense block
```

| method | experts start as | k | route_scale | what is preserved at conversion |
|---|---|---|---|---|
| **copy** (sparse upcycling, 2022) | 8 full copies of the dense block | 2 | 1 (average copies) | **exactly** the dense output, for any router |
| **drop** r=0.5 (drop-upcycling, 2025) | 8 copies, each with half its neurons redrawn | 2 | 1 | about half the signal |
| **partition** (Qwen2 / Lightning LM, our Stage 2) | shared half + slices of 2 shuffled copies of the rest | 4 | 4 (sum pieces) | **the expected value**: each routed neuron is chosen once per token on average |
| **random** | fresh random experts, attention and embeddings kept | 4 | 4 | nothing in the feed-forward block |

**The optimizer state follows the neurons.** Adam keeps two running moments per weight. When a
neuron is copied into an expert, its two moments are copied with it, by the exact same index
operation. Redrawn neurons get zero moments, and the new router starts with no state.
Throwing all of this away and resetting is the common shortcut; carrying it over is part of doing
the conversion properly.

🧠 **Intuition.** A dense feed-forward block is a choir of 1,536 singers, and its output is
everyone singing together.

* **Copy** hires 8 identical choirs and lets the router pick 2. Any pick sounds exactly like the
  original choir, but the choirs start identical and must learn to differ.
* **Partition** splits the singers into small groups and lets each token call a few groups. On
  average every singer is heard once, but for one particular token some singers are heard twice and
  some not at all. That per-token unevenness is the loss jump at conversion.
* **Drop** sends half of each group home and hires random newcomers, so half the song is lost at
  first.
'''),
mathbox("an FFN is a sum over neurons; route scale s = E/c; why copy is exact and partition only on average", r'''
**A feed-forward block is a sum over its neurons.** With $g_j, u_j$ the $j$-th rows of gate and up,
and $W^{\text{down}}_{:,j}$ the $j$-th column of down:

$$\operatorname{FFN}(x) = \sum_{j=1}^{F} \underbrace{W^{\text{down}}_{:,j}\;\operatorname{SiLU}(g_j^\top x)\,(u_j^\top x)}_{\text{neuron } j}.$$

Any grouping of neurons into experts therefore adds back up to the dense block exactly, if every
group is used once.

**Where the route scale comes from.**

* Suppose every dense neuron is placed in $c$ of the $E$ experts, and a token picks $k$ experts.
* Right after conversion the router is near-uniform, so each chosen expert gets weight
  $\approx s/k$, and a given expert is chosen with probability $k/E$.
* Let $M_j$ be the number of chosen experts that contain neuron $j$. Neuron $j$'s coefficient in the
  output is $\tfrac{s}{k} M_j$.
* $\mathbb{E}[M_j] = c\,\tfrac{k}{E}$.

So

$$\mathbb{E}\big[\text{coefficient}_j\big] = \frac{s}{k}\cdot\frac{ck}{E} = \frac{sc}{E} = 1 \quad\Longleftrightarrow\quad s = \frac{E}{c}.$$

One formula covers both methods:

| method | $c$ | $s$ | why |
|---|---|---|---|
| **copy** | $E$ (every expert holds every neuron) | 1 | the outputs are averaged |
| **partition** (ours) | 2 | $8/2 = 4 = k$ | the outputs are summed, so the scale is $k$ |

DeepSeek-V3's "routed scaling factor 2.5" is a constant of this kind.

**Why copy is exact.** Every chosen expert contains neuron $j$, so $M_j = k$ always. The coefficient
is $\sum_{i \in \mathcal{T}} g_i = 1$ exactly, for *any* router.

**Why partition is only right on average.**

* Neuron $j$ sits in $K = 2$ specific experts out of $E = 8$, and $k = 4$ are chosen.
* So $M_j$ is hypergeometric: $M_j \sim \operatorname{Hypergeometric}(E{=}8,\, K{=}2,\, k{=}4)$.

$$P(M_j{=}0) = \frac{\binom{6}{4}}{\binom{8}{4}} = \frac{15}{70}, \qquad P(M_j{=}1) = \frac{\binom21\binom63}{\binom84} = \frac{40}{70}, \qquad P(M_j{=}2) = \frac{\binom62}{\binom84} = \frac{15}{70}$$

$$\mathbb{E}[M_j] = 1, \qquad \operatorname{Var}[M_j] = k\,\frac{K}{E}\Big(1 - \frac{K}{E}\Big)\frac{E - k}{E - 1} = \frac{3}{7} \approx 0.43.$$

Per token, each routed neuron is used 0, 1 or 2 times: right on average, noisy for any one token.
That noise is the partition jump in the table below.

**Drop.** Redrawing a fraction $r$ of each expert's neurons keeps an expected $1 - r$ of the signal
($r = 0.5$ here), and the random neurons add noise. That is why it jumps the most.

**The optimizer state follows the neurons.** Adam's moments update element by element:

$$m \leftarrow \beta_1 m + (1 - \beta_1)\,g, \qquad v \leftarrow \beta_2 v + (1 - \beta_2)\,g^2.$$

Moving neurons is an index map $\pi$, and element-wise updates commute with it. So $(\pi(m), \pi(v))$
are exactly the moments that neuron had: its next update is the one it would have had for the same
gradient.

The bias correction $\hat m = m / (1 - \beta_1^t)$ is about $m$ at $t \approx 1500$. Zeroed moments on
redrawn neurons give them a gentle warm-up: the first steps are scaled down by
$(1-\beta_1)/\sqrt{1-\beta_2} \approx 0.45$.
'''),
md(r'''
**See the math: how often is each dense neuron used, per token?** Simulate uniform routing with
the Stage 2 shape: 8 experts, choose 4, each neuron in 2 experts. For copy, every neuron is used
exactly once (coefficient 1); for partition the coefficient is 0, 1 or 2 with the hypergeometric
probabilities derived above.
'''),
code(r'''
g_demo = torch.Generator().manual_seed(0)
picks = torch.stack([torch.randperm(8, generator=g_demo)[:4] for _ in range(20_000)])   # uniform routing, k=4 of E=8
M = ((picks == 0) | (picks == 4)).sum(1)            # neuron j lives in expert 0 (copy 0) and expert 4 (copy 1)
emp = torch.bincount(M, minlength=3).float() / len(M)
pmf = torch.tensor([15, 40, 15]) / 70
print(f"partition: P(neuron used 0/1/2 times) sampled {[round(v, 3) for v in emp.tolist()]}, "
      f"hypergeometric {[round(v, 3) for v in pmf.tolist()]}; mean {M.float().mean():.3f}, var {M.float().var():.3f} (3/7 = {3 / 7:.3f})")
assert torch.allclose(emp, pmf, atol=0.012)
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.bar([-0.16, 0.84, 1.84], pmf, width=0.3, color=MOE_C, label="partition (c = 2, s = 4)")
ax.bar([1.16], [1.0], width=0.3, color=SERIES[2], label="copy (c = E, s = 1)")
ax.set(xticks=[0, 1, 2], xlabel="coefficient of one dense neuron in one token's output",
       ylabel="probability", title="Copy is exact; partition is right on average", ylim=(0, 1.08))
ax.legend(fontsize=8.5)
savefig(fig, "demo_partition_noise")
'''),
code(r'''
CARRY_ADAM_STATE = True

class W3:
    """A weight and its two Adam moments. Every surgery op is applied to all three identically."""
    __slots__ = ("w", "m", "v")
    def __init__(self, w, m, v):
        self.w, self.m, self.v = w, m, v
    def map(self, fn):
        return W3(fn(self.w), fn(self.m), fn(self.v))
    @staticmethod
    def stack(xs):
        return W3(torch.stack([x.w for x in xs]), torch.stack([x.m for x in xs]), torch.stack([x.v for x in xs]))


def w3_of(p, opt):
    st = opt.state.get(p, {}) if opt is not None else {}
    if CARRY_ADAM_STATE and "exp_avg" in st:
        return W3(p.detach(), st["exp_avg"], st["exp_avg_sq"])
    return W3(p.detach(), torch.zeros_like(p), torch.zeros_like(p))


class Bundle:
    """A set of feed-forward neurons: rows of gate and up, the matching columns of down."""
    def __init__(self, gate, up, down):
        self.gate, self.up, self.down = gate, up, down

    @property
    def n(self):
        return self.gate.w.shape[0]

    def take(self, idx):
        return Bundle(self.gate.map(lambda t: t[idx]), self.up.map(lambda t: t[idx]), self.down.map(lambda t: t[:, idx]))

    def clone(self):
        return Bundle(self.gate.map(torch.clone), self.up.map(torch.clone), self.down.map(torch.clone))

    def redraw(self, frac, gen, std):
        """Drop-upcycling: re-initialise a random `frac` of the neurons (weights ~ N(0, std), moments 0)."""
        b = self.clone()
        k = int(round(frac * self.n))
        idx = torch.randperm(self.n, generator=gen)[:k].to(b.gate.w.device)
        d = b.gate.w.shape[1]
        for t, s, rows in ((b.gate, std[0], True), (b.up, std[1], True), (b.down, std[2], False)):
            shape = (k, d) if rows else (d, k)
            fresh = (torch.randn(shape, generator=gen, dtype=torch.float64) * s).to(t.w)
            if rows:
                t.w[idx], t.m[idx], t.v[idx] = fresh, 0.0, 0.0
            else:
                t.w[:, idx], t.m[:, idx], t.v[:, idx] = fresh, 0.0, 0.0
        return b


def dense_bundle(block, opt):
    f = block.ffn
    return Bundle(w3_of(f.w_gate.weight, opt), w3_of(f.w_up.weight, opt), w3_of(f.w_down.weight, opt))


def expert_bundles(block, opt):
    f = block.ffn
    G, U, D = w3_of(f.gate, opt), w3_of(f.up, opt), w3_of(f.down, opt)
    return [Bundle(G.map(lambda t: t[e]), U.map(lambda t: t[e]), D.map(lambda t: t[e])) for e in range(f.E)]


def bundle_std(b):
    return (b.gate.w.std().item(), b.up.w.std().item(), b.down.w.std().item())


def adam_step(opt):
    for st in opt.state.values():
        if "step" in st:
            return st["step"]
    return None


def assemble(src, src_opt, moe_cfg, layers):
    """Build an MoE model: copy every non-FFN tensor from src, install the per-layer FFN pieces,
    and hand each new parameter the optimizer state that came with its pieces."""
    dev, dt = src.emb.weight.device, src.emb.weight.dtype
    new = GPT(src.c, moe_cfg).to(dev, dt)
    src_sd = src.state_dict()
    with torch.no_grad():
        for k, v in new.state_dict().items():
            if ".ffn." not in k and k in src_sd:
                v.copy_(src_sd[k])
    opt = make_opt(new) if src_opt is not None else None
    step = adam_step(src_opt) if src_opt is not None else None
    carried = []                                       # (param, W3) pairs that bring state along
    if opt is not None and step is not None:
        src_params = dict(src.named_parameters())
        for n, p in new.named_parameters():
            st = src_opt.state.get(src_params.get(n))
            if ".ffn." not in n and st and CARRY_ADAM_STATE:
                opt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    with torch.no_grad():
        for blk, L in zip(new.blocks, layers):
            f = blk.ffn
            if L.get("experts"):
                G = W3.stack([b.gate for b in L["experts"]]); U = W3.stack([b.up for b in L["experts"]])
                D = W3.stack([b.down for b in L["experts"]])
                for p, w3 in ((f.gate, G), (f.up, U), (f.down, D)):
                    p.copy_(w3.w); carried.append((p, w3))
            if L.get("shared") is not None:
                s = L["shared"]
                for p, w3 in ((f.shared.w_gate.weight, s.gate), (f.shared.w_up.weight, s.up), (f.shared.w_down.weight, s.down)):
                    p.copy_(w3.w); carried.append((p, w3))
            if L.get("router") is not None:
                f.router.copy_(L["router"].w); carried.append((f.router, L["router"]))
            for buf in ("bias", "offset"):
                if L.get(buf) is not None:
                    getattr(f, buf).copy_(L[buf])
    if opt is not None and step is not None and CARRY_ADAM_STATE:
        for p, w3 in carried:
            if w3.m.abs().sum() > 0 or w3.v.abs().sum() > 0:
                opt.state[p] = {"step": step.clone() if torch.is_tensor(step) else step,
                                "exp_avg": w3.m.clone().contiguous(), "exp_avg_sq": w3.v.clone().contiguous()}
    return new, opt


MAIN_MOE8 = MoEConfig(n_exp=8, width=192, top_k=4, shared=768)

def convert(src, src_opt, method, seed=0, shared_init="random", **overrides):
    """Dense -> MoE. Returns (model, optimizer, info). overrides: MoEConfig fields (score, balance, ...)."""
    gen = torch.Generator().manual_seed(seed)
    F_, layers = src.c.ffn, []
    if method in ("copy", "drop"):
        cfg = MoEConfig(n_exp=8, width=F_, top_k=2, shared=0, route_scale=1.0)
    elif method == "partition_noshared":
        cfg = MoEConfig(n_exp=16, width=F_ // 8, top_k=8, shared=0)
    else:                                              # "partition", "random"
        cfg = copy.deepcopy(MAIN_MOE8)
        cfg.width, cfg.shared = F_ // 8, F_ // 2
    for k, v in overrides.items():
        setattr(cfg, k, v)
    for blk in src.blocks:
        db = dense_bundle(blk, src_opt)
        std = bundle_std(db)
        if method == "copy":
            layers.append({"experts": [db.clone() for _ in range(cfg.n_exp)]})
        elif method == "drop":
            layers.append({"experts": [db.redraw(0.5, gen, std) for _ in range(cfg.n_exp)]})
        elif method == "random":
            layers.append({})
        else:
            perm = torch.randperm(F_, generator=gen).to(db.gate.w.device)
            if method == "partition_noshared":
                S, Rr = perm[:0], perm
            elif shared_init == "crop":
                S, Rr = torch.arange(cfg.shared, device=perm.device), torch.arange(cfg.shared, F_, device=perm.device)
            else:
                S, Rr = perm[:cfg.shared], perm[cfg.shared:]
            shared = db.take(S) if cfg.shared else None
            if shared is not None and shared_init == "half_fresh":
                shared = shared.redraw(0.5, gen, std)
            copies = cfg.n_exp * cfg.width // len(Rr)
            experts = []
            for _ in range(copies):
                Rc = Rr[torch.randperm(len(Rr), generator=gen).to(Rr.device)]
                experts += [db.take(Rc[j * cfg.width:(j + 1) * cfg.width]) for j in range(len(Rr) // cfg.width)]
            layers.append({"experts": experts, "shared": shared})
    model, opt = assemble(src, src_opt, cfg, layers)
    return model, opt, {"method": method, "cfg": asdict(cfg)}


def grow(src, src_opt, mode, m=4, seed=0, noise=0.01):
    """MoE with E experts -> MoE with E*m experts.
    mode: "copy" | "drop" (redraw half of every clone) | "staggered" (exact copies + clone offsets)
          | "split" (cut each expert into m pieces, k*m chosen: exact, but no growth in total size)."""
    gen = torch.Generator().manual_seed(seed)
    old = src.moe_layers()[0].cfg
    cfg = copy.deepcopy(old)
    if mode == "split":
        cfg.n_exp, cfg.width, cfg.top_k = old.n_exp * m, old.width // m, old.top_k * m
        cfg.route_scale = (old.route_scale or old.top_k) * m
    else:
        cfg.n_exp, cfg.family = old.n_exp * m, m
        cfg.route_scale = old.route_scale or old.top_k
    layers, deltas = [], []
    for blk in src.blocks:
        f = blk.ffn
        exps = expert_bundles(blk, src_opt)
        shared = (Bundle(w3_of(f.shared.w_gate.weight, src_opt), w3_of(f.shared.w_up.weight, src_opt),
                         w3_of(f.shared.w_down.weight, src_opt)) if f.shared is not None else None)
        R0 = w3_of(f.router, src_opt)
        tile = torch.arange(f.E, device=f.router.device).repeat_interleave(m)
        router = R0.map(lambda t: t[tile].clone())
        bias = f.bias[tile].clone()
        offset = torch.zeros_like(bias)
        if mode == "split":
            new_exps = [b.take(torch.arange(j * cfg.width, (j + 1) * cfg.width, device=b.gate.w.device))
                        for b in exps for j in range(m)]
        else:
            std = bundle_std(exps[0])
            new_exps = []
            for b in exps:
                for c in range(m):
                    new_exps.append(b.redraw(0.5, gen, std) if mode == "drop" else b.clone())
            if mode in ("copy", "drop") and noise > 0:
                rs = R0.w.std().item()
                router.w += (torch.randn(router.w.shape, generator=gen, dtype=torch.float64) * noise * rs).to(router.w)
            if mode == "staggered":
                # clone c of every family starts c*delta below clone 0: top-k picks exactly what it picked before
                delta = 1.0 + (f.bias.max() - f.bias.min()).item() + 0.01
                offset = -torch.arange(m, device=bias.device).repeat(f.E).to(bias.dtype) * delta
                deltas.append(delta)
        layers.append({"experts": new_exps, "shared": shared, "router": router, "bias": bias, "offset": offset})
    model, opt = assemble(src, src_opt, cfg, layers)
    return model, opt, {"mode": mode, "cfg": asdict(cfg), "deltas": deltas}


def staggered_schedule(model, window):
    """Offsets shrink linearly to 0 over `window` steps; the bias balancer takes over from there."""
    start = [m.offset.clone() for m in model.moe_layers()]
    def fn(s):
        frac = max(0.0, 1.0 - s / max(1, window))
        for m, o in zip(model.moe_layers(), start):
            m.offset.copy_(o * frac)
    return fn
'''),
md(r'''
### Gates for the surgery

The same exact-equality discipline, on a tiny fp64 model. Each line is a claim from the table
above, checked to ~1e-12.
'''),
code(r'''
torch.manual_seed(0)
_tc = GPTConfig(vocab=64, d=32, n_layer=2, n_head=4, n_kv=2, ctx=16, ffn=64)
_dense = GPT(_tc).double().eval()
_ids = torch.randint(0, 64, (3, 16))
with torch.no_grad():
    _ref = _dense(_ids)

def _logits(m):
    m.eval()
    with torch.no_grad():
        return m(_ids)

_cp, _, _ = convert(_dense, None, "copy")
gate("copy-upcycled MoE gives the dense logits (any router)", torch.allclose(_logits(_cp), _ref, atol=1e-10))

_pt, _, _ = convert(_dense, None, "partition")
with torch.no_grad():
    for _f in _pt.moe_layers():
        _f.router.zero_()                     # equal scores -> every weight = 1/k, x route_scale k = 1
        _f.k = _f.E // 2                      # choose one full copy's worth of slices...
with torch.no_grad():                         # ...and make it exactly one copy: experts 0..3 are copy 0
    for _f in _pt.moe_layers():
        _f.mask[:] = False; _f.mask[: _f.E // 2] = True
        _f.route_scale = float(_f.k)
gate("partition: shared + one full set of slices sums to the dense block", torch.allclose(_logits(_pt), _ref, atol=1e-10))

_m8, _, _ = convert(_dense, None, "partition")
with torch.no_grad():
    for _f in _m8.moe_layers():
        _f.router.normal_(0, 0.5)             # a real, uneven router
        _f.bias.normal_(0, 0.05)
_r8 = _logits(_m8)
_sp, _, _ = grow(_m8, None, "split", m=4)
gate("fine-grained split (E x4, k x4, scale x4) is exact", torch.allclose(_logits(_sp), _r8, atol=1e-10))
_st, _, _info = grow(_m8, None, "staggered", m=4)
gate("staggered-bias clone growth (E x4) is exact at step 0", torch.allclose(_logits(_st), _r8, atol=1e-10),
     f"delta per layer = {[round(x, 3) for x in _info['deltas']]}")
_cg, _, _ = grow(_m8, None, "copy", m=4, noise=0.0)
def _route_ids(m):
    m.eval()
    for f in m.moe_layers():
        f.record = True
    with torch.no_grad():
        m(_ids)
    for f in m.moe_layers():
        f.record = False
    return [f.last["topi"] for f in m.moe_layers()]
_fam = family_share(_route_ids(_cg), 4)
gate("plain clone growth is NOT exact: top-k piles into one clone family",
     not torch.allclose(_logits(_cg), _r8, atol=1e-6), f"tokens whose 4 picks share one family: {_fam:.0%}")

gate("the main recipe uses loss-free balancing and no Switch aux loss",
     MAIN_MOE8.balance == "bias" and MAIN_MOE8.switch_alpha == 0.0 and MAIN_MOE8.seq_alpha <= 1e-4)
'''),
md(r'''
**Optimizer-state gate.** Convert a model that has real Adam state and check that every new
parameter either carries state of exactly its own shape or has none, and that a carried shared-expert
moment is literally the dense moment of those neurons.
'''),
code(r'''
_d2 = GPT(_tc)
_o2 = make_opt(_d2)
for _ in range(2):
    _o2.zero_grad(); _d2(_ids, _ids)[1].backward(); _o2.step()
_g = torch.Generator().manual_seed(0)
_m2, _om2, _ = convert(_d2, _o2, "partition", seed=0)
_shapes_ok = all(st["exp_avg"].shape == p.shape for p, st in _om2.state.items())
_perm = torch.randperm(_tc.ffn, generator=_g)[: _tc.ffn // 2]
_src_m = _o2.state[_d2.blocks[0].ffn.w_gate.weight]["exp_avg"][_perm]
_dst_m = _om2.state[_m2.blocks[0].ffn.shared.w_gate.weight]["exp_avg"]
gate("Adam moments travel with their neurons", _shapes_ok and torch.equal(_src_m, _dst_m),
     f"{len(_om2.state)} of {sum(1 for _ in _m2.parameters())} params carry state; the router starts fresh")
del _d2, _o2, _m2, _om2
'''),
md(r'''
### Convert the real Stage 1 model, and measure what each method preserves

Validation loss of the T1 dense model, then of each converted model **before any MoE training**.
This is the "function preservation" each method promises, measured.
'''),
code(r'''
CONV_METHODS = [("copy", "random"), ("drop", "random"), ("partition", "random"), ("partition", "crop"),
                ("partition", "half_fresh"), ("partition_noshared", "random"), ("random", "random")]
CONV_LABEL = {("copy", "random"): "copy (8 x full, top-2)", ("drop", "random"): "drop r=0.5 (8 x full, top-2)",
              ("partition", "random"): "partition: shared = random half  [main]",
              ("partition", "crop"): "partition: shared = first half (crop)",
              ("partition", "half_fresh"): "partition: shared = half copied, half fresh",
              ("partition_noshared", "random"): "partition, no shared (16 x 192, top-8)",
              ("random", "random"): "random experts (attention kept)"}
if COMPUTE and not have("conversion"):
    rows = []
    for method, si in CONV_METHODS:
        mdl, _, info = convert(dense, None, method, seed=0, shared_init=si)
        rows.append(dict(label=CONV_LABEL[(method, si)], val=evaluate(mdl, BUDGET["val_batches"]),
                         params=n_params(mdl), active=n_active(mdl)))
        del mdl
    R["conversion"] = dict(dense_val=R["val_T1"], dense_params=n_params(dense), rows=rows)
    save_results()

if have("conversion"):
    C = R["conversion"]
    lines = ["| method | val loss right after conversion | jump | total params | active params |", "|---|---|---|---|---|",
             f"| dense model at T1 | {C['dense_val']:.4f} | - | {C['dense_params'] / 1e6:.1f}M | {C['dense_params'] / 1e6:.1f}M |"]
    for r in C["rows"]:
        lines.append(f"| {r['label']} | {r['val']:.4f} | {r['val'] - C['dense_val']:+.4f} | {r['params'] / 1e6:.1f}M | {r['active'] / 1e6:.1f}M |")
    display(Markdown("\n".join(lines)))
'''),
md(r'''
Read the table this way: *copy* costs nothing at conversion, because every expert is the dense
block and the weights average to one. Crop and random choice of the shared half are the same thing,
since a trained layer's neurons have no meaningful order. Partition keeps the signal *on average*
but not per token, and it pays for that with a jump. The random-expert model has lost its whole
feed-forward block. The labs below test whether a small jump now buys a lower loss later.

**Sample text right before and right after surgery** (same seed, same prompt):
'''),
code(r'''
PROMPTS = ["Once upon a time, there was a little", "Tom and Lily went to the park. Tom said,"]
if COMPUTE and not have("samples", "dense_T1"):
    _m8, _, _ = convert(dense, None, "partition", seed=0)
    R.setdefault("samples", {})
    R["samples"]["dense_T1"] = [generate(dense, TOK, p) for p in PROMPTS]
    R["samples"]["moe8_at_conversion"] = [generate(_m8, TOK, p) for p in PROMPTS]
    del _m8
    save_results()
for key, title in (("dense_T1", "dense model at T1"), ("moe8_at_conversion", "MoE-8, zero steps after conversion")):
    if have("samples", key):
        print(f"--- {title}")
        for s in R["samples"][key]:
            print("   ", s.replace("\n", " "))
'''),
md(r'''
**Carry this forward:** a feed-forward block is a sum over neurons, so experts can be cut from it
exactly. *Copy* preserves the function but starts every expert identical. *Partition* starts the
experts different but only preserves the function on average. The optimizer state is part of the
model and is sliced with the same indices.
'''),
]
