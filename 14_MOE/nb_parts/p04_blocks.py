from . import md, code, mathbox

CELLS = [
md(r'''
---
## C · Building blocks: the dense model, the MoE layer, and the gates

> The MoE layer, the router, dropless dispatch and the balancing bias. Everything below is ordinary
> PyTorch; no MoE library.

The **dense model** is a small modern decoder: RoPE, RMSNorm, grouped-query attention, a SwiGLU
feed-forward block, tied embeddings. We call it the "linear model" because its
feed-forward block is a plain stack of linear layers that every token passes through in full.
'''),
code(r'''
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):                      # computed in fp32 whatever the autocast dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)


def rope_cache(T, hd, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, hd, 2).float() / hd)
    f = torch.outer(torch.arange(T).float(), inv)            # [T, hd/2]
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):                                  # x: [B, H, T, hd]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class Attention(nn.Module):
    """Grouped-query attention: n_head query heads share n_kv key/value heads."""
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.h, self.kv, self.hd = c.n_head, c.n_kv, c.d // c.n_head
        self.wq = nn.Linear(c.d, self.h * self.hd, bias=False)
        self.wk = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.wv = nn.Linear(c.d, self.kv * self.hd, bias=False)
        self.wo = nn.Linear(self.h * self.hd, c.d, bias=False)

    def forward(self, x, cos, sin, trace=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if trace is not None:
            trace += [("attn: q", q.shape), ("attn: k after GQA repeat", k.shape), ("attn: out", y.shape)]
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    """The dense feed-forward block: d -> F (gate and up) -> d (down)."""
    def __init__(self, d, f):
        super().__init__()
        self.w_gate = nn.Linear(d, f, bias=False)   # weight [F, d]: one ROW per neuron
        self.w_up = nn.Linear(d, f, bias=False)     # weight [F, d]: one ROW per neuron
        self.w_down = nn.Linear(f, d, bias=False)   # weight [d, F]: one COLUMN per neuron

    def forward(self, x, trace=None):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
'''),
md(r'''
### The MoE layer

One module replaces `SwiGLU`. Read `route` and `dispatch` side by side with this picture:

```
token x [d] ──► router (fp32): logits = x · Wᵣᵀ  [E]
                 scores  = sigmoid(logits)                      (or softmax / √softplus)
                 choose  = top-k of (scores + bias + offset)    ← bias & offset steer the CHOICE only
                 weights = scores[chosen] / Σ scores[chosen] × route_scale
y = Σ_chosen weightᵢ · Expertᵢ(x)   +   SharedExpert(x)
```

* **Experts are stored stacked**: `gate, up: [E, w, d]`, `down: [E, d, w]`. Expert *e* is the slice
  `[e]`, a SwiGLU of inner width *w*.
* **Dropless dispatch**: sort the (token, expert) pairs by expert, pad each expert's group to the
  busiest expert's count *C*, run all experts as three batched matmuls on an `[E, C, d]` tensor,
  and scatter-add the weighted results back to their tokens. When routing is so uneven that padding
  would waste more than half the work (right after growth, or with a collapsed router), it switches
  to one **grouped GEMM** (`F.grouped_mm`, if this PyTorch and GPU support it) over the sorted
  tokens, with no padding at all. The last fallback is one small matmul per expert. All three paths
  are gated against each other. No token is ever dropped unless you ask for a `capacity_factor` (we
  do, in Part M, to measure what dropping would cost).
* **The router runs in fp32** with autocast switched off. Switch Transformer diverged when its router
  ran in 16-bit; a gate below checks this.
* **`route_scale`**: the top-k weights are renormalised to sum to 1 and then multiplied by this.
  When the experts are *pieces* of a dense block (partition), their outputs must be *summed*, so the
  scale is *k*. When the experts are *copies*, their outputs must be *averaged*, so the scale is 1.
  This is the "routed scaling factor" of the DeepSeek models (DeepSeek-V3 uses 2.5), derived in Part F instead
  of quoted.

🧠 **Intuition: a post office.** Every token is a letter addressed to *k* experts. The post office
does four things:

1. Photocopy each letter *k* times.
2. Sort the copies into one bin per expert.
3. Let each expert process its whole bin in one go.
4. Send the answers back and add them up at the sender's address, each answer weighted by how much
   the router trusted that expert.

The sorting is what makes it fast. An expert never looks at tokens that are not addressed to it.

🧠 **Intuition: the balancing bias is a thermostat.** After every step each expert's "door" gets
slightly harder to walk through if it was busier than average, and slightly easier if it was
quieter. The thermostat never touches *how good* the experts are (the gradient), only *which door*
the tokens choose.
'''),
mathbox("the MoE layer, its shapes, dispatch as a permutation, and the bias as a controller", r'''
**The layer.** For a token $x \in \mathbb{R}^d$, with chosen set $\mathcal{T}(x)$, weights $g_i$ and shared
expert $S$:

$$y = \sum_{i \in \mathcal{T}(x)} g_i\,E_i(x) \;+\; S(x), \qquad E_i(x) = W^{\text{down}}_i\Big(\operatorname{SiLU}\big(W^{\text{gate}}_i x\big) \odot W^{\text{up}}_i x\Big).$$

| tensor | shape (ours, Stage 3) |
|---|---|
| tokens entering the layer | $[N, d] = [8192, 384]$ per micro-batch |
| router $W_r$, logits $z$ | $[E, d] = [32, 384]$, $[N, E]$ |
| chosen ids, weights | $[N, k] = [8192, 4]$ |
| `gate`, `up` / `down` | $[E, w, d] = [32, 192, 384]$ / $[E, d, w]$ |
| padded expert batch | $[E, C, d]$, with $C = \max_i n_i$ |
| shared expert | $w_s = 768$: $[768, 384]$ ×2, $[384, 768]$ |

**Dispatch as a permutation.**

1. Repeat each token $k$ times to get $Nk$ (token, expert) pairs.
2. Let $\Pi$ be the permutation that sorts those pairs by expert.
3. Run each expert on its contiguous block, then undo the sort:

$$Y_{\text{pairs}} = \Pi^\top\,\operatorname{blockdiag}\big(E_1, \dots, E_E\big)\,\Pi\,X_{\text{pairs}}, \qquad y_t = \sum_{\text{pairs of } t} g\cdot Y_{\text{pair}}.$$

Padding block $i$ to $C$ rows wastes $EC - Nk$ rows of compute. When that exceeds half of $Nk$, the
layer switches to a grouped GEMM, which needs no padding.

**The bias as a controller.** After each optimizer step, with $n_i$ counted over the whole batch:

$$b_i \;\leftarrow\; b_i + \gamma\,\operatorname{sign}\big(\bar n - n_i\big).$$

* The choice uses $\sigma_i + b_i$, but the weights use $\sigma_i$ only. So $\partial \ell / \partial b_i = 0$: the
  language loss's gradient is untouched. This is the "loss-free" in the name.
* It is an integral (bang-bang) controller: $b_i(t) = b_i(0) + \gamma \sum_{\tau < t} \operatorname{sign}(\bar n - n_i(\tau))$.
  It moves at most $\gamma$ per step, and stops drifting only when the loads are equal; with the
  sign rule it then jitters within about $\pm\gamma$.
* **How fast it acts.** Suppose a busy expert beats an idle one by a score gap $\Delta\sigma$. The busy
  bias falls by $\gamma$ per step and the idle one rises by $\gamma$, so they swap after about
  $\Delta\sigma / (2\gamma)$ steps. A gap of 0.05 at $\gamma = 10^{-3}$ takes about 25 steps.

**The sequence-level auxiliary loss** (DeepSeek-V3), averaged over the sequences in a batch:

$$\mathcal{L}_{\text{seq}} = \alpha \sum_{i=1}^{E} f_i P_i, \qquad f_i = \frac{E}{kT}\sum_{t=1}^{T} \mathbb{1}[i \in \mathcal{T}_t], \qquad P_i = \frac{1}{T}\sum_{t=1}^{T} \tilde\sigma_{t,i}.$$

At perfect balance $f_i = 1$ and $P_i = 1/E$, so $\mathcal{L}_{\text{seq}} = \alpha$. With $\alpha = 10^{-4}$ it is
a light guard against a single sequence collapsing onto a few experts, too small to fight the
language loss.
'''),
code(r'''
SCORE_FNS = {
    "softmax": lambda z: torch.softmax(z, dim=-1),
    "sigmoid": torch.sigmoid,
    "sqrt_softplus": lambda z: torch.sqrt(F.softplus(z)),       # DeepSeek-V4
}


def _probe_grouped_mm():
    """F.grouped_mm runs all experts in one kernel with no padding. Use it only if it exists here and is correct."""
    fn = getattr(F, "grouped_mm", None) or getattr(torch, "_grouped_mm", None)
    if fn is None or DEVICE != "cuda" or PREC == "fp32":
        return None
    try:
        x = torch.randn(8, 16, device=DEVICE, dtype=AMP_DTYPE)
        W = torch.randn(3, 8, 16, device=DEVICE, dtype=AMP_DTYPE)
        offs = torch.tensor([4, 4, 8], device=DEVICE, dtype=torch.int32)          # includes an empty group
        y = fn(x, W.transpose(1, 2), offs=offs)
        ref = torch.cat([x[:4] @ W[0].t(), x[4:] @ W[2].t()])
        return fn if torch.allclose(y.float(), ref.float(), atol=1e-1, rtol=1e-2) else None
    except Exception:
        return None

GROUPED_MM = _probe_grouped_mm()
print("grouped_mm:", "available" if GROUPED_MM else "not available (padded + loop dispatch only)")


def autocast_dtype(t):
    try:
        if torch.is_autocast_enabled(t.device.type):
            return torch.get_autocast_dtype(t.device.type)
    except TypeError:                                            # older torch
        if t.is_cuda and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return t.dtype


class MoEFFN(nn.Module):
    def __init__(self, d, m: MoEConfig, n_layer=8):
        super().__init__()
        E, w = m.n_exp, m.width
        self.d, self.E, self.w, self.k, self.cfg = d, E, w, m.top_k, m
        self.gate = nn.Parameter(torch.randn(E, w, d) * 0.02)
        self.up = nn.Parameter(torch.randn(E, w, d) * 0.02)
        self.down = nn.Parameter(torch.randn(E, d, w) * 0.02 / math.sqrt(2 * n_layer))
        self.router = nn.Parameter(torch.randn(E, d) * 0.002)    # 0.1 x the usual 0.02 (Switch)
        self.shared = SwiGLU(d, m.shared) if m.shared else None
        self.register_buffer("bias", torch.zeros(E))              # loss-free balancing
        self.register_buffer("offset", torch.zeros(E))            # staggered clone offsets (growth lab)
        self.register_buffer("mask", torch.ones(E, dtype=torch.bool))  # pruning (Part L)
        self.register_buffer("counts", torch.zeros(E))            # load since the last bias update
        self.register_buffer("window", torch.zeros(E))            # load since the last log read
        self.route_scale = float(m.top_k if m.route_scale is None else m.route_scale)
        self.gumbel = False             # probabilistic top-k (growth window)
        self.capacity_factor = None     # None = dropless
        self.record = False             # keep this forward's routing for analysis
        self.dispatch_mode = "auto"     # "auto" | "padded" | "grouped" | "loop" (all give the same result)
        self.aux = None
        self.last = {}

    # ---- choosing -------------------------------------------------------------------------
    def route(self, x2):
        with torch.autocast(device_type=x2.device.type, enabled=False):
            rdt = torch.float64 if x2.dtype == torch.float64 else torch.float32   # fp32, never 16-bit
            logits = x2.to(rdt) @ self.router.to(rdt).t()                 # [N, E]
            scores = SCORE_FNS[self.cfg.score](logits)                    # [N, E]
            sel = scores.detach() + self.bias + self.offset               # selection-only terms
            if self.gumbel and self.training:
                # sample k experts without replacement, P ~ exp(z-scored preference): Gumbel top-k
                z = (sel - sel.mean(-1, keepdim=True)) / (sel.std(-1, keepdim=True) + 1e-6)
                u = torch.rand_like(z).clamp_(1e-9, 1 - 1e-9)
                sel = z - torch.log(-torch.log(u))
            sel = sel.masked_fill(~self.mask, float("-inf"))
            topi = sel.topk(self.k, dim=-1).indices                       # [N, k]  (no gradient)
            w = scores.gather(1, topi)                                    # weights from scores, not bias
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9) * self.route_scale
        with torch.no_grad():
            if self.training:
                c = torch.bincount(topi.reshape(-1), minlength=self.E).float()
                self.counts += c
                self.window += c
            if self.record:
                self.last["topi"], self.last["w"] = topi, w.detach()
            self.last["z"] = torch.logsumexp(logits, -1).pow(2).mean()     # router z diagnostic
        return topi, w, scores

    # ---- computing (dropless by default) ----------------------------------------------------
    def dispatch(self, x2, topi, w, trace=None):
        N, d = x2.shape
        flat_e = topi.reshape(-1)                                         # [N*k]
        order = torch.argsort(flat_e, stable=True)                        # group by expert, keep token order
        e_sorted = flat_e[order]
        tok = torch.arange(N, device=x2.device).repeat_interleave(self.k)[order]
        wts = w.reshape(-1)[order]
        counts = torch.bincount(flat_e, minlength=self.E)
        if self.capacity_factor is not None:                              # old-style capacity, for Part M
            cap = math.ceil(N * self.k / self.E * self.capacity_factor)
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(order.numel(), device=x2.device) - starts[e_sorted]
            keep = rank < cap                                             # earlier tokens win the slots
            self.last["dropped"] = int((~keep).sum())
            tok, e_sorted, wts = tok[keep], e_sorted[keep], wts[keep]
            counts = torch.bincount(e_sorted, minlength=self.E)
        dt = autocast_dtype(x2)
        g, u, dn = self.gate.to(dt), self.up.to(dt), self.down.to(dt)
        xs = x2.to(dt)[tok]                                               # [M, d] (token, expert) pairs, by expert
        M = tok.numel()
        C = int(counts.max()) if M else 0                                 # one host sync per layer
        mode = self.dispatch_mode
        if mode == "auto":
            mode = ("padded" if self.E * C <= 1.5 * M + 64 * self.E
                    else "grouped" if (GROUPED_MM is not None and x2.is_cuda and dt in (torch.bfloat16, torch.float16)) else "loop")
        if M and mode == "grouped":
            # uneven routing: one grouped GEMM over the sorted tokens, no padding at all
            offs = torch.cumsum(counts, 0).to(torch.int32)
            h = F.silu(GROUPED_MM(xs, g.transpose(1, 2), offs=offs)) * GROUPED_MM(xs, u.transpose(1, 2), offs=offs)
            ys = GROUPED_MM(h, dn.transpose(1, 2), offs=offs)
        elif M and mode == "padded":
            # balanced enough: pad every expert's group to C rows and run three batched matmuls
            pos = torch.arange(M, device=x2.device) - (torch.cumsum(counts, 0) - counts)[e_sorted]
            buf = xs.new_zeros(self.E, C, d).index_put((e_sorted, pos), xs)          # [E, C, d]
            h = F.silu(torch.bmm(buf, g.transpose(1, 2))) * torch.bmm(buf, u.transpose(1, 2))
            ys = torch.bmm(h, dn.transpose(1, 2))[e_sorted, pos]                    # [M, d]
        else:
            # fallback: one small matmul per expert
            outs, start = [], 0
            for e, c in enumerate(counts.tolist()):
                if c:
                    xe = xs[start:start + c]
                    outs.append((F.silu(xe @ g[e].t()) * (xe @ u[e].t())) @ dn[e].t())
                    start += c
            ys = torch.cat(outs) if outs else xs[:0]
        ys = ys * wts[:, None].to(dt)
        acc = torch.float64 if x2.dtype == torch.float64 else torch.float32     # accumulate the k outputs in fp32+
        y = torch.zeros(N, d, device=x2.device, dtype=acc).index_add_(0, tok, ys.to(acc))
        if trace is not None:
            trace += [("moe: tokens x d", x2.shape), ("moe: top-k ids", topi.shape),
                      ("moe: (token,expert) pairs", tok.shape), ("moe: expert gate stack", self.gate.shape),
                      ("moe: padded expert batch [E, C, d]", (self.E, C, d)),
                      ("moe: routed output", y.shape)]
        return y.to(x2.dtype)

    def reference(self, x2, topi, w):
        """Slow and obviously correct: run every expert on every token, keep the chosen ones."""
        h = F.silu(torch.einsum("nd,ewd->new", x2, self.gate)) * torch.einsum("nd,ewd->new", x2, self.up)
        ye = torch.einsum("new,edw->ned", h, self.down)
        W = torch.zeros(x2.shape[0], self.E, dtype=x2.dtype, device=x2.device).scatter(1, topi, w.to(x2.dtype))
        return torch.einsum("ne,ned->nd", W, ye)

    def forward(self, x, trace=None):
        B, T, d = x.shape
        x2 = x.reshape(-1, d)
        topi, w, scores = self.route(x2)
        y = self.dispatch(x2, topi, w, trace)
        aux = torch.zeros((), device=x.device)
        if self.training and (self.cfg.seq_alpha > 0 or self.cfg.switch_alpha > 0):
            hit = torch.zeros(x2.shape[0], self.E, device=x.device).scatter_(1, topi, 1.0)  # [N, E]
            p = scores / scores.sum(-1, keepdim=True)                                       # affinities
            if self.cfg.seq_alpha > 0:        # DeepSeek-V3 sequence-wise loss: alpha * sum_i f_i P_i per sequence
                f = hit.view(B, T, self.E).mean(1) * (self.E / self.k)
                P = p.view(B, T, self.E).mean(1)
                aux = aux + self.cfg.seq_alpha * (f * P).sum(-1).mean()
            if self.cfg.switch_alpha > 0:     # Switch: alpha * E * sum_i f_i P_i over the batch
                f = hit.mean(0) / self.k
                aux = aux + self.cfg.switch_alpha * self.E * (f * p.mean(0)).sum()
        self.aux = aux
        out = y.view(B, T, d)
        if self.shared is not None:
            out = out + self.shared(x)
        return out

    # ---- balancing: runs after every optimizer step ------------------------------------
    @torch.no_grad()
    def update_bias(self, gamma):
        if self.cfg.balance == "bias" and gamma > 0:
            self.bias += gamma * torch.sign(self.counts.mean() - self.counts)
        self.counts.zero_()

    def read_window(self):
        c = self.window.clone()
        self.window.zero_()
        return c
'''),
md(r'''
### The model that holds either block

`GPT(GCFG)` is the dense model. `GPT(GCFG, MoEConfig(...))` is the same model with every
feed-forward block replaced by `MoEFFN`. Attention, embeddings and norms are identical, which is what
makes the conversion a pure transplant of the feed-forward weights.
'''),
code(r'''
class Block(nn.Module):
    def __init__(self, c, moe=None):
        super().__init__()
        self.n1, self.attn, self.n2 = RMSNorm(c.d), Attention(c), RMSNorm(c.d)
        self.ffn = MoEFFN(c.d, moe, c.n_layer) if moe else SwiGLU(c.d, c.ffn)

    def forward(self, x, cos, sin, trace=None):
        x = x + self.attn(self.n1(x), cos, sin, trace)
        return x + self.ffn(self.n2(x), trace)


class GPT(nn.Module):
    def __init__(self, c: GPTConfig, moe: MoEConfig = None):
        super().__init__()
        self.c, self.moe_cfg = c, moe
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(Block(c, moe) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d)
        cos, sin = rope_cache(c.ctx, c.d // c.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for n, p in self.named_parameters():
            if p.dim() == 2 and not n.endswith("router"):
                std = 0.02 / math.sqrt(2 * c.n_layer) if n.endswith(("wo.weight", "w_down.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx, targets=None, per_seq=False, trace=None):
        T = idx.shape[1]
        x = self.emb(idx)
        if trace is not None:
            trace.append(("embeddings", x.shape))
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T], trace)
            trace = None                       # trace the first block only
        logits = self.norm(x) @ self.emb.weight.t()          # tied output head
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                               reduction="none").view(targets.shape)
        return logits, (loss.mean(1) if per_seq else loss.mean())

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoEFFN)]

    def aux_loss(self):
        return sum((m.aux for m in self.moe_layers() if m.aux is not None), torch.zeros((), device=self.emb.weight.device))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def n_active(model):
    """Parameters one token touches: everything except the experts it did not choose."""
    idle = sum((m.E - m.k) * 3 * m.d * m.w for m in model.moe_layers())
    return n_params(model) - idle


@torch.no_grad()
def generate(model, tok, prompt, n=48, temp=0.8, top_k=40, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt).ids], device=DEVICE)
    was = model.training
    model.eval()
    for _ in range(n):
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=PREC != "fp32"):
            logits = model(ids[:, -model.c.ctx:])[:, -1].float() / temp
        v, i = logits.topk(top_k)
        nxt = i.gather(1, torch.multinomial(torch.softmax(v, -1), 1, generator=g))
        ids = torch.cat([ids, nxt], 1)
        if nxt.item() == 0:
            break
    model.train(was)
    return tok.decode(ids[0].tolist())
'''),
md(r'''
### Shape trace: the dimensions you need to be aware of

One forward pass of a 2-sequence batch through the first block of an MoE model with the final
(Stage 3) shape: 32 routed experts of width 192, top-4, one shared expert of width 768.
'''),
code(r'''
torch.manual_seed(0)
_m = GPT(GCFG, MoEConfig(n_exp=32, width=192, top_k=4, shared=768)).eval()
_tr = []
with torch.no_grad():
    _m(torch.randint(0, GCFG.vocab, (2, 16)), trace=_tr)
for name, shape in _tr:
    print(f"{name:<38} {tuple(shape)}")
_ffn = _m.blocks[0].ffn
print(f"{'router weight (fp32)':<38} {tuple(_ffn.router.shape)}  dtype={_ffn.router.dtype}")
print(f"{'shared expert gate':<38} {tuple(_ffn.shared.w_gate.weight.shape)}")
del _m, _ffn
'''),
md(r'''
> **Pitfall: `nn.Linear` stores weights as `[out, in]`.** A *neuron* of the feed-forward block is a
> **row** of `w_gate.weight` and of `w_up.weight` (shape `[F, d]`), and a **column** of
> `w_down.weight` (shape `[d, F]`). Slice rows of gate/up together with the *same* columns of down,
> or you build an expert out of mismatched halves and nothing errors, it just trains worse.

### Gates: exact checks that must pass before anything trains

Each gate is an equality the code must satisfy. They run in fp64 on the CPU in about a second, so
they run in every mode, including `learn`.
'''),
code(r'''
torch.manual_seed(0)
_d, _F = 16, 32
_dense = SwiGLU(_d, _F).double()
_x = torch.randn(64, _d, dtype=torch.float64)

# 1. An MoE with one expert, top-1 and scale 1 IS the dense block.
_one = MoEFFN(_d, MoEConfig(n_exp=1, width=_F, top_k=1, route_scale=1.0)).double().eval()
with torch.no_grad():
    _one.gate[0].copy_(_dense.w_gate.weight); _one.up[0].copy_(_dense.w_up.weight); _one.down[0].copy_(_dense.w_down.weight)
gate("E=1, k=1 MoE equals the dense SwiGLU", torch.allclose(_one(_x[None])[0], _dense(_x), atol=1e-12))

# 2. Sorted dropless dispatch equals the run-every-expert reference, forward and backward.
_moe = MoEFFN(_d, MoEConfig(n_exp=8, width=12, top_k=3)).double()
_t, _w, _ = _moe.route(_x)
_b = _moe.reference(_x, _t, _w)
_gb = torch.autograd.grad(_b.pow(2).sum(), [_moe.gate, _moe.down, _moe.router], retain_graph=True)
for _path in ("padded", "loop"):
    _moe.dispatch_mode = _path
    _t, _w, _ = _moe.route(_x)
    _a = _moe.dispatch(_x, _t, _w)
    _ga = torch.autograd.grad(_a.pow(2).sum(), [_moe.gate, _moe.down, _moe.router])
    gate(f"{_path} dispatch == reference (forward and gradients)",
         torch.allclose(_a, _b, atol=1e-12) and all(torch.allclose(a, b, atol=1e-10) for a, b in zip(_ga, _gb)))
_moe.dispatch_mode = "auto"
if GROUPED_MM is not None:                       # GPU only: the grouped GEMM path, in the training dtype
    _mg = MoEFFN(64, MoEConfig(n_exp=8, width=32, top_k=3)).to(DEVICE)
    _xg = torch.randn(512, 64, device=DEVICE)
    _outs = {}
    with torch.autocast(DEVICE, dtype=AMP_DTYPE):
        for _path in ("padded", "grouped", "loop"):
            _mg.dispatch_mode = _path
            _tg, _wg, _ = _mg.route(_xg)
            _outs[_path] = _mg.dispatch(_xg, _tg, _wg).float()
    gate("grouped-GEMM dispatch == padded == loop (GPU, training dtype)",
         torch.allclose(_outs["grouped"], _outs["loop"], atol=2e-2, rtol=2e-2) and torch.allclose(_outs["padded"], _outs["loop"], atol=2e-2, rtol=2e-2),
         f"max diff {(_outs['grouped'] - _outs['loop']).abs().max():.1e}")
    del _mg

# 3. The router learns: top-k has no gradient, but the weights it multiplies by do.
gate("router receives a gradient", _ga[2].abs().sum() > 0, f"|grad| = {_ga[2].abs().sum():.3e}")

# 4. Router logits stay fp32 under 16-bit autocast.
_moe32 = MoEFFN(_d, MoEConfig(n_exp=8, width=12, top_k=3))
with torch.autocast("cpu", dtype=torch.bfloat16):
    _, _w32, _s32 = _moe32.route(_x.float())
gate("router runs in fp32 under bf16 autocast", _s32.dtype == torch.float32 and _w32.dtype == torch.float32)

# 5. The balancing bias changes WHICH experts are chosen, never their weights.
_moe.eval()
with torch.no_grad():
    t0, w0, s0 = _moe.route(_x)
    _moe.bias[t0[0, 0]] -= 10.0                      # push token 0's favourite out
    t1, w1, s1 = _moe.route(_x)
    _moe.bias.zero_()
_chosen = set(t1[0].tolist())
_expect = s1[0].gather(0, t1[0]) / s1[0].gather(0, t1[0]).sum() * _moe.route_scale
gate("bias changes the choice, not the weights",
     t0[0, 0].item() not in _chosen and torch.allclose(w1[0], _expect), f"token 0: {t0[0].tolist()} -> {t1[0].tolist()}")
'''),
]
