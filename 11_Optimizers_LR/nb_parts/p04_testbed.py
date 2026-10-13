# %% [markdown]
r"""
## §3 The testbed

Every training experiment from here on uses the pieces defined in this section. Each piece is
built so that comparisons between runs are **paired**: two runs with the same seed start from the
same weights and see the same batches in the same order. Any difference between them then comes
from the one setting that changed.

**Data.**
- TinyStories (GPT-4 version), tokenized with the SmolLM2 tokenizer (49,152 tokens).
- The token stream is cut into non-overlapping windows. A seed fixes a permutation of those
  windows, so no window repeats within a run.
- Validation loss always uses the same fixed set of held-out windows.

**Two model families.**
| | base model | width family |
|---|---|---|
| used in | §4, §5, §6, §M | §7 |
| shape | width 384, 8 layers, 6 query / 2 key-value heads (GQA), SwiGLU 1024, RoPE | width 256…2,048, 4 layers, head size 64, SwiGLU ≈ 8/3·width |
| vocabulary | 49,152, tied input/output embedding | the 8,191 most frequent ids + one "other" id, untied |
| initialization | GPT-2 style: N(0, 0.02), residual projections scaled by 1/√(2·layers) | standard parameterization: N(0, 1/fan_in), embedding N(0, 1) |
| parameters | 31.5M | 7M … 234M |
| tokens per step | 32 × 512 = 16,384 | 64 × 256 = 16,384 |

The width family predicts only the most frequent ids, so that width rather than a 49k-row
embedding dominates its parameter count.

**Training.**
- Optimizer: AdamW with $\beta=(0.9, 0.95)$, $\epsilon=10^{-8}$, weight decay 0.1 on matrices only
  (not on norm gains).
- Clipping: gradient norm 1.0.
- Precision: bf16 autocast with fp32 master weights.
- Logging, on every step that is logged:
  - loss, learning rate, gradient norm, whether clipping fired
  - each tensor's update-to-weight ratio $\lVert\Delta W\rVert/\lVert W\rVert$
"""

# %%
TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
TOK_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json"
TRAIN_MB, VALID_MB = qk(60, 5), qk(8, 2)
EOT = 0                                         # <|endoftext|> in the SmolLM2 vocabulary
V_BASE, V_WIDTH = 49152, 8192


def fetch(url, dest, max_bytes=None, tries=12):
    """Resumable download into dest.part using HTTP Range requests (large downloads do drop)."""
    dest = Path(dest)
    if dest.exists():
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        if max_bytes and have >= max_bytes:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib",
                                                   "Range": f"bytes={have}-{max_bytes - 1 if max_bytes else ''}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                if r.status == 200 and have:            # the server ignored Range: start over
                    f.truncate(0)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if not max_bytes:
                break
        except urllib.error.HTTPError as e:
            if e.code == 416:                           # nothing left to send
                break
            print(f"  download error ({e}); retrying"); time.sleep(min(30, 2 ** attempt))
        except Exception as e:
            print(f"  download error ({e}); retrying"); time.sleep(min(30, 2 ** attempt))
    else:
        raise RuntimeError(f"could not download {url}")
    if max_bytes and part.stat().st_size > max_bytes:
        with open(part, "r+b") as f:
            f.truncate(max_bytes)
    part.replace(dest)
    return dest


def load_tokens(split):
    mb = TRAIN_MB if split == "train" else VALID_MB
    cache = DATA / f"{split}_{mb}mb.npy"
    if cache.exists():
        return np.load(cache)
    url = TRAIN_URL if split == "train" else VALID_URL
    text = fetch(url, DATA / f"tinystories_{split}_{mb}mb.txt", max_bytes=mb * 2**20).read_text(encoding="utf-8", errors="ignore")
    docs = [d.strip() for d in text.split("<|endoftext|>")[:-1] if d.strip()]     # the last piece may be cut mid-story
    enc = TOK.encode_batch(docs)
    ids = np.fromiter((i for e in enc for i in (*e.ids, EOT)), dtype=np.uint16)
    np.save(cache, ids)
    return ids


TOK = None
if MODE != "learn":
    from tokenizers import Tokenizer
    TOK = Tokenizer.from_file(str(fetch(TOK_URL, DATA / "smollm2_tokenizer.json")))
    check(TOK.get_vocab_size() == V_BASE and TOK.token_to_id("<|endoftext|>") == EOT, "SmolLM2 tokenizer: 49,152 ids, <|endoftext|> = 0")
    train_ids, valid_ids = load_tokens("train"), load_tokens("valid")
    counts = np.bincount(train_ids, minlength=V_BASE)
    rank = np.argsort(-counts, kind="stable")
    REMAP = torch.full((V_BASE,), V_WIDTH - 1, dtype=torch.long)                # everything rare → "other"
    REMAP[torch.from_numpy(rank[:V_WIDTH - 1].copy())] = torch.arange(V_WIDTH - 1)
    FREQ_RANK = torch.empty(V_BASE, dtype=torch.long)
    FREQ_RANK[torch.from_numpy(rank.copy())] = torch.arange(V_BASE)
    coverage = counts[rank[:V_WIDTH - 1]].sum() / counts.sum()
    R["data"] = _clean(dict(train_tokens=len(train_ids), valid_tokens=len(valid_ids), train_mb=TRAIN_MB,
                            bytes_per_token=TRAIN_MB * 2**20 / len(train_ids), distinct_ids=int((counts > 0).sum()),
                            top8191_coverage=float(coverage)))
    save_results()
D_INFO = R.get("data", {})
print(f"train tokens {D_INFO.get('train_tokens', 0):,}  valid tokens {D_INFO.get('valid_tokens', 0):,}  "
      f"bytes/token {D_INFO.get('bytes_per_token', 0):.2f}  distinct ids {D_INFO.get('distinct_ids', 0):,}  "
      f"top-8,191 coverage {D_INFO.get('top8191_coverage', 0):.4%}")
if D_INFO:
    check(D_INFO["top8191_coverage"] >= 0.999, "the width family's 8,191 most frequent ids cover ≥ 99.9% of tokens")

# %%
class Windows:
    """Non-overlapping (T+1)-token windows of a token stream; the order is a pure function of the seed."""

    def __init__(self, ids, T):
        self.ids, self.T = torch.from_numpy(ids.astype(np.int64)), T
        self.n = (len(ids) - 1) // T
        self._perm = {}

    def batch(self, seed, step, B):
        if seed not in self._perm:
            self._perm[seed] = torch.from_numpy(np.random.default_rng(10_000 + seed).permutation(self.n))
        k = step * B
        if k + B > self.n and not getattr(self, "_warned", False):
            print(f"  note: step {step} wraps the {self.n:,} training windows"); self._warned = True
        idx = self._perm[seed][torch.arange(k, k + B) % self.n]
        win = self.ids[(idx * self.T)[:, None] + torch.arange(self.T + 1)]
        return win


BASE_T, WIDTH_T = qk(512, 128), qk(256, 64)
BASE_B, WIDTH_B = qk(32, 16), qk(64, 16)            # windows per optimizer step
BASE_ACCUM = qk(4, 2)                                # base model: micro-batch of 8 windows (16 OOMs on 8 GB)
N_EVAL_WIN = qk(384, 32)
if MODE != "learn":
    WIN = {"base": Windows(train_ids, BASE_T), "width": Windows(train_ids, WIDTH_T)}
    EVAL_WIN = {"base": Windows(valid_ids, BASE_T).ids[: N_EVAL_WIN * BASE_T + 1],
                "width": Windows(valid_ids, WIDTH_T).ids[: N_EVAL_WIN * WIDTH_T + 1]}
    print(f"training windows: base {WIN['base'].n:,} of {BASE_T} tokens, width family {WIN['width'].n:,} of {WIDTH_T}")


def eval_batches(fam, micro):
    """The fixed validation set, as micro-batches on the device (cached)."""
    key = (fam, micro)
    if key not in _EVAL_CACHE:
        T = BASE_T if fam == "base" else WIDTH_T
        ids = EVAL_WIN[fam]
        win = ids[(torch.arange(N_EVAL_WIN) * T)[:, None] + torch.arange(T + 1)]
        if fam == "width":
            win = REMAP[win]
        _EVAL_CACHE[key] = [(w[:, :-1].to(DEV), w[:, 1:].to(DEV)) for w in win.split(micro)]
    return _EVAL_CACHE[key]


_EVAL_CACHE = {}

# %% [markdown]
r"""
**Models.** One decoder class serves both families:
- RMSNorm before attention and before the MLP.
- Rotary position embeddings.
- `scaled_dot_product_attention` with grouped key-value heads.
- A SwiGLU MLP.

The width family can also be built in a **muP** variant (§7), which changes three things:
- hidden-matrix learning rates are multiplied by $256/\text{width}$
- the logits are multiplied by $256/\text{width}$
- the output head keeps its width-256 initialization

At width 256 the two variants are the same model, and a `check` below confirms it.
"""

# %%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, n_head, n_kv, ffn):
        super().__init__()
        self.n_head, self.n_kv, self.hd = n_head, n_kv, d // n_head
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * n_kv * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 2 * ffn, bias=False)            # gate and up projections, fused
        self.down = nn.Linear(ffn, d, bias=False)

    def qk(self, x, cos, sin):
        B, T, D = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k, v = self.kv(h).view(B, T, 2, self.n_kv, self.hd).permute(2, 0, 3, 1, 4)
        return rope(q, cos, sin), rope(k, cos, sin), v

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qk(x, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_kv != self.n_head)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        a, b = self.up(self.norm2(x)).chunk(2, -1)
        return x + self.down(F.silu(a) * b)


class LM(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, n_kv, ffn, tied, T_max, init, mup=False, base_width=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_head, n_kv, ffn) for _ in range(n_layer))
        self.norm = RMSNorm(d)
        self.head = None if tied else nn.Linear(d, vocab, bias=False)
        self.mup, self.width_mult = mup, d / base_width
        self.logit_mult = 1 / self.width_mult if mup else 1.0
        hd = d // n_head
        inv = 1e4 ** (-torch.arange(0, hd, 2).float() / hd)
        f = torch.outer(torch.arange(T_max).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)
        resid = 1 / math.sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if p.ndim < 2:
                continue
            is_resid = name.endswith(("o.weight", "down.weight"))
            if init == "gpt2":
                nn.init.normal_(p, 0.0, 0.02 * (resid if is_resid else 1.0))
            elif name == "emb.weight":
                nn.init.normal_(p, 0.0, 1.0)
            elif name == "head.weight":
                nn.init.normal_(p, 0.0, (base_width if mup else d) ** -0.5)
            else:
                nn.init.normal_(p, 0.0, p.shape[1] ** -0.5 * (resid if is_resid else 1.0))

    def hidden_matrices(self):
        return [p for n, p in self.named_parameters() if n.startswith("blocks.") and p.ndim == 2]

    def forward(self, idx, targets=None, collect=None):
        T = idx.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        x = self.emb(idx)
        if collect is not None:
            collect.append(("embedding", x.float().pow(2).mean().sqrt().item()))
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin)
            if collect is not None:
                collect.append((f"block {i}", x.float().pow(2).mean().sqrt().item()))
        x = self.norm(x)
        logits = F.linear(x, self.emb.weight if self.head is None else self.head.weight) * self.logit_mult
        if collect is not None:
            collect.append(("logits", logits.float().pow(2).mean().sqrt().item()))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))

    @torch.no_grad()
    def diagnostics(self, idx):
        """Largest attention logit q·k/sqrt(d) in any layer and head, and the RMS of the output logits."""
        T = idx.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        x, worst = self.emb(idx), 0.0
        mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        for b in self.blocks:
            q, k, _ = b.qk(x, cos, sin)
            k = k.repeat_interleave(b.n_head // b.n_kv, dim=1)
            s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(b.hd)
            worst = max(worst, s.masked_fill(~mask, 0).abs().max().item())
            x = b(x, cos, sin)
        logits = self.forward(idx)
        return dict(max_attn_logit=worst, logit_rms=logits.float().pow(2).mean().sqrt().item())


BASE_ARCH = dict(vocab=V_BASE, d=qk(384, 128), n_layer=qk(8, 2), n_head=qk(6, 2), n_kv=qk(2, 1), ffn=qk(1024, 384),
                 tied=True, T_max=BASE_T, init="gpt2")
BASE_WIDTH_MUP = qk(256, 64)                       # muP's reference width = the smallest width in the sweep


def width_arch(width, mup=False):
    return dict(vocab=V_WIDTH, d=width, n_layer=qk(4, 2), n_head=width // 64, n_kv=width // 64,
                ffn=int(8 * width / 3) // 64 * 64, tied=False, T_max=WIDTH_T, init="fan_in",
                mup=mup, base_width=BASE_WIDTH_MUP)


def make_model(cfg):
    arch = BASE_ARCH if cfg["family"] == "base" else width_arch(cfg["width"], cfg.get("mup", False))
    return LM(**arch)


with torch.device("meta"):
    _m = LM(**BASE_ARCH)
N_BASE = sum(p.numel() for p in _m.parameters())
N_BASE_NONEMB = N_BASE - _m.emb.weight.numel()
print(f"base model: {N_BASE / 1e6:.2f}M parameters ({N_BASE_NONEMB / 1e6:.2f}M outside the tied embedding)")
if MODE != "quick":
    check(abs(N_BASE - 31.46e6) < 0.05e6, "the base model has 31.5M parameters")
WIDTH_PARAMS = {}
for w in qk([256, 512, 1024, 2048], [64, 128, 256, 512]):
    with torch.device("meta"):
        WIDTH_PARAMS[w] = sum(p.numel() for p in LM(**width_arch(w)).parameters())
print("width family parameters: " + ", ".join(f"{w}: {n / 1e6:.1f}M" for w, n in WIDTH_PARAMS.items()))

seed_all(0); _sp = LM(**width_arch(BASE_WIDTH_MUP, mup=False))
seed_all(0); _mp = LM(**width_arch(BASE_WIDTH_MUP, mup=True))
_ids = torch.randint(0, V_WIDTH, (2, 16))
check(torch.equal(_sp(_ids), _mp(_ids)) and _mp.logit_mult == 1.0,
      f"at the reference width {BASE_WIDTH_MUP}, the muP model is the standard model, bit for bit")
del _m, _sp, _mp

# %% [markdown]
r"""
**Learning-rate schedules.** All three start with a linear warmup over $W$ steps. They differ in
what follows:
- **cosine** follows half a cosine down to a floor of 10% of the peak, reached at the planned
  final step
- **WSD** (warmup–stable–decay) holds the peak, then decays to the same floor over the last
  `decay_frac` of the run, either linearly or as $1-\sqrt{x}$
- **constant** holds the peak forever

A WSD run planned for $N$ steps is identical to one planned for $M > N$ until its decay starts. So
a checkpoint taken in the flat phase can be decayed into a finished model at any time.
"""

# %%
def sched_frac(s, step, total):
    """Learning rate at update number step+1, as a fraction of the peak."""
    W, floor = s.get("warmup", 0), s.get("floor", 0.1)
    if step < W:
        return (step + 1) / W
    kind = s["kind"]
    if kind == "constant":
        return 1.0
    if kind == "cosine":
        p = min(1.0, (step - W) / max(1, total - W))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    if kind == "wsd":
        d0 = total - round(s["decay_frac"] * total)
        if step < d0:
            return 1.0
        p = min(1.0, (step - d0 + 1) / max(1, total - d0))
        shape = (1 - p) if s.get("shape", "linear") == "linear" else (1 - math.sqrt(p))
        return floor + (1 - floor) * shape
    raise ValueError(kind)


TOTAL, STOP = qk(300, 60), qk(200, 40)
WARM = qk(30, 6)
BRANCH, COOL = STOP - round(0.2 * STOP), round(0.2 * STOP)          # 160 and 40 in the full run
fig, ax = plt.subplots(figsize=(7.6, 3.0))
xs = np.arange(TOTAL)
for s, col, lab in ((dict(kind="cosine", warmup=WARM), C["blue"], "cosine"),
                    (dict(kind="wsd", warmup=WARM, decay_frac=0.2), C["orange"], "WSD, linear decay over the last 20%"),
                    (dict(kind="wsd", warmup=WARM, decay_frac=0.2, shape="sqrt"), C["yellow"], "WSD, 1−√ decay"),
                    (dict(kind="constant", warmup=WARM), C["aqua"], "constant")):
    ys = [sched_frac(s, i, TOTAL) for i in xs]
    ax.plot(xs + 1, ys, color=col); label_end(ax, TOTAL, ys[-1], lab, col)
ax.axvline(STOP, color=MUTED, lw=0.8, ls="--"); ax.text(STOP + 2, 0.5, f"stop at {STOP}", color=MUTED, fontsize=8.5)
ax.set(xlabel="step", ylabel="learning rate / peak", xlim=(0, TOTAL * 1.55), title=f"Schedules for a {TOTAL}-step plan")
finish(fig, "schedules")
check(all(sched_frac(dict(kind="wsd", warmup=WARM, decay_frac=0.2), i, TOTAL) ==
          sched_frac(dict(kind="wsd", warmup=WARM, decay_frac=COOL / STOP), i, STOP) for i in range(BRANCH)),
      f"WSD planned for {TOTAL} and WSD planned for {STOP} are identical until step {BRANCH}")

# %% [markdown]
r"""
**A hand-written AdamW.** PyTorch's AdamW has no switch for bias correction. So here is AdamW
written out line for line, in the same operation order PyTorch uses, with `bias_correction` as a
flag. With `track=True` it also computes, from the *same* optimizer state, the update both with and
without correction. Their ratio measures what turning correction off would have done at that exact
step (§4).
"""

# %%
class HandAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, bias_correction=True, track=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.bias_correction, self.track, self.counterfactual = bias_correction, track, None

    @torch.no_grad()
    def step(self, closure=None):
        cf = []
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if not st:
                    st["step"], st["exp_avg"], st["exp_avg_sq"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["step"] += 1
                t, m, v = st["step"], st["exp_avg"], st["exp_avg_sq"]
                p.mul_(1 - lr * wd)                                  # decoupled weight decay
                m.lerp_(g, 1 - b1)                                   # m = b1*m + (1-b1)*g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)              # v = b2*v + (1-b2)*g^2
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                if self.bias_correction:
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    p.addcdiv_(m, denom, value=-lr / bc1)
                else:
                    p.addcdiv_(m, v.sqrt().add_(eps), value=-lr)
                if self.track:
                    corrected = (m / bc1) / (v.sqrt() / math.sqrt(bc2) + eps)
                    uncorrected = m / (v.sqrt() + eps)
                    cf.append(torch.stack([uncorrected.norm(), corrected.norm()]))
        if self.track and cf:
            r = torch.stack(cf).float()
            self.counterfactual = (r[:, 0] / r[:, 1]).cpu()


def param_groups(model, wd, hidden_lr_mult=1.0):
    hidden = model.hidden_matrices()
    hid_ids = {id(p) for p in hidden}
    other_2d = [p for p in model.parameters() if p.ndim == 2 and id(p) not in hid_ids]
    gains = [p for p in model.parameters() if p.ndim < 2]
    return [dict(params=hidden, weight_decay=wd, lr_mult=hidden_lr_mult, name="hidden"),
            dict(params=other_2d, weight_decay=wd, lr_mult=1.0, name="embedding/head"),
            dict(params=gains, weight_decay=0.0, lr_mult=1.0, name="norm gains")]


def make_opts(model, oc):
    kind, lr, betas, eps, wd = oc["kind"], oc["lr"], tuple(oc.get("betas", (0.9, 0.95))), oc.get("eps", 1e-8), oc.get("wd", 0.1)
    mult = 1 / model.width_mult if model.mup else 1.0
    groups = param_groups(model, wd, mult)
    for g in groups:
        g["peak"] = lr * g["lr_mult"]
    if kind == "adamw":
        return [torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=DEV == "cuda")]
    if kind == "hand":
        return [HandAdamW(groups, lr=lr, betas=betas, eps=eps, bias_correction=oc.get("bias_correction", True),
                          track=oc.get("track", False))]
    if kind == "radam":
        return [torch.optim.RAdam(groups, lr=lr, betas=betas, eps=eps, decoupled_weight_decay=True)]
    if kind == "muon":
        hidden = dict(params=groups[0]["params"], peak=oc["muon_lr"])
        muon = torch.optim.Muon([hidden], lr=oc["muon_lr"], weight_decay=wd, momentum=0.95, nesterov=True,
                                adjust_lr_fn="match_rms_adamw")
        return [muon, torch.optim.AdamW(groups[1:], lr=lr, betas=betas, eps=eps, fused=DEV == "cuda")]
    raise ValueError(kind)

# %% [markdown]
r"""
**The training loop.** `train(cfg)` runs one configuration. Everything it measures is returned as
plain numbers, and every hook is optional:

| hook | what it records |
|---|---|
| `eval_at` | validation loss (and each eval batch's loss, for paired error bars) after those steps |
| `log_every` | each tensor's update/weight ratio, with and without its decay part, and Adam's step as a fraction of η |
| `detail_every` | spectral update/weight ratio for four matrices; the share of updates smaller than half a bf16 spacing; embedding rows by token frequency |
| `ema` | exponential moving averages of the weights, evaluated at `eval_at` |
| `ckpt_at` | in-memory copies of model and optimizer, to branch from |
| `sample_at` | a short generated story |

A run is marked **diverged** when its loss becomes non-finite or rises above the untrained model's
loss after warmup.
"""

# %%
@torch.no_grad()
def evaluate(model, fam):
    model.eval()
    micro = 8 if fam == "base" else (16 if model.emb.weight.shape[1] >= 2048 else 64)
    losses = []
    for xb, yb in eval_batches(fam, micro):
        with autocast():
            losses.append(model(xb, yb).float())
    model.train()
    per = torch.stack(losses).cpu().numpy()
    return dict(loss=float(per.mean()), batches=per)


@torch.no_grad()
def sample_story(model, prompt="Once upon a time", n=48, seed=0):
    model.eval()
    g = torch.Generator(device=DEV).manual_seed(seed)
    ids = torch.tensor([TOK.encode(prompt).ids], device=DEV)
    for _ in range(n):
        with autocast():
            logits = model(ids[:, -BASE_T:])[:, -1].float() / 0.8
        top = logits.topk(40)
        nxt = top.indices.gather(-1, torch.multinomial(top.values.softmax(-1), 1, generator=g))
        if nxt.item() == EOT:
            break
        ids = torch.cat([ids, nxt], 1)
    model.train()
    return TOK.decode(ids[0].tolist())


def _named_tensors(model):
    return [(n.replace("blocks.", "b").replace(".weight", ""), p) for n, p in model.named_parameters()]


def train(cfg):
    fam = cfg["family"]
    total, stop = cfg["steps"], cfg.get("stop", cfg["steps"])
    seed_all(cfg["seed"])
    model = make_model(cfg).to(DEV)
    opts = make_opts(model, cfg["opt"])
    scaler = torch.amp.GradScaler("cuda") if (AMP == "fp16" and DEV == "cuda") else None
    start = 0
    if cfg.get("resume") is not None:
        ck = cfg["resume"]
        model.load_state_dict(ck["model"])
        for o, sd in zip(opts, ck["opts"]):
            o.load_state_dict(sd)
        start = ck["step"]
    B = BASE_B if fam == "base" else WIDTH_B
    accum = BASE_ACCUM if fam == "base" else (4 if cfg.get("width", 0) >= 2048 else 1)
    names = [n for n, _ in _named_tensors(model)]
    params = [p for _, p in _named_tensors(model)]
    group_of = {}
    for o in opts:
        for g in o.param_groups:
            for p in g["params"]:
                group_of[id(p)] = g
    hist = dict(loss=[], lr=[], gnorm=[], clipped=[])
    out = dict(evals={}, ratios=dict(steps=[], total=[], grad=[], adam_frac=[]), tensors=names,
               detail=dict(steps=[], spectral=[], bf16_lost=[], bf16_lost_floor=[], emb_rows=[]), samples={}, ema_evals={},
               counterfactual=dict(steps=[], ratio=[]), diverged_at=None)
    log_every, detail_every = cfg.get("log_every", 0), cfg.get("detail_every", 0)
    spectral_ids = [names.index(n) for n in ("b0.q", f"b{len(model.blocks) // 2}.up", f"b{len(model.blocks) - 1}.down", "emb")]
    ema_decays = cfg.get("ema") or []
    ema = [[p.detach().float().clone().zero_() for p in params] for _ in ema_decays]
    ckpts, loss0 = {}, None
    wd_frac = (lambda: [group_of[id(p)]["lr"] * group_of[id(p)]["weight_decay"] for p in params])
    for s in range(start, stop):
        frac = sched_frac(cfg["sched"], s, total)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["peak"] * frac
        win = WIN[fam].batch(cfg["seed"], s, B)
        if fam == "width":
            win = REMAP[win]
        win = win.to(DEV, non_blocking=True)
        loss_acc = torch.zeros((), device=DEV)
        for chunk in win.chunk(accum):
            with autocast():
                loss = model(chunk[:, :-1], chunk[:, 1:])
            (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            loss_acc += loss.detach() / accum
        if scaler:
            for o in opts:
                scaler.unscale_(o)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        log_now = log_every and ((s + 1) % log_every == 0 or s < 10 or (s + 1) in cfg.get("eval_at", []))
        detail_now = detail_every and ((s + 1) % detail_every == 0 or s == 0)
        if log_now or detail_now:
            before = [p.detach().clone() for p in params]
            decay = wd_frac()
        if scaler:
            for o in opts:
                scaler.step(o)
            scaler.update()
        else:
            for o in opts:
                o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        L = loss_acc.item()
        loss0 = L if loss0 is None else loss0
        hist["loss"].append(L); hist["lr"].append(frac); hist["gnorm"].append(gnorm.item())
        hist["clipped"].append(bool(gnorm.item() > cfg.get("clip", 1.0)))
        if not math.isfinite(L) or (s >= cfg["sched"].get("warmup", 0) and s > 5 and L > loss0 + 0.5):
            out["diverged_at"] = s + 1
            break
        if log_now or detail_now:
            with torch.no_grad():
                after = [p.detach() for p in params]
                delta = torch._foreach_sub(after, before)
                wn = torch.stack(torch._foreach_norm(before)).float()
                dn = torch.stack(torch._foreach_norm(delta)).float()
                torch._foreach_add_(delta, torch._foreach_mul(before, decay))     # remove the decay part exactly
                gn = torch.stack(torch._foreach_norm(delta)).float()
                numel = torch.tensor([p.numel() for p in params], device=DEV, dtype=torch.float32)
                lrs = torch.tensor([group_of[id(p)]["lr"] for p in params], device=DEV, dtype=torch.float32)
                if log_now:
                    out["ratios"]["steps"].append(s + 1)
                    out["ratios"]["total"].append((dn / wn).cpu())
                    out["ratios"]["grad"].append((gn / wn).cpu())
                    out["ratios"]["adam_frac"].append((gn / (lrs * numel.sqrt())).cpu())
                if detail_now:
                    full_delta = torch._foreach_sub(after, before)
                    out["detail"]["steps"].append(s + 1)
                    out["detail"]["spectral"].append([
                        (torch.linalg.matrix_norm(full_delta[i].float(), ord=2) /
                         torch.linalg.matrix_norm(before[i].float(), ord=2)).item() for i in spectral_ids])
                    lost, lost_floor = [], []
                    for b_, d_ in zip(before, full_delta):
                        half_ulp = torch.ldexp(torch.ones_like(b_), torch.frexp(b_).exponent - 9)
                        lost.append((d_.abs() < half_ulp).float().mean())
                        lost_floor.append((0.1 * d_.abs() < half_ulp).float().mean())      # the same step at a 10x smaller lr
                    out["detail"]["bf16_lost"].append(torch.stack(lost).cpu())
                    out["detail"]["bf16_lost_floor"].append(torch.stack(lost_floor).cpu())
                    e_i = names.index("emb")
                    row_r = full_delta[e_i].norm(dim=1) / before[e_i].norm(dim=1).clamp_min(1e-12)
                    present = torch.zeros(row_r.shape[0], dtype=torch.bool, device=DEV)
                    present[win[:, :-1].reshape(-1)] = True
                    fr = FREQ_RANK.to(DEV) if fam == "base" else torch.arange(row_r.shape[0], device=DEV)
                    buckets = [(fr < 100), (fr >= 100) & (fr < 1000), (fr >= 1000) & (fr < 8000), (fr >= 8000)]
                    out["detail"]["emb_rows"].append([row_r[bk].mean().item() for bk in buckets] +
                                                     [row_r[present].mean().item(), row_r[~present].mean().item() if (~present).any() else None])
                    del full_delta
                del before, delta, after                 # each is a full copy of the model: free it before the next forward pass
        if getattr(opts[0], "counterfactual", None) is not None and ((s + 1) % 5 == 0 or s < 30):
            out["counterfactual"]["steps"].append(s + 1)
            out["counterfactual"]["ratio"].append(opts[0].counterfactual)
        if ema:
            with torch.no_grad():
                for d, e in zip(ema_decays, ema):
                    torch._foreach_lerp_(e, [p.detach().float() for p in params], 1 - d)
        if (s + 1) in cfg.get("ckpt_at", []):
            ckpts[s + 1] = dict(step=s + 1, model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                opts=[copy.deepcopy(_to_cpu(o.state_dict())) for o in opts])
        if (s + 1) in cfg.get("eval_at", []):
            out["evals"][s + 1] = evaluate(model, fam)
            for d, e in zip(ema_decays, ema):
                n_upd = s + 1 - start
                saved = [p.detach().clone() for p in params]
                with torch.no_grad():
                    for p, ev in zip(params, e):
                        p.copy_(ev / (1 - d ** n_upd))                 # bias-corrected average, as in Adam
                out["ema_evals"].setdefault(str(d), {})[s + 1] = evaluate(model, fam)["loss"]
                with torch.no_grad():
                    for p, sv in zip(params, saved):
                        p.copy_(sv)
        if (s + 1) in cfg.get("sample_at", []) and TOK is not None and fam == "base":
            out["samples"][s + 1] = sample_story(model)
    if cfg.get("diag") and out["diverged_at"] is None:
        xb, _ = eval_batches(fam, 8 if fam == "base" else 16)[0]
        out["diag"] = model.diagnostics(xb[:4])
    out["hist"] = hist
    out["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else None
    for k in ("ratios", "detail"):
        for kk, vv in out[k].items():
            out[k][kk] = [x.tolist() if isinstance(x, torch.Tensor) else x for x in vv]
    out["counterfactual"]["ratio"] = [x.tolist() for x in out["counterfactual"]["ratio"]]
    for st in out["evals"]:
        out["evals"][st]["batches"] = out["evals"][st]["batches"].tolist()
    del model, opts
    free_gpu()
    return out, ckpts


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj

# %% [markdown]
r"""
**Helpers for naming, running and reading experiments.** A configuration's name is built from its
settings, so two sections that need the same run share one stored result instead of training
twice.
"""

# %%
def cfg_name(cfg, tag=""):
    o, s = cfg["opt"], cfg["sched"]
    fam = "base" if cfg["family"] == "base" else f"w{cfg['width']}{'-mup' if cfg.get('mup') else ''}"
    opt = o["kind"] + ("" if o.get("bias_correction", True) else "-nobc") + (f"-b2{o['betas'][1]}" if o.get("betas") else "")
    lr = f"lr{o['lr']:.3g}" + (f"-mu{o['muon_lr']:.3g}" if o["kind"] == "muon" else "")
    sch = s["kind"] + f"-w{s.get('warmup', 0)}" + (f"-d{s['decay_frac']:g}{s.get('shape', 'linear')[0]}" if s["kind"] == "wsd" else "") \
        + (f"-fl{s['floor']:g}" if "floor" in s else "")
    stop = f"-stop{cfg['stop']}" if cfg.get("stop", cfg["steps"]) != cfg["steps"] else ""
    return f"{fam}|{opt}|{lr}|{sch}|T{cfg['steps']}{stop}|s{cfg['seed']}" + (f"|{tag}" if tag else "")


BASE_HOOKS = dict(eval_at=sorted({qk(100, 20), qk(150, 30), STOP, qk(240, 48), qk(250, 50), TOTAL}),
                  log_every=10, sample_at=[STOP, TOTAL])


def base_cfg(lr, seed=0, kind="cosine", warmup=WARM, opt="adamw", **kw):
    sched = dict(kind=kind, warmup=warmup)
    sched.update({k: kw.pop(k) for k in ("decay_frac", "shape", "floor") if k in kw})
    o = dict(kind=opt, lr=lr, betas=kw.pop("betas", (0.9, 0.95)), wd=0.1)
    o.update({k: kw.pop(k) for k in ("bias_correction", "track", "muon_lr") if k in kw})
    cfg = dict(family="base", seed=seed, steps=kw.pop("steps", TOTAL), sched=sched, opt=o, clip=1.0)
    cfg.update(BASE_HOOKS)
    cfg.update(kw)
    return cfg


def run(cfg, tag=""):
    """Run (or reuse) one configuration; returns the stored result."""
    return experiment(cfg_name(cfg, tag), lambda: train(cfg)[0])


def final_loss(res, step=None):
    """Validation loss at `step` (default: the last evaluation); inf if the run diverged first."""
    if res is None:
        return float("nan")
    ev = res["evals"]
    key = str(step) if step is not None else max(ev, key=int, default=None)
    if key is None or key not in ev:
        return float("inf")
    return ev[key]["loss"]


def ema_curve(xs, n=20):
    """Bias-corrected exponential moving average with a time constant of n steps (the same correction Adam uses)."""
    out, m, a = [], 0.0, 1 / n
    for i, x in enumerate(xs, 1):
        m = (1 - a) * m + a * x
        out.append(m / (1 - (1 - a) ** i))
    return np.array(out)


def lr_sweep(make_cfg, lrs, metric, factor=2.0, max_ext=2, tag=""):
    """Run make_cfg(lr) for each lr; extend the grid at an edge whose point is the best, up to max_ext times."""
    res = {}
    lrs = sorted(lrs)
    for lr in lrs:
        res[lr] = run(make_cfg(lr), tag)
    for _ in range(max_ext):
        finite = {lr: metric(r) for lr, r in res.items() if math.isfinite(metric(r))}
        if not finite:
            break
        best = min(finite, key=finite.get)
        lo, hi = min(res), max(res)
        if best == lo:
            new = lo / factor
        elif best == hi:
            new = hi * factor
        else:
            break
        res[new] = run(make_cfg(new), tag)
    return dict(sorted(res.items()))

# %% [markdown]
r"""
**Checking the hand-written AdamW against PyTorch's, at full model scale.** §1 checked one weight.
Here every one of the base model's 31.5M weights is checked:
- PyTorch's AdamW and `HandAdamW` each update their own copy of the parameters for 20 steps.
- At each step both get the *same* gradient, computed on the first copy and in float32.
- Any difference that builds up is pure optimizer arithmetic.
"""

# %%
def hand_vs_torch(steps=20):
    seed_all(0)
    m_a = LM(**BASE_ARCH).to(DEV)
    m_b = copy.deepcopy(m_a)
    pa, pb = list(m_a.parameters()), list(m_b.parameters())
    opt_a = torch.optim.AdamW(param_groups(m_a, 0.1), lr=2e-3, betas=(0.9, 0.95), eps=1e-8, foreach=False)
    opt_b = HandAdamW(param_groups(m_b, 0.1), lr=2e-3, betas=(0.9, 0.95), eps=1e-8)
    worst = []
    for s in range(steps):
        win = WIN["base"].batch(0, s, 4).to(DEV)
        m_a(win[:, :-1], win[:, 1:]).backward()                     # float32, no autocast
        for a, b in zip(pa, pb):
            b.grad = a.grad.clone()
        opt_a.step(); opt_b.step()
        opt_a.zero_grad(); opt_b.zero_grad()
        worst.append(max((a - b).abs().max().item() for a, b in zip(pa, pb)))
    del m_a, m_b, opt_a, opt_b
    free_gpu()
    return dict(max_abs_diff_per_step=worst, n_params=sum(p.numel() for p in pa))


hv = experiment("handadamw_vs_torch", hand_vs_torch)
if hv:
    print(f"largest |difference| across {hv['n_params']:,} weights after each of 20 steps: "
          + ", ".join(f"{x:.0e}" for x in hv["max_abs_diff_per_step"][::4]))
    check(max(hv["max_abs_diff_per_step"]) < 1e-6, "HandAdamW matches torch.optim.AdamW on every weight of the base model for 20 steps")
