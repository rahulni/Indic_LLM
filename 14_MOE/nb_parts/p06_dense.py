from . import md, code, mathbox

CELLS = [
md(r'''
---
## E · Stage 1: train the dense ("linear") model

### The training loop, written once and reused by every run

* **AdamW** (β = 0.9/0.95, weight decay 0.1 on matrices, none on norms or the router), gradient
  clipping at 1.0, bf16 autocast (fp16 + `GradScaler` on a T4).
* **Warmup-stable-decay (WSD) schedule.** Stage 1 warms up and then stays at the peak learning rate.
  It never decays, because a model you are about to convert should still be in its fast-learning
  phase. Both branches after T1 share one schedule: a 50-step re-warm, a flat stretch, and a linear
  decay over the last 20% of T1→T3.
* **Micro-batches.** A batch of 64 sequences runs as 2 micro-batches of 32 with gradient
  accumulation, so the MoE fits in 8 GB.
* **After every optimizer step** each MoE layer nudges its balancing bias, using the load
  counted over the *whole* batch, both micro-batches.
* **Logging.** Train loss every step; validation loss, per-expert load, MaxVio and the router
  z-diagnostic every `eval_every` steps.
* **Hot-plug.** Large runs read knobs such as the balancing weight from a file, so they can be
  changed without stopping. Every 50 steps the loop re-reads `hotplug.json` if it exists, so
  `gamma`, `seq_alpha`, `lr_mult` or `freeze_router` can be changed while a run is going.

🧠 **Intuition: why convert in the stable phase.** A model whose learning rate has already decayed
has settled into a narrow valley. Surgery knocks it out of that valley, and with a tiny learning
rate it cannot climb back to a good place. With the learning rate still high, it re-settles
quickly.

🧠 **Intuition: batch size and learning rate are coupled.** Each step follows a noisy estimate of
the gradient. A smaller batch is a noisier estimate. To keep the same amount of "shaking", the step
must shrink with the batch. This notebook learned that the hard way: its first full run gave the
labs a 4× smaller batch at the same learning rate, and every lab's loss *rose*.
'''),
mathbox("the WSD schedule, and the noise scale η/B", r'''
**Warmup-stable-decay**, with warmup length $t_w$, decay start $t_d$ and end $t_{\text{end}}$:

$$\eta(t) = \begin{cases} \eta_{\max}\, t / t_w & t < t_w \\ \eta_{\max} & t_w \le t < t_d \\ \eta_{\max}\Big(1 - 0.9\,\dfrac{t - t_d}{t_{\text{end}} - t_d}\Big) & t \ge t_d \end{cases}$$

Ours: $\eta_{\max} = 1.2\times10^{-3}$. Stage 1 never reaches $t_d$. After T1 there is a 50-step re-warm,
then the last 20% of T1→T3 decays to $0.1\,\eta_{\max}$. The bias speed $\gamma$ is set to 0 for that last
stretch, as in DeepSeek-V3.

**The noise scale.** A minibatch of $B$ sequences gives $\hat g = g + \xi$, with
$\operatorname{Cov}(\xi) \approx \Sigma / B$. One step $\Delta\theta = -\eta \hat g$ adds noise of covariance
$\eta^2 \Sigma / B$.

To make the same progress you need about $1/\eta$ steps, so the noise accumulated over that
progress is about $\eta\,\Sigma / B$. The "temperature" of training is $\eta / B$.

| run | $\eta / B$ |
|---|---|
| main runs | $1.2\times10^{-3} / 64 = 1.9\times10^{-5}$ |
| first labs (batch 16 at the same $\eta$) | $1.2\times10^{-3} / 16 = 7.5\times10^{-5}$, 4× hotter: the loss rose |
| labs now ($\eta \times 16/64$) | $3\times10^{-4} / 16 = 1.9\times10^{-5}$, the main runs' temperature |
'''),
code(r'''
def amp():
    return torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=PREC != "fp32")


def make_opt(model, lr=None):
    dec, nodec = [], []
    for n, p in model.named_parameters():
        (dec if p.dim() >= 2 and not n.endswith("router") else nodec).append(p)
    return torch.optim.AdamW([{"params": dec, "weight_decay": 0.1}, {"params": nodec, "weight_decay": 0.0}],
                             lr=lr or BUDGET["peak_lr"], betas=(0.9, 0.95), eps=1e-8)


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=PREC == "fp16")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=PREC == "fp16")


@torch.no_grad()
def evaluate(model, n_batches, B=32, per_batch=False):
    was = model.training
    model.eval()
    out = []
    for i in range(n_batches):
        x, y = VAL.batch(i * B, B)
        with amp():
            _, l = model(x, y)
        out.append(l.item())
    model.train(was)
    return out if per_batch else float(np.mean(out))


PROBE = None
def probe_batch():
    global PROBE
    if PROBE is None:
        PROBE = VAL.batch(10_000, 8)[0]          # fixed 8 x 256 tokens, never used for val loss
    return PROBE


@torch.no_grad()
def route_probe(model):
    """Top-k ids per layer for the probe batch, in eval mode (hard top-k, no Gumbel)."""
    moes = model.moe_layers()
    was = model.training
    model.eval()
    for m in moes:
        m.record = True
    with amp():
        model(probe_batch())
    out = [m.last["topi"].clone() for m in moes]
    for m in moes:
        m.record = False
    model.train(was)
    return out


def family_share(topis, fam):
    """Fraction of tokens whose k choices all come from one clone family (fam clones per family)."""
    return float(np.mean([((t // fam) == (t[:, :1] // fam)).all(1).float().mean().item() for t in topis]))


def lr_stage1(s):
    return BUDGET["peak_lr"] * min(1.0, (s + 1) / BUDGET["warmup"])

POST = S2 + S3                      # steps after T1 on both branches
DECAY_START = int(0.8 * POST)
def lr_post(s):
    peak = BUDGET["peak_lr"]
    if s < DECAY_START:
        return peak * min(1.0, 0.1 + 0.9 * (s + 1) / BUDGET["rewarm"])
    return peak * (1 - 0.9 * (s - DECAY_START) / max(1, POST - DECAY_START))
def gamma_post(s):                  # DeepSeek-V3: bias updates stop for the final stretch
    return 1.0 if s < DECAY_START else 0.0
LAB_PEAK = BUDGET["peak_lr"] * BUDGET["lab_batch"] / BUDGET["batch"]   # same lr/batch noise scale as the main runs
def lr_lab(s):
    return LAB_PEAK * min(1.0, 0.1 + 0.9 * (s + 1) / BUDGET["rewarm"])
def const1(s):
    return 1.0


SNAP = {}                            # probe routing snapshots (kept in memory, not in results.json)

def train_run(name, model, opt, cursor, steps, B, lr_fn, *, s0=0, gamma_fn=const1, eval_every=50,
              val_batches=16, gumbel_steps=0, offset_fn=None, snap_at=(), eta=False, tokens0=0.0):
    """Train `steps` steps from batch `cursor`. s0 = position inside the schedule (for stage 3)."""
    scaler = make_scaler()
    moes = model.moe_layers()
    fam = moes[0].cfg.family if moes else 1
    log = {k: [] for k in ("loss", "gnorm", "lr", "eval_step", "val", "load", "maxvio", "z", "bias", "family", "tok_s")}
    log.update(name=name, steps=steps, batch=B, tokens0=tokens0, s0=s0,
               params=n_params(model), active=n_active(model))
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hp, t_last, tok_last = {}, time.time(), 0
    t_start = t_last
    model.train()
    for s in range(steps):
        si = s0 + s
        if s % 50 == 0:
            hp = read_hotplug()
            for m in moes:
                m.cfg.seq_alpha = hp.get("seq_alpha", m.cfg.seq_alpha)
                m.router.requires_grad_(not hp.get("freeze_router", False))
        lr = lr_fn(si) * hp.get("lr_mult", 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        for m in moes:
            m.gumbel = s < gumbel_steps
        if offset_fn is not None:
            offset_fn(s)
        if s in snap_at:
            SNAP.setdefault(name, {})[s] = route_probe(model)
        nmb = max(1, B // BUDGET["micro_batch"])            # micro-batches; load is still counted over the whole batch
        loss = 0.0
        for j in range(nmb):
            x, y = TRAIN.batch(cursor + j * (B // nmb), B // nmb)
            with amp():
                _, l = model(x, y)
                total = (l + model.aux_loss()) / nmb
            scaler.scale(total).backward()
            loss = loss + l.detach() / nmb
        cursor += B
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        for m in moes:
            m.update_bias(hp.get("gamma", m.cfg.gamma) * gamma_fn(si))
        log["loss"].append(round(loss.item(), 4))
        log["gnorm"].append(round(gn.item(), 4))
        log["lr"].append(lr)
        if eta and s == min(25, steps - 1):
            report_eta((time.time() - t_start) / (s + 1))
        if (s + 1) % eval_every == 0 or s == steps - 1:
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            dt = time.time() - t_last
            log["tok_s"].append(round((s + 1 - tok_last) * B * GCFG.ctx / max(dt, 1e-9)))
            tok_last = s + 1
            log["eval_step"].append(s + 1)
            log["val"].append(round(evaluate(model, val_batches), 4))
            if moes:
                loads = [m.read_window() for m in moes]
                fr = [(c / c.sum() * m.k).tolist() for c, m in zip(loads, moes)]          # share of tokens per expert
                log["load"].append([[round(v, 4) for v in f] for f in fr])
                log["maxvio"].append([round(((c.max() - c.mean()) / c.mean()).item(), 4) for c in loads])
                log["z"].append(round(float(np.mean([m.last["z"].item() for m in moes])), 3))
                log["bias"].append(round(float(max(m.bias.abs().max().item() for m in moes)), 4))
                if fam > 1:
                    log["family"].append(round(family_share(route_probe(model), fam), 4))
            if s + 1 < steps and len(log["eval_step"]) % 4 == 0:
                print(f"  {name:<14} step {s + 1:>5}/{steps}  train {np.mean(log['loss'][-eval_every:]):.4f}"
                      f"  val {log['val'][-1]:.4f}  {log['tok_s'][-1]:,} tok/s")
            t_last = time.time()
    log["minutes"] = round((time.time() - t_start) / 60, 2)
    print(f"  {name:<14} done: val {log['val'][-1]:.4f}  params {log['params'] / 1e6:.1f}M  active {log['active'] / 1e6:.1f}M"
          f"  ({log['minutes']:.1f} min)")
    if s + 1 in snap_at:
        SNAP.setdefault(name, {})[s + 1] = route_probe(model)
    log["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if DEVICE == "cuda" else None
    R["runs"][name] = log
    save_results()
    return cursor, log


def report_eta(sec_per_step):
    """Projection from the first steps of Stage 1. MoE steps are assumed 1.4x slower (measured in Part M)."""
    rows = [("Stage 1 dense", S1, 1.0, BUDGET["batch"]), ("Stage 2+3 MoE", S2 + S3, 1.4, BUDGET["batch"]),
            ("dense control", S2 + S3, 1.0, BUDGET["batch"]), ("15 lab runs", 15 * SLAB, 1.3, BUDGET["lab_batch"])]
    tot = 0.0
    print(f"  ETA from the first steps ({sec_per_step * 1000:.0f} ms/step at batch {BUDGET['batch']}):")
    for label, n, slow, b in rows:
        t = n * sec_per_step * slow * (b / BUDGET["batch"]) ** 0.7     # small batches are less efficient
        tot += t
        print(f"    {label:<15} ~{t / 60:5.1f} min")
    print(f"    {'total training':<15} ~{tot / 60:5.1f} min")
    R["meta"]["eta_min"] = round(tot / 60, 1)


def save_ckpt(tag, model, opt, cursor):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "moe": asdict(model.moe_cfg) if model.moe_cfg else None, "cursor": cursor}, CKPT / f"{tag}.pt")


def has_ckpt(tag):
    return COMPUTE and RESUME and (CKPT / f"{tag}.pt").exists()


def load_ckpt(tag):
    ck = torch.load(CKPT / f"{tag}.pt", map_location=DEVICE, weights_only=False)
    model = GPT(GCFG, MoEConfig(**ck["moe"]) if ck["moe"] else None).to(DEVICE)
    model.load_state_dict(ck["model"])
    opt = make_opt(model)
    opt.load_state_dict(ck["opt"])
    return model, opt, ck["cursor"]
'''),
md(r'''
### Train it

Stage 1 trains the dense model for 25M tokens at constant learning rate after warmup. The first few
steps are timed, and the cell prints a projected run time for the whole notebook (useful on Colab).
'''),
code(r'''
dense = dense_opt = None
CUR_T1 = S1 * BUDGET["batch"]
if COMPUTE:
    if has_ckpt("dense_T1") and have("runs", "dense"):
        dense, dense_opt, CUR_T1 = load_ckpt("dense_T1")
        print("loaded Stage 1 checkpoint")
    else:
        seed_all(1)
        dense = GPT(GCFG).to(DEVICE)
        dense_opt = make_opt(dense)
        print(f"dense model: {n_params(dense) / 1e6:.2f}M parameters")
        CUR_T1, _ = train_run("dense", dense, dense_opt, 0, S1, BUDGET["batch"], lr_stage1,
                              eval_every=BUDGET["eval_every"], val_batches=BUDGET["val_batches"], eta=True)
        save_ckpt("dense_T1", dense, dense_opt, CUR_T1)
    if not have("val_T1"):
        R["val_T1"] = evaluate(dense, BUDGET["val_batches"])
        save_results()
'''),
code(r'''
def smooth(v, k=25):
    """Trailing mean over k steps; the first k-1 points average whatever is available."""
    v = np.asarray(v, float)
    c = np.cumsum(np.insert(v, 0, 0.0))
    n = np.minimum(np.arange(1, len(v) + 1), k)
    return (c[1:] - c[np.arange(1, len(v) + 1) - n]) / n

def tokens_axis(log, steps):
    return (log["tokens0"] + np.asarray(steps) * log["batch"] * GCFG.ctx) / 1e6

if have("runs", "dense"):
    L = R["runs"]["dense"]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(tokens_axis(L, np.arange(1, len(L["loss"]) + 1)), smooth(L["loss"]), color=DENSE_C, alpha=0.35, lw=1.2, label="train (smoothed)")
    ax.plot(tokens_axis(L, L["eval_step"]), L["val"], color=DENSE_C, marker="o", ms=3.5, label="validation")
    ax.set(xlabel="tokens seen (M)", ylabel="cross-entropy (nats)", title="Stage 1: the dense model learns TinyStories")
    ax.set_ylim(top=min(4.5, max(L["val"]) + 0.2))
    ax.legend()
    ax.annotate(f"T1: val {L['val'][-1]:.3f}", (tokens_axis(L, [L['eval_step'][-1]])[0], L["val"][-1]),
                xytext=(-80, 25), textcoords="offset points", color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    savefig(fig, "stage1_dense")
    print(f"dense: {L['params'] / 1e6:.2f}M parameters, val loss at T1 = {R['val_T1']:.4f}")
'''),
md(r'''
**Carry this forward:** the dense model is the starting point, not a baseline to beat. It is
converted while its loss is still falling fast, because upcycling pays off only when the dense
model still has a lot left to learn.
'''),
]
