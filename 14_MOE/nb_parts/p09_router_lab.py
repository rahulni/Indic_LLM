from . import md, code, mathbox

CELLS = [
md(r'''
---
## H · Router lab: balancing, and the score function

> Routing collapse, the auxiliary loss, the loss-free bias, and softmax vs sigmoid: an open
> question that can only be settled by testing it.

All runs start from the same partition conversion of the T1 model (the Stage 2 shape: shared
768 + 8 × 192, top-4) and read the same batches. Only the router's rules change.

| run | score | balancing | what it tests |
|---|---|---|---|
| no balancing | sigmoid | none | does the router collapse by itself? |
| auxiliary loss | sigmoid | Switch loss, α = 0.01 | the 2021 fix: a gradient that fights the language loss |
| **loss-free bias** | sigmoid | bias, γ = 0.001, + seq loss 1e-4 | the main recipe (same run as "partition [main]" in G) |
| softmax | softmax | bias | Qwen3's score function |
| √softplus | √softplus | bias | DeepSeek-V4's score function |

**Definitions.** An expert's *load* is its share of tokens; the fair share is k/E = 4/8 = 50%.
**MaxVio** = (busiest − mean)/mean, per layer. An expert is **nearly dead** if its share over the last
20% of the run is below 10% of the fair share.

🧠 **Intuition: the favourite child.** A parent gives one child slightly
more attention. That child does a little better, so gets more attention, so does better still.
Meanwhile the others are ignored and stop improving. A router does the same: the expert it picks
gets more gradient, becomes better, and is picked more. Nothing in the language loss stops this.
Balancing is the rule that every child gets their turn.
'''),
mathbox("collapse as positive feedback, and the two restoring forces", r'''
**A cartoon of the loop.** Let $p_i$ be expert $i$'s share of tokens, $q_i$ its quality and $r_i$ the
router's logit for it.

* Experts improve in proportion to the tokens they receive: $\dot q_i \propto p_i$.
* The language-loss gradient moves the router toward better experts: $\dot r_i \propto q_i - \bar q$.
* $p = \operatorname{softmax}(r + b)$.

Together these give the rich-get-richer equation

$$\frac{d\,p_i}{dt} \;\propto\; p_i\,\big(q_i - \bar q_p\big), \qquad \bar q_p = \sum_j p_j q_j.$$

An expert that is slightly ahead grows exponentially. The uniform state is an *unstable* fixed
point, so any small initial asymmetry ends in collapse. (This is a cartoon, not the real network.)

**Restoring force 1: the auxiliary loss.** It adds $-\alpha E\, p_i\big(f_i - \textstyle\sum_j f_j p_j\big)$ to
the logit's gradient: busy experts get pushed down. But it acts on the *same* logits that the
language loss is pushing up, so the two objectives fight. A small $\alpha$ under-corrects; a large
$\alpha$ hurts the language model.

**Restoring force 2: the bias.** $b_i \leftarrow b_i + \gamma\,\operatorname{sign}(1/E - p_i)$ acts outside the
gradient, as an integral controller. Its only fixed point is $p_i = 1/E$, and it cannot be
outvoted by the language loss, because it is not part of it.
'''),
md(r'''
**See the math: a 30-line collapse simulator.** Eight experts, the cartoon dynamics above, 400
steps, with no balancing, with an auxiliary loss, and with the bias. Each line is one expert's share
of tokens. This is the loop in isolation. The real lab follows.
'''),
code(r'''
def collapse_sim(mode, E=8, steps=400, seed=0, lr=0.02, learn=0.05, alpha=0.5, gamma=0.02):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(E, generator=g) * 0.05          # router logits: the learned preference
    q = torch.zeros(E)                              # expert quality
    b = torch.zeros(E)                              # balancing bias (selection only)
    hist = []
    for _ in range(steps):
        p = torch.softmax(r + b, 0)                 # share of tokens each expert receives
        hist.append(p.clone())
        q += learn * p                              # experts improve with the tokens they get
        grad = q - q.mean()                         # language loss: route toward better experts
        if mode == "aux":                           # minus the gradient of alpha*E*sum f_i P_i (f = p, held fixed)
            grad = grad - alpha * E * p * (p - (p * p).sum())
        r += lr * grad
        if mode == "bias":
            b += gamma * torch.sign(1 / E - p)
    return torch.stack(hist)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), sharey=True)
for ax, (mode, title) in zip(axes, (("none", "no balancing"), ("aux", "auxiliary loss"), ("bias", "loss-free bias"))):
    h = collapse_sim(mode)
    for i in range(h.shape[1]):
        ax.plot(h[:, i], color=SERIES[i % 8], lw=1.6)
    ax.axhline(1 / 8, color=MUTED, lw=1)
    ax.set(title=f"{title}: biggest share {h[-1].max():.0%}", xlabel="step")
axes[0].set_ylabel("share of tokens (fair = 12.5%)")
savefig(fig, "demo_collapse_sim")
_none, _bias = collapse_sim("none"), collapse_sim("bias")
assert _none[-1].max() > 0.8 and _bias[-1].max() < 0.2          # the loop collapses; the controller holds
'''),
code(r'''
def tail_shares(log, frac=0.2):
    loads = np.asarray(log["load"], dtype=float)                 # [evals, layers, E]
    n = max(1, int(round(len(loads) * frac)))
    return loads[-n:].mean(0)                                    # [layers, E], sums to k per layer

def nearly_dead(log, k, frac=0.2):
    sh = tail_shares(log, frac)
    return (sh < 0.1 * k / sh.shape[1]).sum(1)                   # per layer

def tail_maxvio(log, frac=0.2):
    mv = np.asarray(log["maxvio"], dtype=float)
    n = max(1, int(round(len(mv) * frac)))
    return float(mv[-n:].mean())

ROUTER_RUNS = {"none": dict(balance="none", seq_alpha=0.0), "aux": dict(balance="none", seq_alpha=0.0, switch_alpha=0.01),
               "softmax": dict(score="softmax"), "sqrt_softplus": dict(score="sqrt_softplus")}
if COMPUTE:
    for key, ov in ROUTER_RUNS.items():
        name = f"router:{key}"
        if have("runs", name):
            continue
        seed_all(10)
        mdl, opt, _ = convert(dense, dense_opt, "partition", seed=0, **ov)
        lab_run(name, mdl, opt, CUR_T1)
        del mdl, opt
        torch.cuda.empty_cache()

def router_log(key):
    return R["runs"]["conv:partition" if key in ("bias", "sigmoid") else f"router:{key}"]
'''),
code(r'''
BAL = [("none", "no balancing", SERIES[2]), ("aux", "auxiliary loss (Switch, 0.01)", SERIES[3]), ("bias", "loss-free bias [main]", SERIES[1])]
if have("runs", "router:none") and have("runs", "conv:partition"):
    none_mv = np.asarray(router_log("none")["maxvio"])[-1]
    L_SHOW = int(np.argmax(none_mv))
    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.45, wspace=0.25)
    for i, (key, label, _) in enumerate(BAL):
        L = router_log(key)
        ld = np.asarray(L["load"])[:, L_SHOW, :]                     # [evals, E]
        ax = fig.add_subplot(gs[0, i])
        k = MAIN_MOE8.top_k
        im = ax.imshow(ld.T / (k / ld.shape[1]), aspect="auto", cmap=SEQ, vmin=0, vmax=2.5, interpolation="nearest",
                       extent=[0, L["eval_step"][-1] * L["batch"] * GCFG.ctx / 1e6, ld.shape[1] - 0.5, -0.5])
        ax.set(title=f"{label}\nlayer {L_SHOW + 1}", xlabel="tokens (M)", ylabel="expert" if i == 0 else None)
        ax.grid(False)
    cb = fig.colorbar(im, ax=fig.axes[:3], fraction=0.02, pad=0.01)
    cb.set_label("load / fair share", color=INK2)
    a = fig.add_subplot(gs[1, :2])
    for key, label, color in BAL:
        L = router_log(key)
        a.plot(np.asarray(L["eval_step"]) * L["batch"] * GCFG.ctx / 1e6, np.asarray(L["maxvio"]).mean(1), color=color, label=label)
    a.set(xlabel="tokens after T1 (M)", ylabel="MaxVio (mean over layers)", title="Imbalance over training (0 = perfectly even)")
    a.legend()
    t = fig.add_subplot(gs[1, 2]); t.axis("off")
    short = {"none": "none", "aux": "aux loss", "bias": "bias [main]"}
    cells = [[short[key], f"{router_log(key)['val'][-1]:.4f}", f"{tail_maxvio(router_log(key)):.2f}",
              str(int(nearly_dead(router_log(key), 4).sum()))] for key, label, _ in BAL]
    tb = t.table(cellText=cells, colLabels=["run", "val loss", "MaxVio", "nearly dead"], loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.6)
    for c in tb.get_celld().values():
        c.set_facecolor(SURFACE); c.set_edgecolor(GRID); c.get_text().set_color(INK2)
    savefig(fig, "lab_balancing")
    R["router_lab"] = {key: dict(val=router_log(key)["val"][-1], maxvio=tail_maxvio(router_log(key)),
                                 dead=nearly_dead(router_log(key), 4).tolist()) for key in ("none", "aux", "bias", "softmax", "sqrt_softplus")}
    save_results()
'''),
md(r'''
How to read it: each heatmap row is one expert, and colour is its share of tokens relative to fair
(1.0 = exactly fair, dark = starved, bright = overloaded). Without balancing the router's early
favourites keep winning, because the experts it picks get more gradient and get better, so it picks
them more. The auxiliary loss pushes back, but through the same router gradient the language
loss uses. The bias pushes back without touching the gradient at all.

In the submitted run, **no balancing reached the lowest loss** (1.742 against 1.751 for the bias)
while starving 12 experts. A loss that drops fastest right now is not the goal; the eventual loss is. The starved experts are capacity that was paid for and will never be used, and
the imbalance would turn into stragglers once experts are spread across GPUs (Part M). The auxiliary
loss held MaxVio near 0.1, and the bias held it near 0.02.

### Score function: softmax, sigmoid or √softplus

Same conversion, same loss-free bias, only the function that turns router logits into scores
changes.
'''),
code(r'''
SCORES = [("softmax", "softmax (Qwen3)", SERIES[4]), ("sigmoid", "sigmoid [main] (DeepSeek-V3)", SERIES[1]),
          ("sqrt_softplus", "sqrt(softplus) (DeepSeek-V4)", SERIES[6])]
if have("runs", "router:softmax") and have("runs", "router:sqrt_softplus"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for key, label, color in SCORES:
        L = router_log(key)
        x = np.asarray(L["eval_step"]) * L["batch"] * GCFG.ctx / 1e6
        a1.plot(np.concatenate([[0], x]), np.concatenate([[L["val0"]], L["val"]]), color=color, label=label, marker="o", ms=2.5)
        a2.plot(x, np.asarray(L["maxvio"]).mean(1), color=color, label=label)
    a1.set(xlabel="tokens after T1 (M)", ylabel="validation loss", title="Loss")
    a2.set(xlabel="tokens after T1 (M)", ylabel="MaxVio (mean over layers)", title="Imbalance")
    a1.legend(fontsize=8)
    savefig(fig, "lab_score_function")
    display(Markdown("| score | val loss | MaxVio (last 20%) |\n|---|---|---|\n" + "\n".join(
        f"| {label} | {router_log(key)['val'][-1]:.4f} | {tail_maxvio(router_log(key)):.3f} |" for key, label, _ in SCORES)))
'''),
md(r'''
**Carry this forward:** a router left alone feeds its favourites. The bias fixes the load without
adding a second objective to the gradient, which is why it replaced the auxiliary loss. The score
function changes how strongly the router's preference reaches the weights. At this scale it made
no measurable difference: all three land within 0.001 nats, and softmax balanced slightly *better*
than sigmoid (P8 missed).
'''),
]
