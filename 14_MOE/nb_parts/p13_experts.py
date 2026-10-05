from . import md, code, mathbox

CELLS = [
md(r'''
---
## L · What did the experts learn?

> Experts do not become subject specialists the way the name suggests; they mostly sort **kinds of tokens** (punctuation, names, function
> words), not subjects. TinyStories has only one subject, so token kind is the only thing left to
> sort by, which makes it a clean place to look.

Every measurement below runs the final MoE-32 over 131K held-out tokens and records each token's
top-4 experts in every layer. Nothing is trained.
'''),
code(r'''
FUNC_WORDS = set("the a an and to of was he she it they his her in on with for but so said is that you i we at as be had were there "
                 "one day him them their this not are can do what up then very".split())
CATS = ["end of text", "punctuation", "number", "capitalised word", "function word", "content word", "word piece"]

def token_category(tid, piece):
    if tid == 0:
        return "end of text"
    core = piece.strip()
    if not core or all(not c.isalnum() for c in core):
        return "punctuation"
    if any(c.isdigit() for c in core):
        return "number"
    if piece.startswith(" ") or piece[:1].isupper():
        if core[0].isupper():
            return "capitalised word"
        return "function word" if core.lower() in FUNC_WORDS else "content word"
    return "word piece"

@torch.no_grad()
def collect_routing(model, n_batches, B=32):
    moes = model.moe_layers()
    model.eval()
    for m in moes:
        m.record = True
    ids, tops = [], [[] for _ in moes]
    for i in range(n_batches):
        x, _ = VAL.batch(i * B, B)
        with amp():
            model(x)
        ids.append(x.reshape(-1).cpu())
        for j, m in enumerate(moes):
            tops[j].append(m.last["topi"].cpu())
    for m in moes:
        m.record = False
    return torch.cat(ids), [torch.cat(t) for t in tops]

def set_overlap(a, b):
    return float((a[:, :, None] == b[:, None, :]).any(-1).float().mean())

if COMPUTE and moe32 is not None and not have("experts"):
    ids, tops = collect_routing(moe32, BUDGET["val_batches"])
    E, k = moe32.moe_layers()[0].E, moe32.moe_layers()[0].k
    vocab_cat = [CATS.index(token_category(i, TOK.decode([i]))) for i in range(GCFG.vocab)]
    cat = torch.tensor(vocab_cat)[ids]
    ex = {"n_tokens": int(len(ids)), "cat_share": torch.bincount(cat, minlength=len(CATS)).div(len(ids)).tolist()}
    # first-choice counts per (token category, expert); lift is computed when plotting
    ex["counts"] = {}
    for layer in (0, 3, 7):
        first = tops[layer][:, 0]
        ex["counts"][str(layer)] = torch.bincount(cat * E + first, minlength=len(CATS) * E).view(len(CATS), E).tolist()
    # top tokens per expert (layer 4): highest lift among tokens seen >= 30 times
    first = tops[3][:, 0]
    cnt_tok = torch.bincount(ids, minlength=GCFG.vocab).float()
    pe = torch.bincount(first, minlength=E).float() / len(first)
    tt = []
    for e in range(E):
        c_e = torch.bincount(ids[first == e], minlength=GCFG.vocab).float()
        lift = c_e / cnt_tok.clamp_min(1) / pe[e].clamp_min(1e-9)
        lift[cnt_tok < 30] = 0
        best = lift.topk(6).indices.tolist()
        tt.append(dict(share=float(pe[e]), tokens=[TOK.decode([i]) for i in best], lift=[round(float(lift[i]), 1) for i in best]))
    ex["top_tokens_layer4"] = tt
    # consecutive tokens sharing their first-choice expert, vs chance
    cons = []
    for layer, t in enumerate(tops):
        f = t[:, 0].view(-1, GCFG.ctx)
        agree = float((f[:, 1:] == f[:, :-1]).float().mean())
        p = torch.bincount(t[:, 0], minlength=E).float() / t.shape[0]
        cons.append(dict(layer=layer + 1, agree=agree, chance=float((p ** 2).sum())))
    ex["consecutive"] = cons
    # load per expert over all k picks (for pruning)
    usage = [torch.bincount(t.reshape(-1), minlength=E).float() for t in tops]
    base = evaluate(moe32, BUDGET["val_batches"])
    def masked_val(drop):                       # drop: {layer: [experts]}
        for li, m in enumerate(moe32.moe_layers()):
            m.mask[:] = True
            m.mask[drop.get(li, [])] = False
        v = evaluate(moe32, BUDGET["val_batches"])
        for m in moe32.moe_layers():
            m.mask[:] = True
        return v
    half = {li: u.argsort()[: E // 2].tolist() for li, u in enumerate(usage)}
    flat = torch.stack(usage).flatten()
    top3 = flat.topk(3).indices.tolist()
    super3 = {}
    for g in top3:
        super3.setdefault(g // E, []).append(g % E)
    rng = np.random.default_rng(0)
    rand = []
    for _ in range(5):
        pick = rng.choice(len(flat), 3, replace=False)
        dct = {}
        for g in pick:
            dct.setdefault(int(g) // E, []).append(int(g) % E)
        rand.append(masked_val(dct))
    ex["prune"] = dict(base=base, half=masked_val(half), top3=masked_val(super3), top3_ids=[(g // E + 1, g % E) for g in top3],
                       top3_share=[float(flat[g] / usage[0].sum()) for g in top3], random3=rand)
    # how fast routing settles: overlap of each snapshot's top-4 with the end-of-stage top-4
    settle = {}
    for run in ("moe8", "moe32"):
        snaps = SNAP.get(run, {})
        if len(snaps) >= 2:
            last = snaps[max(snaps)]
            n = R["runs"][run]["steps"]
            settle[run] = dict(frac=[s / n for s in sorted(snaps)],
                               overlap=[[set_overlap(a, b) for a, b in zip(snaps[s], last)] for s in sorted(snaps)])
    if settle:
        ex["settling"] = settle
    R["experts"] = ex
    save_results()
'''),
md(r'''
### Which kinds of tokens go where

Colour is **lift**: how much more (warm) or less (cool) often an expert is a category's first
choice than it is overall. 1× is grey; a row of grey means that category is spread evenly. Blank
cells had fewer than 5 expected tokens, too few to say.

🧠 **Intuition.** Lift asks: "if I know this token is punctuation, how much more likely is expert
7 to be its first choice than for an average token?" A lift of 4× means expert 7 is a punctuation
specialist; 1× means it does not care.
'''),
mathbox("lift, chance agreement, and routing overlap", r'''
**Lift.** Let $n_{c,e}$ count tokens of category $c$ whose first choice is expert $e$, with $n_c$, $n_e$
the row and column totals and $N$ the total:

$$\operatorname{lift}(c, e) = \frac{P(e \mid c)}{P(e)} = \frac{n_{c,e}/n_c}{n_e/N} = \frac{n_{c,e}}{\mathbb{E}[n_{c,e}]}, \qquad \mathbb{E}[n_{c,e}] = \frac{n_c\,n_e}{N} \text{ under independence.}$$

We blank cells with $\mathbb{E}[n_{c,e}] < 5$.

**Chance agreement.** Draw two tokens independently. Their first choices agree with probability

$$\sum_{i=1}^{E} p_i^2, \qquad p_i = \text{expert } i\text{'s share of first choices}.$$

That equals $1/E$ for uniform routing (1/32 ≈ 3.1%) and is larger when routing is uneven.
Consecutive tokens agreeing far more often than $\sum p_i^2$ means routing follows local
structure: words, phrases, syntax.

**Routing overlap**, comparing the chosen set at step $s$ with the end of the stage, on a fixed
probe batch:

$$\operatorname{overlap}(s) = \frac{1}{N}\sum_{t=1}^{N} \frac{\lvert \mathcal{T}_t^{(s)} \cap \mathcal{T}_t^{(\text{end})} \rvert}{k}.$$

For random routing this is about $k/E$ (0.125 in Stage 3).
'''),
code(r'''
if have("experts"):
    ex = R["experts"]
    rows = [i for i, s in enumerate(ex["cat_share"]) if s >= 0.005]          # skip categories that barely occur
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.5 * len(rows) + 1.2), sharey=True)
    for ax, layer in zip(axes, ("0", "3", "7")):
        Cn = np.asarray(ex["counts"][layer], float)[rows]
        expected = Cn.sum(1, keepdims=True) * Cn.sum(0, keepdims=True) / Cn.sum()
        lift = np.where(expected >= 5, Cn / np.maximum(expected, 1e-9), np.nan)       # too few tokens to say: blank
        M = np.log2(np.clip(lift, 1 / 8, 8))
        im = ax.imshow(M, aspect="auto", cmap=DIV, vmin=-3, vmax=3, interpolation="nearest")
        ax.set(title=f"layer {int(layer) + 1}", xlabel="expert")
        ax.set_yticks(range(len(rows)), [f"{CATS[i]} ({ex['cat_share'][i]:.0%})" for i in rows])
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.ax.set_yticklabels(["1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x"])
    cb.set_label("lift (first choice)", color=INK2)
    savefig(fig, "experts_by_token_kind")
'''),
md(r'''
**The most characteristic tokens of each expert** (layer 4, first choice). For each expert, these are
the tokens with the highest lift among tokens seen at least 30 times.
'''),
code(r'''
if have("experts"):
    tt = R["experts"]["top_tokens_layer4"]
    order = sorted(range(len(tt)), key=lambda e: -max(tt[e]["lift"] or [0]))
    lines = ["| expert | share of first choices | most characteristic tokens (lift) |", "|---|---|---|"]
    for e in order:
        toks = ", ".join(f"`{t.replace('|', '¦').replace(chr(10), '⏎')!r}` {l}x" for t, l in zip(tt[e]["tokens"], tt[e]["lift"]))
        lines.append(f"| {e} | {tt[e]['share']:.1%} | {toks} |")
    display(Markdown("\n".join(lines)))
'''),
md(r'''
### Neighbouring tokens share experts; and how fast routing settles

The left panel shows how often two consecutive tokens get the same first-choice expert, compared
with chance (Σpᵢ², the agreement two random tokens would have). Mixtral measured 24–28% against
12.5%. The right panel shows how much of the end-of-stage routing is already in place at each point
of the stage (OLMoE: *"60% settled after 1% of training"*).

Our routing settles **late**, and that is expected here. OLMoE's figure is for a model trained from
scratch with a fixed router. Our Stage 3 sampled its routing (Gumbel) for the first 15%, and its
bias balancer kept moving loads until the final decay. Every stage also starts from a freshly
converted router, so the end-of-stage routing is still being formed.
'''),
code(r'''
if have("experts"):
    ex = R["experts"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.4))
    cons = ex["consecutive"]
    x = np.arange(1, len(cons) + 1)
    a1.bar(x - 0.18, [c["agree"] for c in cons], width=0.34, color=MOE_C, label="consecutive tokens")
    a1.bar(x + 0.18, [c["chance"] for c in cons], width=0.34, color=MUTED, label="chance (sum p^2)")
    a1.set(xlabel="layer", ylabel="same first-choice expert", title="Consecutive-token agreement vs chance", xticks=x)
    a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a1.legend()
    if "settling" in ex:
        for run, color, lab in (("moe8", SERIES[2], "Stage 2 (8 experts)"), ("moe32", MOE_C, "Stage 3 (32 experts)")):
            if run in ex["settling"]:
                s = ex["settling"][run]
                ov = np.asarray(s["overlap"])
                a2.plot(np.asarray(s["frac"]) * 100, ov.mean(1), color=color, marker="o", ms=3.5, label=lab)
        a2.set(xlabel="% of the stage trained", ylabel="top-4 overlap with end of stage", title="How fast routing settles", ylim=(0, 1.02))
        a2.set_xscale("symlog", linthresh=1)
        a2.set_xticks([0, 1, 2, 5, 10, 25, 50, 100], ["0", "1", "2", "5", "10", "25", "50", "100"])
        a2.set_xlim(0, 110)
        a2.legend()
    else:
        a2.axis("off")
    savefig(fig, "experts_consecutive_settling")
'''),
md(r'''
### Remove experts and see what breaks

*REAP (2025):* half the experts can be removed with under 2% loss. *Super Experts (2025):* three
experts out of 6,144 hold the model up. Masked experts can no longer be chosen; each token picks
its 4 from the rest.
'''),
code(r'''
if have("experts"):
    p = R["experts"]["prune"]
    rel = lambda v: (v - p["base"]) / p["base"]
    rows = ["| experts removed | val loss | change |", "|---|---|---|",
            f"| none | {p['base']:.4f} | - |",
            f"| least-used half in every layer (128 of 256) | {p['half']:.4f} | {rel(p['half']):+.2%} |",
            f"| the 3 most-used experts (layer, id) = {p['top3_ids']} | {p['top3']:.4f} | {rel(p['top3']):+.2%} |",
            f"| 3 random experts, mean of 5 draws | {np.mean(p['random3']):.4f} | {rel(np.mean(p['random3'])):+.2%} |"]
    display(Markdown("\n".join(rows)))
'''),
md(r'''
**Carry this forward:** specialization comes out of training, and in a one-domain corpus it can only
be about the kind of token: end-of-text, punctuation and capitalised words each have experts that
prefer them, and neighbouring tokens share experts far more often than chance. Half of the routed
experts could be removed for under 1% loss here. That is the redundancy REAP measured, and the same
redundancy a careful growth strategy tries not to build in. The "super expert" effect did not
appear at this scale: removing the three busiest experts cost no more than removing three at random
(P11 missed).
'''),
]
