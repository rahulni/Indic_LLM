from . import md, code, mathbox

CELLS = [
md(r'''
---
## J · Growing 8 → 32 experts, and the clone-family trap

> Lightning LM grew 20 → 460 experts by cloning, and its first attempt collapsed
> "along clone families". This part reproduces that failure on purpose and then avoids it.

Growing means every expert becomes a **family** of 4 clones and the router rows are tiled, so
the 4 clones of a family get (almost) the same score. With hard top-4, a token's four highest scores
are then usually **all four clones of its single best family**. Before growth the token used four
*different* experts; after growth it uses one expert four times. The output changes, and the router
has no way to tell the clones apart.

| variant | clones start as | selection right after growth |
|---|---|---|
| copy + hard top-k | exact copies, router noise 1% | piles into one family |
| drop + hard top-k | half of each clone's neurons redrawn | piles into one family |
| **drop + Gumbel top-k [main]** | half redrawn; sampled selection for the first 15% of the window | spread by sampling |
| staggered-bias copy *(an untested idea of ours)* | exact copies, clone *c* starts with selection offset −c·Δ | **exactly as before growth** (gated in F) |

The staggered variant uses the balancing machinery itself. At step 0 only clone 0 of each family
can win, so routing is unchanged. The offset then shrinks linearly to zero over the window, and the
loss-free bias balancer hands tokens over to the idle clones gradually.

🧠 **Intuition.** Clone a teacher into four identical twins and seat them side by side. Every
student who wanted that teacher now sees four equally good options, and with "take the top four"
picks all four twins instead of four different teachers. The fixes:

* **Gumbel:** draw lots among near-equals, so the twins *and* the other teachers get some
  students.
* **Staggering:** introduce the twins one at a time.
'''),
mathbox("why clones tie, the Gumbel-max trick, and the staggered-offset guarantee", r'''
**Tied clones.** Tiling the router gives clone $c$ of family $f$ the logit $z_{f,c} = z_f + \varepsilon_{f,c}$,
with tiny noise $\varepsilon$.

Let $f^\star$ be the token's best family and $f'$ the runner-up. If $z_{f^\star} - z_{f'} > 2\max\lvert\varepsilon\rvert$,
then all $m$ clones of $f^\star$ outrank every clone of every other family. With $m = k = 4$ the
chosen set is exactly family $f^\star$: one expert, used four times.

That holds for almost every token, because $\varepsilon$ is tiny. Measured: 97.8% of tokens at the
start of Stage 3 (P5).

**The Gumbel-max trick.** With $G_i$ i.i.d. standard Gumbel noise, $G_i = -\log(-\log U_i)$ and
$U_i \sim \mathrm{Uniform}(0,1)$:

$$P\Big(\arg\max_i\,(z_i + G_i) = j\Big) = \frac{e^{z_j}}{\sum_i e^{z_i}},$$

and the top-$k$ of $z + G$ is a sample of $k$ experts without replacement in proportion to $e^{z}$
(the Plackett–Luce model).

* Tied clones get *equal* chances.
* Other families get their softmax share.
* So during the window every clone receives some gradient and the twins can start to differ.

We apply it to per-token standardised scores, $(\sigma + b - \text{mean})/\text{std}$, so it is
scale-free.

**The staggered guarantee.** Clone $c$ starts with selection offset $-c\Delta$. With scores in
$(0,1)$, the selection value of clone $c$ of family $f$ is $v_{f,c} = \sigma_f + b_f - c\Delta$.

For any $c \ge 1$ and any families $f, f'$:

$$v_{f,c} \;\le\; \sigma_f + b_f - \Delta \;<\; 1 + b_{\max} - \Delta \;\le\; b_{\min} \;<\; \sigma_{f'} + b_{f'} = v_{f',0}, \qquad \text{whenever } \Delta \ge 1 + (b_{\max} - b_{\min}).$$

Every clone 0 outranks every other clone, so the top-$k$ is exactly the clone-0s of the old top-$k$
families. With identical weights, that is an identical output. The gate in Part F checks it.

The catch, measured below: clones 1–3 receive no tokens until the offset is gone. When it
vanishes they are still identical to clone 0, and the tie returns.
'''),
md(r'''
**See the math: four tied clones.** Two families of four identical clones each. Family A's logit
is 1.0 and family B's is 0.8, plus noise of 0.001, with k = 4. Hard top-k always takes all of family
A. Gumbel top-k spreads picks over every clone, in proportion to $e^{z}$.
'''),
code(r'''
g_demo = torch.Generator().manual_seed(1)
zc = torch.tensor([1.0] * 4 + [0.8] * 4) + 0.001 * torch.randn(8, generator=g_demo)
hard = torch.zeros(8)
hard[zc.topk(4).indices] = 1.0
U = torch.rand(20_000, 8, generator=g_demo).clamp(1e-9, 1 - 1e-9)
noisy = zc - torch.log(-torch.log(U))
gum = torch.zeros(8).index_add_(0, noisy.topk(4, dim=1).indices.flatten(), torch.ones(80_000)) / 20_000
first = torch.bincount(noisy.argmax(1), minlength=8) / 20_000
assert torch.allclose(first, torch.softmax(zc, 0), atol=0.01)        # Gumbel-max: argmax ~ softmax
fig, ax = plt.subplots(figsize=(7.5, 2.8))
x = np.arange(8)
ax.bar(x - 0.18, hard, width=0.34, color=SERIES[2], label="hard top-4")
ax.bar(x + 0.18, gum, width=0.34, color=MOE_C, label="Gumbel top-4")
ax.set(xticks=x, xticklabels=[f"A{c}" for c in range(4)] + [f"B{c}" for c in range(4)], ylim=(0, 1.08),
       ylabel="share of tokens that pick it", title="Four tied clones per family: who gets picked")
ax.legend(fontsize=8.5)
savefig(fig, "demo_gumbel_ties")
print("hard top-4 picks family A only:", hard.tolist(), "| Gumbel:", [round(v, 2) for v in gum.tolist()])
'''),
code(r'''
GROW_RUNS = [("copy", "copy + hard top-k", SERIES[2]), ("drop", "drop + hard top-k", SERIES[3]),
             ("gumbel", "drop + Gumbel top-k [main]", SERIES[1]), ("staggered", "staggered-bias copy", SERIES[6])]
GWIN = max(1, int(0.15 * SLAB))
if COMPUTE:
    R.setdefault("growth", {})
    for key, _, _ in GROW_RUNS:
        name = f"grow:{key}"
        if have("runs", name):
            continue
        seed_all(20)
        mdl, opt, info = grow(moe8, moe8_opt, "drop" if key == "gumbel" else key, m=4, seed=0)
        R["growth"][key] = dict(family_at_0=family_share(route_probe(mdl), 4))
        lab_run(name, mdl, opt, CUR_T2, gumbel_steps=GWIN if key == "gumbel" else 0,
                offset_fn=staggered_schedule(mdl, GWIN) if key == "staggered" else None)
        del mdl, opt
        torch.cuda.empty_cache()
    save_results()
'''),
code(r'''
if have("runs", "grow:staggered"):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 3.5))
    for key, label, color in GROW_RUNS:
        L = R["runs"][f"grow:{key}"]
        x = np.asarray(L["eval_step"]) * L["batch"] * GCFG.ctx / 1e6
        a1.plot(np.concatenate([[0], x]), np.concatenate([[L["val0"]], L["val"]]), color=color, marker="o", ms=2.5, label=label)
        a2.plot(np.concatenate([[0], x]), np.concatenate([[R["growth"][key]["family_at_0"]], L["family"]]), color=color, label=label)
        dead = [int((np.asarray(l) < 0.1 * 4 / 32).sum()) for l in L["load"]]
        a3.plot(x, dead, color=color, label=label)
    a1.axhline(R["points"]["moe8_T2"], color=MUTED, lw=1)
    a1.text(0, R["points"]["moe8_T2"], " MoE-8 before growth", color=INK2, va="bottom", fontsize=8)
    a1.set(xlabel="tokens after growth (M)", ylabel="validation loss", title="Loss")
    a2.set(xlabel="tokens after growth (M)", ylabel="share of tokens", title="All 4 picks from one clone family", ylim=(-0.03, 1.03))
    a3.set(xlabel="tokens after growth (M)", ylabel="experts (all 8 layers, of 256)", title="Starved experts (< 10% of fair share)")
    fig.subplots_adjust(bottom=0.24)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    savefig(fig, "lab_growth")
    rows = ["| variant | val before growth | val at step 0 | val after window | one-family share at step 0 | nearly dead at end |", "|---|---|---|---|---|---|"]
    for key, label, _ in GROW_RUNS:
        L = R["runs"][f"grow:{key}"]
        rows.append(f"| {label} | {R['points']['moe8_T2']:.4f} | {L['val0']:.4f} | {L['val'][-1]:.4f} | "
                    f"{R['growth'][key]['family_at_0']:.0%} | {int(nearly_dead(L, 4).sum())} |")
    display(Markdown("\n".join(rows)))
'''),
md(r'''
**What the submitted run showed.** All four variants ended within 0.01 nats of each other:

| variant | val loss | note |
|---|---|---|
| copy + hard | 1.729 | |
| staggered | 1.729 | |
| drop + hard | 1.737 | |
| drop + Gumbel | 1.738 | the registered main recipe, last |

* **The pile-up is real.** At step 0, 98% of tokens put all four picks inside one family under every
  hard-routed variant (P5). The bias balancer then spread it, to about 40% by the end of the window.
* **Redrawing half the neurons did not pay at this size.** Plain copies jumped *more* at step 0
  (4 identical clones of one piece add up to 4× that piece), but recovered best. This matches
  Amazon's Expert Upcycling result for 32 → 64 experts, where plain copying beat drop-upcycling.
  Lightning LM's drop + sampling recipe was built for a 23× clone-out (20 → 460); here the growth is
  4×.
* **Staggering removed the step-0 jump but only postponed the pile-up.** While the offsets were
  shrinking, only clone 0 of each family got tokens. Clones 1–3 were therefore still identical when
  the offsets reached zero, the families tied, and the bump arrived about 0.2M tokens later.
  Letting the balancer remove each offset only once the clones have diverged is the obvious next
  version.

### Stage 3: the main path grows to 32 experts and trains to T3

The main path uses the recipe registered before the run: **drop + Gumbel top-k**, which is
Lightning LM's fix. Total parameters go from 26M to 68.5M; active parameters stay at about 19M
(the router grows from 8 to 32 rows).
'''),
code(r'''
moe32 = moe32_opt = None
if COMPUTE:
    if has_ckpt("moe32_T3") and have("runs", "moe32"):
        moe32, moe32_opt, _ = load_ckpt("moe32_T3")
        print("loaded Stage 3 checkpoint")
    else:
        seed_all(3)
        moe32, moe32_opt, _ = grow(moe8, moe8_opt, "drop", m=4, seed=0)
        R["points"]["moe32_T2"] = evaluate(moe32, BUDGET["val_batches"])
        R["growth"]["main_family_at_0"] = family_share(route_probe(moe32), 4)
        R["samples"]["moe32_at_growth"] = [generate(moe32, TOK, p) for p in PROMPTS]
        print(f"MoE-32: {n_params(moe32) / 1e6:.2f}M total, {n_active(moe32) / 1e6:.2f}M active; val right after growth "
              f"{R['points']['moe32_T2']:.4f}; tokens with all 4 picks in one family: {R['growth']['main_family_at_0']:.0%}")
        _, _ = train_run("moe32", moe32, moe32_opt, CUR_T2, S3, BUDGET["batch"], lr_post, s0=S2, gamma_fn=gamma_post,
                         eval_every=BUDGET["eval_every"], val_batches=BUDGET["val_batches"],
                         gumbel_steps=int(0.15 * S3), snap_at=snap_steps(S3), tokens0=T2_TOK)
        save_ckpt("moe32_T3", moe32, moe32_opt, CUR_T2 + S3 * BUDGET["batch"])
    R["points"]["moe32_T3"] = evaluate(moe32, BUDGET["val_batches"])
    R["samples"]["moe32_T3"] = [generate(moe32, TOK, p) for p in PROMPTS]
    R["params"] = {"dense": n_params(dense), "moe8": n_params(moe8), "moe32": n_params(moe32),
                   "dense_active": n_active(dense), "moe8_active": n_active(moe8), "moe32_active": n_active(moe32)}
    save_results()
'''),
md(r'''
**Carry this forward:** when you grow by cloning, the router cannot tell the clones apart, so
selection collapses onto one family per token. Either break the tie on purpose (sample the top-k
for a while) or remove it on purpose (stagger the clones and let the balancer hand tokens over).
'''),
]
