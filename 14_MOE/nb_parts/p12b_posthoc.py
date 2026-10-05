from . import md, code

CELLS = [
md(r'''
---
## K.2 · Post-hoc: was it the growth recipe, or growing at all?

> **Added after the first full run, and labelled as such.** Everything above was fixed before that
> run. Prediction P3 (MoE-32 beats the dense control) missed, and most of the gap traced back to
> the growth step at T2. That growth used the recipe registered in advance (drop + Gumbel), which
> came *last* in the growth lab. This section asks two follow-up questions. Their predictions,
> P13–P15, were written down before these runs. P1–P12 were not touched.

Two more branches start from the same T2 checkpoint, on the same batches with the same
learning-rate curve as the registered Stage 3:

| branch | from T2 | question it answers |
|---|---|---|
| **MoE-32, plain-copy growth** | 8 → 32 by exact copies of each expert (router tiled + 1% noise), hard top-4 | was the *recipe* the problem? |
| **MoE-8, no growth** | keep training the 8-expert model | was *growing at all* worth it in 12M tokens? |
'''),
code(r'''
POSTHOC_RUNS = [("moe32", "MoE-32, drop + Gumbel growth (registered)", MOE_C, "-"),
                ("moe32_copy", "MoE-32, plain-copy growth (post-hoc)", SERIES[2], "-"),
                ("moe8_cont", "MoE-8, no growth (post-hoc)", SERIES[3], "-"),
                ("control_b", "dense control", DENSE_C, "--")]
moe32c = moe8c = None
if COMPUTE:
    for v in ("dense_opt", "control_opt", "moe32_opt"):        # free optimizer state no longer needed
        globals()[v] = None
    torch.cuda.empty_cache()
    if has_ckpt("moe32copy_T3") and have("runs", "moe32_copy"):
        moe32c, _, _ = load_ckpt("moe32copy_T3")
    else:
        if moe8_opt is None:
            moe8, moe8_opt, _ = load_ckpt("moe8_T2")
        seed_all(4)
        moe32c, opt_c, _ = grow(moe8, moe8_opt, "copy", m=4, seed=0)
        R["points"]["moe32copy_T2"] = evaluate(moe32c, BUDGET["val_batches"])
        R["growth"]["copy_main_family_at_0"] = family_share(route_probe(moe32c), 4)
        train_run("moe32_copy", moe32c, opt_c, CUR_T2, S3, BUDGET["batch"], lr_post, s0=S2, gamma_fn=gamma_post,
                  eval_every=BUDGET["eval_every"], val_batches=BUDGET["val_batches"], tokens0=T2_TOK)
        save_ckpt("moe32copy_T3", moe32c, opt_c, CUR_T2 + S3 * BUDGET["batch"])
        del opt_c
    if has_ckpt("moe8_T3") and have("runs", "moe8_cont"):
        moe8c, _, _ = load_ckpt("moe8_T3")
    else:
        moe8c, opt8c, cur = load_ckpt("moe8_T2")              # a fresh copy of the T2 model + Adam state
        assert cur == CUR_T2
        train_run("moe8_cont", moe8c, opt8c, cur, S3, BUDGET["batch"], lr_post, s0=S2, gamma_fn=gamma_post,
                  eval_every=BUDGET["eval_every"], val_batches=BUDGET["val_batches"], tokens0=T2_TOK)
        save_ckpt("moe8_T3", moe8c, opt8c, cur + S3 * BUDGET["batch"])
        del opt8c
    torch.cuda.empty_cache()
    R["points"]["moe32copy_T3"] = evaluate(moe32c, BUDGET["val_batches"])
    R["points"]["moe8cont_T3"] = evaluate(moe8c, BUDGET["val_batches"])
    if not have("final_pair", "moe32_copy"):
        R["final_pair"]["moe32_copy"] = evaluate(moe32c, BUDGET["final_val_batches"], per_batch=True)
        R["final_pair"]["moe8_cont"] = evaluate(moe8c, BUDGET["final_val_batches"], per_batch=True)
    R["samples"]["moe32copy_T3"] = [generate(moe32c, TOK, p) for p in PROMPTS]
    fp = R["final_pair"]
    R["posthoc"] = {name: dict(val_T3=float(np.mean(fp[name])),
                               vs_control=paired_stats(fp[name], fp["control"]),
                               vs_moe32=paired_stats(fp[name], fp["moe32"]),
                               params=n_params(mdl), active=n_active(mdl))
                    for name, mdl in (("moe32_copy", moe32c), ("moe8_cont", moe8c))}
    save_results()
'''),
code(r'''
if have("posthoc") and have("runs", "moe32_copy"):
    P = R["points"]
    starts = {"moe32": P["moe32_T2"], "moe32_copy": P["moe32copy_T2"], "moe8_cont": P["moe8_T2"], "control_b": P["control_T2"]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for name, label, color, ls in POSTHOC_RUNS:
        L = R["runs"][name]
        x = np.concatenate([[T2_TOK / 1e6], tokens_axis(L, L["eval_step"])])
        y = np.concatenate([[starts[name]], L["val"]])
        for ax in (a1, a2):
            ax.plot(x, y, color=color, ls=ls, marker="o" if ls == "-" else None, ms=2.5, lw=2 if ls == "-" else 1.6, label=label)
    a1.set(xlabel="tokens seen (M)", ylabel="validation loss", title="Three ways to spend Stage 3 (T2 → T3)")
    a1.legend(fontsize=8)
    ends = [R["runs"][n]["val"][-1] for n, *_ in POSTHOC_RUNS]
    a2.set_xlim(T2_TOK / 1e6 + 0.7 * S3 * TOK_STEP / 1e6, (T2_TOK + S3 * TOK_STEP) / 1e6 * 1.005)
    a2.set_ylim(min(ends) - 0.01, max(ends) + 0.05)
    a2.set(xlabel="tokens seen (M)", title="Zoom: the last 30%")
    savefig(fig, "posthoc_stage3")
    ph = R["posthoc"]
    rows = ["| T3 model | total / active | val loss (64 batches) | minus dense control | minus registered MoE-32 |", "|---|---|---|---|---|"]
    rows.append(f"| dense control | {R['params']['dense'] / 1e6:.1f}M / {R['params']['dense_active'] / 1e6:.1f}M "
                f"| {np.mean(R['final_pair']['control']):.4f} | - | - |")
    rows.append(f"| MoE-32, drop + Gumbel (registered) | {R['params']['moe32'] / 1e6:.1f}M / {R['params']['moe32_active'] / 1e6:.1f}M "
                f"| {np.mean(R['final_pair']['moe32']):.4f} | {R['paired']['mean']:+.4f} [{R['paired']['lo']:+.4f}, {R['paired']['hi']:+.4f}] | - |")
    for name, label in (("moe32_copy", "MoE-32, plain-copy growth"), ("moe8_cont", "MoE-8, no growth")):
        r, c, m = ph[name], ph[name]["vs_control"], ph[name]["vs_moe32"]
        rows.append(f"| {label} (post-hoc) | {r['params'] / 1e6:.1f}M / {r['active'] / 1e6:.1f}M | {r['val_T3']:.4f} "
                    f"| {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] |")
    display(Markdown("\n".join(rows) + "\n\nNegative = the row's model is better. Brackets: 95% bootstrap interval over batches."))
'''),
md(r'''
**What the post-hoc runs showed** (all three post-hoc predictions held).

1. **The recipe mattered, a little.** Plain-copy growth finished 0.005 nats below the registered
   drop + Gumbel recipe (P13), even though it *jumped more* at T2: 1.966 against 1.892. Four
   identical copies of one partition piece add up to four times that piece. Jump size alone did not
   predict the outcome. Copied clones keep their full function and only need to diverge; redrawn
   clones must relearn half their neurons.
2. **Growing at all mattered more.** Not growing beat both growth recipes. MoE-8 simply continued
   to T3 finished 0.017 below the registered MoE-32 (P14), with 26M parameters against 68.5M. A
   growth step costs 0.17–0.24 nats on the spot, and 12M tokens at this scale did not pay that back.
   Four times the experts at the same active compute needs far more tokens to earn its keep.
3. **No MoE beat the dense control (P15), but the gap was closing.** MoE-8 trailed the dense model
   by 0.009 at T2 and by 0.003 at T3 (CI 0.002 to 0.004). That is consistent with the T1 conversion
   cost being repaid slowly. Whether it overtakes with more tokens is the next experiment; it was not
   run here.

**For a growth plan at larger scale:** treat every growth step as a loss you must buy back. Grow only when
the next stage has many times the tokens that the jump takes to recover. Prefer growth that keeps
each expert's function (copies, or staggered copies), because redrawing neurons slows the recovery.
And grow early, while the learning rate is high, not in the last stretch before decay.
''', "posthoc_narrative"),
]
