from . import md, code

CELLS = [
md(r'''
---
## I · Stage 2: the main run (dense → MoE-8) and the dense control

From the T1 checkpoint, two branches run on **identical batches** with an **identical learning-rate
curve**:

* **MoE path.** Partition-convert (shared 768 + 8 × 192, top-4, sigmoid, loss-free bias), carry the
  Adam state, and train 8M tokens to T2. Stage 3 then continues it to T3.
* **Dense control.** The same dense model and optimizer state, trained on the same 20M tokens to T3.

The routing of a fixed probe batch is snapshotted at 0, 1, 2, 5, 10, 25, 50 and 100% of each MoE
stage, so Part L can measure how quickly routing settles.
'''),
code(r'''
def snap_steps(n):
    return sorted({int(round(f * n)) for f in (0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)})

T1_TOK, T2_TOK = S1 * TOK_STEP, (S1 + S2) * TOK_STEP
moe8 = moe8_opt = None
CUR_T2 = CUR_T1 + S2 * BUDGET["batch"]
if COMPUTE:
    R.setdefault("points", {})
    R["points"]["dense_T1"] = R["val_T1"]
    if has_ckpt("moe8_T2") and have("runs", "moe8"):
        moe8, moe8_opt, CUR_T2 = load_ckpt("moe8_T2")
        print("loaded Stage 2 checkpoint")
    else:
        seed_all(2)
        moe8, moe8_opt, _ = convert(dense, dense_opt, "partition", seed=0)
        R["points"]["moe8_T1"] = evaluate(moe8, BUDGET["val_batches"])
        print(f"MoE-8: {n_params(moe8) / 1e6:.2f}M total, {n_active(moe8) / 1e6:.2f}M active; "
              f"val right after conversion {R['points']['moe8_T1']:.4f} (dense {R['val_T1']:.4f})")
        CUR_T2, _ = train_run("moe8", moe8, moe8_opt, CUR_T1, S2, BUDGET["batch"], lr_post, s0=0, gamma_fn=gamma_post,
                              eval_every=BUDGET["eval_every"], val_batches=BUDGET["val_batches"],
                              snap_at=snap_steps(S2), tokens0=T1_TOK)
        save_ckpt("moe8_T2", moe8, moe8_opt, CUR_T2)
    R["points"]["moe8_T2"] = evaluate(moe8, BUDGET["val_batches"])
    R.setdefault("samples", {})["moe8_T2"] = [generate(moe8, TOK, p) for p in PROMPTS]
    save_results()
'''),
code(r'''
control = control_opt = None
if COMPUTE:
    if has_ckpt("control_T3") and have("runs", "control_b"):
        control, control_opt, _ = load_ckpt("control_T3")
        print("loaded dense-control checkpoint")
    else:
        control, control_opt, cur = load_ckpt("dense_T1")          # a fresh copy of the T1 model + Adam state
        assert cur == CUR_T1
        cur, _ = train_run("control_a", control, control_opt, cur, S2, BUDGET["batch"], lr_post, s0=0,
                           eval_every=BUDGET["eval_every"], val_batches=BUDGET["val_batches"], tokens0=T1_TOK)
        R["points"]["control_T2"] = evaluate(control, BUDGET["val_batches"])
        cur, _ = train_run("control_b", control, control_opt, cur, S3, BUDGET["batch"], lr_post, s0=S2,
                           eval_every=BUDGET["eval_every"], val_batches=BUDGET["val_batches"], tokens0=T2_TOK)
        save_ckpt("control_T3", control, control_opt, cur)
    R["points"]["control_T3"] = evaluate(control, BUDGET["val_batches"])
    R["samples"]["control_T3"] = [generate(control, TOK, p) for p in PROMPTS]
    save_results()
'''),
code(r'''
if have("runs", "moe8") and have("runs", "control_a"):
    M, Cn, D = R["runs"]["moe8"], R["runs"]["control_a"], R["runs"]["dense"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    xd = tokens_axis(D, D["eval_step"])
    keep = xd > T1_TOK / 1e6 * 0.6
    ax.plot(xd[keep], np.asarray(D["val"])[keep], color=DENSE_C, marker="o", ms=3, label="dense, Stage 1")
    ax.plot(np.concatenate([[T1_TOK / 1e6], tokens_axis(Cn, Cn["eval_step"])]), np.concatenate([[R["points"]["dense_T1"]], Cn["val"]]),
            color=DENSE_C, ls="--", marker="o", ms=3, label="dense control")
    ax.plot(np.concatenate([[T1_TOK / 1e6], tokens_axis(M, M["eval_step"])]), np.concatenate([[R["points"]["moe8_T1"]], M["val"]]),
            color=MOE_C, marker="o", ms=3, label="MoE-8 (converted at T1)")
    ax.axvline(T1_TOK / 1e6, color=MUTED, lw=1)
    ax.text(T1_TOK / 1e6, ax.get_ylim()[1], "  T1: convert", color=INK2, va="top", fontsize=8.5)
    ax.set(xlabel="tokens seen (M)", ylabel="validation loss", title="Stage 2: after conversion the MoE keeps training")
    ax.legend()
    savefig(fig, "stage2")
    p = R["points"]
    print(f"dense at T1 {p['dense_T1']:.4f} -> MoE-8 right after conversion {p['moe8_T1']:.4f} -> MoE-8 at T2 {p['moe8_T2']:.4f}"
          f"   |   dense control at T2 {p.get('control_T2', float('nan')):.4f}")
'''),
md(r'''
**Carry this forward:** conversion is a step inside one training run, not a restart. The batches,
the learning-rate curve and the optimizer state all continue across it.
'''),
]
