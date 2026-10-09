from . import md, code

CELLS = [
md(r'''
---
## G · Conversion lab: which starting point learns fastest?

> Sparse upcycling (copy), partition, drop-upcycling, and "is it worth starting from the dense
> model at all?"

Six short runs, all starting from the **same** T1 checkpoint (weights *and* Adam state), all reading
the **same** batches for 1.2M tokens at batch 16, with the same learning rate.

> **A lesson from the first full run.** The labs first ran at the main runs' peak learning rate
> (1.2e-3) with a batch 4× smaller. Every lab's loss *rose*, the dense one included (1.81 → 1.98),
> because gradient noise grows with lr / batch. The labs now use lr × 16/64 = 3e-4, which keeps the
> main runs' noise scale. The main stages were unaffected and were not re-run, and no prediction was
> changed.

| run | experts | active params | note |
|---|---|---|---|
| dense, continued | (none) | 18.9M | the bar every MoE must clear |
| **partition [main]** | shared 768 + 8 × 192, top-4 | 18.9M | our Stage 2 recipe |
| partition, no shared | 16 × 192, top-8 | 18.9M | the open design question "shared or none?" |
| copy | 8 × 1536, top-2 | 33.1M | exact at conversion, but twice the compute per token |
| drop r=0.5 | 8 × 1536 with half redrawn, top-2 | 33.1M | the drop-upcycling paper's best rate |
| random experts | shared 768 + 8 × 192, top-4 | 18.9M | keeps attention and embeddings, throws the FFN away |

Copy and drop do twice the feed-forward work per token, so they are *not* a fair race against the
others. They are here to show what exact preservation buys.
'''),
code(r'''
LAB_VAL = max(2, BUDGET["val_batches"] // 2)
LAB_COLORS = {"dense": SERIES[0], "partition": SERIES[1], "partition_noshared": SERIES[4], "copy": SERIES[2],
              "drop": SERIES[3], "random": SERIES[6]}
LAB_NAMES = {"dense": "dense, continued", "partition": "partition [main]", "partition_noshared": "partition, no shared",
             "copy": "copy (2x compute)", "drop": "drop r=0.5 (2x compute)", "random": "random experts"}

def lab_run(name, model, opt, cursor, gumbel_steps=0, offset_fn=None, tokens0=0.0):
    """One short branch at batch 16. val0 = validation loss before its first step."""
    v0 = round(evaluate(model, LAB_VAL), 4)
    train_run(name, model, opt, cursor, SLAB, BUDGET["lab_batch"], lr_lab, eval_every=BUDGET["lab_eval_every"],
              val_batches=LAB_VAL, gumbel_steps=gumbel_steps, offset_fn=offset_fn, tokens0=tokens0)
    R["runs"][name]["val0"] = v0
    save_results()

if COMPUTE:
    for key in ("dense", "partition", "partition_noshared", "copy", "drop", "random"):
        name = f"conv:{key}"
        if have("runs", name):
            continue
        seed_all(10)
        if key == "dense":
            mdl, opt, _ = load_ckpt("dense_T1")
        else:
            mdl, opt, _ = convert(dense, dense_opt, key, seed=0)
        lab_run(name, mdl, opt, CUR_T1)
        del mdl, opt
        torch.cuda.empty_cache()
'''),
code(r'''
def lab_curve(ax, name, label, color, ls="-"):
    L = R["runs"][name]
    x = np.concatenate([[0], np.asarray(L["eval_step"]) * L["batch"] * GCFG.ctx / 1e6])
    y = np.concatenate([[L["val0"]], L["val"]])
    ax.plot(x, y, color=color, ls=ls, label=label, marker="o", ms=2.5)
    return y

if have("runs", "conv:partition"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    finals = {}
    for key in ("dense", "partition", "partition_noshared", "copy", "drop", "random"):
        lab = LAB_NAMES[key] + (f" (starts at {R['runs']['conv:random']['val0']:.1f})" if key == "random" else "")
        y = lab_curve(a1, f"conv:{key}", lab, LAB_COLORS[key])
        finals[key] = y[-1]
        lab_curve(a2, f"conv:{key}", LAB_NAMES[key], LAB_COLORS[key])
    a1.set(xlabel="tokens after T1 (M)", ylabel="validation loss", title="All six starting points")
    a1.set_ylim(min(finals.values()) - 0.05, R["runs"]["conv:dense"]["val0"] + 0.6)
    a2.set(xlabel="tokens after T1 (M)", title="Zoom: the last half")
    lo = min(finals.values())
    a2.set_xlim(BUDGET["lab"] / 2e6, BUDGET["lab"] / 1e6 * 1.02)
    a2.set_ylim(lo - 0.02, lo + 0.12)
    a1.legend(fontsize=8, ncol=2)
    savefig(fig, "lab_conversion")
    rows = ["| start | val at conversion | val after lab | vs dense continued | total | active |", "|---|---|---|---|---|---|"]
    for key in ("dense", "partition", "partition_noshared", "copy", "drop", "random"):
        L = R["runs"][f"conv:{key}"]
        rows.append(f"| {LAB_NAMES[key]} | {L['val0']:.4f} | {L['val'][-1]:.4f} | {L['val'][-1] - finals['dense']:+.4f} "
                    f"| {L['params'] / 1e6:.1f}M | {L['active'] / 1e6:.1f}M |")
    display(Markdown("\n".join(rows)))
'''),
md(r'''
**What the submitted run showed.** After 1.2M tokens, nothing beat simply continuing the dense
model:

| run | val loss | note |
|---|---|---|
| dense, continued | 1.733 | |
| copy | 1.742 | at twice the compute |
| partition [main] | 1.751 | |
| partition, no shared | 1.775 | |
| drop | 1.809 | |
| random experts | 2.41 | |

**Carry this forward:** a conversion costs loss up front, and 1.2M tokens is not enough to repay it
(the main run gives the MoE 20M). The shared expert earned its place: 1.751 with it against 1.775
without, at the same active size and with *fewer* total parameters. The random-experts run shows
how much of a model's knowledge lives in its feed-forward blocks.
'''),
]
