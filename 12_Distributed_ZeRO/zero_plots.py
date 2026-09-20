"""zero_plots: the figures for the notebook. Dark theme, matching the rest of the project.

Plotting is not the lesson, so it lives here rather than in the notebook. Every function
takes plain numbers (already measured) and returns a matplotlib Figure.

Palette: the first five dark-mode categorical slots, validated on this surface (#0d1117)
for colour-vision deficiency and contrast. Colour always follows the entity: a stage or a
memory category keeps the same colour in every figure.
"""
from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullFormatter

SURFACE, INK, MUTED, EDGE, GRID = "#0d1117", "#e6edf3", "#8b949e", "#30363d", "#21262d"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
STAGE_NAMES = {0: "DDP (ZeRO-0)", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
STAGE_COLORS = {s: SERIES[s] for s in range(4)}
CATS = ["weights", "grads", "optimizer", "activations", "temp"]
CAT_LABELS = {"weights": "weights", "grads": "gradients",
              "optimizer": "optimizer states (fp32 master + Adam m, v)",
              "activations": "activations", "temp": "temporary buffers"}
CAT_COLORS = dict(zip(CATS, SERIES))
# one-hue sequential ramp (blue), dark -> light, so bigger values stand out on dark
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"])
MiB = 2 ** 20


def apply_theme():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": EDGE,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "figure.dpi": 130,
    })


def _mib(x):
    return x / MiB


def _ms_by_cat(snap: dict) -> dict:
    """Collapse a ledger snapshot into the five categories the figures use."""
    return {"weights": snap["weights"], "grads": snap["grads"],
            "optimizer": snap["master"] + snap["adam_m"] + snap["adam_v"],
            "activations": snap["activations"], "temp": snap["temp"]}


# ---------------------------------------------------------------------------------------------
def ring_heatmap(traces: dict, N: int):
    """traces[rank] = [(phase, step, contributions-per-chunk), ...]. One panel per step:
    rows = GPUs, columns = chunks, cell = how many GPUs' data that chunk now contains."""
    n_steps = len(traces[0])
    fig, axes = plt.subplots(1, n_steps, figsize=(1.55 * n_steps + 1, 2.6), sharey=True)
    for k, ax in enumerate(axes):
        grid = [traces[r][k][2] for r in range(N)]
        ax.imshow(grid, cmap=BLUE_RAMP, vmin=1, vmax=N)
        for r in range(N):
            for c in range(N):
                v = grid[r][c]
                ax.text(c, r, str(v), ha="center", va="center", fontsize=9,
                        color=SURFACE if v >= 0.6 * N else INK, fontweight="bold")
        phase, step = traces[0][k][0], traces[0][k][1]
        ax.set_title("start" if phase == "start" else f"{'RS' if phase[0] == 'r' else 'AG'} {step}",
                     fontsize=9)
        ax.set_xticks(range(N), [f"c{c}" for c in range(N)], fontsize=7)
        ax.set_yticks(range(N), [f"GPU{r}" for r in range(N)], fontsize=7)
        ax.grid(False)
    fig.suptitle(f"Ring all-reduce on {N} GPUs: how many GPUs' data each chunk holds "
                 f"(reduce-scatter, then all-gather)", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def what_each_gpu_holds(stages: dict, psi_label: str):
    """stages[s] = dict(snapshot=end-of-backward ledger, formula=bytes, peak=bytes).
    Horizontal stacked bars of model states (rank 0), formula tick and peak marker."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    order = [3, 2, 1, 0]
    for y, s in enumerate(order):
        parts = _ms_by_cat(stages[s]["snapshot"])
        left = 0.0
        for c in ["weights", "grads", "optimizer"]:
            w = _mib(parts[c])
            ax.barh(y, w, left=left, height=0.46, color=CAT_COLORS[c], edgecolor=SURFACE,
                    linewidth=2, label=CAT_LABELS[c] if y == 0 else None)
            left += w
        ax.plot([_mib(stages[s]["formula"])] * 2, [y - 0.32, y + 0.32], color=INK, lw=2)
        ax.plot(_mib(stages[s]["peak"]), y, marker="D", ms=7, color=MUTED,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label="peak incl. activations + temporaries" if y == 0 else None)
        ax.text(max(left, _mib(stages[s]["peak"])) + 0.25, y,
                f"{left:.2f} MiB  ({stages[s]['formula'] / stages[0]['formula'] * 100:.0f}% of DDP)",
                va="center", fontsize=9, color=INK)
    ax.plot([], [], color=INK, lw=2, label="ZeRO formula")
    ax.set_yticks(range(4), [STAGE_NAMES[s] for s in order])
    ax.set_xlabel("MiB held by one GPU at the end of backward (rank 0 of 32)")
    ax.set_xlim(0, _mib(max(stages[0]["peak"], stages[0]["formula"])) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What each GPU holds: model states under each ZeRO stage  ({psi_label})",
                 loc="left")
    ax.legend(loc="lower right", fontsize=8.5, ncol=1)
    fig.tight_layout()
    return fig


def cluster_heatmap(peaks: dict, world: int, cols: int = 8):
    """peaks[s] = list of per-GPU peak bytes. One panel per stage, GPUs on a grid."""
    rows = math.ceil(world / cols)
    vmax = max(max(v) for v in peaks.values()) / MiB
    vmin = min(min(v) for v in peaks.values()) / MiB
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
    norm = matplotlib.colors.LogNorm(vmin=vmin * 0.9, vmax=vmax * 1.05)
    for s, ax in enumerate(axes):
        grid = [[peaks[s][r * cols + c] / MiB if r * cols + c < world else float("nan")
                 for c in range(cols)] for r in range(rows)]
        im = ax.imshow(grid, cmap=BLUE_RAMP, norm=norm)
        for r in range(rows):
            for c in range(cols):
                if r * cols + c < world:
                    v = grid[r][c]
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                            color=SURFACE if v > vmax * 0.35 else INK)
        ax.set_title(f"{STAGE_NAMES[s]}: {sum(peaks[s]) / MiB:.0f} MiB in total", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("peak MiB per GPU", color=MUTED)
    ticks = [vmin * (vmax / vmin) ** (i / 4) for i in range(5)]
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.yaxis.set_tick_params(color=MUTED)
    fig.suptitle(f"All {world} virtual GPUs, peak memory per GPU (each cell is one GPU)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig


def memory_timeline(timelines: dict):
    """timelines[s] = list of ledger events (phase, total, w, g, opt, act, temp)."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.2))
    for s, ax in enumerate(axes):
        ev = timelines[s]
        x = list(range(len(ev)))
        ys = [[_mib(e[i]) for e in ev] for i in (2, 3, 4, 5, 6)]
        ax.stackplot(x, ys, colors=[CAT_COLORS[c] for c in CATS], alpha=0.9,
                     labels=[CAT_LABELS[c] for c in CATS], linewidth=0)
        # shade the phases
        def span(prefix):
            idx = [i for i, e in enumerate(ev) if e[0].startswith(prefix)]
            return (min(idx), max(idx)) if idx else None
        for prefix, name in (("fwd", "forward"), ("bwd", "backward"), ("step", "optimizer step")):
            sp = span(prefix)
            if sp:
                ax.axvspan(sp[0], sp[1], color=INK, alpha=0.04, lw=0)
                ax.text((sp[0] + sp[1]) / 2, 1.0, name, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color=MUTED)
        peak = max(e[1] for e in ev)
        ax.set_ylabel("MiB")
        ax.set_xlim(0, len(ev) - 1)
        ax.set_title(f"{STAGE_NAMES[s]}: peak {_mib(peak):.2f} MiB", loc="left", fontsize=10,
                     pad=12)
        ax.set_xticks([])
        if s == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    axes[-1].set_xlabel("allocation / free events on GPU 0 during one training step  →")
    fig.suptitle("Memory on GPU 0 through one step: DDP keeps every model state resident; "
                 "ZeRO-3 gathers, uses and frees one unit at a time", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def comm_and_compute(summary: dict):
    """summary[s] = dict(bytes_by_op, calls, flops_fwd, flops_bwd, opt_elements)."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    xs = list(range(4))
    labels = [STAGE_NAMES[s].replace(" (ZeRO-0)", "") for s in range(4)]
    ops = [("all_reduce", SERIES[0]), ("reduce_scatter", SERIES[1]), ("all_gather", SERIES[2])]
    ax = axes[0]
    bottom = [0.0] * 4
    for op, col in ops:
        vals = [summary[s]["bytes_by_op"].get(op, 0) / 1e6 for s in range(4)]
        ax.bar(xs, vals, bottom=bottom, width=0.5, color=col, edgecolor=SURFACE, linewidth=2,
               label=op.replace("_", "-"))
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, b in zip(xs, bottom):
        ax.text(x, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_title("MB sent per GPU per step", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(0, max(bottom) * 1.3)
    ax = axes[1]
    calls = [summary[s]["calls"] for s in range(4)]
    ax.bar(xs, calls, width=0.5, color=SERIES[0])
    for x, c in zip(xs, calls):
        ax.text(x, c, str(c), ha="center", va="bottom", fontsize=8.5)
    ax.set_title("collective calls per step", loc="left")
    ax.set_ylim(0, max(calls) * 1.25)
    ax = axes[2]
    w = 0.34
    fw = [summary[s]["flops_fwd"] / 1e6 for s in range(4)]
    bw = [summary[s]["flops_bwd"] / 1e6 for s in range(4)]
    ax.bar([x - w / 2 for x in xs], fw, width=w, color=SERIES[0], edgecolor=SURFACE, lw=2,
           label="forward")
    ax.bar([x + w / 2 for x in xs], bw, width=w, color=SERIES[1], edgecolor=SURFACE, lw=2,
           label="backward")
    ax.set_title("MFLOPs per GPU per step (counted)", loc="left")
    ax.set_ylim(0, max(bw) * 1.35)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax = axes[3]
    el = [summary[s]["opt_elements"] for s in range(4)]
    ax.bar(xs, el, width=0.5, color=SERIES[2])
    ax.set_yscale("log")
    for x, e in zip(xs, el):
        ax.text(x, e, f"{e:,}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title("optimizer elements updated per GPU", loc="left")
    ax.set_ylim(min(el) / 2, max(el) * 6)
    for ax in axes:
        ax.set_xticks(xs, labels, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("What changes per step: ZeRO-3 sends 1.5x the bytes; forward/backward "
                 "compute does not change; optimizer work drops by N", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def scaling(ns: list, measured: dict, formula: dict):
    """measured[s] / formula[s] = bytes-per-parameter values over ns (model states / P)."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for s in range(4):
        ax.plot(ns, formula[s], color=STAGE_COLORS[s], lw=2, alpha=0.55)
        ax.plot(ns, measured[s], "o", color=STAGE_COLORS[s], ms=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=STAGE_NAMES[s])
        ax.text(ns[-1] * 1.12, measured[s][-1], f"{STAGE_NAMES[s]}  {measured[s][-1]:.2f} B",
                va="center", fontsize=9)
    for y, txt in ((4, "ZeRO-1 floor: 4 B (weights + grads never shrink)"),
                   (2, "ZeRO-2 floor: 2 B (weights never shrink)")):
        ax.axhline(y, color=MUTED, lw=1)
        ax.text(ns[0], y + 0.25, txt, fontsize=8, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlim(ns[0] * 0.8, ns[-1] * 3.2)
    ax.set_ylim(0, 17.5)
    ax.set_xlabel("number of GPUs N")
    ax.set_ylabel("bytes per parameter, per GPU")
    ax.set_title("Scaling out: memory per GPU for model states (dots measured, lines = formula)",
                 loc="left")
    ax.legend(loc="center", bbox_to_anchor=(0.62, 0.62), fontsize=8.5)
    fig.tight_layout()
    return fig


def loss_curves(losses: dict, long_run: list | None = None):
    n = 2 if long_run else 1
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 3.8), squeeze=False)
    ax = axes[0][0]
    styles = ["-", "--", "-.", ":"]
    for s in range(4):
        ax.plot(range(1, len(losses[s]) + 1), losses[s], styles[s], color=STAGE_COLORS[s],
                lw=2, label=STAGE_NAMES[s])
    diff = max(abs(a - b) for s in (1, 2, 3) for a, b in zip(losses[s], losses[0]))
    ax.text(0.98, 0.95, f"max |loss - DDP loss| = {diff:g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlabel("step")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("training loss (mean over 32 GPUs)")
    ax.set_title("Four stages, one curve: ZeRO changes memory, not maths", loc="left")
    ax.legend(fontsize=8.5, loc="center right")
    if long_run:
        ax = axes[0][1]
        ax.plot(range(1, len(long_run) + 1), long_run, color=STAGE_COLORS[3], lw=2)
        ax.set_xlabel("step")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("training loss")
        ax.set_title(f"ZeRO-3 on 32 virtual GPUs, {len(long_run)} steps", loc="left")
    fig.tight_layout()
    return fig


def oom_ladder(capacities: list, predicted: dict, observed: dict):
    """predicted/observed[(s, i)] = True if stage s fits in capacities[i]."""
    fig, ax = plt.subplots(figsize=(1.9 * len(capacities) + 2.4, 3.2))
    ax.set_xlim(-0.5, len(capacities) - 0.5)
    ax.set_ylim(-0.5, 3.5)
    for i in range(len(capacities)):
        for s in range(4):
            ok = observed[(s, i)]
            match = ok == predicted[(s, i)]
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (i - 0.42, 3 - s - 0.36), 0.84, 0.72, boxstyle="round,pad=0,rounding_size=0.06",
                color=GOOD if ok else CRITICAL, alpha=0.85, lw=0))
            ax.text(i, 3 - s, ("trains" if ok else "out of memory") + ("" if match else "  (!)"),
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(range(len(capacities)), [f"{c / MiB:.2f} MiB" for c in capacities])
    ax.set_yticks(range(4), [STAGE_NAMES[s] for s in (3, 2, 1, 0)])
    ax.set_xlabel("memory capacity of each of the 32 virtual GPUs")
    ax.grid(False)
    ax.set_title("Out-of-memory ladder: what happened on each capacity (every cell matched "
                 "the prediction made before running)" if all(observed[k] == predicted[k]
                                                             for k in observed)
                 else "Out-of-memory ladder: what happened ((!) = differs from prediction)",
                 loc="left")
    fig.tight_layout()
    return fig


def grad_accum_and_ckpt(accum: dict, ckpt: dict):
    """accum[s] = (bytes G=1, bytes G=G); ckpt = dict(no=..., yes=...) with act, flops."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    xs = list(range(4))
    w = 0.34
    ax = axes[0]
    g = accum["G"]
    one = [accum[s][0] / 1e6 for s in range(4)]
    many = [accum[s][1] / 1e6 for s in range(4)]
    ax.bar([x - w / 2 for x in xs], one, width=w, color=SERIES[0], edgecolor=SURFACE, lw=2,
           label="1 micro-batch per step")
    ax.bar([x + w / 2 for x in xs], many, width=w, color=SERIES[1], edgecolor=SURFACE, lw=2,
           label=f"{g} micro-batches per step (grad accumulation)")
    for x, v in zip(xs, many):
        ax.text(x + w / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [STAGE_NAMES[s].replace(" (ZeRO-0)", "") for s in range(4)])
    ax.set_title("MB sent per GPU per optimizer step", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(0, max(many) * 1.3)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    vals = [ckpt["no"]["act"] / MiB, ckpt["yes"]["act"] / MiB]
    ax.bar([0, 1], vals, width=0.5, color=[SERIES[3], SERIES[3]])
    for x, v in zip([0, 1], vals):
        ax.text(x, v, f"{v:.2f} MiB", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks([0, 1], ["keep activations", "activation\ncheckpointing"])
    ax.set_title("peak activations per GPU (ZeRO-3)", loc="left")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    bottom = [0.0, 0.0]
    for key, col, lab in (("forward", SERIES[0], "forward"), ("recompute", SERIES[4], "recompute"),
                          ("backward", SERIES[1], "backward")):
        v = [ckpt["no"]["flops"].get(key, 0) / 1e6, ckpt["yes"]["flops"].get(key, 0) / 1e6]
        ax.bar([0, 1], v, bottom=bottom, width=0.5, color=col, edgecolor=SURFACE, lw=2, label=lab)
        bottom = [b + x for b, x in zip(bottom, v)]
    for x, b in zip([0, 1], bottom):
        ax.text(x, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks([0, 1], ["keep activations", "activation\ncheckpointing"])
    ax.set_title("MFLOPs per GPU per step", loc="left")
    ax.legend(fontsize=7.5, loc="upper left", ncol=3)
    ax.set_ylim(0, max(bottom) * 1.35)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def real_hardware(rows: list, gpu_mem_gb: float, title: str):
    """rows: zero_theory.real_hardware_rows(). Bars = model states per GPU for each stage;
    diamonds = states + activations (with the rows' recompute setting)."""
    rows = [r for r in rows if r["variant"] in ("DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3")]
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    w = 0.19
    for s in range(4):
        xs, states, totals = [], [], []
        for i, m in enumerate(models):
            r = next(r for r in rows if r["model"] == m and r["stage"] == s)
            xs.append(i + (s - 1.5) * w)
            states.append(r["states_gb"])
            totals.append(r["total_gb"])
        ax.bar(xs, states, width=w * 0.9, color=STAGE_COLORS[s], label=STAGE_NAMES[s])
        ax.plot(xs, totals, "D", ms=6, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label="+ activations (selective recompute)" if s == 3 else None)
    ax.axhline(gpu_mem_gb, color=CRITICAL, lw=1.5)
    ax.text(-0.45, gpu_mem_gb * 1.12, f"{gpu_mem_gb:.0f} GB per GPU", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel("GB per GPU (log scale)")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
