"""Every figure in the report, generated from assets/results.json and assets/ladder.json.

Conventions held to throughout, so the whole set reads as one system:

  * colour identifies the *variant* and nothing else, assigned in a fixed order and never
    recycled - the baseline is blue in every chart it appears in;
  * every series is also direct-labelled, so identity never rests on colour alone;
  * one y-axis per chart, always (two scales on one frame is the single most effective way
    to make an honest chart lie);
  * grid and axes recede, marks are thin, and numbers sit in text ink rather than series
    colour.

The palette is the validated dark set: worst adjacent CVD Delta E 8.4, worst normal-vision
19.3, all six above 3:1 on the surface.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE = "#14141c"
INK = "#ffffff"
INK_2 = "#c3c2b7"
INK_3 = "#87867e"
GRID = "#2b2b36"

# fixed slot order; a variant keeps its hue in every figure
COLORS = {
    "store": "#3987e5",           # slot 1 blue    - the baseline
    "checkpoint": "#d95926",      # slot 2 orange  - the honest rival
    "euler": "#199e70",           # slot 3 aqua    - symplectic Euler
    "midpoint": "#c98500",        # slot 4 yellow  - leapfrog
    "coupling": "#d55181",        # slot 5 magenta - RevNet
    "euler_implicit": "#008300",  # slot 6 green   - the conditional one
}
LABEL = {
    "store": "baseline (stored)",
    "checkpoint": "checkpointing",
    "euler": "symplectic Euler",
    "midpoint": "midpoint",
    "coupling": "coupling",
    "euler_implicit": "implicit Euler",
}
ORDER = ["store", "checkpoint", "euler", "midpoint", "coupling", "euler_implicit"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "xtick.color": INK_3, "ytick.color": INK_3,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "figure.dpi": 130, "lines.linewidth": 2.0, "lines.markersize": 5,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _frame(ax, title=None, sub=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK, loc="left", pad=16 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_3, fontsize=9, va="bottom")
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_axisbelow(True)


def _end_label(ax, x, y, text, color, dx=6):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, 0),
                color=color, fontsize=9, va="center", fontweight="medium")


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
def _gutter_labels(ax, items, x, min_gap):
    """Place right-hand labels at their series' height, nudged apart so none overlap.

    Several variants here have *identical* memory curves (symplectic Euler and midpoint
    allocate the same tensors), so their labels land on the same pixel. Greedy vertical
    separation keeps every one readable without moving it far from what it names.
    """
    items = sorted(items, key=lambda it: -it[0])
    placed = []
    for y, text, color in items:
        if placed and placed[-1][0] - y < min_gap:
            y = placed[-1][0] - min_gap
        placed.append((y, text, color))
    for y, text, color in placed:
        ax.annotate(text, (x, y), color=color, fontsize=9, va="center",
                    fontweight="medium", annotation_clip=False)


def memory_vs_batch(ladder, out_dir):
    """The headline: what each variant costs as the batch grows, and where it dies."""
    style()
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    rows = ladder["rows"]
    chunked = ladder.get("chunked", {}).get("rows", [])
    cap = ladder["rows"][0].get("vram_cap_gib") if ladder.get("rows") else None
    labels, ymax = [], 0

    def series(data, mode, dash, tag):
        nonlocal ymax
        pts = sorted({(r["batch"], r["peak_alloc_gib"]) for r in data
                      if r["mode"] == mode and not r.get("degraded")})
        if not pts:
            return
        xs, ys = zip(*pts)
        ymax = max(ymax, max(ys))
        ax.plot(xs, ys, color=COLORS[mode], marker="o", ms=4.5 if not dash else 3.5,
                ls="--" if dash else "-", alpha=.9 if dash else 1)
        ax.plot([xs[-1]], [ys[-1]], marker="X", ms=11, color=COLORS[mode],
                mec=SURFACE, mew=2, zorder=5)
        labels.append((ys[-1], f"{tag} · max {xs[-1]}", COLORS[mode]))

    # symplectic Euler and midpoint allocate identically; draw one line for the pair
    for mode in ("store", "checkpoint", "midpoint", "coupling"):
        tag = {"midpoint": "midpoint = symplectic Euler"}.get(mode, LABEL[mode])
        series(rows, mode, False, tag)
    series(chunked, "midpoint", True, "same, chunked cross entropy")

    if cap:
        ax.axhline(cap, color=INK_3, ls=":", lw=1.2)
        ax.text(ax.get_xlim()[1], cap + ymax * .015, f"{cap:.2f} GiB usable  ",
                color=INK_3, fontsize=9, va="bottom", ha="right")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    _frame(ax, "Peak memory against batch size",
           "X marks the largest batch that ran; the next size up ran out of memory",
           "batch size (sequences of 512 tokens)", "peak allocated (GiB)")
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 * 3.6)
    _gutter_labels(ax, labels, x1 * 1.12, ymax * 0.075)
    return save(fig, out_dir, "memory_vs_batch.png")


def memory_vs_depth(depth_rows, out_dir):
    """The claim itself: stored memory grows with depth, reversible memory does not."""
    style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for mode in ORDER:
        pts = sorted([(r["n_layer"], r["per_batch_gib"]) for r in depth_rows
                      if r["mode"] == mode])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLORS[mode], marker="o", label=LABEL[mode])
        _end_label(ax, xs[-1], ys[-1], LABEL[mode], COLORS[mode])
    _frame(ax, "What one batch costs, against depth",
           "batch 16 throughout; the fixed weight and optimiser floor is subtracted",
           "layers", "memory per batch (GiB)")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.45)
    ax.legend(frameon=False, labelcolor=INK_2, loc="upper left", fontsize=9)
    return save(fig, out_dir, "memory_vs_depth.png")


def loss_curves(results, out_dir, key="final_val_loss"):
    """Training trajectories. The baseline and implicit Euler share a forward pass, so
    the interesting thing is whether their curves come apart."""
    style()
    fig, ax = plt.subplots(figsize=(8.2, 5))
    for r in results:
        mode, name = r["spec"]["mode"], r["spec"]["name"]
        if "probe" in r["spec"].get("tags", []) or name.startswith("D"):
            continue
        hist = r["history"]
        xs = [h["tokens"] / 1e6 for h in hist]
        ys = [h["loss"] for h in hist]
        ax.plot(xs, ys, color=COLORS[mode], label=LABEL[mode])
        _end_label(ax, xs[-1], ys[-1], f"{LABEL[mode]} · {r[key]:.3f}", COLORS[mode])
    _frame(ax, "Training loss against tokens seen",
           "same seed, same sampler stream, same schedule, same 50M-token budget",
           "tokens (millions)", "training loss")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.5)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)
    return save(fig, out_dir, "loss_curves.png")


def speed_vs_memory(results, out_dir):
    """The actual trade. Every point is direct-labelled: with all pairs on screen at once
    colour alone would not separate six series safely."""
    style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        if r["spec"]["name"].startswith("D2"):
            continue
        mode = r["spec"]["mode"]
        x, y = r["peak_alloc_gib"], r["tokens_per_sec"]
        ax.scatter([x], [y], s=110, color=COLORS[mode], edgecolor=SURFACE,
                   linewidth=2, zorder=4)
        tag = LABEL[mode] + (f"  (batch {r['spec']['batch_size']})"
                             if r["spec"]["name"].startswith("D") else "")
        ax.annotate(tag, (x, y), textcoords="offset points", xytext=(10, 4),
                    color=INK_2, fontsize=9)
    _frame(ax, "What reversibility costs and what it buys",
           "up is faster, left is smaller - the baseline sits top-right",
           "peak allocated (GiB)", "throughput (tokens/second)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    return save(fig, out_dir, "speed_vs_memory.png")


def memory_decomposition(results, out_dir):
    """Where the peak actually goes - the figure that stops 'O(1) memory' being a slogan."""
    style()
    runs = [r for r in results if not r["spec"]["name"].startswith("D2")]
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(runs) + 2.2))
    ys = range(len(runs))
    for i, r in enumerate(runs):
        state = r["state_gib"]
        logits = min(r["analytic"]["logits_gib"], max(0.0, r["per_batch_gib"]))
        rest = max(0.0, r["per_batch_gib"] - logits)
        mode = r["spec"]["mode"]
        # 2px surface gaps between segments
        ax.barh(i, state, color=INK_3, height=0.55, edgecolor=SURFACE, linewidth=2)
        ax.barh(i, rest, left=state, color=COLORS[mode], height=0.55,
                edgecolor=SURFACE, linewidth=2)
        ax.barh(i, logits, left=state + rest, color=COLORS[mode], alpha=0.42,
                height=0.55, edgecolor=SURFACE, linewidth=2)
        ax.text(state + rest + logits + 0.06, i, f"{r['peak_alloc_gib']:.2f} GiB",
                va="center", color=INK_2, fontsize=9)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{LABEL[r['spec']['mode']]}  b={r['spec']['batch_size']}"
                        for r in runs], color=INK_2, fontsize=9)
    ax.invert_yaxis()
    _frame(ax, "Where the peak goes",
           "grey: weights, gradients, Adam states  ·  solid: activations  ·  faded: fp32 logits",
           "GiB", None)
    ax.grid(True, axis="x", alpha=0.5)
    ax.grid(False, axis="y")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.16)
    return save(fig, out_dir, "memory_decomposition.png")


def reconstruction_drift(drifts, out_dir):
    """How far the reconstructed activations sit from the true ones, layer by layer."""
    style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for mode, series in drifts.items():
        xs = list(range(len(series)))
        ax.plot(xs, [max(v, 1e-12) for v in series], color=COLORS[mode],
                marker="o", label=LABEL[mode])
        _end_label(ax, xs[0], max(series[0], 1e-12), LABEL[mode], COLORS[mode], dx=-4)
    ax.set_yscale("log")
    _frame(ax, "Reconstruction error, layer by layer",
           "walking backwards from the last layer to the first; fp32 stream, bf16 compute",
           "layer (backward walk runs right to left)", "max |reconstructed - true|")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="lower right")
    return save(fig, out_dir, "reconstruction_drift.png")


def euler_condition(report_by_gamma, out_dir):
    """Why the obvious reading of 'reversible Euler' is conditional, in two panels."""
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    gammas = sorted(report_by_gamma)
    lips = [max(report_by_gamma[g]["lipschitz"]) for g in gammas]
    a1.plot(gammas, lips, color=COLORS["euler_implicit"], marker="o")
    a1.axhline(1.0, color="#e66767", ls="--", lw=1.4)
    a1.text(gammas[0], 1.04, " contraction fails above this line", color="#e66767", fontsize=9)
    a1.set_xscale("log")
    a1.set_yscale("log")
    _frame(a1, "The condition", "Lip(G) = gamma x Lip(F) must stay below 1",
           "gamma (LayerScale init)", "Lip(G)")

    for g in gammas:
        sweep = report_by_gamma[g]["sweep"]
        xs = [s["iters"] for s in sweep]
        ys = [max(s["err"], 1e-12) for s in sweep]
        lip = max(report_by_gamma[g]["lipschitz"])
        ok = lip < 1
        a2.plot(xs, ys, marker="o", color=COLORS["euler_implicit"] if ok else "#e66767",
                alpha=0.5 + 0.5 * ok, ls="-" if ok else "--")
        _end_label(a2, xs[-1], ys[-1], f"gamma {g} (Lip {lip:.2f})",
                   COLORS["euler_implicit"] if ok else "#e66767")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    _frame(a2, "The consequence", "more iterations only help on the right side of the line",
           "fixed-point iterations", "reconstruction error")
    a2.set_xlim(right=a2.get_xlim()[1] * 3.4)
    fig.tight_layout()
    return save(fig, out_dir, "euler_condition.png")


def throughput(bench, out_dir):
    """Speed at thermal equilibrium, order-balanced. Bars, because it is one number each."""
    style()
    rows = sorted(bench["summary"], key=lambda r: -r["tokens_per_sec"])
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(rows) + 1.9))
    base = [r for r in rows if r["mode"] == "store"][0]["tokens_per_sec"]
    for i, r in enumerate(rows):
        ax.barh(i, r["tokens_per_sec"], color=COLORS[r["mode"]], height=.58,
                edgecolor=SURFACE, linewidth=2)
        ax.plot([min(r["passes"]), max(r["passes"])], [i, i], color=SURFACE, lw=2,
                solid_capstyle="butt")
        ax.text(r["tokens_per_sec"] + base * .012, i,
                f"{r['tokens_per_sec']:,.0f}   {base/r['tokens_per_sec']:.2f}x slower",
                va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[r["mode"]] for r in rows], color=INK_2, fontsize=9)
    ax.invert_yaxis()
    _frame(ax, "Throughput at thermal equilibrium",
           "order-balanced: each variant measured once forwards and once backwards, "
           "then averaged", "tokens / second", None)
    ax.grid(True, axis="x", alpha=.5)
    ax.grid(False, axis="y")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.3)
    return save(fig, out_dir, "throughput.png")


def diagnostics(diag, out_dir):
    """Three things that only show up once a reversible model is actually training."""
    style()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 4.3))

    for mode, v in diag.items():
        pts = [(d["step"], d["recon_h0"]) for d in v["diagnostics"] if "recon_h0" in d]
        if not pts:
            continue
        xs, ys = zip(*pts)
        a1.plot(xs, ys, color=COLORS[mode], label=LABEL[mode])
        _end_label(a1, xs[-1], ys[-1], LABEL[mode].split()[0], COLORS[mode])
    a1.set_yscale("log")
    _frame(a1, "Reconstruction error", "how far the rebuilt h0 sits from the true one",
           "training step", "max |rebuilt - true|")
    a1.set_xlim(right=a1.get_xlim()[1] * 1.45)

    lip = [(d["step"], d["lipschitz"]) for d in diag["euler_implicit"]["diagnostics"]
           if "lipschitz" in d]
    if lip:
        xs, ys = zip(*lip)
        a2.plot(xs, ys, color=COLORS["euler_implicit"])
        a2.axhline(1.0, color="#e66767", ls="--", lw=1.4)
        a2.text(xs[0], 1.15, " below this line the inverse converges", color="#e66767",
                fontsize=9)
        a2.set_yscale("log")
    _frame(a2, "Implicit Euler leaves its safe region",
           "gamma is learnable, and training pushes it the wrong way",
           "training step", "Lip(G)")

    par = [(d["step"], d["parasitic"]) for d in diag["midpoint"]["diagnostics"]
           if "parasitic" in d]
    if par:
        xs, ys = zip(*par)
        a3.plot(xs, ys, color=COLORS["midpoint"])
    _frame(a3, "Midpoint's parasitic mode grows",
           "the odd and even chains drift apart - leapfrog's known failure",
           "training step", "mean |h_l - h_{l-1}|")
    fig.tight_layout()
    return save(fig, out_dir, "diagnostics.png")


def integrator_figures(out_dir):
    """Regenerate the two figures that characterise the integrators themselves.

    These need a model rather than a results file, so they used to be produced only by
    notebook 01 - which meant `python -m revlm.plots` could not rebuild everything the
    README references, and the staleness guard would tell you to run a command that could
    not fix it. They belong here.
    """
    import torch

    from .model import GPT, GPTConfig
    from . import reversible as R

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}
    for g in (0.02, 0.05, 0.1, 0.25):
        cfg = GPTConfig(n_layer=4, block_size=128, gamma_init=g)
        torch.manual_seed(0)
        m = GPT(cfg, mode="euler_implicit").to(dev)
        h0 = m.embed(torch.randint(0, cfg.vocab_size, (2, 128), device=dev))
        torch.manual_seed(7)
        report[g] = R.euler_implicit_report(m, h0, iters=(1, 2, 4, 8, 16, 32))
        del m

    drifts = {}
    for mode in ("euler", "midpoint", "coupling"):
        cfg = GPTConfig(n_layer=10, block_size=256)
        torch.manual_seed(0)
        m = GPT(cfg, mode=mode).to(dev)
        h0 = m.embed(torch.randint(0, cfg.vocab_size, (4, 256), device=dev))
        drifts[mode] = R.reconstruction_drift(m, h0)
        del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return [euler_condition(report, out_dir), reconstruction_drift(drifts, out_dir)]


def all_figures(out_dir="assets"):
    res = json.load(open(os.path.join(out_dir, "results.json")))
    made = []
    lad = res.get("ladder") or {}
    if lad.get("rows"):
        made.append(memory_vs_batch(lad, out_dir))
    if lad.get("depth_rows"):
        made.append(memory_vs_depth(lad["depth_rows"], out_dir))
    made.append(loss_curves(res["runs"], out_dir))
    made.append(speed_vs_memory(res["runs"], out_dir))
    made.append(memory_decomposition(res["runs"], out_dir))
    bench_path = os.path.join(out_dir, "throughput.json")
    if os.path.exists(bench_path):
        made.append(throughput(json.load(open(bench_path)), out_dir))
    diag_path = os.path.join(out_dir, "diagnostics.json")
    if os.path.exists(diag_path):
        made.append(diagnostics(json.load(open(diag_path)), out_dir))
    made.extend(integrator_figures(out_dir))
    return made


if __name__ == "__main__":
    for p in all_figures():
        print("wrote", p)
