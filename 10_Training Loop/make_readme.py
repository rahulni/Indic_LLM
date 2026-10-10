"""Render README.md from assets/results.json, so every number in it comes from the stored full run.

Usage:  python make_readme.py
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "assets" / "results.json").read_text(encoding="utf-8"))
NB = "inside_the_training_loop.ipynb"
COLAB = f"https://colab.research.google.com/github/rahulni/Indic_LLM/blob/main/10_Training%20Loop/{NB}"


def g(path, default=None):
    cur = R
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig(name, alt):
    return f"![{alt}](assets/{name}.png)" if (HERE / "assets" / f"{name}.png").exists() else ""


def mean_sd(xs, fmt="{:+.4f}"):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return fmt.format(xs.mean())
    return f"{fmt.format(xs.mean())} ± {fmt.format(xs.std(ddof=1)).lstrip('+')}"


def arr(y):
    return np.array([np.nan if v is None else v for v in y], dtype=float)


def twin_rows(tw):
    rows = []
    for sd in sorted(tw, key=int):
        ok, bug = tw[sd]["token"], tw[sd]["avg_of_avg"]
        gap = 100 * (arr(bug["loss"]) - arr(bug["true_loss"])) / arr(bug["true_loss"])
        rows.append(dict(seed=int(sd), gap=float(np.median(gap)), val_ok=ok["val"]["loss"][-1], val_bug=bug["val"]["loss"][-1],
                         bucket=(np.array(bug["bucket_val"]) - np.array(ok["bucket_val"])).tolist()))
    return rows


def step_txt(x):
    return "never" if x is None else str(x)


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


env = g("env", {})
parts = []

# ---------------------------------------------------------------- header
parts.append(f"""# Inside the Training Loop

*Making a small language model tell the truth about itself.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

One notebook, [`{NB}`]({NB}), takes a real training loop apart and checks every piece with a measurement. It doubles as study notes: each section gives the intuition, the math, the code, and an `assert` that fails loudly if the claim is wrong.

Two models share one modern decoder class (RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings):

- **Model A**: 31.5M parameters, trained from scratch on TinyStories with SmolLM2's 49,152-token vocabulary. Used for every training experiment.
- **SmolLM2-135M**: the real pretrained weights, loaded into the same class. Used for the shape tour, a gradient check on trained weights, MFU at real width, and a short fine-tune.

Every number below comes from one full run on an **{env.get('gpu', 'GPU')}** ({env.get('vram_gib', '?')} GiB, sm{env.get('capability', '?').replace('.', '')}), torch {env.get('torch', '?')}, finished on {env.get('date', '?')}, all on AC power in the laptop's Turbo mode. The run was interrupted once, when the laptop went to sleep. It was resumed, and experiments already stored were reused rather than re-run, which is why some notebook cells say "reusing the result stored earlier". All of the numbers are stored in [`assets/results.json`](assets/results.json).
""")

# ---------------------------------------------------------------- at a glance
glance = []
ta = g("tour_A")
if ta:
    big = max(ta["rows"], key=lambda r: r[3])
    glance.append(("What is the biggest tensor in a step?", f"the logits, B×T×V = {big[1]} ({big[3] / 2**20:.0f} MiB in fp32), more than all of Model A's weights"))
gc = g("grad_check")
if gc:
    rows = gc["A"] + gc["smol"]
    glance.append(("Does `backward()` agree with nudging one weight?",
                   f"yes: {len(rows)} entries, every parameter type, two models, float64; worst {min(r['digits'] for r in rows):.1f} matching digits"))
ag, tw = g("accum_gradients"), g("accum_twins_seeds")
if ag:
    glance.append(("How wrong is the average-of-averages gradient?",
                   f"median {np.median([r['bug_err'] for r in ag]):.1e} relative error vs {np.median([r['ok_err'] for r in ag]):.1e} for token-weighted accumulation"))
if tw:
    rws = twin_rows(tw)
    glance.append(("…and what does it do to training?",
                   f"over {len(rws)} seeds the logged loss is off by {mean_sd([r['gap'] for r in rws], '{:+.1f}')}%, and the model is tilted towards "
                   f"short stories in {sum(r['bucket'][0] < 0 < r['bucket'][-1] for r in rws)} of {len(rws)} seeds; overall held-out loss "
                   f"{mean_sd([r['val_bug'] - r['val_ok'] for r in rws])} vs correct"))
det = g("stress_detect_seeds") or {}
fxs = g("stress_detect_fixed_seeds") or {}
if det:
    first_ = sum(d["norm_step"] is not None and (d["loss_step"] is None or d["norm_step"] < d["loss_step"]) for d in det.values())
    lds = [v["loss_step"] - v["norm_step"] for f_ in fxs.values() for v in f_.values() if v["norm_step"] is not None and v["loss_step"] is not None]
    n_cases = sum(len(f_) for f_ in fxs.values())
    all_lead = bool(lds) and len(lds) == n_cases and all(l > 0 for l in lds)
    glance.append(("Did the gradient norm move before the loss?",
                   (f"in an instability created on purpose, against a fixed pre-ramp baseline: **yes, {min(lds)}–{max(lds)} steps earlier** in all {n_cases} seed × threshold cases. "
                    if all_lead else (f"against a fixed pre-ramp baseline the lead ranged {min(lds)}–{max(lds)} steps. " if lds else ""))
                   + f"The rolling-window rule fixed in advance flagged the norm first in only {first_} of {len(det)} seeds, so that detector missed it. "
                   + ("A healthy run showed no event at all." if g("natural_events") == [] else "")))
bt = g("bad_batch_seeds_table")
if bt:
    glance.append(("Does clipping help?", f"against a burst of noise batches it cut the worst damage in {sum(r['damage_clip'] < r['damage_none'] for r in bt)} "
                   f"of {len(bt)} seeds ({mean_sd([r['damage_none'] for r in bt])} → {mean_sd([r['damage_clip'] for r in bt])} held-out loss)"))
mh = g("mfu_headline")
if mh:
    glance.append(("MFU of the training loop", f"**{pct(mh['mfu_spec'])}** of the spec peak ({mh['tflops']:.1f} TFLOP/s); {pct(mh['mfu_obs_clock'])} at the clock actually observed"))
glance.append(("0.1 in fp32 / bf16 / fp8 E4M3", "`0x3DCCCCCD` / `0x3DCD` / `0x1D`, which store 0.1000000015 / 0.1000977 / 0.1015625"))
glance.append(("Which format to train in?", "bf16 autocast with fp32 master weights and fp32 AdamW state"))
parts.append("## Results at a glance\n\n" + table(glance, ["question", "measured answer"]))

# ---------------------------------------------------------------- 1. shapes
if ta:
    dims = [("B", "sequences in the micro-batch", 8, 4), ("T", "positions per sequence", 512, 512), ("C", "residual-stream width", 384, 576),
            ("H", "query heads", 6, 9), ("H_kv", "key/value heads (GQA)", 2, 3), ("D_h", "width of one head, C/H", 64, 64),
            ("F", "MLP hidden width", 1024, 1536), ("V", "vocabulary size", "49,152", "49,152")]
    def size(n):
        return f"{n / 2**20:.1f} MiB" if n >= 2**20 else (f"{n / 1024:.0f} KiB" if n >= 1024 else f"{n} B")
    rows = [(r[0], r[1], r[2], size(r[3]), r[4]) for r in ta["rows"]]
    parts.append(f"""## 1. Every tensor in one step

{table(dims, ["symbol", "meaning", "Model A", "SmolLM2-135M"])}

Model A, one micro-batch of B={ta['B']} × T={ta['T']}, layer 0 in full (layers {ta['skipped'][0]}–{ta['skipped'][1]} repeat it exactly), recorded in fp32 (under bf16 autocast every matmul output becomes bfloat16). The shapes come from forward hooks on every module, plus `rec()` calls inside attention for the tensors that are not module outputs:

{table(rows, ["tensor", "shape", "dtype", "size", "what the dimensions mean"])}

Every gradient and both AdamW states have exactly their weight's shape: 31,463,808 weights → 31,463,808 gradients + 2 × 31,463,808 optimizer numbers. The tied output head is the embedding matrix, so it gets one gradient, accumulated from two uses.""")

# ---------------------------------------------------------------- 2. gradient check
if gc:
    by_a = {r["param"]: r for r in gc["A"]}
    rows = []
    for ra, rs in zip(gc["A"], gc["smol"]):
        kind = ra["param"].split(".")[-2] if ra["param"].count(".") >= 2 else ra["param"].split(".")[0]
        rows.append((kind, ra["autograd_s"], ra["numeric_s"], f"{ra['digits']:.1f}", f"{rs['digits']:.1f}"))
    dis = [(d["case"], f"{d['rel_err']:.1e}", d["why"]) for d in gc["disagree"]]
    sweep = gc["fp32_sweep"]
    best32 = min(r["rel_err"] for r in sweep)
    parts.append(f"""## 2. One gradient, checked by hand

Nudge one weight by ±ε, run the whole model each time, and compare the slope with `w.grad`. This is done for the largest-gradient entry of ten parameter types, in **float64**, eval mode, ε = 1e-5. Model A uses its trained weights, SmolLM2 its pretrained ones:

{table(rows, ["parameter", "Model A: autograd", "Model A: nudge", "Model A: matching digits", "SmolLM2: matching digits"])}

In fp32 the same check cannot do better than a relative error of {best32:.1e}, whatever ε you pick. That is why the check runs in float64. When they **disagree** on purpose, each case points at a real bug class:

{table(dis, ["setup", "relative error", "why"])}""")

# ---------------------------------------------------------------- 3. accumulation
if ag and tw:
    rws = twin_rows(tw)
    first = tw[sorted(tw, key=int)[0]]
    nk = np.median(np.array(first["token"]["n_k"]), axis=0).astype(int).tolist()
    D = np.array([r["bucket"] for r in rws])
    trows = [(r["seed"], f"{r['gap']:+.2f}%", f"{r['val_ok']:.4f}", f"{r['val_bug']:.4f}", f"{r['val_bug'] - r['val_ok']:+.4f}",
              ", ".join(f"{v:+.3f}" for v in r["bucket"])) for r in rws]
    trows.append(("mean ± sd", mean_sd([r["gap"] for r in rws], "{:+.2f}") + "%", mean_sd([r["val_ok"] for r in rws], "{:.4f}"),
                  mean_sd([r["val_bug"] for r in rws], "{:.4f}"), mean_sd([r["val_bug"] - r["val_ok"] for r in rws]),
                  ", ".join(f"{v:+.3f}" for v in D.mean(0))))
    tilt = sum(r["bucket"][0] < 0 < r["bucket"][-1] for r in rws)
    parts.append(f"""## 3. Gradient accumulation, and the average-of-averages bug

The correct recipe counts the real tokens $N$ in the whole global batch first, then backpropagates each micro-batch's *summed* loss divided by $N$. The bug (in major frameworks until 2024) averages each micro-batch over its own tokens, then averages those averages. That gives a token in micro-batch $k$ the weight $1/(K n_k)$ instead of $1/N$. The worked example: micro-batches with 4, 4 and 2 tokens and mean losses 2.0, 2.0, 5.0 give **2.6** correctly and **3.0** by the shortcut, 15.4% apart.

**Gradients.** Stories were grouped by length into 4 micro-batches per global batch (median tokens per micro-batch: {nk}). Over {len(ag)} real global batches in fp32, against the one-big-batch gradient: token-weighted accumulation differs by a median **{np.median([r['ok_err'] for r in ag]):.1e}** (fp32 summation noise, cosine {np.median([r['ok_cos'] for r in ag]):.7f}); the average of averages by **{np.median([r['bug_err'] for r in ag]):.1e}** (cosine {np.median([r['bug_cos'] for r in ag]):.4f}).

{fig("accum_gradient_error", "gradient error of the two accumulation methods")}

**Training.** Pairs of runs share an initialization and a batch order; one accumulates correctly, one averages averages. Each pair was repeated with {len(rws)} seeds:

{table(trows, ["seed", "buggy run: logged loss vs its true value (median)", "final held-out: correct", "final held-out: buggy", "buggy − correct", "buggy − correct by length quartile (short → long)"])}

The logged number is wrong every step, in every seed. In the model, the bug's fingerprint (better on the shortest stories, worse on the longest, as the $1/(Kn_k)$ weighting predicts) appears in {tilt} of {len(rws)} seeds. Compare the overall held-out difference with its seed-to-seed spread before reading much into it: it is small, and that smallness is how the bug survived.

{fig("accum_bug", "training curves for token-weighted vs average-of-averages accumulation")}

With packed batches (every micro-batch exactly B×T tokens) the two methods are identical to float64 precision, which is how the bug hid.""")

# ---------------------------------------------------------------- 4. grad norm
st, cs, sct = g("stress_seeds"), g("clip_stats"), g("stress_clip_table")
if st and det:
    nat = g("natural_events", [])
    nat_txt = ("In the healthy baseline the same rule flagged " + ", ".join(
        f"the norm at step {e['norm_step']}" + (f" (loss followed at {e['loss_step']})" if e["loss_step"] is not None else " (the loss never followed)")
        for e in nat) + ".") if nat else ("In the healthy baseline run the rule found no departure at all, in either trace: a healthy run has "
                                            "no step where the norm visibly moves first, so the lead is measured on an instability created on purpose.")
    pre_rows = [(sd, step_txt(d["norm_step"]), step_txt(d["loss_step"]),
                 (d["loss_step"] - d["norm_step"]) if d["norm_step"] is not None and d["loss_step"] is not None else "—") for sd, d in det.items()]
    fx_rows = []
    for sd, f_ in fxs.items():
        cells = [sd]
        for z in ("3.0", "4.0", "6.0"):
            v = f_.get(z, {})
            tn, tl = v.get("norm_step"), v.get("loss_step")
            cells.append(f"{step_txt(tn)} / {step_txt(tl)}" + (f" (lead {tl - tn})" if tn is not None and tl is not None else ""))
        fx_rows.append(tuple(cells))
    lds = [v["loss_step"] - v["norm_step"] for f_ in fxs.values() for v in f_.values() if v["norm_step"] is not None and v["loss_step"] is not None]
    bt_rows = []
    if bt:
        bt_rows = [(r["seed"], f"{r['spike']:.1f} (typical {r['typical']:.2f})", f"{r['damage_clip']:+.4f}", f"{r['damage_none']:+.4f}",
                    f"{r['end_clip']:.4f}", f"{r['end_none']:.4f}") for r in bt]
        bt_rows.append(("mean ± sd", "", mean_sd([r["damage_clip"] for r in bt]), mean_sd([r["damage_none"] for r in bt]),
                        mean_sd([r["end_clip"] for r in bt], "{:.4f}"), mean_sd([r["end_none"] for r in bt], "{:.4f}")))
    delays = [int(r[3]) - int(r[2]) for r in (sct or []) if r[2] != "never" and r[3] != "never"]
    first_ = sum(d["norm_step"] is not None and (d["loss_step"] is None or d["norm_step"] < d["loss_step"]) for d in det.values())
    parts.append(f"""## 4. Does the gradient norm move before the loss?

The norm is logged at every step (before clipping). The detection rule was fixed in code before any run was looked at: a trace "moves" at step $t$ when $\\log(\\text{{trace}})$ sits more than 4 robust standard deviations above the median of the previous 50 steps, for 3 steps in a row. {nat_txt}

**Stress test.** Starting from the trained Model A, the learning rate was raised geometrically, with no clipping, until training broke. This was repeated over {len(det)} seeds (batch orders):

{table(pre_rows, ["seed", "pre-registered rule: norm departs at step", "loss departs at step", "lead (steps)"])}

Under the rule fixed in advance, the norm was flagged first in **{first_} of {len(det)}** seeds. That verdict stands as recorded: this detector does not show a lead.

{fig("gradnorm_leads_loss", "learning rate, gradient norm and loss during the stress test")}

That rule compares each step with the 50 before it, so a slow drift keeps raising its own baseline: the loss creeps upward well before the rule fires, if it fires at all. A second view, **added after seeing the first result** (the first verdict stands as recorded), measures both traces against a fixed baseline taken before the ramp. Each cell gives the steps where the norm and the loss depart:

{table(fx_rows, ["seed", "z > 3: norm / loss", "z > 4", "z > 6"])}

{f"Across all seeds and thresholds the norm led by {min(lds)}–{max(lds)} steps (median {np.median(lds):.0f})." if lds else ""} Why: near a minimum $\\mathcal{{L}} \\approx \\mathcal{{L}}^* + \\tfrac12\\lambda x^2$ while $\\lVert g\\rVert = \\lambda|x|$. An instability multiplies the norm from its own small baseline, but the loss changes on top of a large constant $\\mathcal{{L}}^*$. The per-step loss is also measured on different text every step, so it is the noisier trace, and a real change stands out later in it.

**Does clipping rescue a learning rate that is too high?** Every stress run was repeated with clipping at 1.0. The breaking point (the first step after the ramp starts where the loss exceeds 1.5× its pre-ramp median) was defined before any clipped run was looked at:

{table(sct, ["seed", "loss before the ramp", "breaks at step: no clip", "breaks at step: clip at 1.0", "mean loss, last 50 steps: no clip", "clip at 1.0"]) if sct else ""}

{f"Clipping moved the breaking point by {np.mean(delays):+.1f} steps on average (per seed: {delays})" + (", so it does **not** rescue a learning rate that is too high. " if abs(np.mean(delays)) <= 15 else ". ") + "AdamW divides each update by a running estimate of the gradient's size, so scaling every gradient down by the same factor changes its step much less than the clip factor suggests." if delays else ""}

**What clipping is for.** Three batches of random tokens were injected into a healthy run, with and without clipping at 1.0, for each seed:

{table(bt_rows, ["seed", "grad norm on the noise", "worst damage: clip at 1.0", "no clipping", "held-out at end: clip at 1.0", "no clipping"]) if bt_rows else ""}

{f"Clipping reduced the worst damage in {sum(r['damage_clip'] < r['damage_none'] for r in bt)} of {len(bt)} seeds. AdamW already limits how far one batch can move the weights, so the benefit is bounded here; it grows with how big and how frequent the spikes are, which is the case for having it on from step one." if bt else ""}{(f" By the end, though, the unclipped run was slightly *lower* in {sum(r['end_none'] < r['end_clip'] for r in bt)} of {len(bt)} seeds (by {np.mean([r['end_clip'] - r['end_none'] for r in bt]):.4f} on average). One untested explanation: the noise spike inflates AdamW's second-moment estimate, which shrinks the following steps like a brief learning-rate cut." if bt and sum(r['end_none'] < r['end_clip'] for r in bt) > len(bt) / 2 else "")}

{f"Baseline gradient norms after warmup: median {cs['p50']:.3f}, 99th percentile {cs['p99']:.3f}, max {cs['max']:.3f}. A threshold should sit above ordinary steps and below spikes." if cs else ""}""")

# ---------------------------------------------------------------- 5. MFU
mfu, wf, prof = g("mfu"), g("mfu_waterfall"), g("profile")
if mh and mfu:
    pk = g("peak", {})
    wd = mfu["width"]
    w_items = list(wd.items())
    ab = []
    for key, title in (("precision", "matmul precision"), ("attention", "attention kernel"), ("micro_batch", "micro-batch"),
                       ("optimizer", "optimizer kernel"), ("sync", "host sync"), ("loader", "data loading")):
        for n, r in mfu[key].items():
            ab.append((title, n, f"{r['tok_s'] / 1e3:.1f}K", pct(r["mfu"])))
    for n, r in w_items:
        ab.append(("model width (4×512)", n, f"{r['tok_s'] / 1e3:.1f}K", pct(r["mfu"])))
    gs = mfu["gemm_model_shapes"]
    verdict, fixes_txt = [], ""
    if wf:
        cat = prof.get("cat_ms", {}) if prof else {}
        busy = sum(cat.values()) or 1
        top_other = sorted(((k, v) for k, v in cat.items() if k not in ("matmul", "attention")), key=lambda kv: -kv[1])[:3]
        items = {
            "other": (wf["lost_other"], f"**{pct(wf['lost_other'], 0)} of peak goes to kernels that are not matmuls.** The biggest: " +
                      ", ".join(f"{k} ({100 * v / busy:.0f}% of GPU time)" for k, v in top_other) +
                      ". These are the softmax and cross-entropy over a 49,152-word vocabulary, the fp32↔bf16 casts autocast inserts, and the residual adds, norms and RoPE. They move memory and do few FLOPs. A fused linear + cross-entropy kernel and `torch.compile` target exactly this slice."),
            "mm": (wf["lost_mm"], f"**{pct(wf['lost_mm'], 0)} of peak is lost inside the matmuls,** which reach {pct(wf['mm_eff'], 0)} of peak. "
                   f"Width 384 makes them thin: the model's {gs['qkv-sized']['shape']} matmul runs at {gs['qkv-sized']['tflops']:.1f} TFLOP/s vs "
                   f"{mfu['gemm_big_tflops']:.1f} for a 4096³ one, and the width sweep takes MFU from {pct(w_items[0][1]['mfu'])} ({w_items[0][0]}) "
                   f"to {pct(w_items[3][1]['mfu'])} ({w_items[3][0]})."),
            "idle": (wf["lost_idle"], f"**{pct(wf['lost_idle'], 0)} of peak is lost to an idle GPU** between ~{wf['kernels_per_step']:.0f} kernel launches per step. "
                     "At this model size the GPU queue stays full, so this slice is small."),
        }
        for k in sorted(items, key=lambda k: -items[k][0]):
            verdict.append("- " + items[k][1])
        fix_for = {"other": "fuse the non-matmul kernels (a fused linear + cross-entropy kernel that never materializes fp32 logits, fused norm/RoPE, `torch.compile`, which needs triton and was not available on this Windows setup)",
                   "mm": "give the matmuls more work per launch (a wider model, or a bigger micro-batch where memory allows)",
                   "idle": "cut launch and sync overhead (CUDA graphs, no per-step `.item()`)"}
        fixes_txt = "The fixes follow the ranking: " + "; then ".join(fix_for[k] for k in sorted(items, key=lambda k: -items[k][0])) + ". On Hopper or Blackwell, fp8 matmuls would add to all of it."
        if mh.get("mfu_obs_clock"):
            verdict.append(f"- **The clock.** The laptop held {mh['clock']:.0f} MHz rather than its 2,100 MHz boost; at the observed clock MFU is {pct(mh['mfu_obs_clock'])}.")
    parts.append(f"""## 5. MFU, reported honestly

$\\text{{MFU}} = \\dfrac{{(6N + 12LTC)\\times\\text{{tokens/s}}}}{{\\text{{peak FLOP/s}}}}$. Peak for this GPU: {pk.get('source', '')} = {pk.get('flops', 0) / 1e12:.1f} TFLOP/s (dense bf16, fp32 accumulation). It was measured warm (sustained load first, until the clock settled), with every comparison interleaved A-B-B-A so thermal drift cancels, and the SM clock recorded beside every number.

{table([("spec peak at full boost", f"{pk.get('flops', 0) / 1e12:.1f}", pct(mh['mfu_spec'])),
        (f"spec peak at the observed clock ({mh['clock']:.0f} MHz)",
         f"{pk.get('flops', 0) / 1e12 * mh['clock'] / (env.get('clocks', {}).get('sm_max') or mh['clock']):.1f}", pct(mh['mfu_obs_clock'])),
        ("best measured: one 4096³ bf16 matmul", f"{mh['gemm_tflops']:.1f}", pct(mh['mfu_gemm']))],
       ["measured against", "TFLOP/s", "MFU"])}

Model A at micro-batch 8 × 512 trains at **{mh['tok_s'] / 1e3:.1f}K tokens/s = {mh['tflops']:.1f} TFLOP/s** of model arithmetic.

{fig("mfu_waterfall", "where the gap from 100% of peak to the measured MFU goes")}

**What is costing the distance to 40%** (from `torch.profiler`, every GPU kernel classified):

{chr(10).join(verdict)}

{fixes_txt}

<details><summary>All the one-factor-at-a-time measurements</summary>

{table(ab, ["factor", "configuration", "tokens/s", "MFU"])}

{fig("mfu_vs_width", "MFU vs model width")}
</details>""")

# ---------------------------------------------------------------- 6. 0.1
bits = g("bits_0p1")
prec, gm = g("precision_seeds"), g("grad_magnitudes")
if bits:
    rows = [(b[0], b[1], b[3], f"`{b[5]}`", f"`{b[6]}`", b[7], b[8]) for b in bits]
    prec_rows = []
    labels = dict(fp32="fp32", tf32="TF32", bf16="bf16 autocast", fp16="fp16 + loss scaling", fp16_noscale="fp16, no scaling")
    if prec:
        prec_rows = [(labels.get(p, p), f"{r['micro_bs']}×{r['accum']}", mean_sd([s_["val"]["loss"][-1] for s_ in r["seeds"]], "{:.4f}"),
                      ", ".join(f"{s_['val']['loss'][-1]:.4f}" for s_ in r["seeds"]),
                      f"{np.mean([s_['tok_s'] for s_ in r['seeds']]) / 1e3:.1f}K", f"{max(s_['peak_mem_gib'] for s_ in r['seeds']):.2f} GiB")
                     for p, r in prec.items()]
    gm_rows = []
    if gm:
        for kind in ("weights", "activations"):
            r = gm[kind]
            gm_rows.append((kind, f"{r['median']:.1e}", pct(r["fp16_zero_S1"], 3), pct(r["fp16_subnormal_S1"]), pct(r["fp16_zero_S1024"], 4), pct(r["bf16_zero"], 4)))
    parts.append(f"""## 6. The number 0.1 in fp32, bf16 and fp8 E4M3, bit by bit

$0.1 = 1.6 \\times 2^{{-4}}$, and $0.6$ in binary is $0.1001\\,1001\\,1001\\ldots$ (the block 1001 repeats forever). Store the exponent as $-4 + \\text{{bias}}$, keep $M$ mantissa bits, and round to nearest by looking at the bits that were cut:

{table(rows, ["format", "exponent field", "rounding", "sign exponent mantissa", "hex", "stored value", "relative error"])}

{fig("bits_of_0p1", "0.1 in five formats, bit by bit")}

These bit patterns were computed with exact rational arithmetic, then checked against PyTorch's own conversions, which agree bit-for-bit. The same encoder matches PyTorch on 560,000 random values and every rounding tie.

**Which would I train in? bf16 mixed precision**: bf16 matmuls and activations, with fp32 master weights, AdamW state and reductions.

- **Range like fp32.** fp16's 5-bit exponent flushes anything below ~3e-8 to zero. Measured on Model A's real gradients:

{table(gm_rows, ["gradient", "median magnitude", "fp16: becomes 0", "fp16: subnormal (digits lost)", "fp16 ×1024: becomes 0", "bf16: becomes 0"]) if gm_rows else ""}

- **Speed and memory.** bf16 runs on the tensor cores at full rate, with half the bytes per activation. The same model and data in five precisions, three seeds each:

{table(prec_rows, ["precision", "micro-batch × accumulation", "final held-out loss, mean ± sd", "per seed", "tokens/s", "peak memory"]) if prec_rows else ""}

- **Master weights must stay fp32.** Near 1.0, bf16 numbers are 0.0078 apart, so an update of 0.001 rounds away completely: a weight of 1.0 plus a hundred such updates stays exactly 1.0 in bf16 (and reaches 1.1 in fp32).
- **fp8 E4M3** stores 0.1 with a 1.6% error. It needs a per-tensor or per-block scale (unscaled, a gradient-sized tensor is flushed to zero entirely) and hardware with fp8 tensor cores (Hopper or Blackwell, not this Ampere GPU). It is the right choice for the big matmuls there, not for everything.""")

# ---------------------------------------------------------------- memory
mp, mt = g("memory_phases"), g("memory_tricks")
if mp:
    rows = [(r[0], f"{r[1]:,.0f} MiB", f"{r[2]:,.0f} MiB" if r[2] else "—") for r in mp["rows"]]
    tr = []
    if mt:
        for k, v in mt["res"].items():
            mdl, lab = k.split(" | ")
            Bk = mt["B"] if mdl == "A" else mt["Bs"]
            tr.append((("Model A" if mdl == "A" else "SmolLM2-135M") + f", {Bk}×{mt['T']}", lab,
                       f"{v['peak_gib']:.2f} GiB" if v else "out of memory", f"{v['step_ms']:.0f} ms" if v else "—"))
    parts.append(f"""## Also measured: what a step costs in memory

Training holds 16 bytes per weight before any activation: weight, gradient, fp32 master copy and AdamW's two running averages. Measured on Model A's first step against that prediction:

{table(rows, ["phase", "measured", "predicted"])}

{table(tr, ["model, micro-batch", "configuration", "peak memory", "step time"]) if tr else ""}""")

# ---------------------------------------------------------------- run it
secs = g("_seconds", {})
parts.append(f"""## Running it

| `MODE` | What runs | Time |
|---|---|---|
| `learn` | Cheap demos live; every training run replayed from `assets/results.json` (fetched from GitHub if missing) | minutes, on a CPU |
| `quick` | Everything on a tiny model and a 5 MB slice; writes `assets/quick/`, never the results above | ~10 min on a GPU |
| `full` | The runs behind this README | {sum(secs.values()) / 60:.0f} min of experiments on the GPU above, plus downloads; a Colab T4 is estimated (not measured) at 1–1.5 h |

Set `MODE` in the first code cell, or run headless:

```
TL_MODE=full python -m nbconvert --to notebook --execute --inplace {NB} --ExecutePreprocessor.timeout=-1
```

On first run the notebook downloads TinyStories (first 100 MB of the train file, all of the validation file) and SmolLM2-135M (270 MB) into `data/`, resuming interrupted downloads with HTTP range requests.

| File | What it is |
|---|---|
| [`{NB}`]({NB}) | the notebook, executed, outputs included |
| `nb_source.py` | its source as a plain script (runs top to bottom with `python nb_source.py`) |
| `build_notebook.py` | turns `nb_source.py` into the notebook |
| `make_readme.py` | writes this README from `assets/results.json` |
| [`CHEATSHEET.md`](CHEATSHEET.md) | one-page refresher |
| `assets/` | figures and `results.json` from the full run |
""")

(HERE / "README.md").write_text("\n\n".join(p for p in parts if p) + "\n", encoding="utf-8")
print(f"wrote README.md ({sum(len(p) for p in parts):,} characters)")
