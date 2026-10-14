# Optimizers and Learning Rates, Measured

*What `optimizer.step()` does, taken apart and measured: Adam by hand, bias correction, warmup and the
update-to-weight ratio, cosine vs WSD, Muon vs AdamW, and how the best learning rate moves with width.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahulni/Indic_LLM/blob/main/11_Optimizers_LR/optimizers_and_learning_rates.ipynb)

A gradient says which way each weight should move, but not how far. Every optimizer is a rule for the
distance. This notebook checks those rules one at a time:
- the arithmetic, by computing Adam by hand and against PyTorch's internal state
- the training effects, by small runs that change one thing at a time against a tuned baseline

Every comparison is tuned on both sides and paired by seed. Every threshold used to call a result
was written down before the first measurement. A one-page refresher is in
[`CHEATSHEET.md`](CHEATSHEET.md).

**The measured run:** about 5.2 hours of training (311 min) on an NVIDIA GeForce RTX 3070 Laptop GPU.
- bf16, torch 2.11.0+cu128, Python 3.14.5
- 2026-10-11, on AC power, Turbo power plan

The notebook opens in **learn** mode. That mode replays these results in about a minute, and needs
no GPU (see [Running it](#running-it)).

**Contents**

| | |
|---|---|
| results | [Answers at a glance](#answers-at-a-glance) · [Decision rules, fixed before the first measurement](#decision-rules-fixed-before-the-first-measurement) |
| the arithmetic | [§R The rules before Adam](#r-the-rules-before-adam) · [§1 Adam by hand](#1-adam-by-hand) · [§2 Bias correction](#2-bias-correction) |
| training-time choices | [§3 The testbed](#3-the-testbed) · [§4 Bias correction in a real model](#4-bias-correction-in-a-real-model) · [§5 Update-to-weight ratio and warmup](#5-update-to-weight-ratio-and-warmup) · [§6 Cosine vs WSD, stopped at 200](#6-cosine-vs-wsd-stopped-at-200) · [§M Muon vs AdamW, both tuned](#m-muon-vs-adamw-both-tuned) |
| scaling | [§7 Learning rate vs width, and muP](#7-learning-rate-vs-width-and-mup) |
| method | [§8 Tune both sides](#8-tune-both-sides) · [Measured outcomes](#measured-outcomes) · [Limitations](#limitations) · [Running it](#running-it) · [References](#references) |


## Answers at a glance

| question | measured answer | where |
|---|---|---|
| Does Adam by hand match PyTorch? | **Yes.** Step by step, PyTorch's own m, v, m̂, v̂ and step agree with the hand computation to **7 decimal places**. Against a 50-digit reference, float64 is within 3e-15 and float32 within 2e-08. The one outlier is the fused float32 kernel, whose v̂ is off by 3.3e-06 because it rounds 1−β2 in float32; its step is still within 1e-08. | [§1](#1-adam-by-hand) |
| Bias correction off: when does the difference stop mattering? | The first step is **3.16×** too large, and the gap *grows* to 6.57× at step 12 (6.24× at step 20). It is within 1% only after **3,925 steps** at β2 = 0.999, or **76** at β2 = 0.95. In the 31.5M model, the loss gap re-enters seed noise at: β2 0.95, no warmup: never within 300; β2 0.95, warmup 30: step 16; β2 0.999, no warmup: never within 300; β2 0.999, warmup 30: never within 300. | [§2](#2-bias-correction), [§4](#4-bias-correction-in-a-real-model) |
| When does warmup stop changing the update-to-weight ratio? | It changes the ratio from **step 1**, by construction (its learning rate is smaller). Measured against no warmup at all (the rule fixed in advance), its effect **never ends** within the run: skipping warmup lets matrices move by up to 20% of their size in one step (vs 4.3% with 30 warmup steps), and that run never catches up. Among warmed-up runs, the warmup *length* stops mattering at **step 148 / 138** (6 vs 30, 30 vs 100; exploratory). | [§5](#5-update-to-weight-ratio-and-warmup) |
| Cosine vs WSD, both stopped at 200 of 300: which to keep? | **Keep WSD.** As stopped: cosine 2.7544 ± 0.0185, WSD 2.7544 ± 0.0185 (3 seeds). Tuned on its own floor, cosine's best is *no decay at all*, so up to step 200 it is the same run as WSD's flat phase, and the two tie exactly. What separates them is the option to decay late: with ≤20% more compute, WSD's best option (checkpoint 200 + 40-step cooldown) beats cosine's (continued to 240) by **-0.0274 ± 0.0019**. The margin shrank as cosine's floor was tuned (10%: -0.104, 20%: -0.086, 50%: -0.056, 100%: -0.027) and never flipped. | [§6](#6-cosine-vs-wsd-stopped-at-200) |
| Best learning rate at width 4,096, and how confident? | **Use 2.7e-04**, expected within 2.5e-04–2.9e-04. Minima: 256: 5.6e-03, 512: 2.5e-03, 1024: 1.2e-03; best lr ∝ width^-1.08 (inverse-width would be −1). A fit on three widths predicted the held-out width 2048 to within 0.10 octave, which sets the band. Confidence **high** by the rule fixed in advance, for this depth, batch and step count. Under muP the minima move only 0.27 octave across widths (vs 2.19 under SP). | [§7](#7-learning-rate-vs-width-and-mup) |
| Tune both sides: does Muon beat AdamW here? | **Yes, by about 1.43×** (1.37–1.48× across seeds; paired loss difference -0.2255 ± 0.0154), with both sides tuned on the same grid. Against a mis-tuned AdamW the same Muon run looks 1.59× faster. | [§M](#m-muon-vs-adamw-both-tuned), [§8](#8-tune-both-sides) |

## Decision rules, fixed before the first measurement

Stored in `results.json` on the first measured run; every rerun `check`s that they have not changed.

- **When bias correction stops mattering.** The per-step multiplier (1−β₁ᵗ)/√(1−β₂ᵗ) is within p of 1 for every later step. Headline p = 1%; 5% and 10% also reported. In a real model: the 20-step EMA of the paired loss gap stays inside ±2 seed s.d. from that step on.
- **When warmup stops changing the ratio.** For each tensor, the 9-step rolling median of its update/weight ratio stays within p of the no-warmup run's ratio from that step on. Headline p = 10%; 5% and 20% for sensitivity.
- **Which model to keep at step 200.** Both runs are stopped at step 200. For each, take the best finished model reachable with at most 20% more compute (cosine: its step-200 or step-240 model; WSD: its step-200 model or a 40-step cooldown from it). Keep the run whose option has the lower paired-mean validation loss. If they are within 2 paired standard errors, keep WSD, because it can still continue.
- **Where a learning-rate minimum is.** Vertex of a parabola in log₂(lr) through the five √2-spaced points around the argmin of the seed-mean final loss. 90% interval from 2,000 bootstrap resamples over seeds.
- **How confident the width-4,096 answer is.** High: the width-2,048 prediction is off by < 0.5 octave AND the 90% interval at 4,096 is narrower than 1 octave. Medium: off by < 1 octave. Otherwise low.
- **When an optimizer speedup counts.** A speedup is claimed only if the tuned-vs-tuned paired loss difference exceeds 2 standard errors. Speedup = steps the tuned AdamW run needs to reach its final loss, divided by the steps the challenger needs to reach that same loss.

## §R The rules before Adam

The notebook first rebuilds the ideas Adam combines, each checked against a standard worked example:
- gradient descent's stability condition, and why one learning rate cannot serve two curvatures
- momentum, which cancels what alternates and keeps what is consistent
- per-parameter learning rates
- what the optimizer costs in memory (16 bytes per weight with AdamW)
- the √batch rule for Adam
- Muon's orthogonalized update

![gradient descent on two curvatures](assets/gd_two_curvatures.png)

![Newton-Schulz orthogonalization](assets/muon_orthogonalize.png)

## §1 Adam by hand

One weight at w = 1, lr = 1e-3, β = (0.9, 0.999), ε = 1e-8, five gradients. The exact (50-digit)
computation:

| t | g | m | v | m̂ | v̂ | step | w |
|---|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.0500 | 0.000250 | 0.5000 | 0.2500 | -0.001000 | 0.999000 |
| 2 | 0.40 | 0.0850 | 0.000410 | 0.4474 | 0.2050 | -0.000988 | 0.998012 |
| 3 | 0.60 | 0.1365 | 0.000769 | 0.5037 | 0.2567 | -0.000994 | 0.997018 |
| 4 | 0.45 | 0.1678 | 0.000971 | 0.4881 | 0.2431 | -0.000990 | 0.996028 |
| 5 | 0.55 | 0.2061 | 0.001273 | 0.5032 | 0.2550 | -0.000996 | 0.995031 |

**Step by step against PyTorch.** PyTorch's numbers are read from `optimizer.state` after each step
(float32, the default dtype). PyTorch stores m (`exp_avg`) and v (`exp_avg_sq`); m̂ and v̂ are derived
from them. The two sides agree to at least **7 decimal places**:

| t |  | m | v | m̂ | v̂ | step |
|---|---|---|---|---|---|---|
| 1 | by hand | 0.05000000 | 0.00025000 | 0.50000000 | 0.25000000 | -0.00100000 |
|  | PyTorch | 0.05000000 | 0.00025000 | 0.50000001 | 0.25000001 | -0.00099999 |
| 2 | by hand | 0.08500000 | 0.00040975 | 0.44736842 | 0.20497749 | -0.00098813 |
|  | PyTorch | 0.08500000 | 0.00040975 | 0.44736843 | 0.20497751 | -0.00098813 |
| 3 | by hand | 0.13650000 | 0.00076934 | 0.50369004 | 0.25670337 | -0.00099414 |
|  | PyTorch | 0.13650000 | 0.00076934 | 0.50369004 | 0.25670339 | -0.00099415 |
| 4 | by hand | 0.16785000 | 0.00097107 | 0.48807793 | 0.24313218 | -0.00098985 |
|  | PyTorch | 0.16785000 | 0.00097107 | 0.48807794 | 0.24313219 | -0.00098985 |
| 5 | by hand | 0.20606500 | 0.00127260 | 0.50319895 | 0.25502952 | -0.00099642 |
|  | PyTorch | 0.20606500 | 0.00127260 | 0.50319894 | 0.25502953 | -0.00099641 |

**Every implementation against the exact values** (largest absolute difference over the five steps):

| implementation | m | v | m̂ | v̂ | step | w |
|---|---|---|---|---|---|---|
| by hand, float64 | 2.8e-17 | 1.3e-18 | 1.1e-16 | 3.0e-15 | 6.9e-18 | 1.1e-16 |
| torch float64, for-loop | 2.8e-17 | 1.3e-18 | 1.1e-16 | 3.0e-15 | 5.2e-17 | 1.1e-16 |
| torch float64, foreach | 2.8e-17 | 1.3e-18 | 1.1e-16 | 3.0e-15 | 5.2e-17 | 1.1e-16 |
| torch float32, for-loop | 2.6e-09 | 7.2e-11 | 7.7e-09 | 2.2e-08 | 1.4e-08 | 1.3e-08 |
| torch float32, foreach | 2.6e-09 | 7.2e-11 | 7.7e-09 | 2.2e-08 | 1.4e-08 | 1.3e-08 |
| torch float64, fused CUDA | 5.6e-17 | 1.3e-18 | 1.7e-16 | 3.0e-15 | 5.2e-17 | 1.1e-16 |
| torch float32, fused CUDA | 4.4e-08 | 1.6e-08 | 1.2e-07 | 3.3e-06 | 1.4e-08 | 1.3e-08 |

The fused float32 kernel computes 1−β2 in float32, which is off by 1.3e-5 relative. Its own bias
correction uses the same rounded β2, so the step is unaffected. `adam_by_hand.py` runs the same check
on its own and exits 1 on any mismatch.

## §2 Bias correction

Without correction, the step is too large by exactly $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$, whatever the
gradients. $m$ recovers from its zero start in tens of steps, $v$ in thousands, so the gap **widens**
over the first dozen steps before it closes.

![first 20 steps with and without bias correction](assets/bias_correction_first20.png)

Steps until the per-step multiplier stays within 1/5/10% of 1. After about the first 44 steps, only $v$'s correction matters:

| β2 | 1% | 5% | 10% |
|---|---|---|---|
| 0.95 | 76 | 43 | 6 |
| 0.98 | 195 | 118 | 87 |
| 0.99 | 391 | 237 | 175 |
| 0.999 | 3,925 | 2,375 | 1,751 |

![multiplier vs step for several beta2](assets/bias_correction_long.png)

Summed over the run, the overshoot is permanent: about **1,213 η** of extra
displacement at β2 = 0.999, with consistently signed gradients.

## §3 The testbed

All training runs use TinyStories with the SmolLM2 tokenizer:

| | base model | width family |
|---|---|---|
| used in | §4–§6, §M | §7 |
| shape | width 384, 8 layers, GQA 6/2 heads, SwiGLU, RoPE | width 256–2,048, 4 layers, head size 64 |
| vocabulary | 49,152, tied | 8,192 most frequent ids, untied |
| parameters | 31.5M | 7M–234M |
| tokens per step | 16,384 | 16,384 |

**Fair comparisons.**
- **Paired runs.** Runs with the same seed start from the same weights and see the same batches.
- **A verified hand-written optimizer.** The hand-written AdamW used in §4 to switch bias correction
  off matches `torch.optim.AdamW` on all 31.5M weights for 20 steps (largest difference 0).
- **A tuned base learning rate.** The base model's peak learning rate is tuned on a bracketed factor-2
  grid; the best is **1e-03**.

**Noise levels.**
- Seed-to-seed s.d. of validation loss: 0.0235.
- The same seed run twice: 0.00000. This is the GPU noise floor; here the
  runs are deterministic.
- Untrained loss: 10.83, against ln 49,152 = 10.80.

![base model learning-rate tuning](assets/base_lr_tuning.png)

![learning-rate schedules](assets/schedules.png)

## §4 Bias correction in a real model

The 31.5M base model for 300 steps at the tuned learning rate, with bias correction on and off, on the
same seed and data:

| setting | val loss, on | val loss, off | off − on | gap back inside seed noise from step |
|---|---|---|---|---|
| β2 0.95, no warmup | 3.5534 | 2.9547 | -0.5987 | never within 300 |
| β2 0.95, warmup 30 | 2.6041 | 2.6313 | +0.0272 | 16 |
| β2 0.999, no warmup | 3.5265 | 4.2360 | +0.7095 | never within 300 |
| β2 0.999, warmup 30 | 2.6179 | 4.5932 | +1.9753 | never within 300 |

- **β2 = 0.999.** Switching correction off is costly and never recovers within the run: the
  multiplier is still about 2× at step 300, and warmup does not hide it.
- **β2 = 0.95.** The uncorrected early steps are *smaller* (0.45η at step 1). They act as a built-in
  warmup, so without warmup, switching correction off *helps*.

![loss with and without bias correction in the base model](assets/bias_correction_model.png)

From the same optimizer state, the uncorrected/corrected update ratio of the median tensor follows
§2's closed form to within 3.7%:

![counterfactual update ratio vs closed form](assets/bias_correction_counterfactual.png)

## §5 Update-to-weight ratio and warmup

Each matrix's $\lVert\Delta W\rVert/\lVert W\rVert$ is logged on every step. Runs warmed up over W steps
are compared with one that is not, at the same constant peak learning rate.

Warmup changes the ratio from step 1, because its learning rate is smaller by construction, so the
informative question is where that influence **ends**. It counts as ended at the first step after
which the 9-step rolling median stays within the threshold of the no-warmup run's (median over
matrices):

| warmup W | 5% | 10% (headline) | 20% | ratio settles by | range over matrices (10%) |
|---|---|---|---|---|---|
| 6 | never within 300 | never within 300 | never within 300 | 168 | 100–never within 300 |
| 30 | never within 300 | never within 300 | never within 300 | 159 | 110–never within 300 |
| 100 | never within 300 | never within 300 | never within 300 | 160 | 198–never within 300 |

**Two findings.**
- **Against no warmup, the effect never ends.** Skipping warmup leaves the run permanently worse
  (val 3.1002 vs 2.4540), and its ratios stay lower.
- **Among warmed-up runs, the length stops mattering at about step
  148 / 138** (6 vs 30, 30 vs 100; exploratory).

![update-to-weight ratio heatmap and medians](assets/update_ratio_warmup.png)

**What drives the ratio.**
- **Adam's step as a fraction of η** falls from 1.00 at step 1 to
  0.20 once gradients decorrelate.
- **The decay part.** Decoupled decay alone contributes ηλ = 1.0e-04 per step,
  against a late median ratio of 8.9e-03.
- **Spectral vs Frobenius.** The spectral ratio is
  1.6× for the first query matrix, 2.0× for the a middle up-projection, 1.7× for the last down-projection, 1.0× for the embedding the Frobenius ratio, so updates concentrate in a
  few directions.
- **The fp32 master copy.** 19% of updates are smaller than half a bf16 spacing
  at the peak learning rate, and 80% at a 10× smaller one. Without the
  copy, those updates would be rounded away.

![adam fraction, spectral ratio, bf16 rounding](assets/update_ratio_anatomy.png)

Embedding rows (tied head), median ratio by token frequency:

| rows | median update/weight |
|---|---|
| 100 most frequent | 5.0e-03 |
| rank 100–1k | 7.8e-03 |
| rank 1k–8k | 6.1e-03 |
| rank ≥ 8k (rare/unseen) | 5.5e-03 |
| in this batch | 8.6e-03 |
| not in this batch | 5.5e-03 |

A RAdam run with **no** warmup reaches 2.5769, against AdamW at warmup
0: 3.1002, 6: 2.5163, 30: 2.4540, 100: 2.4345 (one seed each).

## §6 Cosine vs WSD, stopped at 200

The same model, planned for 300 steps under each schedule, is stopped at step 200.

**Tuning.** Both schedules were tuned by the same rule: a learning-rate grid, then the schedule's own
knob, each extended at an edge until bracketed.
- **Cosine** had 15 trials. Its floor kept improving all the way to 100%, which
  means no decay at all. Validation loss at 300 by floor:
  0%: 2.643, 10%: 2.580, 20%: 2.544, 30%: 2.517, 40%: 2.503, 50%: 2.489, 60%: 2.475, 70%: 2.463, 80%: 2.458, 90%: 2.457, 100%: 2.454.
- **WSD** had 9 trials. Its best decay is the last 10%, linear.
- **Both** ended at a peak learning rate of 1e-03.

So in this short-run regime the tuned cosine arm is a constant learning rate, the same run as WSD's
flat phase. The rows below that coincide do so because they are the same model.

Three paired seeds; compute is in steps:

| model | compute | val loss | s.d. |
|---|---|---|---|
| cosine-300, stopped at 200 | 200 | 2.7544 | 0.0185 |
| WSD-300, stopped at 200 | 200 | 2.7544 | 0.0185 |
| cosine planned for 200 | 200 | 2.7544 | 0.0185 |
| WSD checkpoint 160 + 40-step cooldown | 200 | 2.7431 | 0.0234 |
| WSD flat phase, weight EMA 0.9 (constant lr + EMA) | 200 | 2.7169 | 0.0217 |
| cosine-300, continued to 240 | 240 | 2.6175 | 0.0136 |
| WSD checkpoint 200 + 40-step cooldown | 240 | 2.5901 | 0.0163 |
| cosine planned for 240 | 240 | 2.6175 | 0.0136 |
| cosine-300, completed | 300 | 2.4641 | 0.0091 |
| WSD-300, completed | 300 | 2.4183 | 0.0110 |

![cosine vs WSD loss and schedules](assets/cosine_vs_wsd.png)

![every finished model with its compute](assets/cosine_vs_wsd_arms.png)

**The state of each model at step 200:**

|  | val loss | train loss | lr now | grad norm | clipped | update/weight |
|---|---|---|---|---|---|---|
| cosine-300 at step 200 | 2.7544 | 2.8001 | 1.0e-03 | 0.726 | 3% | 8.9e-03 |
| WSD-300 at step 200 | 2.7544 | 2.8001 | 1.0e-03 | 0.726 | 3% | 8.9e-03 |

**Decision (rule fixed in advance): keep the WSD run.**
- Best option within +20% compute: cosine → cosine-300 continued to 240; WSD → WSD 200 + 40-step cooldown.
- Paired difference WSD − cosine: -0.0274 ± 0.0019. That is the SE over seeds; the
  eval-sampling SE is 0.0007.
- As stopped, with no cooldown: WSD − cosine = +0.0000.

**How the decision moved as cosine's floor was tuned** (3 paired seeds each; the first pass stopped
at 20%):

| cosine floor | WSD − cosine | SE | keep |
|---|---|---|---|
| 10% | -0.1039 | 0.0050 | WSD |
| 20% | -0.0862 | 0.0039 | WSD |
| 50% | -0.0563 | 0.0018 | WSD |
| 100% | -0.0274 | 0.0019 | WSD |

**Samples at step 200** (seed 0):
- cosine: *Once upon a time, there was a little boy named Tim. Tim had a favorite toy. It loved to run and play outside. One day, Tim and his friend, Sam, came to play. It was a big mess! Tim told Sam, "*
- WSD: *Once upon a time, there was a little boy named Tim. Tim had a favorite toy. It loved to run and play outside. One day, Tim and his friend, Sam, came to play. It was a big mess! Tim told Sam, "*
- WSD, checkpoint 160 + cooldown: *Once upon a time, there was a little boy named Tim. Tim was a very special boy. He had many friends who loved to play with. One day, Tim found a small toy that was very small. He wanted to see what made. Tim*

## §M Muon vs AdamW, both tuned

**Setup.**
- Muon (momentum 0.95, Nesterov, `match_rms_adamw`) updates the 40 hidden matrices.
- AdamW updates the embedding and the norm gains.
- Each optimizer's learning rate was tuned on the same grid with the same number of points (AdamW
  5, Muon 5), and both ran on the same 3 seeds.

**Tuned vs tuned.**
- Final loss: Muon 2.3809, AdamW 2.6064.
- Paired difference: **-0.2255 ± 0.0154**. A speedup is claimed.
- Speedup to AdamW's final loss: 1.37×, 1.43×, 1.48×.

The same tuned Muon run against differently tuned AdamW baselines:

| baseline | loss difference | apparent speedup |
|---|---|---|
| AdamW at 5e-04 (half its best) | -0.3436 | 1.59× |
| AdamW at 2e-03 (double its best) | -0.2618 | 1.48× |
| AdamW at its tuned lr | -0.1989 | 1.37× |

![muon vs adamw](assets/muon_vs_adamw.png)

## §7 Learning rate vs width, and muP

**Protocol.**
- **Model:** the width family (4 layers, head size 64, 8,192-id vocabulary, untied).
- **Held fixed:** everything except width, i.e. a 16,384-token batch, 300 steps, warmup + cosine.
- **Minimum:** the vertex of a parabola in log₂(lr) through five √2-spaced points, 3 seeds each, with
  a 90% bootstrap interval.

| width | parameters | best lr | 90% interval | loss there | cost of a 2× error |
|---|---|---|---|---|---|
| 256 | 7.2M | 5.63e-03 | 5.54e-03 – 5.73e-03 | 2.4944 | +0.0810 |
| 512 | 20.8M | 2.53e-03 | 2.48e-03 – 2.59e-03 | 2.3651 | +0.0954 |
| 1024 | 66.6M | 1.20e-03 | 1.19e-03 – 1.21e-03 | 2.2527 | +0.1079 |

![loss vs learning rate at each width, and best lr vs width](assets/lr_vs_width.png)

**Fit, then test on a held-out width.**
- Fit on three widths: log₂(best lr) = -1.11·log₂(width) + c, slope 90% interval
  -1.13 to -1.10.
- Predicted for width 2048: 5.51e-04. Measured: 5.92e-04, off by
  **+0.10 octave**.
- Refit on four widths: slope -1.08.

**Width 4096: use 2.7e-04.**
- **Expected range: 2.5e-04–2.9e-04.** This is the error
  the same fit made one octave out.
- **The bootstrap interval (2.7e-04–2.8e-04) sees seed noise only**, so it understates the
  real uncertainty.
- **Confidence: high**, by the rule fixed in advance.
- **Scope.** It holds for this depth, batch, step count and the standard parameterization. A
  different batch needs the √batch adjustment.
- **For comparison**, scaling width 256's best by 1/width gives 3.5e-04.

![attention logit and output logit growth vs learning rate](assets/lr_cliff_diagnostics.png)

**Does one learning rate suit every layer?** Update/weight ratio at each width's best learning rate,
over the second half of the run:

| width | embedding | hidden matrices | output head |
|---|---|---|---|
| 256 | 5.9e-04 | 9.6e-03 | 6.4e-03 |
| 512 | 2.8e-04 | 7.8e-03 | 5.2e-03 |
| 1024 | 1.4e-04 | 5.9e-03 | 4.1e-03 |

**What muP changes.**
- **Coordinate check.** After 10 steps at one learning rate, the logit RMS grows
  2.0× from width 256 to 2048 under SP, against
  1.1× under muP.
- **Sweep.** The best learning rate spreads 2.19 octaves across widths under SP, and
  0.27 under muP.
- **Transfer.** Carrying width 256's best learning rate (5.7e-03) unchanged to width
  2048 gives 3.3320 under SP and 2.2280 under muP. SP tuned at
  2048 reaches 2.1804.

![muP coordinate check](assets/mup_coord_check.png)

| width | muP best lr | loss |
|---|---|---|
| 256 | 5.39e-03 | 2.4951 |
| 512 | 4.77e-03 | 2.3698 |
| 1024 | 4.45e-03 | 2.2922 |

![muP sweep](assets/mup_sweep.png)

## §8 Tune both sides

The same tuned challenger against baselines run away from their own best, from grids already
measured (seed 0). In the learning-rate grids, both schedules use their default knobs.

**Mis-tuning the baseline's learning rate:**

| comparison | apparent advantage when fair | worst mis-tuned baseline | octaves off |
|---|---|---|---|
| WSD vs cosine (default knobs) | +0.1704 | +1.5399 | +3 |
| Muon vs AdamW | +0.1989 | +1.5684 | +3 |
| cosine vs WSD (default knobs) | -0.1704 | +1.3040 | +3 |

**Mis-tuning the baseline's own knob.**
- Against cosine at its tuned floor (100%), tuned WSD is ahead by 0.0483 at step 300.
- Against cosine at the usual 10% default floor, the same WSD run looks 0.1739 ahead.
- This notebook's first pass made exactly that mistake (§6).

![apparent advantage vs baseline tuning](assets/tune_both_sides.png)

**Tuning budget per side:**

| comparison | trials | tuned | held fixed |
|---|---|---|---|
| cosine vs WSD | 15 vs 9 trials | lr grid + own knob, edge-extended (floor / decay length) | warmup, β, weight decay, batch, data, seeds |
| Muon vs AdamW | 5 vs 5 lr points, 3 seeds each | Muon's lr; AdamW's lr | momentum 0.95 / β (0.9, 0.95), decay 0.1, schedule |
| SP vs muP (width) | 30 vs 24 runs | lr grid per width | depth, batch, steps, schedule |

## Measured outcomes

The notebook records each expectation with `observe(...)`. It is measured, not guaranteed, and a failed one is reported as failed. 20 of 24 held.

- ✅ the fused kernel's v is off by 1.29e-05 relative, the same size as float32's 1 - 0.999 error
- ✅ re-running the same seed moves the validation loss less than changing the seed does
- ✅ β2=0.999 without warmup: switching bias correction off costs +0.709 in final loss (seed s.d. 0.024)
- ❌ expected warmup to shrink that cost (β2=0.999); measured +0.709 without warmup and +1.975 with it
- ❌ expected switching bias correction off to be harmless at β2=0.95 (within 2 seed s.d. = 0.047); measured -0.599 without warmup and +0.027 with it
- ✅ in the real model the uncorrected/corrected update ratio follows §2's closed form (median tensor within 5%)
- ❌ expected warmup's effect on the update/weight ratio (against no warmup) to end within the 300-step run, for every warmup length
- ✅ warmup over 30 steps lowers the largest ratio from 2.00e-01 to 4.33e-02
- ✅ for every warmup length, the ratio stops being affected only after the warmup itself has ended
- ✅ Adam's step falls from 1.00 η at step 1 to 0.20 η once gradients decorrelate
- ✅ at a decayed learning rate most updates are smaller than half a bf16 spacing: without an fp32 copy they would vanish
- ✅ with a tied head, rare-token rows still move every step (the output softmax gives every row a gradient)
- ❌ expected the cosine model to have the lower loss as stopped at step 200; measured WSD − cosine = +0.0000 (the same run: tuned cosine never decays)
- ✅ a 40-step cooldown from WSD's step-200 checkpoint vs cosine-300 at the same compute: -0.0274 ± 0.0019
- ✅ a 40-step cooldown from WSD's step-200 checkpoint vs a cosine run planned for 240 from the start: -0.0274 ± 0.0019 (within 2 SE or better)
- ✅ averaging WSD's flat-phase weights (EMA 0.9) vs a cosine run planned for 200: -0.0375 ± 0.0021 (within 2 SE or better)
- ✅ a mis-tuned AdamW baseline inflates Muon's apparent speedup from 1.37x to 1.59x
- ✅ the best achievable loss falls as width grows (wider is better at its own best learning rate)
- ✅ width 2048: the best learning rate is bracketed by the points measured around the prediction
- ✅ muP keeps the output logits flatter across width than SP does
- ✅ muP narrows the spread of best learning rates (2.19 → 0.27 octaves)
- ✅ carrying width 256's learning rate to width 2048 works better under muP than under SP
- ✅ the cosine floor is bracketed: its best value (1) is inside the values tried (0–1) or at the end of its feasible range
- ✅ the WSD decay length is bracketed: its best value (0.1) is inside the values tried (0.05–0.3)

## Limitations

- **Horizon.** 300 steps × 16,384 tokens is about 0.16 tokens per parameter for the base model, which
  is very early training. Learning-rate optima and schedule rankings shift with run length, and longer
  horizons were not tested.
- **Seeds.** Three paired seeds give 2 degrees of freedom, so differences within 2 standard errors are
  called ties.
- **Data.** There is one dataset (TinyStories).
- **Held fixed in the width sweep:** depth, batch, warmup and β.
- **muP-lite.** The muP used here rescales hidden-layer learning rates and the output multiplier only.
- **Weight decay.** ηλT ≈ 0.06, so decay barely acts within these runs.
- **Validation reuse.** The schedule knobs and the EMA decay were chosen on the validation set used
  for the comparison.
- **Tuned cosine is constant LR.** Cosine's best floor was 100%, so the comparison WSD wins is "decay
  at the end" against "no decay". A genuine cosine decay at any floor loses by more.
- **Sequential tuning.** Learning rate and knob were tuned one after the other, not jointly.
- **Width 2,048 used one seed.** The width-4,096 band therefore uses that held-out error rather than
  the seed-only bootstrap.
- **Timing.** One run's wall time includes the machine sleeping, so it is excluded from the timing
  totals. Its result is unaffected.

## Running it

| `MODE` | what runs | time |
|---|---|---|
| `learn` (default) | the arithmetic sections live; every training result is replayed from `assets/results.json` | ~1 min, plus a one-time ~10 MB download if the file is missing |
| `quick` | tiny models through every code path; writes `assets/quick/` | ~10 min on a GPU |
| `full` | every measured run, resumable | ~5 h on the RTX 3070 Laptop above; a Colab T4 is roughly 2–3× slower (estimate) |

```bash
pip install -r requirements.txt
python adam_by_hand.py                                     # the hand-vs-PyTorch check on its own
python build_notebook.py                                   # nb_parts/*.py -> optimizers_and_learning_rates.ipynb
OPT_MODE=full python -m nbconvert --to notebook --execute --inplace optimizers_and_learning_rates.ipynb --ExecutePreprocessor.timeout=-1
python make_readme.py                                      # this file, from assets/results.json
```

- **Resumable.** Each experiment is saved to `results.json` as soon as it finishes, so an interrupted
  full run resumes where it stopped.
- **T4 rehearsal.** `OPT_PRECISION=fp16` rehearses the T4 path (fp16 + GradScaler) on any GPU.

| file | what it is |
|---|---|
| `optimizers_and_learning_rates.ipynb` | the notebook, executed in full mode |
| `nb_parts/p00_setup.py` … `p12_answers.py` | the notebook's source, one file per section |
| `adam_by_hand.py` | a standalone Adam-by-hand vs PyTorch check |
| `build_notebook.py`, `make_readme.py` | build the notebook and this README |
| `assets/results.json`, `assets/*.png` | every measured number, and the figures |
| `CHEATSHEET.md` | a one-page refresher |
| `requirements.txt` | dependencies for a local run (Colab has them) |

## References

| topic | reference |
|---|---|
| Adam | Kingma & Ba (2015), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| AdamW | Loshchilov & Hutter (2019), [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |
| RAdam | Liu et al. (2020), [arXiv:1908.03265](https://arxiv.org/abs/1908.03265) |
| warmup and the update/weight ratio | Kosson et al. (2024), [arXiv:2410.23922](https://arxiv.org/abs/2410.23922) |
| small-scale proxies for instabilities | Wortsman et al. (2023), [arXiv:2309.14322](https://arxiv.org/abs/2309.14322) |
| μTransfer | Yang et al. (2022), [arXiv:2203.03466](https://arxiv.org/abs/2203.03466) |
| spectral condition for feature learning | Yang, Simon & Bernstein (2023), [arXiv:2310.17813](https://arxiv.org/abs/2310.17813) |
| scaling exponents across parameterizations | Everett et al. (2024), [arXiv:2407.05872](https://arxiv.org/abs/2407.05872) |
| cooldowns and WSD | Hägele et al. (2024), [arXiv:2405.18392](https://arxiv.org/abs/2405.18392); Hu et al. (2024), [arXiv:2404.06395](https://arxiv.org/abs/2404.06395); Wen et al. (2024), [arXiv:2410.05192](https://arxiv.org/abs/2410.05192) |
| AdamW decay as an EMA | Wang & Aitchison (2024), [arXiv:2405.13698](https://arxiv.org/abs/2405.13698) |
| √batch scaling for Adam | Malladi et al. (2022), [arXiv:2205.10287](https://arxiv.org/abs/2205.10287) |
| critical batch size | McCandlish et al. (2018), [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) |
| Muon | Jordan et al. (2024), [blog](https://kellerjordan.github.io/posts/muon/); Liu et al. (2025), [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) |
| MuonClip | Kimi Team (2025), [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) |
| QK-norm | Dehghani et al. (2023), [arXiv:2302.05442](https://arxiv.org/abs/2302.05442) |
| data and tokenizer | TinyStories, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759); SmolLM2, [arXiv:2502.02737](https://arxiv.org/abs/2502.02737) |
