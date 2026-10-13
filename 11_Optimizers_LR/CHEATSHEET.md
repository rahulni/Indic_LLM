# Optimizers and learning rates: one-page refresher

Section numbers (§) refer to `optimizers_and_learning_rates.ipynb`. Numbers marked *measured* come
from `assets/results.json`; the README has the full tables.

## The update rules (§R, §1)

| rule | update | state per weight |
|---|---|---|
| gradient descent | $w \leftarrow w - \eta g$ | none |
| momentum | $m \leftarrow \beta_1 m + (1-\beta_1)g$, then step along $m$ | $m$ |
| per-parameter scale | $v \leftarrow \beta_2 v + (1-\beta_2)g^2$, step $\eta g/\sqrt v$ | $v$ |
| Adam | $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$, step $-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ | $m, v$ |
| AdamW | Adam step, then $-\eta\lambda w$ outside the ratio | $m, v$ |
| Muon (2-D hidden matrices only) | momentum matrix → Newton–Schulz → singular values ≈ 1 | $m$ |

- **Stability of plain gradient descent.** Each step multiplies the distance to the minimum by
  $1-\eta c$ (where $c$ is the curvature), so it needs $|1-\eta c|<1$. One $\eta$ cannot suit two
  curvatures.
- **Memory.** About $1/(1-\beta)$ steps: $\beta_1 = 0.9$ remembers about 10 steps, $\beta_2 = 0.999$
  about 1,000.
- **Adam's step.** It is about $\eta$ when gradients agree, and at most
  $\eta\max(1, (1-\beta_1)/\sqrt{1-\beta_2})$, which is $3.16\eta$ at $\beta_2=0.999$. The maximum is
  hit on **rare tokens**: a long run of zero gradients followed by one non-zero gradient.
- **Weight decay.** L2-in-Adam divides the decay by $\sqrt{\hat v}$; with no loss gradient it shrinks
  every weight by $\eta$ regardless of size. AdamW shrinks each weight by $\eta\lambda w$.
- **Memory cost.** 16 bytes per weight in mixed precision: bf16 weight 2 + bf16 grad 2 + fp32 master
  copy 4 + $m$ 4 + $v$ 4. Half of that is optimizer state.

## PyTorch facts worth knowing (§1, §3)

- `optimizer.state[p]` holds `exp_avg` ($m$), `exp_avg_sq` ($v$) and `step`. There is no $\hat m$ or
  $\hat v$: both corrections are folded into `lr/bc1` and `sqrt(v)/sqrt(bc2) + eps`.
- The fused CUDA kernel computes $1-\beta_2$ in float32, so its $v$ is off by 1.3e-5 relative. Its
  step is not affected.
- `Adam(weight_decay=λ)` is L2. `AdamW(weight_decay=λ)` is decoupled.
- Bias correction cannot be switched off in either; that needs a hand-written AdamW (§3).
- `torch.optim.Muon` takes 2-D parameters only, and `adjust_lr_fn="match_rms_adamw"` puts its learning
  rate on AdamW's scale.

## Bias correction (§2, §4)

- Uncorrected step / corrected step $= (1-\beta_1^t)/\sqrt{1-\beta_2^t}$, exactly, for any gradient
  sequence.
- At $\beta_2 = 0.999$: 3.16× at step 1, a peak of **6.57× at step 12**, still 6.24× at step 20. The
  gap *widens* first.
- It is within 1% of 1 after **3,925 steps** at $\beta_2=0.999$, or **76** at $\beta_2=0.95$. Past
  ~44 steps only $v$'s correction matters.
- The overshoot integrates to a permanent offset of about 1,200 η of extra displacement
  ($\beta_2=0.999$, consistent gradients).
- *Measured.* At $\beta_2 = 0.999$, warmup does **not** hide a missing correction. The multiplier is
  still ~2× at step 300, and the final loss was +0.7 to +2.0 worse in a 300-step run.
- *Measured.* At $\beta_2 = 0.95$ the uncorrected early steps are *smaller* (0.45η at step 1), a
  built-in warmup. Without warmup, switching correction off *helped*; with warmup, the two runs were
  indistinguishable from step 16.

## Update-to-weight ratio and warmup (§5)

- Log $\lVert\Delta W\rVert/\lVert W\rVert$ for every matrix from step one. Healthy is around
  $10^{-3}$.
- What sets it: $\eta$ × Adam's fraction $\rho$ (≈1 when gradients agree, ≈0.2–0.3 once they are
  noise) × $\sqrt{n}/\lVert W\rVert$, plus a decay floor of $\eta\lambda$.
- Warmup exists because $\rho \approx 1$ at initialization, exactly when weights are small relative to
  their updates.
- Warmup's effect on the ratio starts at step 1 by construction. The useful number is when it
  **ends**.
- *Measured.* Without warmup the first steps moved matrices by up to **20%** of their size, and the
  run never caught up: val 3.10 vs 2.45, and its ratios stayed lower all run. So compared with no
  warmup, warmup's mark never fades within 300 steps.
- *Measured.* Among warmed-up runs, the length (6 / 30 / 100 steps) stopped mattering around **step
  140–150**. Adam's step fell from 1.0η to about **0.2η**. RAdam with no warmup recovered most, but
  not all, of the damage.
- *Measured.* Keep an fp32 master copy: **19%** of updates are below half a bf16 spacing at the peak
  learning rate, and **80%** at a 10× smaller one.

## Schedules (§6)

- **Cosine** fixes the run length before step 1. Stopped early, it is stranded mid-decay.
- **WSD** holds the peak and decays only at the end. A flat-phase checkpoint plus a short cooldown is a
  finished model, and the same checkpoint can also keep training.
- Compare schedules by what you can do when the run is cut short, compute-matched and tuned on both
  sides.
- *Measured* (planned for 300, stopped at 200, 3 paired seeds): **keep WSD**.
  - As stopped, the two tie exactly (2.754), because tuned cosine never decays and so matches WSD's
    flat phase.
  - 2.590 after a 40-step cooldown, vs 2.618 for continuing at the same compute.
  - Weight EMA in the flat phase was the best model at 200 steps of compute.
  - In this early-training regime, every decay that starts early hurts. Tuned on its own floor,
    cosine's best was **100%**, no decay at all, so "tuned cosine" here is constant LR.
- *Measured.* Tuning cosine's floor properly cut WSD's apparent margin about fourfold (−0.104 at the
  usual 10% floor, −0.027 at 100%), and the verdict never flipped. Tune the baseline's own knobs, not
  just its learning rate.

## Batch and width (§R.5, §7)

- **Batch.** Global batch = micro × GPUs × accumulation. For SGD the learning rate scales linearly
  with batch; for Adam, with √batch. Both stop working past the critical batch size.
- **Width under SP.** The best learning rate falls with width as width^b. *Measured* (4 layers,
  16k-token batch, 300 steps): b ≈ **−1.1**, with minima 5.6e-3 / 2.5e-3 / 1.2e-3 at 256 / 512 /
  1,024. Width 2,048 was predicted within **0.1 octave**. **Width 4,096: use ≈ 2.7e-4** (2.5–2.9e-4).
  Extrapolate only after testing on a held-out width.
- **muP.** It scales hidden-layer learning rates by 1/width and multiplies the logits by 1/width.
  The minimum then stays put, and a small sweep transfers. Check the implementation first with a
  coordinate check.
- *Measured.* The minima spread 0.27 octaves under muP vs 2.19 under SP. Width 256's learning rate
  used at width 2,048 cost +1.15 nats under SP and +0.05 under muP.

## Before believing "X beats Y" (§M, §8)

*Measured.* Tuned Muon vs tuned AdamW gave **1.43×** (paired, 3 seeds). Against AdamW at half its
best learning rate, the same Muon run looked 1.59× faster.

1. Did both sides get the same number of tuning trials, on the same grid?
2. Is each side's best learning rate bracketed, not at a grid edge?
3. Are the runs paired (same seeds, same data order)?
4. Is the difference bigger than 2 paired standard errors, and bigger than the same-seed noise floor?
5. Is the comparison compute-matched?
6. Were the thresholds fixed before looking?
