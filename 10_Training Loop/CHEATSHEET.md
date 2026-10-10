# The training loop on one page

A refresher for [`inside_the_training_loop.ipynb`](inside_the_training_loop.ipynb). Section numbers point into the notebook.

## One step

```python
for micro in global_batch:                         # K micro-batches = what fits; the global batch is a choice
    loss = model(x, y, reduction="sum") / N        # N = real (non-padding) tokens in the WHOLE global batch
    loss.backward()                                # adds into .grad
norm = clip_grad_norm_(params, 1.0)                # log it every step: it moves before the loss does
optimizer.step()
optimizer.zero_grad(set_to_none=True)              # .grad accumulates; forgetting this is silent
```

- step = K forwards + K backwards + **one** update. Global batch = micro-batch × K × GPUs.
- Loss at step 0 must be ≈ ln V (ln 49152 = 10.80). If it is not, stop.

## Gradients (§2–§4, §8)

| Idea | Formula |
|---|---|
| a gradient is the answer to a nudge | $\dfrac{\mathcal{L}(w+\varepsilon)-\mathcal{L}(w-\varepsilon)}{2\varepsilon}$, error $O(\varepsilon^2)$ + roundoff $O(u/\varepsilon)$ |
| backprop = chain rule, one link at a time | $x=2, w_1=3, w_2=4, t=20$: $\;8 \to 48 \to 32 \to 64$ |
| a linear layer $y=Wx$ backwards | $\bar x = W^\top\bar y$, $\;\bar W = \bar y\,x^\top$, so backward ≈ 2× forward |
| check a gradient | float64, `model.eval()` (no dropout), fixed batch, ε ≈ 1e-5. Expect 7+ matching digits |

## Accumulation and the average-of-averages bug (§9–§10)

- Correct: $\mathcal{L} = \frac{1}{N}\sum_k\sum_i \ell_{k,i} = \sum_k \frac{n_k}{N}\bar{\mathcal{L}}_k$.
- Bug: $\frac{1}{K}\sum_k \bar{\mathcal{L}}_k$ weights a token by $\frac{1}{K n_k}$. Short micro-batches get extra votes. Example: (4 tokens, 2.0), (4, 2.0), (2, 5.0) give **2.6** correctly, **3.0** by the bug, 15.4% apart.
- Invisible whenever all $n_k$ are equal (packed batches). That is how it survived until 2024.

## Gradient norm and clipping (§11)

- $\lVert g\rVert = \sqrt{\sum g^2}$; clip: $g \leftarrow g\cdot\min(1, c/\lVert g\rVert)$ (8.4 → ×0.119). The direction is unchanged.
- The norm should lead the loss: $\mathcal{L}\approx\mathcal{L}^*+\tfrac12\lambda x^2$ hides a change behind $\mathcal{L}^*$, while $\lVert g\rVert=\lambda|x|$ shows it. Measured here: against a fixed pre-trouble baseline it led by a few dozen steps in every seed, but a rolling-window detector missed it. Compare against a reference you set *before* trouble starts.
- Clipping limits damage from abnormal batches. It does not rescue a learning rate that is too high. Measured: the breaking point barely moved, because AdamW normalizes the step anyway.
- Choose the threshold from the measured norm distribution, not habit.

## Floats (§12–§17)

$x = (-1)^s \times 2^{e-b}\times(1+m/2^M)$, bias $b = 2^{E-1}-1$, ε $=2^{-M}$, digits $=(M+1)\log_{10}2$.

| Format | E | M | Largest | Smallest normal | Digits | 0.1 is stored as |
|---|---|---|---|---|---|---|
| fp32 | 8 | 23 | 3.4e38 | 1.2e-38 | 7.2 | `0 01111011 10011001100110011001101` = 0x3DCCCCCD → 0.1000000015 |
| fp16 | 5 | 10 | 65504 | 6.1e-5 | 3.3 | `0 01011 1001100110` = 0x2E66 → 0.0999756 |
| bf16 | 8 | 7 | 3.4e38 | 1.2e-38 | 2.4 | `0 01111011 1001101` = 0x3DCD → 0.1000977 |
| fp8 E4M3 | 4 | 3 | 448 | 0.0156 | 1.2 | `0 0011 101` = 0x1D → 0.1015625 |
| fp8 E5M2 | 5 | 2 | 57344 | 6.1e-5 | 0.9 | `0 01011 10` = 0x2E → 0.09375 |
| fp4 E2M1 | 2 | 1 | 6 | 1 | 0.6 | (needs a block scale) |

How 0.1 was derived: $0.1 = 1.6\times2^{-4}$. 0.6 in binary is `0.1001 1001 1001…`. The exponent field is $-4+b$. Keep $M$ bits, then round up if the first cut bit is 1 and anything after it is nonzero.

- **Exponent bits buy range; mantissa bits buy detail.** Look at range first: running out turns numbers into 0 or ∞.
- **bf16 beat fp16** because gradients live at 1e-8 to 1e-3, and fp16 flushes anything below ~3e-8 to zero. fp16 needs loss scaling (×S, then ÷S); bf16 does not.
- **Master weights stay fp32.** Near 1.0, bf16 values are 0.0078 apart, so an update of 0.001 rounds away.
- **fp8/fp4 need scales.** Use one per tensor (fp8) or per block: MXFP8 = 8.25 bits per value, NVFP4 = (16×4+8)/16 = 4.5 bits per value. An outlier inflates its block's scale.
- **Train in**: bf16 autocast + fp32 master weights and AdamW state. On a T4: fp16 + GradScaler. On Hopper/Blackwell: fp8 matmuls with scales.

## Cost (§18–§21)

- Memory: **16 bytes per weight** (2 bf16 weight + 2 grad + 4 fp32 master + 8 AdamW) before activations. 9B → 134 GiB, so one 80 GB card holds ≈5B params of state.
- Biggest activation of a small LLM: the logits, B·T·V. Fix: chunked cross-entropy. Activation checkpointing ≈ +⅓ compute for most of the activation memory.
- FLOPs per token: $6N + 12LTC$. MFU $=$ FLOPs/token × tokens/s ÷ peak. Peak = SMs × FLOP/clk/SM × clock.
- 9B at 12K tok/s on 8×H100 → 648 / 7,912 TFLOP/s = **8.2%**. Healthy is 35–50%.
- MFU = (busy/wall) × (matmul share of busy) × (matmul efficiency). Small models lose on all three: thin matmuls, many small kernels, Python launch gaps.
- Measure warm, interleave A-B-B-A, and write the clock next to every number. Laptop GPUs throttle.
- Dashboard: **loss** (learning?), **grad norm** (trouble?), **tokens/s** (slower?), **MFU** (wasted?).

## Failures that stay quiet (§22)

| Failure | Cheap check |
|---|---|
| checkpoint loads with missing/renamed keys | strict loading; held-out loss right after reload |
| forgotten `zero_grad` | gradient norm grows step after step |
| average of averages | normalize by tokens; compare against a one-big-batch gradient |
| padding counted in the loss | `ignore_index=-100`; count real tokens |
| wrong gradients | float64 gradient check against the nudge |
| fp16 underflow | fraction of zero gradients; bf16 or GradScaler |
| recompute without the same RNG | gradients with recompute on vs off |
| eval in train mode | `model.eval()` + `torch.no_grad()` |
| silent slowdown | tokens/s and MFU next to the GPU clock |

**Print things, and check things. The loss curve is not the thing that will tell you.**
