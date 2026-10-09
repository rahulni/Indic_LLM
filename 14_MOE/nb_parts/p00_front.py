from . import md, code

REPO = "rahulni/Indic_LLM/blob/main/14_MOE/14_dense_to_moe.ipynb"

CELLS = [
md(rf'''
# Dense → Mixture-of-Experts: train a linear model, convert it, keep training

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO})
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/{REPO})

> **Goal.** Train a dense ("linear") model, convert it into a mixture-of-experts, and show that the
> converted model keeps training and its loss keeps falling, at a size that fits one laptop or Colab
> GPU.

This notebook does that with a ~19M-parameter transformer trained on TinyStories. It converts the
model into a mixture-of-experts **twice**, dense → 8 experts → 32 experts, the way Lightning LM grew.
A dense control trains on the same batches with the same schedule for comparison.

It is also written to be **reopened later as a refresher**:
- Every published number it relies on is recomputed and asserted in code.
- Every mechanism (routing, balancing, dropping, growing, placement) is implemented in plain
  PyTorch, checked by an exact gate, and then measured in a short lab.
- Every idea comes with a plain-language intuition and a collapsible derivation.
- Set `MODE = "learn"` and it replays the saved results on a CPU in about a minute.
'''),
md(r'''
## Results at a glance

_Filled in from `assets/results.json` after the full run (`python nb_source.py --inject-results`)._
''', "glance"),
md(r'''
## Contents

| part | what |
|---|---|
| **1 · Setup** | modes, precision, budgets |
| **A · Refresher** | the core ideas with intuition and math, and published numbers as asserted code: total vs active, router by hand, combinations, capacity, aux loss, traffic |
| **B · Toy** | the whole experiment in 2-D: a linear model → a mixture of linear experts |
| **C · Building blocks** | dense GPT, the MoE layer (router, dropless dispatch, bias), shape trace, gates |
| **D · Data** | TinyStories, BPE-4096, paired batch order |
| **E · Stage 1** | train the dense model |
| **F · Surgery** | copy / partition / drop, Adam-state slicing, what each preserves |
| **G · Conversion lab** | six starting points, raced |
| **H · Router lab** | no balancing vs aux loss vs loss-free bias; softmax vs sigmoid vs √softplus |
| **I · Stage 2** | dense → MoE-8, plus the dense control |
| **J · Growth** | 8 → 32 experts; the clone-family trap and two ways out |
| **K · Result** | the main curve, paired test, samples, checklist |
| **K.2 · Post-hoc** | was it the growth recipe, or growing at all? |
| **L · Experts** | what they learned, how fast routing settles, what pruning costs |
| **M · Systems** | token dropping, EP placement, memory and time |
| **N · Wrap-up** | predictions scored, open design questions, cheat sheet, quiz |

**How the experiment is built**

```
tokens:   0 ─────────── T1 = 25M ─────── T2 = 33M ─────────────── T3 = 45M
MoE path: DENSE 18.9M   │ MoE-8 (26M)     │ MoE-32 (68M total, 18.9M active)
                        │ partition        │ clone x4 + redraw half + Gumbel top-k
control:                └─ DENSE continues on the same batches, same LR curve ──► T3
```
'''),
]
