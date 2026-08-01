# Proxy screen

The harness for §13 of [`MIXTURE_PLAN.md`](../MIXTURE_PLAN.md). Seven arms at 1B,
then a 3B confirmation of the baseline and the winner.

```bash
python arms.py                              # print + validate all arm configs
python arms.py --json out/                  # write one JSON per arm
python run_proxy.py --dry-run               # full control flow, no GPU
python run_proxy.py --dry-run --arm A6 --stage train
```

`--dry-run` exercises the real config generation, mixture schedule, sequence
ladder, LR schedule and decision logic with a stub trainer. Everything it prints
about cost and shape is real; no loss numbers are produced, because none exist.

## Arms

| Arm | Change | What it tests |
|---|---|---|
| A0 | — | Reference |
| A1 | Agentic → 0%, freed share to web | Whether a 2% agentic lane buys measurable tool use |
| A2 | Protected floor removed | Whether the floor is load-bearing |
| A3 | Code → 34%, web → 42% | The repetition ceiling, deliberately violated |
| A4 | No phases, same totals | Whether curriculum ordering matters at all |
| A5 | Selector disabled | Whether the 8× claim justifies 2× corpus |
| A6 | Crossfades removed | Whether the 15% band overlap earns its complexity |

A4 and A5 are the arms that can invalidate large parts of the plan, which is
why they exist.

## What the harness enforces

- **Corpus scaling.** Pools are subsampled by the budget factor so proxy epochs
  equal full-run epochs per lane. Without this A3 runs at ~0.005 epochs and
  cannot observe repetition at all.
- **Length-homogeneous batches.** `seq_len_at()` returns one length per batch;
  nothing below 4,096. Tokens-per-batch is held at 2,097,152 across the ladder
  by halving the example count at each doubling, so a length change is never
  also a batch-size change.
- **Band crossfade.** `mixture_at()` interpolates linearly across 15% of the
  outgoing phase. Compare `--arm A0` against `--arm A6` at frac 0.38 to see it.
- **LR schedule.** `lr_at()` runs warmup → cosine to 10% of peak → linear to
  zero across phase D. The anneal is a *learning-rate* event as much as a data
  event.
- **Arm derivation.** Every arm is a declared mutation of the baseline, and
  `recompute()` re-derives shares, selector multipliers, trained tokens and
  epochs afterwards. An arm cannot silently keep the baseline's numbers.

`validate()` fails the run if any arm's phase shares do not sum to 100.

## A bug this harness already caught

The LR warmup was specified as 15B absolute tokens — 0.5% of the 3T run, but
**75% of a 20B proxy arm**. Every arm would have been compared while still
warming up. It is now a fraction (`warmup_frac`), and the dry-run trace is what
surfaced it.

## To run for real

Needs `torch`, `transformers`, `datasets`, plus the cleaned corpora.

1. `--stage pools` — wire to the session-4 pipeline output; stream each lane's
   sources through the same dedup/decontamination gates until `target_tokens`
   is reached. Drop documents below 4,096 rather than padding them.
2. `--stage train` — the dataloader calls `mixture_at()`, `seq_len_at()` and
   `lr_at()` per step. Loss masking follows §12 of the plan: tool output and
   user turns are read, never trained on.
3. `--stage eval` — lm-evaluation-harness (MMLU, GSM8K), bigcode-evaluation-harness
   (HumanEval+, MBPP+), BFCL, RULER, FLORES-200 with chrF++. Indic per language,
   never averaged.
4. `--stage decide` — applies the pre-registered rules mechanically.

Cost: ~$253/arm at 1B (8×H100, 40% MFU, $3/GPU-h), $5,814 for the full screen
including the 3B confirmation — 0.38% of the 3T run it protects.
