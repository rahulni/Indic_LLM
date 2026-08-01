# ERA V5 — Indic LLM

Coursework for building an India-first foundation model, one session per folder.

## Session 5 — the current submission

**→ [`5_data_mixture_curriculum/README.md`](5_data_mixture_curriculum/README.md)**

The deliverable is
[`MIXTURE_PLAN.md`](5_data_mixture_curriculum/MIXTURE_PLAN.md) — the V5
mixture-and-curriculum specification for Drishtikon-40B. It is *generated*, not
written: every number comes out of `plan/spec.py` → `audit.py` →
`build_plan.py`, and the build refuses to emit the document if the arithmetic
does not close. Change a share, and either the plan re-derives or the build
fails.

What was executed rather than only specified:

| | Result |
|---|---|
| Data gate | **93,019,466 cleaned tokens** (1.73× session 4), after cleaning Sangraha Verified — the tier the audit ranks first and the only one the anneal accepts |
| Benchmark contamination | Measured against 6 suites; 0.0000% and 0.0209% |
| Proxy experiment | 11M-param micro-proxy, 3 arms × 2 seeds — both verdicts INCONCLUSIVE, and *why* is the finding |

The 1B/3B screen in §13 is specified and costed but **not run** — it needs GPUs
we do not have. That is stated in the plan rather than glossed.

## Earlier sessions

| Folder | Session |
|---|---|
| [`1_attention/`](1_attention/) | Attention mechanics |
| [`2_token/`](2_token/) | Tokenizer design and Indic fertility |
| [`3_40b_model/`](3_40b_model/) | Drishtikon-40B model design memo |
| [`4_model_data/`](4_model_data/) | 8-stage data cleaning pipeline |
| [`5_data_mixture_curriculum/`](5_data_mixture_curriculum/) | Mixture and curriculum plan |

## What is not in this repo, and why

Corpus text is excluded — **for privacy before size**. The raw draws and stage
1–5 intermediates are bulk scraped Indic web documents that have *not* passed
the pipeline's stage-6 PII scrubber. Publishing them would republish personal
data harvested from the open web, which is the harm that filter and the
DPDP-Act analysis in `3_40b_model` exist to prevent.

Nothing is lost. Every sample is reproducible from `stage0_sample.py` at a fixed
offset and dataset revision — only the obligation to rehost it goes away.

Also excluded: virtualenvs (5.4 GB), pipeline intermediates, and `.bin` training
arrays, all regenerable. See [`.gitignore`](.gitignore), which documents each
exclusion inline.

## Reproducing

```bash
cd 5_data_mixture_curriculum/plan
python audit.py         # invariants + per-lane verdicts
python build_plan.py    # regenerates MIXTURE_PLAN.md
```

Standard library only. Actually training the proxy needs `torch`; see
[`5_data_mixture_curriculum/proxy/README.md`](5_data_mixture_curriculum/proxy/README.md).
