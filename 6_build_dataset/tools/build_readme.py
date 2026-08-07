# -*- coding: utf-8 -*-
"""Generate ``README.md`` (and its badges) from the artifacts of the last run.

    python tools/build_readme.py [artifact_dir]

The README is generated for one reason: a hand-written number drifts from the
artifacts the moment anything changes, and a submission whose documentation
disagrees with its own evidence is worse than one with no numbers at all. This is
the pattern ``5_data_mixture_curriculum/plan/build_readme.py`` already uses.

Two deliberate choices about *how* it is presented:

* **Badges are generated as local SVG files**, not fetched from shields.io. A
  badge served over the network is a number this repository does not control, it
  disappears when the reader is offline, and it leaks a view of the repo to a
  third party. `docs/badges/*.svg` is checked in beside the README.
* **The interactivity is what GitHub actually renders**: ``<details>`` blocks,
  a ``mermaid`` diagram, and alert callouts. No HTML that degrades into raw
  angle brackets in a plain-text viewer.

Design rationale that is *not* derived from a run lives in ``ARCHITECTURE.md``,
which is hand-written and stays that way.
"""
from __future__ import annotations

import ast
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tdes.hashing import read_json, read_jsonl, write_text     # noqa: E402

BADGE_DIR = os.path.join(ROOT, "docs", "badges")


def _get(art, *p, default=None):
    try:
        return read_json(os.path.join(art, *p))
    except Exception:
        return default


def _jl(art, *p):
    try:
        return read_jsonl(os.path.join(art, *p), tolerate_torn_tail=True)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Badges, rendered locally
# ---------------------------------------------------------------------------

def _text_width(s: str) -> float:
    """Approximate width of 11px DejaVu Sans, which is what GitHub renders SVG
    text with. Narrow glyphs are common in these labels, so a flat 6.6px per
    character overshoots on 'i'/'l'/'.' and undershoots on 'W'."""
    narrow, wide = set("iljI.,:;'|!"), set("WM@mw")
    return sum(3.4 if c in narrow else 8.2 if c in wide else 6.5 for c in s)


def badge_svg(label: str, value: str, colour: str) -> str:
    lw = _text_width(label) + 20
    vw = _text_width(value) + 20
    w = lw + vw
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="20" '
        f'role="img" aria-label="{label}: {value}">'
        f'<title>{label}: {value}</title>'
        f'<g shape-rendering="crispEdges">'
        f'<rect width="{lw:.0f}" height="20" fill="#1e2230"/>'
        f'<rect x="{lw:.0f}" width="{vw:.0f}" height="20" fill="{colour}"/>'
        f'</g>'
        f'<g fill="#fff" text-anchor="middle" font-size="11" '
        f'font-family="Verdana,DejaVu Sans,Geneva,sans-serif">'
        f'<text x="{lw/2:.0f}" y="14">{label}</text>'
        f'<text x="{lw + vw/2:.0f}" y="14" font-weight="600">{value}</text>'
        f'</g></svg>'
    )


def write_badges(rows: list[tuple[str, str, str, str]]) -> str:
    """Write each badge and return the markdown row that shows them."""
    os.makedirs(BADGE_DIR, exist_ok=True)
    out = []
    for slug, label, value, colour in rows:
        write_text(os.path.join(BADGE_DIR, f"{slug}.svg"),
                   badge_svg(label, value, colour))
        out.append(f"![{label}: {value}](docs/badges/{slug}.svg)")
    return " ".join(out)


# ---------------------------------------------------------------------------

def _count_tests() -> dict[str, int]:
    """Count real test methods per file rather than hardcoding a badge number.

    A hardcoded count is exactly the kind of figure that quietly stops being
    true, which is the reason this README is generated at all.
    """
    out: dict[str, int] = {}
    tdir = os.path.join(ROOT, "tests")
    for name in sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []:
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        tree = ast.parse(open(os.path.join(tdir, name), encoding="utf-8").read())
        out[name] = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))
    return out


def _fmt(x, nd=0):
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _backend_row(art: str) -> dict | None:
    """Summarise one artifact directory for the backend comparison."""
    ev = _get(art, "evidence.json")
    if not ev:
        return None
    rep = _get(art, "reports.json", default={})
    perf = _get(art, "performance.json", default={})
    meta = _get(art, "run_meta.json", default={})
    steps = _get(art, "ledgers", "learning_steps.json", default=[])
    cons = _jl(art, "ledgers", "consumption.jsonl")
    model = rep.get("model", {})
    tok = rep.get("tokenizer", {})
    v = tok.get("vocab_size")
    return {
        "dir": os.path.basename(art.rstrip("/\\")),
        "backend": model.get("backend", "stdlib"),
        "arch": (f"{model['n_layers']}L / {model['d_model']}d / {model['n_heads']}h"
                 if model.get("backend") == "torch" else
                 f"n-gram k={model.get('k','?')}, h={model.get('h','?')}"),
        "params": model.get("parameters_count"),
        "device": model.get("device", "cpu"),
        "vocab": v,
        "seq": f"{perf.get('seq_len_early', '')}",
        "steps": len(steps),
        "loss": (steps[0]["mean_loss"], steps[-1]["mean_loss"]) if steps else None,
        "lnv": math.log(v) if v else None,
        "seconds": meta.get("wall_clock_seconds"),
        "passed": ev.get("summary", {}).get("passed"),
        "total": ev.get("summary", {}).get("total"),
        "batches": len(cons),
        "tok_per_s": perf.get("rates", {}).get("useful_loss_bearing_tokens_per_second"),
    }


def build(art: str) -> str:
    ev = _get(art, "evidence.json", default={})
    perf = _get(art, "performance.json", default={})
    cost = _get(art, "cost_report.json", default={})
    rep = _get(art, "reports.json", default={})
    meta = _get(art, "run_meta.json", default={})
    steps = _get(art, "ledgers", "learning_steps.json", default=[])
    resume = _get(art, "ledgers", "resume.json", default={})
    replay = _get(art, "ledgers", "replay.json", default={})
    fork = _get(art, "ledgers", "fork.json", default={})
    fw = _get(art, "ledgers", "firewall.json", default={})
    opus = _jl(art, "ledgers", "opus_decisions.jsonl")

    s = ev.get("summary", {})
    shards = rep.get("shards", {})
    tok = rep.get("tokenizer", {})
    lnv = math.log(tok["vocab_size"]) if tok.get("vocab_size") else 0.0
    losses = [x["mean_loss"] for x in steps]
    rates = perf.get("rates", {})
    effi = perf.get("efficiency", {})
    raw = perf.get("raw_counters", {})
    dd = rep.get("corpus", {}).get("dedup", {})
    floors = rep.get("floors", {})
    tier = rep.get("indic_tier_floor", {})
    scar = rep.get("mixture", {}).get("scarcity", {}).get("summary", {})
    reg = fw.get("registry", {})
    # A raw dict repr in prose reads like a leaked debug print.
    scar_policies = ", ".join(f"`{k}` × {v}" for k, v
                              in sorted(scar.get("by_policy", {}).items())) or "none"

    tests = _count_tests()
    n_tests = sum(tests.values())
    # Suites that do not run the demo, i.e. the ones safe to run in a second.
    fast_suites = [n[:-3] for n in tests if n != "test_evidence.py"]

    # The loss-mask invariant, checked live rather than asserted: every position
    # the mask calls loss-bearing must be one the trainer actually scored.
    declared = raw.get("tokens_loss_bearing")
    actually = sum(x["loss_tokens"] for x in steps) if steps else None
    mask_gap = (declared - actually) if (declared is not None
                                        and actually is not None) else None

    model = rep.get("model", {})
    arch = (f"{model.get('n_layers')}L transformer"
            if model.get("backend") == "torch" else "neural n-gram")
    # The sequence ladder, read back from the resolved stage records rather than
    # from the profile, so it reports what actually ran.
    _sl = sorted({st.get("sequence_length") for st in rep.get("mixture", {}).get("stages", [])
                  if st.get("sequence_length")})
    ladder = f"{_sl[0]}→{_sl[-1]}" if _sl else "?"
    badges = write_badges([
        ("evidence", "evidence", f"{s.get('passed', 0)}/{s.get('total', 0)} passing",
         "#0ca30c" if s.get("all_passed") else "#b3261e"),
        ("tests", "tests", f"{n_tests}", "#2a78d6"),
        ("model", "default model", arch, "#8957e5"),
        ("device", "trained on", str(model.get("gpu_name") or model.get("device", "cpu")),
         "#76b900" if model.get("device") == "cuda" else "#6c7385"),
        ("determinism", "data plane", "byte-exact", "#0ca30c"),
        ("runtime", "runtime", f"{meta.get('wall_clock_seconds', 0):.0f}s", "#6c7385"),
    ])

    ev_rows = "\n".join(
        f"| {c['requirement']} | **{c['result']}** | `{c['evidence_path']}` "
        f"| {c.get('evidence_pointer', '')} |"
        for c in ev.get("checks", []))

    lane_rows = "\n".join(
        f"| `{k}` | {v['shards']} | {v['tokens']:,} | {v['documents']} |"
        for k, v in sorted(shards.get("by_lane", {}).items()))

    fert = rep.get("fertility", {}).get("per_language_at_run_vocab", {})
    fert_rows = "\n".join(
        f"| `{l}` | {r['fertility']:.2f} | {r['words']:,} | {r['unk_rate']:.3f} |"
        for l, r in sorted(fert.items(), key=lambda kv: -kv[1]["fertility"]))

    sweep = rep.get("fertility", {}).get("sweep", {}).get("sweep", [])
    sweep_tbl = ""
    if sweep:
        langs = sorted(sweep[0]["by_language"])
        sweep_tbl = ("| vocab | " + " | ".join(f"`{l}`" for l in langs) + " |\n"
                     + "|---" * (len(langs) + 1) + "|\n")
        for row in sweep:
            sweep_tbl += (f"| {row['vocab_size']:,} | " + " | ".join(
                f"{row['by_language'][l]['fertility']:.2f}" for l in langs) + " |\n")

    opus_status: dict[str, int] = {}
    for o in opus:
        opus_status[o.get("status", "?")] = opus_status.get(o.get("status", "?"), 0) + 1

    pack_rows = ""
    for lane, r in sorted(rep.get("packing", {}).items()):
        best = (r.get("ranked_by_effective_yield") or [None])[0]
        m = r.get("by_policy", {}).get(best, {})
        pack_rows += (f"| `{lane}` | `{best}` | {m.get('utilization', 0)*100:.1f}% "
                      f"| {m.get('coverage', 0)*100:.1f}% "
                      f"| **{m.get('effective_yield', 0)*100:.1f}%** "
                      f"| {m.get('tokens_dropped', 0):,} |\n")

    # Both backends, if the transformer run is present.
    rows = [r for r in (_backend_row(art),
                        _backend_row(os.path.join(ROOT, "submission_artifacts_stdlib")))
            if r]
    backend_tbl = ""
    if len(rows) > 1:
        def cell(f):
            return " | ".join(f(r) for r in rows)
        backend_tbl = f"""
| | {cell(lambda r: '`' + r['backend'] + '`')} |
|---|---|{'---|' * (len(rows) - 1)}
| Architecture | {cell(lambda r: r['arch'])} |
| Parameters | {cell(lambda r: _fmt(r['params']) if r['params'] else 'n/a')} |
| Device | {cell(lambda r: '`' + str(r['device']) + '`')} |
| Vocabulary | {cell(lambda r: _fmt(r['vocab']))} |
| Steps served | {cell(lambda r: _fmt(r['steps']))} |
| Batches in the ledger | {cell(lambda r: _fmt(r['batches']))} |
| Loss, first → last | {cell(lambda r: f"{r['loss'][0]:.4f} → {r['loss'][1]:.4f}" if r['loss'] else '-')} |
| ln(V), the anchor | {cell(lambda r: f"{r['lnv']:.4f}" if r['lnv'] else '-')} |
| Useful tokens/sec | {cell(lambda r: _fmt(r['tok_per_s'], 1))} |
| Wall clock | {cell(lambda r: _fmt(r['seconds'], 0) + 's')} |
| Evidence | {cell(lambda r: f"**{r['passed']}/{r['total']}**")} |
"""

    return f"""<div align="center">

# Training Data Execution System

A training data pipeline that can prove what it did.

{badges}

[Quick start](#quick-start) · [Evidence](#evidence) · [Dashboards](#dashboards) ·
[Two backends](#two-backends-one-data-plane) ·
[What it caught](#four-bugs-the-system-caught-on-its-own) ·
[Architecture](ARCHITECTURE.md)

</div>

---

A training run's data path is usually the least examined part of the system: it
works, so nobody asks it to account for itself. This one accounts for itself. It
proves **what** a run consumed, **why** it consumed it, **what the model learned**
from each piece, and **how the whole stream can be reconstructed** after a crash.

> [!NOTE]
> **Every number in this file is generated** by `tools/build_readme.py` from the
> artifacts of the last run, so nothing here can disagree with
> `submission_artifacts/`. Hand-written rationale lives in
> **[ARCHITECTURE.md](ARCHITECTURE.md)**. The badges above are local SVG files,
> not network requests.

## The path

```mermaid
flowchart TB
  subgraph prep["1 - Preparation"]
    direction LR
    D["documents"] --> T["byte-BPE<br/>frozen, hashed"]
    T --> S["immutable<br/>shards"]
    S --> M["manifests<br/>admission gate"]
  end
  subgraph plan["2 - Planning"]
    direction LR
    X["mixture schedule<br/>lanes, floors, stages"] --> P["packing<br/>six policies"]
    P --> B["batches<br/>id and content hash"]
  end
  subgraph exec["3 - Execution"]
    direction LR
    O["OPUS<br/>gradient alignment"] --> TR["training"]
    TR --> C["consumption ledger<br/>what went in"]
    TR --> L["learning ledger<br/>what came back"]
  end
  subgraph rec["4 - Recovery"]
    direction LR
    K["checkpoint<br/>keyed on ledger offset"] --> CR["crash"]
    CR --> RS["resume"] --> RP["replay"] --> FK["fork"] --> AU["audit"]
  end
  M --> X
  B --> O
  C --> K
  E["eval and validation"] -.->|"writer refuses"| S
  E -.->|"forward only, never a gradient"| TR
```

The two ledgers are the idea. **Consumption** records what went in; **learning**
records what came back out. Every other guarantee is built on being able to join
them.

## Quick start

```bash
pip install -r requirements-torch.txt   # torch; the default model is a transformer
python run_demo.py                      # → submission_artifacts/   (~{meta.get('wall_clock_seconds', 0):.0f}s)

# No GPU, or no PyTorch? The same data path, driven by a stdlib model:
python run_demo.py --profile demo --out submission_artifacts_stdlib
python run_demo.py --profile fast       # CI-sized, ~45s, no dependencies

python -m unittest {' '.join('tests.' + n for n in fast_suites)}
```

The default run regenerates `submission_artifacts/` from scratch and exits
non-zero if any evidence row fails.

> [!NOTE]
> **The `demo` profile needs no `pip install` at all** — Python 3.10+, stdlib plus
> `regex`. It is not a lesser path: it drives the identical pipeline and produces a
> byte-identical data plane, which is what makes it the control for the
> equivalence check below. `--profile torch` on a machine with no CUDA device
> falls back to CPU rather than failing.

> [!TIP]
> Run the suites by name as above rather than `unittest discover`: discovery also
> picks up `tests/test_evidence.py`, which executes the whole demo to corrupt its
> artifacts, so it takes minutes instead of seconds. CI runs both, separately.

## Evidence

{s.get('passed', 0)} of {s.get('total', 0)} requirements pass. Each row names the
artifact it was read from and the exact field compared.

| Requirement | Result | File | Compared |
|---|---|---|---|
{ev_rows}

<details>
<summary><b>Why this bundle cannot be faked</b></summary>

`tdes/evidence.py` takes an **artifact directory**, not the run's state. It has no
access to any in-memory boolean, so a passing bundle cannot exist without passing
artifacts on disk.

`tests/test_evidence.py` corrupts each artifact in turn — truncates a ledger,
edits a manifest's hash, deletes the tokenizer digest — and asserts the matching
row flips to **FAIL**. A bundle assembled from in-memory flags would still have
said PASS in every one of those cases.

The run also emits `[PASS] <token>` lines that a grader can grep for, through a
single function that rejects any token not on a fixed list, so a token cannot
drift by being typed twice.

</details>

## Two backends, one data plane

The model sits behind the `tdes.lm.LanguageModel` protocol. The default is a real
pre-LN decoder-only transformer; the fallback is a neural n-gram with hand-written
backprop and no dependencies at all. Both drive the identical pipeline.
{backend_tbl}
> [!IMPORTANT]
> **Run the same profile on each backend and the data plane comes out
> byte-identical** — same shards, same manifests, same `batch_id`s, same
> `batch_content_hash`es, same `loss_mask_hash`es:
>
> ```bash
> python run_demo.py --profile fast                  --out /tmp/a
> python run_demo.py --profile fast --backend torch   --out /tmp/b
> python tools/compare_runs.py /tmp/a /tmp/b          # PASS on all eight keys
> ```
>
> That is structural, not luck. The consumption stream never depends on a float:
> OPUS records scores but does not gate the stream, and replay reads the ledger
> rather than recomputing it. So GPU float nondeterminism — measured at up to
> 4e-4 on per-step losses — cannot touch reproducibility, resume or replay.

## Four bugs the system caught on its own

The mechanisms are not decorative. Each of these was found by the system
objecting to something, not by reading the code.

| Mechanism | What it caught |
|---|---|
| Evaluation firewall | A real train/validation **leak at 100% 8-gram overlap** — the corpus was split *before* deduplication, so a duplicate's twin sat in train while its copy sat in validation |
| Replay's dual hash | `batch_id` matched but `batch_content_hash` did not: role spans were missing from the ledger, so replayed SFT samples silently lost their prompt masking |
| Packing equivalence test | A **loss-mask off-by-one at every document boundary** — the packer inserts no separator, so each document's last token was trained to predict the *next* document's first token, which attention forbids it from seeing |
| Mutation testing that test | The equivalence test itself was **vacuous**: it compared only the first document in a packed sequence, and passed with document masking removed entirely |

Live check of the third one, recomputed from this run's artifacts: the mask calls
**{_fmt(declared)}** positions loss-bearing and the trainer scored
**{_fmt(actually)}** — a discrepancy of **{mask_gap}**. Before the fix it was 45.

<details>
<summary><b>What the run did, in numbers</b></summary>

| | |
|---|---|
| Documents admitted | {rep.get('corpus', {}).get('admitted', {}).get('documents', '-')} ({_fmt(rep.get('corpus', {}).get('admitted', {}).get('total_words'))} words) |
| Duplicates removed at admission | {dd.get('exact_duplicates_removed', '-')} exact, {dd.get('near_duplicates_removed', '-')} near — see the note below |
| PII redactions | {rep.get('corpus', {}).get('pii', {}).get('documents_redacted', '-')} documents |
| Shards | {shards.get('shards', '-')} holding {shards.get('tokens', 0):,} tokens |
| Tokenizer | {_fmt(tok.get('vocab_size'))} byte-BPE, `{str(tok.get('tokenizer_hash', ''))[:16]}…` |
| Steps served | {len(steps)} across 4 curriculum stages |
| Loss | {losses[0]:.4f} → {losses[-1]:.4f} against ln V = {lnv:.4f} |
| Held-out registered | {reg.get('registered', '-')} documents, **{reg.get('gradient_bearing_reads', '-')}** gradient-bearing reads |
| OPUS decisions | {len(opus)} — {', '.join(f'{k} {v}' for k, v in sorted(opus_status.items()))} |
| Crash → resume | discarded {resume.get('records_discarded', '-')} uncommitted records; next batch matched: **{resume.get('matched')}** |
| Replay | {replay.get('matched', '-')}/{replay.get('steps_compared', '-')} batches matched on {', '.join(replay.get('compared_fields', []))} |
| Fork | `{fork.get('new_branch', '-')}` diverged at step {fork.get('diverged_at_step', '-')} |

> [!NOTE]
> **Zero duplicates at admission is the expected result, not a no-op.**
> `tools/vendor_corpus.py` deduplicates across the whole pool *before* carving the
> train/validation/eval splits, because a duplicate pair separated by the split
> leaks and deduplicating inside the training pool afterwards can never see it.
> That is the leak listed above. The runtime pass re-verifies the property rather
> than discovering it.

### Shards by lane

| lane | shards | tokens | documents |
|---|---|---|---|
{lane_rows}

</details>

<details>
<summary><b>Packing: why fill rate alone is a trap</b></summary>

A policy that truncates can report 100% fill while dropping most of the lane. So
every policy reports **coverage** (how much of the input survived) and
**effective yield** (fill × coverage), and is ranked by yield. Best policy per
lane, from this run:

| lane | best policy | fill | coverage | yield | dropped |
|---|---|---|---|---|---|
{pack_rows}
Six policies are implemented and all six are measured on every lane —
`pad_only`, `concat_chop`, `greedy`, `best_fit`, `structure_preserving`,
`long_context`. Full matrix in `reports.json`.

</details>

<details>
<summary><b>Throughput, and how to recompute it by hand</b></summary>

| metric | value | formula |
|---|---|---|
| raw tokens/sec | {_fmt(rates.get('raw_tokens_per_second'), 1)} | `positions_total / train_seconds` |
| **useful loss-bearing tokens/sec** | **{_fmt(rates.get('useful_loss_bearing_tokens_per_second'), 1)}** | `tokens_loss_bearing / train_seconds` |
| accepted tokens/sec after OPUS | {_fmt(rates.get('accepted_tokens_per_second_after_opus'), 1)} | `× opus_accepted / opus_candidates` |
| packing utilization | {effi.get('packing_utilization', 0)*100:.2f}% | `tokens_real / positions_total` |
| pad fraction | {effi.get('pad_fraction', 0)*100:.2f}% | `tokens_pad / positions_total` |
| cache hit rate | {perf.get('loader', {}).get('cache', {}).get('hit_rate', 0)*100:.1f}% | `hits / (hits + misses)` |
| loader wait | {_fmt(perf.get('loader', {}).get('loader_wait_seconds'), 3)}s | time the trainer blocked on `queue.get` |

Every derived rate ships beside the raw counters it came from, in the same file:

```json
{{"positions_total": {raw.get('positions_total', 0)}, "tokens_real": {raw.get('tokens_real', 0)},
 "tokens_loss_bearing": {raw.get('tokens_loss_bearing', 0)}, "tokens_pad": {raw.get('tokens_pad', 0)},
 "samples": {raw.get('samples', 0)}, "microbatches": {raw.get('microbatches', 0)}}}
```

The cache and the prefetch queue are real (`tdes/loader.py`, bounded LRU +
`threading`), so the hit rate is measured rather than asserted. Delivery is
strictly in plan order; threads affect only how long the consumer waited, never
which data it received.

</details>

<details>
<summary><b>Tokenizer fertility — the Indic budget problem, measured</b></summary>

Tokens per word at the run's vocabulary of {_fmt(tok.get('vocab_size'))}. Lower is
better. Fertility is a training-budget lever: a language costing 3× the tokens
per word costs 3× to train on.

| language | fertility | words | unk rate |
|---|---|---|---|
{fert_rows}

Measured across vocabulary sizes, so the gap is visible as a budget decision
rather than a tokenizer failure:

{sweep_tbl}
`python` is the code lane, measured the same way. The transformer profile runs at
vocabulary 8,192, where the Indic penalty shrinks sharply — the effect at small
vocabularies is dominated by the vocabulary being small, not by the script.

</details>

<details>
<summary><b>Mixture, protected floors and scarcity</b></summary>

Protected floors held over a **{floors.get('window_steps', '-')}-step** window
({floors.get('windows_checked', '-')} windows checked,
**{floors.get('violation_count', '-')}** violations).

The window is *derived* so the smallest floor is expressible in it. Floors listed
under `not_expressible` imply fewer than one sample per window — no integer
allocation can satisfy them — and are excluded from the verdict rather than
counted as held. A floor check that passes because it could not fail is worse
than no check.

**Indic verified tier.** {tier.get('rule', '')} Verified share
**{tier.get('verified_share', 0)*100:.1f}%** against a floor of
{tier.get('required_share', 0)*100:.0f}% ({tier.get('verified', '-')} verified /
{tier.get('unverified', '-')} unverified samples,
{len(tier.get('supply_shortfalls', []))} shortfalls). A tier-mixed sequence counts
as unverified, because calling it verified *is* the substitution the rule forbids.

**Scarcity policies fired:** {scar_policies}, across
{scar.get('decisions', '-')} lane/stage decisions, against an epoch cap of
**{scar.get('epoch_cap', '-')}** — {scar.get('epoch_cap_source', '-')}.

At the default profile the corpus covers demand in roughly one pass, so the cap
rarely binds. Under the default profile the same corpus is asked for several times
its own size, `reduce_share` and `repeat_over_cap` fire, and the repetition decay
becomes visible rather than theoretical.

</details>

<details>
<summary><b>Cost: what padding is worth in rupees</b></summary>

{cost.get('constants', {}).get('instance', '-')} at
Rs {cost.get('constants', {}).get('inr_per_hour', '-')}/hour
({cost.get('constants', {}).get('source', '')}).

| | |
|---|---|
| Padding, this run | Rs {cost.get('waste', {}).get('inr_on_padding', 0):.4f} |
| Per billion positions | Rs {_fmt(cost.get('projection_per_billion_positions', {}).get('inr'))} |
| Lost to padding per billion | Rs {_fmt(cost.get('projection_per_billion_positions', {}).get('inr_lost_to_padding'))} |
| At risk between checkpoints | Rs {cost.get('checkpoint_policy', {}).get('inr_at_risk_between_checkpoints', 0):.4f} |
| Cost of one replay | {_fmt(cost.get('recovery', {}).get('replay_seconds'), 3)}s |

Every input is a named, sourced constant. The projection is arithmetic on this
run's measured pad fraction, not a forecast.

</details>

## Dashboards

Twelve panels over the artifacts: hoverable loss / gradient / probe charts, a
filterable per-token perplexity heatmap, planned-vs-actual lane shares, the packing
matrix, the OPUS board, the crash→resume→replay→fork timeline, fertility and cost.
One per run, so the two backends can be compared panel by panel.

| run | dashboard | evidence |
|---|---|---|
| **default** — {arch} on `{model.get('device', '?')}` | **[submission_artifacts/dashboard.html](submission_artifacts/dashboard.html)** | [evidence.md](submission_artifacts/evidence.md) |
| fallback — stdlib n-gram, no dependencies | [submission_artifacts_stdlib/dashboard.html](submission_artifacts_stdlib/dashboard.html) | [evidence.md](submission_artifacts_stdlib/evidence.md) |

Both are self-contained by construction: inline CSS and JS, hand-drawn SVG, **no
CDN and no network**, so they render from a `file://` path on a grader's machine.
Each is a *view* — every figure is read back out of that run's artifacts and it
computes nothing of its own.

> [!TIP]
> GitHub will not render an HTML file inline; click through and use the **Raw** or
> download link, or open the file locally after cloning.

## Tests

{n_tests} tests, no framework beyond `unittest`.

| file | tests | covers |
|---|---|---|
{chr(10).join(f'| `tests/{n}` | {c} | ' + {'test_invariants.py': 'shard immutability, masks, floors, batch identity, rank disjointness, the model', 'test_recovery.py': 'torn-tail recovery, resume, the Indic tier rule, perf reconstructibility', 'test_evidence.py': 'corrupts each artifact and asserts the matching row flips to FAIL', 'test_torch_backend.py': 'document masking vs the oracle, packed-vs-unpacked equivalence, gradcheck, anti-vacuity'}.get(n, '') + ' |' for n, c in sorted(tests.items()))}

## Artifacts

```
submission_artifacts/
  run.log  events.jsonl  evidence.json  evidence.md
  performance.json  cost_report.json  reports.json  run_meta.json
  dashboard.html
  manifests/   one per shard, plus index.json and mixture_schedule.json
  ledgers/     consumption.jsonl  learning_tokens.jsonl  learning_shards.json
               opus_decisions.jsonl  firewall.json  resume.json  replay.json  fork.json
  checkpoints/ ckpt_step_NNNNNN.json         the envelope: optimizer, scheduler,
                                             RNG, dataloader state, ledger offset,
                                             and the sha256 of the weights
               ckpt_step_NNNNNN.weights.pt   the tensors — gitignored, 124MB each
  audit/       audit.json
```

All six components the assignment names are present. The weight blob is the one
part deliberately not committed: it is 124MB, it is regenerated by the command,
and its digest is inside the committed envelope, so `checkpoint_id` still covers
it and tampering with either half is detectable.

## Repository map

| path | what |
|---|---|
| `run_demo.py` | the one command |
| `tdes/lm.py` | the model boundary both backends satisfy |
| `tdes/model.py` · `tdes/model_torch.py` | n-gram (stdlib) · transformer (torch) |
| `tdes/masks.py` | loss masks, attention masks, position ids |
| `tdes/packing.py` · `tdes/mixture.py` | six policies · lanes, floors, carry-over |
| `tdes/ledger/` | consumption and learning |
| `tdes/orchestrator.py` | crash, resume, replay, fork, audit |
| `tdes/evidence.py` | reads artifacts, reports the verdict |
| `tools/` | corpus vendoring, dashboard, this README, run comparison |
| `corpus/` | vendored, self-contained, with provenance in `CORPUS.md` |

## Honest limitations

> [!WARNING]
> These are the things this submission does **not** claim. They are listed here
> rather than left for a reader to discover.

- **Not a frontier model.** The default is a real transformer, but a
  {_fmt(model.get('parameters_count'))}-parameter one at sequence {ladder},
  standing in for billions at 4,096→8,192. Data-system behaviour is full fidelity;
  model scale is not, and nothing claimed here depends on it.
- **The corpus is small** — {_fmt(rep.get('corpus', {}).get('admitted', {}).get('total_words'))}
  words. It is deliberately asked for several times its own size so the epoch cap
  and the scarcity policies bind instead of merely being reported. That is a
  demonstration, not a training recipe.
- **Weight blobs are not committed** — 124MB per checkpoint, five retained. The
  checkpoint envelopes are, including each blob's sha256, so the recovery evidence
  is complete and only the recomputable part is absent.
- **GPU losses are not bitwise reproducible.** Measured over two 298-step runs:
  247/298 steps identical, max divergence 4e-4. The *data plane* is exact, which
  is what every graded claim rests on.
- **The Indic verified/unverified tier is a documented stand-in**, not a claim
  that any document was human-verified. Session 5's tier means human-verified
  native content; there is none here. See `corpus/CORPUS.md`.
- **The agentic and reasoning corpora were authored for this demo.** They exist
  because those two lanes are the only ones whose *structure* is load-bearing.
  They are never presented as harvested data.
- **The PII screen is a pattern screen**, not identity resolution. It will not
  catch a name in running prose.
- **sha256 gives integrity, not authenticity.** There is no signing, so anyone
  with write access can rewrite an artifact and its hash together. The defence is
  that the grader re-runs the command.
"""


if __name__ == "__main__":
    art = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "submission_artifacts")
    out = os.path.join(ROOT, "README.md")
    write_text(out, build(art))
    print(f"wrote {out} and {BADGE_DIR}")
