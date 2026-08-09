# -*- coding: utf-8 -*-
"""Wiring: build every component from a config, in dependency order.

``run_demo.py`` is the narrative -- it decides what to demonstrate and what to
log. This module is the plumbing that gets a :class:`RunState` into existence,
kept separate so the tests can construct the same system without going through
the demo script.
"""
from __future__ import annotations

import os

from .batching import build_pools, build_sample
from .config import (DATALOADER_VERSION, LANE_PACKING, RunConfig, STAGES)
from .corpus import admit
from .firewall import EvalRegistry
from .hashing import hash_source_tree
from .ledger.consumption import ConsumptionLedger
from .ledger.learning import LearningLedger
from .loader import ShardLoader
from .manifest import build_manifest, summarise, validate
from .mixture import compile_schedule
from . import lm
from .opus import OpusSelector
from .orchestrator import RunState
from .packing import compare_policies, pack
from .shards import build_shards, load_shard_tokens
from .tokenizer import fertility, fertility_sweep, load_frozen, train_and_freeze
from .trainer import LRSchedule, Trainer, ValidationProbe

SHARD_TARGET_TOKENS = 4000


def build_everything(cfg: RunConfig, *, log=None, total_steps: int) -> dict:
    """Construct the whole system. Returns the state plus every report."""
    out_dir = cfg.out_dir
    shard_dir = os.path.join(out_dir, "shards")
    reports: dict = {}

    # -- corpus admission -------------------------------------------------
    adm = admit(cfg.corpus_dir, docs_per_lane_cap=cfg.profile.docs_per_lane_cap)
    reports["corpus"] = adm["reports"]
    if log:
        r = adm["reports"]
        log.info(f"admitted {r['admitted']['documents']} documents "
                 f"({r['dedup']['exact_duplicates_removed']} exact + "
                 f"{r['dedup']['near_duplicates_removed']} near duplicates removed, "
                 f"{r['pii']['documents_redacted']} PII-redacted)")

    # -- tokenizer --------------------------------------------------------
    # Scoped by vocabulary size: the 512-token tokenizer the stdlib profiles use
    # and the 8,192-token one the transformer uses are different frozen
    # artifacts, and a shard is only meaningful relative to exactly one of them.
    # Sharing one directory would have let a run silently overwrite the tokenizer
    # another run's committed shards were built with.
    tok_dir = os.path.join(cfg.corpus_dir, "tokenizer", f"v{cfg.profile.vocab_size}")
    train_texts = [d["text"] for d in adm["documents"]]
    tok, tstats = train_and_freeze(train_texts, cfg.profile.vocab_size, tok_dir)
    tok, thash = load_frozen(tok_dir)                # verify on every load
    reports["tokenizer"] = tstats
    if log:
        log.check("tokenizer_hash_verified", thash == tstats["tokenizer_hash"],
                  vocab_size=tok.vocab_size, unit_mode=tok.unit_mode,
                  hash=thash[:16], merges=tstats["n_merges"])

    by_lang: dict[str, list[str]] = {}
    for d in adm["documents"]:
        by_lang.setdefault(d.get("language", "?"), []).append(d["text"])
    reports["fertility"] = {
        "per_language_at_run_vocab": {
            l: fertility(tok, t) for l, t in sorted(by_lang.items())},
        "sweep": fertility_sweep(train_texts, by_lang,
                                 list(cfg.profile.fertility_sweep_sizes)),
    }

    # -- firewall ---------------------------------------------------------
    registry = EvalRegistry()
    registry.register_all(adm["heldout"])
    reports["firewall_registry"] = registry.summary()

    # -- shards + manifests ----------------------------------------------
    tree = hash_source_tree(os.path.dirname(os.path.abspath(__file__)))
    manifests: list[dict] = []
    rejected: list[dict] = []
    shard_paths: dict[str, str] = {}
    items: dict[str, list[dict]] = {}
    stage_of_lane = {l: STAGES[0].stage_id for l in LANE_PACKING}

    for lane, docs in sorted(adm["by_lane"].items()):
        for s in build_shards(docs, tok, vocab_size=tok.vocab_size, lane=lane,
                              target_tokens=SHARD_TARGET_TOKENS,
                              out_dir=shard_dir, tokenizer_hash=thash):
            overlap = _screen_shard(s, registry)
            m = build_manifest(
                s, tokenizer_hash=thash, cleaning_hash=tree["combined"],
                cleaning_per_file=tree["per_file"],
                dedup_report=adm["reports"]["dedup"],
                pii_report=adm["reports"]["pii"],
                curriculum_stage=stage_of_lane.get(lane, "A"),
                eval_overlap=overlap)
            ok, reasons = validate(m)
            if ok:
                manifests.append(m)
                shard_paths[s["shard_id"]] = s["path"]
                items.setdefault(lane, []).extend(
                    [dict(sp, shard_id=s["shard_id"]) for sp in s["spans"]])
            else:
                rejected.append({"shard_id": s["shard_id"], "reasons": reasons})

    reports["shards"] = summarise(manifests)
    reports["shards"]["rejected_at_gate"] = rejected
    if log:
        log.event("shards created", shards=len(manifests),
                  tokens=reports["shards"]["tokens"])
        log.event("manifests validated", admitted=len(manifests),
                  rejected=len(rejected))

    # -- packing comparison ----------------------------------------------
    reports["packing"] = {
        lane: compare_policies(its, cfg.profile.seq_len_early)
        for lane, its in sorted(items.items())
    }

    # -- schedule ---------------------------------------------------------
    plan = compile_schedule(cfg, manifests, total_steps=total_steps)
    reports["mixture"] = {k: v for k, v in plan.items() if k != "steps"}
    if log:
        log.event("mixture compiled", stages=len(plan["stages"]),
                  steps=total_steps,
                  planned=plan["planned_shares"]["by_lane_share"])

    # -- pools + loader ---------------------------------------------------
    seq_lens = sorted({cfg.seq_len_for(s) for s in STAGES})
    pools = build_pools(plan, items, LANE_PACKING, cfg.seed, seq_lens)
    loader = ShardLoader(shard_paths, tok.vocab_size,
                         cache_capacity=cfg.profile.cache_capacity,
                         workers=cfg.profile.loader_workers,
                         prefetch_depth=cfg.profile.prefetch_depth)

    # -- validation probe -------------------------------------------------
    probe = _build_probe(cfg, adm, tok, thash, out_dir, registry)

    # -- model / trainer / ledgers ---------------------------------------
    P = cfg.profile
    model = lm.build(cfg, tok.vocab_size, total_steps=total_steps)
    reports["model"] = lm.describe(model)
    if log:
        d = reports["model"]
        log.info(f"model: {d.get('backend')} {d.get('class')} "
                 f"ln(V)={model.initial_loss():.4f}"
                 + (f"  {d.get('n_layers')}L/{d.get('d_model')}d/"
                    f"{d.get('n_heads')}h  {d.get('parameters_count'):,} params "
                    f"on {d.get('device')} ({d.get('amp_dtype')})"
                    if d.get("backend") == "torch" else ""))
    schedule = LRSchedule(0.30, total_steps)
    trainer = Trainer(model, schedule)

    ledger_dir = os.path.join(out_dir, "ledgers")
    cons = ConsumptionLedger(os.path.join(ledger_dir, "consumption.jsonl"),
                             run_id=cfg.run_id, branch_id="main")
    # Full token trace for an early, a middle and a late window -- enough to see
    # the same shard get easier -- with aggregates over every step.
    q = max(2, total_steps // 10)
    learn = LearningLedger(
        os.path.join(ledger_dir, "learning_tokens.jsonl"),
        os.path.join(ledger_dir, "learning_shards.json"),
        trace_windows=[(0, q), (total_steps // 2, total_steps // 2 + q),
                       (max(0, total_steps - q), total_steps + 1)])

    protected = {l for st in STAGES for l in st.protected_floors}
    selector = OpusSelector(accept_ratio=P.opus_accept_ratio,
                            prefix_tokens=P.opus_prefix_tokens,
                            protected_lanes=protected)

    state = RunState(cfg, model=model, trainer=trainer, schedule=schedule,
                     pools=pools, loader=loader, schedule_plan=plan,
                     cons_ledger=cons, learn_ledger=learn, tokenizer=tok,
                     tokenizer_hash=thash, registry=registry, probe=probe,
                     selector=selector)

    return {"state": state, "reports": reports, "manifests": manifests,
            "items": items, "shard_paths": shard_paths, "plan": plan,
            "admission": adm, "source_tree": tree}


def _screen_shard(shard: dict, registry: EvalRegistry) -> dict:
    """Check a built shard's documents against the held-out registry."""
    hits = [s["doc_id"] for s in shard["spans"] if s["doc_id"] in registry.entries]
    if hits:
        return {"contamination_status": "OVERLAP",
                "eval_overlap_status": "OVERLAP",
                "detail": {"overlapping_doc_ids": sorted(hits)[:10]}}
    return {"contamination_status": "CLEAN", "eval_overlap_status": "CLEAN",
            "detail": {"checked_against": len(registry.entries)}}


def _build_probe(cfg: RunConfig, adm: dict, tok, thash: str, out_dir: str,
                 registry: EvalRegistry) -> ValidationProbe:
    """Tokenize validation documents into probe samples.

    They go through the *same* shard writer, but into a directory outside the
    training shard set, and ``never_train`` is cleared only for this local copy
    so the writer will accept them. The firewall registry still holds their ids,
    so any attempt to put them on a gradient path is caught by
    ``assert_not_gradient_bearing``.
    """
    vdocs = [dict(d, never_train=False) for d in adm["heldout"]["validation"]]
    if not vdocs:
        return ValidationProbe([], registry=registry)
    vdir = os.path.join(out_dir, "probe_shards")
    vshards = build_shards(vdocs, tok, vocab_size=tok.vocab_size, lane="validation",
                           target_tokens=10 ** 9, out_dir=vdir, tokenizer_hash=thash)
    vtoks = {s["shard_id"]: load_shard_tokens(s["path"], tok.vocab_size) for s in vshards}
    vitems = [dict(sp, shard_id=s["shard_id"]) for s in vshards for sp in s["spans"]]
    seq_len = cfg.profile.seq_len_early

    # Pack per capability lane, not across all validation text at once.
    #
    # The validation split carries the lane each document came from, and the
    # learning ledger's "loss delta before and after exposure" is only meaningful
    # per lane -- an aggregate cannot say that exposure to code helped code.
    # Packing everything together with concat_chop merges lanes into one
    # sequence, which is what collapsed `by_lane` to a single `validation` key.
    lane_of = {d["doc_id"]: d.get("lane", "validation") for d in vdocs}
    by_lane: dict[str, list[dict]] = {}
    for it in vitems:
        by_lane.setdefault(lane_of.get(it.get("doc_id"), "validation"), []).append(it)

    samples, i = [], 0
    for lane in sorted(by_lane):
        # A couple of sequences per lane keeps the probe cheap; it runs at every
        # checkpoint boundary and is forward-only.
        for q in pack(by_lane[lane], seq_len, "concat_chop")[:3]:
            samples.append(build_sample(
                q, vtoks, seq_len=seq_len,
                attention_policy=cfg.attention_policy,
                position_policy=cfg.position_policy_early,
                lane=lane, sample_index=i))
            i += 1
    return ValidationProbe(samples, registry=registry)
