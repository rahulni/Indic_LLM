# -*- coding: utf-8 -*-
"""Authored trace data for the agentic and reasoning lanes.

These are written by hand for this demo. They are NOT harvested data and are
never presented as such -- ``corpus/CORPUS.md`` says so plainly.

They exist because the two lanes they feed are the only ones whose *structure*
is load-bearing:

  * a reasoning trace has a prompt that must be masked out of the loss and a
    response that must not be;
  * an agentic trace additionally has tool observations, which the model reads
    but must never be trained to predict -- otherwise it learns to hallucinate
    environment responses instead of calling the tool.

Each turn is ``(role, text)`` where role is one of ``user``, ``assistant``,
``tool_call``, ``tool_obs``. The loss-mask policy in ``tdes/masks.py`` keys off
exactly these roles, so a mistake here shows up as a failing mask test rather
than as a silently wrong training signal.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Reasoning lane: user question -> assistant chain-of-thought + answer.
# Loss on the assistant turn only.
# ---------------------------------------------------------------------------

REASONING_TRACES: list[list[tuple[str, str]]] = [
    [("user", "A train leaves Chennai at 06:40 and reaches Bengaluru at 12:55. It halts twice for 8 minutes each. What is the moving time?"),
     ("assistant", "Total elapsed time is from 06:40 to 12:55, which is 6 hours 15 minutes, or 375 minutes. Two halts of 8 minutes remove 16 minutes. Moving time is 375 - 16 = 359 minutes, that is 5 hours 59 minutes.")],

    [("user", "If a shard contains 1,048,576 tokens and each optimizer step consumes 384 tokens, how many steps does the shard cover?"),
     ("assistant", "Divide the shard size by the per-step consumption: 1048576 / 384 = 2730.67. A step cannot be fractional, so the shard covers 2730 complete steps and leaves 256 tokens, which carry over into the next shard rather than being discarded.")],

    [("user", "Why does initial cross-entropy loss equal the natural log of the vocabulary size?"),
     ("assistant", "At initialisation the model has no information, so it spreads probability uniformly across all V tokens, giving each one probability 1/V. Cross-entropy is the negative log of the probability assigned to the correct token, so it is -ln(1/V) = ln(V). For a 131072-token vocabulary that is 11.78. This makes step-zero loss a sanity check: if it is not close to ln(V), the loss function, the masking, or the label alignment is wrong.")],

    [("user", "A tokenizer produces 13 tokens per word for Telugu and 1.3 for English. A 1,000-word document in each language is packed into 4,096-token sequences. How many sequences does each need?"),
     ("assistant", "The Telugu document becomes 13,000 tokens, which needs ceil(13000/4096) = 4 sequences. The English document becomes 1,300 tokens, which fits inside a single sequence with 2,796 positions to spare. The same semantic content costs four times the sequences purely because of tokenizer fertility, which is why fertility is a training-budget decision and not a cosmetic one.")],

    [("user", "Is 391 prime?"),
     ("assistant", "Check divisors up to sqrt(391), which is about 19.8. It is odd, so 2 is out. Digit sum is 13, so not divisible by 3. It does not end in 0 or 5. 391/7 = 55.86, 391/11 = 35.5, 391/13 = 30.1, 391/17 = 23 exactly. So 391 = 17 x 23 and it is not prime.")],

    [("user", "Two runs start from the same checkpoint. One replays the recorded data stream, the other regenerates it from a seed. Which comparison is trustworthy?"),
     ("assistant", "The replay. Regenerating from a seed reproduces the order only if the worker count, shard set, library versions and process state are all identical, and none of those are guaranteed across a restart. If the data differs, an observed improvement cannot be attributed to the change under test rather than to different data. Replaying the recorded stream fixes the data exactly, so the only remaining difference is the one being tested.")],

    [("user", "A model averages 2.3 loss at mid-training. A new shard scores 1.2. Should it be trained on?"),
     ("assistant", "No, not at this stage. A shard that is easier than the running average tells you the model already predicts it well, so the gradient it produces is small and the compute spent on it buys little. The useful move is to record the measurement and schedule that shard earlier in a future run, where it would have been informative, rather than discarding it outright.")],

    [("user", "What is 17% of 4,096?"),
     ("assistant", "10% of 4096 is 409.6 and 7% is 286.72. Adding them gives 696.32. So 17% of 4096 is 696.32, or 696 whole tokens if the quantity has to be an integer.")],

    [("user", "Why is padding worse than merely wasteful?"),
     ("assistant", "Padding is wasteful because the GPU computes over positions that carry no information. It is worse than wasteful because if pad positions reach the loss, the model learns that predicting a pad after a pad is correct, which is trivially easy. The average loss then falls for a reason unrelated to capability, so the loss curve stops being a measurement of learning and starts being a measurement of how much padding is present."),],

    [("user", "A best-fit packer reduces sequences from 16 to 5 for the same documents. What is the utilization gain?"),
     ("assistant", "If every sequence has the same fixed length, the total compute is proportional to the number of sequences. Going from 16 to 5 is a reduction of 11/16, about 68.75%. Put the other way, the same documents now occupy 31.25% of the compute they did under pad-only packing, and the tokens freed are available for real data.")],

    [("user", "Explain why an epoch is not a useful recovery coordinate for LLM pretraining."),
     ("assistant", "In pretraining the corpus is large enough relative to the compute budget that most data is seen roughly once, so a run is effectively single-epoch. Saying 'resume at epoch 3' identifies nothing. What identifies a position is the offset into the recorded stream of consumed batches, because that is unique, monotonic and independent of how many times the corpus happens to be traversed.")],

    [("user", "A candidate pool is filtered at a fixed 25% acceptance rate. Early batches score 0.9 on average; later ones score 0.1. What does that mean?"),
     ("assistant", "Because the acceptance rate is fixed, the selector always returns 25% and can never report that the pool is empty. Only the absolute scores can. A mean falling from 0.9 to 0.1 means the selector is now choosing the best of a bad set rather than a good set, so the pool is exhausted. Watching the accepted count would have shown nothing; watching the score distribution shows it immediately.")],
]


# ---------------------------------------------------------------------------
# Agentic lane: user request -> tool calls -> tool observations -> answer.
# Loss on assistant and tool_call turns; user and tool_obs are context only.
# ---------------------------------------------------------------------------

AGENTIC_TRACES: list[list[tuple[str, str]]] = [
    [("user", "How many shards in the manifest directory are tagged as the indic lane?"),
     ("tool_call", "grep -l '\"capability_lane\": \"indic\"' submission_artifacts/manifests/*.json | wc -l"),
     ("tool_obs", "7"),
     ("assistant", "Seven shard manifests carry the indic capability lane.")],

    [("user", "Check whether the frozen tokenizer still matches its recorded hash."),
     ("tool_call", "sha256sum corpus/tokenizer/tokenizer.json"),
     ("tool_obs", "3f1c9e0a52b7d4886ac1f0e9b2d7143c5a8e6b04f9c2d1e7a3b5c8d0f2e4a6b8  corpus/tokenizer/tokenizer.json"),
     ("tool_call", "cat corpus/tokenizer/tokenizer.sha256"),
     ("tool_obs", "3f1c9e0a52b7d4886ac1f0e9b2d7143c5a8e6b04f9c2d1e7a3b5c8d0f2e4a6b8"),
     ("assistant", "The digests match, so the tokenizer has not changed since it was frozen. Token ids produced now mean the same thing they meant when the shards were written.")],

    [("user", "The run crashed at step 68. Find the last checkpoint and its ledger offset."),
     ("tool_call", "ls submission_artifacts/checkpoints/"),
     ("tool_obs", "ckpt_step_00060.json  ckpt_step_00050.json  ckpt_step_00040.json"),
     ("tool_call", "python -c \"import json;d=json.load(open('submission_artifacts/checkpoints/ckpt_step_00060.json'));print(d['global_step'], d['ledger_offset'])\""),
     ("tool_obs", "60 360"),
     ("assistant", "The newest checkpoint is at global step 60 with ledger offset 360. Resuming truncates any ledger records past offset 360 and serves record 361 next, so the eight steps completed after the checkpoint are re-run rather than skipped.")],

    [("user", "Did any evaluation document reach a training batch?"),
     ("tool_call", "python tools/audit_firewall.py --check-eval-leak"),
     ("tool_obs", "scanned 612 packed samples; eval canaries found: 0; blocked at admission: 3"),
     ("assistant", "No evaluation content reached a loss-bearing batch. Three candidate documents were blocked at admission, which is the firewall working as intended rather than a failure.")],

    [("user", "What fraction of the last batch was padding?"),
     ("tool_call", "python tools/inspect_batch.py --step 42 --field pad_fraction"),
     ("tool_obs", "{\"step\": 42, \"total_positions\": 384, \"pad_positions\": 27, \"pad_fraction\": 0.0703}"),
     ("assistant", "About 7% of the positions in step 42 were padding, so roughly 93% of the compute went to real tokens.")],

    [("user", "Find which lane fell furthest below its planned share."),
     ("tool_call", "python tools/mixture_report.py --sort-by drift --top 3"),
     ("tool_obs", "agentic  planned 0.030  actual 0.021  drift -0.009\nlongctx  planned 0.030  actual 0.026  drift -0.004\nindic    planned 0.080  actual 0.079  drift -0.001"),
     ("assistant", "The agentic lane is furthest below plan, at 2.1% against a planned 3.0%. Its protected floor is 1.0%, so it is still above the floor and no override was required, but the shortfall is worth tracing to whether agentic shards were exhausted or were being rejected by the selector.")],

    [("user", "Show the highest-perplexity tokens in the Telugu shard."),
     ("tool_call", "python tools/token_trace.py --shard indic_te_0001 --top 5 --by perplexity"),
     ("tool_obs", "pos 41  ppl 84.2  'ష'\npos 12  ppl 71.9  'క్ష'\npos 88  ppl 66.4  'ఞ'\npos 57  ppl 51.0  'ృ'\npos 3   ppl 44.8  'ౌ'"),
     ("assistant", "The difficulty concentrates on conjuncts and vowel signs rather than on whole words. That points at the tokenizer rather than the corpus: at this vocabulary size those clusters are being split into fragments the model has to reassemble, which is exactly the fertility cost showing up as perplexity.")],

    [("user", "Verify the replay reproduced the original batches."),
     ("tool_call", "python tools/compare_runs.py --original submission_artifacts/ledgers/consumption.jsonl --replay submission_artifacts/ledgers/replay.jsonl --range 20:40"),
     ("tool_obs", "compared 20 batches; batch_id mismatches: 0; content_hash mismatches: 0; span mismatches: 0"),
     ("assistant", "All twenty replayed batches match the originals on batch id, content hash and token spans. The data stream is reproducible over that interval.")],

    [("user", "How much did padding cost in rupees this run?"),
     ("tool_call", "python -c \"import json;d=json.load(open('submission_artifacts/cost_report.json'));print(d['padding'])\""),
     ("tool_obs", "{\"pad_positions\": 4812, \"total_positions\": 46080, \"pad_fraction\": 0.1044, \"inr_wasted\": 0.42}"),
     ("assistant", "About 10.4% of positions were padding, costing roughly 0.42 rupees at demo scale. The number is trivial here, but the same fraction on a run billed at 2,700 rupees an hour for several weeks is the difference between packing policies being a detail and being a budget line.")],

    [("user", "List the shards that OPUS rejected for stage mismatch."),
     ("tool_call", "python tools/opus_report.py --reason stage_mismatch"),
     ("tool_obs", "longctx_0002  stage A  seq_len 64  doc_len 311\nlongctx_0004  stage A  seq_len 64  doc_len 402"),
     ("assistant", "Two long-context shards were rejected in stage A because their documents exceed the stage's 64-token sequence length. That is the reservation policy working: those documents are held for the later rung where the sequence length can hold them whole, rather than being chopped now.")],
]
