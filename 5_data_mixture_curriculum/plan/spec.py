# -*- coding: utf-8 -*-
"""
spec.py - the declarative mixture-and-curriculum specification for Drishtikon-40B (V5).

Nothing here is prose. Every number in MIXTURE_PLAN.md is derived from this file
by audit.py and rendered by build_plan.py. If a number is not in this file or
computed from it, it does not appear in the plan.

Provenance tags on every supply figure:
  MEASURED  - produced by our own session-4 cleaning pipeline (results.json)
  PUBLISHED - stated on a dataset card or in a paper, cited by URL
  DERIVED   - published sample count x a stated mean-tokens-per-sample assumption
  ESTIMATED - our own estimate, flagged as such and never load-bearing alone
"""

# ---------------------------------------------------------------------------
# 0. Budget
# ---------------------------------------------------------------------------

# Session 3 (3_40b_model/plan-notes.md) locked 14T for a 40B dense design memo.
# Session 5 asked the question directly and the answer was "between 2.4 to 4
# trillion tokens" for what V5 actually trains on. Those are different numbers
# for different things, so the budget is a parameter and the plan is audited
# across the whole range rather than asserting one of them.
#
# 3.0T is primary: the midpoint of the range the session actually named.
BUDGET_TOKENS = 3_000_000_000_000
MODEL_PARAMS = 40_000_000_000

BUDGET_SCENARIOS = [
    (2.4e12, "low end of the range named in session 5"),
    (3.0e12, "PRIMARY - midpoint of the session-5 range"),
    (4.0e12, "high end of the range named in session 5"),
    (14.0e12, "session-3 design memo for a 40B dense model"),
]

# Post-training budgets sit OUTSIDE the 14T (session 3), restated so the
# anneal is not confused with SFT.
POST_TRAINING = {"sft": 25e9, "dpo": 10e9, "rl_rollouts": 20e9}

# Repetition ceiling. Muennighoff et al., "Scaling Data-Constrained Language
# Models" (NeurIPS 2023, arXiv:2305.16264). 4.0 is our hard physical cap and no
# lane may plan past it.
EPOCH_CAP_DEFAULT = 4.0

# [REVIEW FIX 7] The 4-epoch "cliff" is a summary, not the result. The paper
# fits a smooth decay in which every repetition past the first is worth less
# than fresh data:
#
#     D' = U_D + U_D * R_D* * (1 - exp(-R_D / R_D*)),   R_D = epochs - 1
#
# An earlier draft cited this paper as its ceiling authority while using a
# cruder rule than the paper supplies - repetition below 4 epochs was priced as
# free. It is not. At the 4.0 cap the decay delivers ~3.73 effective epochs, so
# the last epoch of any capped lane is worth about a quarter of a fresh one.
#
# This does not change physical feasibility (you still consume epoch_cap passes
# of real tokens); it changes the VALUE those tokens deliver, which is reported
# separately so the discount is visible rather than assumed away.
EPOCH_DECAY = dict(
    r_star_data=15.387,
    citation="Muennighoff et al., Scaling Data-Constrained Language Models, "
             "NeurIPS 2023 (arXiv:2305.16264), eq. 6",
    confidence="R_D* is quoted from the paper's fitted parameters. Corroborated "
               "by the paper's own prose gloss that R* is roughly the half-life "
               "of epochs, with repeated tokens retaining value out to ~15 "
               "repetitions. Worth re-checking against the published table "
               "before this number is used to justify raising any epoch cap.",
)

# ---------------------------------------------------------------------------
# 0b. Tokenizer units. [REVIEW FIX 3]
#
# The first draft of this plan added token counts that were produced by at
# least four different tokenizers - FineWeb-Edu under GPT-2, DCLM and
# Nemotron-CC under GPT-NeoX, The Stack v2 under StarCoder2, Sangraha under an
# Indic sentencepiece, our own pipeline under cl100k - and then separately
# claimed a 1.33x credit for our Indic-efficient tokenizer. That double-counts:
# you cannot both hold supply in someone else's tokens and bank the efficiency
# gain on top.
#
# The fix is to make WORDS the invariant unit. Every published token count is
# divided by its own tokenizer's fertility to recover words, then multiplied by
# ours exactly once. Epoch counts are only meaningful when supply and demand are
# in the same units, and the 14T budget is denominated in OUR tokens.
#
# Fertility is tokens per word. The two MEASURED rows come from our own session-4
# pipeline (4_model_data/assignment/analysis_tokenizer_fertility.json).
# ---------------------------------------------------------------------------

TOKENIZER_FERTILITY = {
    # tokenizer  ->  content class -> tokens per word
    "ours": {          # Drishtikon-180K, BrahmicTokenizer-131K derived (session 3)
        "english_prose": 1.30, "code": 2.20, "indic": 1.683,
        "multilingual": 1.45, "math_prose": 1.45,
    },
    "cl100k": {
        "english_prose": 1.563,   # MEASURED, session-4 reasoning_sft corpus
        "indic": 13.268,          # MEASURED, session-4 telugu_web corpus
        "code": 2.60, "multilingual": 2.40, "math_prose": 1.70,
    },
    "muril": {"indic": 2.296},    # MEASURED, session-4 telugu_web corpus
    "gpt2": {"english_prose": 1.60, "code": 2.80, "math_prose": 1.75},
    "neox": {"english_prose": 1.55, "code": 2.55, "math_prose": 1.70,
             "multilingual": 2.10, "indic": 7.50},
    "starcoder2": {"code": 2.40, "english_prose": 1.58},
    "indic_sp": {"indic": 2.30},  # assumed Indic sentencepiece for Sangraha's counts
}

# Provenance of each fertility figure, so which conversions
# rest on measurement and which on assumption.
FERTILITY_PROVENANCE = {
    ("cl100k", "english_prose"): "MEASURED (session 4, reasoning_sft: 1.563 tok/word over 4,867,426 words)",
    ("cl100k", "indic"): "MEASURED (session 4, telugu_web: 13.268 tok/word over 3,480,047 words)",
    ("muril", "indic"): "MEASURED (session 4, telugu_web: 2.296 tok/word, 5.78x better than cl100k)",
    ("ours", "indic"): "DERIVED (MuRIL's measured 2.296 x BrahmicTokenizer-131K's reported 26.7% reduction)",
    ("ours", "english_prose"): "DERIVED (o200k-parity claim in BrahmicTokenizer-131K, conservatively rounded)",
    ("indic_sp", "indic"): "ESTIMATED - the single most load-bearing assumption in the conversion; see the sensitivity note in the plan",
}

# Which tokenizer and content class each source's published count is in.
# Defaults by lane, overridden per source where they differ.
LANE_UNITS_DEFAULT = {
    "web":       ("neox", "english_prose"),
    "code":      ("starcoder2", "code"),
    "reasoning": ("neox", "math_prose"),
    "indic":     ("indic_sp", "indic"),
    "multiling": ("neox", "multilingual"),
    "longctx":   ("neox", "english_prose"),
    "civic":     ("cl100k", "english_prose"),
    "agentic":   ("cl100k", "code"),
    "parallel":  ("indic_sp", "indic"),
}

# Anything we DERIVED or ESTIMATED ourselves is already in our own units - we
# invented the mean-tokens-per-sample, so we get to declare the tokenizer. Only
# PUBLISHED counts need converting.
SELF_UNIT_PROVENANCE = ("DERIVED", "ESTIMATED", "MEASURED")

# ---------------------------------------------------------------------------
# 1. Phase schedule. Weights are fractions of the 14T budget and must sum to 1.
# ---------------------------------------------------------------------------

PHASES = [
    ("A", "Foundation",     0.40, "Breadth. Model cannot yet use premium data, so premium data is not spent on it."),
    ("B", "Consolidation",  0.32, "Code and reasoning ramp; long-context packing begins."),
    ("C", "Specialisation", 0.25, "Scarce lanes concentrate; sequence length reaches 128k."),
    ("D", "Anneal",         0.03, "The reserve. Verified tiers only, highest difficulty, highest reasoning bands."),
]

# ---------------------------------------------------------------------------
# 2. Capability lanes. Per-phase share of that phase's tokens, in percent.
#    Each phase column MUST sum to 100. Whole-run shares are DERIVED from these
#    by weighted average - they are an output, not an input.
# ---------------------------------------------------------------------------

# The Phase D column was revised after [REVIEW FIX 2] added a supply audit to
# the anneal reserve. The first draft asked for 12% Indic and 10% long-context
# in the cooldown; under the reserve's own eligibility rules (Sangraha Verified
# only, one epoch, dependency-scored long documents only) those tiers could
# supply 47.1B and 35.8B against 50.4B and 42.0B of demand - a 9.5B shortfall
# that the plan had declared but never checked. Indic drops to 11%, long-context
# to 8%, and the freed 3% goes to code and web, which have eligible headroom.
LANES = {
    #  id                     label                            A     B     C     D
    "web":       dict(label="General web & knowledge",  phases=(68.0, 52.0, 32.0, 13.0)),
    "code":      dict(label="Code",                     phases=(16.0, 25.0, 34.0, 22.0)),
    "reasoning": dict(label="Reasoning & math",         phases=( 3.0,  7.0, 13.0, 26.0)),
    "indic":     dict(label="Indic (22 languages)",     phases=( 4.0,  5.0,  6.0, 11.0)),
    "multiling": dict(label="World multilingual",       phases=( 5.0,  4.0,  2.0,  0.0)),
    "longctx":   dict(label="Long-context native",      phases=( 1.0,  3.0,  5.0,  8.0)),
    "civic":     dict(label="India civic & legal",      phases=( 1.5,  2.0,  3.0,  4.0)),
    "agentic":   dict(label="Agentic / tool-use",       phases=( 0.5,  1.0,  4.0, 16.0)),
    "parallel":  dict(label="Parallel / cross-lingual", phases=( 1.0,  1.0,  1.0,  0.0)),
}

# Fraction of each lane's tokens that actually carry loss. The rest is context
# the model reads but is never punished for failing to predict: the issue text
# in a SWE-bench sample, the 4,000-line stack trace a tool returns, the user
# turn in a trajectory. We pay compute for masked tokens, so the budget is
# denominated in raw tokens - but capability comes from the loss-bearing share,
# and the two must be reported side by side or the agentic lane looks 3x bigger
# than it is.
LOSS_BEARING = {
    "web": 1.00, "code": 1.00, "reasoning": 0.85, "indic": 1.00,
    "multiling": 1.00, "longctx": 1.00, "civic": 1.00,
    "agentic": 0.30, "parallel": 1.00,
}

# ---------------------------------------------------------------------------
# 3. Inventory. Real supply, per source.
#    dedup_keep - fraction surviving cross-corpus dedup against the rest of the
#                 lane. Web sources are all Common Crawl derivatives and overlap
#                 heavily; Sangraha's three tiers are disjoint by construction.
#    epoch_cap  - how many times we are willing to repeat THIS source.
# ---------------------------------------------------------------------------

INVENTORY = [
    # -- web ---------------------------------------------------------------
    dict(lane="web", tier=None, name="Nemotron-CC", samples=None, tokens=6.30e12,
         provenance="PUBLISHED", dedup_keep=0.65, epoch_cap=2.0,
         url="https://arxiv.org/abs/2412.02595",
         note="Common Crawl derivative; overlaps DCLM and FineWeb heavily."),
    dict(lane="web", tier=None, name="DCLM-Baseline", samples=None, tokens=4.00e12,
         provenance="PUBLISHED", dedup_keep=0.65, epoch_cap=2.0,
         url="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0",
         note="Common Crawl derivative."),
    dict(lane="web", tier=None, name="FineWeb-Edu", samples=None, tokens=1.30e12,
         provenance="PUBLISHED", dedup_keep=0.65, epoch_cap=4.0, src_tokenizer="gpt2",
         url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
         note="Highest-quality web tier; the only web source eligible for the anneal."),

    # -- code --------------------------------------------------------------
    dict(lane="code", tier=None, name="The Stack v2 (dedup)", samples=600e6, tokens=900e9,
         provenance="PUBLISHED", dedup_keep=1.00, epoch_cap=4.0,
         url="https://huggingface.co/datasets/bigcode/the-stack-v2",
         note="Already deduplicated at source. 600M files / ~900B tokens per the session-5 inventory widget."),
    dict(lane="code", tier=None, name="Repo-level packs (India-Stack SDKs, ONDC, UPI)", samples=None, tokens=8e9,
         provenance="ESTIMATED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://github.com/ONDC-Official",
         note="Session-3 novel source. Estimate; not load-bearing - 0.3% of the lane."),

    # -- reasoning & math --------------------------------------------------
    dict(lane="reasoning", tier=None, name="arXiv full-text", samples=None, tokens=28e9,
         provenance="PUBLISHED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://huggingface.co/datasets/EleutherAI/proof-pile-2", note=""),
    dict(lane="reasoning", tier=None, name="FineMath (3+)", samples=None, tokens=34e9,
         provenance="PUBLISHED", dedup_keep=0.85, epoch_cap=4.0,
         url="https://huggingface.co/datasets/HuggingFaceTB/finemath", note=""),
    dict(lane="reasoning", tier=None, name="OpenWebMath", samples=None, tokens=14.7e9,
         provenance="PUBLISHED", dedup_keep=0.70, epoch_cap=4.0,
         url="https://huggingface.co/datasets/open-web-math/open-web-math",
         note="Overlaps FineMath and DCLM; heaviest dedup haircut in the lane."),
    dict(lane="reasoning", tier=None, name="AlgebraicStack", samples=None, tokens=11e9,
         provenance="PUBLISHED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://huggingface.co/datasets/EleutherAI/proof-pile-2", note=""),
    dict(lane="reasoning", tier=None, name="OpenThoughts2 (long CoT)", samples=1.0e6, tokens=4.0e9,
         provenance="DERIVED", dedup_keep=0.95, epoch_cap=4.0,
         url="https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M",
         note="DERIVED: 1.0M samples x 4,000 tok/sample assumed mean."),
    dict(lane="reasoning", tier=None, name="OpenR1-Math-220k", samples=220e3, tokens=1.32e9,
         provenance="DERIVED", dedup_keep=0.95, epoch_cap=4.0,
         url="https://huggingface.co/datasets/open-r1/OpenR1-Math-220k",
         note="DERIVED: 220k samples x 6,000 tok/sample assumed mean."),

    # -- indic: the four tiers --------------------------------------------
    dict(lane="indic", tier="verified", name="Sangraha Verified (22 langs)", samples=None, tokens=64.3061e9,
         provenance="PUBLISHED", dedup_keep=1.00, epoch_cap=4.0,
         url="https://huggingface.co/datasets/ai4bharat/sangraha",
         note="Human-verified web + OCR from Indic PDFs + ASR from video/podcast. The only Indic tier eligible for the anneal."),
    dict(lane="indic", tier="unverified", name="Sangraha Unverified (14 langs)", samples=None, tokens=24.3077e9,
         provenance="PUBLISHED", dedup_keep=1.00, epoch_cap=2.0,
         url="https://huggingface.co/datasets/ai4bharat/sangraha",
         note="Perplexity-filtered from existing multilingual corpora. Epoch cap halved: this is the tier our own pipeline measured, and it lost 8.0% at the quality filter."),
    dict(lane="indic", tier="unverified", name="IndicCorp v2", samples=None, tokens=20.9e9,
         provenance="PUBLISHED", dedup_keep=0.50, epoch_cap=3.0,
         url="https://huggingface.co/datasets/ai4bharat/IndicCorpV2",
         note="Same crawl lineage as Sangraha; 50% dedup haircut applied."),
    dict(lane="indic", tier="translated", name="Sangraha Synthetic (MT + romanised, 14 langs)", samples=None, tokens=162.7079e9,
         provenance="PUBLISHED", dedup_keep=1.00, epoch_cap=1.5,
         url="https://huggingface.co/datasets/ai4bharat/sangraha",
         note="AI4Bharat labels this 'synthetic', but it is WikiMedia English machine-translated into 14 languages and then transliterated. That is the TRANSLATED tier, not the synthetic one. Epoch cap 1.5 because translationese compounds under repetition."),

    # -- world multilingual ------------------------------------------------
    dict(lane="multiling", tier=None, name="CulturaX (167 langs, non-Indic slice)", samples=None, tokens=1.20e12,
         provenance="ESTIMATED", dedup_keep=0.70, epoch_cap=2.0,
         url="https://huggingface.co/datasets/uonlp/CulturaX",
         note="ESTIMATED: CulturaX is 6.3T total; non-Indic non-English slice taken at ~19%."),

    # -- long-context native ----------------------------------------------
    dict(lane="longctx", tier=None, name="Repo-level concatenated packs (from Stack v2)", samples=None, tokens=90e9,
         provenance="ESTIMATED", dedup_keep=1.00, epoch_cap=3.0,
         url="https://huggingface.co/datasets/bigcode/the-stack-v2",
         note="ESTIMATED. Budget-neutral: these tokens are re-packed Stack v2 and are charged to THIS lane, not to code, so they are never double-counted."),
    dict(lane="longctx", tier=None, name="Books / PG-19 / public-domain long-form", samples=None, tokens=12e9,
         provenance="PUBLISHED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://huggingface.co/datasets/deepmind/pg19", note=""),
    dict(lane="longctx", tier=None, name="ProLong long-context corpus", samples=None, tokens=40e9,
         provenance="PUBLISHED", dedup_keep=0.85, epoch_cap=3.0,
         url="https://arxiv.org/abs/2410.02660",
         note="~40B tokens of curated long-dependency continued-pretraining data."),

    # -- india civic & legal ----------------------------------------------
    dict(lane="civic", tier=None, name="Indian court judgments (SC + 25 HCs)", samples=10e6, tokens=30e9,
         provenance="DERIVED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://www.indiankanoon.org/",
         note="DERIVED: ~10M judgments x 3,000 tok/judgment assumed mean."),
    dict(lane="civic", tier=None, name="Panchayat-to-Parliament records (session-3 corpus)", samples=None, tokens=12e9,
         provenance="ESTIMATED", dedup_keep=0.95, epoch_cap=4.0, url="",
         note="ESTIMATED. Session-3 novel corpus; not yet built, so it is counted as supply only at the ESTIMATED tag and the shortfall it creates is carried as manufactured."),

    # -- agentic / tool-use -----------------------------------------------
    dict(lane="agentic", tier=None, name="Nemotron-SWE-v1 (OpenHands trajectories)", samples=59e3, tokens=1.77e9,
         provenance="DERIVED", dedup_keep=0.80, epoch_cap=4.0,
         url="https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1",
         note="DERIVED: 59k trajectories x 30,000 tok/trajectory assumed mean. Issues sourced from SWE-Gym and R2E-Gym, hence the 0.80 dedup keep."),
    dict(lane="agentic", tier=None, name="SWE-smith", samples=50_137, tokens=1.00e9,
         provenance="DERIVED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://huggingface.co/datasets/SWE-bench/SWE-smith",
         note="DERIVED: 50,137 task instances from 128 repos x 20,000 tok assumed mean."),
    dict(lane="agentic", tier=None, name="AgentInstruct", samples=1.1e6, tokens=1.65e9,
         provenance="DERIVED", dedup_keep=0.85, epoch_cap=4.0,
         url="https://arxiv.org/abs/2407.03502",
         note="DERIVED: 1.1M pairs x 1,500 tok assumed mean. Short single-step samples, not long-horizon trajectories."),
    dict(lane="agentic", tier=None, name="ToolBench", samples=120e3, tokens=0.080e9,
         provenance="PUBLISHED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://github.com/OpenBMB/ToolBench",
         note="120k samples but only ~80M tokens - the session-5 inventory's own warning about sizing in samples instead of tokens."),
    dict(lane="agentic", tier=None, name="xLAM function-calling-60k", samples=60e3, tokens=0.060e9,
         provenance="DERIVED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
         note="DERIVED: 60k x 1,000 tok."),
    dict(lane="agentic", tier=None, name="Glaive function-calling v2", samples=113e3, tokens=0.113e9,
         provenance="DERIVED", dedup_keep=0.85, epoch_cap=4.0,
         url="https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2",
         note="DERIVED: 113k x 1,000 tok."),
    dict(lane="agentic", tier=None, name="SWE-Gym", samples=2_400, tokens=0.060e9,
         provenance="DERIVED", dedup_keep=0.50, epoch_cap=4.0,
         url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
         note="DERIVED: 2.4k tasks x 25,000 tok. Half discarded as a subset of Nemotron-SWE's issue pool."),

    # -- parallel / cross-lingual -----------------------------------------
    dict(lane="parallel", tier=None, name="BPCC (Bharat Parallel Corpus Collection)", samples=230e6, tokens=18.4e9,
         provenance="DERIVED", dedup_keep=0.90, epoch_cap=4.0,
         url="https://huggingface.co/datasets/ai4bharat/BPCC",
         note="DERIVED: 230M sentence pairs x ~40 tok/side x 2 sides."),
    dict(lane="parallel", tier=None, name="Samanantar", samples=49.7e6, tokens=4.0e9,
         provenance="DERIVED", dedup_keep=0.40, epoch_cap=4.0,
         url="https://huggingface.co/datasets/ai4bharat/samanantar",
         note="DERIVED: 49.7M pairs x ~40 tok x 2. Largely absorbed into BPCC - 60% haircut."),
]

# ---------------------------------------------------------------------------
# 4. Protected always-on floor.
#    Minimum share of EVERY 1,000-step window, enforced before the online
#    selector runs. Rationale (session 5): the OPUS-style selector scores a
#    sample on its first ~512 tokens against a benchmark-gradient proxy. An
#    agentic trajectory opens with a masked issue body and a tool log, and an
#    Indic document scores against benchmarks that are overwhelmingly English.
#    Both get discarded for reasons that have nothing to do with their value.
#    The floor is the answer to a known selector pathology, not a preference.
# ---------------------------------------------------------------------------

FLOOR = {
    "indic":     dict(pct=3.00, note="of which Sangraha Verified >= 1.50 pct; unverified may never substitute for the verified half"),
    "longctx":   dict(pct=1.50, note="sequences >= 32k only; shorter packs do not count toward this floor"),
    "reasoning": dict(pct=1.00, note="long-CoT traces only, not the math corpus"),
    "agentic":   dict(pct=1.00, note="multi-step trajectories only; single-turn function-call pairs do not count"),
    "civic":     dict(pct=0.50, note="India-context legal/administrative"),
}

# The selector may freely allocate the remainder. Keep-fraction is the share of
# candidate samples OPUS retains from the unprotected pool.
#
# [REVIEW FIX 4] keep_fraction is not a cosmetic detail - it is a multiplier on
# every supply requirement in the plan, and an earlier draft used it only in
# prose. Session 5, line 463: "if we keep the 50% fraction ... if you collect one
# token actually only half will be trained on", and line 465 confirms discarded
# batches are thrown away rather than re-offered. So the corpus has to cover
# CANDIDATE tokens, not trained tokens.
#
# The multiplier is per-lane, not global: data admitted under the protected
# floor bypasses the selector entirely and is consumed 1:1, while everything
# above the floor is drawn at 1/keep_fraction. A lane that is mostly floor is
# barely affected; a lane with no floor pays the full 2x.
SELECTOR = dict(keep_fraction=0.50, scored_prefix_tokens=512, candidate_batch=1024,
                refresh_every_steps=1000,
                claimed_efficiency="8x (the OPUS paper's claim, quoted in session 5: "
                                   "the loss reached at 20B tokens would otherwise "
                                   "have taken 160B)")

# ---------------------------------------------------------------------------
# 5. Anneal reserve (Phase D). Composition in percent of the reserve.
#    Must sum to 100.
# ---------------------------------------------------------------------------

# [REVIEW FIX 2] Eligibility yield: the fraction of a lane's supply that
# actually survives the reserve's own admission rules. The first draft declared
# these rules and never checked whether enough data passes them - the reserve
# was the one section where the quality bar was highest and the supply audit was
# absent. audit.anneal_supply_audit() now enforces it, and the reserve is held
# to ONE epoch: repeating data during a cooldown defeats the purpose of a
# cooldown.
ANNEAL_YIELD = {
    "web":       dict(frac=0.10, basis="top-decile FineWeb-Edu classifier score only"),
    "code":      dict(frac=0.15, basis="repos with a runnable test suite that goes green in the harness; most of Stack v2 has no executable tests"),
    "reasoning": dict(frac=0.35, basis="items with a checkable final answer that the generator got right"),
    "indic":     dict(frac=1.00, basis="Sangraha Verified only - the tier restriction IS the filter, so yield is 1.0 of that tier and 0.0 of every other"),
    "longctx":   dict(frac=0.25, basis="documents with measured long-range dependency, not merely long"),
    "civic":     dict(frac=0.40, basis="judgments and proceedings clearing the licence gate"),
    "agentic":   dict(frac=0.60, basis="trajectories with a positive verifier reward; manufactured rollouts are verifier-gated at generation so they clear at a higher rate than scraped ones"),
    "multiling": dict(frac=0.10, basis="not admitted - Phase D share is 0"),
    "parallel":  dict(frac=0.10, basis="not admitted - Phase D share is 0"),
}
ANNEAL_MAX_EPOCHS = 1.0

# Order in which the Indic lane is filled. Real human-authored text is drawn
# before machine translation, and synthetic is the last resort rather than a
# design goal. At the 3T budget this ordering means the translated tier is
# barely touched: we do not need machine-translated Wikipedia when verified and
# unverified real text can cover the lane inside their epoch caps. At 14T the
# ordering inverts the conclusion, which is exactly what the budget sweep shows.
INDIC_TIER_PRIORITY = ["verified", "unverified", "translated", "synthetic"]

ANNEAL_ELIGIBILITY = [
    "Sangraha Verified only. Unverified and translated Indic tiers are barred from the reserve.",
    "Code samples must have passed execution in the harness (tests green), not merely parsed.",
    "Agentic trajectories must carry a positive verifier reward; failed-and-unrecovered trajectories are barred.",
    "Synthetic data of any kind must have passed a verifier (execution, proof-check, or citation-grounding). Unverified synthetic is barred.",
    "Every document must clear 12-gram decontamination against all evaluation suites, re-run at reserve-build time rather than inherited.",
]

# ---------------------------------------------------------------------------
# 6. Difficulty bands and reasoning-length bands.
# ---------------------------------------------------------------------------

DIFFICULTY_BANDS = [
    dict(id="D1", label="Foundational", phase_mix=(55, 35, 18,  5),
         example_source="MBPP task 2",
         example="Write a function to find the shared elements from two given lists.",
         why="Single concept, single function, no state. Present in bulk early and never removed entirely - it is what keeps the tokenizer and syntax priors sharp."),
    dict(id="D2", label="Standard", phase_mix=(35, 42, 37, 20),
         example_source="GSM8K train, item 1",
         example="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
         why="Two-to-four step chain, one arithmetic path, verifiable answer (72). This is the band that moves GSM8K and MBPP+."),
    dict(id="D3", label="Hard", phase_mix=( 9, 20, 33, 45),
         example_source="SWE-bench Verified, django__django-11099",
         example="ASCIIUsernameValidator and UnicodeUsernameValidator accept a username with a trailing newline, because r'^[\\w.@+-]+$' lets $ match before a final newline. Locate both validators across the repo, change the anchor to \\Z, keep the hidden tests green. (Django ticket #30257, PR #11099.)",
         why="Requires localisation across an unfamiliar repo before a one-line edit. This is the band SWE-bench Verified actually tests, and the band we have least of."),
    dict(id="D4", label="Frontier", phase_mix=( 1,  3, 12, 30),
         example_source="Terminal-Bench / SWE-bench Live-Pro class",
         example="On a bare container, install and configure a headless X server on display :99, bring up the GUI test suite, diagnose the failure the logs report, and make all 1,240 tests pass.",
         why="Multi-hour, multi-tool, requires recovery from a failed call. Almost none of this exists as public data - it is the lane we manufacture in the harness."),
]

# Reasoning-length bands. The control token is supplied in the prompt; the
# model is trained to answer WITHIN the band. Sachin's question in session 5 -
# why not let the model choose - is answered by the reward asymmetry: if depth
# is rewarded only through final correctness, the model always picks maximum
# depth. The band is an input, not an output.
LENGTH_BANDS = [
    dict(id="low", token_budget=(0, 128), control="<think:low>", share_of_reasoning_lane=40,
         example_q="What is 43 divided by 17?",
         example_trace="17x2=34, remainder 9; 9/17 is a bit over 0.5 -> ~2.53",
         example_a="approximately 2.53"),
    dict(id="medium", token_budget=(129, 1024), control="<think:medium>", share_of_reasoning_lane=35,
         example_q="How many integers between 1 and 1000 are divisible by 3 or 5?",
         example_trace="Inclusion-exclusion. floor(1000/3)=333, floor(1000/5)=200, floor(1000/15)=66. 333+200-66.",
         example_a="467"),
    dict(id="high", token_budget=(1025, 8192), control="<think:high>", share_of_reasoning_lane=20,
         example_q="How many 4-digit positive integers have digits summing to exactly 20?",
         example_trace="Stars and bars over d1 in 1..9 and d2,d3,d4 in 0..9 with sum 20, inclusion-exclusion on the upper bounds; enumerate the correction terms.",
         example_a="COMPUTED_AT_BUILD"),
    dict(id="ultra", token_budget=(8193, 65536), control="<think:ultra>", share_of_reasoning_lane=5,
         example_q="Migrate this 40k-line service from SQLAlchemy 1.4 to 2.0 across 63 modules, keeping all 1,240 tests green and the public API unchanged.",
         example_trace="Plan the migration order from the dependency graph; convert Query to select(); run the suite; read failures; revise; repeat until green.",
         example_a="a merged patch series with a green suite"),
]

# Enforcement: over-length traces are truncated at the band ceiling and the
# sample is dropped rather than trained past the boundary, so the band is a
# hard contract instead of a suggestion.
BAND_ENFORCEMENT = dict(overlength_policy="truncate-and-drop", tolerance_pct=10)

# ---------------------------------------------------------------------------
# 6b. Learning-rate schedule. [REVIEW FIX 8]
#
# Without this the anneal is undefined. A cooldown's effect comes from the LR
# decaying to near-zero while the highest-quality data is in front of the
# model; an earlier draft described only the data half and called it an anneal.
# The reserve in section 5 and this schedule are the same mechanism.
# ---------------------------------------------------------------------------

LR_SCHEDULE = dict(
    peak=3.0e-4,
    # Expressed as a FRACTION, not an absolute token count. An earlier draft
    # used 15B absolute, which is 0.5% of the 3T run but 75% of a 20B proxy arm
    # - the proxy would have spent three quarters of its life warming up and
    # every arm would have been compared in the wrong regime. Caught by the
    # harness dry-run.
    warmup_frac=0.005,
    warmup_shape="linear",
    main_shape="cosine-to-10pct",
    main_note="Cosine from peak down to 10% of peak across phases A-C, NOT to "
              "zero. Decaying to zero before the anneal would spend the "
              "cooldown's only lever before the cooldown starts.",
    anneal_shape="linear-to-zero",
    anneal_note="Linear decay from 10% of peak to 0 across the whole of phase "
                "D. Linear rather than cosine because the reserve is short and "
                "cosine spends most of a short window near its start value.",
    anneal_start_frac=0.10,
    min_lr=0.0,
    rewarm="none - phase D is a continuation of the phase C decay, not a "
           "restart. Re-warming into an anneal discards the stability the "
           "cooldown exists to produce.",
    why_this_shape="The reserve is 3% of the budget. Decay-to-zero across "
                   "exactly that window is what makes the final tokens "
                   "disproportionately influential, which is the entire reason "
                   "for holding premium data back.",
)

# ---------------------------------------------------------------------------
# 6c. Sequence-length ladder. [TRANSCRIPT REQUIREMENT]
#
# Session 5 was specific and an earlier draft was vague. Three hard rules:
#   - Every example in a batch has the SAME length. Mixed-length batches are
#     not possible (session 5: "In a batch all example have same length").
#   - Nothing below 4,096 tokens. Padding short samples burns compute for
#     nothing ("Shorter one is a loss of compute for us").
#   - Length doubles up the ladder; V4 ran 4K -> 8K -> 16K -> 32K -> 64K.
#
# Long context is therefore a BATCH CONSTRUCTION decision, not only a data
# decision, and the ladder is part of the curriculum spec rather than an
# implementation detail.
# ---------------------------------------------------------------------------

SEQ_MIN = 4096
SEQ_LADDER = [
    dict(seq_len=4096,   phase="A", share_of_phase=100, batch_examples=512,
         note="Everything starts here. 4,096 tokens is already a long document."),
    dict(seq_len=8192,   phase="B", share_of_phase=70,  batch_examples=256,
         note="First doubling. Code repos and arXiv papers begin to fit whole."),
    dict(seq_len=16384,  phase="B", share_of_phase=30,  batch_examples=128,
         note="Introduced inside phase B so the jump into C is not a cliff."),
    dict(seq_len=32768,  phase="C", share_of_phase=55,  batch_examples=64,
         note="Repo-level packs and full court judgments."),
    dict(seq_len=65536,  phase="C", share_of_phase=45,  batch_examples=32,
         note="Agentic trajectories at full length; a 100k-token session is "
              "truncated here rather than in the middle of a tool call."),
    dict(seq_len=65536,  phase="D", share_of_phase=100, batch_examples=32,
         note="The anneal does not introduce a new length. Changing sequence "
              "length and data quality at the same moment makes an unstable "
              "step impossible to attribute."),
]

# Tokens-per-batch is held roughly constant as length grows, by halving the
# example count at each doubling. This keeps the optimiser's effective batch
# size stable across the ladder - otherwise every length change is also a
# batch-size change, and the loss spike could be either.
SEQ_CONSTANT_TOKENS_PER_BATCH = True

# ---------------------------------------------------------------------------
# 6d. Band transitions. [TRANSCRIPT REQUIREMENT]
#
# Session 5 reported this from V4 directly: changing a band sharply made the
# loss jump and the gradient norm spike, and it took manual intervention to
# recover. "You cannot introduce Sanskrit suddenly and say start speaking
# shloka." Bands must diffuse into each other rather than switch.
#
# An earlier draft had hard boundaries at every phase edge and every difficulty
# band change, which is precisely the failure mode described.
# ---------------------------------------------------------------------------

BAND_TRANSITION = dict(
    overlap_frac=0.15,
    overlap_note="Each phase boundary is a linear crossfade over 15% of the "
                 "OUTGOING phase's tokens. During the crossfade both mixtures "
                 "are sampled, interpolated linearly, so no lane share ever "
                 "steps discontinuously.",
    grad_norm_target=0.2,
    grad_norm_trip=0.5,
    trip_action="If gradient norm exceeds the trip value for more than 100 "
                "consecutive steps during a crossfade, the crossfade is "
                "extended (its remaining fraction is doubled) rather than the "
                "LR being cut. Cutting LR treats the symptom; the cause is the "
                "mixture moving too fast.",
    monitored=["gradient norm", "per-lane loss", "loss spike count per 1k steps"],
    why="V4 evidence, reported first-hand in session 5: sharp band changes "
        "produced visible loss and gradient-norm excursions that required "
        "manual intervention to recover from.",
)

# ---------------------------------------------------------------------------
# 6e. Decontamination. [REVIEW ITEM]
#
# An earlier draft stated a 12-gram rule and never measured anything. A rule
# without a number is not evidence. This declares what gets measured, against
# what, and what the plan does when contamination is found.
# ---------------------------------------------------------------------------

DECONTAMINATION = dict(
    ngram=12,
    method="12-gram exact overlap plus MinHash near-duplicate at Jaccard 0.8, "
           "run per lane against the test split of every suite in section 8.",
    suites=["SWE-bench Verified", "SWE-bench Live-Pro", "Terminal-Bench",
            "tau-bench", "BFCL v4", "LiveCodeBench", "AIME 2026", "GPQA",
            "FrontierMath", "MMLU-Pro", "MILU", "FLORES-200", "RULER",
            "GSM8K", "HumanEval+", "MBPP+"],
    when="At corpus build time AND again at reserve-build time. Inherited "
         "decontamination is not trusted, because the anneal draws from pools "
         "that were filtered under a different set of rules.",
    action="Strip the contaminated document, not just the matching span - a "
           "document containing a verbatim test item is evidence the whole "
           "document is downstream of the benchmark.",
    report="Contamination rate per lane per suite, published with the run. A "
           "lane reporting exactly zero against a suite is treated as a "
           "detector failure to be investigated, not as a clean result.",
    measured=True,
    measured_note="Run by decon/measure.py against the corpora we actually "
                  "hold. Results in decon_results.json and section 13.5.",
    measured_scope="2 of 9 lanes and 6 of the 16 suites listed above. The other "
                   "seven lanes have no cleaned corpus yet, and the remaining "
                   "suites are either gated (FLORES-200), not on the HF "
                   "datasets-server in a scannable form (SWE-bench family, "
                   "Terminal-Bench, tau-bench, BFCL), or not yet released in a "
                   "test split we can pull (AIME 2026, FrontierMath). The "
                   "measurement is real within that scope and claims nothing "
                   "outside it.",
)

# ---------------------------------------------------------------------------
# 6f. Sensitivity. [REVIEW ITEM]
#
# Which assumptions, if wrong, change a conclusion. Each entry names the
# parameter, its plausible range, and what breaks at the extremes. audit.py
# re-runs the whole plan at each bound rather than reasoning about it in prose.
# ---------------------------------------------------------------------------

SENSITIVITY = [
    dict(id="sangraha_fertility",
         label="Sangraha's tokenizer (unstated by AI4Bharat)",
         kind="fertility", target=("indic_sp", "indic"),
         base=2.30, low=1.80, high=13.268,
         basis="AI4Bharat publishes 251.3B tokens and documents no tokenizer. "
               "Checked: the HF dataset card, arXiv:2403.06350 abstract and "
               "full PDF, and the IndicLLMSuite GitHub README - none state it. "
               "The high bound is our own MEASURED cl100k fertility on Telugu; "
               "the low bound is an aggressive Indic sentencepiece.",
         breaks="At the high bound the Indic lane loses ~83% of its apparent "
                "supply and flips from verified-only to mostly manufactured."),
    dict(id="web_dedup",
         label="Cross-corpus dedup survival for web",
         kind="dedup", target="web",
         base=0.65, low=0.40, high=0.85,
         basis="A single estimate covering the shared Common Crawl lineage of "
               "Nemotron-CC, DCLM and FineWeb-Edu. Never measured.",
         breaks="At the low bound the largest lane in the plan needs "
                "repetition even at 3T."),
    dict(id="agentic_traj_tokens",
         label="Mean tokens per agentic trajectory",
         kind="source_tokens", target="Nemotron-SWE-v1 (OpenHands trajectories)",
         base=1.77e9, low=0.59e9, high=5.31e9,
         basis="DERIVED from 59k trajectories x an assumed 30,000 tok/trajectory. "
               "Bounds are 10k and 90k tok/trajectory.",
         breaks="Nothing. Even at 3x the assumed length the lane stays "
                "overwhelmingly manufactured, which is why the agentic "
                "conclusion is robust to an assumption that is itself weak."),
    dict(id="civic_judgment_tokens",
         label="Mean tokens per Indian court judgment",
         kind="source_tokens", target="Indian court judgments (SC + 25 HCs)",
         base=30e9, low=10e9, high=60e9,
         basis="DERIVED from ~10M judgments x an assumed 3,000 tok each. The "
               "corpus has never been built, so both factors are estimates.",
         breaks="At the low bound the civic lane becomes substantially "
                "manufactured and needs a COLLECT plan it does not have."),
]

# ---------------------------------------------------------------------------
# 6g. Cleaning priority.
#
# An earlier draft derived this table from manufactured_pct alone. At the 3T
# budget that silently dropped Indic off the list entirely - because Indic is
# 0% manufactured here - while section 2 was simultaneously arguing that the
# cleaning must switch to the Verified tier. The document told the reader to do
# something and then omitted it from the table saying what to do.
#
# Priority is therefore declared, not derived. "Starved" is not the same as
# "needs manufacturing": a lane can be well supplied in aggregate and still be
# cleaning the wrong tier of it.
# ---------------------------------------------------------------------------

CLEANING_PRIORITY = [
    dict(lane="indic", rank=1,
         why_now="Highest value per hour of work. Not manufactured at this "
                 "budget, but the tier we have actually cleaned is the wrong "
                 "one: unverified supplies 11.1% of the lane and is barred from "
                 "the anneal, while Verified supplies 88.9% and is the only "
                 "tier the cooldown accepts. The data exists, is licensed "
                 "CC-BY-4.0, and needs no generation.",
         action="**Switch tiers.** Session 4 cleaned Sangraha *unverified*; "
                "move to Sangraha *Verified*. The training slice must start at "
                "row >= 300 of verified/tel, because rows 0-299 are already the "
                "frozen held-out Golden Proxy."),
    dict(lane="reasoning", rank=2,
         why_now="35% manufactured, but the binding constraint is legal rather "
                 "than volumetric.",
         action="Fix the licence gate before scaling: 3 of the 4 sampled "
                "sources failed it (one undeclared, two AGPL-3.0). Replace them "
                "or the lane is unusable regardless of how clean it is."),
    dict(lane="agentic", rank=3,
         why_now="82% manufactured - the most starved lane in the plan - but "
                 "cleaning is not the bottleneck.",
         action="Not a cleaning task. Build the container/shell/verifier "
                "harness first; there is nothing to clean until it produces "
                "trajectories."),
    dict(lane="civic", rank=4,
         why_now="Supply exists in the world but sits in no corpus.",
         action="COLLECT, not clean: court judgments and assembly proceedings. "
                "Highest raw token yield per hour, gated on licence clearance."),
    dict(lane="parallel", rank=5,
         why_now="0.97% of budget.",
         action="Lowest priority - the plan absorbs total failure of this lane "
                "without renegotiation."),
]

# What running the priority-1 job actually turned up. Measured, not predicted:
# the numbers below come from cleaning/results_verified.json and from a direct
# scan of both raw corpora using the pipeline's own stopword list and threshold.
CLEANING_FINDINGS = [
    dict(
        id="stopword_length_bias",
        title="The quality filter has a length bias, and it costs 10% of the "
              "only anneal-eligible Indic tier",
        measured=dict(
            threshold="stopword_hits_first_250 < 2 rejects the document",
            stopword_list_size=41,
            unverified_median_hits=10.0,
            verified_median_hits=8.0,
            unverified_mean_words=313,
            verified_mean_words=257,
            unverified_pct_below=4.58,
            verified_pct_below=10.28,
            verified_observed_reject_pct=9.90,
        ),
        finding=(
            "Sangraha Verified loses 16.58% at the quality filter against the "
            "unverified tier's 8.01% - twice the cull rate on what is supposed "
            "to be the better tier. The dominant reason in both corpora is the "
            "layer-1 `too_few_stopwords` heuristic (roughly 60% of all "
            "removals in each), not the trained classifier. A direct scan "
            "explains it: verified documents are ~18% shorter (257 vs 313 mean "
            "words), and the rule counts stopword hits in a FIXED 250-word "
            "window with a FIXED threshold of 2. A short document has fewer "
            "chances to clear a fixed bar. The predicted rejection rate from "
            "that scan (10.28%) matches the observed one (9.90%)."
        ),
        not_established=(
            "WHY verified documents are shorter cannot be established from the "
            "data we hold. Sangraha's verified/tel parquet carries no "
            "sub-source labels, so OCR fragments and concise curated articles "
            "are indistinguishable in these records. An earlier draft of this "
            "note asserted an OCR/ASR explanation; that was speculation and is "
            "withdrawn. The length bias is measured; its cause is not."
        ),
        action=(
            "Normalise the rule by length - hits per 100 words rather than hits "
            "in the first 250 - before this tier is cleaned at scale. On the "
            "current threshold, ~10% of the only tier the anneal will accept is "
            "discarded for being concise."
        ),
    ),
]

# ---------------------------------------------------------------------------
# 7. Loss-masking policy. Session 5's central distinction.
# ---------------------------------------------------------------------------

MASKING = [
    dict(segment="User turn / issue body",          policy="MASKED",  why="Predicting the user's question teaches nothing about answering it."),
    dict(segment="Assistant plan / reasoning trace", policy="TRAINED", why="This is the behaviour being learned."),
    dict(segment="Assistant tool call",              policy="TRAINED", why="Choosing the right tool with the right arguments is the capability."),
    dict(segment="Tool result / stderr / stack trace", policy="MASKED", why="Training here teaches the model to imitate a Python interpreter. It must READ the 4,000-line trace, never reproduce it."),
    dict(segment="Final assistant answer",           policy="TRAINED", why="The deliverable."),
    dict(segment="Verifier / test outcome",          policy="REWARD",  why="Not cross-entropy. Enters at stage 4+, outside this plan's scope."),
]

# ---------------------------------------------------------------------------
# 8. Benchmarks each lane is accountable for. A lane with no benchmark has no
#    business holding budget.
# ---------------------------------------------------------------------------

LANE_BENCHMARKS = {
    "web":       ["MMLU-Pro (>=85.2, Gemma-4-31B parity)", "GPQA (>=84.3)", "SimpleQA"],
    "code":      ["LiveCodeBench (>=80.0)", "SWE-bench Verified", "HumanEval+/MBPP+"],
    "reasoning": ["AIME 2026 (>=89.2)", "GPQA", "FrontierMath", "GSM8K"],
    "indic":     ["MILU (>=75, beats GPT-4o's real 72)", "IndicGenBench", "FLORES-200 (never averaged)"],
    "multiling": ["FLORES-200 non-Indic", "MGSM"],
    "longctx":   ["RULER @32k/128k", "LongBench-v2", "HELMET"],
    "civic":     ["MILU Law/Governance and Arts/Humanities sub-scores", "Drishtikon-Bench (session-3 novel)"],
    "agentic":   ["tau-bench", "BFCL v4", "Terminal-Bench", "WebArena", "GAIA", "DPI-Gym (session-3 novel)"],
    "parallel":  ["FLORES-200 en<->xx chrF++", "IN22-Gen"],
}

# ---------------------------------------------------------------------------
# 8b. Manufacturing plan.
#
# Any lane whose demand exceeds its capped supply must make up the difference.
# "We will generate it" is wishful accounting unless it carries a rate and a
# price, so every shortfall below is costed. Two distinct kinds:
#   GENERATE - tokens that do not exist and must be produced by inference
#              (and, for agentic, by executing a real environment).
#   COLLECT  - tokens that exist in the world but are not yet in any corpus;
#              the cost is crawling, OCR and licensing, not GPU time.
# ---------------------------------------------------------------------------

MANUFACTURING = {
    "agentic": dict(
        kind="GENERATE",
        method="Verifier-gated rollouts in a live harness: real container, real shell, real test suite. Only trajectories whose verifier returns green are kept.",
        mean_tokens_per_unit=25_000,   # one trajectory
        generated_frac=0.30,           # the rest is tool output and env text, not model-generated
        env_seconds_per_unit=60,
        parallel_workers=10_000,
        generator_params=40e9,
        gpus=512,
        risk="This is the single largest infrastructure commitment in the plan and the thing V4 did not have. If the harness lands late, the agentic lane cannot be filled by any amount of scraping.",
        fallback="If the harness delivers under 50% of planned rollouts, agentic drops to its 1.00% floor and the freed share goes to code, which has repetition headroom.",
    ),
    "reasoning": dict(
        kind="GENERATE",
        method="Synthetic textbooks and graded exercise sets (Cosmopedia / phi-4 method) plus long-CoT distillation at the four length bands, every item gated on a checkable final answer.",
        mean_tokens_per_unit=2_000,
        generated_frac=1.00,
        env_seconds_per_unit=0,
        parallel_workers=0,
        generator_params=40e9,
        gpus=512,
        risk="Synthetic math inherits the generator's error modes. Mitigated by answer-checking every item and discarding the trace when the final answer is wrong.",
        fallback="If verified yield falls below 60%, cut the lane to 5.0% and raise web; a smaller honest reasoning lane beats a large unverified one.",
    ),
    "indic": dict(
        kind="GENERATE",
        method="Oral-Commons ASR (session-3 novel: elicited speech via ASHA/community-radio networks) + OCR of Indic print + in-language synthetic textbooks generated from the verified tier, never from the translated tier.",
        mean_tokens_per_unit=1_500,
        generated_frac=0.70,           # 30% comes from ASR/OCR of real speech and print
        env_seconds_per_unit=0,
        parallel_workers=0,
        generator_params=40e9,
        gpus=256,
        risk="Generating Indic text from a model that is itself weak in Indic risks a feedback loop that amplifies translationese.",
        fallback="Generate only from Sangraha Verified seeds, and hold the synthetic tier out of the anneal entirely (already enforced by the eligibility rules).",
    ),
    "civic": dict(
        kind="COLLECT",
        method="Direct ingestion of court judgments, state assembly proceedings and gazettes that exist as public record but sit in no pretraining corpus. Cost is crawling, OCR and licence clearance, not GPU time.",
        mean_tokens_per_unit=3_000,
        generated_frac=0.0,
        env_seconds_per_unit=0,
        parallel_workers=0,
        generator_params=0,
        gpus=0,
        risk="Licensing. The ANI v. OpenAI proceedings leave Indian TDM exemptions unsettled (session-3 analysis).",
        fallback="Restrict to sources with explicit government-open-data licences and absorb the shortfall into the Indic lane.",
    ),
    # Added only because the unit conversion in section 0b pushed this lane past
    # its capped supply and the build refused to emit the document without it -
    # which is the invariant working as intended.
    "longctx": dict(
        kind="GENERATE",
        method="Retrieval-clustered long-context assembly: group topically related documents and concatenate them into coherent long sequences with genuine cross-document dependencies, plus multi-hop QA constructed across the assembled span. Only the linking questions are generated; the body is re-assembled real text.",
        mean_tokens_per_unit=128_000,
        generated_frac=0.05,
        env_seconds_per_unit=0,
        parallel_workers=0,
        generator_params=40e9,
        gpus=128,
        risk="Naive concatenation produces long sequences with no long-range dependency, which trains position handling but not retrieval over distance - the failure mode LADM and ProLong were written to diagnose.",
        fallback="Gate every assembled sequence on a measured attention-dependency score; discard the bottom half rather than shipping filler.",
    ),
    "parallel": dict(
        kind="GENERATE",
        method="Back-translation with IndicTrans2 over the verified Indic tier and over FineWeb-Edu, filtered by round-trip chrF++.",
        mean_tokens_per_unit=80,
        generated_frac=1.00,
        env_seconds_per_unit=0,
        parallel_workers=0,
        generator_params=1.1e9,        # IndicTrans2, not the 40B
        gpus=64,
        risk="Back-translated pairs are the weakest data in the plan.",
        fallback="This lane is 0.97% of the budget. If it fails entirely the plan absorbs it without renegotiation.",
    ),
}

# ---------------------------------------------------------------------------
# 9. Proxy experiment. The plan is a hypothesis until this runs.
# ---------------------------------------------------------------------------

PROXY = dict(
    scale_1b=dict(params=1.0e9, tokens_per_arm=20e9, seq_len=8192),
    scale_3b=dict(params=3.0e9, tokens_per_arm=60e9, seq_len=8192),

    # [REVIEW FIX 1] Scale the CORPUS, not just the model.
    #
    # The first draft ran 1B params on 20B tokens - 20 tokens/param, against the
    # full run's 350. At 20B total tokens the code lane draws ~4.7B against
    # ~830B of usable supply: 0.006 epochs. Arm A3 exists to falsify the
    # repetition ceiling and would never have repeated a single token. It would
    # have silently degenerated into "more code, less web at constant
    # freshness" while appearing to test a data-constrained regime.
    #
    # Every lane's candidate pool is therefore subsampled by the same factor as
    # the budget, so the proxy sits at the SAME epoch count per lane as the full
    # run. Repetition becomes observable, and A3 can actually fail.
    #
    # This does not fix the tokens/param gap - that is a separate and
    # unfixable-at-this-price limitation, disclosed rather than papered over.
    corpus_scaled=True,
    tokens_per_param_note=(
        "The proxy runs at 20 tok/param against the full run's 350. Corpus "
        "scaling makes the DATA-CONSTRAINED regime match; it does not make the "
        "OVER-TRAINED regime match. Mixture conclusions that depend on being far "
        "past Chinchilla-optimal are not tested by this screen, and the 3B "
        "confirmation at 20 tok/param does not close that gap either. The "
        "honest scope of this experiment is: relative mixture ranking under "
        "matched repetition pressure."
    ),
    cluster=dict(gpus=8, device="H100-SXM", peak_bf16_flops=989e12, mfu=0.40, usd_per_gpu_hour=3.00),
    full_run=dict(gpus=4096, mfu=0.40, usd_per_gpu_hour=3.00),
    arms=[
        dict(id="A0", name="Baseline",
             change="The phase schedule in section 2, compressed proportionally into 20B tokens.",
             tests="Nothing on its own. It is the reference every other arm is differenced against."),
        dict(id="A1", name="Agentic-starved",
             change="Agentic lane set to 0.0% in every phase; the freed share goes to web.",
             tests="Whether a 1.75% agentic lane buys measurable tool-use capability, or whether it is too thin to matter and should be deferred entirely to SFT."),
        dict(id="A2", name="Floor-off",
             change="Protected floor removed. The selector may discard Indic, agentic, long-context and civic freely.",
             tests="Whether the floor is load-bearing. If A2 matches A0 on Indic metrics, the floor is costing 7% of every batch for nothing and should be cut."),
        dict(id="A3", name="Code-heavy",
             change="Code raised to 34% whole-run (4.76T, ~5.3 epochs of Stack v2); web drops to 42%.",
             tests="Whether the 4-epoch repetition ceiling is real for code. This arm deliberately violates it, so the plan's central supply constraint is falsifiable rather than assumed."),
        dict(id="A4", name="Static-mix",
             change="No phases. The whole-run average mixture applied uniformly from step 0, identical totals, no anneal.",
             tests="Whether curriculum ordering is worth anything at all, or whether only the totals matter. This is the arm that could invalidate the entire document."),
        dict(id="A5", name="Selector-off",
             change="OPUS-style selection disabled entirely. Same trained-token budget, same mixture, but every candidate batch is trained on rather than the top half.",
             tests="The most expensive unexamined assumption in the plan. Selection costs 2x corpus (section 5.2) plus per-refresh compute, justified by a claimed 8x efficiency. No other arm touches it. If A0 does not beat A5 by a wide margin, the honest move is to delete the selector, halve the corpus requirement, and reclaim the throughput."),
        dict(id="A6", name="Sharp-transitions",
             change="Band crossfades removed: phase boundaries and difficulty-band changes step discontinuously, as the first draft of this plan specified.",
             tests="Whether the 15% crossfade in section 6d earns its complexity. V4 evidence says sharp changes spike the loss and gradient norm, but that was observed, not ablated. This arm measures loss-spike count and peak gradient norm rather than end-task accuracy."),
    ],
    metrics=[
        dict(lane="code",      metric="HumanEval+ pass@1",            floor_note="1B models score in the teens; the delta is the signal, not the level."),
        dict(lane="code",      metric="MBPP+ pass@1",                 floor_note=""),
        dict(lane="agentic",   metric="BFCL v4 simple + multiple",    floor_note="Restricted to the non-live subsets a 1B model can register at all."),
        dict(lane="reasoning", metric="GSM8K 8-shot",                 floor_note=""),
        dict(lane="web",       metric="MMLU 5-shot",                  floor_note="The common-sense guard rail. Any arm that wins its lane by regressing MMLU >1.0 pt is rejected."),
        dict(lane="indic",     metric="FLORES-200 en->{hi,te,ta,bn} chrF++", floor_note="Reported per language, never averaged."),
        dict(lane="indic",     metric="MILU 4-language subset",       floor_note=""),
        dict(lane="longctx",   metric="RULER @8k and @32k",           floor_note=""),
        dict(lane="all",       metric="Per-lane held-out loss",       floor_note="Uses the disjoint held-out slices already frozen by the session-4 pipeline (telugu_heldout.jsonl, reasoning_sft_heldout.jsonl)."),
    ],
    # Pre-registered. Written before the runs, so the result cannot be
    # reinterpreted after the fact.
    decisions=[
        dict(rule="Keep agentic at 1.75%",
             iff="A0 - A1 >= +2.0 pts BFCL simple+multiple AND MMLU regression <= 0.5 pts",
             else_="Cut agentic to its 1.00% floor and move the difference to code; defer agentic capability to SFT and RL."),
        dict(rule="Keep the 7% protected floor",
             iff="A2 shows >= 3.0 chrF++ drop on FLORES en->te OR >= 2.0 pts drop on the MILU subset",
             else_="Drop the floor to Indic-only at 1.5% and return ~5% of every batch to the selector."),
        dict(rule="Hold code at 23.5% rather than 34%",
             iff="A3 - A0 < +1.5 pts HumanEval+ OR A3 MMLU regression > 1.0 pt",
             else_="Raise code toward 34% and accept ~5.3 epochs of Stack v2, revising the 4-epoch ceiling with our own evidence."),
        dict(rule="Keep the four-phase curriculum and the anneal reserve",
             iff="A0 - A4 >= +1.0 pt on the mean of {HumanEval+, BFCL, GSM8K} with no single lane regressing more than 1.0 pt",
             else_="Abandon phasing, ship the static mixture, and reallocate the 3% reserve into Phase C."),
        dict(rule="Keep the OPUS-style selector at all",
             iff="A0 - A5 >= +1.5 pts on the mean of {HumanEval+, GSM8K, MMLU} at equal trained tokens",
             else_="Delete the selector. That halves the corpus requirement from 5.79T to 3.00T, removes the per-refresh throughput tax, and makes the protected floor unnecessary - the single largest simplification available to this plan."),
        dict(rule="Keep the 15% band crossfade",
             iff="A6 shows >= 2x the loss-spike count of A0, or peak gradient norm above 0.5 during any transition",
             else_="Drop the crossfade and step the bands directly; the complexity is not buying stability."),
        dict(rule="Promote to the 3B confirmation run",
             iff="the 1B ranking of A0 against the winning variant is stable across 3 seeds (sign of the delta unchanged)",
             else_="Re-run at 1B with a wider token budget before spending 3B compute; a sign-unstable delta at 1B is noise, not a finding."),
    ],
)

# ---------------------------------------------------------------------------
# 10. Data-gating status. MEASURED by the session-4 pipeline.
# ---------------------------------------------------------------------------

GATING = dict(
    # Cumulative across session 4 and the session-5 Verified run.
    measured_clean_tokens=93_019_466,
    measured_clean_docs=30_032,
    measured_raw_tokens=112_010_145,
    measured_raw_docs=35_756,
    deterministic=True,
    session4_clean_tokens=53_781_200,
    added_this_session=39_238_266,
    growth_multiple=93_019_466 / 53_781_200,
    source="4_model_data/assignment/results.json (session 4) + "
           "5_data_mixture_curriculum/cleaning/results_verified.json (session 5)",
    corpora=[
        dict(name="Sangraha unverified / Telugu", lane="indic", tier="unverified",
             raw_tokens=50_002_626, raw_docs=11_607, final_docs=10_632,
             session="4",
             note="Cleaned the tier this plan gives the SMALLEST weight to and bars from the anneal."),
        dict(name="Reasoning / SFT distillation mix (4 sources)", lane="reasoning", tier=None,
             raw_tokens=12_006_709, raw_docs=9_553, final_docs=7_846,
             session="4",
             note="3 of 4 sources failed the license gate (one undeclared, two AGPL-3.0)."),
        dict(name="Sangraha VERIFIED / Telugu", lane="indic", tier="verified",
             raw_tokens=50_000_810, raw_docs=14_596, final_docs=11_554,
             session="5",
             note="The priority-1 target from section 16, now actually cleaned. Training slice "
                  "starts at row 300 of verified/tel because rows 0-299 are session 4's frozen "
                  "held-out Golden Proxy; disjointness is asserted, not assumed. Adds 39.2M "
                  "cleaned tokens in the only tier the anneal accepts."),
    ],
)
