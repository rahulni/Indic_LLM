"""Corpus registry.

The pipeline runs over two deliberately different corpora, because one
corpus could not exercise all eight strategies.

  telugu_web    AI4Bharat Sangraha, unverified/tel. A raw Indic web crawl.
                Exercises: normalization (Brahmic joiners), language ID
                (the te/tel bug), Indic quality-filter bias, Indic PII.
                Cannot exercise: ghost tags (no conversation markers exist
                in a web crawl), and - as originally sampled - dedup.

  reasoning_sft A four-source mix of reasoning/SFT distillation sets from
                the same Hugging Face profile the assignment's example link
                points at. Ships PRE-FLATTENED text with literal ChatML
                markers already in the stored `text` field, so ghost tags
                are real and not manufactured. Two of its four sources
                overlap byte-identically, which makes cross-source (global)
                deduplication a real measured effect rather than a diagram.
                Its licenses genuinely differ, including one source with no
                declared license at all - which is what the manifest gating
                rule is for.

Nothing here is chosen for convenience. Each source was inspected before
being added: the marker counts, the cross-source overlap, and the license
fields were all verified against the live dataset before this file was
written.
"""
from __future__ import annotations

import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSIGNMENT_DIR = os.path.dirname(PIPELINE_DIR)
RAW_DIR = os.path.join(ASSIGNMENT_DIR, "raw_sample")


CORPORA: dict[str, dict] = {
    "telugu_web": {
        "id": "telugu_web",
        "name": "AI4Bharat Sangraha — unverified tier — Telugu",
        "kind": "web_crawl",
        "claimed_lang": "tel",
        "script": "telugu",
        "hf_url": "https://huggingface.co/datasets/ai4bharat/sangraha",
        "raw_file": "telugu_raw.jsonl",
        "heldout_file": "telugu_heldout.jsonl",
        # Contiguous slice, NOT a uniform random draw. A uniform 8% sample of a
        # 150k-doc shard has ~0.6% odds of containing both halves of any given
        # duplicate pair, which is why the first round of this pipeline found
        # essentially no duplicates. A contiguous slice preserves pairs.
        "sampling": {
            "strategy": "contiguous_slice",
            "target_tokens": 50_000_000,
            "reason": (
                "A uniform random sub-sample destroys duplicate pairs by construction. "
                "Deduplication is the session's headline topic, so the sample has to be "
                "drawn in a way that can actually contain duplicates."
            ),
        },
        "sources": [
            {
                "key": "sangraha_unverified_tel",
                "hf_id": "ai4bharat/sangraha",
                "data_files": "unverified/tel/data-0.parquet",
                "license": "CC-BY-4.0",
                "text_field": "text",
            }
        ],
        "heldout": {
            "hf_id": "ai4bharat/sangraha",
            "data_files": "verified/tel/*.parquet",
            "n_docs": 300,
            "note": "Disjoint verified/tel slice - the Golden Proxy role, never in the training pool.",
        },
        "ner": {
            "model": "kuppuluri/telugu_bertu_ner",
            "license": "MIT",
            "labels": ["B-PERSON", "I-ORG", "B-ORG", "I-LOC", "B-MISC", "I-MISC", "I-PERSON", "B-LOC", "O"],
            "person_group": "PERSON",
            "note": (
                "ai4bharat/IndicNER shares a publisher with the dataset and was the obvious "
                "first choice, but it is gated and returned a real 401 on download. This "
                "ungated MIT-licensed model is the one that actually works."
            ),
        },
        "has_code": False,
        "quality_classifier": True,
    },
    "reasoning_sft": {
        "id": "reasoning_sft",
        "name": "Reasoning / SFT distillation mix — four sources",
        "kind": "conversation_sft",
        "claimed_lang": "eng",
        "script": "latin",
        "hf_url": "https://huggingface.co/lordx64",
        "raw_file": "reasoning_sft_raw.jsonl",
        "heldout_file": "reasoning_sft_heldout.jsonl",
        "sampling": {
            "strategy": "contiguous_slice_per_source",
            "target_tokens": 12_000_000,
            "reason": (
                "Contiguous per-source slices taken from the head of each source, so the "
                "cross-source overlap between fable-sft-combined-v2 and "
                "agentic-distill-fable-5-sft survives into the sample and global dedup has "
                "something real to find."
            ),
        },
        # Licenses below are read live from the HF API at sampling time and
        # cross-checked against these declarations; a mismatch is reported.
        # Each source carries its OWN token budget. A single shared budget
        # consumed in source order starves whatever comes last - and what comes
        # last here is precisely the pair whose overlap makes global dedup
        # measurable, so the two fable sources get equal, independent slices
        # taken from the head of each.
        "sources": [
            {
                "key": "opus_reasoning",
                "hf_id": "lordx64/reasoning-distill-opus-4-7-max-sft",
                "license": "apache-2.0",
                "text_field": "text",
                "target_tokens": 4_000_000,
                "max_rows": 7823,
            },
            {
                "key": "kimi_reasoning",
                "hf_id": "lordx64/reasoning-distill-kimi-k2-6-max-sft",
                "license": None,  # genuinely absent - the gating rule must block this
                "text_field": "text",
                "target_tokens": 3_000_000,
                "max_rows": 2500,
            },
            {
                "key": "fable_combined",
                "hf_id": "lordx64/fable-sft-combined-v2",
                "license": "agpl-3.0",
                "text_field": "text",
                "target_tokens": 2_500_000,
                "max_rows": 3000,
            },
            {
                "key": "fable_agentic",
                "hf_id": "lordx64/agentic-distill-fable-5-sft",
                "license": "agpl-3.0",
                "text_field": "text",
                "target_tokens": 2_500_000,
                "max_rows": 3000,
            },
        ],
        "heldout": {
            "hf_id": "lordx64/reasoning-distill-opus-4-7-max-sft",
            "n_docs": 300,
            "offset_after_training_slice": True,
            "note": "Rows past the end of the training slice - disjoint by construction.",
        },
        "ner": {
            "model": "dslim/bert-base-NER",
            "license": "MIT",
            "labels": None,  # this model ships correct labels in its own config
            "person_group": "PER",
            "note": "English corpus, so the Telugu NER model does not apply; this is the ungated English equivalent.",
        },
        "has_code": True,
        # The trained quality classifier is fit on Telugu labels drawn from
        # corpus A. It does not transfer to English SFT text, and pretending it
        # does would be exactly the kind of stand-in this pipeline refuses to
        # ship silently. Layer 1 (heuristics) runs for real; layer 2 is skipped
        # and reported as skipped.
        "quality_classifier": False,
    },
}

DEFAULT_CORPUS = "telugu_web"

# Session 4's two corpora, in the order its results.json records them. This
# list drives run_all.py and is deliberately NOT extended by the session-5
# addition below - regenerating results.json with a third corpus would alter an
# artifact that has already been submitted.
ORDER = ["telugu_web", "reasoning_sft"]

# --------------------------------------------------------------------------
# Session-5 addition. Additive only: nothing above this line changed.
#
# The session-5 mixture audit found that session 4 cleaned the WRONG Indic
# tier. At a 3T budget the unverified tier supplies 11.1% of the Indic lane and
# is barred from the anneal outright, while Sangraha Verified supplies 88.9% and
# is the only tier the cooldown accepts. So the cleaning moves to Verified.
#
# CONTAMINATION CONSTRAINT: telugu_web's held-out set is ALREADY drawn from
# verified/tel rows 0-299 (see sampling_report_telugu_web.json). Training on
# those rows would train directly on the frozen Golden Proxy. skip_rows=300
# makes the training slice disjoint by construction; run_verified.py additionally
# asserts it against the doc_ids in telugu_heldout.jsonl rather than trusting the
# offset.
# --------------------------------------------------------------------------

CORPORA["telugu_verified"] = {
    "id": "telugu_verified",
    "name": "AI4Bharat Sangraha — verified tier — Telugu",
    "kind": "web_crawl",
    "claimed_lang": "tel",
    "script": "telugu",
    "hf_url": "https://huggingface.co/datasets/ai4bharat/sangraha",
    "raw_file": "telugu_verified_raw.jsonl",
    "heldout_file": "telugu_verified_heldout.jsonl",
    "sampling": {
        "strategy": "contiguous_slice_after_heldout",
        "target_tokens": 50_000_000,
        "reason": (
            "Contiguous, like the unverified draw, so duplicate pairs survive into "
            "the sample. Offset past row 300 because rows 0-299 of this exact "
            "shard are telugu_web's frozen held-out set."
        ),
    },
    "sources": [
        {
            "key": "sangraha_verified_tel",
            "hf_id": "ai4bharat/sangraha",
            "data_files": "verified/tel/*.parquet",
            "license": "CC-BY-4.0",
            "text_field": "text",
            # The whole point. Wired through by stage0_sample.sample_corpus().
            "skip_rows": 300,
        }
    ],
    "heldout": {
        "hf_id": "ai4bharat/sangraha",
        "data_files": "verified/tel/*.parquet",
        "n_docs": 300,
        "offset_after_training_slice": True,
        "note": (
            "Starts past this corpus's own last training row, so it is disjoint "
            "from both this training slice and telugu_web's held-out rows 0-299."
        ),
    },
    "ner": {
        "model": "kuppuluri/telugu_bertu_ner",
        "license": "MIT",
        "labels": ["B-PERSON", "I-ORG", "B-ORG", "I-LOC", "B-MISC", "I-MISC", "I-PERSON", "B-LOC", "O"],
        "person_group": "PERSON",
        "note": "Same ungated Telugu NER model the unverified corpus uses.",
    },
    "has_code": False,
    # The stage-4 classifier was fit on 200 hand-labelled Telugu UNVERIFIED web
    # documents. The verified tier is curated web + OCR + ASR - related, but a
    # different distribution. We run it (unlike the English corpus, where the
    # labels said nothing at all) and disclose the transfer caveat rather than
    # presenting the score as clean in-domain.
    "quality_classifier": True,
    "quality_classifier_caveat": (
        "Labels are 200 Telugu unverified-web documents. Applied here to "
        "verified-tier text (curated web + OCR + ASR), which is the same "
        "language and script but not the same distribution. Reported with this "
        "caveat attached."
    ),
}

ORDER_SESSION5 = ["telugu_verified"]


def get(corpus_id: str) -> dict:
    if corpus_id not in CORPORA:
        raise KeyError(f"unknown corpus {corpus_id!r}; known: {sorted(CORPORA)}")
    return CORPORA[corpus_id]


def raw_path(corpus_id: str) -> str:
    return os.path.join(RAW_DIR, get(corpus_id)["raw_file"])


def heldout_path(corpus_id: str) -> str:
    return os.path.join(RAW_DIR, get(corpus_id)["heldout_file"])
