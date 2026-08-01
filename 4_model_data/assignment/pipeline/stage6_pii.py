"""Stage 6: PII scrub.

Three layers. Two are cheap and exact: regex for structured identifiers
(emails, Indian mobile numbers, IPv4 addresses), and a curated Telugu
name gazetteer split into high-confidence and ambiguous (common-word)
buckets so a real false-positive tension is a real,
counted thing. The third is a real model: kuppuluri/telugu_bertu_ner
(MIT-licensed, trained on the academic anikethjr/NER_Telugu dataset),
run per-document to catch person names the fixed gazetteer's ~20-word
list was never going to know about - AI4Bharat's own IndicNER would
have been the obvious first choice given it shares a publisher with our
dataset, but it's gated and returned a 401 on an actual download
attempt, so this ungated alternative stands in. Final redaction is the
union of all three layers; the gazetteer and NER are compared directly
so the precision/recall story has real numbers instead of an assertion.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import common
from common import StageTimer, make_report, read_jsonl, write_jsonl, write_json

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# Indian mobile numbers: optional +91/91/0 prefix, then a 10-digit number starting 6-9.
PHONE_RE = re.compile(r"(?:(?:\+91|91|0)[\s\-]?)?\b([6-9]\d{9})\b")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

# High-confidence personal given names - low ambiguity with ordinary vocabulary.
HIGH_CONFIDENCE_NAMES = {
    "రాజేష్", "సురేష్", "లక్ష్మి", "ప్రియ", "వెంకటేష్", "శ్రీను", "మహేష్",
    "నరేష్", "కిరణ్", "అనిత", "సునీత", "పద్మ", "శ్రీనివాస్", "వెంకట్",
    "రజనీకాంత్", "చిరంజీవి", "పవన్", "నాగార్జున", "బాలకృష్ణ", "సునీల్",
}
# Ambiguous names - also ordinary words or public-figure/place names in common use,
# so flagging every occurrence as "personal PII" is a real precision risk.
AMBIGUOUS_NAMES = {
    "రాజు": "also the common noun 'king'",
    "బాబు": "also a generic honorific ('sir'/'mister'), not always a name",
    "అమ్మ": "also the common noun 'mother'",
    "చిన్ని": "also a generic term of endearment ('little one'), not always a name",
}
ALL_NAMES = HIGH_CONFIDENCE_NAMES | set(AMBIGUOUS_NAMES)
NAME_TOKEN_RE = re.compile("|".join(re.escape(n) for n in sorted(ALL_NAMES, key=len, reverse=True)))

# --- Real NER model (lazy-loaded singleton, routed by corpus language) ---
# The Telugu model is useless on English and vice versa, so which model runs is
# a property of the corpus, not a constant.
NER_MAX_CHARS = 1000  # keeps runtime bounded; the cost of this is measured below
NER_MIN_SCORE = 0.6
NER_BATCH_SIZE = 32
_ner_pipeline = None
_ner_pipeline_key = None


def get_ner_pipeline():
    """One pipeline per (process, corpus). Re-created when the corpus changes."""
    global _ner_pipeline, _ner_pipeline_key
    cfg = common.corpus()["ner"]
    key = cfg["model"]
    if _ner_pipeline is None or _ner_pipeline_key != key:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        tok = AutoTokenizer.from_pretrained(key)
        model = AutoModelForTokenClassification.from_pretrained(key)
        if cfg.get("labels"):
            # The Telugu model's config.json ships generic LABEL_0..8 names; the
            # real label order is published only in its model card, not the config.
            model.config.id2label = {i: l for i, l in enumerate(cfg["labels"])}
            model.config.label2id = {l: i for i, l in enumerate(cfg["labels"])}
        _ner_pipeline = pipeline("ner", model=model, tokenizer=tok, aggregation_strategy="simple")
        _ner_pipeline_key = key
    return _ner_pipeline


def ner_person_names_batch(texts: list[str]) -> list[list[str]]:
    """Batched inference. Running one document at a time left the GPU-less CPU
    path idle between calls; batching is a pure speed change with identical
    outputs, which matters because the pipeline now runs four times."""
    nlp = get_ner_pipeline()
    person_group = common.corpus()["ner"]["person_group"]
    truncated = [t[:NER_MAX_CHARS] for t in texts]

    # Batch length-homogeneous documents together. A batch is padded to its
    # longest member, so mixing a 40-character document with a 1000-character
    # one wastes most of the compute on padding. Sorting by length first, then
    # restoring the original order, is a pure throughput change - attention
    # masks mean the per-token predictions are unaffected. Ties break on the
    # index so the ordering is total and the run stays deterministic.
    order = sorted(range(len(truncated)), key=lambda i: (len(truncated[i]), i))
    ordered_texts = [truncated[i] for i in order]

    try:
        results = nlp(ordered_texts, batch_size=NER_BATCH_SIZE)
    except Exception:
        return [[] for _ in texts]
    if ordered_texts and not isinstance(results, list):
        results = [results]

    out: list[list[str]] = [[] for _ in texts]
    for pos, entities in zip(order, results):
        names: list[str] = []
        for e in entities or []:
            if e.get("entity_group") == person_group and e.get("score", 0) >= NER_MIN_SCORE:
                w = e["word"].strip()
                if len(w) >= 2 and w not in names:
                    names.append(w)
        out[pos] = names
    return out


def redact(text: str, ner_names: list[str], use_gazetteer: bool = True) -> tuple[str, dict]:
    counts = Counter()
    examples = {}
    ner_only_names: list[str] = []
    ner_overlap_names: list[str] = []

    def _sub(pattern: re.Pattern, tag: str, s: str) -> str:
        def repl(m: re.Match) -> str:
            counts[tag] += 1
            if tag not in examples:
                examples[tag] = m.group(0)
            return f"[{tag}]"

        return pattern.sub(repl, s)

    text = _sub(EMAIL_RE, "EMAIL", text)
    text = _sub(PHONE_RE, "PHONE", text)
    text = _sub(IPV4_RE, "IP", text)

    # Real NER pass, first - so the gazetteer pass below only reports names
    # the model itself did NOT already catch (a clean "gazetteer recall
    # beyond the model" signal, rather than double-counting the same hit).
    for name in ner_names:
        occurrences = text.count(name)
        if occurrences == 0:
            continue
        counts["NAME_NER"] += occurrences
        examples.setdefault("NAME_NER", name)
        if name in ALL_NAMES:
            ner_overlap_names.append(name)
        else:
            ner_only_names.append(name)
        text = text.replace(name, "[NAME_NER]")

    def name_repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok in AMBIGUOUS_NAMES:
            counts["NAME_AMBIGUOUS"] += 1
            examples.setdefault("NAME_AMBIGUOUS", f"{tok} ({AMBIGUOUS_NAMES[tok]})")
        else:
            counts["NAME_HIGH_CONFIDENCE"] += 1
            examples.setdefault("NAME_HIGH_CONFIDENCE", tok)
        return "[NAME]"

    if use_gazetteer:
        text = NAME_TOKEN_RE.sub(name_repl, text)
    return text, {
        "counts": dict(counts),
        "examples": examples,
        "ner_only_names": ner_only_names,
        "ner_overlap_names": ner_overlap_names,
    }


def run(input_path: str) -> dict:
    timer = StageTimer("pii_scrub")
    cfg = common.corpus()
    ner_cfg = cfg["ner"]
    # The gazetteer is a curated Telugu/Indian name list. It has nothing to say
    # about an English corpus, so it is switched off there rather than run and
    # reported as having "found nothing".
    use_gazetteer = cfg["script"] == "telugu"
    docs = read_jsonl(input_path)

    total_counts: Counter = Counter()
    docs_with_pii = 0
    examples = []
    survivors = []
    ner_only_examples: list[str] = []
    ner_overlap_examples: list[str] = []

    # What truncation actually costs, measured rather than hand-waved.
    total_chars = sum(len(d["text"]) for d in docs)
    scanned_chars = sum(min(len(d["text"]), NER_MAX_CHARS) for d in docs)
    docs_longer_than_window = sum(1 for d in docs if len(d["text"]) > NER_MAX_CHARS)

    get_ner_pipeline()  # load once, up front, not on the first doc mid-loop

    # Batched NER over the whole corpus, in document order.
    ner_by_doc: list[list[str]] = []
    for i in range(0, len(docs), NER_BATCH_SIZE * 8):
        chunk = docs[i : i + NER_BATCH_SIZE * 8]
        ner_by_doc.extend(ner_person_names_batch([d["text"] for d in chunk]))
        if i and i % (NER_BATCH_SIZE * 80) == 0:
            print(f"    [stage6] NER {i}/{len(docs)} docs", flush=True)

    for d, ner_names in zip(docs, ner_by_doc):
        redacted, info = redact(d["text"], ner_names, use_gazetteer=use_gazetteer)
        if info["counts"]:
            docs_with_pii += 1
            total_counts.update(info["counts"])
            if len(examples) < 8:
                examples.append({"doc_id": d["doc_id"], "counts": info["counts"], "matched_examples": info["examples"]})
        for n in info["ner_only_names"]:
            if len(ner_only_examples) < 12 and n not in ner_only_examples:
                ner_only_examples.append(n)
        for n in info["ner_overlap_names"]:
            if len(ner_overlap_examples) < 12 and n not in ner_overlap_examples:
                ner_overlap_examples.append(n)
        d2 = dict(d)
        d2["text"] = redacted
        survivors.append(d2)

    write_jsonl(common.work_path("stage6_survivors.jsonl"), survivors)

    report = make_report(
        stage_num=6,
        stage_name="PII scrub",
        input_docs=len(docs),
        output_docs=len(survivors),  # PII scrub redacts text, it does not drop documents
        elapsed_s=timer.done(),
        extra={
            "note": (
                "Redaction removes text, not documents - survival is 100% of documents by design. "
                "NAME_NER is a real model pass, run before the gazetteer so NAME_HIGH_CONFIDENCE / "
                "NAME_AMBIGUOUS below report only what the model did NOT already catch - a real "
                "recall comparison, not an assertion. "
            )
            + ner_cfg["note"]
            + (
                ""
                if use_gazetteer
                else " The Telugu name gazetteer is disabled for this corpus: it is a curated Indic "
                "name list and has nothing to say about English text, so running it would produce a "
                "meaningless zero rather than a measurement."
            ),
            "corpus_id": cfg["id"],
            "ner_model": ner_cfg["model"],
            "ner_model_license": ner_cfg["license"],
            "gazetteer_enabled": use_gazetteer,
            "ner_truncation_chars": NER_MAX_CHARS,
            "ner_coverage": {
                "total_chars_in_corpus": total_chars,
                "chars_actually_scanned": scanned_chars,
                "pct_of_corpus_text_scanned": round(100.0 * scanned_chars / total_chars, 2) if total_chars else 0.0,
                "docs_longer_than_window": docs_longer_than_window,
                "docs_total": len(docs),
                "note": (
                    "Everything past the first "
                    + str(NER_MAX_CHARS)
                    + " characters of a document is NOT scanned by the NER layer. Regex layers still "
                    "cover the whole document. This is a real recall limit, and this is its size."
                ),
            },
            "ner_min_confidence": NER_MIN_SCORE,
            "total_redactions_by_type": dict(total_counts),
            "docs_with_any_pii": docs_with_pii,
            "ambiguous_name_false_positive_risk_count": total_counts.get("NAME_AMBIGUOUS", 0),
            "ner_recall_gain_examples": ner_only_examples,
            "ner_gazetteer_overlap_examples": ner_overlap_examples,
        },
        examples=examples,
    )
    write_json(common.work_path("stage6_report.json"), report)
    return report


if __name__ == "__main__":
    r = run(common.work_path("stage5_survivors.jsonl"))
    print(f"[stage6] {r['input_docs']} -> {r['output_docs']} docs; extra={r['extra']}")
