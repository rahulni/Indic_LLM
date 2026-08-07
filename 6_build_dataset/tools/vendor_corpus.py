# -*- coding: utf-8 -*-
"""Build ``corpus/`` -- the self-contained, git-tracked demo corpus.

Run once; its output is committed. The grader never runs this, because the
submission must not depend on paths outside its own directory or on the network.

    python tools/vendor_corpus.py

Everything written here passes through :func:`canonical_text` first, so the
bytes are LF + NFC regardless of the platform that produced them. That matters:
the identity of every shard downstream is ``sha256`` over these bytes, and the
authoring machine has ``core.autocrlf=true``.
"""
from __future__ import annotations

import io
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from tdes.dedup import deduplicate                                     # noqa: E402
from tdes.determinism import canonical_text, stable_shuffle          # noqa: E402
from tdes.hashing import sha256_text, write_jsonl, write_text, ensure_dir  # noqa: E402
from tools.corpus_seed import REASONING_TRACES, AGENTIC_TRACES        # noqa: E402

CORPUS = os.path.join(ROOT, "corpus")

# ---------------------------------------------------------------------------
# Sources.
#
# The .txt files were fetched by 2_token/fetch_data.py from the MediaWiki API at
# https://{lang}.wikipedia.org/w/api.php as plain-text extracts. Wikipedia text
# is CC BY-SA 4.0. That is the real provenance and it is what CORPUS.md states.
# ---------------------------------------------------------------------------

WIKI_LICENSE = "CC-BY-SA-4.0"
WIKI_SOURCE = "Wikipedia plain-text extract via MediaWiki API (fetched by 2_token/fetch_data.py)"

TEXT_SOURCES = [
    # (src relative to repo root, lane, language, script)
    ("2_token/data/en.txt",       "web",       "eng", "Latin"),
    ("2_token/data/en_prose.txt", "web",       "eng", "Latin"),
    ("2_token/data/es.txt",       "multiling", "spa", "Latin"),
    ("2_token/data/hi.txt",       "indic",     "hin", "Devanagari"),
    ("2_token/data/mr.txt",       "indic",     "mar", "Devanagari"),
    ("2_token/data/ne.txt",       "indic",     "nep", "Devanagari"),
    ("2_token/data/te.txt",       "indic",     "tel", "Telugu"),
    ("2_token/data/kn.txt",       "indic",     "kan", "Kannada"),
]

# Real code, taken from this repository. Vendored rather than referenced so the
# submission stays self-contained.
CODE_SOURCES = [
    "4_model_data/assignment/pipeline/common.py",
    "4_model_data/assignment/pipeline/stage2_normalize.py",
    "4_model_data/assignment/pipeline/stage5_dedup.py",
    "4_model_data/assignment/pipeline/stage6_pii.py",
    "4_model_data/assignment/pipeline/stage8_manifest.py",
    "5_data_mixture_curriculum/plan/audit.py",
    "5_data_mixture_curriculum/microproxy/train.py",
    "2_token/bpe.py",
]

MIN_DOC_CHARS = 120
LONGCTX_MIN_CHARS = 2400     # documents long enough to be worth a late rung

# Canary strings. Distinctive enough that an n-gram scan finds them and no
# natural document contains them by accident.
CANARY_EVAL = "ZZQX-EVAL-CANARY-7F3A9C21-DO-NOT-TRAIN"
CANARY_VALIDATION = "ZZQX-VALIDATION-CANARY-B48D6E05-READ-ONLY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_source(rel: str) -> str:
    path = os.path.join(REPO, rel)
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        return canonical_text(f.read())


def split_documents(text: str) -> list[str]:
    """Split on blank lines. Wikipedia extracts are paragraph-structured, so a
    blank-line boundary is a real semantic boundary rather than an arbitrary cut."""
    parts = re.split(r"\n\s*\n", text)
    out = []
    for p in parts:
        p = p.strip()
        # Drop bare section headers like "== History ==": they are structure,
        # not prose, and they would dominate a small corpus with boilerplate.
        if len(p) >= MIN_DOC_CHARS and not re.fullmatch(r"=+[^=]+=+", p):
            out.append(p)
    return out


def script_purity(text: str, script: str) -> float:
    """Fraction of letters belonging to the expected script."""
    ranges = {
        "Devanagari": (0x0900, 0x097F),
        "Telugu": (0x0C00, 0x0C7F),
        "Kannada": (0x0C80, 0x0CFF),
        "Latin": (0x0041, 0x024F),
    }
    lo, hi = ranges[script]
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)


def indic_quality_score(text: str, purity: float) -> float:
    """Composite quality score used to assign the Indic tier.

    Read the honesty note in CORPUS.md before trusting this for anything beyond
    the demo. Session 5's `verified` tier means *human-verified native content*
    (Sangraha Verified). We have no human verification available here, and the
    vendored Wikipedia text is uniformly high-purity -- 80% of documents score
    exactly 1.0 on script purity alone -- so thresholding purity would produce a
    98/2 split and would be an arbitrary number dressed up as a measurement.

    Instead the score combines three signals that are genuinely computed:

      * script purity          -- how much of the text is actually in-script
      * length adequacy        -- very short fragments are less reliable
      * prose density          -- list/table-like text is structurally poorer
                                  than running prose

    The tier boundary is the corpus median of this score, so both tiers are
    populated by construction. That is a *stand-in* chosen to exercise Session
    5's tier rule, not a claim that any document was verified by a human.
    """
    n_lines = text.count("\n") + 1
    avg_line = len(text) / n_lines
    # Running prose has long lines; list-like extracts have many short ones.
    prose_density = min(1.0, avg_line / 200.0)
    length_adequacy = min(1.0, len(text) / 800.0)
    return round(purity * (0.5 + 0.3 * length_adequacy + 0.2 * prose_density), 6)


def split_python_units(src: str, base: str) -> list[tuple[str, str]]:
    """Split a Python file at top-level ``def``/``class``/decorator boundaries.

    These are real structural boundaries in real code, which is exactly what
    best-fit packing is supposed to respect: a unit that fits inside a sequence
    should never be cut in half. Splitting also turns 8 large files into enough
    documents for the packer to have real work to do.
    """
    lines = src.split("\n")
    starts = [
        i for i, ln in enumerate(lines)
        if re.match(r"^(?:def |class |@\w)", ln)
    ]
    if not starts:
        return [(base, src)]
    units: list[tuple[str, str]] = []
    header = "\n".join(lines[: starts[0]]).strip()
    if len(header) >= MIN_DOC_CHARS:
        units.append((f"{base}__header", header))
    bounds = starts + [len(lines)]
    for k in range(len(starts)):
        body = "\n".join(lines[bounds[k]: bounds[k + 1]]).rstrip()
        if len(body) >= MIN_DOC_CHARS:
            name = re.sub(r"[^0-9A-Za-z_]", "_", lines[bounds[k]][:40]).strip("_")
            units.append((f"{base}__{k:02d}_{name}", body))
    return units


def make_doc(doc_id: str, lane: str, text: str, **kw) -> dict:
    text = canonical_text(text)
    d = {
        "doc_id": doc_id,
        "lane": lane,
        "text": text,
        "chars": len(text),
        "words": len(text.split()),
        "content_sha256": sha256_text(text),
        "never_train": False,
        "split": "train",
    }
    d.update(kw)
    return d


def turns_to_text(turns: list[tuple[str, str]]) -> tuple[str, list[dict]]:
    """Flatten role turns into one document, recording each turn's char span.

    The spans are what ``masks.py`` later converts into a loss mask. Storing
    character offsets (not token offsets) keeps this file tokenizer-independent,
    so retraining the tokenizer does not invalidate the corpus.
    """
    parts, spans, cursor = [], [], 0
    for role, body in turns:
        marker = {"user": "<user>", "assistant": "<assistant>",
                  "tool_call": "<tool_call>", "tool_obs": "<tool_obs>"}[role]
        chunk = f"{marker}\n{body}\n"
        parts.append(chunk)
        spans.append({"role": role, "start": cursor, "end": cursor + len(chunk)})
        cursor += len(chunk)
    return "".join(parts), spans


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build() -> dict:
    docs: list[dict] = []
    stats: dict[str, dict] = {}

    # --- prose lanes ------------------------------------------------------
    for rel, lane, lang, script in TEXT_SOURCES:
        raw = read_source(rel)
        chunks = split_documents(raw)
        for i, chunk in enumerate(chunks):
            purity = script_purity(chunk, script)
            # Long documents are diverted to the longctx lane, where they will
            # be reserved for the late sequence-length rung instead of being
            # chopped at 64 tokens in stage A.
            target_lane = "longctx" if len(chunk) >= LONGCTX_MIN_CHARS else lane
            docs.append(make_doc(
                f"{lang}_{os.path.basename(rel).split('.')[0]}_{i:04d}",
                target_lane, chunk,
                language=lang, script=script,
                source=WIKI_SOURCE, source_file=rel, license=WIKI_LICENSE,
                provenance_tier="public-encyclopaedic",
                contributor="upstream 2_token/fetch_data.py",
                indic_tier=None, script_purity=round(purity, 4),
                quality_score=(indic_quality_score(chunk, purity)
                               if lang not in ("eng", "spa") else None),
            ))
        stats[rel] = {"documents": len(chunks)}

    # Indic tier: median split on the composite quality score. Assigned after
    # every Indic document exists, because the boundary is defined relative to
    # the corpus rather than as a free-floating constant. See CORPUS.md.
    indic = [d for d in docs if d["language"] in ("hin", "mar", "nep", "tel", "kan")]
    scores = sorted(d["quality_score"] for d in indic)
    threshold = scores[len(scores) // 2] if scores else 0.0
    for d in indic:
        d["indic_tier"] = "verified" if d["quality_score"] >= threshold else "unverified"
    stats["indic_tier"] = {
        "threshold": threshold,
        "verified": sum(1 for d in indic if d["indic_tier"] == "verified"),
        "unverified": sum(1 for d in indic if d["indic_tier"] == "unverified"),
        "method": "median of composite quality score (purity x length x prose density)",
    }

    # --- code lane --------------------------------------------------------
    n_units = 0
    for rel in CODE_SOURCES:
        src = read_source(rel)
        base = os.path.basename(rel)[:-3]
        for unit_id, body in split_python_units(src, base):
            n_units += 1
            docs.append(make_doc(
                f"code_{unit_id}", "code", body,
                language="python", script="Latin",
                source=f"this repository ({rel})", source_file=rel,
                license="course-work (author-retained)",
                provenance_tier="first-party",
                contributor="upstream 2_token, 4_model_data, 5_data_mixture_curriculum",
                indic_tier=None, script_purity=None, quality_score=None,
                # Best-fit packing must not split a unit that fits whole.
                atomic_unit=True,
            ))
    stats["code"] = {"files": len(CODE_SOURCES), "documents": n_units}

    # --- authored trace lanes --------------------------------------------
    for name, traces, lane in (("reasoning", REASONING_TRACES, "reasoning"),
                               ("agentic", AGENTIC_TRACES, "agentic")):
        for i, turns in enumerate(traces):
            text, spans = turns_to_text(turns)
            docs.append(make_doc(
                f"{lane}_{i:04d}", lane, text,
                language="eng", script="Latin",
                source="authored for this demo (tools/corpus_seed.py)",
                source_file="tools/corpus_seed.py",
                license="course-work (author-retained)",
                provenance_tier="authored-synthetic",
                contributor="authored for this submission",
                role_spans=spans, indic_tier=None, script_purity=None,
            ))
        stats[name] = {"documents": len(traces)}

    # --- deduplicate BEFORE splitting ------------------------------------
    #
    # This ordering is load-bearing and was found the hard way. The first
    # version split first and deduplicated afterwards, inside the training pool
    # only. Because `en.txt` and `en_prose.txt` share 23 identical paragraphs,
    # a duplicate pair could be separated by the split -- one copy into
    # validation, its twin left in train -- and deduplicating within the train
    # pool afterwards could never see it.
    #
    # The result was a genuine train/validation leak, which the firewall's
    # n-gram check caught at 100% overlap. The firewall is the right safety net,
    # but the real fix is here: collapse duplicates across the whole pool first,
    # so no two splits can ever hold the same content.
    pre = len(docs)
    docs = deduplicate(docs)["documents"]
    stats["pre_split_dedup"] = {
        "input": pre,
        "removed": pre - len(docs),
        "surviving": len(docs),
        "why": "duplicates must be collapsed before splitting, or a pair can "
               "straddle the train/held-out boundary and leak",
    }

    # --- held-out splits --------------------------------------------------
    # Carved deterministically from the deduplicated pool, then removed from it,
    # so a document can never appear on both sides. Canaries are appended so a
    # leak is detectable by content and not only by hash.
    docs = stable_shuffle(docs, seed="corpus-split-v1")
    by_lane: dict[str, list[dict]] = {}
    for d in docs:
        by_lane.setdefault(d["lane"], []).append(d)

    eval_docs, val_docs, train_docs = [], [], []
    for lane, group in sorted(by_lane.items()):
        n = len(group)
        n_eval = max(1, n // 12)
        n_val = max(1, n // 12)
        for j, d in enumerate(group):
            if j < n_eval:
                d = dict(d, split="eval", never_train=True,
                         text=canonical_text(d["text"] + "\n" + CANARY_EVAL),
                         canary=CANARY_EVAL, benchmark_id=f"demo-bench-{lane}")
                d["content_sha256"] = sha256_text(d["text"])
                d["doc_id"] = "eval_" + d["doc_id"]
                eval_docs.append(d)
            elif j < n_eval + n_val:
                d = dict(d, split="validation", never_train=True,
                         text=canonical_text(d["text"] + "\n" + CANARY_VALIDATION),
                         canary=CANARY_VALIDATION, benchmark_id=f"demo-val-{lane}")
                d["content_sha256"] = sha256_text(d["text"])
                d["doc_id"] = "val_" + d["doc_id"]
                val_docs.append(d)
            else:
                train_docs.append(d)

    # --- write ------------------------------------------------------------
    ensure_dir(CORPUS)
    train_by_lane: dict[str, list[dict]] = {}
    for d in train_docs:
        train_by_lane.setdefault(d["lane"], []).append(d)

    written = {}
    for lane, group in sorted(train_by_lane.items()):
        group = sorted(group, key=lambda x: x["doc_id"])
        path = os.path.join(CORPUS, lane, "documents.jsonl")
        write_jsonl(path, group)
        written[lane] = len(group)

    write_jsonl(os.path.join(CORPUS, "eval", "documents.jsonl"),
                sorted(eval_docs, key=lambda x: x["doc_id"]))
    write_jsonl(os.path.join(CORPUS, "validation", "documents.jsonl"),
                sorted(val_docs, key=lambda x: x["doc_id"]))

    summary = {
        "train_by_lane": written,
        "eval": len(eval_docs),
        "validation": len(val_docs),
        "total": len(docs),
        "source_stats": stats,
    }
    write_corpus_md(summary, train_by_lane, eval_docs, val_docs)
    return summary


def write_corpus_md(summary, train_by_lane, eval_docs, val_docs) -> None:
    lines = [
        "# Corpus provenance",
        "",
        "Everything under `corpus/` is vendored and committed so the demo runs with no",
        "network access and no dependency on paths outside this directory. Regenerate with",
        "`python tools/vendor_corpus.py`.",
        "",
        "All files are written LF-normalised and NFC-normalised. Their bytes are the",
        "identity of every shard downstream, so `.gitattributes` marks `corpus/**` as",
        "binary to stop git rewriting line endings on checkout.",
        "",
        "## Sources",
        "",
        "| Lane | Origin | License | Notes |",
        "|---|---|---|---|",
        f"| `web`, `multiling`, `indic`, `longctx` | {WIKI_SOURCE} | {WIKI_LICENSE} | Plain-text extracts, split on blank lines into paragraph documents. |",
        "| `code` | This repository's own Python files (sessions 2, 4, 5) | Course work, author-retained | Real source files, vendored whole so file boundaries are genuine. |",
        "| `reasoning`, `agentic` | **Authored for this demo** (`tools/corpus_seed.py`) | Course work, author-retained | See the honesty note below. |",
        "",
        "## Honesty note on the authored lanes",
        "",
        "The `reasoning` and `agentic` documents were **written by hand for this",
        "submission**. They are not harvested data and are not presented as such.",
        "",
        "They exist because those two lanes are the only ones whose *structure* is",
        "load-bearing: a reasoning trace needs a prompt to mask out of the loss, and an",
        "agentic trace additionally needs tool observations, which the model must read but",
        "must never be trained to predict. Each document carries explicit `role_spans`, and",
        "`tdes/masks.py` keys its loss mask off exactly those roles.",
        "",
        "## Indic tiers are a documented stand-in, not a verification claim",
        "",
        "Session 5 distinguishes *verified* from *unverified* Indic supply (Sangraha",
        "Verified means human-verified native content) and forbids the unverified tier",
        "from substituting for the verified part of the protected floor.",
        "",
        "**We have no human verification available for this corpus, and we do not claim",
        "any.** The vendored Wikipedia text is also uniformly high-purity -- over 80% of",
        "documents score exactly 1.0 on script purity -- so thresholding purity alone",
        "would yield a 98/2 split: an arbitrary number dressed up as a measurement.",
        "",
        "Instead `indic_tier` is assigned by splitting the corpus at the **median of a",
        "composite quality score** that combines three genuinely computed signals:",
        "script purity, length adequacy, and prose density (running prose versus",
        "list-like extracts). Both `script_purity` and `quality_score` are stored on every",
        "document, so the assignment can be re-derived and checked by a test.",
        "",
        "This is a stand-in chosen so Session 5's tier *rule* can actually be exercised.",
        "It is not a statement that any document here was verified by a human.",
        "",
        f"Threshold used: `{summary['source_stats'].get('indic_tier', {}).get('threshold', 0):.6f}` "
        f"({summary['source_stats'].get('indic_tier', {}).get('verified', 0)} verified / "
        f"{summary['source_stats'].get('indic_tier', {}).get('unverified', 0)} unverified).",
        "",
        "## Splits",
        "",
        "Splits are carved deterministically per lane and are disjoint by construction --",
        "a document is moved out of the train pool, never copied. Held-out documents carry",
        "a distinctive canary string so a leak is detectable by content scan and not only",
        "by content hash.",
        "",
        "**Deduplication runs before the split, and that ordering matters.** An earlier",
        "version split first. Because `en.txt` and `en_prose.txt` share 23 identical",
        "paragraphs, a duplicate pair could be separated by the split -- one copy into",
        "validation, its twin left in train -- and deduplicating inside the training pool",
        "afterwards could never see it. That produced a real train/validation leak, which",
        "the firewall's n-gram check caught at 100% overlap. The firewall is the right",
        "safety net; collapsing duplicates across the whole pool first is the actual fix.",
        "",
        "| Split | Documents | Canary | Permission |",
        "|---|---|---|---|",
        f"| train | {sum(summary['train_by_lane'].values())} | - | admitted to training |",
        f"| validation | {summary['validation']} | `{CANARY_VALIDATION}` | readable for evaluation, **never gradient-bearing** |",
        f"| eval | {summary['eval']} | `{CANARY_EVAL}` | **never read during training** |",
        "",
        "## Train documents by lane",
        "",
        "| Lane | Documents | Packing policy |",
        "|---|---|---|",
    ]
    from tdes.config import LANE_PACKING
    for lane in sorted(train_by_lane):
        lines.append(f"| `{lane}` | {len(train_by_lane[lane])} | `{LANE_PACKING.get(lane, '-')}` |")
    lines += [
        "",
        "## Document schema",
        "",
        "```json",
        '{"doc_id": "...", "lane": "indic", "text": "...", "chars": 512, "words": 78,',
        ' "content_sha256": "...", "language": "tel", "script": "Telugu",',
        ' "source": "...", "source_file": "...", "license": "CC-BY-SA-4.0",',
        ' "provenance_tier": "public-encyclopaedic", "indic_tier": "verified",',
        ' "script_purity": 0.9931, "never_train": false, "split": "train"}',
        "```",
        "",
    ]
    write_text(os.path.join(CORPUS, "CORPUS.md"), "\n".join(lines))


if __name__ == "__main__":
    s = build()
    print("train by lane:")
    for k, v in sorted(s["train_by_lane"].items()):
        print(f"  {k:12s} {v:5d}")
    print(f"  {'validation':12s} {s['validation']:5d}")
    print(f"  {'eval':12s} {s['eval']:5d}")
    print(f"  {'TOTAL':12s} {s['total']:5d}")
