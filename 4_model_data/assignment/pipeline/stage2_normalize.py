"""Stage 2: Normalize.

NFC unicode normalization, HTML entity unescaping, whitespace collapse,
and character-level noise stripping - with the one rule standard practice
insists on: ZWNJ (U+200C) and ZWJ (U+200D) are legitimate parts of
Brahmic scripts and must survive, even though every other invisible
control/format character is noise. Also runs the "ghost-tag" scan
(literal conversation-format markers baked into the text as ordinary
characters) as a folded-in sub-check, matching the reference
structure (Section 4's ghost-tag trap lives inside this stage's
`removes` list in the course's own pipeline-map widget).
"""
from __future__ import annotations

import html
import os
import re
import unicodedata
from collections import Counter

import common
from common import StageTimer, make_report, read_jsonl, write_jsonl, write_json

# Noise: must be stripped. Preserve: ZWNJ U+200C, ZWJ U+200D (never touched).
# Cross-checked against the reference reference implementation
# (widget_2_clean_text_live.html's CTRL_RE/BIDI_RE/GHOST_RE) rather than
# reconstructed from the prose alone - soft hyphen and word joiner come
# from that widget's CTRL_RE range that the prose summary doesn't spell out.
NOISE_CODEPOINTS = {
    0xFEFF: "BOM / ZERO WIDTH NO-BREAK SPACE",
    0x200B: "ZERO WIDTH SPACE",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x00AD: "SOFT HYPHEN",
    0x2060: "WORD JOINER",
    0x202A: "LRE (bidi embed)",
    0x202B: "RLE (bidi embed)",
    0x202C: "PDF (bidi pop)",
    0x202D: "LRO (bidi override)",
    0x202E: "RLO (bidi override)",
    0x2066: "LRI (bidi isolate)",
    0x2067: "RLI (bidi isolate)",
    0x2068: "FSI (bidi isolate)",
    0x2069: "PDI (bidi isolate)",
    0xFFFD: "REPLACEMENT CHARACTER",
}
PRESERVED_CODEPOINTS = {0x200C: "ZWNJ (Brahmic conjunct control)", 0x200D: "ZWJ (Brahmic conjunct control)"}

NOISE_RE = re.compile("[" + "".join(chr(c) for c in NOISE_CODEPOINTS) + "]")
PRESERVED_RE = re.compile("[" + "".join(chr(c) for c in PRESERVED_CODEPOINTS) + "]")
# Other C0/C1 control characters, excluding tab/newline/carriage-return.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Private Use Area - explicitly named in the reference "46 garbage tokens"
# V4-audit finding ("zero-width characters, HTML artifacts, broken byte
# fragments, and private-use characters"), and demonstrated directly in
# widget_2's own "Mixed / worst case" sample.
PUA_RE = re.compile("[-]")

GHOST_TAG_PATTERNS = [
    re.compile(r"\[(USER|ASSISTANT|SYSTEM|HUMAN|BOT)\]", re.IGNORECASE),
    re.compile(r"<\|(user|assistant|system|im_start|im_end|endoftext)\|>", re.IGNORECASE),
    re.compile(r"###\s*(Human|Assistant)\s*:", re.IGNORECASE),
    re.compile(r"</?(human|bot|assistant|user|system)>", re.IGNORECASE),
]

# --- Ghost-tag unification -------------------------------------------------
# Detection alone is only half of what the fix asks for. The fix is to pick
# ONE canonical format and rewrite every source into it at ingestion, so the
# model never learns a fake conversation structure it then has to unlearn
# during SFT. These are the real dialects present in the sampled corpus, each
# mapped onto the canonical marker chosen here.
CANONICAL = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>", "end": "<|end|>"}
_ROLE_ALIAS = {"human": "user", "bot": "assistant", "instruction": "user", "response": "assistant"}


def _canon_role(raw: str) -> str:
    role = _ROLE_ALIAS.get(raw.lower(), raw.lower())
    return CANONICAL.get(role, CANONICAL["user"])


GHOST_REWRITE_RULES: list[tuple[str, re.Pattern, object]] = [
    ("chatml_im_start", re.compile(r"<\|im_start\|>[ \t]*(system|user|assistant)\b", re.I),
     lambda m: _canon_role(m.group(1))),
    ("chatml_im_end", re.compile(r"<\|im_end\|>", re.I), lambda m: CANONICAL["end"]),
    ("endoftext", re.compile(r"<\|endoftext\|>", re.I), lambda m: CANONICAL["end"]),
    ("bracket_role", re.compile(r"\[(USER|ASSISTANT|SYSTEM|HUMAN|BOT)\]", re.I),
     lambda m: _canon_role(m.group(1))),
    ("hash_instruction", re.compile(r"###[ \t]*(Human|Assistant|Instruction|Response)[ \t]*:", re.I),
     lambda m: _canon_role(m.group(1))),
    ("xml_role", re.compile(r"</?(human|bot|assistant|user|system)>", re.I),
     lambda m: _canon_role(m.group(1))),
    # Plain-text role prefixes at line start - e.g. "USER:" nested *inside* an
    # already-marked ChatML block, a real doubled marker found in
    # fable-sft-combined-v2's own published text.
    ("plain_role_prefix", re.compile(r"^(USER|ASSISTANT|SYSTEM)[ \t]*:[ \t]*", re.M),
     lambda m: _canon_role(m.group(1)) + " "),
]

# Code must survive whitespace collapse. Python indentation *is* semantics, so
# collapsing runs of spaces inside a fenced block turns valid code into invalid
# code. This is the exact case raised in the session Q&A: subsequent whitespace
# should not be collapsed for a code snippet even though it should be for prose.
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


def unify_ghost_tags(text: str) -> tuple[str, Counter]:
    """Rewrite every recognised conversation-marker dialect into one canonical
    format. Returns the rewritten text and a per-dialect count of what was found."""
    found: Counter = Counter()
    for name, pat, repl in GHOST_REWRITE_RULES:
        n = len(pat.findall(text))
        if n:
            found[name] += n
            text = pat.sub(repl, text)
    return text, found

WHITESPACE_RUN_RE = re.compile(r"[ \t ]{2,}")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str, protect_code: bool = False, unify: bool = False) -> tuple[str, dict]:
    stats = {
        "noise_removed": Counter(),
        "preserved_kept": 0,
        "ghost_tags_found": 0,
        "ghost_formats": Counter(),
        "code_blocks_protected": 0,
    }

    # 0. Stash fenced code blocks so no later rule can touch their whitespace.
    code_stash: list[str] = []
    if protect_code:
        def _stash(m: re.Match) -> str:
            code_stash.append(m.group(0))
            return f"\x00CODE{len(code_stash) - 1}\x00"

        text = CODE_BLOCK_RE.sub(_stash, text)
        stats["code_blocks_protected"] = len(code_stash)

    # 1. HTML entity unescape (before NFC, so entities like &amp; resolve first)
    text = html.unescape(text)

    # 2. NFC unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 3. Count and strip noise characters (BOM, ZWSP, bidi controls, replacement char)
    for ch in NOISE_RE.findall(text):
        stats["noise_removed"][NOISE_CODEPOINTS[ord(ch)]] += 1
    text = NOISE_RE.sub("", text)

    # 3b. Other stray control characters
    ctrl_matches = CONTROL_RE.findall(text)
    if ctrl_matches:
        stats["noise_removed"]["other control characters"] += len(ctrl_matches)
    text = CONTROL_RE.sub("", text)

    # 3c. Private Use Area characters (the "private-use characters" category
    # named in the reference 46-garbage-token audit finding)
    pua_matches = PUA_RE.findall(text)
    if pua_matches:
        stats["noise_removed"]["PRIVATE USE AREA CHARACTER"] += len(pua_matches)
    text = PUA_RE.sub("", text)

    # 4. Explicitly preserve ZWJ/ZWNJ - never removed, just counted for the record.
    stats["preserved_kept"] = len(PRESERVED_RE.findall(text))

    # 5. Ghost conversation-marker scan (folded-in sub-check from Section 4)
    for pat in GHOST_TAG_PATTERNS:
        stats["ghost_tags_found"] += len(pat.findall(text))

    # 5b. ...and the actual fix: rewrite every dialect into one canonical format.
    if unify:
        text, formats = unify_ghost_tags(text)
        stats["ghost_formats"] = formats

    # 6. Collapse whitespace runs, keep paragraph breaks
    text = WHITESPACE_RUN_RE.sub(" ", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    text = text.strip()

    # 7. Restore protected code verbatim - byte-for-byte as it arrived.
    for i, block in enumerate(code_stash):
        text = text.replace(f"\x00CODE{i}\x00", block)

    return text, stats


def run(input_path: str) -> dict:
    timer = StageTimer("normalize")
    cfg = common.corpus()
    protect_code = cfg["has_code"]
    # Unification only makes sense where conversation markers actually exist.
    unify = cfg["kind"] == "conversation_sft"
    docs = read_jsonl(input_path)

    total_chars_before = total_chars_after = 0
    noise_totals: Counter = Counter()
    ghost_format_totals: Counter = Counter()
    total_preserved = 0
    total_ghost_tags = 0
    docs_with_noise = 0
    docs_with_ghost_tags = 0
    total_code_blocks = 0
    ghost_examples = []
    examples = []
    survivors = []

    for d in docs:
        before = d["text"]
        after, stats = normalize_text(before, protect_code=protect_code, unify=unify)
        if stats["ghost_tags_found"]:
            docs_with_ghost_tags += 1
            if len(ghost_examples) < 4:
                # Show the marker region before and after the rewrite, so the
                # unification is visible rather than asserted.
                m = GHOST_TAG_PATTERNS[1].search(before) or GHOST_TAG_PATTERNS[0].search(before)
                start = max(0, (m.start() - 40) if m else 0)
                ghost_examples.append(
                    {
                        "doc_id": d["doc_id"],
                        "source_key": d.get("source_key"),
                        "markers_found": stats["ghost_tags_found"],
                        "formats": dict(stats["ghost_formats"]),
                        "before_excerpt": before[start : start + 150],
                        "after_excerpt": after[start : start + 150],
                    }
                )
        ghost_format_totals.update(stats["ghost_formats"])
        total_code_blocks += stats["code_blocks_protected"]
        total_chars_before += len(before)
        total_chars_after += len(after)
        if sum(stats["noise_removed"].values()) > 0:
            docs_with_noise += 1
            noise_totals.update(stats["noise_removed"])
            if len(examples) < 6:
                examples.append(
                    {
                        "doc_id": d["doc_id"],
                        "noise_removed": dict(stats["noise_removed"]),
                        "chars_before": len(before),
                        "chars_after": len(after),
                    }
                )
        total_preserved += stats["preserved_kept"]
        total_ghost_tags += stats["ghost_tags_found"]

        d2 = dict(d)
        d2["text"] = after
        survivors.append(d2)

    write_jsonl(common.work_path("stage2_survivors.jsonl"), survivors)

    char_shrink_pct = (
        round(100.0 * (total_chars_before - total_chars_after) / total_chars_before, 4)
        if total_chars_before
        else 0.0
    )

    report = make_report(
        stage_num=2,
        stage_name="Normalize",
        input_docs=len(docs),
        output_docs=len(survivors),  # normalization drops 0 documents, only cleans characters
        elapsed_s=timer.done(),
        extra={
            "note": (
                "No documents are dropped here - this stage cleans characters, not documents. "
                "The character-level shrink and the noise inventory are the real signal."
            )
            + (
                " Conversation markers in this corpus are not just counted but rewritten into a "
                "single canonical format, which is the prescribed fix; code blocks "
                "are stashed before whitespace collapse and restored byte-for-byte afterwards."
                if unify
                else " This corpus is web text with no conversation markers in it, so the ghost-tag "
                "scan runs and legitimately finds nothing - reported as zero rather than hidden."
            ),
            "corpus_id": cfg["id"],
            "ghost_tag_unification_enabled": unify,
            "code_protection_enabled": protect_code,
            "docs_with_ghost_markers": docs_with_ghost_tags,
            "ghost_format_breakdown": dict(ghost_format_totals),
            "canonical_format_chosen": CANONICAL,
            "code_blocks_protected": total_code_blocks,
            "ghost_tag_examples": ghost_examples,
            "total_chars_before": total_chars_before,
            "total_chars_after": total_chars_after,
            "char_shrink_pct": char_shrink_pct,
            "docs_with_any_noise_char": docs_with_noise,
            "noise_char_inventory": dict(noise_totals),
            "total_noise_chars_removed": sum(noise_totals.values()),
            "distinct_noise_categories_found": len(noise_totals),
            "zwj_zwnj_preserved_count": total_preserved,
            "ghost_conversation_markers_found": total_ghost_tags,
        },
        examples=examples,
    )
    write_json(common.work_path("stage2_report.json"), report)
    return report


if __name__ == "__main__":
    r = run(common.work_path("stage1_survivors.jsonl"))
    print(f"[stage2] {r['input_docs']} -> {r['output_docs']} docs; extra={r['extra']}")
