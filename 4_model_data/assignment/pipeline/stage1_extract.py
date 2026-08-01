"""Stage 1: Extract.

What this stage does depends on what it is fed, which is the point that
keeps recurring: the same cleaning rule is right for one corpus and
destructive for another.

  telugu_web     Sangraha's `unverified` tier ships already-extracted
                 plain text - AI4Bharat's crawler already pulled article
                 bodies out of the page. There is no HTML to extract, so
                 this is a verification pass: find and strip residual
                 markup an upstream extractor missed, and say plainly
                 that nothing was dropped.

  reasoning_sft  SFT text contains fenced code blocks, and code contains
                 things that look exactly like HTML tags (`<div>`,
                 `<T>`, `<-`). Running the markup stripper over a code
                 block silently corrupts the code. So code blocks are
                 masked out before the markup pass and restored after,
                 and the number of protected blocks is reported. This is
                 the concrete form of the standard warning that you must
                 know whether you are cleaning prose or code.
"""
from __future__ import annotations

import re

import common
from common import StageTimer, make_report, read_jsonl, write_jsonl, write_json

HTML_TAG_RE = re.compile(r"<[^>\n]{1,80}>")
HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{2,5}|#x[0-9a-fA-F]{2,5});")
BOILERPLATE_LINE_RE = re.compile(
    r"^\s*(home|about( us)?|contact( us)?|privacy policy|terms( (of|&) (use|service))?|"
    r"copyright ©?.*|cookie policy|subscribe|read more)\s*$",
    re.IGNORECASE,
)

# Fenced code blocks (``` ... ```) and inline code spans (` ... `).
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]{1,200}`")
# Conversation markers must survive this stage untouched - stage 2 owns them.
# Without this guard the HTML tag regex would eat `<|im_start|>` here, and the
# ghost-tag statistics downstream would silently read zero.
CHAT_MARKER_RE = re.compile(r"<\|[a-z_]{1,20}\|>", re.IGNORECASE)


def _mask(text: str, patterns: list[re.Pattern]) -> tuple[str, list[str]]:
    """Replace protected spans with placeholders that no cleaning rule matches."""
    stash: list[str] = []

    def repl(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\x00PROTECTED{len(stash) - 1}\x00"

    for pat in patterns:
        text = pat.sub(repl, text)
    return text, stash


def _unmask(text: str, stash: list[str]) -> str:
    for i, original in enumerate(stash):
        text = text.replace(f"\x00PROTECTED{i}\x00", original)
    return text


def strip_residual_markup(text: str, protect_code: bool) -> tuple[str, dict]:
    patterns = [CHAT_MARKER_RE]
    if protect_code:
        patterns = [CODE_BLOCK_RE, INLINE_CODE_RE, CHAT_MARKER_RE]
    masked, stash = _mask(text, patterns)

    tags_found = len(HTML_TAG_RE.findall(masked))
    entities_found = len(HTML_ENTITY_RE.findall(masked))
    masked = HTML_TAG_RE.sub(" ", masked)

    kept_lines = []
    boilerplate_dropped = 0
    for line in masked.split("\n"):
        if BOILERPLATE_LINE_RE.match(line.strip()):
            boilerplate_dropped += 1
            continue
        kept_lines.append(line)
    masked = "\n".join(kept_lines)

    return _unmask(masked, stash), {
        "tags": tags_found,
        "entities": entities_found,
        "boilerplate": boilerplate_dropped,
        "protected_spans": len(stash),
    }


def run() -> dict:
    timer = StageTimer("extract")
    cfg = common.corpus()
    protect_code = cfg["has_code"]
    docs = read_jsonl(common.raw_sample())

    total_tags = total_entities = total_boilerplate = total_protected = 0
    docs_with_tags = 0
    docs_with_protected_code = 0
    examples = []
    survivors = []

    for d in docs:
        cleaned, s = strip_residual_markup(d["text"], protect_code)
        if s["tags"] or s["entities"] or s["boilerplate"]:
            docs_with_tags += 1
            if len(examples) < 5 and (s["tags"] or s["boilerplate"]):
                examples.append(
                    {
                        "doc_id": d["doc_id"],
                        "html_tags_found": s["tags"],
                        "boilerplate_lines_dropped": s["boilerplate"],
                        "protected_spans": s["protected_spans"],
                        "note": "residual markup stripped; protected spans left untouched",
                    }
                )
        if s["protected_spans"]:
            docs_with_protected_code += 1
        total_tags += s["tags"]
        total_entities += s["entities"]
        total_boilerplate += s["boilerplate"]
        total_protected += s["protected_spans"]

        d2 = dict(d)
        d2["text"] = cleaned
        survivors.append(d2)

    write_jsonl(common.work_path("stage1_survivors.jsonl"), survivors)

    if protect_code:
        note = (
            "This corpus contains fenced code blocks, and code legitimately contains "
            "angle-bracket sequences that the HTML tag stripper would destroy. Code spans "
            "and conversation markers are masked before the markup pass and restored after, "
            "so the stripper only ever sees prose. That masking is the whole point of the "
            "stage here: the corpus decides which rules are safe."
        )
    else:
        note = (
            "Sangraha's unverified tier is pre-extracted plain text, not raw HTML, so this "
            "stage runs a verification pass rather than a full trafilatura-style extraction. "
            "It found residual markup in a small number of documents and stripped it; zero "
            "documents were dropped at this stage."
        )

    report = make_report(
        stage_num=1,
        stage_name="Extract",
        input_docs=len(docs),
        output_docs=len(survivors),
        elapsed_s=timer.done(),
        extra={
            "note": note,
            "corpus_id": cfg["id"],
            "code_protection_enabled": protect_code,
            "docs_with_any_residual_markup": docs_with_tags,
            "docs_with_protected_spans": docs_with_protected_code,
            "total_protected_spans": total_protected,
            "total_html_tags_stripped": total_tags,
            "total_html_entities_seen": total_entities,
            "total_boilerplate_lines_dropped": total_boilerplate,
        },
        examples=examples,
    )
    write_json(common.work_path("stage1_report.json"), report)
    return report


if __name__ == "__main__":
    r = run()
    print(f"[stage1] {r['input_docs']} -> {r['output_docs']} docs ({r['survival_pct']}% survive)")
