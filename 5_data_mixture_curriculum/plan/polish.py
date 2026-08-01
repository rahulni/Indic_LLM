# -*- coding: utf-8 -*-
"""
polish.py - presentation pass over the generated Markdown.

Applied after build_plan.py has produced the document, so it works uniformly
across all ~40 tables instead of being hand-applied at each emit site. Nothing
here changes a number; it only changes how the numbers read.

What it does, and why each earns its place:
  - right-aligns numeric table columns, so digits line up down the column. In a
    document that is almost entirely figures, left-ragged numbers are the single
    biggest legibility cost.
  - builds a linked table of contents, because 680 lines with 40 sections is not
    navigable by scrolling.
  - promotes marked paragraphs to GitHub alert callouts, which render as tinted
    panels on GitHub and in the HTML view.
"""

import re

NUM = re.compile(r"^[\s*_`]*[+\-−~<>≈$]?[\d,]+(?:\.\d+)?\s*(?:%|×|x|T|B|M|K|d|h|ep|"
                 r"bpb|GB|MB|tok|/\S*)?[\s*_`]*$")


def _cells(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_divider(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", line)) and "-" in line


def align_tables(md):
    """Rewrite each table's divider row so numeric columns are right-aligned.

    A column counts as numeric when a clear majority of its non-empty body cells
    parse as a number with an optional unit. Header text is ignored - headers are
    words even when the column is figures."""
    lines = md.split("\n")
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.strip().startswith("|") and i + 1 < n and _is_divider(lines[i + 1]):
            head = _cells(line)
            body = []
            j = i + 2
            while j < n and lines[j].strip().startswith("|"):
                body.append(_cells(lines[j]))
                j += 1

            aligns = []
            for c in range(len(head)):
                vals = [r[c] for r in body if c < len(r) and r[c].strip()]
                if not vals:
                    aligns.append("---")
                    continue
                hits = sum(1 for v in vals if NUM.match(v))
                aligns.append("---:" if hits >= max(1, int(len(vals) * 0.6)) else "---")

            out.append(line)
            out.append("|" + "|".join(aligns) + "|")
            out.extend(lines[i + 2:j])
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def build_toc(md, min_level=2, max_level=3):
    """A linked contents block, derived from the headings actually present."""
    entries = []
    in_code = False
    for line in md.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = re.match(r"^(#{2,3})\s+(.*)$", line)
        if not m:
            continue
        lvl = len(m.group(1))
        if lvl < min_level or lvl > max_level:
            continue
        title = m.group(2).strip()
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[\s]+", "-", slug).strip("-")
        entries.append((lvl, title, slug))

    lines = ["<details open>", "<summary><b>Contents</b></summary>", ""]
    for lvl, title, slug in entries:
        indent = "" if lvl == 2 else "  "
        lines.append(f"{indent}- [{title}](#{slug})")
    lines += ["", "</details>"]
    return "\n".join(lines)


ALERT = re.compile(r"^\[\[(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\]\s*(.*)$",
                   re.DOTALL)


def promote_alerts(md):
    """Turn `[[WARNING]] text` paragraphs into GitHub alert callouts.

    build_plan.py marks a paragraph by prefixing it; this converts the marker
    into the blockquote form GitHub tints. Keeping the marker in the renderer
    and the syntax here means the emit sites stay readable."""
    out = []
    for para in md.split("\n\n"):
        m = ALERT.match(para.strip())
        if m:
            kind, body = m.group(1), m.group(2).strip()
            body = " ".join(body.split("\n"))
            out.append(f"> [!{kind}]\n> {body}")
        else:
            out.append(para)
    return "\n\n".join(out)


def apply(md, toc_after=None):
    """Run the full pass. `toc_after` is a literal line to insert the TOC below."""
    md = promote_alerts(md)
    md = align_tables(md)
    if toc_after:
        idx = md.find(toc_after)
        if idx != -1:
            cut = idx + len(toc_after)
            md = md[:cut] + "\n\n" + build_toc(md) + md[cut:]
    return md
