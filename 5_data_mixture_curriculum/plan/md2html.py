# -*- coding: utf-8 -*-
"""
md2html.py - a small, deterministic Markdown renderer.

Deliberately not a general Markdown implementation. It handles exactly the
constructs MIXTURE_PLAN.md and README.md actually use, and nothing else, so the
site is a *view* of the generated documents rather than a second copy of them
that can drift. Feed it the same .md the build already emits and the HTML cannot
disagree with the Markdown.
"""

import html
import re

_INLINE_CODE = re.compile(r"`([^`]+)`")
# Non-greedy and permits nested emphasis: the plan contains spans like
# "**infeasible ... where the *web* lane runs dry**". A [^*]+ body silently
# fails on those and leaks literal asterisks into the page.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_SECT = re.compile(r"§(\d+(?:\.\d+)?)")


def slugify(text):
    s = re.sub(r"[^\w\s.-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", s).strip("-")


_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def inline(text):
    """Escape first, then re-introduce only the markup we permit."""
    # Images are dropped rather than rendered. The only images in these
    # documents are shields.io badges, and a published artifact's CSP blocks
    # every external host - they would render as broken-image icons. The stat
    # row at the top of the HTML view carries the same figures natively.
    text = _IMG.sub("", text)
    out = html.escape(text, quote=False)
    holds = []

    def hold(fragment):
        holds.append(fragment)
        return f"\x00{len(holds)-1}\x00"

    out = _INLINE_CODE.sub(lambda m: hold(f"<code>{m.group(1)}</code>"), out)
    out = _LINK.sub(
        lambda m: hold(
            f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>'),
        out)
    out = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _ITAL.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    # Cross-references become real jump links - the single most useful
    # affordance in a document that refers to itself constantly.
    out = _SECT.sub(
        lambda m: f'<a class="xref" href="#sec-{m.group(1)}">§{m.group(1)}</a>',
        out)
    for i, frag in enumerate(holds):
        out = out.replace(f"\x00{i}\x00", frag)
    return out


_VERDICTS = ("SUPPLY-OK", "REPEAT", "GENERATE", "COLLECT", "INFEASIBLE")


def _verdict_chip(cell):
    """Encode state in FORM as well as colour: chip + glyph + label, so the
    status never rides on hue alone.

    The cell must be EXACTLY a verdict token (bare, backticked, or bolded).
    An earlier version matched anywhere in the cell and replaced the whole
    cell with the chip - which silently deleted prose like "COLLECT, not
    clean: court judgments and assembly proceedings" down to a single chip.
    A renderer that drops content is worse than one that renders it plainly."""
    bare = cell.strip().strip("*").strip("`").strip()
    if bare not in _VERDICTS:
        return None
    kind = bare
    glyph = {"SUPPLY-OK": "●", "REPEAT": "◐", "GENERATE": "○",
             "COLLECT": "◇", "INFEASIBLE": "✕"}[kind]
    cls = {"SUPPLY-OK": "ok", "REPEAT": "warn", "GENERATE": "serious",
           "COLLECT": "info", "INFEASIBLE": "critical"}[kind]
    return (f'<span class="chip chip-{cls}">'
            f'<span class="chip-glyph" aria-hidden="true">{glyph}</span>{kind}</span>')


def _cells(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_divider(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", line)) and "-" in line


def render(md):
    """Markdown -> (html, toc) where toc is a list of (level, id, title)."""
    lines = md.split("\n")
    out, toc = [], []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        # raw HTML block (the <details> contents block) - passed through as-is
        if re.match(r"^\s*</?(details|summary)\b", line, re.I):
            out.append(line)
            i += 1
            continue

        # fenced code
        if line.startswith("```"):
            lang = line[3:].strip()
            body = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            code = html.escape("\n".join(body), quote=False)
            if lang.lower() == "mermaid":
                # Artifacts render mermaid natively from <pre class="mermaid">.
                out.append(f'<pre class="mermaid">{code}</pre>')
            else:
                out.append(f'<pre class="code" data-lang="{html.escape(lang)}">'
                           f"<code>{code}</code></pre>")
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            title = m.group(2).strip()
            sec = re.match(r"^(\d+(?:\.\d+)?)\.?\s", title)
            hid = f"sec-{sec.group(1)}" if sec else slugify(title)
            if lvl <= 3:
                toc.append((lvl, hid, title))
            out.append(
                f'<h{lvl} id="{hid}" class="h{lvl}">'
                f'<a class="anchor" href="#{hid}" aria-label="Link to this section">#</a>'
                f"{inline(title)}</h{lvl}>")
            i += 1
            continue

        # table
        if line.strip().startswith("|") and i + 1 < n and _is_divider(lines[i + 1]):
            head = _cells(line)
            # Carry the Markdown alignment markers into the HTML, so the
            # right-aligned numeric columns polish.py produces line up here too.
            aligns = []
            for spec_cell in _cells(lines[i + 1]):
                right = spec_cell.endswith(":")
                left = spec_cell.startswith(":")
                aligns.append("right" if (right and not left)
                              else ("center" if (right and left) else "left"))
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1

            def _al(idx):
                a = aligns[idx] if idx < len(aligns) else "left"
                return "" if a == "left" else f' class="ta-{a}"'

            th = "".join(f"<th{_al(k)}>{inline(c)}</th>"
                         for k, c in enumerate(head))
            body = []
            for r in rows:
                tds = []
                for k, c in enumerate(r):
                    chip = _verdict_chip(c)
                    tds.append(f"<td{_al(k)}>{chip if chip else inline(c)}</td>")
                body.append("<tr>" + "".join(tds) + "</tr>")
            out.append(
                '<div class="tablewrap" tabindex="0" role="region" '
                'aria-label="Scrollable table">'
                f"<table><thead><tr>{th}</tr></thead>"
                f'<tbody>{"".join(body)}</tbody></table></div>')
            continue

        # blockquote, incl. GitHub alert callouts
        if line.startswith(">"):
            body = []
            while i < n and lines[i].startswith(">"):
                body.append(lines[i].lstrip(">").strip())
                i += 1
            joined = " ".join(x for x in body if x)
            am = re.match(r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*)$",
                          joined, re.S)
            if am:
                kind = am.group(1)
                glyph = {"NOTE": "i", "TIP": "✦", "IMPORTANT": "!",
                         "WARNING": "▲", "CAUTION": "✕"}[kind]
                out.append(
                    f'<div class="alert alert-{kind.lower()}">'
                    f'<div class="alert-h"><span class="alert-g" aria-hidden="true">'
                    f'{glyph}</span>{kind.title()}</div>'
                    f"<div>{inline(am.group(2))}</div></div>")
            else:
                out.append(f"<blockquote>{inline(joined)}</blockquote>")
            continue

        # lists
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>")
            continue

        if not line.strip():
            i += 1
            continue

        para = []
        while i < n and lines[i].strip() and not lines[i].startswith(("#", "|", ">", "```")) \
                and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
            para.append(lines[i].strip())
            i += 1
        if para:
            out.append(f'<p>{inline(" ".join(para))}</p>')

    return "\n".join(out), toc
