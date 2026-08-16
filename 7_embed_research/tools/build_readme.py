"""Renders README.md as a styled HTML page in the dashboards' theme.

GitHub Pages serves a raw .md file as plain text, so the README link on the
landing page was the one surface in the submission with no design at all. This
turns it into a page that matches the dashboards, without touching README.md
itself -- the markdown stays the canonical source and keeps rendering normally
on GitHub.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.chart_kit import (  # noqa: E402
    BASE_CSS,
    PALETTE_CSS,
    REPO_TREE_URL,
    THEME_BOOT_JS,
    THEME_TOGGLE_HTML,
    THEME_TOGGLE_JS,
)

# The page is written into submission_artifacts/, so a link that README.md
# writes relative to the repo root has to be re-based. Source files point at
# GitHub (where they render with syntax highlighting); generated pages stay
# local, one directory up.
REPO_BLOB_BASE = "https://github.com/rahulni/Indic_LLM/blob/main/7_embed_research/"

# GitHub's alert syntax: a blockquote whose first line is "[!NOTE]". markdown-it
# has no rule for it, so it arrives as literal text and is promoted afterwards.
ALERT_RE = re.compile(
    r"<blockquote>\s*<p>\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*?)</blockquote>",
    re.IGNORECASE | re.DOTALL,
)

README_CSS = """
.md { max-width: 82ch; }
.md h1 { font-size: 2.1rem; margin: 0 0 18px 0; }
.md h2 { font-size: 1.4rem; margin: 44px 0 14px 0; padding-top: 20px;
  border-top: 1px solid var(--border); }
.md h3 { font-size: 1.06rem; margin: 30px 0 10px 0; }
.md p, .md li { color: var(--text-secondary); }
.md li { margin: 6px 0; }
.md strong { color: var(--text-primary); }
/* A bold link is still a link: without this, `[**Track A**](...)` inherits the
   strong colour and reads as plain bold text. */
.md a strong { color: inherit; }
.md table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 20px 0; }
.md th, .md td { text-align: left; padding: 9px 11px; border-bottom: 1px solid var(--gridline);
  vertical-align: top; }
.md th { color: var(--text-muted); font-weight: 600; font-size: 0.76rem;
  text-transform: uppercase; letter-spacing: 0.03em; }
.md td { color: var(--text-secondary); }
.md tbody tr:hover td { background: var(--surface-2); }
.md blockquote { margin: 20px 0; padding: 2px 20px; border-left: 3px solid var(--border-strong);
  background: var(--surface-1); border-radius: 0 10px 10px 0; }
.md hr { border: none; border-top: 1px solid var(--border); margin: 40px 0; }
.md .alert { margin: 22px 0; padding: 16px 20px; border-radius: 12px;
  border: 1px solid var(--border); border-left-width: 3px; background: var(--surface-1); }
.md .alert > p:first-child { margin-top: 0; }
.md .alert > p:last-child { margin-bottom: 0; }
.md .alert-title { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.07em;
  text-transform: uppercase; margin-bottom: 8px; }
.md .alert-note { border-left-color: var(--series-1); }
.md .alert-note .alert-title { color: var(--series-1); }
.md .alert-warning, .md .alert-caution { border-left-color: var(--series-2); }
.md .alert-warning .alert-title, .md .alert-caution .alert-title { color: var(--series-2); }
.md .alert-tip, .md .alert-important { border-left-color: var(--good); }
.md .alert-tip .alert-title, .md .alert-important .alert-title { color: var(--good); }
.md .table-scroll { overflow-x: auto; }
.readme-nav { margin: 0 0 28px 0; font-size: 0.85rem; }
"""


def rebase_links(html_text: str) -> str:
    """Points every relative link at something that resolves from this page.

    Generated .html pages sit one level up; everything else is source, and is
    sent to GitHub rather than to a raw file Pages would serve as plain text.
    """

    def replace(match: re.Match) -> str:
        href = match.group(1)
        if href.startswith(("http://", "https://", "#", "mailto:")):
            return match.group(0)
        target, _, anchor = href.partition("#")
        if target.endswith(".html"):
            new = f"../{target}"
        else:
            new = REPO_BLOB_BASE + target
        return f'href="{new}{"#" + anchor if anchor else ""}"'

    return re.sub(r'href="([^"]+)"', replace, html_text)


def promote_alerts(html_text: str) -> str:
    """Turns GitHub's `> [!NOTE]` blockquotes into styled callout blocks.

    Blockquotes are never nested in this README, so matching non-greedily to the
    first </blockquote> is unambiguous.
    """

    def replace(match: re.Match) -> str:
        kind = match.group(1).lower()
        inner = match.group(2).lstrip()
        # A one-paragraph alert leaves prose that still needs its <p>; an alert
        # whose body is a list closes the paragraph immediately, and reopening
        # one there would emit a stray empty paragraph.
        opener = "" if inner.startswith("</p>") else "<p>"
        return (
            f'<div class="alert alert-{kind}">'
            f'<div class="alert-title">{kind}</div>{opener}{inner}</div>'
        )

    return ALERT_RE.sub(replace, html_text)


def wrap_tables(html_text: str) -> str:
    """Wide tables scroll inside their own box rather than the page."""
    return html_text.replace("<table>", '<div class="table-scroll"><table>').replace(
        "</table>", "</table></div>"
    )


def main() -> int:
    src = (ROOT / "README.md").read_text(encoding="utf-8")
    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    rendered = md.render(src)
    rendered = promote_alerts(rendered)
    rendered = rebase_links(rendered)
    rendered = wrap_tables(rendered)

    title = "Kronecker Embedding V2 — README"
    body = f"""<title>{html.escape(title)}</title>
<script>{THEME_BOOT_JS}</script>
<style>
{PALETTE_CSS}
{BASE_CSS}
{README_CSS}
.page {{ max-width: 900px; margin: 0 auto; padding: 40px 20px 64px 20px; }}
</style>
{THEME_TOGGLE_HTML}
<div class="viz-root page">
<div class="readme-nav"><a href="index.html">&larr; All tracks</a></div>
<div class="md">
{rendered}
</div>
<div class="page-footer">
<a href="index.html">&larr; All tracks</a> &middot;
<a href="{REPO_TREE_URL}">Code + README on GitHub</a>
</div>
</div>
<script>{THEME_TOGGLE_JS}</script>
"""

    out = ROOT / "submission_artifacts" / "readme.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
