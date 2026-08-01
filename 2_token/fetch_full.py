"""Rebuild data/<lang>.txt as a FULLER corpus: readable prose (extract) PLUS image
captions and data-table (wikitable) text -- but NOT the infobox fact-dump,
references, or navigation. Answers "you missed the captions/tables below images".

The infobox is deliberately excluded (per decision): it's mostly numbers/names/dates
that only inflate fertility.
"""
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

UA = "ERA-BPE-Assignment/1.0 (rahul.phil@gmail.com)"
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
LANGS = {"en": ("India", "English"), "hi": ("भारत", "Hindi"), "te": ("భారతదేశం", "Telugu"),
         "es": ("India", "Spanish"), "kn": ("ಭಾರತ", "Kannada"), "mr": ("भारत", "Marathi"),
         "ne": ("भारत", "Nepali")}

VOID = {"img", "br", "hr", "input", "meta", "link", "source", "area", "base",
        "col", "embed", "param", "track", "wbr"}
SKIP_TAGS = {"script", "style", "sup"}
SKIP_CLASS = ["reference", "reflist", "navbox", "infobox", "metadata",
              "mw-editsection", "noprint", "sistersitebox", "mw-empty-elt",
              "gallerybox", "toc", "hatnote", "ambox"]


def api(lang, params):
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def prose(lang, title):
    d = api(lang, {"action": "query", "prop": "extracts", "explaintext": "1",
                   "titles": title, "format": "json", "redirects": "1"})
    return next(iter(d["query"]["pages"].values())).get("extract", "")


class CapsTables(HTMLParser):
    """Collect ONLY figcaption text and wikitable text, skipping ref/nav/infobox."""

    def __init__(self):
        super().__init__()
        self.stack = []          # frames: (tag, eff_skip, eff_cap, eff_table)
        self.caps = []
        self.tables = []

    def _eff(self):
        if not self.stack:
            return False, False, False
        return self.stack[-1][1], self.stack[-1][2], self.stack[-1][3]

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        cls = dict(attrs).get("class", "") or ""
        pskip, pcap, ptab = self._eff()
        skip = pskip or tag in SKIP_TAGS or any(s in cls for s in SKIP_CLASS)
        cap = pcap or tag == "figcaption" or "thumbcaption" in cls
        tab = ptab or (tag == "table" and "wikitable" in cls)
        self.stack.append((tag, skip, cap, tab))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        skip, cap, tab = self._eff()
        if skip:
            return
        t = data.strip()
        if not t:
            return
        if cap:
            self.caps.append(t)
        elif tab:
            self.tables.append(t)


def caps_tables(lang, title):
    d = api(lang, {"action": "parse", "page": title, "prop": "text",
                   "format": "json", "redirects": "1"})
    html = d["parse"]["text"]["*"]
    p = CapsTables()
    p.feed(html)
    caps = re.sub(r"\s+", " ", " ".join(p.caps))
    tables = re.sub(r"\s+", " ", " ".join(p.tables))
    return caps, tables


def main():
    for lang, (title, name) in LANGS.items():
        try:
            body = prose(lang, title)
            caps, tables = caps_tables(lang, title)
        except Exception as e:  # noqa: BLE001
            print(f"[{lang}] ERROR: {e}")
            continue
        combined = "\n\n".join(x for x in (body, caps, tables) if x)
        with open(os.path.join(DATA_DIR, f"{lang}.txt"), "w", encoding="utf-8") as f:
            f.write(combined)
        print(f"[{lang}] {name:8s} prose={len(body):>6d}  +caps={len(caps):>5d}  "
              f"+tables={len(tables):>6d}  total={len(combined):>6d} chars")


if __name__ == "__main__":
    main()
