"""ENGLISH ONLY. Fetch the India article with the MOST complete text possible:
prose + ALL captions (figure, thumb, AND gallery captions) + data tables.

Bug fix vs fetch_full.py: gallery captions (div.gallerytext inside div.gallerybox)
were being skipped, which dropped captions like the Dhyan Chand hockey image.

Writes (English only, does not touch other languages):
  data/en_prose.txt  -- prose only (baseline)
  data/en_full.txt   -- prose + captions + tables (fullest)
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
TITLE = "India"

VOID = {"img", "br", "hr", "input", "meta", "link", "source", "area", "base",
        "col", "embed", "param", "track", "wbr"}
SKIP_TAGS = {"script", "style", "sup"}
# NOTE: 'gallerybox' removed so gallery captions are kept; infobox still excluded.
SKIP_CLASS = ["reference", "reflist", "navbox", "infobox", "metadata",
              "mw-editsection", "noprint", "sistersitebox", "mw-empty-elt",
              "toc", "hatnote", "ambox"]
CAP_CLASS = ["thumbcaption", "gallerytext"]   # + <figcaption> tag


def api(params):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def prose():
    d = api({"action": "query", "prop": "extracts", "explaintext": "1",
             "titles": TITLE, "format": "json", "redirects": "1"})
    return next(iter(d["query"]["pages"].values())).get("extract", "")


class CapsTables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []      # (tag, skip, cap, tab)
        self.caps, self.tables = [], []

    def _eff(self):
        return self.stack[-1][1:] if self.stack else (False, False, False)

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        cls = dict(attrs).get("class", "") or ""
        pskip, pcap, ptab = self._eff()
        skip = pskip or tag in SKIP_TAGS or any(s in cls for s in SKIP_CLASS)
        cap = pcap or tag == "figcaption" or any(c in cls for c in CAP_CLASS)
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


def caps_tables():
    d = api({"action": "parse", "page": TITLE, "prop": "text",
             "format": "json", "redirects": "1"})
    p = CapsTables()
    p.feed(d["parse"]["text"]["*"])
    caps = re.sub(r"\s+", " ", " ".join(p.caps))
    tables = re.sub(r"\s+", " ", " ".join(p.tables))
    return caps, tables


def main():
    body = prose()
    caps, tables = caps_tables()
    full = "\n\n".join(x for x in (body, caps, tables) if x)
    with open(os.path.join(DATA_DIR, "en_prose.txt"), "w", encoding="utf-8") as f:
        f.write(body)
    with open(os.path.join(DATA_DIR, "en_full.txt"), "w", encoding="utf-8") as f:
        f.write(full)
    has_hockey = "Dhyan Chand" in caps
    print(f"prose={len(body)}  caps={len(caps)}  tables={len(tables)}  full={len(full)}")
    print(f"gallery caption fix -> 'Dhyan Chand' caption captured: {has_hockey}")


if __name__ == "__main__":
    main()
