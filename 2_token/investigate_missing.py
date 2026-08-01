"""Check how much of each Wikipedia page is MISSING from the prose extract we use.

The API prop=extracts&explaintext returns readable prose only; it drops the infobox,
image captions, tables and references. Here we fetch the fully rendered HTML
(action=parse) and compare visible-text size + count captions/infobox, per language.
"""
import json
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
LANGS = {"en": ("India", "English"), "hi": ("भारत", "Hindi"), "te": ("భారతదేశం", "Telugu"),
         "es": ("India", "Spanish"), "kn": ("ಭಾರತ", "Kannada"), "mr": ("भारत", "Marathi"),
         "ne": ("भारत", "Nepali")}


def api(lang, params):
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def extract_words(lang, title):
    d = api(lang, {"action": "query", "prop": "extracts", "explaintext": "1",
                   "titles": title, "format": "json", "redirects": "1"})
    p = next(iter(d["query"]["pages"].values()))
    return p.get("extract", "")


class Visible(HTMLParser):
    SKIP = {"script", "style", "sup"}  # sup = reference markers

    def __init__(self):
        super().__init__()
        self.buf = []
        self.caps = []
        self.skip = 0
        self.in_cap = 0
        self.in_infobox = 0
        self.infobox_chars = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag in self.SKIP:
            self.skip += 1
        if tag == "figcaption" or "thumbcaption" in cls:
            self.in_cap += 1
        if "infobox" in cls:
            self.in_infobox += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "figcaption" and self.in_cap:
            self.in_cap -= 1
        # infobox/thumbcaption end tracked loosely (div); reset on table/div close
        if tag in ("div", "table") and self.in_infobox:
            self.in_infobox = max(0, self.in_infobox - 1)

    def handle_data(self, data):
        if self.skip:
            return
        t = data.strip()
        if not t:
            return
        self.buf.append(data)
        if self.in_cap:
            self.caps.append(t)
        if self.in_infobox:
            self.infobox_chars += len(data)


def full_html_text(lang, title):
    d = api(lang, {"action": "parse", "page": title, "prop": "text",
                   "format": "json", "redirects": "1"})
    html = d["parse"]["text"]["*"]
    v = Visible()
    v.feed(html)
    text = re.sub(r"\s+", " ", " ".join(v.buf))
    return text, v.caps, v.infobox_chars


def wc(s):
    return len(s.split())


print(f'{"lang":8s} {"extract_w":>9s} {"full_w":>8s} {"missing%":>8s} {"captions":>8s} {"cap_w":>6s} {"infobox_c":>9s}')
for lang, (title, name) in LANGS.items():
    try:
        ex = extract_words(lang, title)
        full, caps, infobox_c = full_html_text(lang, title)
    except Exception as e:  # noqa: BLE001
        print(f"[{lang}] ERROR: {e}")
        continue
    ew, fw = wc(ex), wc(full)
    miss = 100 * (fw - ew) / fw if fw else 0
    cap_w = wc(" ".join(caps))
    print(f'{name:8s} {ew:9d} {fw:8d} {miss:7.1f}% {len(caps):8d} {cap_w:6d} {infobox_c:9d}')
