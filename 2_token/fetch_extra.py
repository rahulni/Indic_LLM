"""Phase B (real data): enlarge each language's corpus with genuine in-language text.

For every language we pull the plain-text extracts of the articles LINKED FROM the
"India" page on that wiki (namespace 0). This is real, native text on related
topics -- the honest way to lower fertility for data-scarce languages (Telugu,
Kannada, Nepali) instead of gaming the metric.

Output: data/<lang>_extra/links.txt
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

UA = "ERA-BPE-Assignment/1.0 (rahul.phil@gmail.com)"
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

# (title on that wiki, human name)  -- same India article as fetch_data.py
LANGS = {
    "en": ("India", "English"), "hi": ("भारत", "Hindi"), "te": ("భారతదేశం", "Telugu"),
    "es": ("India", "Spanish"), "kn": ("ಭಾರತ", "Kannada"), "mr": ("भारत", "Marathi"),
    "ne": ("भारत", "Nepali"),
}
GPLLIMIT = 60         # linked pages fetched per request
MAX_CHARS = 350_000   # stop once we have this much real text per language (plenty)


def api(lang, params, retries=4):
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 -- transient network / rate limits
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def fetch_links_text(lang, title):
    """Concatenated plain-text extracts of pages linked from `title`, capped at
    MAX_CHARS so we don't harvest millions of characters."""
    params = {
        "action": "query", "generator": "links", "titles": title,
        "gpllimit": str(GPLLIMIT), "gplnamespace": "0",
        "prop": "extracts", "explaintext": "1", "exlimit": "5",
        "format": "json", "redirects": "1",
    }
    texts, total = [], 0
    cont = {}
    while total < MAX_CHARS:
        data = api(lang, {**params, **cont})
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            ex = p.get("extract")
            if ex and len(ex) > 200:  # skip stubs
                texts.append(ex)
                total += len(ex)
        if "continue" in data and total < MAX_CHARS:
            cont = data["continue"]
            time.sleep(0.3)  # be polite
        else:
            break
    return "\n\n".join(texts)


def main():
    for lang, (title, name) in LANGS.items():
        try:
            text = fetch_links_text(lang, title)
        except Exception as e:  # noqa: BLE001
            print(f"[{lang}] ERROR: {e}")
            continue
        d = os.path.join(DATA_DIR, f"{lang}_extra")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "links.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[{lang}] {name:8s} +{len(text):>8d} chars real linked-article text")


if __name__ == "__main__":
    main()
