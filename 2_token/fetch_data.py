"""Fetch the "India" Wikipedia article as plain text in several languages.

Fixed languages : English (en), Hindi (hi), Telugu (te)
Candidate 4th   : Spanish (es), Kannada (kn), Marathi (mr), Nepali (ne)

Output: data/<lang>.txt  (one file per language, UTF-8)
"""
import json
import os
import sys
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252 by default
except Exception:  # noqa: BLE001
    pass

UA = "ERA-BPE-Assignment/1.0 (rahul.phil@gmail.com)"

# lang code -> (article title on that wiki, human-readable name)
LANGS = {
    "en": ("India",        "English"),
    "hi": ("भारत", "Hindi"),      # भारत
    "te": ("భారతదేశం", "Telugu"),  # భారతదేశం
    "es": ("India",        "Spanish"),
    "kn": ("ಭಾರತ", "Kannada"),    # ಭಾರತ
    "mr": ("भारत", "Marathi"),    # भारत
    "ne": ("भारत", "Nepali"),     # भारत
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch_extract(lang: str, title: str) -> str:
    """Return the plain-text extract of `title` from the `lang` wikipedia."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "format": "json",
        "redirects": "1",
    }
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    summary = []
    for lang, (title, name) in LANGS.items():
        try:
            text = fetch_extract(lang, title)
        except Exception as e:  # noqa: BLE001
            print(f"[{lang}] ERROR: {e}")
            continue
        path = os.path.join(DATA_DIR, f"{lang}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        summary.append((lang, name, len(text)))
        print(f"[{lang}] {name:8s} title={title!r:20s} {len(text):>7d} chars -> {path}")
    print("\nDone. Fetched", len(summary), "languages.")


if __name__ == "__main__":
    main()
