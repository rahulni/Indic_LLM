"""Fetch the India Wikipedia plain-text extracts for the partitioned-BPE project.

Saves formatversion=2 JSON so that `data["query"]["pages"][0]["extract"]` matches
what multilingual_bpe_report.py and bpe_x1_english.py expect.

Output: data/india_<lang>_extract.json  for en, hi, te, mr
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
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

# lang code -> (article title on that wiki, human-readable name)
LANGS = {
    "en": ("India", "English"),
    "hi": ("भारत", "Hindi"),
    "te": ("భారతదేశం", "Telugu"),
    "mr": ("भारत", "Marathi"),
}


def fetch(lang: str, title: str) -> dict:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for lang, (title, name) in LANGS.items():
        data = fetch(lang, title)
        page = data["query"]["pages"][0]
        path = os.path.join(DATA_DIR, f"india_{lang}_extract.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[{lang}] {name:8s} title={page.get('title')!r}  "
              f"{len(page.get('extract', '')):>7d} chars -> {path}")


if __name__ == "__main__":
    main()
