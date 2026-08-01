"""Machine-translation augmentation (deep-translator / Google, no API key).

Two modes:
  * default  : translate ENGLISH source that is NOT the India article (data/en_extra)
               into thin languages -> data_aug/<lang>.txt  (adds domain-adjacent vocab,
               limits leakage).
  * indomain : translate the ENGLISH India article (data/en.txt) into a language ->
               data_aug/<lang>_indomain.txt. This reinforces the eval article's exact
               vocabulary and LOWERS that language's fertility -- but it is in-domain
               MT leakage (translationese + same topic we score on). Disclose it.

Usage:  python augment_translate.py            # default mode
        python augment_translate.py indomain   # in-domain (te by default)

CAVEATS: MT output is translationese; augmenting only some languages is corpus
engineering; the in-domain mode is leakage. All surfaced in the widget.
"""
import os
import re
import sys
import time

from deep_translator import GoogleTranslator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
AUG_DIR = os.path.join(ROOT, "data_aug")

TARGETS = {"kn": 60_000, "ne": 40_000, "te": 40_000}   # default (non-India) mode
INDOMAIN_TARGETS = {"te": 65_000}                        # in-domain (India article) mode
CHUNK = 4500


def chunk_text(text, size):
    parts, buf = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(buf) + len(sent) + 1 > size:
            if buf:
                parts.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        parts.append(buf)
    return parts


def translate_text(src, lang):
    tr = GoogleTranslator(source="en", target=lang)
    out, chunks = [], chunk_text(src, CHUNK)
    for i, ch in enumerate(chunks):
        for attempt in range(4):
            try:
                out.append(tr.translate(ch))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    print(f"  [{lang}] chunk {i} failed: {e}")
                else:
                    time.sleep(2 * (attempt + 1))
        if i % 5 == 0:
            print(f"  [{lang}] {i+1}/{len(chunks)} chunks")
        time.sleep(0.2)
    return "\n".join(out)


def read_capped(path, n):
    with open(path, encoding="utf-8") as f:
        return f.read()[:n]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    os.makedirs(AUG_DIR, exist_ok=True)
    if mode == "indomain":
        for lang, n in INDOMAIN_TARGETS.items():
            print(f"[{lang}] in-domain: translating ~{n} chars of the English India article -> {lang}")
            text = translate_text(read_capped(os.path.join(DATA_DIR, "en.txt"), n), lang)
            with open(os.path.join(AUG_DIR, f"{lang}_indomain.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[{lang}] wrote {len(text)} chars -> data_aug/{lang}_indomain.txt")
    else:
        src_path = os.path.join(DATA_DIR, "en_extra", "links.txt")
        if not os.path.exists(src_path):
            src_path = os.path.join(DATA_DIR, "en.txt")
        for lang, n in TARGETS.items():
            print(f"[{lang}] default: translating ~{n} chars of non-India English -> {lang}")
            text = translate_text(read_capped(src_path, n), lang)
            with open(os.path.join(AUG_DIR, f"{lang}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[{lang}] wrote {len(text)} chars -> data_aug/{lang}.txt")


if __name__ == "__main__":
    main()
