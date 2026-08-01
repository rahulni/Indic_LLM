"""Compute the English-only fertility story and emit english_results.json, then
inject it into english_widget_template.html -> english_widget.html.

Everything here is ENGLISH ONLY.
"""
import json
import os
import sys
from collections import Counter

from bpe import BPETokenizer, pretokenize

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
VOCAB = 10_000


def load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return f.read()


def train(text, mpf=2):
    pts = pretokenize(text)
    freqs = Counter(pt for pt, _ in pts)
    tok = BPETokenizer(unit_mode="char")
    info = tok.train(dict(freqs), VOCAB, min_pair_freq=mpf)
    return tok, pts, info


def fertility(tok, pts):
    cache, words, wtoks = {}, 0, 0
    for pt, isw in pts:
        n = cache.get(pt)
        if n is None:
            n = len(tok.encode_pretoken(pt))
            cache[pt] = n
        if isw:
            words += 1
            wtoks += n
    return round(wtoks / words, 4) if words else 0.0, words


def strat(name, text, kind, note, mpf=2):
    tok, pts, info = train(text, mpf)
    x, w = fertility(tok, pts)
    return {"name": name, "X": x, "words": w, "vocab": info["vocab_size"],
            "kind": kind, "note": note}, tok, pts


def show_tokens(tok, pt):
    return [t.replace("</w>", "⏎") for t in tok.encode_pretoken(pt)]


def main():
    prose = load("en_prose.txt")
    full = load("en_full.txt")

    strategies = []
    s, _, _ = strat("Prose only", prose, "honest", "Readable article text only."); strategies.append(s)
    s, tok_full, pts_full = strat("Prose + captions + tables", full, "honest",
                                  "Your idea — includes the gallery captions (Dhyan Chand) now fixed. Helps a little."); strategies.append(s)
    # shared-tokenizer reference from the multilingual run, if available
    shared_x = None
    rp = os.path.join(ROOT, "results.json")
    if os.path.exists(rp):
        d = json.load(open(rp, encoding="utf-8"))
        try:
            shared_x = [l for l in d["results"]["es"]["languages"] if l["code"] == "en"][0]["fertility"]
        except Exception:  # noqa: BLE001
            shared_x = None
    if shared_x is not None:
        strategies.append({"name": "English inside the shared 10k (submitted)", "X": shared_x,
                           "words": None, "vocab": None, "kind": "honest",
                           "note": "Sharing the vocab with Spanish actually helps English a bit."})
    s, tok_mpf1, pts_mpf1 = strat("min_pair_freq = 1", full, "overfit",
                                  "Merges even one-off subwords → memorises this article’s proper nouns. The ONLY way below 1.2."
                                  , mpf=1); strategies.append(s)

    # proper-noun "wall": how they tokenize honestly vs memorised
    examples = []
    for w in ["vijayanagara", "balochistan", "brahmaputra", "ramanujan",
              "chandigarh", "aryabhata", "dhyan", "aṣṭādhyāyī".lower()]:
        examples.append({
            "word": w,
            "honest": show_tokens(tok_full, w),
            "memorised": show_tokens(tok_mpf1, w),
        })

    honest_floor = min(s["X"] for s in strategies if s["kind"] == "honest")
    best_below = next((s["X"] for s in strategies if s["kind"] == "overfit"), None)

    out = {
        "meta": {
            "title": "English BPE fertility — how low can it honestly go?",
            "source": "Wikipedia “India” (prose + captions + tables)",
            "target": 1.2,
            "honest_floor": honest_floor,
            "memorised": best_below,
            "vocab_budget": VOCAB,
            "caption_fix": "Gallery captions (e.g. the Dhyan Chand hockey caption) are now extracted.",
        },
        "strategies": strategies,
        "examples": examples,
        "tokens": tok_full.vocab,          # honest tokenizer vocabulary
        "n_base": tok_full.base_alphabet.__len__(),
    }
    json.dump(out, open(os.path.join(ROOT, "english_results.json"), "w", encoding="utf-8"), ensure_ascii=False)

    # inject into the template
    tmpl = open(os.path.join(ROOT, "english_widget_template.html"), encoding="utf-8").read()
    html = tmpl.replace("/*__DATA__*/", json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    open(os.path.join(ROOT, "english_widget.html"), "w", encoding="utf-8").write(html)
    print(f"honest floor X={honest_floor}  memorised X={best_below}  tokens={len(out['tokens'])}")
    print("wrote english_results.json + english_widget.html")


if __name__ == "__main__":
    main()
