"""Main-project scoring on the partitioned_bpe data (Marathi as the 4th language).

Loads the India-article extracts DIRECTLY from partitioned_bpe/data/india_*_extract.json
(no re-download), trains ONE joint 10k-vocab BPE over English+Hindi+Telugu+Marathi with
the main project's profiles, and reports X1..X4, spread and score = 1000/(X_max-X_min).

Output: results_marathi_partitioned.json (consumed by build_marathi_widget.py).
"""
import json
import os
import sys
from collections import Counter

from bpe import BPETokenizer, pretokenize
from morph import split_suffixes

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
PART_DATA = os.path.join(ROOT, "partitioned_bpe", "data")
VOCAB_BUDGET = 10_000

LANGS = ["en", "hi", "te", "mr"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}
LOCAL = {"en": "English", "hi": "हिन्दी", "te": "తెలుగు", "mr": "मराठी"}
DRAVIDIAN = {"te"}

PROFILES = [
    {"key": "baseline", "label": "Baseline · codepoint BPE", "kind": "honest",
     "note": "Classic BPE on Unicode codepoints, no normalization tricks.",
     "unit_mode": "char", "strip_joiners": False, "morph": False, "min_pair_freq": 2},
    {"key": "akshara", "label": "Akshara units + joiner strip", "kind": "honest",
     "note": "Grapheme-cluster base units keep Indic syllables intact; ZWNJ/ZWJ stripped.",
     "unit_mode": "akshara", "strip_joiners": True, "morph": False, "min_pair_freq": 2},
    {"key": "te_morph", "label": "Akshara + Telugu morphology — SUBMITTED", "kind": "linguistic",
     "note": "Additionally splits Telugu case/plural/postposition suffixes so recurring "
             "morphemes share tokens.",
     "unit_mode": "akshara", "strip_joiners": True, "morph": True, "min_pair_freq": 2},
]
PRIMARY = "te_morph"


def eval_text(lang):
    path = os.path.join(PART_DATA, f"india_{lang}_extract.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    page = payload["query"]["pages"][0]
    return page["title"], page["extract"]


def _pretoks(text, lang, profile):
    pts = pretokenize(text, strip_joiners=profile["strip_joiners"])
    if profile["morph"] and lang in DRAVIDIAN:
        out = []
        for pt, is_word in pts:
            if is_word:
                out.extend((piece, True) for piece in split_suffixes(pt))
            else:
                out.append((pt, is_word))
        pts = out
    return pts


def corpus_stats(lang, tok, profile, cache):
    title, text = eval_text(lang)
    words = word_tokens = total_pretokens = total_tokens = chars = 0
    unique_words = set()
    for pt, is_word in _pretoks(text, lang, profile):
        n = cache.get(pt)
        if n is None:
            n = len(tok.encode_pretoken(pt))
            cache[pt] = n
        total_pretokens += 1
        total_tokens += n
        chars += len(pt)
        if is_word:
            words += 1
            unique_words.add(pt)
            word_tokens += n
    fert = word_tokens / words if words else 0.0
    return {"code": lang, "name": NAMES[lang], "local_name": LOCAL[lang], "page_title": title,
            "chars": chars, "words": words, "unique_words": len(unique_words),
            "word_tokens": word_tokens, "total_pretokens": total_pretokens,
            "total_tokens": total_tokens, "fertility": round(fert, 4)}


def run(profile, keep_tokens=False):
    combined = Counter()
    for lang in LANGS:
        _, text = eval_text(lang)
        combined.update(pt for pt, _ in _pretoks(text, lang, profile))
    tok = BPETokenizer(unit_mode=profile["unit_mode"])
    info = tok.train(dict(combined), VOCAB_BUDGET, min_pair_freq=profile["min_pair_freq"])
    cache = {}
    lang_stats = [corpus_stats(l, tok, profile, cache) for l in LANGS]
    ferts = sorted(((s["name"], s["fertility"]) for s in lang_stats), key=lambda t: t[1], reverse=True)
    x_max, x_min = ferts[0][1], ferts[-1][1]
    spread = round(x_max - x_min, 4)
    score = round(1000.0 / spread, 2) if spread > 0 else None
    out = {"languages": lang_stats,
           "sorted_ratios": [{"name": n, "X": x} for n, x in ferts],
           "x_max": x_max, "x_min": x_min, "spread": spread, "score": score,
           "vocab_size": info["vocab_size"], "n_base": info["n_base"], "n_merges": info["n_merges"]}
    if keep_tokens:
        out["tokens"] = tok.vocab
    return out


def main():
    per_profile = {}
    for prof in PROFILES:
        r = run(prof, keep_tokens=(prof["key"] == PRIMARY))
        per_profile[prof["key"]] = r
        print(f"  {prof['key']:10s} spread={r['spread']:.4f}  score={r['score']}  "
              f"X: " + "  ".join(f"{s['name']}={s['fertility']:.4f}" for s in r["languages"]))

    primary = per_profile[PRIMARY]
    base = per_profile["baseline"]
    bx = {l["code"]: l["fertility"] for l in base["languages"]}
    for l in primary["languages"]:
        l["baseline_fertility"] = bx.get(l["code"])
    out = {
        "meta": {
            "source": "partitioned_bpe/data/india_*_extract.json (Wikipedia 'India' article intros)",
            "vocab_budget": VOCAB_BUDGET,
            "languages": [NAMES[l] for l in LANGS],
            "metric": "fertility X = word_tokens / words (avg tokens per word; lower = better)",
            "score_formula": "1000 / (X_max - X_min) over the 4 languages",
            "primary_profile": PRIMARY,
        },
        "profiles": [
            {"key": p["key"], "label": p["label"], "kind": p["kind"], "note": p["note"],
             "langX": {l["code"]: l["fertility"] for l in per_profile[p["key"]]["languages"]},
             "spread": per_profile[p["key"]]["spread"], "score": per_profile[p["key"]]["score"]}
            for p in PROFILES
        ],
        "primary": primary,
    }
    dest = os.path.join(ROOT, "results_marathi_partitioned.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\nPrimary profile: {PRIMARY}  spread={primary['spread']}  "
          f"score={primary['score']} -> {dest}")


if __name__ == "__main__":
    main()
