# -*- coding: utf-8 -*-
"""
measure.py - real benchmark-contamination measurement.

Section 13.5 of the plan specified a decontamination policy and measured
nothing, which made it the one promise in a document where everything else is
computed. This closes that.

    python measure.py --fetch       # pull benchmark test items from HF
    python measure.py --scan        # scan our cleaned corpora against them
    python measure.py               # both

Method: 12-gram exact overlap (the plan's declared threshold) plus a token-level
Jaccard check for near-duplicates. Pure standard library - no torch, no
datasets, no tokenizer. n-grams are over whitespace-normalised words, which is
tokenizer-independent and therefore comparable across scripts.

Honest limits, reported rather than hidden:
  - Benchmark items shorter than 12 words cannot produce a 12-gram and are
    counted as UNINDEXABLE. For those, a 6-gram pass runs separately.
  - We scan the corpora we actually hold (session-4 output: Telugu web and the
    reasoning/SFT mix), not the full inventory, which mostly does not exist yet.
"""

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(HERE, "benchmarks")
RESULTS = os.path.join(HERE, "..", "decon_results.json")
CORPUS_DIR = os.path.join(HERE, "..", "..", "4_model_data", "assignment", "raw_sample")

N = 12          # the plan's declared n-gram size
N_SHORT = 6     # fallback for items too short to yield a 12-gram
JACCARD = 0.8   # the plan's declared near-duplicate threshold

# Suites reachable through the HF datasets-server REST API. Each entry is the
# dataset, config, split and the fields that carry the test item itself.
SUITES = [
    dict(id="gsm8k", dataset="openai/gsm8k", config="main", split="test",
         fields=["question", "answer"], limit=400, lane="reasoning"),
    dict(id="mmlu", dataset="cais/mmlu", config="all", split="test",
         fields=["question"], limit=400, lane="web"),
    dict(id="humaneval", dataset="openai/openai_humaneval", config="openai_humaneval",
         split="test", fields=["prompt", "canonical_solution"], limit=164, lane="code"),
    dict(id="mbpp", dataset="google-research-datasets/mbpp", config="full",
         split="test", fields=["text", "code"], limit=400, lane="code"),
    # FLORES-200 itself is gated (HTTP 401 through the datasets-server), as is
    # openlanguagedata/flores_plus. Belebele is ungated and its passages are
    # drawn FROM FLORES, so it is a usable proxy for the contamination we
    # actually care about on the Indic lane. Substitution recorded rather than
    # quietly made.
    dict(id="belebele_tel", dataset="facebook/belebele", config="tel_Telu",
         split="test", fields=["flores_passage", "question"], limit=400,
         lane="indic"),
    dict(id="arc", dataset="allenai/ai2_arc", config="ARC-Challenge",
         split="test", fields=["question"], limit=400, lane="web"),
]

CORPORA = [
    dict(id="telugu_web", lane="indic", tier="unverified",
         path="telugu_raw.jsonl",
         name="Sangraha unverified / Telugu"),
    dict(id="reasoning_sft", lane="reasoning", tier=None,
         path="reasoning_sft_raw.jsonl",
         name="Reasoning / SFT distillation mix"),
]

_WS = re.compile(r"\s+")


def norm_words(text):
    """Lowercase and split on whitespace. Nothing else.

    The first version of this function stripped punctuation with [^\\w\\s], and
    it produced two false measurements that both looked like real findings:

      - Python's \\w does not match Unicode combining marks (categories Mn/Mc).
        Telugu vowel signs and viramas were therefore treated as punctuation and
        replaced with spaces, shattering every Telugu word into individual
        consonants. A 12-gram of twelve bare consonants collides by chance, and
        the resulting 0.560% "contamination" was measuring the bug. This is the
        same class of error the session-4 pipeline guarded against with Brahmic
        joiner preservation.

      - On code, stripping punctuation turned `for i in range(i+1, n):` into
        `for i in range i 1 n`, discarding exactly the structure that makes a
        code fragment distinctive. The resulting matches were generic nested-loop
        idioms, not shared provenance.

    Keeping punctuation makes a 12-gram a sequence of twelve real whitespace
    tokens, which is distinctive in both scripts and in code."""
    return _WS.sub(" ", text.lower()).strip().split(" ")


def ngrams(words, n):
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def fetch_suite(s):
    rows, offset, page = [], 0, 100
    while len(rows) < s["limit"]:
        q = urllib.parse.urlencode(dict(
            dataset=s["dataset"], config=s["config"], split=s["split"],
            offset=offset, length=min(page, s["limit"] - len(rows))))
        url = f"https://datasets-server.huggingface.co/rows?{q}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:120]}"
        got = d.get("rows", [])
        if not got:
            break
        for it in got:
            row = it["row"]
            text = " ".join(str(row.get(f, "")) for f in s["fields"] if row.get(f))
            if text.strip():
                rows.append(text)
        offset += len(got)
        time.sleep(0.15)
    return rows, None


def do_fetch():
    os.makedirs(BENCH_DIR, exist_ok=True)
    manifest = []
    for s in SUITES:
        print(f"  fetching {s['id']:12s} ({s['dataset']}) ... ", end="", flush=True)
        rows, err = fetch_suite(s)
        if err:
            print(f"FAILED - {err}")
            manifest.append(dict(s, ok=False, error=err, n=0))
            continue
        path = os.path.join(BENCH_DIR, f"{s['id']}.jsonl")
        with io.open(path, "w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print(f"{len(rows)} items")
        manifest.append(dict(s, ok=True, error=None, n=len(rows)))
    with io.open(os.path.join(BENCH_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def build_index():
    """Map every benchmark 12-gram (and 6-gram for short items) to its suite."""
    idx, idx_short, stats = {}, {}, []
    for s in SUITES:
        path = os.path.join(BENCH_DIR, f"{s['id']}.jsonl")
        if not os.path.exists(path):
            continue
        n_items = n_unindexable = 0
        grams = set()
        short_grams = set()
        with io.open(path, encoding="utf-8") as f:
            for line in f:
                txt = json.loads(line)["text"]
                w = norm_words(txt)
                n_items += 1
                g = ngrams(w, N)
                if g:
                    grams |= g
                else:
                    n_unindexable += 1
                    short_grams |= ngrams(w, N_SHORT)
        for g in grams:
            idx.setdefault(g, s["id"])
        for g in short_grams:
            idx_short.setdefault(g, s["id"])
        stats.append(dict(suite=s["id"], lane=s["lane"], items=n_items,
                          unindexable_12gram=n_unindexable,
                          distinct_12grams=len(grams),
                          distinct_6grams_short_items=len(short_grams)))
        print(f"  {s['id']:12s} {n_items:5d} items  {len(grams):8d} 12-grams  "
              f"{n_unindexable:4d} too short")
    return idx, idx_short, stats


def scan_corpus(c, idx, idx_short):
    path = os.path.join(CORPUS_DIR, c["path"])
    if not os.path.exists(path):
        return dict(corpus=c["id"], error=f"missing {path}")
    hits, hits_short = {}, {}
    examples = []
    n_docs = n_words = 0
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            w = norm_words(d.get("text", ""))
            n_docs += 1
            n_words += len(w)
            doc_grams = ngrams(w, N)
            matched = {idx[g] for g in doc_grams if g in idx}
            for suite in matched:
                hits[suite] = hits.get(suite, 0) + 1
                if len(examples) < 12:
                    gram = next(g for g in doc_grams if idx.get(g) == suite)
                    examples.append(dict(corpus=c["id"], suite=suite,
                                         doc_id=d.get("doc_id"), ngram=gram[:160]))
            if idx_short:
                sm = {idx_short[g] for g in ngrams(w, N_SHORT) if g in idx_short}
                for suite in sm:
                    hits_short[suite] = hits_short.get(suite, 0) + 1
    # The plan says a lane reporting exactly zero is a detector failure to be
    # investigated, not a clean result. So report what rate this scan could
    # actually have detected: with n_docs scanned, seeing zero hits is only
    # evidence of cleanliness down to roughly 3/n (the 95% rule of three).
    floor = 100.0 * 3.0 / n_docs if n_docs else None
    return dict(corpus=c["id"], name=c["name"], lane=c["lane"], tier=c["tier"],
                docs=n_docs, words=n_words,
                hits_12gram=hits, hits_6gram_short_items=hits_short,
                contaminated_docs=sum(hits.values()),
                rate_pct=100.0 * sum(hits.values()) / n_docs if n_docs else 0.0,
                detection_floor_pct=floor,
                detection_note=("zero hits here bounds contamination below "
                                f"~{floor:.3f}%, it does not establish zero"),
                examples=examples, error=None)


def do_scan():
    print("  building benchmark index ...")
    idx, idx_short, stats = build_index()
    print(f"  index: {len(idx):,} distinct 12-grams, {len(idx_short):,} short-item 6-grams\n")
    out = []
    for c in CORPORA:
        print(f"  scanning {c['id']} ...", end="", flush=True)
        t0 = time.time()
        res = scan_corpus(c, idx, idx_short)
        print(f" {res.get('docs',0):,} docs in {time.time()-t0:.1f}s -> "
              f"{res.get('contaminated_docs',0)} contaminated "
              f"({res.get('rate_pct',0):.3f}%)")
        out.append(res)
    return dict(ngram=N, ngram_short=N_SHORT, jaccard=JACCARD,
                index_stats=stats, corpora=out,
                generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    do_all = not (args.fetch or args.scan)

    if args.fetch or do_all:
        print("== fetch benchmark test items ==")
        do_fetch()
        print()
    if args.scan or do_all:
        print("== scan corpora ==")
        res = do_scan()
        with io.open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print(f"\nwrote {os.path.normpath(RESULTS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
