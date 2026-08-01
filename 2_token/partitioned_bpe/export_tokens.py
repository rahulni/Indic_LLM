"""Export the FULL token list for each language's tokenizer, make the tokens
downloadable, and build an interactive widget.

The base engine (multilingual_bpe_report.py) discards the learned tokens, so this
script reuses its EXACT algorithm (pair_counts / merge_pair, same tie-breaking) to
re-derive every token up to the vocab size the optimizer selected for each language
-- read from results/multilingual_bpe_report.json.

Outputs:
  results/tokens/tokens_<code>.txt   one token per line (base alphabet + merges)
  results/tokens/all_tokens.txt      all languages, sectioned
  results/tokens/all_tokens.json     {code: {language, count, tokens:[...]}}
  widget.html                        ratios + score + constraint checks + token
                                     browser + Download buttons (see all tokens)
"""
import json
import os
import sys
from collections import Counter

import multilingual_bpe_report as eng

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
TOKENS_DIR = os.path.join(RESULTS, "tokens")


def record_tokens(code: str, target_vocab: int) -> list[str]:
    """Re-run the identical BPE and record every token (base + merges in order)."""
    path = os.path.join(DATA_DIR, f"india_{code}_extract.json")
    _title, text = eng.load_wikipedia_extract(__import__("pathlib").Path(path))
    words = eng.extract_words(text, code)
    vocab = Counter({tuple(w): f for w, f in Counter(words).items()})
    base = sorted({s for syms in vocab for s in syms})
    tokens = list(base)
    while len(tokens) < target_vocab:
        pairs = eng.pair_counts(vocab)
        if not pairs:
            break
        best_pair, _freq = max(pairs.items(), key=lambda it: (it[1], it[0]))
        vocab = eng.merge_pair(vocab, best_pair)
        tokens.append(best_pair[0] + best_pair[1])
    return tokens


def main():
    report_path = os.path.join(RESULTS, "multilingual_bpe_report.json")
    if not os.path.exists(report_path):
        raise SystemExit("Run multilingual_bpe_report.py first (results/…json missing).")
    report = json.load(open(report_path, encoding="utf-8"))

    os.makedirs(TOKENS_DIR, exist_ok=True)
    langs = []
    all_tokens = {}
    combined_lines = []

    for lang in report["languages"]:
        code = lang["code"]
        target = lang["selected_bpe_vocab"]
        tokens = record_tokens(code, target)
        assert len(tokens) == target, f"{code}: recorded {len(tokens)} != selected {target}"
        base_n = lang["base_vocab_size"]

        with open(os.path.join(TOKENS_DIR, f"tokens_{code}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(tokens) + "\n")
        all_tokens[code] = {"language": lang["language"], "count": len(tokens), "tokens": tokens}
        combined_lines.append(f"# {lang['language']} ({code}) - {len(tokens)} tokens "
                              f"({base_n} base + {len(tokens) - base_n} merges)")
        combined_lines.extend(tokens)
        combined_lines.append("")

        langs.append({**lang, "base_tokens": base_n,
                      "merge_tokens": len(tokens) - base_n, "tokens": tokens})
        print(f"[{code}] {lang['language']:8s} tokens={len(tokens)} "
              f"({base_n} base + {len(tokens) - base_n} merges)  X={lang['ratio']:.6f}")

    with open(os.path.join(TOKENS_DIR, "all_tokens.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(combined_lines))
    with open(os.path.join(TOKENS_DIR, "all_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(all_tokens, f, ensure_ascii=False, indent=2)

    ratios = [l["ratio"] for l in langs]
    english = next(l["ratio"] for l in langs if l["code"] == "en")
    data = {
        "total_vocab": report["total_vocab"],
        "spread": report["spread"],
        "score": report["assignment_score"],
        "english": english,
        "pass_total": report["total_vocab"] < 10000,
        "pass_english": english < 1.2,
        "x_max": max(ratios), "x_min": min(ratios),
        "languages": [{k: v for k, v in l.items() if k != "tokens"} for l in langs],
        "tokens": {l["code"]: l["tokens"] for l in langs},
    }

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    with open(os.path.join(ROOT, "widget.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\ntotal_vocab={data['total_vocab']} (<10000: {data['pass_total']})  "
          f"English X={english:.6f} (<1.2: {data['pass_english']})  score={data['score']:.2f}")
    print("wrote results/tokens/*  and  widget.html")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>India BPE - Tokens & Score (partitioned)</title>
<style>
  :root{--ink:#17202a;--muted:#5f6f7e;--faint:#8b98a5;--line:#dbe3ea;--panel:#f6f9fb;
        --accent:#0f766e;--accent-soft:#0f766e14;--good:#15803d;--bad:#b42318;
        --mono:ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace;
        --sans:system-ui,"Segoe UI",Roboto,Arial,sans-serif;}
  *{box-sizing:border-box}
  body{margin:0;background:#fff;color:var(--ink);font-family:var(--sans);line-height:1.5}
  main{width:min(1080px,calc(100% - 32px));margin:28px auto 72px}
  h1{font-size:26px;margin:0 0 4px}
  .sub{color:var(--muted);margin:0 0 22px;font-size:14px}
  .metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:14px}
  @media(max-width:720px){.metrics{grid-template-columns:repeat(2,1fr)}}
  .metric{border:1px solid var(--line);border-radius:10px;padding:13px 14px;background:var(--panel)}
  .metric .label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}
  .metric .value{display:block;font-size:22px;font-weight:700;font-family:var(--mono);overflow-wrap:anywhere}
  .metric.score .value{color:var(--accent)}
  .checks{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 22px}
  .pill{border-radius:999px;padding:6px 12px;font-size:13px;color:#fff;font-family:var(--mono)}
  .pill.ok{background:var(--good)} .pill.no{background:var(--bad)}
  .tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin-bottom:26px}
  table{width:100%;border-collapse:collapse;font-size:13.5px}
  th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
  th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}
  thead th{background:#eef4f7;color:#33424e;font-size:11px;text-transform:uppercase;letter-spacing:.03em}
  tbody tr:last-child td{border-bottom:0}
  td .x{color:var(--muted);margin-left:5px;font-size:11px}
  h2{font-size:16px;margin:0 0 4px}
  .h2sub{color:var(--muted);font-size:13px;margin:0 0 12px}
  .toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px}
  .tabs{display:flex;gap:6px;flex-wrap:wrap}
  .tab{font-family:var(--mono);font-size:13px;padding:7px 12px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;color:var(--muted)}
  .tab[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
  .search{flex:1;min-width:180px;font-family:var(--mono);font-size:13px;padding:9px 11px;border:1px solid var(--line);border-radius:8px}
  .search:focus{outline:2px solid var(--accent);outline-offset:1px}
  .dl{font-family:var(--mono);font-size:12.5px;padding:8px 12px;border-radius:8px;border:1px solid var(--accent);background:var(--accent-soft);color:var(--accent);cursor:pointer}
  .dl:hover{background:var(--accent);color:#fff}
  .dlrow{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
  .tokgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:6px}
  .tok{font-family:var(--mono);font-size:12.5px;border:1px solid var(--line);border-radius:7px;padding:5px 8px;background:#fff;display:flex;justify-content:space-between;gap:6px;overflow:hidden}
  .tok.base{border-left:3px solid var(--faint)} .tok.merge{border-left:3px solid var(--accent)}
  .tok .t{white-space:pre;overflow:hidden;text-overflow:ellipsis}
  .tok .i{color:var(--faint);font-size:10px}
  .foot{margin-top:16px;font-size:12.5px;color:var(--muted)}
  .more{font-family:var(--mono);font-size:13px;padding:8px 15px;border-radius:8px;border:1px solid var(--accent);background:#fff;color:var(--accent);cursor:pointer;margin-top:12px}
  .more:disabled{opacity:.4;cursor:default}
  code{font-family:var(--mono);background:var(--panel);padding:1px 5px;border-radius:5px}
</style></head>
<body><main>
  <h1>India BPE - tokens, ratios &amp; self-score</h1>
  <p class="sub">Language-partitioned word-internal BPE. Four per-language vocabularies under one
     shared cap. <b>Score = 1000 / (X<sub>max</sub> - X<sub>min</sub>)</b>. Fetched live from Wikipedia.</p>

  <div class="metrics">
    <div class="metric"><span class="label">Total vocab (all langs)</span><span class="value" id="mVocab"></span></div>
    <div class="metric"><span class="label">Ratio spread</span><span class="value" id="mSpread"></span></div>
    <div class="metric score"><span class="label">Self-score</span><span class="value" id="mScore"></span></div>
    <div class="metric"><span class="label">English X1</span><span class="value" id="mEng"></span></div>
  </div>
  <div class="checks" id="checks"></div>

  <div class="tablewrap"><table>
    <thead><tr><th>Sort</th><th>Language</th><th>Page</th><th>Words</th><th>Unique</th>
      <th>Vocab (base+merges)</th><th>Tokens</th><th>X = tokens/words</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table></div>

  <h2>Tokenizer vocabulary</h2>
  <p class="h2sub">Every token in each language's tokenizer. Search, browse, and download.</p>
  <div class="dlrow" id="dlrow"></div>
  <div class="toolbar">
    <div class="tabs" id="tabs"></div>
    <input class="search" id="search" placeholder="search tokens..." autocomplete="off">
  </div>
  <div class="tokgrid" id="tokgrid"></div>
  <button class="more" id="more">Show more</button>
  <div class="foot" id="foot"></div>
</main>

<script>
const DATA = /*__DATA__*/;
const $ = s => document.querySelector(s);
const fmt = n => n.toLocaleString("en-US");
const esc = s => s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const XID = {en:"X1",hi:"X2",te:"X3",mr:"X4"};

$("#mVocab").textContent = fmt(DATA.total_vocab);
$("#mSpread").textContent = DATA.spread.toFixed(6);
$("#mScore").textContent = (DATA.score>1e9?"inf":DATA.score.toFixed(2));
$("#mEng").textContent = DATA.english.toFixed(6);
$("#checks").innerHTML =
  `<span class="pill ${DATA.pass_total?'ok':'no'}">Total vocab &lt; 10000: ${DATA.pass_total}</span>`+
  `<span class="pill ${DATA.pass_english?'ok':'no'}">English X1 &lt; 1.2: ${DATA.pass_english}</span>`;

const sorted = [...DATA.languages].sort((a,b)=>b.ratio-a.ratio);
$("#tbody").innerHTML = sorted.map((l,i)=>
  `<tr><td>${i+1} <span class="x">${l.x_label}</span></td>`+
  `<td>${esc(l.language)}</td><td>${esc(l.page_title)}</td>`+
  `<td>${fmt(l.word_count)}</td><td>${fmt(l.unique_words)}</td>`+
  `<td>${fmt(l.selected_bpe_vocab)} <span class="x">${l.base_tokens}+${l.merge_tokens}</span></td>`+
  `<td>${fmt(l.encoded_token_occurrences)}</td>`+
  `<td><b>${l.ratio.toFixed(6)}</b></td></tr>`).join("");

// ---- downloads
function download(name, text, mime){
  const blob = new Blob([text], {type: mime || "text/plain;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1000);
}
function allTokensText(){
  return DATA.languages.map(l=>{
    const t = DATA.tokens[l.code];
    return `# ${l.language} (${l.code}) - ${t.length} tokens\n` + t.join("\n");
  }).join("\n\n");
}
$("#dlrow").innerHTML =
  `<button class="dl" id="dlAll">Download ALL tokens (.txt)</button>`+
  `<button class="dl" id="dlJson">Download all_tokens.json</button>`+
  DATA.languages.map(l=>`<button class="dl" data-c="${l.code}">${l.code}.txt</button>`).join("");
$("#dlAll").onclick = ()=>download("india_all_tokens.txt", allTokensText());
$("#dlJson").onclick = ()=>download("india_all_tokens.json",
  JSON.stringify(DATA.tokens, null, 2), "application/json");
$("#dlrow").querySelectorAll("[data-c]").forEach(b=>{
  b.onclick = ()=>download(`india_tokens_${b.dataset.c}.txt`, DATA.tokens[b.dataset.c].join("\n"));
});

// ---- token browser
let cur = "en", shown = 0, PAGE = 200, filt = [];
$("#tabs").innerHTML = DATA.languages.map(l=>
  `<button class="tab" data-c="${l.code}" aria-selected="${l.code===cur}">${l.language} `+
  `<span class="x">${fmt(l.selected_bpe_vocab)}</span></button>`).join("");
function apply(){
  const q = $("#search").value.trim().toLowerCase();
  const toks = DATA.tokens[cur];
  const base = DATA.languages.find(l=>l.code===cur).base_tokens;
  filt = [];
  for(let i=0;i<toks.length;i++){
    if(q && !toks[i].toLowerCase().includes(q)) continue;
    filt.push([i, toks[i], i<base]);
  }
  shown = Math.min(PAGE, filt.length); paint();
}
function paint(){
  const base = DATA.languages.find(l=>l.code===cur).base_tokens;
  $("#tokgrid").innerHTML = filt.slice(0,shown).map(([i,t,isBase])=>
    `<div class="tok ${isBase?'base':'merge'}" title="${esc(t)}"><span class="t">${esc(t)||' '}</span>`+
    `<span class="i">${isBase?'-':'#'+(i-base+1)}</span></div>`).join("");
  $("#more").disabled = shown>=filt.length;
  $("#foot").innerHTML = `Showing <b>${fmt(Math.min(shown,filt.length))}</b> of <b>${fmt(filt.length)}</b> `+
    `matching tokens - ${DATA.languages.find(l=>l.code===cur).language} tokenizer `+
    `(<code>${base}</code> base chars + merges, no <code>&lt;/w&gt;</code> marker).`;
}
$("#tabs").querySelectorAll(".tab").forEach(b=>b.onclick=()=>{
  cur=b.dataset.c; $("#tabs").querySelectorAll(".tab").forEach(x=>x.setAttribute("aria-selected",x===b));
  $("#search").value=""; apply();
});
$("#search").addEventListener("input", apply);
$("#more").onclick = ()=>{ shown=Math.min(shown+PAGE, filt.length); paint(); };
apply();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
