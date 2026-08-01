"""Inject results.json into widget_template.html -> india_bpe_widget.html"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT, "results.json"), encoding="utf-8") as f:
    data = json.load(f)

# best candidate code for the widget's default tab
best = max(data["results"].values(), key=lambda r: (r["score"] or -1))
data["meta"]["best_candidate_code"] = best["candidate"]

with open(os.path.join(ROOT, "widget_template.html"), encoding="utf-8") as f:
    tmpl = f.read()

payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
html = tmpl.replace("/*__DATA__*/", payload)

out = os.path.join(ROOT, "india_bpe_widget.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {out}  ({len(html)/1024:.0f} KB)  best={best['candidate_name']} score={best['score']}")
