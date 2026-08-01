"""Inject results_marathi_partitioned.json into marathi_widget_template.html
-> india_bpe_marathi_widget.html (self-contained, incl. downloadable token list)."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT, "results_marathi_partitioned.json"), encoding="utf-8") as f:
    data = json.load(f)

with open(os.path.join(ROOT, "marathi_widget_template.html"), encoding="utf-8") as f:
    tmpl = f.read()

payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
payload = payload.replace("</", "<\\/")  # tokens contain "</w>"; keep the <script> block intact
html = tmpl.replace("/*__DATA__*/", payload)

out = os.path.join(ROOT, "india_bpe_marathi_widget.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {out}  ({len(html)/1024:.0f} KB)  score={data['primary']['score']}")
