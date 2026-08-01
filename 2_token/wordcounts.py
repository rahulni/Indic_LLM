import json, sys
sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open("results.json", encoding="utf-8"))
res = d["results"]
hdr = f'{"lang":8s} {"words":>7s} {"unique":>7s} {"chars":>7s} {"wtoks":>7s} {"X":>6s}'
print(hdr)
for l in res["es"]["languages"]:
    if l["code"] in ("en", "hi", "te"):
        print(f'{l["name"]:8s} {l["words"]:7d} {l["unique_words"]:7d} {l["chars"]:7d} {l["word_tokens"]:7d} {l["fertility"]:6.3f}')
print("-- candidate 4th --")
for c in ["es", "kn", "mr", "ne"]:
    l = [x for x in res[c]["languages"] if x["code"] == c][0]
    print(f'{l["name"]:8s} {l["words"]:7d} {l["unique_words"]:7d} {l["chars"]:7d} {l["word_tokens"]:7d} {l["fertility"]:6.3f}')
