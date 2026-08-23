#!/usr/bin/env python3
"""Copy the built site into 8_attention/, where GitHub Pages serves it from.

Pages on this repository is configured to serve the branch contents directly rather
than a workflow artifact, which is why the earlier submission appears at
    https://rahulni.github.io/Indic_LLM/7_embed_research/submission_artifacts/index.html
A file committed at 8_attention/index.html therefore appears at
    https://rahulni.github.io/Indic_LLM/8_attention/

Vite builds into app/dist. That directory stays gitignored because it is a build
artifact of a build artifact; this script promotes the finished output to the published
location so exactly one copy is committed.

`assets/` is cleared before copying rather than merged. Vite fingerprints filenames, so
merging would silently accumulate every previous build's orphaned bundles - the
repository would grow forever and nobody would notice.

It also writes build-info.json recording a fingerprint of the content the site was built
from. See the note on SOURCES below for why that exists.

Usage:
    cd app && npm run build
    cd .. && python tools/publish_site.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "app" / "dist"
INDEX = ROOT / "index.html"
ASSETS = ROOT / "assets"

# Everything Pages needs. Anything else Vite emits is listed explicitly rather than
# copied wholesale, so a stray file cannot end up published by accident.
COPY_FILES = ["index.html"]
COPY_DIRS = ["assets"]

# The content files the site is built from.
#
# check_published.py compares a fingerprint of these against the one recorded here at
# publish time, which answers the question that actually matters - was the live site
# built from the data currently in the repository - without demanding byte-identical
# builds across platforms.
#
# That distinction was learned the hard way. The previous check diffed the committed
# bundle against a fresh build, and failed on every CI run: this repository is authored
# on Windows where the working tree is CRLF, git stores LF, and the Linux runner
# therefore compiles different source bytes and produces a different content hash. Line
# endings, npm resolution and per-platform esbuild binaries all move those bytes, and
# none of them mean the published site is stale.
SOURCES = [
    "app/src/data/mechanisms.json",
    "app/src/data/eras.json",
    "app/src/data/glossary.json",
    "app/src/data/checks.json",
    "app/src/data/formulas.ts",
    "app/src/data/resources.ts",
]


def fingerprint() -> str:
    """SHA-256 over the content files, with line endings normalised.

    Normalising CRLF to LF is what makes this comparable between a Windows working tree
    and a Linux runner, which is the entire point.
    """
    digest = hashlib.sha256()
    for rel in SOURCES:
        raw = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        digest.update(rel.encode())
        digest.update(raw)
    return digest.hexdigest()


def main() -> int:
    if not DIST.exists():
        print(f"error: {DIST.relative_to(ROOT)} does not exist - run `npm run build` first")
        return 1

    built_index = DIST / "index.html"
    if not built_index.exists():
        print("error: the build produced no index.html")
        return 1

    # A build made without the right base would publish asset URLs that 404 in place.
    # Catching it here is cheaper than catching it in a browser after a push.
    html = built_index.read_text(encoding="utf-8")
    if "/Indic_LLM/8_attention/assets/" not in html:
        print(
            "error: built index.html does not reference /Indic_LLM/8_attention/assets/.\n"
            "       The base path is wrong; rebuild with the default BASE_PATH."
        )
        return 1

    if ASSETS.exists():
        shutil.rmtree(ASSETS)

    for name in COPY_DIRS:
        src = DIST / name
        if src.exists():
            shutil.copytree(src, ROOT / name)

    for name in COPY_FILES:
        shutil.copy2(DIST / name, ROOT / name)

    published = sorted(ASSETS.iterdir()) if ASSETS.exists() else []
    total = sum(p.stat().st_size for p in published) + INDEX.stat().st_size

    info = {
        "data_fingerprint": fingerprint(),
        "assets": [p.name for p in published],
    }
    (ROOT / "build-info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )

    print(f"published to {ROOT.name}/")
    print(f"  index.html + {len(published)} asset file(s), {total / 1024:.0f} KB total")
    print(f"  data fingerprint {info['data_fingerprint'][:12]}")
    print("  live at https://rahulni.github.io/Indic_LLM/8_attention/ once committed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
