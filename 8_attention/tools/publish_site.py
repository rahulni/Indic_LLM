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

Usage:
    cd app && npm run build
    cd .. && python tools/publish_site.py
"""

from __future__ import annotations

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

    published = sorted(p for p in ASSETS.iterdir()) if ASSETS.exists() else []
    total = sum(p.stat().st_size for p in published) + INDEX.stat().st_size

    print(f"published to {ROOT.name}/")
    print(f"  index.html + {len(published)} asset file(s), {total / 1024:.0f} KB total")
    print("  live at https://rahulni.github.io/Indic_LLM/8_attention/ once committed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
