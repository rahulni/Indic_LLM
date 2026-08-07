# -*- coding: utf-8 -*-
"""TDES -- a Training Data Execution System.

A small but complete data path:

    documents -> tokenized shards -> manifests -> mixture schedule -> packing
    -> batches -> training -> consumption ledger -> learning ledger
    -> checkpoint -> crash -> resume -> replay -> audit

Stdlib plus ``regex`` only. Nothing here is a stub: the model computes real
cross-entropy with hand-written backprop, the selector computes a real gradient
cosine, the loader owns a real cache, and every number in the evidence bundle is
read back off an artifact the run produced.
"""

__version__ = "1.0.0"
