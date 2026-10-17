"""Deterministic text embeddings for the PacketShield-Agentic control plane.

Produces a fixed-dimension, L2-normalized embedding from character n-gram
hashes (SHA-256). This is deliberately dependency-free so the pgvector
similarity path works without an external embedding model: the same function
feeds both the CVE seed loader (agent/load_cves.py) and live queries
(agent/orchestrator.py), so cosine similarity in pgvector is consistent.

For production, swap embed_text() for a real encoder (e.g. a
sentence-transformers model served over HTTP) and re-run the seed loader —
the pgvector schema and query code are unchanged.
"""

from __future__ import annotations

import hashlib
import math
import re

DIM = 64
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _ngrams(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    grams: list[str] = []
    for tok in tokens:
        grams.append(tok)
        for i in range(len(tok) - 2):
            grams.append(tok[i : i + 3])
    return grams


def embed_text(text: str, dim: int = DIM) -> list[float]:
    """Return a deterministic L2-normalized embedding of `text`."""
    vec = [0.0] * dim
    for gram in _ngrams(text):
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def to_pgvector_literal(vec: list[float]) -> str:
    """Format an embedding as a pgvector input literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
