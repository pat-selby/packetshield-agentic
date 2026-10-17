#!/usr/bin/env python3
"""Seed the pgvector CVE table from agent/cve_seed.json.

Usage:
    DATABASE_URL=postgresql://packetshield:packetshield@localhost:5432/packetshield \
        python agent/load_cves.py [--seed agent/cve_seed.json]

Embeddings are produced by agent/embeddings.py so they are comparable with
the live queries issued by the Threat Analyst agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings import DIM, embed_text, to_pgvector_literal  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="agent/cve_seed.json")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 2

    import asyncpg

    with open(args.seed, "r", encoding="utf-8") as fh:
        seed = json.load(fh)

    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        inserted = 0
        for entry in seed:
            text = f"{entry['cve_id']} {entry['description']} " + " ".join(
                entry.get("keywords", [])
            )
            vec = to_pgvector_literal(embed_text(text, DIM))
            await conn.execute(
                """
                INSERT INTO cve_vectors (cve_id, description, severity, cvss, embedding)
                VALUES ($1, $2, $3, $4, $5::vector)
                ON CONFLICT (cve_id) DO UPDATE SET
                    description = EXCLUDED.description,
                    severity    = EXCLUDED.severity,
                    cvss        = EXCLUDED.cvss,
                    embedding   = EXCLUDED.embedding
                """,
                entry["cve_id"],
                entry["description"],
                entry["severity"],
                entry["cvss"],
                vec,
            )
            inserted += 1
        print(f"Seeded {inserted} CVE vectors into cve_vectors.")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
