-- PacketShield-Agentic pgvector schema.
-- Applied automatically by docker-compose (mounted into
-- /docker-entrypoint-initdb.d), or manually with:
--   psql "$DATABASE_URL" -f agent/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cve_vectors (
    cve_id      TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    severity    TEXT NOT NULL,
    cvss        NUMERIC NOT NULL,
    embedding   vector(64) NOT NULL
);

-- Cosine-similarity search benefits from an IVFFlat index once the table
-- grows; harmless on small seed data.
CREATE INDEX IF NOT EXISTS cve_vectors_embedding_idx
    ON cve_vectors USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
