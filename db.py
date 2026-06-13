"""Postgres + pgvector helpers (Replit-managed Postgres)."""
import os
import psycopg
from pgvector.psycopg import register_vector

EMBED_DIM = 1024

def _url():
    url = os.environ["DATABASE_URL"]
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url

def get_conn():
    conn = psycopg.connect(_url(), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn

def init_schema(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS evidence (
            id        BIGSERIAL PRIMARY KEY,
            source    TEXT,
            question  TEXT,
            answer    TEXT,
            embedding vector({EMBED_DIM})
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS evidence_embedding_idx
        ON evidence USING hnsw (embedding vector_cosine_ops)
    """)

def count_rows(conn):
    return conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]

def upsert_rows(conn, rows):
    """rows: list of (source, question, answer, embedding)."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO evidence (source, question, answer, embedding) "
            "VALUES (%s, %s, %s, %s)",
            rows,
        )

def search(conn, query_embedding, k=3):
    """Cosine-nearest k rows. Returns (question, answer, source, score)."""
    vec = "[" + ",".join(str(x) for x in query_embedding) + "]"
    return conn.execute(
        "SELECT question, answer, source, 1 - (embedding <=> %s::vector) AS score "
        "FROM evidence ORDER BY embedding <=> %s::vector LIMIT %s",
        (vec, vec, k),
    ).fetchall()
