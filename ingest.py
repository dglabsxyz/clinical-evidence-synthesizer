"""Load a MedQuAD subset from HuggingFace, embed with Qwen, store in pgvector.

Usage:  python ingest.py [N]      # N = number of Q&A pairs (default 500)
"""
import sys
from datasets import load_dataset
import db
import embeddings

def pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    conn = db.get_conn()
    db.init_schema(conn)
    print("Existing rows:", db.count_rows(conn))

    print("Loading MedQuAD (lavita/MedQuAD) from HuggingFace...")
    ds = load_dataset("lavita/MedQuAD", split="train")
    cols = ds.column_names
    qcol = pick(cols, "question")
    acol = pick(cols, "answer")
    scol = pick(cols, "document_url", "url", "document_source", "source")
    print("Detected columns -> question:", qcol, "answer:", acol, "source:", scol)

    ds = ds.filter(lambda r: r.get(qcol) and r.get(acol))
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    questions = [r[qcol] for r in ds]
    answers = [r[acol] for r in ds]
    sources = [(r.get(scol) if scol else None) or "MedQuAD" for r in ds]

    print(f"Embedding {len(questions)} Q&A pairs via {embeddings.EMBED_MODEL}...")
    texts = [f"{q}\n{a}" for q, a in zip(questions, answers)]
    vectors = embeddings.embed_texts(texts)

    rows = list(zip(sources, questions, answers, vectors))
    db.upsert_rows(conn, rows)
    print("Inserted", len(rows), "rows. Total now:", db.count_rows(conn))

if __name__ == "__main__":
    main()
