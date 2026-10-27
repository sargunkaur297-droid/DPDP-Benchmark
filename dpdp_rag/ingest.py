
"""Ingest the chunked corpus into a persistent Chroma collection.

Progress is checkpointed after every batch so ingestion can resume.
Document embeddings use the same SentenceTransformer model as queries.
"""

import hashlib
import json

import chromadb
from sentence_transformers import SentenceTransformer

from . import chunker, config

STATE_FILE = config.DB_DIR / ".ingest_state.json"
BATCH = config.EMBED_BATCH

_embedder = None


def _corpus_hash(docs: list[dict]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x["id"]):
        h.update(d["id"].encode())
        h.update(d["text"].encode())
    return h.hexdigest()


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"corpus_hash": None, "done": []}


def _save_state(state: dict) -> None:
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed document chunks using the configured embedding model."""
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)

    return _embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def main(reset: bool = False) -> None:
    docs = chunker.build_documents()
    chash = _corpus_hash(docs)

    state = _load_state()

    if state.get("corpus_hash") != chash:
        if state.get("corpus_hash") is not None:
            print("Corpus changed since last run — restarting from scratch")
        state = {"corpus_hash": chash, "done": []}

    done = set(state["done"])

    chroma = chromadb.PersistentClient(path=str(config.DB_DIR))

    if reset and config.COLLECTION in [
        c.name for c in chroma.list_collections()
    ]:
        chroma.delete_collection(config.COLLECTION)
        state = {"corpus_hash": chash, "done": []}
        done = set()

    col = chroma.get_or_create_collection(
        config.COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    pending = [d for d in docs if d["id"] not in done]

    if not pending and col.count() >= len(docs):
        print("Nothing to do — checkpoint and collection are up to date")
        return

    if not done and col.count() >= len(docs):
        done.update(d["id"] for d in docs)
        state["done"] = sorted(done)
        _save_state(state)
        pending = []
        print(f"Seeded checkpoint from existing collection ({len(docs)} vectors)")
    else:
        pending = [d for d in docs if d["id"] not in done]

    print(
        f"{len(docs)} chunks, {len(done)} already ingested, "
        f"{len(pending)} to do"
    )

    for i in range(0, len(pending), BATCH):
        part = pending[i:i + BATCH]

        col.upsert(
            ids=[d["id"] for d in part],
            documents=[d["text"] for d in part],
            metadatas=[d["metadata"] for d in part],
            embeddings=_embed([d["text"] for d in part]),
        )

        done.update(d["id"] for d in part)
        state["done"] = sorted(done)
        _save_state(state)

        print(f"  embedded {len(done)}/{len(docs)}")

    print(f"Done. Collection '{config.COLLECTION}' at {config.DB_DIR}")


if __name__ == "__main__":
    main(reset=True)