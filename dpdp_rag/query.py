
"""Retrieval + generation with section-level citations."""

import math
import os
import re
import threading
import time
from collections import Counter
from datetime import date
from typing import Callable

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from . import config

_STOP = {
    "the", "and", "for", "must", "how", "what", "which", "does", "did",
    "with", "that", "from", "are", "was", "were", "has", "have", "not",
    "can", "may", "within", "into", "under", "any", "its", "their",
}

_BM25_K1, _BM25_B = 1.5, 0.75

AUTHORITY = {
    "act": 1.0,
    "rules": 1.0,
    "notification": 1.0,
    "corrigendum": 1.0,
    "parliament_qa": 0.9,
    "summary": 0.8,
    "faq": 0.7,
    "draft_rules": 0.65,
}

CTX_SAFETY = 192
MSG_OVERHEAD = 8
BLOCK_OVERHEAD = 12
HISTORY_SHARE = 0.25

_bm25_cache: dict = {
    "size": None, "docs": None, "df": None, "avgdl": 0.0
}
_bm25_lock = threading.Lock()

_dense_cache: dict = {
    "size": None, "device": None, "matrix": None,
    "documents": None, "metadatas": None,
}
_dense_lock = threading.Lock()
_chroma_lock = threading.RLock()

_chroma_client = None
_chroma_collection = None
_embedder = None
_groq_client = None

ProgressCallback = Callable[[str, int], None]


def _report(
    callback: ProgressCallback | None,
    stage: str,
    progress: int,
) -> None:
    if callback is not None:
        callback(stage, progress)


def _retry(
    fn,
    *,
    what: str,
    attempts: int = 5,
    base_wait: float = 5.0,
):
    """Retry transient API or inference failures with backoff."""
    last = None

    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            wait = base_wait * (2 ** i)
            print(
                f"[retry] {what} failed ({e}); backing off "
                f"{wait:.0f}s ({i + 1}/{attempts})"
            )
            if i < attempts - 1:
                time.sleep(wait)

    detail = f": {type(last).__name__}: {last}" if last else ""
    raise RuntimeError(
        f"{what} failed after {attempts} attempts{detail}"
    ) from last


def _get_embedder():
    """Load the configured sentence-transformer model once."""
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)

    return _embedder


def _get_groq():
    """Create the Groq client once using the configured API key."""
    global _groq_client

    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Configure it in Streamlit Secrets."
            )

        _groq_client = Groq(api_key=api_key)

    return _groq_client


def _collection():
    """Return one process-wide Chroma collection."""
    global _chroma_client, _chroma_collection

    with _chroma_lock:
        if _chroma_collection is None:
            _chroma_client = chromadb.PersistentClient(
                path=str(config.DB_DIR)
            )
            _chroma_collection = (
                _chroma_client.get_or_create_collection(
                    config.COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
            )

        return _chroma_collection


def warmup() -> int:
    """Initialize Chroma before concurrent worker slots start."""
    with _chroma_lock:
        return _collection().count()


def _tokenize(text: str) -> list[str]:
    return [
        _stem(t)
        for t in re.findall(
            r"[a-z][a-z\-]{2,}|\d{2,}", text.lower()
        )
        if t not in _STOP
    ]


def _bm25_search(question: str, n: int) -> list[dict]:
    """Run keyword-based BM25 retrieval over the corpus."""
    col = _collection()

    with _chroma_lock:
        total = col.count()

    with _bm25_lock:
        if (
            _bm25_cache["size"] != total
            or _bm25_cache["docs"] is None
        ):
            with _chroma_lock:
                got = col.get(
                    include=["documents", "metadatas"]
                )

            docs = []
            df: Counter = Counter()

            for doc, meta in zip(
                got["documents"], got["metadatas"]
            ):
                tokens = _tokenize(doc)
                docs.append({
                    "text": doc,
                    "meta": meta,
                    "tokens": tokens,
                    "len": len(tokens),
                })
                df.update(set(tokens))

            _bm25_cache.update(
                size=total,
                docs=docs,
                df=df,
                avgdl=(
                    sum(d["len"] for d in docs)
                    / max(len(docs), 1)
                ),
            )

    docs = _bm25_cache["docs"]
    df = _bm25_cache["df"]
    avgdl = _bm25_cache["avgdl"]

    n_docs = len(docs)
    q_tokens = _tokenize(question)
    scored = []

    for d in docs:
        tf = Counter(d["tokens"])
        score = 0.0

        for t in q_tokens:
            if t not in tf:
                continue

            idf = math.log(
                1 + (n_docs - df[t] + 0.5)
                / (df[t] + 0.5)
            )
            f = tf[t]

            score += (
                idf * f * (_BM25_K1 + 1)
                / (
                    f + _BM25_K1 * (
                        1 - _BM25_B
                        + _BM25_B * d["len"] / max(avgdl, 1)
                    )
                )
            )

        if score > 0:
            scored.append({
                "text": d["text"],
                "meta": d["meta"],
                "bm25": score,
            })

    scored.sort(key=lambda h: -h["bm25"])
    return scored[:n]


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[:-len(suf)]
    return w


SYSTEM_PROMPT_TEMPLATE = """You are a legal research assistant specialised in
India's Digital Personal Data Protection Act, 2023 (DPDP Act) and the Digital
Personal Data Protection Rules, 2025 (DPDP Rules).

Rules:
- Answer using only the provided context. Check all retrieved passages
  carefully for requested facts, especially numbers, dates, penalties,
  and exceptions. If the answer is present, answer directly.
- For dates expressed as offsets, calculate from the publication date
  stated in the context. Eighteen months after 13 November 2025 is
  13 May 2027; one year after that date is 13 November 2026.
- Distinguish the DPDP Act's commencement notification from Rule 1
  of the DPDP Rules. Do not assume the Act's Schedule commenced
  immediately unless the retrieved notification explicitly says so.
- Cite every legal claim using the exact source labels in the context.
  Never invent facts or citations.
- If the context genuinely does not establish an answer, say so.
"""


def system_prompt() -> str:
    """Return the legal assistant system prompt."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        today=date.today().strftime("%d %B %Y")
    )


def _hybrid_rerank(
    question: str,
    dense: list[dict],
    bm25: list[dict],
    k: int,
) -> list[dict]:
    """Fuse dense and BM25 rankings, weighted by source authority."""
    by_id = {}

    for rank, h in enumerate(dense):
        h = dict(h)
        h["dense_rank"] = rank
        key = (
            h["meta"]["unit"]
            + f"#{h['meta']['part']}:"
            + h["text"][:40]
        )
        by_id[key] = h

    for rank, h in enumerate(bm25):
        key = (
            h["meta"]["unit"]
            + f"#{h['meta']['part']}:"
            + h["text"][:40]
        )
        h = by_id.setdefault(key, dict(h))
        h.setdefault("dense_rank", len(dense) + rank)
        h["bm25_rank"] = rank

    hits = list(by_id.values())
    nd, nb = max(len(dense), 1), max(len(bm25), 1)

    for h in hits:
        dense_score = 1 - h.get("dense_rank", nd) / nd
        bm25_score = 1 - h.get("bm25_rank", nb) / nb

        h["authority"] = AUTHORITY.get(
            h["meta"].get("doc_type"), 0.8
        )
        h["score"] = (
            0.55 * dense_score + 0.45 * bm25_score
        ) * h["authority"]

    ranked = sorted(hits, key=lambda h: -h["score"])
    return ranked[:k]


def retrieve(
    question: str,
    k: int = config.TOP_K,
    progress: ProgressCallback | None = None,
) -> list[dict]:
    """Retrieve relevant passages using dense and keyword search."""
    _report(progress, "Opening the legal index", 12)
    col = _collection()

    _report(progress, "Embedding your question", 20)

    emb = _get_embedder().encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    _report(progress, "Searching semantic matches", 30)

    with _chroma_lock:
        result_count = min(max(16, k * 2), col.count() or 16)

    dense = _gpu_dense_search(col, emb, result_count)

    if dense is None:
        with _chroma_lock:
            res = col.query(
                query_embeddings=[emb],
                n_results=min(max(16, k * 2), col.count() or 16),
                include=[
                    "documents", "metadatas", "distances"
                ],
            )

        dense = [
            {"text": doc, "meta": meta, "distance": dist}
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ]

    _report(progress, "Checking exact legal terms", 40)
    bm25 = _bm25_search(question, n=max(16, k * 2))

    _report(progress, "Ranking the strongest sources", 48)

    return _hybrid_rerank(question, dense, bm25, k)


def _gpu_dense_search(
    collection,
    query_embedding: list[float],
    n: int,
) -> list[dict] | None:
    """Use GPU-backed dense search when PyTorch and an accelerator exist."""
    if os.environ.get("DPDP_GPU_RETRIEVAL", "1") == "0":
        return None

    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        device = "cuda"
    elif (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    else:
        return None

    with _chroma_lock:
        total = collection.count()

    if total == 0:
        return None

    with _dense_lock:
        if (
            _dense_cache["size"] != total
            or _dense_cache["device"] != device
            or _dense_cache["matrix"] is None
        ):
            with _chroma_lock:
                got = collection.get(
                    include=[
                        "embeddings", "documents", "metadatas"
                    ]
                )

            if (
                got["embeddings"] is None
                or len(got["embeddings"]) == 0
            ):
                return None

            matrix = torch.tensor(
                got["embeddings"],
                dtype=torch.float32,
                device=device,
            )
            matrix = torch.nn.functional.normalize(matrix, dim=1)

            _dense_cache.update(
                size=total,
                device=device,
                matrix=matrix,
                documents=got["documents"],
                metadatas=got["metadatas"],
            )
            print(
                f"[gpu] cached {total} retrieval vectors on {device}"
            )

    query = torch.tensor(
        query_embedding,
        dtype=torch.float32,
        device=device,
    )
    query = torch.nn.functional.normalize(query, dim=0)

    scores = _dense_cache["matrix"] @ query
    values, indices = torch.topk(scores, k=min(n, total))

    return [
        {
            "text": _dense_cache["documents"][index],
            "meta": _dense_cache["metadatas"][index],
            "distance": 1.0 - float(score),
        }
        for score, index in zip(
            values.detach().cpu().tolist(),
            indices.detach().cpu().tolist(),
        )
    ]


def _block(hit: dict, n: int = 0) -> str:
    """Render a passage with its source citation."""
    m = hit["meta"]
    text = hit["text"]

    embedded = f"{m['source'].split(' (')[0]}, {m['unit']}."
    first, sep, rest = text.partition("\n")

    if sep and first.strip() == embedded:
        text = rest

    return f"[{n}] {m['source']}, {m['unit']}\n{text}"


def build_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(
        _block(h, i) for i, h in enumerate(hits, 1)
    )


def _tokens(text: str) -> int:
    """Estimate token count conservatively."""
    return int(len(text) / 3.4) + 1


def _fit_history(
    history: list[dict],
    budget: int,
) -> list[dict]:
    """Keep recent complete turns that fit the budget."""
    kept, used = [], 0

    for msg in reversed(history):
        cost = _tokens(msg["content"]) + MSG_OVERHEAD

        if used + cost > budget:
            break

        kept.append(msg)
        used += cost

    kept.reverse()

    if kept and kept[0]["role"] == "assistant":
        kept = kept[1:]

    return kept


def _fit_context(
    hits: list[dict],
    budget: int,
) -> list[dict]:
    """Keep the highest-ranked passages that fit the budget."""
    kept, used = [], 0

    for h in hits:
        cost = _tokens(_block(h)) + BLOCK_OVERHEAD

        if kept and used + cost > budget:
            break

        kept.append(h)
        used += cost

    return kept


def answer(
    question: str,
    k: int = config.TOP_K,
    model: str = config.CHAT_MODEL,
    history: list[dict] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[str, list[dict]]:
    """Generate a grounded answer and return the passages used."""
    hits = retrieve(question, k, progress=progress)
    history = history or []

    _report(progress, "Building the grounded legal prompt", 55)

    prompt = system_prompt()

    budget = (
        config.NUM_CTX
        - config.MAX_OUTPUT_TOKENS
        - CTX_SAFETY
        - _tokens(prompt)
        - _tokens(question)
    )

    kept_history = _fit_history(
        history,
        int(max(budget, 0) * HISTORY_SHARE),
    )

    budget -= sum(
        _tokens(m["content"]) + MSG_OVERHEAD
        for m in kept_history
    )

    kept = _fit_context(hits, budget)

    if (
        len(kept) < len(hits)
        or len(kept_history) < len(history)
    ):
        print(
            f"[budget] context budget: kept {len(kept)}/{len(hits)} "
            f"passages, {len(kept_history)}/{len(history)} history messages"
        )

    context = build_context(kept)

    messages = [{"role": "system", "content": prompt}]
    messages.extend(kept_history)
    messages.append({
        "role": "user",
        "content": (
            f"Context:\n\n{context}\n\nQuestion: {question}"
        ),
    })

    def generate() -> str:
        parts: list[str] = []
        characters = 0
        last_progress = 59

        stream = _get_groq().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            text = chunk.choices[0].delta.content or ""
            parts.append(text)
            characters += len(text)

            estimate = min(
                96,
                60 + int(
                    36 * characters
                    / max(config.MAX_OUTPUT_TOKENS * 4, 1)
                ),
            )

            if estimate > last_progress:
                last_progress = estimate
                _report(
                    progress,
                    f"Writing the grounded answer · ~{characters // 4} tokens",
                    estimate,
                )

        return "".join(parts)

    _report(progress, "Starting answer generation with Groq", 60)

    answer_text = _retry(
        generate,
        what="Groq chat completion",
    )

    _report(progress, "Finalizing citations", 98)

    return answer_text, kept
