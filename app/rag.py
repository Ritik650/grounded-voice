"""Domain grounding: hybrid retrieval over a document knowledge base.

Embeddings come from fastembed (ONNX Runtime) rather than sentence-transformers,
which would drag torch into an otherwise torch-free image for the same MiniLM-class
quality.

Retrieval is hybrid on purpose. Voice queries arrive via ASR, so they carry
transcription noise and are phrased conversationally ("what's the refund window on a
damaged item") while the KB is written in documentation register. Dense embeddings
handle that paraphrase gap; BM25 catches the exact identifiers -- order numbers, SKUs,
policy names -- that dense vectors blur and that ASR usually gets right because they
are spelled out. Reciprocal Rank Fusion merges the two rankings without needing a
tuned score-scale mapping between them.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from .config import settings

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass
class Chunk:
    text: str
    source: str
    score: float = 0.0

    def cite(self) -> str:
        return f"[{self.source}] {self.text}"


def chunk_document(text: str, source: str, size: int | None = None, overlap: int | None = None):
    """Paragraph-aware chunking with character overlap.

    Splitting on blank lines first keeps semantically whole passages together; the
    fixed-size fallback only kicks in for paragraphs that are too long on their own.
    Overlap exists so a fact straddling a boundary is still fully present in one chunk.
    """
    size = size or settings.chunk_chars
    overlap = overlap or settings.chunk_overlap
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[Chunk] = []
    buf = ""
    for para in paragraphs:
        if len(para) > size:
            if buf:
                chunks.append(Chunk(buf, source))
                buf = ""
            for i in range(0, len(para), size - overlap):
                chunks.append(Chunk(para[i : i + size], source))
        elif len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            chunks.append(Chunk(buf, source))
            buf = para
    if buf:
        chunks.append(Chunk(buf, source))
    return chunks


class KnowledgeBase:
    def __init__(self, url: str | None = None, collection: str | None = None):
        url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.client = (
            QdrantClient(location=":memory:")
            if url == ":memory:"
            else QdrantClient(url=url, prefer_grpc=False)
        )

        from fastembed import SparseTextEmbedding, TextEmbedding

        self.dense = TextEmbedding(settings.embed_model)
        self.sparse = SparseTextEmbedding("Qdrant/bm25")
        self._dense_dim = len(next(iter(self.dense.embed(["dimension probe"]))))

    # -- ingestion ------------------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(size=self._dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                # IDF modifier is required for BM25 scoring to mean anything; without
                # it every term is weighted equally and common words dominate.
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def ingest_texts(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        dense_vecs = list(self.dense.embed(texts))
        sparse_vecs = list(self.sparse.embed(texts))

        points = [
            models.PointStruct(
                id=uuid.uuid5(uuid.NAMESPACE_URL, f"{c.source}:{i}:{_hash(c.text)}").hex,
                vector={
                    "dense": dense.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload={"text": c.text, "source": c.source},
            )
            for i, (c, dense, sparse) in enumerate(
                zip(chunks, dense_vecs, sparse_vecs, strict=True)
            )
        ]
        self.client.upsert(self.collection, points=points, wait=True)
        return len(points)

    def ingest_directory(self, directory: Path | None = None, recreate: bool = True) -> int:
        directory = Path(directory or settings.kb_dir)
        files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if not files:
            log.warning("No knowledge-base documents found in %s", directory)

        self.ensure_collection(recreate=recreate)
        chunks: list[Chunk] = []
        for path in files:
            chunks.extend(chunk_document(path.read_text(encoding="utf-8"), source=path.name))
        count = self.ingest_texts(chunks)
        log.info("Ingested %d chunks from %d documents", count, len(files))
        return count

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    # -- retrieval ------------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        k = k or settings.retrieval_top_k
        if not query.strip() or self.count() == 0:
            return []

        dense_vec = next(iter(self.dense.embed([query]))).tolist()
        sparse_raw = next(iter(self.sparse.embed([query])))
        sparse_vec = models.SparseVector(
            indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist()
        )

        # Over-fetch on each arm (k*4) so fusion has enough depth to reorder; RRF on
        # two top-k lists that barely overlap just returns the dense list back.
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=k * 4),
                models.Prefetch(query=sparse_vec, using="sparse", limit=k * 4),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [
            Chunk(
                text=p.payload.get("text", ""),
                source=p.payload.get("source", "unknown"),
                score=float(p.score),
            )
            for p in response.points
        ]

    def context_for(self, query: str, k: int | None = None) -> tuple[str, list[Chunk]]:
        """Retrieved chunks rendered as a prompt-ready context block."""
        chunks = self.retrieve(query, k=k)
        context = "\n\n".join(f"[{i + 1}] ({c.source}) {c.text}" for i, c in enumerate(chunks))
        return context, chunks

    def vocabulary_hint(self, max_terms: int = 40) -> str:
        """Domain terms to bias ASR toward, passed as Whisper's initial_prompt.

        Grounding is not only an LLM concern: if ASR mangles a product name, retrieval
        never sees it. Capitalised multi-word terms in the KB are a cheap proxy for the
        proper nouns Whisper is most likely to get wrong.
        """
        if self.count() == 0:
            return ""
        records, _ = self.client.scroll(self.collection, limit=200, with_payload=True)
        text = " ".join(r.payload.get("text", "") for r in records)
        terms = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:[ -][A-Z][a-zA-Z0-9]+)*\b", text)
        seen: dict[str, int] = {}
        for t in terms:
            if len(t) > 3:
                seen[t] = seen.get(t, 0) + 1
        ranked = sorted(seen, key=lambda t: -seen[t])[:max_terms]
        return ", ".join(ranked)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
