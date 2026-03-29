import os
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from schemas import PolicyChunk, RetrievalResult

load_dotenv()

VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", "./data/vectorstore"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))


class PolicyRetriever:
    """
    Wraps the FAISS index and embedding model.
    Call .retrieve(queries) to get relevant policy chunks.
    """

    def __init__(self):
        self._index = None
        self._chunks = None
        self._model = None
        self._loaded = False

    def load(self):
        """Lazy-load: only hits disk when first retrieval is needed."""
        if self._loaded:
            return

        index_path = VECTOR_DB_PATH / "policy_index.faiss"
        meta_path = VECTOR_DB_PATH / "chunks_metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {VECTOR_DB_PATH}. "
                "Run: python data_pipeline.py --create-sample-policies"
            )

        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self._chunks = pickle.load(f)
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._loaded = True
        print(f"[Retriever] Loaded {len(self._chunks)} chunks from {VECTOR_DB_PATH}")

    def retrieve(self, queries: List[str], top_k: int = TOP_K) -> RetrievalResult:
        """
        Run multiple queries, merge results, deduplicate, filter by threshold.

        Why multiple queries? The triage agent produces 2-4 targeted queries
        (e.g., "refund window 30 days" AND "damaged item policy"). Running all
        of them and deduplicating by chunk_id gives better recall than a single
        broad query.
        """
        self.load()

        seen_ids = set()
        all_chunks: List[PolicyChunk] = []

        for query in queries:
            # Embed + normalize for cosine similarity
            query_vec = self._model.encode([query], show_progress_bar=False)
            query_vec = np.array(query_vec, dtype="float32")
            faiss.normalize_L2(query_vec)

            scores, indices = self._index.search(query_vec, top_k)

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                chunk_meta = self._chunks[idx]
                chunk_id = chunk_meta["chunk_id"]

                # Skip if already seen or below threshold
                if chunk_id in seen_ids:
                    continue
                if float(score) < SIMILARITY_THRESHOLD:
                    continue

                seen_ids.add(chunk_id)
                all_chunks.append(PolicyChunk(
                    chunk_id=chunk_id,
                    source=chunk_meta["source"],
                    content=chunk_meta["content"],
                    score=float(score),
                ))

        # Sort by score descending
        all_chunks.sort(key=lambda c: c.score, reverse=True)

        return RetrievalResult(
            chunks=all_chunks,
            queries_used=queries,
            total_retrieved=len(all_chunks),
        )

    def format_context(self, result: RetrievalResult) -> str:
        """
        Format retrieved chunks into a prompt-ready context string.
        Each chunk is labeled with its source for the LLM to cite.
        """
        if not result.chunks:
            return "NO RELEVANT POLICY FOUND."

        parts = []
        for i, chunk in enumerate(result.chunks, 1):
            parts.append(
                f"[POLICY SOURCE {i}: {chunk.source}]\n{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)


# Singleton — one instance reused across all graph executions
retriever = PolicyRetriever()
