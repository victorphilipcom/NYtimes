"""Embedding service abstraction — supports local and OpenAI backends."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import duckdb
import numpy as np

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


class EmbeddingBackend(ABC):
    """Abstract embedding backend."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (n, dim) array."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class LocalEmbeddingBackend(EmbeddingBackend):
    """Uses sentence-transformers for local embedding."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or get_settings().local_embedding_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Uses OpenAI API for embeddings."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or get_settings().openai_embedding_model
        self._client = None
        self._dim: int | None = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=get_settings().openai_api_key)
        return self._client

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        # OpenAI allows up to 2048 inputs; batch in chunks of 100
        all_embeddings = []
        for i in range(0, len(texts), 100):
            chunk = texts[i : i + 100]
            response = client.embeddings.create(model=self._model_name, input=chunk)
            chunk_embs = [d.embedding for d in response.data]
            all_embeddings.extend(chunk_embs)
            if self._dim is None:
                self._dim = len(chunk_embs[0])
        return np.array(all_embeddings, dtype=np.float32)

    def dimension(self) -> int:
        if self._dim is None:
            # Probe with a dummy
            emb = self.embed_batch(["test"])
            self._dim = emb.shape[1]
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


def get_embedding_backend() -> EmbeddingBackend:
    """Factory function to get the configured embedding backend."""
    settings = get_settings()
    if settings.embedding_backend == "openai":
        if not settings.has_openai:
            log.warning("openai_key_missing_falling_back_to_local")
            return LocalEmbeddingBackend()
        return OpenAIEmbeddingBackend()
    return LocalEmbeddingBackend()


def embed_articles(
    conn: duckdb.DuckDBPyConnection,
    articles: list[dict],
    batch_size: int = 64,
) -> int:
    """Embed articles and store in article_embeddings table.

    Args:
        conn: DuckDB connection
        articles: List of dicts with article_id and normalized_text
        batch_size: Batch size for embedding

    Returns:
        Number of articles embedded
    """
    if not articles:
        return 0

    backend = get_embedding_backend()
    model_name = backend.model_name
    total = 0

    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        texts = [a["normalized_text"] for a in batch]
        ids = [a["article_id"] for a in batch]

        embeddings = backend.embed_batch(texts)
        now = datetime.utcnow()

        for j, (aid, emb) in enumerate(zip(ids, embeddings)):
            emb_blob = pickle.dumps(emb)
            conn.execute(
                """INSERT INTO article_embeddings (article_id, embedding, embedding_model, embedded_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT (article_id) DO UPDATE SET
                       embedding = ?, embedding_model = ?, embedded_at = ?""",
                [aid, emb_blob, model_name, now, emb_blob, model_name, now],
            )
        total += len(batch)
        log.info("embedded_batch", batch_num=i // batch_size + 1, articles=len(batch))

    return total
