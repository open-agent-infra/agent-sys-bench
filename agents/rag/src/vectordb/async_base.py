from typing import Dict, Optional, List, Union, Iterable, Any
from abc import ABC, abstractmethod
import asyncio
import uuid
from langchain_core.documents import Document


class AsyncBaseVectorDBClient(ABC):
    """Async base class for vector database clients."""

    @abstractmethod
    async def store(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store documents and embeddings asynchronously."""
        pass

    async def add_documents(
        self, documents: list[Document], embeddings: list[list[float]], **kwargs: Any
    ) -> list[str]:
        """Add documents with embeddings asynchronously."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return await self.store(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            **kwargs,
        )

    async def search(
        self, query: str, embedding: list[float], search_type: str, **kwargs: Any
    ) -> list[Document]:
        """Search documents asynchronously."""
        if search_type == "similarity":
            return await self.similarity_search(query, embedding, **kwargs)
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = await self.similarity_search_with_relevance_scores(
                query=query, embedding=embedding, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        elif search_type == "mmr":
            return await self.max_marginal_relevance_search(
                query=query, embedding=embedding, **kwargs
            )
        else:
            msg = (
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity', 'similarity_score_threshold'"
                " or 'mmr'."
            )
            raise ValueError(msg)

    @abstractmethod
    async def similarity_search(
        self, query: str, embedding: list[float], **kwargs: Any
    ) -> list[Document]:
        """Perform similarity search asynchronously."""
        raise NotImplementedError

    @abstractmethod
    async def batch_similarity_search(
        self, queries: list[str], embeddings: list[list[float]], **kwargs: Any
    ) -> list[list[Document]]:
        """Perform batch similarity search asynchronously."""
        raise NotImplementedError

    @abstractmethod
    async def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with scores asynchronously."""
        raise NotImplementedError

    async def similarity_search_with_relevance_scores(
        self, query: str, embedding: list[float], **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with relevance scores asynchronously."""
        raise NotImplementedError

    async def max_marginal_relevance_search(
        self, query: str, embedding: list[float], **kwargs: Any
    ) -> list[Document]:
        """Perform max marginal relevance search asynchronously."""
        raise NotImplementedError

    @abstractmethod
    async def drop_collection(self) -> None:
        """Drop collection asynchronously."""
        raise NotImplementedError

    @abstractmethod
    async def collection_exists(self) -> bool:
        """Check if collection exists asynchronously."""
        raise NotImplementedError

    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics asynchronously."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Close client asynchronously."""
        raise NotImplementedError


class AsyncVectorDBClientMixin:
    """Mixin class to add async capabilities to existing sync clients."""

    def __init__(self, sync_client):
        self.sync_client = sync_client
        self._executor = None

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run synchronous function in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    async def store(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store documents and embeddings asynchronously."""
        return await self._run_in_executor(
            self.sync_client.store, texts, embeddings, metadatas, ids=ids, **kwargs
        )

    async def similarity_search(
        self, query: str, embedding: list[float], **kwargs: Any
    ) -> list[Document]:
        """Perform similarity search asynchronously."""
        return await self._run_in_executor(
            self.sync_client.similarity_search, query, embedding, **kwargs
        )

    async def batch_similarity_search(
        self, queries: list[str], embeddings: list[list[float]], **kwargs: Any
    ) -> list[list[Document]]:
        """Perform batch similarity search asynchronously."""
        return await self._run_in_executor(
            self.sync_client.batch_similarity_search, queries, embeddings, **kwargs
        )

    async def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with scores asynchronously."""
        return await self._run_in_executor(
            self.sync_client.similarity_search_with_score, *args, **kwargs
        )

    async def drop_collection(self) -> None:
        """Drop collection asynchronously."""
        return await self._run_in_executor(self.sync_client.drop_collection)

    async def collection_exists(self) -> bool:
        """Check if collection exists asynchronously."""
        if hasattr(self.sync_client, "collection_exists"):
            return await self._run_in_executor(self.sync_client.collection_exists)
        # Fallback implementation
        try:
            await self.get_collection_stats()
            return True
        except:
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics asynchronously."""
        if hasattr(self.sync_client, "get_collection_stats"):
            return await self._run_in_executor(self.sync_client.get_collection_stats)
        return {"status": "unknown"}

    async def close(self) -> None:
        """Close client asynchronously."""
        if hasattr(self.sync_client, "close"):
            await self._run_in_executor(self.sync_client.close)
        if self._executor:
            self._executor.shutdown()


def create_async_client(sync_client) -> AsyncVectorDBClientMixin:
    """Create an async wrapper for a synchronous vector DB client."""
    return AsyncVectorDBClientMixin(sync_client)
