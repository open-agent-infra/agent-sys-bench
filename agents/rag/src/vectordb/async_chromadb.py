from typing import Dict, Optional, List, Union, Iterable, Any
import asyncio
import uuid
import httpx
from langchain_core.documents import Document
from .async_base import AsyncBaseVectorDBClient


def _results_to_docs_and_scores(results: Any) -> List[tuple[Document, float]]:
    """Convert ChromaDB results to documents and scores."""
    return [
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class AsyncChromaVectorDBClient(AsyncBaseVectorDBClient):
    """Async ChromaDB vector database client."""

    def __init__(
        self,
        uri: str,
        collection_name: str,
        collection_metadata: Optional[Dict] = None,
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self._client = None
        self._collection = None
        self._http_client = None

    async def _ensure_client(self):
        """Ensure ChromaDB client is initialized."""
        if self._client is None:
            # Try to use async ChromaDB client if available
            try:
                import chromadb

                # Check if AsyncHttpClient is available (newer versions)
                if hasattr(chromadb, "AsyncHttpClient"):
                    host, port = self.uri.split("://")[1].split(":")
                    self._client = chromadb.AsyncHttpClient(host=host, port=int(port))
                    await self._ensure_collection_async()
                else:
                    # Fallback to sync client
                    host, port = self.uri.split("://")[1].split(":")
                    self._client = chromadb.HttpClient(host=host, port=int(port))
                    await self._ensure_collection_sync()
            except Exception:
                # Fallback to sync client
                import chromadb

                host, port = self.uri.split("://")[1].split(":")
                self._client = chromadb.HttpClient(host=host, port=int(port))
                await self._ensure_collection_sync()

    async def _ensure_collection_async(self):
        """Ensure collection exists using async client."""
        if self._collection is None:
            self._collection = await self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata=self.collection_metadata,
            )

    async def _ensure_collection_sync(self):
        """Ensure collection exists using sync client in executor."""
        if self._collection is None:
            # Run in executor since chromadb operations are sync
            loop = asyncio.get_event_loop()
            self._collection = await loop.run_in_executor(
                None,
                lambda: self._client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata=self.collection_metadata,
                ),
            )

    async def store(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store documents and embeddings asynchronously."""
        await self._ensure_client()

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        texts = list(texts)

        # Run the storage operation in executor
        loop = asyncio.get_event_loop()

        def _store_sync():
            if metadatas:
                # Fill metadatas with empty dicts if somebody
                # did not specify metadata for all texts
                length_diff = len(texts) - len(metadatas)
                if length_diff:
                    metadatas.extend([{}] * length_diff)

                empty_ids = []
                non_empty_ids = []
                for idx, m in enumerate(metadatas):
                    if m:
                        non_empty_ids.append(idx)
                    else:
                        empty_ids.append(idx)

                if non_empty_ids:
                    metadatas_filtered = [metadatas[idx] for idx in non_empty_ids]
                    texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                    embeddings_with_metadatas = (
                        [embeddings[idx] for idx in non_empty_ids]
                        if embeddings
                        else None
                    )
                    ids_with_metadata = [ids[idx] for idx in non_empty_ids]

                    try:
                        self._collection.upsert(
                            metadatas=metadatas_filtered,
                            embeddings=embeddings_with_metadatas,
                            documents=texts_with_metadatas,
                            ids=ids_with_metadata,
                        )
                    except ValueError as e:
                        if "Expected metadata value to be" in str(e):
                            msg = (
                                "Try filtering complex metadata from the document using "
                                "langchain_community.vectorstores.utils.filter_complex_metadata."
                            )
                            raise ValueError(e.args[0] + "\n\n" + msg)
                        else:
                            raise e

                if empty_ids:
                    texts_without_metadatas = [texts[j] for j in empty_ids]
                    embeddings_without_metadatas = (
                        [embeddings[j] for j in empty_ids] if embeddings else None
                    )
                    ids_without_metadatas = [ids[j] for j in empty_ids]
                    self._collection.upsert(
                        embeddings=embeddings_without_metadatas,
                        documents=texts_without_metadatas,
                        ids=ids_without_metadatas,
                    )
            else:
                self._collection.upsert(
                    embeddings=embeddings,
                    documents=texts,
                    ids=ids,
                )
            return ids

        return await loop.run_in_executor(None, _store_sync)

    async def similarity_search(
        self,
        query: str,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Perform similarity search asynchronously."""
        docs_and_scores = await self.similarity_search_with_score(
            query, embedding, k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def _query_collection(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """Query collection asynchronously."""
        await self._ensure_client()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                **kwargs,
            ),
        )

    async def similarity_search_with_score(
        self,
        query: str,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with scores asynchronously."""
        results = await self._query_collection(
            query_embeddings=[embedding],
            n_results=k,
            where=filter,
            **kwargs,
        )
        return _results_to_docs_and_scores(results)

    async def batch_similarity_search(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        k: int = 4,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Perform batch similarity search asynchronously."""
        if not queries or not embeddings:
            return []

        results = await self._query_collection(
            query_embeddings=embeddings,
            n_results=k,
            **kwargs,
        )

        # Process results for each query
        all_docs = []
        for i in range(len(queries)):
            query_results = {
                "documents": [results["documents"][i]],
                "metadatas": [results["metadatas"][i]],
                "distances": [results["distances"][i]],
            }
            docs_and_scores = _results_to_docs_and_scores(query_results)
            all_docs.append([doc for doc, _ in docs_and_scores])

        return all_docs

    async def drop_collection(self) -> None:
        """Drop collection asynchronously."""
        await self._ensure_client()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self._client.delete_collection(name=self.collection_name)
        )
        self._collection = None

    async def collection_exists(self) -> bool:
        """Check if collection exists asynchronously."""
        await self._ensure_client()

        loop = asyncio.get_event_loop()
        try:
            collections = await loop.run_in_executor(
                None, lambda: self._client.list_collections()
            )
            return any(col.name == self.collection_name for col in collections)
        except Exception:
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics asynchronously."""
        await self._ensure_client()

        loop = asyncio.get_event_loop()
        try:
            count = await loop.run_in_executor(None, lambda: self._collection.count())
            return {"count": count, "name": self.collection_name}
        except Exception as e:
            return {"error": str(e), "name": self.collection_name}

    async def close(self) -> None:
        """Close client asynchronously."""
        if self._client:
            await self._client.close()
        if self._http_client:
            await self._http_client.aclose()
