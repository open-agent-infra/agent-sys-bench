from typing import Dict, Optional, List, Union, Iterable, Any
import asyncio
import uuid
import httpx
from langchain_core.documents import Document
from .async_base import AsyncBaseVectorDBClient


class AsyncMilvusVectorDBClient(AsyncBaseVectorDBClient):
    """Async Milvus vector database client using native async operations."""

    def __init__(
        self,
        uri: str,
        collection_name: str,
        embedding_dimension: int,
        collection_metadata: Optional[Dict] = None,
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.collection_metadata = collection_metadata
        self._client = None
        self._http_client = None
        self._metric_type = "COSINE"

    async def _ensure_client(self):
        """Ensure Milvus client is initialized."""
        if self._client is None:
            # Try to use native async Milvus client if available
            try:
                # Check if AsyncMilvusClient is available (newer versions)
                from pymilvus import AsyncMilvusClient

                self._client = AsyncMilvusClient(self.uri)
                await self._ensure_collection_async()
            except ImportError:
                # Fallback to sync client with executor
                from pymilvus import MilvusClient
                from pymilvus.client.constants import ConsistencyLevel

                loop = asyncio.get_event_loop()
                self._client = await loop.run_in_executor(
                    None, lambda: MilvusClient(self.uri)
                )
                await self._ensure_collection_sync()

    async def _ensure_collection_async(self):
        """Ensure collection exists using async client."""
        import logging

        # Temporarily suppress logging for expected collection existence check
        milvus_logger = logging.getLogger("pymilvus")
        original_level = milvus_logger.level
        milvus_logger.setLevel(logging.CRITICAL)

        try:
            # Try to load the collection to check if it exists
            # This will raise an exception if the collection doesn't exist
            await self._client.load_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            await self._client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dimension,
                primary_field_name="id",
                id_type="str",
                auto_id=False,
                consistency_level="Strong",
                metric_type=self._metric_type,
                max_length=len(str(uuid.uuid4())) + 1,
            )
        finally:
            # Restore original logging level
            milvus_logger.setLevel(original_level)

    async def _ensure_collection_sync(self):
        """Ensure collection exists using sync client in executor."""
        from pymilvus.client.constants import ConsistencyLevel

        loop = asyncio.get_event_loop()

        def _create_collection():
            if not self._client.has_collection(self.collection_name):
                self._client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.embedding_dimension,
                    primary_field_name="id",
                    id_type="str",
                    auto_id=False,
                    consistency_level=ConsistencyLevel.Strong,
                    metric_type=self._metric_type,
                    max_length=len(str(uuid.uuid4())) + 1,
                )

        await loop.run_in_executor(None, _create_collection)

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

        if len(embeddings) == 0:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        texts = list(texts)

        data = [
            {
                "id": ids[idx],
                "vector": embedding,
                "text": texts[idx],
                "metadata": metadata,
            }
            for idx, (embedding, metadata) in enumerate(
                zip(embeddings, metadatas or [{}] * len(embeddings))
            )
        ]

        insertion_size = kwargs.get("insertion_size", 6400000)
        embedding_dim = len(embeddings[0])
        batch_size = insertion_size // embedding_dim

        async def _store_batch(batch_data):
            # Check if we have native async client
            if hasattr(self._client, "upsert") and asyncio.iscoroutinefunction(
                self._client.upsert
            ):
                # Use native async method
                return await self._client.upsert(
                    collection_name=self.collection_name,
                    data=batch_data,
                )
            else:
                # Fallback to executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self._client.upsert(
                        collection_name=self.collection_name,
                        data=batch_data,
                    ),
                )

        # Process batches concurrently
        tasks = []
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            tasks.append(_store_batch(batch))

        if tasks:
            await asyncio.gather(*tasks)

        return ids

    async def similarity_search(
        self,
        query: str,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Perform similarity search asynchronously."""
        await self._ensure_client()

        # Check if we have native async client
        if hasattr(self._client, "search") and asyncio.iscoroutinefunction(
            self._client.search
        ):
            # Use native async method
            ret = await self._client.search(
                collection_name=self.collection_name,
                data=[embedding],
                limit=k,
                search_params={"metric_type": "COSINE", "params": {}},
                output_fields=["text", "metadata"],
            )
        else:
            # Fallback to executor for sync client
            loop = asyncio.get_event_loop()

            def _search():
                # Check if collection exists (only for sync client)
                if hasattr(self._client, "has_collection"):
                    if not self._client.has_collection(self.collection_name):
                        return []

                return self._client.search(
                    collection_name=self.collection_name,
                    data=[embedding],
                    limit=k,
                    search_params={"metric_type": "COSINE", "params": {}},
                    output_fields=["text", "metadata"],
                )

            ret = await loop.run_in_executor(None, _search)

        # Handle empty results
        if not ret or not ret[0]:
            return []

        docs = [
            Document(
                page_content=doc["entity"]["text"],
                metadata={**doc["entity"]["metadata"], "distance": doc["distance"]},
            )
            for doc in ret[0]
        ]
        # Sort by distance (higher is better for cosine similarity)
        docs.sort(key=lambda doc: doc.metadata["distance"], reverse=True)
        return docs

    async def batch_similarity_search(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Perform batch similarity search asynchronously."""
        await self._ensure_client()

        if not queries or not embeddings:
            return []

        # Check if we have native async client
        if hasattr(self._client, "search") and asyncio.iscoroutinefunction(
            self._client.search
        ):
            # Use native async method
            ret = await self._client.search(
                collection_name=self.collection_name,
                data=embeddings,
                limit=k,
                search_params={"metric_type": "COSINE", "params": {}},
                output_fields=["text", "metadata"],
            )
        else:
            # Fallback to executor for sync client
            loop = asyncio.get_event_loop()

            def _batch_search():
                # Check if collection exists (only for sync client)
                if hasattr(self._client, "has_collection"):
                    if not self._client.has_collection(self.collection_name):
                        return [[]]

                return self._client.search(
                    collection_name=self.collection_name,
                    data=embeddings,
                    limit=k,
                    search_params={"metric_type": "COSINE", "params": {}},
                    output_fields=["text", "metadata"],
                )

            ret = await loop.run_in_executor(None, _batch_search)

        # Handle empty results
        if not ret:
            return []

        all_docs = [
            [
                Document(
                    page_content=doc["entity"]["text"],
                    metadata={**doc["entity"]["metadata"], "distance": doc["distance"]},
                )
                for doc in docs
            ]
            for docs in ret
        ]

        # Sort by distance for each query result
        for docs in all_docs:
            docs.sort(key=lambda doc: doc.metadata["distance"], reverse=True)

        return all_docs

    async def similarity_search_with_score(
        self,
        query: str,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with scores asynchronously."""
        await self._ensure_client()

        # Check if we have native async client
        if hasattr(self._client, "search") and asyncio.iscoroutinefunction(
            self._client.search
        ):
            # Use native async method
            ret = await self._client.search(
                collection_name=self.collection_name,
                data=[embedding],
                limit=k,
                search_params={"metric_type": "COSINE", "params": {}},
                output_fields=["text", "metadata"],
            )
        else:
            # Fallback to executor for sync client
            loop = asyncio.get_event_loop()

            def _search_with_score():
                # Check if collection exists (only for sync client)
                if hasattr(self._client, "has_collection"):
                    if not self._client.has_collection(self.collection_name):
                        return []

                return self._client.search(
                    collection_name=self.collection_name,
                    data=[embedding],
                    limit=k,
                    search_params={"metric_type": "COSINE", "params": {}},
                    output_fields=["text", "metadata"],
                )

            ret = await loop.run_in_executor(None, _search_with_score)

        # Handle empty results
        if not ret or not ret[0]:
            return []

        docs_and_scores = [
            (
                Document(
                    page_content=doc["entity"]["text"],
                    metadata=doc["entity"]["metadata"],
                ),
                doc["distance"],
            )
            for doc in ret[0]
        ]
        # Sort by distance (higher is better for cosine similarity)
        docs_and_scores.sort(key=lambda x: x[1], reverse=True)
        return docs_and_scores

    async def drop_collection(self) -> None:
        """Drop collection asynchronously."""
        await self._ensure_client()

        # Check if we have native async client
        if hasattr(self._client, "drop_collection") and asyncio.iscoroutinefunction(
            self._client.drop_collection
        ):
            # Use native async method
            await self._client.drop_collection(collection_name=self.collection_name)
        else:
            # Fallback to executor for sync client
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._client.drop_collection(
                    collection_name=self.collection_name
                ),
            )

    async def collection_exists(self) -> bool:
        """Check if collection exists asynchronously."""
        await self._ensure_client()

        # Check if we have native async client
        if hasattr(self._client, "load_collection") and not hasattr(
            self._client, "has_collection"
        ):
            # This is AsyncMilvusClient - use try/catch approach
            import logging

            # Temporarily suppress logging for expected collection existence check
            milvus_logger = logging.getLogger("pymilvus")
            original_level = milvus_logger.level
            milvus_logger.setLevel(logging.CRITICAL)

            try:
                await self._client.load_collection(self.collection_name)
                return True
            except Exception:
                return False
            finally:
                # Restore original logging level
                milvus_logger.setLevel(original_level)
        else:
            # This is sync client - use executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self._client.has_collection(self.collection_name)
            )

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics asynchronously."""
        await self._ensure_client()

        loop = asyncio.get_event_loop()

        def _get_stats():
            try:
                # Check if collection exists (only for sync client)
                if hasattr(self._client, "has_collection"):
                    if not self._client.has_collection(self.collection_name):
                        return {
                            "count": 0,
                            "name": self.collection_name,
                            "exists": False,
                        }

                stats = self._client.get_collection_stats(
                    collection_name=self.collection_name
                )
                return {"name": self.collection_name, "exists": True, **stats}
            except Exception as e:
                return {"error": str(e), "name": self.collection_name, "exists": False}

        return await loop.run_in_executor(None, _get_stats)

    async def close(self) -> None:
        """Close client asynchronously."""
        if self._client:
            await self._client.close()
        if self._http_client:
            await self._http_client.aclose()
