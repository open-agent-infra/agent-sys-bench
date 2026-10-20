import asyncio
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import VDBStoringInput, VDBStoringOutput
from .shared import VDBSearchingInput, VDBSearchingOutput
from .settings import get_settings
from .async_utils import get_logger, async_run_in_executor, get_async_vdb_client
from vectordb.async_base import create_async_client


@agsbench_rag_fsm.fast_service
async def vdb_store_async(
    item: VDBStoringInput, context: RequestContext = None
) -> VDBStoringOutput:
    """Store documents in vector database asynchronously."""
    item.vdb_type = get_settings().vdb_type
    item.vdb_uri = get_settings().vdb_uri
    insertion_size = get_settings().insertion_size

    embeddings = item.embeddings
    if len(embeddings) == 0:
        get_logger().warning(
            f"{context.request_id} No embeddings to store, skip {item}"
        )
        return VDBStoringOutput(count=0)

    dimension = len(embeddings[0])
    vdb_client = await get_async_vdb_client(
        vdb_type=item.vdb_type,
        vdb_uri=item.vdb_uri,
        collection_name=item.collection_name,
        embedding_dimension=dimension,
    )

    if item.metadatas is None:
        try:
            item.metadatas = [{"source": doc.metadata["source"]} for doc in item.docs]
        except (AttributeError, KeyError):
            item.metadatas = [{} for _ in item.docs]

    ids = await vdb_client.store(
        texts=[doc.page_content for doc in item.docs],
        embeddings=embeddings,
        metadatas=item.metadatas,
        ids=item.ids,
        insertion_size=insertion_size,
    )

    return VDBStoringOutput(count=len(ids))


@agsbench_rag_fsm.fast_service
async def vdb_search_async(
    item: VDBSearchingInput, context: RequestContext = None
) -> VDBSearchingOutput:
    """Search vector database asynchronously."""
    item.vdb_type = get_settings().vdb_type
    item.vdb_uri = get_settings().vdb_uri
    search_args = get_settings().search_args
    item.k = search_args.get("k", 4) if search_args else 4

    dimension = len(item.embedding)
    vdb_client = await get_async_vdb_client(
        vdb_type=item.vdb_type,
        vdb_uri=item.vdb_uri,
        collection_name=item.collection_name,
        embedding_dimension=dimension,
    )

    documents = await vdb_client.similarity_search(
        query=item.query, embedding=item.embedding, k=item.k
    )

    return VDBSearchingOutput(docs=documents)


@agsbench_rag_fsm.fast_service
async def batch_vdb_search_async(
    items: list[VDBSearchingInput], context: RequestContext = None
) -> list[VDBSearchingOutput]:
    """Perform batch vector database search asynchronously."""
    if len(items) == 0:
        return []

    item = items[0]
    item.vdb_type = get_settings().vdb_type
    item.vdb_uri = get_settings().vdb_uri
    search_args = get_settings().search_args
    item.k = search_args.get("k", 4) if search_args else 4

    dimension = len(item.embedding)
    vdb_client = await get_async_vdb_client(
        vdb_type=item.vdb_type,
        vdb_uri=item.vdb_uri,
        collection_name=item.collection_name,
        embedding_dimension=dimension,
    )

    all_documents = await vdb_client.batch_similarity_search(
        queries=[item.query for item in items],
        embeddings=[item.embedding for item in items],
        k=item.k,
    )

    return [VDBSearchingOutput(docs=docs) for docs in all_documents]


# Advanced async vector store operations
class AsyncVectorStoreManager:
    """Advanced async vector store manager with additional capabilities."""

    def __init__(self):
        self._clients = {}

    async def get_client(
        self,
        vdb_type: str,
        vdb_uri: str,
        collection_name: str,
        embedding_dimension: int = None,
    ):
        """Get or create async VDB client with caching."""
        client_key = f"{vdb_type}:{vdb_uri}:{collection_name}"

        if client_key not in self._clients:
            self._clients[client_key] = await get_async_vdb_client(
                vdb_type=vdb_type,
                vdb_uri=vdb_uri,
                collection_name=collection_name,
                embedding_dimension=embedding_dimension,
            )

        return self._clients[client_key]

    async def store_with_retry(
        self,
        item: VDBStoringInput,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> VDBStoringOutput:
        """Store documents with retry logic."""
        for attempt in range(max_retries):
            try:
                return await vdb_store_async(item)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                get_logger().warning(
                    f"Store attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        raise RuntimeError("All store attempts failed")

    async def search_with_fallback(
        self,
        item: VDBSearchingInput,
        fallback_collections: list[str] = None,
    ) -> VDBSearchingOutput:
        """Search with fallback to other collections."""
        try:
            return await vdb_search_async(item)
        except Exception as e:
            if not fallback_collections:
                raise e

            get_logger().warning(
                f"Search in {item.collection_name} failed: {e}. Trying fallback collections..."
            )

            for fallback_collection in fallback_collections:
                try:
                    fallback_item = item.model_copy()
                    fallback_item.collection_name = fallback_collection
                    return await vdb_search_async(fallback_item)
                except Exception as fallback_e:
                    get_logger().warning(
                        f"Fallback search in {fallback_collection} failed: {fallback_e}"
                    )
                    continue

            raise RuntimeError("All search attempts failed")

    async def parallel_store_multiple_collections(
        self,
        items: list[VDBStoringInput],
        max_concurrency: int = 5,
    ) -> list[VDBStoringOutput]:
        """Store documents in multiple collections concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def store_with_semaphore(item):
            async with semaphore:
                return await vdb_store_async(item)

        tasks = [store_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    async def parallel_search_multiple_collections(
        self,
        items: list[VDBSearchingInput],
        max_concurrency: int = 5,
    ) -> list[VDBSearchingOutput]:
        """Search multiple collections concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def search_with_semaphore(item):
            async with semaphore:
                return await vdb_search_async(item)

        tasks = [search_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    async def health_check(
        self, vdb_type: str, vdb_uri: str, collection_name: str
    ) -> dict:
        """Perform health check on vector database."""
        try:
            client = await self.get_client(vdb_type, vdb_uri, collection_name)

            # Check if collection exists
            exists = await client.collection_exists()
            if not exists:
                return {"status": "error", "message": "Collection does not exist"}

            # Get collection stats
            stats = await client.get_collection_stats()

            return {
                "status": "healthy",
                "collection_exists": exists,
                "stats": stats,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    async def cleanup_collections(
        self, collections_to_drop: list[tuple[str, str, str]]
    ):
        """Clean up multiple collections asynchronously."""

        async def drop_collection(vdb_type, vdb_uri, collection_name):
            try:
                client = await self.get_client(vdb_type, vdb_uri, collection_name)
                await client.drop_collection()
                return {"collection": collection_name, "status": "dropped"}
            except Exception as e:
                return {
                    "collection": collection_name,
                    "status": "error",
                    "error": str(e),
                }

        tasks = [
            drop_collection(vdb_type, vdb_uri, collection_name)
            for vdb_type, vdb_uri, collection_name in collections_to_drop
        ]

        return await asyncio.gather(*tasks)


# Global async vector store manager instance
_async_vstore_manager = None


async def get_async_vstore_manager() -> AsyncVectorStoreManager:
    """Get async vector store manager instance."""
    global _async_vstore_manager
    if _async_vstore_manager is None:
        _async_vstore_manager = AsyncVectorStoreManager()
    return _async_vstore_manager
