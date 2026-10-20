import asyncio
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import (
    EmbeddingDocsInput,
    EmbeddingDocsOutput,
    EmbeddingQueryInput,
    EmbeddingQueryOutput,
)
from .settings import get_settings
from .async_utils import get_embedding_fn, async_run_in_executor, AsyncBatchProcessor


class AsyncEmbeddingEngine:
    """Async embedding engine for processing documents and queries."""

    def __init__(self):
        self.batch_processor = AsyncBatchProcessor()

    async def embed_documents_batch(
        self,
        docs: list[str],
        embedding_model: str,
        embedding_device: str,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed documents in batches asynchronously."""
        if not docs:
            return []

        fn = get_embedding_fn(
            embedding_model=embedding_model, embedding_device=embedding_device
        )

        if batch_size <= 0:
            # Process all at once
            return await self._embed_with_native_async(fn, docs, "embed_documents")

        self.batch_processor.batch_size = batch_size

        # Process in batches
        async def process_batch(batch_docs):
            return await self._embed_with_native_async(
                fn, batch_docs, "embed_documents"
            )

        return await self.batch_processor.process_batches(docs, process_batch)

    async def embed_query_single(
        self, query: str, embedding_model: str, embedding_device: str
    ) -> list[float]:
        """Embed a single query asynchronously."""
        fn = get_embedding_fn(
            embedding_model=embedding_model, embedding_device=embedding_device
        )
        return await self._embed_with_native_async(fn, query, "embed_query")

    async def embed_queries_batch(
        self, queries: list[str], embedding_model: str, embedding_device: str
    ) -> list[list[float]]:
        """Embed multiple queries asynchronously."""
        if not queries:
            return []

        fn = get_embedding_fn(
            embedding_model=embedding_model, embedding_device=embedding_device
        )

        # Use embed_documents for batch processing of queries
        return await self._embed_with_native_async(fn, queries, "embed_documents")

    async def _embed_with_native_async(
        self, embedding_fn, input_data, method_name: str
    ):
        """Try to use native async embedding methods, fallback to executor."""
        method = getattr(embedding_fn, method_name)

        # Check if the embedding function has async methods
        async_method_name = f"a{method_name}"
        if hasattr(embedding_fn, async_method_name):
            async_method = getattr(embedding_fn, async_method_name)
            if asyncio.iscoroutinefunction(async_method):
                try:
                    return await async_method(input_data)
                except Exception as e:
                    # If async method fails, fallback to sync
                    pass

        # Fallback to sync method in executor
        return await async_run_in_executor(method, input_data)


# Global async embedding engine instance
_async_embedding_engine = None


async def get_async_embedding_engine() -> AsyncEmbeddingEngine:
    """Get async embedding engine instance."""
    global _async_embedding_engine
    if _async_embedding_engine is None:
        _async_embedding_engine = AsyncEmbeddingEngine()
    return _async_embedding_engine


@agsbench_rag_fsm.fast_service
async def embed_documents_async(
    item: EmbeddingDocsInput, context: RequestContext = None
) -> EmbeddingDocsOutput:
    """Embed documents asynchronously."""
    engine = await get_async_embedding_engine()

    # Use settings from global config
    item.embedding_batch_size = get_settings().embedding_batch_size
    embedding_model = get_settings().embedding_model
    embedding_device = get_settings().embedding_device

    docs = [doc.page_content for doc in item.docs]
    embeddings = await engine.embed_documents_batch(
        docs=docs,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        batch_size=item.embedding_batch_size,
    )

    return EmbeddingDocsOutput(embeddings=embeddings)


@agsbench_rag_fsm.fast_service
async def embed_query_async(
    item: EmbeddingQueryInput, context: RequestContext = None
) -> EmbeddingQueryOutput:
    """Embed a query asynchronously."""
    engine = await get_async_embedding_engine()

    # Use settings from global config
    embedding_model = get_settings().embedding_model
    embedding_device = get_settings().embedding_device

    embedding = await engine.embed_query_single(
        query=item.query,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
    )

    return EmbeddingQueryOutput(embedding=embedding)


@agsbench_rag_fsm.fast_service
async def batch_embed_query_async(
    items: list[EmbeddingQueryInput], context: RequestContext = None
) -> list[EmbeddingQueryOutput]:
    """Embed multiple queries asynchronously."""
    if not items:
        return []

    engine = await get_async_embedding_engine()

    # Use settings from global config
    embedding_model = get_settings().embedding_model
    embedding_device = get_settings().embedding_device

    queries = [item.query for item in items]
    embeddings = await engine.embed_queries_batch(
        queries=queries,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
    )

    return [EmbeddingQueryOutput(embedding=embedding) for embedding in embeddings]


# Concurrent embedding functions for high-throughput request
async def embed_documents_concurrent(
    docs_batches: list[list[str]],
    embedding_model: str,
    embedding_device: str,
    max_concurrency: int = 5,
) -> list[list[list[float]]]:
    """Embed multiple document batches concurrently."""
    engine = await get_async_embedding_engine()

    async def embed_batch(docs):
        return await engine.embed_documents_batch(
            docs=docs,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            batch_size=0,  # Process each batch as a whole
        )

    # Limit concurrency to avoid overwhelming the embedding service
    semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_with_semaphore(docs):
        async with semaphore:
            return await embed_batch(docs)

    tasks = [embed_with_semaphore(docs) for docs in docs_batches]
    return await asyncio.gather(*tasks)


async def embed_queries_concurrent(
    query_batches: list[list[str]],
    embedding_model: str,
    embedding_device: str,
    max_concurrency: int = 5,
) -> list[list[list[float]]]:
    """Embed multiple query batches concurrently."""
    engine = await get_async_embedding_engine()

    async def embed_batch(queries):
        return await engine.embed_queries_batch(
            queries=queries,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
        )

    # Limit concurrency to avoid overwhelming the embedding service
    semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_with_semaphore(queries):
        async with semaphore:
            return await embed_batch(queries)

    tasks = [embed_with_semaphore(queries) for queries in query_batches]
    return await asyncio.gather(*tasks)


# Adaptive batch sizing for optimal performance
class AdaptiveEmbeddingProcessor:
    """Adaptive embedding processor that adjusts batch sizes based on performance."""

    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 128):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 8
        self.performance_history = []

    async def embed_documents_adaptive(
        self, docs: list[str], embedding_model: str, embedding_device: str
    ) -> list[list[float]]:
        """Embed documents with adaptive batch sizing."""
        if not docs:
            return []

        engine = await get_async_embedding_engine()

        # Start timing
        import time

        start_time = time.time()

        embeddings = await engine.embed_documents_batch(
            docs=docs,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            batch_size=self.current_batch_size,
        )

        # Calculate performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        throughput = len(docs) / processing_time if processing_time > 0 else 0

        # Update performance history
        self.performance_history.append(
            {
                "batch_size": self.current_batch_size,
                "throughput": throughput,
                "processing_time": processing_time,
                "doc_count": len(docs),
            }
        )

        # Adjust batch size based on performance
        self._adjust_batch_size()

        return embeddings

    def _adjust_batch_size(self):
        """Adjust batch size based on performance history."""
        if len(self.performance_history) < 2:
            return

        recent_performance = self.performance_history[-2:]

        # If throughput improved, continue in the same direction
        if recent_performance[1]["throughput"] > recent_performance[0]["throughput"]:
            if (
                recent_performance[1]["batch_size"]
                > recent_performance[0]["batch_size"]
            ):
                # Increase was beneficial, continue increasing
                self.current_batch_size = min(
                    self.current_batch_size * 1.2, self.max_batch_size
                )
            else:
                # Decrease was beneficial, continue decreasing
                self.current_batch_size = max(
                    self.current_batch_size * 0.8, self.min_batch_size
                )
        else:
            # Performance degraded, reverse direction
            if (
                recent_performance[1]["batch_size"]
                > recent_performance[0]["batch_size"]
            ):
                # Increase was detrimental, decrease
                self.current_batch_size = max(
                    self.current_batch_size * 0.8, self.min_batch_size
                )
            else:
                # Decrease was detrimental, increase
                self.current_batch_size = min(
                    self.current_batch_size * 1.2, self.max_batch_size
                )

        self.current_batch_size = int(self.current_batch_size)
