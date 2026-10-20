import asyncio
from typing import List
from fast_service import RequestContext
from .shared import agsbench_rag_fsm, Document
from .shared import DocSplittingInput, DocSplittingOutput
from .async_utils import get_splitter, get_logger, async_run_in_executor
from .settings import get_settings


class AsyncChunkingEngine:
    """Async chunking engine for document splitting."""

    def __init__(self):
        pass

    async def execute(
        self,
        splitter_model: str,
        chunk_size: int,
        chunk_overlap: int,
        docs: List[Document],
        request_id: str = None,
    ) -> List[Document]:
        """Execute document chunking asynchronously."""
        get_logger().debug(f"AsyncChunkingEngine.execute start: {request_id}")

        # Get text splitter (this is cached and fast)
        text_splitter = get_splitter(
            splitter_model=splitter_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Run the splitting operation in executor since it's CPU-intensive
        chunks = await async_run_in_executor(text_splitter.split_documents, docs)

        get_logger().debug(f"AsyncChunkingEngine.execute end: {request_id}")
        return chunks


class AsyncChunkingClient:
    """Async chunking client."""

    def __init__(self, engine_type: str = "local"):
        self.engine_type = engine_type
        self.engine = AsyncChunkingEngine()

    async def execute(
        self, item: DocSplittingInput, context: RequestContext = None
    ) -> list[Document]:
        """Execute chunking asynchronously."""
        return await self.engine.execute(
            splitter_model=item.splitter_model,
            chunk_size=item.chunk_size,
            chunk_overlap=item.chunk_overlap,
            docs=item.docs,
            request_id=context.request_id if context else None,
        )


# Global async chunking client instance
_async_chunking_client = None


async def get_async_chunking_client() -> AsyncChunkingClient:
    """Get async chunking client instance."""
    global _async_chunking_client
    if _async_chunking_client is None:
        engine_type = get_settings().chunking_engine or "local"
        _async_chunking_client = AsyncChunkingClient(engine_type=engine_type)
    return _async_chunking_client


@agsbench_rag_fsm.fast_service
async def split_docs_async(
    item: DocSplittingInput, context: RequestContext = None
) -> DocSplittingOutput:
    """Split documents asynchronously."""
    client = await get_async_chunking_client()

    # Use settings from global config
    item.splitter_model = get_settings().splitter_model
    item.chunk_size = get_settings().chunk_size
    item.chunk_overlap = get_settings().chunk_overlap

    splits = await client.execute(item=item, context=context)
    return DocSplittingOutput(docs=splits)


# Advanced async chunking functions
class AdvancedAsyncChunkingEngine:
    """Advanced async chunking engine with additional capabilities."""

    def __init__(self):
        self.base_engine = AsyncChunkingEngine()

    async def adaptive_chunking(
        self,
        docs: List[Document],
        base_chunk_size: int = 1000,
        base_overlap: int = 200,
        splitter_model: str = "recursive-character",
        context: RequestContext = None,
    ) -> List[Document]:
        """Adaptive chunking that adjusts parameters based on document characteristics."""
        if not docs:
            return []

        # Analyze document characteristics
        total_length = sum(len(doc.page_content) for doc in docs)
        avg_doc_length = total_length / len(docs)

        # Adjust chunk size based on average document length
        if avg_doc_length < 500:
            # Small documents - use smaller chunks
            chunk_size = min(base_chunk_size, int(avg_doc_length * 0.8))
            overlap = min(base_overlap, chunk_size // 5)
        elif avg_doc_length > 5000:
            # Large documents - use larger chunks
            chunk_size = int(base_chunk_size * 1.5)
            overlap = int(base_overlap * 1.2)
        else:
            # Medium documents - use default settings
            chunk_size = base_chunk_size
            overlap = base_overlap

        get_logger().debug(
            f"Adaptive chunking: avg_doc_length={avg_doc_length}, "
            f"chunk_size={chunk_size}, overlap={overlap}"
        )

        return await self.base_engine.execute(
            splitter_model=splitter_model,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            docs=docs,
            request_id=context.request_id if context else None,
        )

    async def semantic_chunking(
        self,
        docs: List[Document],
        similarity_threshold: float = 0.8,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        context: RequestContext = None,
    ) -> List[Document]:
        """Semantic chunking based on content similarity (placeholder implementation)."""
        # This is a simplified implementation
        # In practice, you'd use sentence embeddings to determine semantic boundaries

        get_logger().debug("Performing semantic chunking (simplified implementation)")

        # For now, fall back to adaptive chunking
        return await self.adaptive_chunking(
            docs=docs,
            base_chunk_size=max_chunk_size,
            base_overlap=min_chunk_size // 4,
            context=context,
        )

    async def hierarchical_chunking(
        self,
        docs: List[Document],
        chunk_sizes: List[int] = [500, 1000, 2000],
        overlap_ratio: float = 0.2,
        splitter_model: str = "recursive-character",
        context: RequestContext = None,
    ) -> dict[int, List[Document]]:
        """Create hierarchical chunks at multiple granularities."""
        results = {}

        # Create chunks at different sizes concurrently
        async def create_chunks_for_size(size):
            overlap = int(size * overlap_ratio)
            return await self.base_engine.execute(
                splitter_model=splitter_model,
                chunk_size=size,
                chunk_overlap=overlap,
                docs=docs,
                request_id=context.request_id if context else None,
            )

        tasks = [create_chunks_for_size(size) for size in chunk_sizes]
        chunk_results = await asyncio.gather(*tasks)

        for size, chunks in zip(chunk_sizes, chunk_results):
            results[size] = chunks
            # Add hierarchy metadata
            for i, chunk in enumerate(chunks):
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.update(
                    {
                        "chunk_level": size,
                        "chunk_index": i,
                        "hierarchy_id": f"{size}_{i}",
                    }
                )

        return results

    async def parallel_chunking(
        self,
        doc_batches: List[List[Document]],
        splitter_model: str = "recursive-character",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_concurrency: int = 5,
        context: RequestContext = None,
    ) -> List[List[Document]]:
        """Process multiple document batches in parallel."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def chunk_batch_with_semaphore(docs):
            async with semaphore:
                return await self.base_engine.execute(
                    splitter_model=splitter_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    docs=docs,
                    request_id=context.request_id if context else None,
                )

        tasks = [chunk_batch_with_semaphore(batch) for batch in doc_batches]
        return await asyncio.gather(*tasks)

    async def content_aware_chunking(
        self,
        docs: List[Document],
        content_types: dict[str, dict] = None,
        default_chunk_size: int = 1000,
        default_overlap: int = 200,
        context: RequestContext = None,
    ) -> List[Document]:
        """Chunk documents based on their content type."""
        if content_types is None:
            content_types = {
                "code": {
                    "chunk_size": 500,
                    "overlap": 50,
                    "splitter": "recursive-character-python",
                },
                "text": {
                    "chunk_size": 1000,
                    "overlap": 200,
                    "splitter": "recursive-character",
                },
                "markdown": {
                    "chunk_size": 800,
                    "overlap": 160,
                    "splitter": "recursive-character",
                },
            }

        # Group documents by content type
        doc_groups = {}
        for doc in docs:
            content_type = (
                doc.metadata.get("content_type", "text") if doc.metadata else "text"
            )
            if content_type not in doc_groups:
                doc_groups[content_type] = []
            doc_groups[content_type].append(doc)

        # Process each group with appropriate settings
        all_chunks = []

        async def process_group(content_type, group_docs):
            config = content_types.get(
                content_type,
                {
                    "chunk_size": default_chunk_size,
                    "overlap": default_overlap,
                    "splitter": "recursive-character",
                },
            )

            return await self.base_engine.execute(
                splitter_model=config["splitter"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["overlap"],
                docs=group_docs,
                request_id=context.request_id if context else None,
            )

        tasks = [
            process_group(content_type, group_docs)
            for content_type, group_docs in doc_groups.items()
        ]

        group_results = await asyncio.gather(*tasks)

        # Combine all chunks
        for chunks in group_results:
            all_chunks.extend(chunks)

        return all_chunks


# Global advanced async chunking engine instance
_advanced_async_chunking_engine = None


async def get_advanced_async_chunking_engine() -> AdvancedAsyncChunkingEngine:
    """Get advanced async chunking engine instance."""
    global _advanced_async_chunking_engine
    if _advanced_async_chunking_engine is None:
        _advanced_async_chunking_engine = AdvancedAsyncChunkingEngine()
    return _advanced_async_chunking_engine
