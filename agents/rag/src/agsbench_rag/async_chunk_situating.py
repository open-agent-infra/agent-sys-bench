import asyncio
from fast_service import RequestContext
from langchain_core.documents import Document
from .shared import agsbench_rag_fsm
from .shared import SituatingChunksInput, SituatingChunksOutput
from .async_utils import get_logger, async_run_in_executor, AsyncBatchProcessor
from .settings import get_settings


class AsyncChunkSituatingEngine:
    """Async chunk situating engine for adding contextual information to chunks."""

    def __init__(self):
        self.batch_processor = AsyncBatchProcessor()

    async def situate_chunk(
        self,
        chunk: Document,
        source_doc: Document,
        context: RequestContext = None,
    ) -> Document:
        """Add contextual information to a single chunk using LLM."""
        from .async_generation import get_async_generation_engine

        engine = await get_async_generation_engine()

        try:
            # Generate contextual information using the same template as sync version
            contextual_info = await engine.generate_with_template(
                template_name="contextual_retrieval",
                template_vars={
                    "doc_content": source_doc.page_content,
                    "chunk_content": chunk.page_content,
                },
                context=context,
            )

            # Create enhanced chunk with LLM-generated context (same format as sync version)
            enhanced_content = f"{chunk.page_content}\n\n{contextual_info}"
            enhanced_chunk = Document(
                page_content=enhanced_content,
                metadata={
                    **(chunk.metadata or {}),
                    "has_context": True,
                    "contextual_info": contextual_info,
                    "doc_id": (
                        source_doc.metadata.get("doc_id")
                        if source_doc.metadata
                        else None
                    ),
                },
            )

            get_logger().debug(
                f"situate_chunk output {context.request_id}: {contextual_info}"
            )
            return enhanced_chunk

        except Exception as e:
            get_logger().error(f"situate_chunk Error invoking LLM: {e}")
            # Fall back to returning original chunk
            return Document(
                page_content=chunk.page_content,
                metadata={
                    **(chunk.metadata or {}),
                    "has_context": False,
                    "error": str(e),
                    "doc_id": (
                        source_doc.metadata.get("doc_id")
                        if source_doc.metadata
                        else None
                    ),
                },
            )

    async def situate_chunks_batch(
        self,
        chunks: list[Document],
        docs: list[Document],
        batch_size: int = 10,
        context: RequestContext = None,
    ) -> list[Document]:
        """Situate multiple chunks in batches."""
        if not chunks:
            return []

        # Create a mapping from doc_id to document
        doc_map = {}
        for doc in docs:
            if doc.metadata and "doc_id" in doc.metadata:
                doc_map[doc.metadata["doc_id"]] = doc

        async def process_chunk_batch(chunk_batch):
            """Process a batch of chunks."""
            enhanced_chunks = []

            for chunk in chunk_batch:
                # Find the source document for this chunk
                source_doc = None
                if chunk.metadata and "doc_id" in chunk.metadata:
                    source_doc = doc_map.get(chunk.metadata["doc_id"])

                if source_doc is None:
                    # If no source document found, use the first document as fallback
                    source_doc = docs[0] if docs else chunk

                enhanced_chunk = await self.situate_chunk(
                    chunk=chunk,
                    source_doc=source_doc,
                    context=context,
                )
                enhanced_chunks.append(enhanced_chunk)

            return enhanced_chunks

        # Process chunks in batches
        return await self.batch_processor.process_batches(chunks, process_chunk_batch)


# Global async chunk situating engine instance
_async_chunk_situating_engine = None


async def get_async_chunk_situating_engine() -> AsyncChunkSituatingEngine:
    """Get async chunk situating engine instance."""
    global _async_chunk_situating_engine
    if _async_chunk_situating_engine is None:
        _async_chunk_situating_engine = AsyncChunkSituatingEngine()
    return _async_chunk_situating_engine


@agsbench_rag_fsm.fast_service
async def situate_chunks_async(
    item: SituatingChunksInput, context: RequestContext = None
) -> SituatingChunksOutput:
    """Situate chunks with contextual information asynchronously."""
    engine = await get_async_chunk_situating_engine()

    get_logger().debug(
        f"Situating {len(item.chunks)} chunks with context from {len(item.docs)} documents"
    )

    enhanced_chunks = await engine.situate_chunks_batch(
        chunks=item.chunks,
        docs=item.docs,
        batch_size=get_settings().situating_batch_size or 10,
        context=context,
    )

    get_logger().debug(
        f"Enhanced {len(enhanced_chunks)} chunks with contextual information"
    )

    return SituatingChunksOutput(chunks=enhanced_chunks)


# Advanced async chunk situating functions
class AdvancedAsyncChunkSituatingEngine:
    """Advanced async chunk situating engine with LLM-based contextual enhancement."""

    def __init__(self):
        self.base_engine = None

    async def get_base_engine(self):
        """Get base chunk situating engine."""
        if self.base_engine is None:
            self.base_engine = await get_async_chunk_situating_engine()
        return self.base_engine

    async def llm_based_situating(
        self,
        chunk: Document,
        source_doc: Document,
        context: RequestContext = None,
    ) -> Document:
        """Use LLM to generate contextual information for a chunk."""
        from .async_generation import get_async_generation_engine

        engine = await get_async_generation_engine()

        try:
            # Generate contextual information using the same template as sync version
            contextual_info = await engine.generate_with_template(
                template_name="contextual_retrieval",
                template_vars={
                    "doc_content": source_doc.page_content,
                    "chunk_content": chunk.page_content,
                },
                context=context,
            )

            # Create enhanced chunk with LLM-generated context (same format as sync version)
            enhanced_content = f"{chunk.page_content}\n\n{contextual_info}"
            enhanced_chunk = Document(
                page_content=enhanced_content,
                metadata={
                    **(chunk.metadata or {}),
                    "has_llm_context": True,
                    "contextual_info": contextual_info,
                    "doc_id": (
                        source_doc.metadata.get("doc_id")
                        if source_doc.metadata
                        else None
                    ),
                },
            )

            return enhanced_chunk

        except Exception as e:
            get_logger().warning(
                f"LLM-based situating failed: {e}. Falling back to basic situating."
            )
            # Fall back to basic situating
            base_engine = await self.get_base_engine()
            return await base_engine.situate_chunk(chunk, source_doc, context)

    async def parallel_llm_situating(
        self,
        chunks: list[Document],
        docs: list[Document],
        max_concurrency: int = 5,
        context: RequestContext = None,
    ) -> list[Document]:
        """Situate chunks using LLM with controlled concurrency."""
        if not chunks:
            return []

        # Create a mapping from doc_id to document
        doc_map = {}
        for doc in docs:
            if doc.metadata and "doc_id" in doc.metadata:
                doc_map[doc.metadata["doc_id"]] = doc

        semaphore = asyncio.Semaphore(max_concurrency)

        async def situate_with_semaphore(chunk):
            async with semaphore:
                # Find the source document for this chunk
                source_doc = None
                if chunk.metadata and "doc_id" in chunk.metadata:
                    source_doc = doc_map.get(chunk.metadata["doc_id"])

                if source_doc is None:
                    # If no source document found, use the first document as fallback
                    source_doc = docs[0] if docs else chunk

                return await self.llm_based_situating(
                    chunk=chunk,
                    source_doc=source_doc,
                    context=context,
                )

        tasks = [situate_with_semaphore(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    async def adaptive_situating(
        self,
        chunks: list[Document],
        docs: list[Document],
        use_llm_threshold: int = 100,  # Use LLM for chunks shorter than this
        context: RequestContext = None,
    ) -> list[Document]:
        """Adaptively choose between basic and LLM-based situating."""
        if not chunks:
            return []

        # Separate chunks based on length
        short_chunks = []
        long_chunks = []

        for chunk in chunks:
            if len(chunk.page_content) < use_llm_threshold:
                short_chunks.append(chunk)
            else:
                long_chunks.append(chunk)

        # Process short chunks with LLM (more detailed context needed)
        # Process long chunks with basic method (already have enough context)

        tasks = []

        if short_chunks:
            tasks.append(
                self.parallel_llm_situating(
                    chunks=short_chunks,
                    docs=docs,
                    context=context,
                )
            )

        if long_chunks:
            base_engine = await self.get_base_engine()
            tasks.append(
                base_engine.situate_chunks_batch(
                    chunks=long_chunks,
                    docs=docs,
                    context=context,
                )
            )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)

        # Combine results
        all_enhanced_chunks = []
        for result in results:
            all_enhanced_chunks.extend(result)

        return all_enhanced_chunks

    async def hierarchical_situating(
        self,
        chunks: list[Document],
        docs: list[Document],
        hierarchy_levels: list[str] = ["document", "section", "paragraph"],
        context: RequestContext = None,
    ) -> list[Document]:
        """Situate chunks with hierarchical context at multiple levels."""
        if not chunks:
            return []

        enhanced_chunks = []

        for chunk in chunks:
            # Build hierarchical context
            hierarchical_context = {}

            # Document level context
            source_doc = None
            if chunk.metadata and "doc_id" in chunk.metadata:
                for doc in docs:
                    if (
                        doc.metadata
                        and doc.metadata.get("doc_id") == chunk.metadata["doc_id"]
                    ):
                        source_doc = doc
                        break

            if source_doc:
                hierarchical_context["document"] = source_doc.page_content[:200]

            # Section level context (simplified - would need better parsing)
            if "section" in hierarchy_levels:
                # This is a placeholder - in practice, you'd parse document structure
                hierarchical_context["section"] = chunk.page_content[:100]

            # Paragraph level context
            if "paragraph" in hierarchy_levels:
                hierarchical_context["paragraph"] = chunk.page_content

            # Create enhanced chunk with hierarchical context
            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata={
                    **(chunk.metadata or {}),
                    "hierarchical_context": hierarchical_context,
                    "context_levels": hierarchy_levels,
                },
            )

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks


# Global advanced async chunk situating engine instance
_advanced_async_chunk_situating_engine = None


async def get_advanced_async_chunk_situating_engine() -> (
    AdvancedAsyncChunkSituatingEngine
):
    """Get advanced async chunk situating engine instance."""
    global _advanced_async_chunk_situating_engine
    if _advanced_async_chunk_situating_engine is None:
        _advanced_async_chunk_situating_engine = AdvancedAsyncChunkSituatingEngine()
    return _advanced_async_chunk_situating_engine
