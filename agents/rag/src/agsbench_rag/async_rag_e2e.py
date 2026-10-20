import asyncio
from fast_service import RequestContext
from .settings import (
    LLMSettings,
    VDBSettings,
    SplitterSettings,
    EmbeddingSettings,
    RetrieverSettings,
    GenerationSettings,
)
from .shared import agsbench_rag_fsm
from .shared import RAGE2EInput, RAGE2EOutput
from .shared import IndexingInput, RAGChattingInput, CollectionClearingInput
from .async_utils import get_logger, get_settings, close_async_vdb_client
from .async_rag_indexing import (
    indexing_async,
    contextual_indexing_async,
    clear_collection_async,
)
from .async_rag_chatting import (
    naive_rag_chatting_async,
    multiquery_rag_chatting_async,
    dynamic_rag_chatting_async,
)


def extract_all_settings(item: RAGE2EInput):
    """Extract all settings from input item."""
    if item.splitter_settings is None:
        splitter_settings = SplitterSettings.from_default_settings()
    else:
        splitter_settings = item.splitter_settings

    if item.vdb_settings is None:
        vdb_settings = VDBSettings.from_default_settings()
    else:
        vdb_settings = item.vdb_settings

    if item.embedding_settings is None:
        embedding_settings = EmbeddingSettings.from_default_settings()
    else:
        embedding_settings = item.embedding_settings

    if item.retriever_settings is None:
        retriever_settings = RetrieverSettings.from_default_settings()
    else:
        retriever_settings = item.retriever_settings

    if item.generation_settings is None:
        generation_settings = GenerationSettings.from_default_settings()
    else:
        generation_settings = item.generation_settings

    return (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    )


@agsbench_rag_fsm.fast_service
async def naive_rag_async(
    item: RAGE2EInput, context: RequestContext = None
) -> RAGE2EOutput:
    """Naive RAG end-to-end asynchronously."""
    get_logger().debug(f"naive_rag_async input {context.request_id}: {item}")

    (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    ) = extract_all_settings(item=item)

    # Index documents
    indexing_item = IndexingInput(
        uri=item.uri,
        type=item.type,
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
        embedding_batch_size=embedding_settings.embedding_batch_size,
    )

    y1 = await indexing_async(item=indexing_item, context=context)
    get_logger().debug(f"Naive RAG indexing output {context.request_id}: {y1}")

    # Chat with RAG
    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )

    y2 = await naive_rag_chatting_async(item=chatting_item, context=context)
    get_logger().debug(f"Naive RAG chatting output {context.request_id}: {y2}")

    # Clear collection if requested
    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
            embedding_model=embedding_settings.embedding_model,
            embedding_device=embedding_settings.embedding_device,
        )
        y3 = await clear_collection_async(item=clearing_item, context=context)
        get_logger().debug(f"Naive RAG clearing output {context.request_id}: {y3}")

    close_async_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    get_logger().debug(f"Naive RAG completed {context.request_id}")
    return RAGE2EOutput(answer=y2.answer)


@agsbench_rag_fsm.fast_service
async def multiquery_rag_async(
    item: RAGE2EInput, context: RequestContext = None
) -> RAGE2EOutput:
    """Multi-query RAG end-to-end asynchronously."""
    get_logger().debug(f"multiquery_rag_async input {context.request_id}: {item}")

    (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    ) = extract_all_settings(item=item)

    # Index documents
    indexing_item = IndexingInput(
        uri=item.uri,
        type=item.type,
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
        embedding_batch_size=embedding_settings.embedding_batch_size,
    )

    y1 = await indexing_async(item=indexing_item, context=context)
    get_logger().debug(f"Multiquery RAG indexing output {context.request_id}: {y1}")

    # Chat with multi-query RAG
    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )

    y2 = await multiquery_rag_chatting_async(item=chatting_item, context=context)
    get_logger().debug(f"Multiquery RAG chatting output {context.request_id}: {y2}")

    # Clear collection if requested
    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
            embedding_model=embedding_settings.embedding_model,
            embedding_device=embedding_settings.embedding_device,
        )
        y3 = await clear_collection_async(item=clearing_item, context=context)
        get_logger().debug(f"Multiquery RAG clearing output {context.request_id}: {y3}")

    close_async_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    get_logger().debug(f"Multiquery RAG completed {context.request_id}")
    return RAGE2EOutput(answer=y2.answer)


@agsbench_rag_fsm.fast_service
async def advanced_rag_async(
    item: RAGE2EInput, context: RequestContext = None
) -> RAGE2EOutput:
    """Advanced RAG end-to-end asynchronously (alias for multiquery_rag_async)."""
    return await multiquery_rag_async(item=item, context=context)


@agsbench_rag_fsm.fast_service
async def dynamic_rag_async(
    item: RAGE2EInput, context: RequestContext = None
) -> RAGE2EOutput:
    """Dynamic RAG end-to-end asynchronously."""
    get_logger().debug(f"dynamic_rag_async input {context.request_id}: {item}")

    (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    ) = extract_all_settings(item=item)

    # Index documents
    indexing_item = IndexingInput(
        uri=item.uri,
        type=item.type,
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
        embedding_batch_size=embedding_settings.embedding_batch_size,
    )

    y1 = await indexing_async(item=indexing_item, context=context)
    get_logger().debug(f"Dynamic RAG indexing output {context.request_id}: {y1}")

    # Chat with dynamic RAG
    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )

    y2 = await dynamic_rag_chatting_async(item=chatting_item, context=context)
    get_logger().debug(f"Dynamic RAG chatting output {context.request_id}: {y2}")

    # Clear collection if requested
    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
            embedding_model=embedding_settings.embedding_model,
            embedding_device=embedding_settings.embedding_device,
        )
        y3 = await clear_collection_async(item=clearing_item, context=context)
        get_logger().debug(f"Dynamic RAG clearing output {context.request_id}: {y3}")

    close_async_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    get_logger().debug(f"Dynamic RAG completed {context.request_id}")
    return RAGE2EOutput(answer=y2.answer)


@agsbench_rag_fsm.fast_service
async def contextual_rag_async(
    item: RAGE2EInput, context: RequestContext = None
) -> RAGE2EOutput:
    """Contextual RAG end-to-end asynchronously."""
    get_logger().debug(f"contextual_rag_async input {context.request_id}: {item}")

    (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    ) = extract_all_settings(item=item)

    # Index documents with contextual chunking
    indexing_item = IndexingInput(
        uri=item.uri,
        type=item.type,
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
        embedding_batch_size=embedding_settings.embedding_batch_size,
    )

    y1 = await contextual_indexing_async(item=indexing_item, context=context)
    get_logger().debug(f"Contextual RAG indexing output {context.request_id}: {y1}")

    # Chat with multi-query RAG (using contextual chunks)
    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )

    y2 = await multiquery_rag_chatting_async(item=chatting_item, context=context)
    get_logger().debug(f"Contextual RAG chatting output {context.request_id}: {y2}")

    # Clear collection if requested
    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
            embedding_model=embedding_settings.embedding_model,
            embedding_device=embedding_settings.embedding_device,
        )
        y3 = await clear_collection_async(item=clearing_item, context=context)
        get_logger().debug(f"Contextual RAG clearing output {context.request_id}: {y3}")

    close_async_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    get_logger().debug(f"Contextual RAG completed {context.request_id}")
    return RAGE2EOutput(answer=y2.answer)


# Advanced async RAG E2E functions
class AdvancedAsyncRAGE2EEngine:
    """Advanced async RAG E2E engine with additional capabilities."""

    def __init__(self):
        pass

    async def parallel_rag_strategies(
        self,
        item: RAGE2EInput,
        strategies: list[str] = ["naive", "multiquery", "dynamic"],
        context: RequestContext = None,
    ) -> dict[str, RAGE2EOutput]:
        """Run multiple RAG strategies in parallel and return all results."""
        tasks = []

        if "naive" in strategies:
            tasks.append(("naive", naive_rag_async(item=item, context=context)))

        if "multiquery" in strategies:
            tasks.append(
                ("multiquery", multiquery_rag_async(item=item, context=context))
            )

        if "advanced" in strategies:
            tasks.append(("advanced", advanced_rag_async(item=item, context=context)))

        if "dynamic" in strategies:
            tasks.append(("dynamic", dynamic_rag_async(item=item, context=context)))

        if "contextual" in strategies:
            tasks.append(
                ("contextual", contextual_rag_async(item=item, context=context))
            )

        # Execute all strategies concurrently
        strategy_names = [name for name, _ in tasks]
        strategy_tasks = [task for _, task in tasks]

        results = await asyncio.gather(*strategy_tasks, return_exceptions=True)

        # Combine results
        strategy_results = {}
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                strategy_results[strategy_names[i]] = result
            else:
                get_logger().warning(f"Strategy {strategy_names[i]} failed: {result}")

        return strategy_results

    async def adaptive_rag_e2e(
        self,
        item: RAGE2EInput,
        context: RequestContext = None,
    ) -> RAGE2EOutput:
        """Adaptively choose the best RAG strategy based on query characteristics."""
        query_length = len(item.query.split())

        # Simple heuristics for strategy selection
        if query_length < 5:
            # Short query - use naive RAG
            return await naive_rag_async(item=item, context=context)
        elif query_length > 20:
            # Complex query - use dynamic RAG
            return await dynamic_rag_async(item=item, context=context)
        else:
            # Medium query - use multiquery RAG
            return await multiquery_rag_async(item=item, context=context)

    async def ensemble_rag_e2e(
        self,
        item: RAGE2EInput,
        strategies: list[str] = ["naive", "multiquery"],
        voting_method: str = "majority",
        context: RequestContext = None,
    ) -> RAGE2EOutput:
        """Run multiple RAG strategies and ensemble their results."""
        strategy_results = await self.parallel_rag_strategies(
            item=item, strategies=strategies, context=context
        )

        if not strategy_results:
            raise RuntimeError("All RAG strategies failed")

        # For now, return the first successful result
        # In practice, you'd implement sophisticated ensembling
        first_result = next(iter(strategy_results.values()))

        # Add metadata about the ensemble
        ensemble_answer = f"Ensemble result from {len(strategy_results)} strategies: {first_result.answer}"

        return RAGE2EOutput(answer=ensemble_answer)


# Global advanced async RAG E2E engine instance
_advanced_async_rag_e2e_engine = None


async def get_advanced_async_rag_e2e_engine() -> AdvancedAsyncRAGE2EEngine:
    """Get advanced async RAG E2E engine instance."""
    global _advanced_async_rag_e2e_engine
    if _advanced_async_rag_e2e_engine is None:
        _advanced_async_rag_e2e_engine = AdvancedAsyncRAGE2EEngine()
    return _advanced_async_rag_e2e_engine
