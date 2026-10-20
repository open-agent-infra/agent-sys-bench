import asyncio
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import RAGChattingInput, RAGChattingOutput
from .shared import (
    QueryTranslationInput,
    RetrievalInput,
    PostRetrievalInput,
    GenerationInput,
)
from .shared import RetrievalJudgingInput
from .async_pre_retrieval import query_translation_async
from .async_retrieval import retrieval_async, batch_retrieval_async
from .async_post_retrieval import post_retrieval_async
from .async_generation import generation_async
from .async_judging import retrieval_judger_async
from .async_utils import get_logger
from .settings import get_settings


@agsbench_rag_fsm.fast_service
async def naive_rag_chatting_async(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    """Naive RAG chatting asynchronously."""
    get_logger().debug(f"naive_rag_chatting_async input {context.request_id}: {item}")

    # Retrieve relevant documents
    retrieval_output = await retrieval_async(
        item=RetrievalInput(
            query=item.query,
            vdb_type=item.vdb_settings.vdb_type,
            vdb_uri=item.vdb_settings.vdb_uri,
            collection_name=item.vdb_settings.collection_name,
            embedding_model=item.embedding_settings.embedding_model,
            embedding_device=item.embedding_settings.embedding_device,
            search_args=item.retriever_settings.search_args,
        ),
        context=context,
    )
    get_logger().debug(
        f"naive_rag_chatting_async retrieval output {context.request_id}: {retrieval_output}"
    )

    # Prepare context from retrieved documents
    query_context = "\n\n".join([doc.page_content for doc in retrieval_output.docs])

    # Generate answer
    generation_output = await generation_async(
        item=GenerationInput(
            query=item.query,
            context=query_context,
            method=item.generation_settings.generation_method,
            method_args=item.generation_settings.generation_method_args,
        ),
        context=context,
    )
    get_logger().debug(
        f"naive_rag_chatting_async generation output {context.request_id}: {generation_output}"
    )

    return RAGChattingOutput(answer=generation_output.answer)


@agsbench_rag_fsm.fast_service
async def multiquery_rag_chatting_async(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    """Multi-query RAG chatting asynchronously."""
    get_logger().debug(
        f"multiquery_rag_chatting_async input {context.request_id}: {item}"
    )

    # Generate multiple query variations
    qtrans_output = await query_translation_async(
        item=QueryTranslationInput(query=item.query, method="MultiQuery"),
        context=context,
    )
    get_logger().debug(
        f"multiquery_rag_chatting_async query_translation output {context.request_id}: {qtrans_output}"
    )

    queries = qtrans_output.queries

    # Retrieve documents for all queries in parallel
    retrieval_inputs = [
        RetrievalInput(
            query=query,
            vdb_type=item.vdb_settings.vdb_type,
            vdb_uri=item.vdb_settings.vdb_uri,
            collection_name=item.vdb_settings.collection_name,
            embedding_model=item.embedding_settings.embedding_model,
            embedding_device=item.embedding_settings.embedding_device,
            search_args=item.retriever_settings.search_args,
        )
        for query in queries
    ]

    retrieval_outputs = await batch_retrieval_async(
        items=retrieval_inputs, context=context
    )
    get_logger().debug(
        f"multiquery_rag_chatting_async batch retrieval output {context.request_id}: {retrieval_outputs}"
    )

    # Post-process retrieved documents (fusion, reranking, etc.)
    post_retrieval_output = await post_retrieval_async(
        item=PostRetrievalInput(
            retrieved=retrieval_outputs,
            fusion_policy=get_settings().fusion_policy,
            reranking_args=(
                {
                    "model": get_settings().reranker_model,
                    "engine": get_settings().reranker_engine,
                    "query": item.query,
                }
                if get_settings().reranker_model
                else None
            ),
        ),
        context=context,
    )
    get_logger().debug(
        f"multiquery_rag_chatting_async post_retrieval output {context.request_id}: {post_retrieval_output}"
    )

    # Prepare context from processed documents
    query_context = "\n\n".join(
        [doc.page_content for doc in post_retrieval_output.docs]
    )

    # Generate answer
    generation_output = await generation_async(
        item=GenerationInput(
            query=item.query,
            context=query_context,
            method=item.generation_settings.generation_method,
            method_args=item.generation_settings.generation_method_args,
        ),
        context=context,
    )
    get_logger().debug(
        f"multiquery_rag_chatting_async generation output {context.request_id}: {generation_output}"
    )

    return RAGChattingOutput(answer=generation_output.answer)


@agsbench_rag_fsm.fast_service
async def dynamic_rag_chatting_async(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    """Dynamic RAG chatting with retrieval judging asynchronously."""
    get_logger().debug(f"dynamic_rag_chatting_async input {context.request_id}: {item}")

    # Check if retrieval is needed
    retrieval_judger_output = await retrieval_judger_async(
        item=RetrievalJudgingInput(question=item.query),
        context=context,
    )
    get_logger().debug(
        f"dynamic_rag_chatting_async retrieval_judger output {context.request_id}: {retrieval_judger_output}"
    )

    if not retrieval_judger_output.judgement:
        # Direct generation without retrieval
        generation_output = await generation_async(
            item=GenerationInput(
                query=item.query,
                context="",
                method="direct",
            ),
            context=context,
        )
        get_logger().debug(
            f"dynamic_rag_chatting_async direct generation output {context.request_id}: {generation_output}"
        )
        return RAGChattingOutput(answer=generation_output.answer)

    # Proceed with multi-query RAG if retrieval is needed
    return await multiquery_rag_chatting_async(item=item, context=context)


@agsbench_rag_fsm.fast_service
async def HyDE_rag_chatting_async(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    """HyDE RAG chatting asynchronously."""
    get_logger().debug(f"HyDE_rag_chatting_async input {context.request_id}: {item}")

    # Generate hypothetical document
    qtrans_output = await query_translation_async(
        item=QueryTranslationInput(query=item.query, method="HyDE"),
        context=context,
    )
    get_logger().debug(
        f"HyDE_rag_chatting_async query_translation output {context.request_id}: {qtrans_output}"
    )

    queries = qtrans_output.queries
    generated_docs_for_retrieval = queries[
        0
    ]  # HyDE generates one hypothetical document

    # Use the hypothetical document for retrieval
    new_item = item.model_copy(
        update={"query": generated_docs_for_retrieval}, deep=True
    )
    return await naive_rag_chatting_async(item=new_item, context=context)


# Advanced async RAG chatting functions
class AdvancedAsyncRAGChattingEngine:
    """Advanced async RAG chatting engine with additional capabilities."""

    def __init__(self):
        pass

    async def adaptive_rag_chatting(
        self,
        item: RAGChattingInput,
        strategies: list[str] = ["naive", "multiquery", "dynamic"],
        context: RequestContext = None,
    ) -> RAGChattingOutput:
        """Adaptive RAG that tries multiple strategies and selects the best result."""
        get_logger().debug(f"adaptive_rag_chatting with strategies: {strategies}")

        # Try different strategies concurrently
        tasks = []

        if "naive" in strategies:
            tasks.append(
                ("naive", naive_rag_chatting_async(item=item, context=context))
            )

        if "multiquery" in strategies:
            tasks.append(
                (
                    "multiquery",
                    multiquery_rag_chatting_async(item=item, context=context),
                )
            )

        if "dynamic" in strategies:
            tasks.append(
                ("dynamic", dynamic_rag_chatting_async(item=item, context=context))
            )

        if "hyde" in strategies:
            tasks.append(("hyde", HyDE_rag_chatting_async(item=item, context=context)))

        # Execute all strategies concurrently
        strategy_names = [name for name, _ in tasks]
        strategy_tasks = [task for _, task in tasks]

        try:
            results = await asyncio.gather(*strategy_tasks, return_exceptions=True)

            # Filter successful results
            successful_results = []
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    successful_results.append((strategy_names[i], result))
                else:
                    get_logger().warning(
                        f"Strategy {strategy_names[i]} failed: {result}"
                    )

            if not successful_results:
                raise RuntimeError("All RAG strategies failed")

            # For now, return the first successful result
            # In practice, you'd implement a quality scoring mechanism
            best_strategy, best_result = successful_results[0]
            get_logger().debug(f"Selected strategy: {best_strategy}")

            return best_result

        except Exception as e:
            get_logger().error(f"Adaptive RAG failed: {e}")
            # Fallback to naive RAG
            return await naive_rag_chatting_async(item=item, context=context)

    async def conversational_rag_chatting(
        self,
        item: RAGChattingInput,
        conversation_history: list[dict],
        context: RequestContext = None,
    ) -> RAGChattingOutput:
        """RAG chatting with conversation history awareness."""
        # Enhance query with conversation context
        if conversation_history:
            # Create a context-aware query
            recent_context = "\n".join(
                [
                    f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
                    for turn in conversation_history[-3:]  # Use last 3 turns
                ]
            )

            enhanced_query = f"Conversation context:\n{recent_context}\n\nCurrent question: {item.query}"

            # Create new item with enhanced query
            enhanced_item = item.model_copy(update={"query": enhanced_query}, deep=True)

            return await multiquery_rag_chatting_async(
                item=enhanced_item, context=context
            )
        else:
            # No conversation history, use standard RAG
            return await naive_rag_chatting_async(item=item, context=context)

    async def multi_domain_rag_chatting(
        self,
        item: RAGChattingInput,
        domain_collections: dict[str, str],  # domain -> collection_name
        context: RequestContext = None,
    ) -> RAGChattingOutput:
        """RAG chatting across multiple domain-specific collections."""
        # Retrieve from all domain collections concurrently
        domain_results = {}

        async def retrieve_from_domain(domain, collection_name):
            domain_item = item.model_copy(deep=True)
            domain_item.vdb_settings.collection_name = collection_name

            try:
                return await naive_rag_chatting_async(item=domain_item, context=context)
            except Exception as e:
                get_logger().warning(f"Domain {domain} retrieval failed: {e}")
                return None

        tasks = [
            retrieve_from_domain(domain, collection)
            for domain, collection in domain_collections.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        successful_answers = []
        for i, result in enumerate(results):
            if isinstance(result, RAGChattingOutput):
                domain = list(domain_collections.keys())[i]
                successful_answers.append(f"From {domain}: {result.answer}")

        if not successful_answers:
            return RAGChattingOutput(
                answer="I couldn't find relevant information in any domain."
            )

        # Combine answers from different domains
        combined_answer = "\n\n".join(successful_answers)

        return RAGChattingOutput(answer=combined_answer)


# Global advanced async RAG chatting engine instance
_advanced_async_rag_chatting_engine = None


async def get_advanced_async_rag_chatting_engine() -> AdvancedAsyncRAGChattingEngine:
    """Get advanced async RAG chatting engine instance."""
    global _advanced_async_rag_chatting_engine
    if _advanced_async_rag_chatting_engine is None:
        _advanced_async_rag_chatting_engine = AdvancedAsyncRAGChattingEngine()
    return _advanced_async_rag_chatting_engine
