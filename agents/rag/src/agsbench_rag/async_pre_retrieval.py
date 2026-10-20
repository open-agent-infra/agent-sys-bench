import asyncio
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import QueryTranslationInput, QueryTranslationOutput
from .async_generation import get_async_generation_engine
from .async_utils import get_logger


@agsbench_rag_fsm.fast_service
async def query_translation_async(
    item: QueryTranslationInput, context: RequestContext = None
) -> QueryTranslationOutput:
    """Translate/expand queries asynchronously."""
    get_logger().debug(f"query_translation_async input {context.request_id}: {item}")

    if item.method == "MultiQuery":
        return await multi_query_generation_async(item.query, context)
    elif item.method == "HyDE":
        return await hyde_query_generation_async(item.query, context)
    elif item.method == "StepBack":
        return await step_back_query_generation_async(item.query, context)
    elif item.method == "Decomposition":
        return await decomposition_query_generation_async(item.query, context)
    else:
        raise NotImplementedError(
            f"Unsupported query translation method: {item.method}"
        )


async def multi_query_generation_async(
    query: str, context: RequestContext = None
) -> QueryTranslationOutput:
    """Generate multiple query variations asynchronously."""
    engine = await get_async_generation_engine()

    response = await engine.generate_with_template(
        template_name="multi_query",
        template_vars={"question": query},
        context=context,
    )

    # Parse the response to extract individual queries
    queries = [q.strip() for q in response.split("\n") if q.strip()]

    # Ensure we have at least the original query
    if not queries:
        queries = [query]
    elif query not in queries:
        queries.insert(0, query)

    # Limit to reasonable number of queries
    queries = queries[:5]

    get_logger().debug(
        f"multi_query_generation_async output {context.request_id}: {queries}"
    )

    return QueryTranslationOutput(
        queries=queries,
        qtypes=["original" if q == query else "variation" for q in queries],
    )


async def hyde_query_generation_async(
    query: str, context: RequestContext = None
) -> QueryTranslationOutput:
    """Generate hypothetical document for HyDE asynchronously."""
    engine = await get_async_generation_engine()

    hypothetical_doc = await engine.generate_with_template(
        template_name="hyde",
        template_vars={"question": query},
        context=context,
    )

    get_logger().debug(
        f"hyde_query_generation_async output {context.request_id}: {hypothetical_doc}"
    )

    return QueryTranslationOutput(
        queries=[hypothetical_doc], qtypes=["hypothetical_document"]
    )


async def step_back_query_generation_async(
    query: str, context: RequestContext = None
) -> QueryTranslationOutput:
    """Generate step-back query for broader context asynchronously."""
    engine = await get_async_generation_engine()

    step_back_query = await engine.generate_with_template(
        template_name="query_step_back",
        template_vars={"question": query},
        context=context,
    )

    # Clean up the response
    step_back_query = step_back_query.strip()
    if step_back_query.startswith("Stepback:"):
        step_back_query = step_back_query[9:].strip()

    get_logger().debug(
        f"step_back_query_generation_async output {context.request_id}: {step_back_query}"
    )

    return QueryTranslationOutput(
        queries=[query, step_back_query], qtypes=["original", "step_back"]
    )


async def decomposition_query_generation_async(
    query: str, context: RequestContext = None
) -> QueryTranslationOutput:
    """Decompose complex query into sub-questions asynchronously."""
    engine = await get_async_generation_engine()

    response = await engine.generate_with_template(
        template_name="query_decomposition",
        template_vars={"question": query},
        context=context,
    )

    # Parse the response to extract sub-questions
    sub_questions = []
    for line in response.split("\n"):
        line = line.strip()
        if line and not line.startswith("Sub-questions:"):
            # Remove numbering if present
            if line[0].isdigit() and "." in line[:5]:
                line = line.split(".", 1)[1].strip()
            if line.startswith("- "):
                line = line[2:].strip()
            if line:
                sub_questions.append(line)

    # Ensure we have the original query
    if query not in sub_questions:
        sub_questions.insert(0, query)

    # Limit to reasonable number of sub-questions
    sub_questions = sub_questions[:5]

    get_logger().debug(
        f"decomposition_query_generation_async output {context.request_id}: {sub_questions}"
    )

    return QueryTranslationOutput(
        queries=sub_questions,
        qtypes=["original" if q == query else "sub_question" for q in sub_questions],
    )


# Advanced async pre-retrieval functions
class AdvancedAsyncPreRetrievalEngine:
    """Advanced async pre-retrieval engine with additional capabilities."""

    def __init__(self):
        pass

    async def adaptive_query_expansion(
        self,
        query: str,
        query_complexity_threshold: int = 50,
        context: RequestContext = None,
    ) -> QueryTranslationOutput:
        """Adaptively choose query expansion strategy based on query characteristics."""
        query_length = len(query.split())

        if query_length < 5:
            # Short query - use multi-query expansion
            return await multi_query_generation_async(query, context)
        elif query_length > query_complexity_threshold:
            # Complex query - use decomposition
            return await decomposition_query_generation_async(query, context)
        else:
            # Medium query - use step-back
            return await step_back_query_generation_async(query, context)

    async def multi_strategy_query_expansion(
        self,
        query: str,
        strategies: list[str] = ["MultiQuery", "StepBack"],
        context: RequestContext = None,
    ) -> dict[str, QueryTranslationOutput]:
        """Apply multiple query expansion strategies concurrently."""
        tasks = []

        if "MultiQuery" in strategies:
            tasks.append(("MultiQuery", multi_query_generation_async(query, context)))

        if "HyDE" in strategies:
            tasks.append(("HyDE", hyde_query_generation_async(query, context)))

        if "StepBack" in strategies:
            tasks.append(("StepBack", step_back_query_generation_async(query, context)))

        if "Decomposition" in strategies:
            tasks.append(
                ("Decomposition", decomposition_query_generation_async(query, context))
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

    async def contextual_query_enhancement(
        self,
        query: str,
        domain_context: str = None,
        user_context: str = None,
        context: RequestContext = None,
    ) -> QueryTranslationOutput:
        """Enhance query with additional context information."""
        engine = await get_async_generation_engine()

        # Build context information string
        context_info = ""
        if domain_context:
            context_info += f"\nDomain context: {domain_context}"
        if user_context:
            context_info += f"\nUser context: {user_context}"

        enhanced_query = await engine.generate_with_template(
            template_name="contextual_query_enhancement",
            template_vars={"query": query, "context_info": context_info},
            context=context,
        )

        enhanced_query = enhanced_query.strip()
        if enhanced_query.startswith("Enhanced query:"):
            enhanced_query = enhanced_query[15:].strip()

        return QueryTranslationOutput(
            queries=[query, enhanced_query], qtypes=["original", "enhanced"]
        )

    async def semantic_query_clustering(
        self,
        queries: list[str],
        similarity_threshold: float = 0.8,
        context: RequestContext = None,
    ) -> dict[str, list[str]]:
        """Cluster semantically similar queries (placeholder implementation)."""
        # This would require embedding the queries and computing similarities
        # For now, return a simple grouping

        clusters = {"cluster_0": queries}
        return clusters


# Global advanced async pre-retrieval engine instance
_advanced_async_pre_retrieval_engine = None


async def get_advanced_async_pre_retrieval_engine() -> AdvancedAsyncPreRetrievalEngine:
    """Get advanced async pre-retrieval engine instance."""
    global _advanced_async_pre_retrieval_engine
    if _advanced_async_pre_retrieval_engine is None:
        _advanced_async_pre_retrieval_engine = AdvancedAsyncPreRetrievalEngine()
    return _advanced_async_pre_retrieval_engine
