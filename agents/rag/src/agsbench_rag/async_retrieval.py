import asyncio
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import RetrievalInput, RetrievalOutput
from .shared import EmbeddingQueryInput, VDBSearchingInput
from .async_embedding import embed_query_async, batch_embed_query_async
from .async_vstore import vdb_search_async, batch_vdb_search_async
from .settings import get_settings
from .async_utils import get_logger, async_gather_with_concurrency


@agsbench_rag_fsm.fast_service
async def retrieval_async(
    item: RetrievalInput, context: RequestContext = None
) -> RetrievalOutput:
    """Retrieve documents asynchronously."""
    get_logger().debug(f"Retrieving documents for query: {item.query}")

    # Embed the query
    embedding_output = await embed_query_async(
        item=EmbeddingQueryInput(
            query=item.query,
            embedding_model=item.embedding_model,
            embedding_device=item.embedding_device,
        ),
        context=context,
    )

    # Search vector database
    search_output = await vdb_search_async(
        item=VDBSearchingInput(
            vdb_type=item.vdb_type,
            vdb_uri=item.vdb_uri,
            collection_name=item.collection_name,
            query=item.query,
            embedding=embedding_output.embedding,
            k=item.search_args.get("k", 4) if item.search_args else 4,
        ),
        context=context,
    )

    return RetrievalOutput(docs=search_output.docs)


@agsbench_rag_fsm.fast_service
async def batch_retrieval_async(
    items: list[RetrievalInput], context: RequestContext = None
) -> list[RetrievalOutput]:
    """Retrieve documents for multiple queries asynchronously."""
    if not items:
        return []

    parallel_retrieval = get_settings().parallel_retrieval
    parallel_search = get_settings().parallel_search

    for item in items:
        get_logger().debug(f"retrieval input {context.request_id}: {item}")

    if parallel_retrieval:
        if parallel_search:
            # Parallel embedding and search
            embedding_outputs = await batch_embed_query_async(
                items=[
                    EmbeddingQueryInput(
                        query=item.query,
                        embedding_model=item.embedding_model,
                        embedding_device=item.embedding_device,
                    )
                    for item in items
                ],
                context=context,
            )

            search_outputs = await batch_vdb_search_async(
                items=[
                    VDBSearchingInput(
                        vdb_type=item.vdb_type,
                        vdb_uri=item.vdb_uri,
                        collection_name=item.collection_name,
                        query=item.query,
                        embedding=embedding_output.embedding,
                        k=item.search_args.get("k", 4) if item.search_args else 4,
                    )
                    for item, embedding_output in zip(items, embedding_outputs)
                ],
                context=context,
            )

            ret = [RetrievalOutput(docs=output.docs) for output in search_outputs]
        else:
            # Parallel individual retrievals
            tasks = [retrieval_async(item=item, context=context) for item in items]
            ret = await async_gather_with_concurrency(tasks, max_concurrency=10)
    else:
        # Sequential retrieval
        ret = []
        for item in items:
            retrieval_output = await retrieval_async(item=item, context=context)
            ret.append(retrieval_output)

    for rout in ret:
        get_logger().debug(f"retrieval output {context.request_id}: {rout}")

    return ret


# Advanced async retrieval functions
class AsyncRetrievalEngine:
    """Advanced async retrieval engine with additional capabilities."""

    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency

    async def multi_collection_retrieval(
        self,
        query: str,
        collections: list[str],
        embedding_model: str,
        embedding_device: str,
        vdb_type: str,
        vdb_uri: str,
        k: int = 4,
        context: RequestContext = None,
    ) -> dict[str, RetrievalOutput]:
        """Retrieve from multiple collections concurrently."""
        # Create retrieval items for each collection
        retrieval_items = [
            RetrievalInput(
                query=query,
                collection_name=collection,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
                vdb_type=vdb_type,
                vdb_uri=vdb_uri,
                search_args={"k": k},
            )
            for collection in collections
        ]

        # Retrieve from all collections concurrently
        results = await batch_retrieval_async(items=retrieval_items, context=context)

        # Return results mapped by collection name
        return {collection: result for collection, result in zip(collections, results)}

    async def adaptive_retrieval(
        self,
        query: str,
        primary_collection: str,
        fallback_collections: list[str],
        embedding_model: str,
        embedding_device: str,
        vdb_type: str,
        vdb_uri: str,
        min_results: int = 3,
        max_k: int = 20,
        context: RequestContext = None,
    ) -> RetrievalOutput:
        """Adaptive retrieval that adjusts k and tries fallback collections."""
        current_k = 4

        # Try primary collection first
        while current_k <= max_k:
            try:
                result = await retrieval_async(
                    item=RetrievalInput(
                        query=query,
                        collection_name=primary_collection,
                        embedding_model=embedding_model,
                        embedding_device=embedding_device,
                        vdb_type=vdb_type,
                        vdb_uri=vdb_uri,
                        search_args={"k": current_k},
                    ),
                    context=context,
                )

                if len(result.docs) >= min_results:
                    return result

                current_k *= 2  # Double k and try again
            except Exception as e:
                get_logger().warning(f"Primary collection retrieval failed: {e}")
                break

        # Try fallback collections
        for fallback_collection in fallback_collections:
            try:
                result = await retrieval_async(
                    item=RetrievalInput(
                        query=query,
                        collection_name=fallback_collection,
                        embedding_model=embedding_model,
                        embedding_device=embedding_device,
                        vdb_type=vdb_type,
                        vdb_uri=vdb_uri,
                        search_args={"k": max_k},
                    ),
                    context=context,
                )

                if len(result.docs) >= min_results:
                    return result
            except Exception as e:
                get_logger().warning(
                    f"Fallback collection {fallback_collection} failed: {e}"
                )
                continue

        # Return empty result if all attempts failed
        return RetrievalOutput(docs=[])

    async def semantic_routing_retrieval(
        self,
        query: str,
        collection_routing_map: dict[str, list[str]],  # topic -> collections
        embedding_model: str,
        embedding_device: str,
        vdb_type: str,
        vdb_uri: str,
        k: int = 4,
        context: RequestContext = None,
    ) -> dict[str, RetrievalOutput]:
        """Route queries to appropriate collections based on semantic similarity."""
        # This is a simplified version - in practice, you'd use a classifier
        # For now, we'll search all collections and return results grouped by topic

        all_collections = []
        collection_to_topic = {}

        for topic, collections in collection_routing_map.items():
            all_collections.extend(collections)
            for collection in collections:
                collection_to_topic[collection] = topic

        # Retrieve from all collections
        multi_results = await self.multi_collection_retrieval(
            query=query,
            collections=all_collections,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            vdb_type=vdb_type,
            vdb_uri=vdb_uri,
            k=k,
            context=context,
        )

        # Group results by topic
        topic_results = {}
        for collection, result in multi_results.items():
            topic = collection_to_topic[collection]
            if topic not in topic_results:
                topic_results[topic] = RetrievalOutput(docs=[])
            topic_results[topic].docs.extend(result.docs)

        return topic_results

    async def hybrid_retrieval(
        self,
        query: str,
        dense_collections: list[str],
        sparse_collections: list[str],
        embedding_model: str,
        embedding_device: str,
        vdb_type: str,
        vdb_uri: str,
        k: int = 4,
        alpha: float = 0.7,  # Weight for dense retrieval
        context: RequestContext = None,
    ) -> RetrievalOutput:
        """Hybrid retrieval combining dense and sparse methods."""
        # Retrieve from dense collections
        dense_results = await self.multi_collection_retrieval(
            query=query,
            collections=dense_collections,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            vdb_type=vdb_type,
            vdb_uri=vdb_uri,
            k=k,
            context=context,
        )

        # Retrieve from sparse collections (if any)
        sparse_results = await self.multi_collection_retrieval(
            query=query,
            collections=sparse_collections,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            vdb_type=vdb_type,
            vdb_uri=vdb_uri,
            k=k,
            context=context,
        )

        # Combine and rerank results
        all_docs = []

        # Add dense results with weight
        for result in dense_results.values():
            for i, doc in enumerate(result.docs):
                score = alpha * (1.0 / (i + 1))  # Simple ranking score
                doc.metadata["hybrid_score"] = score
                doc.metadata["source_type"] = "dense"
                all_docs.append(doc)

        # Add sparse results with weight
        for result in sparse_results.values():
            for i, doc in enumerate(result.docs):
                score = (1 - alpha) * (1.0 / (i + 1))  # Simple ranking score
                doc.metadata["hybrid_score"] = score
                doc.metadata["source_type"] = "sparse"
                all_docs.append(doc)

        # Sort by hybrid score and return top k
        all_docs.sort(key=lambda x: x.metadata.get("hybrid_score", 0), reverse=True)

        return RetrievalOutput(docs=all_docs[:k])


# Global async retrieval engine instance
_async_retrieval_engine = None


async def get_async_retrieval_engine() -> AsyncRetrievalEngine:
    """Get async retrieval engine instance."""
    global _async_retrieval_engine
    if _async_retrieval_engine is None:
        _async_retrieval_engine = AsyncRetrievalEngine()
    return _async_retrieval_engine
