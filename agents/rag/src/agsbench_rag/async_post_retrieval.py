import asyncio
from fast_service import RequestContext
from langchain_core.documents import Document
from .shared import agsbench_rag_fsm
from .shared import PostRetrievalInput, PostRetrievalOutput, RetrievalOutput
from .async_utils import get_logger, async_run_in_executor
from .settings import get_settings
from .post_retrieval import reciprocal_rank_fusion
from .post_retrieval import get_unique_union as unique_fusion


def all_fusion(documents: list[list[Document]]) -> list[Document]:
    """Simple concatenation of all documents."""
    all_docs = []
    for doc_list in documents:
        all_docs.extend(doc_list)
    return all_docs


class AsyncReranker:
    """Async reranker for document reranking."""

    def __init__(self):
        pass

    async def rerank_documents(
        self,
        query: str,
        documents: list[Document],
        model: str = None,
        engine: str = None,
        top_k: int = None,
    ) -> list[Document]:
        """Rerank documents asynchronously."""
        if not model or not documents:
            return documents

        if engine == "cohere":
            return await self._cohere_rerank(query, documents, model, top_k)
        elif engine and engine.startswith("http"):
            return await self._http_rerank(query, documents, model, engine, top_k)
        else:
            # Fallback to simple scoring
            return await self._simple_rerank(query, documents, top_k)

    async def _cohere_rerank(
        self,
        query: str,
        documents: list[Document],
        model: str,
        top_k: int = None,
    ) -> list[Document]:
        """Rerank using Cohere API asynchronously."""
        try:
            # This would require async Cohere client
            # For now, use sync version in executor
            def _sync_cohere_rerank():
                import cohere
                from langchain.retrievers.document_compressors import CohereRerank

                reranker = CohereRerank(model=model)
                texts = [doc.page_content for doc in documents]

                # Use the reranker
                reranked_docs = reranker.compress_documents(
                    documents=documents, query=query
                )

                if top_k:
                    return reranked_docs[:top_k]
                return reranked_docs

            return await async_run_in_executor(_sync_cohere_rerank)

        except Exception as e:
            get_logger().warning(f"Cohere reranking failed: {e}. Using simple rerank.")
            return await self._simple_rerank(query, documents, top_k)

    async def _http_rerank(
        self,
        query: str,
        documents: list[Document],
        model: str,
        engine: str,
        top_k: int = None,
    ) -> list[Document]:
        """Rerank using HTTP-based reranking service asynchronously."""
        try:
            # Use the cached reranker utility function
            def _sync_http_rerank():
                from .async_utils import get_async_reranker

                reranker = get_async_reranker(model=model, engine=engine)

                # Use the reranker
                reranked_docs = reranker.compress_documents(
                    documents=documents, query=query
                )
                reranked_docs = sorted(
                    reranked_docs,
                    key=lambda x: x.metadata["relevance_score"],
                    reverse=True,
                )
                if top_k:
                    return reranked_docs[:top_k]
                return reranked_docs

            get_logger().debug(f"Using HTTP reranking with engine: {engine}")
            return await async_run_in_executor(_sync_http_rerank)

        except Exception as e:
            get_logger().warning(f"HTTP reranking failed: {e}. Using simple rerank.")
            return await self._simple_rerank(query, documents, top_k)

    async def _simple_rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = None,
    ) -> list[Document]:
        """Simple reranking based on keyword overlap."""
        query_words = set(query.lower().split())

        def score_document(doc):
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            return overlap / len(query_words) if query_words else 0

        # Score and sort documents
        scored_docs = [(score_document(doc), doc) for doc in documents]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        reranked_docs = [doc for _, doc in scored_docs]

        if top_k:
            return reranked_docs[:top_k]
        return reranked_docs


# Global async reranker instance
_async_reranker = None


async def get_async_reranker() -> AsyncReranker:
    """Get async reranker instance."""
    global _async_reranker
    if _async_reranker is None:
        _async_reranker = AsyncReranker()
    return _async_reranker


@agsbench_rag_fsm.fast_service
async def post_retrieval_async(
    item: PostRetrievalInput, context: RequestContext = None
) -> PostRetrievalOutput:
    """Post-process retrieved documents asynchronously."""
    get_logger().debug(f"post_retrieval_async input {context.request_id}: {item}")

    # Handle both single and multiple retrieval outputs
    if isinstance(item.retrieved, RetrievalOutput):
        retrieved_lists = [item.retrieved.docs]
    else:
        retrieved_lists = [output.docs for output in item.retrieved]

    # Apply fusion policy
    if item.fusion_policy == "reciprocal_rank":
        fused_docs = await async_run_in_executor(
            reciprocal_rank_fusion, retrieved_lists
        )
    elif item.fusion_policy == "unique":
        fused_docs = await async_run_in_executor(unique_fusion, retrieved_lists)
    elif item.fusion_policy == "all":
        fused_docs = await async_run_in_executor(all_fusion, retrieved_lists)
    else:
        # Default to reciprocal rank fusion
        fused_docs = await async_run_in_executor(
            reciprocal_rank_fusion, retrieved_lists
        )

    get_logger().debug(f"Fusion result: {len(fused_docs)} documents")

    # Apply reranking if specified
    if item.reranking_args and fused_docs:
        reranker = await get_async_reranker()
        fused_docs = await reranker.rerank_documents(
            query=item.reranking_args.get("query", ""),
            documents=fused_docs,
            model=item.reranking_args.get("model"),
            engine=item.reranking_args.get("engine"),
            top_k=item.reranking_args.get("top_k"),
        )
        get_logger().debug(f"Reranking result: {len(fused_docs)} documents")

    # Apply compression if specified
    if item.compression_args and fused_docs:
        # Placeholder for document compression
        # In practice, you'd implement document compression logic
        get_logger().debug("Document compression not implemented yet")

    # Apply selection if specified
    if item.selection_args and fused_docs:
        # Apply document selection based on criteria
        max_docs = item.selection_args.get("max_docs")
        if max_docs and len(fused_docs) > max_docs:
            fused_docs = fused_docs[:max_docs]
            get_logger().debug(f"Selection result: {len(fused_docs)} documents")

    get_logger().debug(
        f"post_retrieval_async output {context.request_id}: {len(fused_docs)} documents"
    )

    return PostRetrievalOutput(docs=fused_docs)


# Advanced async post-retrieval functions
class AdvancedAsyncPostRetrievalEngine:
    """Advanced async post-retrieval engine with additional capabilities."""

    def __init__(self):
        self.reranker = None

    async def get_reranker(self):
        """Get reranker instance."""
        if self.reranker is None:
            self.reranker = await get_async_reranker()
        return self.reranker

    async def multi_stage_reranking(
        self,
        query: str,
        documents: list[Document],
        stages: list[dict],
        context: RequestContext = None,
    ) -> list[Document]:
        """Apply multiple reranking stages sequentially."""
        current_docs = documents

        for i, stage in enumerate(stages):
            get_logger().debug(f"Applying reranking stage {i+1}: {stage}")

            reranker = await self.get_reranker()
            current_docs = await reranker.rerank_documents(
                query=query,
                documents=current_docs,
                model=stage.get("model"),
                engine=stage.get("engine"),
                top_k=stage.get("top_k"),
            )

            get_logger().debug(f"Stage {i+1} result: {len(current_docs)} documents")

        return current_docs

    async def diversity_reranking(
        self,
        query: str,
        documents: list[Document],
        diversity_threshold: float = 0.7,
        max_docs: int = 10,
        context: RequestContext = None,
    ) -> list[Document]:
        """Rerank documents to maximize diversity while maintaining relevance."""
        if not documents:
            return documents

        # This is a simplified implementation
        # In practice, you'd use embeddings to compute document similarities

        selected_docs = []
        remaining_docs = documents.copy()

        # Always include the top document
        if remaining_docs:
            selected_docs.append(remaining_docs.pop(0))

        # Select diverse documents
        while len(selected_docs) < max_docs and remaining_docs:
            best_doc = None
            best_score = -1

            for doc in remaining_docs:
                # Simple diversity score based on content overlap
                diversity_score = self._compute_diversity_score(doc, selected_docs)

                if diversity_score > best_score:
                    best_score = diversity_score
                    best_doc = doc

            if best_doc and best_score > diversity_threshold:
                selected_docs.append(best_doc)
                remaining_docs.remove(best_doc)
            else:
                break

        return selected_docs

    def _compute_diversity_score(
        self, doc: Document, selected_docs: list[Document]
    ) -> float:
        """Compute diversity score for a document against selected documents."""
        if not selected_docs:
            return 1.0

        doc_words = set(doc.page_content.lower().split())

        min_similarity = 1.0
        for selected_doc in selected_docs:
            selected_words = set(selected_doc.page_content.lower().split())

            if doc_words and selected_words:
                intersection = len(doc_words.intersection(selected_words))
                union = len(doc_words.union(selected_words))
                similarity = intersection / union if union > 0 else 0
                min_similarity = min(min_similarity, similarity)

        return 1.0 - min_similarity  # Higher score for more diverse documents

    async def adaptive_fusion(
        self,
        retrieved_lists: list[list[Document]],
        query: str,
        fusion_strategies: list[str] = ["reciprocal_rank", "unique"],
        context: RequestContext = None,
    ) -> list[Document]:
        """Adaptively choose the best fusion strategy."""
        # Try different fusion strategies
        fusion_results = {}

        for strategy in fusion_strategies:
            if strategy == "reciprocal_rank":
                result = await async_run_in_executor(
                    reciprocal_rank_fusion, retrieved_lists
                )
            elif strategy == "unique":
                result = await async_run_in_executor(unique_fusion, retrieved_lists)
            elif strategy == "all":
                result = await async_run_in_executor(all_fusion, retrieved_lists)
            else:
                continue

            fusion_results[strategy] = result

        # For now, return reciprocal rank fusion result
        # In practice, you'd evaluate which strategy produces the best results
        return fusion_results.get("reciprocal_rank", [])

    async def semantic_deduplication(
        self,
        documents: list[Document],
        similarity_threshold: float = 0.9,
        context: RequestContext = None,
    ) -> list[Document]:
        """Remove semantically similar documents."""
        if not documents:
            return documents

        # This is a simplified implementation using text overlap
        # In practice, you'd use embeddings for semantic similarity

        unique_docs = []

        for doc in documents:
            is_duplicate = False

            for unique_doc in unique_docs:
                similarity = self._compute_text_similarity(doc, unique_doc)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)

        get_logger().debug(
            f"Deduplication: {len(documents)} -> {len(unique_docs)} documents"
        )

        return unique_docs

    def _compute_text_similarity(self, doc1: Document, doc2: Document) -> float:
        """Compute text similarity between two documents."""
        words1 = set(doc1.page_content.lower().split())
        words2 = set(doc2.page_content.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


# Global advanced async post-retrieval engine instance
_advanced_async_post_retrieval_engine = None


async def get_advanced_async_post_retrieval_engine() -> (
    AdvancedAsyncPostRetrievalEngine
):
    """Get advanced async post-retrieval engine instance."""
    global _advanced_async_post_retrieval_engine
    if _advanced_async_post_retrieval_engine is None:
        _advanced_async_post_retrieval_engine = AdvancedAsyncPostRetrievalEngine()
    return _advanced_async_post_retrieval_engine
