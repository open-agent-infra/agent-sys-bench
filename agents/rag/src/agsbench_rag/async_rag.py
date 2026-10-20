"""
Async RAG Module

This module provides asynchronous versions of all RAG functionality,
maintaining the same interfaces as the synchronous versions while
improving performance through concurrent execution.

Key Features:
- Async document loading with concurrent file I/O
- Async embedding with batch processing
- Async vector database operations
- Async retrieval with parallel search
- Async generation with LLM calls
- Async end-to-end RAG pipelines

Usage:
    from agsbench_rag.async_rag import (
        indexing_async,
        naive_rag_chatting_async,
        naive_rag_async,
        warm_up_async
    )

    # Async indexing
    result = await indexing_async(indexing_input, context)

    # Async RAG chatting
    response = await naive_rag_chatting_async(chatting_input, context)

    # Async end-to-end RAG
    output = await naive_rag_async(e2e_input, context)
"""

# Import all async functions and classes
from .async_utils import (
    async_run_in_executor,
    async_gather_with_concurrency,
    get_async_lock,
    warm_up_async,
    AsyncFileHandler,
    AsyncHTTPClient,
    AsyncBatchProcessor,
)

from .async_loading import (
    load_docs_async,
    AsyncLoadingEngine,
    AsyncLoadingClient,
    get_async_loading_client,
)

from .async_chunking import (
    split_docs_async,
    AsyncChunkingEngine,
    AsyncChunkingClient,
    get_async_chunking_client,
    AdvancedAsyncChunkingEngine,
    get_advanced_async_chunking_engine,
)

from .async_chunk_situating import (
    situate_chunks_async,
    AsyncChunkSituatingEngine,
    get_async_chunk_situating_engine,
    AdvancedAsyncChunkSituatingEngine,
    get_advanced_async_chunk_situating_engine,
)

from .async_embedding import (
    embed_documents_async,
    embed_query_async,
    batch_embed_query_async,
    AsyncEmbeddingEngine,
    get_async_embedding_engine,
    embed_documents_concurrent,
    embed_queries_concurrent,
    AdaptiveEmbeddingProcessor,
)

from .async_vstore import (
    vdb_store_async,
    vdb_search_async,
    batch_vdb_search_async,
    get_async_vdb_client,
    AsyncVectorStoreManager,
    get_async_vstore_manager,
)

from .async_retrieval import (
    retrieval_async,
    batch_retrieval_async,
    AsyncRetrievalEngine,
    get_async_retrieval_engine,
)

from .async_pre_retrieval import (
    query_translation_async,
    multi_query_generation_async,
    hyde_query_generation_async,
    step_back_query_generation_async,
    decomposition_query_generation_async,
    AdvancedAsyncPreRetrievalEngine,
    get_advanced_async_pre_retrieval_engine,
)

from .async_post_retrieval import (
    post_retrieval_async,
    AsyncReranker,
    get_async_reranker,
    AdvancedAsyncPostRetrievalEngine,
    get_advanced_async_post_retrieval_engine,
)

from .async_generation import (
    generation_async,
    normal_generation_async,
    direct_generation_async,
    decomposition_recursive_generation_async,
    decomposition_individual_final_generation_async,
    step_back_generation_async,
    HyDE_generation_async,
    AsyncGenerationEngine,
    get_async_generation_engine,
    AdvancedAsyncGenerationEngine,
    get_advanced_async_generation_engine,
)

from .async_judging import (
    retrieval_judger_async,
    doc_relevance_judger_async,
    hallucination_judger_async,
    answer_judger_async,
    AdvancedAsyncJudgingEngine,
    get_advanced_async_judging_engine,
)

from .async_rag_indexing import (
    indexing_async,
    contextual_indexing_async,
    clear_collection_async,
    AsyncIndexingEngine,
    get_async_indexing_engine,
)

from .async_rag_chatting import (
    naive_rag_chatting_async,
    multiquery_rag_chatting_async,
    dynamic_rag_chatting_async,
    HyDE_rag_chatting_async,
    AdvancedAsyncRAGChattingEngine,
    get_advanced_async_rag_chatting_engine,
)

from .async_rag_e2e import (
    naive_rag_async,
    multiquery_rag_async,
    advanced_rag_async,
    dynamic_rag_async,
    contextual_rag_async,
    AdvancedAsyncRAGE2EEngine,
    get_advanced_async_rag_e2e_engine,
)

# Import vector database async clients
from vectordb.async_base import AsyncBaseVectorDBClient, create_async_client
from vectordb.async_chromadb import AsyncChromaVectorDBClient
from vectordb.async_milvusdb import AsyncMilvusVectorDBClient

# Import shared models (same as sync version)
from .shared import (
    DocLoadingInput,
    DocLoadingOutput,
    DocSplittingInput,
    DocSplittingOutput,
    SituatingChunksInput,
    SituatingChunksOutput,
    EmbeddingDocsInput,
    EmbeddingDocsOutput,
    EmbeddingQueryInput,
    EmbeddingQueryOutput,
    VDBStoringInput,
    VDBStoringOutput,
    VDBSearchingInput,
    VDBSearchingOutput,
    IndexingInput,
    IndexingOutput,
    CollectionClearingInput,
    QueryTranslationInput,
    QueryTranslationOutput,
    RetrievalInput,
    RetrievalOutput,
    PostRetrievalInput,
    PostRetrievalOutput,
    GenerationInput,
    GenerationOutput,
    RAGChattingInput,
    RAGChattingOutput,
    JudgingOutput,
    RetrievalJudgingInput,
    DocRelevanceJudgingInput,
    HallucinationJudgingInput,
    AnswerJudgingInput,
    RAGE2EInput,
    RAGE2EOutput,
)

# Import settings (same as sync version)
from .settings import (
    init_settings,
    get_settings,
    LLMSettings,
    SplitterSettings,
    VDBSettings,
    EmbeddingSettings,
    RetrieverSettings,
    GenerationSettings,
    LoggingSettings,
)


__all__ = [
    # Utility functions
    "async_run_in_executor",
    "async_gather_with_concurrency",
    "get_async_lock",
    "warm_up_async",
    "AsyncFileHandler",
    "AsyncHTTPClient",
    "AsyncBatchProcessor",
    # Document loading
    "load_docs_async",
    "AsyncLoadingEngine",
    "AsyncLoadingClient",
    "get_async_loading_client",
    # Document chunking
    "split_docs_async",
    "AsyncChunkingEngine",
    "AsyncChunkingClient",
    "get_async_chunking_client",
    "AdvancedAsyncChunkingEngine",
    "get_advanced_async_chunking_engine",
    # Chunk situating
    "situate_chunks_async",
    "AsyncChunkSituatingEngine",
    "get_async_chunk_situating_engine",
    "AdvancedAsyncChunkSituatingEngine",
    "get_advanced_async_chunk_situating_engine",
    # Embedding
    "embed_documents_async",
    "embed_query_async",
    "batch_embed_query_async",
    "AsyncEmbeddingEngine",
    "get_async_embedding_engine",
    "embed_documents_concurrent",
    "embed_queries_concurrent",
    "AdaptiveEmbeddingProcessor",
    # Vector store
    "vdb_store_async",
    "vdb_search_async",
    "batch_vdb_search_async",
    "get_async_vdb_client",
    "AsyncVectorStoreManager",
    "get_async_vstore_manager",
    # Retrieval
    "retrieval_async",
    "batch_retrieval_async",
    "AsyncRetrievalEngine",
    "get_async_retrieval_engine",
    # Pre-retrieval
    "query_translation_async",
    "multi_query_generation_async",
    "hyde_query_generation_async",
    "step_back_query_generation_async",
    "decomposition_query_generation_async",
    "AdvancedAsyncPreRetrievalEngine",
    "get_advanced_async_pre_retrieval_engine",
    # Post-retrieval
    "post_retrieval_async",
    "AsyncReranker",
    "get_async_reranker",
    "AdvancedAsyncPostRetrievalEngine",
    "get_advanced_async_post_retrieval_engine",
    # Generation
    "generation_async",
    "normal_generation_async",
    "direct_generation_async",
    "decomposition_recursive_generation_async",
    "decomposition_individual_final_generation_async",
    "step_back_generation_async",
    "HyDE_generation_async",
    "AsyncGenerationEngine",
    "get_async_generation_engine",
    "AdvancedAsyncGenerationEngine",
    "get_advanced_async_generation_engine",
    # Judging
    "retrieval_judger_async",
    "doc_relevance_judger_async",
    "hallucination_judger_async",
    "answer_judger_async",
    "AdvancedAsyncJudgingEngine",
    "get_advanced_async_judging_engine",
    # RAG indexing
    "indexing_async",
    "contextual_indexing_async",
    "clear_collection_async",
    "AsyncIndexingEngine",
    "get_async_indexing_engine",
    # RAG chatting
    "naive_rag_chatting_async",
    "multiquery_rag_chatting_async",
    "dynamic_rag_chatting_async",
    "HyDE_rag_chatting_async",
    "AdvancedAsyncRAGChattingEngine",
    "get_advanced_async_rag_chatting_engine",
    # RAG E2E
    "naive_rag_async",
    "multiquery_rag_async",
    "advanced_rag_async",
    "dynamic_rag_async",
    "contextual_rag_async",
    "AdvancedAsyncRAGE2EEngine",
    "get_advanced_async_rag_e2e_engine",
    # Vector database clients
    "AsyncBaseVectorDBClient",
    "create_async_client",
    "AsyncChromaVectorDBClient",
    "AsyncMilvusVectorDBClient",
    # Data models (same as sync)
    "DocLoadingInput",
    "DocLoadingOutput",
    "DocSplittingInput",
    "DocSplittingOutput",
    "SituatingChunksInput",
    "SituatingChunksOutput",
    "EmbeddingDocsInput",
    "EmbeddingDocsOutput",
    "EmbeddingQueryInput",
    "EmbeddingQueryOutput",
    "VDBStoringInput",
    "VDBStoringOutput",
    "VDBSearchingInput",
    "VDBSearchingOutput",
    "IndexingInput",
    "IndexingOutput",
    "CollectionClearingInput",
    "QueryTranslationInput",
    "QueryTranslationOutput",
    "RetrievalInput",
    "RetrievalOutput",
    "PostRetrievalInput",
    "PostRetrievalOutput",
    "GenerationInput",
    "GenerationOutput",
    "RAGChattingInput",
    "RAGChattingOutput",
    "JudgingOutput",
    "RetrievalJudgingInput",
    "DocRelevanceJudgingInput",
    "HallucinationJudgingInput",
    "AnswerJudgingInput",
    "RAGE2EInput",
    "RAGE2EOutput",
    # Settings (same as sync)
    "init_settings",
    "get_settings",
    "LLMSettings",
    "SplitterSettings",
    "VDBSettings",
    "EmbeddingSettings",
    "RetrieverSettings",
    "GenerationSettings",
    "LoggingSettings",
]
