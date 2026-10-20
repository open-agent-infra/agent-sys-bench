"""
Search task module for deep research operations.
"""

from typing import Dict, Any, Optional, List, Union, Literal, cast, Annotated
import hashlib
import logging
from pydantic import BaseModel, Field
from fastapi import Body
import asyncio
from collections import defaultdict
import itertools
from concurrent.futures import ThreadPoolExecutor

from agsbench_deep_research.core.schema.io import TaskRequest, TaskResponse
from agsbench_deep_research.core.registry_setup import get_service_manager
from agsbench_deep_research.core.service.search import SearchServiceRequest, SearchServiceResponse
from agsbench_deep_research.core.service.search import (
    RetrievedDocument,
    SearchQueryResponse,
    DEFAULT_MAX_SEARCH_RESULTS,
)
from ..shared import agsbench_deep_research_fsm, RequestContext
from ..shared import DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL
from ..shared import DEFAULT_SPLITTING_MODEL, DEFAULT_VDB_COLLECTION
from ..shared import DEFAULT_SEARCH_POST_PROCESSING_CONFIG
from ..state import Section, Sections, Queries, Feedback, SearchResultSummary
from ..utils import (
    deduplicate_search_results,
    get_formatted_search_results,
)
from ..task.llm_task import (
    SummarizeRetrievedDocumentRequest,
    SummarizeRetrievedDocumentResponse,
    summarize_retrieved_document_with_llm,
    summarize_retrieved_document_with_llm_async,
)
from ..task.splitting_task import (
    SplittingTaskRequest,
    SplittingTaskResponse,
    split_texts,
    split_texts_async,
)
from ..task.embedding_task import (
    EmbeddingTaskRequest,
    EmbeddingTaskResponse,
    embed_texts,
    embed_texts_async,
    embed_docs,
    embed_docs_async,
)
from ..task.vdb_task import (
    VDBStoreRequest,
    VDBStoreResponse,
    VDBSearchRequest,
    VDBSearchResponse,
    VDBClearRequest,
    VDBClearResponse,
    store_vdb,
    store_vdb_async,
    search_vdb,
    search_vdb_async,
    clear_vdb,
    clear_vdb_async,
)


logger = logging.getLogger(__name__)


class SearchTaskRequest(TaskRequest):
    """Request schema for search tasks in deep research."""

    task_name: str = "search_task"
    """Name of the task, used for identification."""

    queries: List[str] = Field(..., description="The search queries to execute.")
    search_api: str = Field(
        ..., description="The search API to use (e.g., 'tavily', 'perplexity')."
    )
    search_api_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Configuration parameters for the search API."
    )
    max_tokens_per_source: Optional[int] = Field(
        default=5000,
        description="Maximum number of tokens to return per source in the search results.",
    )
    include_raw_content: bool = Field(
        default=True,
        description="Whether to include raw content in the search results.",
    )
    deduplication_strategy: Literal["keep_first", "keep_last"] = Field(
        default="keep_first",
        description="Strategy for deduplicating search results with the same url. "
        "'keep_first' retains the first occurrence, while 'keep_last' retains the last occurrence.",
    )
    search_post_processing: Optional[Dict[str, Any]] = Field(
        default=DEFAULT_SEARCH_POST_PROCESSING_CONFIG,
        description="Optional post-processing parameters for the search results. "
        "Example1: {'operation': 'summarize', 'summarization_model': '<model_name>', 'summarization_model_kwargs': {'max_tokens': 4096, 'temperature': 0.0} }"
        "Example2: {'operation': 'split_and_rerank', 'splitting_model': '<splitter_name>', 'chunk_size': 1500, 'chunk_overlap': 200, 'embedding_model': '<model_name>', 'vdb': <vdb_name>, 'temperature': 0.0}",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the search task request."""
        return {
            "task_name": self.task_name,
            "queries_count": len(self.queries),
            "search_api": self.search_api,
            "search_api_config_keys": list(self.search_api_config.keys()),
            "max_tokens_per_source": self.max_tokens_per_source,
            "include_raw_content": self.include_raw_content,
            "deduplication_strategy": self.deduplication_strategy,
            "search_post_processing_keys": (
                list(self.search_post_processing.keys())
                if isinstance(self.search_post_processing, dict)
                else []
            ),
        }


class SearchTaskResponse(TaskResponse):
    """Response schema for search tasks in deep research."""

    task_name: str = "search_task"
    """Name of the task, used for identification."""

    merged_content: str = Field(
        ...,
        description="Merged content from all search results, formatted for readability.",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the search task response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "merged_content_length": len(self.merged_content),
            }
        )
        return stats


class SearchWithQueriesResponse(TaskResponse):
    """Response schema for search tasks with queries."""

    task_name: str = "search_with_queries"
    """Name of the task, used for identification."""

    results: Optional[Union[SearchQueryResponse, List[SearchQueryResponse]]] = Field(
        None,
        description="Search results from the search API, containing individual query responses.",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the search with queries response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "results_count": len(self.results) if self.results else 0,
                "results_type": (
                    type(self.results).__name__ if self.results is not None else "None"
                ),
            }
        )
        return stats


@agsbench_deep_research_fsm.fast_service
def search_with_queries(
    request: SearchTaskRequest,
    context: RequestContext = None,
) -> SearchWithQueriesResponse:
    """Search with the specified query and search API.
    This function executes a search using the provided query and search API,
    returning a response containing the search results.
    Args:
        request (SearchTaskRequest): The request object containing the search query,
            search API, and optional configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.
    Returns:
        SearchWithQueriesResponse: The response object containing the search results.
    """
    logger.debug(
        "Executing search with request: %s",
        request.get_statistics(),
    )
    # Get the search service
    service_manager = get_service_manager()
    search_service = service_manager.get_service("search")

    if not search_service:
        raise ValueError("Search service is not available.")
    # Prepare the search request
    max_results = (
        request.search_api_config.get("max_results", DEFAULT_MAX_SEARCH_RESULTS)
        if request.search_api_config
        else DEFAULT_MAX_SEARCH_RESULTS
    )
    search_request = SearchServiceRequest(
        queries=request.queries,
        engine=request.search_api,
        max_results=max_results,
        kwargs=request.search_api_config or {},
    )
    search_response: SearchServiceResponse = search_service.serve(
        request=search_request
    )
    return search_response


@agsbench_deep_research_fsm.async_fast_service
async def search_with_queries_async(
    request: SearchTaskRequest,
    context: RequestContext = None,
) -> SearchWithQueriesResponse:
    """Asynchronous version of the search task function.

    Args:
        request (SearchTaskRequest): The request object containing the search query,
            search API, and optional configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.

    Returns:
        SearchWithQueriesResponse: The response object containing the search results.
    """
    logger.debug(
        "Asynchronously executing search with request: %s",
        request.get_statistics(),
    )
    # Get the search service
    service_manager = get_service_manager()
    search_service = service_manager.get_service("search")

    if not search_service:
        raise ValueError("Search service is not available.")
    # Prepare the search request
    max_results = (
        request.search_api_config.get("max_results", DEFAULT_MAX_SEARCH_RESULTS)
        if request.search_api_config
        else DEFAULT_MAX_SEARCH_RESULTS
    )
    search_request = SearchServiceRequest(
        queries=request.queries,
        engine=request.search_api,
        max_results=max_results,
        kwargs=request.search_api_config or {},
    )
    search_response: SearchServiceResponse = await search_service.serve_async(
        request=search_request
    )
    return search_response


class SummarizeSearchResultsRequest(TaskRequest):
    """Request schema for summarizing search results."""

    task_name: str = "summarize_search_results"
    """Name of the task, used for identification."""

    search_results: List[RetrievedDocument] = Field(
        ...,
        description="List of search results to summarize.",
    )
    model: str = Field(..., description="Model to use for summarization.")
    max_tokens: Optional[int] = Field(
        default=4096, description="Maximum number of tokens for the summary."
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Temperature for the summarization model."
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for random number generation (if needed)"
    )
    extra_gen_kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional generation parameters for the summarization model.",
    )
    max_char_to_include: Optional[int] = Field(
        default=30_000,
        description="Maximum number of characters to include from the document for summarization.",
    )
    parallelize: bool = Field(
        default=True,
        description="Whether to parallelize the summarization of search results.",
    )
    parallelization_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration parameters for parallelization, if applicable.",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the summarization task."""
        return {
            "task_name": self.task_name,
            "search_results_count": len(self.search_results),
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_char_to_include": self.max_char_to_include,
            "parallelize": self.parallelize,
            "parallelization_config": self.parallelization_config,
        }


class SummarizeSearchResultsResponse(TaskResponse):
    """Response schema for summarizing search results."""

    task_name: str = "summarize_search_results"
    """Name of the task, used for identification."""

    summaries: List[SearchResultSummary] = Field(
        ...,
        description="List of summaries for the search results.",
    )


@agsbench_deep_research_fsm.fast_service
def summarize_search_results(
    request: SummarizeSearchResultsRequest,
    context: RequestContext = None,
) -> SummarizeSearchResultsResponse:
    """Summarize the provided search results using the specified model.
    This function takes a list of search results and generates summaries for each result
    using the specified model.

    Args:
        request (SummarizeSearchResultsRequest): The request object containing the search results,
            model, and configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.

    Returns:
        SummarizeSearchResultsResponse: The response object containing the summaries of the search results.
    """
    logger.debug(
        "Summarizing search results with request: %s",
        request.get_statistics(),
    )

    def summarize_helper(
        document: RetrievedDocument,
    ) -> SearchResultSummary:
        """Helper function to summarize a single retrieved document."""
        summary_request = SummarizeRetrievedDocumentRequest(
            document=document,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            extra_gen_kwargs=request.extra_gen_kwargs,
            max_char_to_include=request.max_char_to_include,
        )
        summary_response = summarize_retrieved_document_with_llm(
            request=summary_request, context=context
        )
        summary = SearchResultSummary(
            summary=summary_response.summary,
            key_excerpts=summary_response.key_excerpts,
        )
        return summary

    # Create a list to hold the summaries
    summaries = []
    if request.parallelize:
        # using threadpool for parallelization
        max_concurrency = request.parallelization_config.get("n_workers", 32)
        try:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = [
                    executor.submit(
                        summarize_helper,
                        document=result,
                    )
                    for result in request.search_results
                ]
            for future in futures:
                summaries.append(future.result())
        except Exception as e:
            logger.exception("Error during asynchronous summarization: %s", str(e))
            raise e
    else:
        # Iterate over each search result and summarize it
        for result in request.search_results:
            summaries.append(summarize_helper(result))

    # Create the response object
    response = SummarizeSearchResultsResponse(
        request_id=request.request_id, summaries=summaries
    )

    return response


@agsbench_deep_research_fsm.async_fast_service
async def summarize_search_results_async(
    request: SummarizeSearchResultsRequest,
    context: RequestContext = None,
) -> SummarizeSearchResultsResponse:
    """Asynchronous version of the summarize search results function.

    Args:
        request (SummarizeSearchResultsRequest): The request object containing the search results,
            model, and configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.

    Returns:
        SummarizeSearchResultsResponse: The response object containing the summaries of the search results.
    """
    logger.debug(
        "Asynchronously summarizing search results with request: %s",
        request.get_statistics(),
    )

    # Create a list of tasks to summarize each search result
    tasks = []
    for result in request.search_results:
        summary_request = SummarizeRetrievedDocumentRequest(
            document=result,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            extra_gen_kwargs=request.extra_gen_kwargs,
            max_char_to_include=request.max_char_to_include,
        )
        tasks.append(
            summarize_retrieved_document_with_llm_async(
                request=summary_request, context=context
            )
        )
    # Run all summarization tasks concurrently
    summary_responses: List[SummarizeRetrievedDocumentResponse] = await asyncio.gather(
        *tasks
    )
    # Create a list to hold the summaries
    summaries = [
        SearchResultSummary(summary=resp.summary, key_excerpts=resp.key_excerpts)
        for resp in summary_responses
    ]

    # Create the response object
    response = SummarizeSearchResultsResponse(
        request_id=request.request_id, summaries=summaries
    )

    return response


@agsbench_deep_research_fsm.fast_service
def stitch_documents_by_url(
    documents: List[RetrievedDocument],
) -> List[RetrievedDocument]:
    """Stitch documents by URL to combine content from the same URL into a single document.
    This function iterates through the list of documents and combines the content of documents
    that have the same URL.

    Args:
        documents (List[RetrievedDocument]): The list of documents to stitch.

    Returns:
        List[RetrievedDocument]: The stitched list of documents.
    """
    logger.debug("Stitching documents by URL for %d documents", len(documents))
    url_to_docs: defaultdict[str, list[RetrievedDocument]] = defaultdict(list)
    url_to_snippet_hashes: defaultdict[str, set[str]] = defaultdict(set)
    for doc in documents:
        snippet_hash = hashlib.sha256(doc.content.encode()).hexdigest()
        url = doc.url
        # deduplicate snippets by the content
        if snippet_hash in url_to_snippet_hashes[url]:
            continue

        url_to_docs[url].append(doc)
        url_to_snippet_hashes[url].add(snippet_hash)

    # stitch retrieved chunks into a single doc per URL
    stitched_docs = []
    for docs in url_to_docs.values():
        stitched_doc = RetrievedDocument(
            content="\n\n".join([f"...{doc.content}..." for doc in docs]),
            title=docs[0].title,
            url=docs[0].url,
            score=max(doc.score for doc in docs),
            id=docs[0].id,
            raw_content="\n\n".join([doc.raw_content or doc.content for doc in docs]),
            extra=docs[0].extra or {},
        )
        stitched_docs.append(stitched_doc)

    return stitched_docs


@agsbench_deep_research_fsm.fast_service
def split_and_rerank_search_results(
    docs: List[RetrievedDocument],
    query: Annotated[str, Body(..., description="The query to use for reranking")],
    process_cfg: Dict[str, Any] = None,
    context: RequestContext = None,
) -> List[RetrievedDocument]:
    """Split and rerank search results based on the specified operation.
    This function applies splitting and reranking operations to the provided search results.

    Args:
        docs (List[RetrievedDocument]): The list of search results to process.
        process_cfg (Dict[str, Any]): The processing configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.

    Returns:
        List[RetrievedDocument]: The processed search results.
    """
    logger.debug(
        "Splitting and reranking %d search results with configuration: %s",
        len(docs),
        process_cfg,
    )
    # split the documents into chunks
    splitting_request = SplittingTaskRequest(
        texts=[doc.raw_content or doc.content for doc in docs],
        metadatas=[doc.model_dump(include=["id", "title", "url"]) for doc in docs],
        model=process_cfg.get("splitting_model", DEFAULT_SPLITTING_MODEL),
        chunk_size=process_cfg.get("chunk_size", 1500),
        chunk_overlap=process_cfg.get("chunk_overlap", 200),
    )
    splitting_response = split_texts(request=splitting_request, context=context)
    if not splitting_response.success:
        logger.error(
            "Splitting task failed: %s", splitting_response.error or "Unknown error"
        )
        return docs
    if len(splitting_response.chunks) == 0:
        logger.warning("No chunks were created from the splitting task.")
        return docs
    # Assuming the splitting response contains chunks
    chunks = splitting_response.chunks
    chunk_metadatas = splitting_response.metadatas or [{}] * len(chunks)
    # Now we need to embed the chunks
    embedding_request = EmbeddingTaskRequest(
        texts=chunks,
        model=process_cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
    )
    embedding_response = embed_docs(request=embedding_request, context=context)
    if not embedding_response.success:
        logger.error(
            "Embedding task failed: %s", embedding_response.error or "Unknown error"
        )
        return docs
    embeddings = embedding_response.embeddings
    # Now we need to store the chunks and embeddings in the vector database
    collection_name = (
        process_cfg.get("vdb_collection", DEFAULT_VDB_COLLECTION)
        + "_"
        + hashlib.md5(query.encode()).hexdigest()
        + "_"
        + (context.call_id if context else "UnknownContextCallID")
    )

    def clear_collection():
        """Helper function to clear the vector database collection."""
        vdb_clear_request = VDBClearRequest(
            collection_name=collection_name,
        )
        vdb_clear_response = clear_vdb(request=vdb_clear_request, context=context)
        if not vdb_clear_response.success:
            logger.error(
                "VDB clear task failed: %s",
                vdb_clear_response.error or "Unknown error",
            )

    vdb_store_request = VDBStoreRequest(
        texts=chunks,
        embeddings=embeddings,
        metadatas=chunk_metadatas,
        ids=None,  # do not pass ids, they will be generated by the VDB
        collection_name=collection_name,
    )
    vdb_store_response = store_vdb(request=vdb_store_request, context=context)
    if not vdb_store_response.success:
        logger.error(
            "VDB store task failed: %s",
            vdb_store_response.error or "Unknown error",
        )
        clear_collection()  # Clear the collection if store failed
        return docs
    # Now we need to embed the query
    query_embedding_request = EmbeddingTaskRequest(
        texts=[query],
        model=process_cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
    )
    query_embedding_response = embed_texts(
        request=query_embedding_request, context=context
    )
    if (
        not query_embedding_response.success
        or not query_embedding_response.embeddings
        or len(query_embedding_response.embeddings) == 0
    ):
        logger.error(
            "Query embedding task failed: %s",
            query_embedding_response.error or "Unknown error",
        )
        clear_collection()  # Clear the collection if embedding failed
        return docs
    query_embedding = query_embedding_response.embeddings[0]
    # Now we need to search the vector database
    vdb_search_request = VDBSearchRequest(
        query_texts=[query],
        query_embeddings=[query_embedding],
        collection_name=collection_name,
        k=process_cfg.get("vdb_topk", 5),  # Number of results to return
    )
    vdb_search_response = search_vdb(request=vdb_search_request, context=context)
    if not vdb_search_response.success:
        logger.error(
            "VDB search task failed: %s",
            vdb_search_response.error or "Unknown error",
        )
        clear_collection()  # Clear the collection if search failed
        return docs
    clear_collection()  # Clear the collection to avoid cluttering it with temporary data
    # Replace the original docs with the search results from the VDB
    retrieved_data = vdb_search_response.results[0]
    docs = [
        RetrievedDocument(
            title=data.metadata.get("title", ""),
            url=data.metadata.get("url", ""),
            content=data.text,
            score=data.score,
            id=data.metadata.get("id", data.id),
            raw_content=data.text,
            extra=data.metadata,
        )
        for data in retrieved_data
    ]
    return docs


@agsbench_deep_research_fsm.async_fast_service
async def split_and_rerank_search_results_async(
    docs: List[RetrievedDocument],
    query: Annotated[str, Body(..., description="The query to use for reranking")],
    process_cfg: Dict[str, Any] = None,
    context: RequestContext = None,
) -> List[RetrievedDocument]:
    """Asynchronous version of the split and rerank search results function.

    Args:
        docs (List[RetrievedDocument]): The list of search results to process.
        query (str): The query to use for reranking.
        process_cfg (Dict[str, Any]): The processing configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.

    Returns:
        List[RetrievedDocument]: The processed search results.
    """
    logger.debug(
        "Asynchronously splitting and reranking %d search results with configuration: %s",
        len(docs),
        process_cfg,
    )
    # split the documents into chunks
    splitting_request = SplittingTaskRequest(
        texts=[doc.raw_content or doc.content for doc in docs],
        metadatas=[doc.model_dump(include=["id", "title", "url"]) for doc in docs],
        model=process_cfg.get("splitting_model", DEFAULT_SPLITTING_MODEL),
        chunk_size=process_cfg.get("chunk_size", 1500),
        chunk_overlap=process_cfg.get("chunk_overlap", 200),
    )
    splitting_response = await split_texts_async(
        request=splitting_request, context=context
    )
    if not splitting_response.success:
        logger.error(
            "Splitting task failed: %s", splitting_response.error or "Unknown error"
        )
        return docs
    if len(splitting_response.chunks) == 0:
        logger.warning("No chunks were created from the splitting task.")
        return docs
    # Assuming the splitting response contains chunks
    chunks = splitting_response.chunks
    chunk_metadatas = splitting_response.metadatas or [{}] * len(chunks)
    # Now we need to embed the chunks
    embedding_request = EmbeddingTaskRequest(
        texts=chunks,
        model=process_cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
    )
    embedding_response = await embed_docs_async(
        request=embedding_request, context=context
    )
    if not embedding_response.success:
        logger.error(
            "Embedding task failed: %s", embedding_response.error or "Unknown error"
        )
        return docs
    embeddings = embedding_response.embeddings
    collection_name = (
        process_cfg.get("vdb_collection", DEFAULT_VDB_COLLECTION)
        + "_"
        + hashlib.md5(query.encode()).hexdigest()
        + "_"
        + (context.call_id if context else "UnknownContextCallID")
    )

    async def clear_collection_async():
        """Helper function to clear the vector database collection."""
        vdb_clear_request = VDBClearRequest(
            collection_name=collection_name,
        )
        vdb_clear_response = await clear_vdb_async(
            request=vdb_clear_request, context=context
        )
        if not vdb_clear_response.success:
            logger.error(
                "VDB clear task failed: %s",
                vdb_clear_response.error or "Unknown error",
            )

    # Now we need to store the chunks and embeddings in the vector database
    vdb_store_request = VDBStoreRequest(
        texts=chunks,
        embeddings=embeddings,
        metadatas=chunk_metadatas,
        ids=None,  # do not pass ids, they will be generated by the VDB
        collection_name=collection_name,
    )
    vdb_store_response = await store_vdb_async(
        request=vdb_store_request, context=context
    )
    if not vdb_store_response.success:
        logger.error(
            "VDB store task failed: %s",
            vdb_store_response.error or "Unknown error",
        )
        await clear_collection_async()  # Clear the collection if store failed
        return docs
    # Now we need to embed the query
    query_embedding_request = EmbeddingTaskRequest(
        texts=[query],
        model=process_cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
    )
    query_embedding_response = await embed_texts_async(
        request=query_embedding_request, context=context
    )
    if (
        not query_embedding_response.success
        or not query_embedding_response.embeddings
        or len(query_embedding_response.embeddings) == 0
    ):
        logger.error(
            "Query embedding task failed: %s",
            query_embedding_response.error or "Unknown error",
        )
        await clear_collection_async()  # Clear the collection if embedding failed
        return docs
    query_embedding = query_embedding_response.embeddings[0]
    # Now we need to search the vector database
    vdb_search_request = VDBSearchRequest(
        query_texts=[query],
        query_embeddings=[query_embedding],
        collection_name=collection_name,
        k=process_cfg.get("vdb_topk", 5),  # Number of results to return
    )
    vdb_search_response = await search_vdb_async(
        request=vdb_search_request, context=context
    )
    if not vdb_search_response.success:
        logger.error(
            "VDB search task failed: %s",
            vdb_search_response.error or "Unknown error",
        )
        await clear_collection_async()  # Clear the collection if search failed
        return docs
    await clear_collection_async()  # Clear the collection to avoid cluttering it with temporary data
    # Replace the original docs with the search results from the VDB
    retrieved_data = vdb_search_response.results[0]
    docs = [
        RetrievedDocument(
            title=data.metadata.get("title", ""),
            url=data.metadata.get("url", ""),
            content=data.text,
            score=data.score,
            id=data.metadata.get("id", data.id),
            raw_content=data.text,
            extra=data.metadata,
        )
        for data in retrieved_data
    ]
    return docs


@agsbench_deep_research_fsm.fast_service
def post_process_search_results(
    docs: List[RetrievedDocument],
    process_cfg: Dict[str, Any] = None,
    context: RequestContext = None,
) -> List[RetrievedDocument]:
    """Post-process search results based on the specified operation.
    This function applies post-processing operations such as summarization or split and rerank
    to the provided search results.

    Args:
        docs (List[RetrievedDocument]): The list of search results to post-process.
        process_cfg (Dict[str, Any]): The post-processing configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.

    Returns:
        List[RetrievedDocument]: The post-processed search results.
    """
    logger.debug(
        "Post-processing %d search results with configuration: %s",
        len(docs),
        process_cfg,
    )
    if not process_cfg:
        logger.warning(
            "No post-processing configuration provided, returning original docs."
        )
        return docs
    if process_cfg.get("operation") == "summarize":
        summarize_request = SummarizeSearchResultsRequest(
            search_results=docs,
            model=process_cfg.get("summarization_model", DEFAULT_LLM_MODEL),
            max_tokens=process_cfg.get("summarization_model_kwargs", {}).get(
                "max_tokens", 4096
            ),
            temperature=process_cfg.get("summarization_model_kwargs", {}).get(
                "temperature", 0.0
            ),
            seed=process_cfg.get("summarization_model_kwargs", {}).get("seed", None),
            extra_gen_kwargs=process_cfg.get("summarization_model_kwargs", {}).get(
                "extra_gen_kwargs", {}
            ),
            max_char_to_include=process_cfg.get("max_char_to_include", 30_000),
            parallelize=process_cfg.get("parallelize_summarization", True),
            parallelization_config={
                "n_workers": process_cfg.get("parallelize_summarization_n_workers", 32)
            },
        )
        summarize_response = summarize_search_results(
            request=summarize_request, context=context
        )
        for i, summary in enumerate(summarize_response.summaries):
            if summary.summary:
                docs[i].content = summary.get_formatted_summary()
                docs[i].extra = docs[i].extra or {}
                docs[i].extra["summary"] = summary
    elif process_cfg.get("operation") == "split_and_rerank":
        # group the documents by query and then split_and_rerank each group
        query_to_docs: Dict[str, List[RetrievedDocument]] = defaultdict(list)
        for doc in docs:
            # Use the first query as the key for grouping
            if not doc.extra or "query" not in doc.extra:
                logger.warning(
                    "Document %s has no query in extra metadata, skipping.",
                    doc.id,
                )
                continue
            query = doc.extra.get("query", "")
            query_to_docs[query].append(doc)
        # Now process each group of documents
        new_docs: List[RetrievedDocument] = []
        if process_cfg.get("parallelize_split_and_rerank", True):
            # using threadpool for parallelization
            max_concurrency = process_cfg.get(
                "parallelize_split_and_rerank_n_workers", 32
            )
            try:
                with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                    futures = [
                        executor.submit(
                            split_and_rerank_search_results,
                            group_docs,
                            query,
                            process_cfg,
                            context=context,
                        )
                        for query, group_docs in query_to_docs.items()
                    ]
                for future in futures:
                    new_docs.extend(future.result())
            except Exception as e:
                logger.exception(
                    "Error during asynchronous splitting and reranking: %s", str(e)
                )
                raise e
        else:
            for query, group_docs in query_to_docs.items():
                # split and rerank the documents for this query
                group_docs = cast(
                    List[RetrievedDocument],
                    split_and_rerank_search_results(
                        docs=group_docs,
                        query=query,
                        process_cfg=process_cfg,
                        context=context,
                    ),
                )
                new_docs.extend(group_docs)
        # now stitch the documents by URL to combine content from the same URL into a single document
        docs = stitch_documents_by_url(new_docs)

        # Optionally, you can also summarize the search results from the VDB
        if process_cfg.get("summarize_after_rerank", False):
            summarize_request = SummarizeSearchResultsRequest(
                search_results=docs,
                model=process_cfg.get("summarization_model", DEFAULT_LLM_MODEL),
                max_tokens=process_cfg.get("summarization_model_kwargs", {}).get(
                    "max_tokens", 4096
                ),
                temperature=process_cfg.get("summarization_model_kwargs", {}).get(
                    "temperature", 0.0
                ),
                seed=process_cfg.get("summarization_model_kwargs", {}).get(
                    "seed", None
                ),
                extra_gen_kwargs=process_cfg.get("summarization_model_kwargs", {}).get(
                    "extra_gen_kwargs", {}
                ),
                max_char_to_include=process_cfg.get("max_char_to_include", 30_000),
                parallelize=process_cfg.get("parallelize_summarization", True),
                parallelization_config={
                    "n_workers": process_cfg.get(
                        "parallelize_summarization_n_workers", 32
                    )
                },
            )
            summarize_response = summarize_search_results(
                request=summarize_request, context=context
            )
            for i, summary in enumerate(summarize_response.summaries):
                if summary.summary:
                    docs[i].content = summary.get_formatted_summary()
                    docs[i].extra = docs[i].extra or {}
                    docs[i].extra["summary"] = summary
    # Add more post-processing operations as needed
    else:
        logger.warning(
            "Unknown search post-processing operation: %s",
            process_cfg.get("operation"),
        )
    logger.debug(
        "Post-processing complete. Returning %d documents.",
        len(docs),
    )
    return docs


@agsbench_deep_research_fsm.async_fast_service
async def post_process_search_results_async(
    docs: List[RetrievedDocument],
    process_cfg: Dict[str, Any] = None,
    context: RequestContext = None,
) -> List[RetrievedDocument]:
    logger.debug(
        "Asynchronously post-processing %d search results with configuration: %s",
        len(docs),
        process_cfg,
    )
    if not process_cfg:
        logger.warning(
            "No post-processing configuration provided, returning original docs."
        )
        return docs
    if process_cfg.get("operation") == "summarize":
        summarize_request = SummarizeSearchResultsRequest(
            search_results=docs,
            model=process_cfg.get("summarization_model", DEFAULT_LLM_MODEL),
            max_tokens=process_cfg.get("summarization_model_kwargs", {}).get(
                "max_tokens", 4096
            ),
            temperature=process_cfg.get("summarization_model_kwargs", {}).get(
                "temperature", 0.0
            ),
            seed=process_cfg.get("summarization_model_kwargs", {}).get("seed", None),
            extra_gen_kwargs=process_cfg.get("summarization_model_kwargs", {}).get(
                "extra_gen_kwargs", {}
            ),
            max_char_to_include=process_cfg.get("max_char_to_include", 30_000),
            parallelize=process_cfg.get("parallelize_summarization", True),
            parallelization_config={
                "n_workers": process_cfg.get("parallelize_summarization_n_workers", 32)
            },
        )
        summarize_response = await summarize_search_results_async(
            request=summarize_request, context=context
        )
        for i, summary in enumerate(summarize_response.summaries):
            if summary.summary:
                docs[i].content = summary.get_formatted_summary()
                docs[i].extra = docs[i].extra or {}
                docs[i].extra["summary"] = summary
    elif process_cfg.get("operation") == "split_and_rerank":
        # group the documents by query and then split_and_rerank each group
        query_to_docs: Dict[str, List[RetrievedDocument]] = defaultdict(list)
        for doc in docs:
            # Use the first query as the key for grouping
            if not doc.extra or "query" not in doc.extra:
                logger.warning(
                    "Document %s has no query in extra metadata, skipping.",
                    doc.id,
                )
                continue
            query = doc.extra.get("query", "")
            query_to_docs[query].append(doc)
        # Now process each group of documents
        new_docs: List[RetrievedDocument] = []
        for query, group_docs in query_to_docs.items():
            # split and rerank the documents for this query
            group_docs = cast(
                List[RetrievedDocument],
                await split_and_rerank_search_results_async(
                    docs=group_docs,
                    query=query,
                    process_cfg=process_cfg,
                    context=context,
                ),
            )
            new_docs.extend(group_docs)
        # now stitch the documents by URL to combine content from the same URL into a single document
        docs = stitch_documents_by_url(new_docs)

        # Optionally, you can also summarize the search results from the VDB
        if process_cfg.get("summarize_after_rerank", False):
            summarize_request = SummarizeSearchResultsRequest(
                search_results=docs,
                model=process_cfg.get("summarization_model", DEFAULT_LLM_MODEL),
                max_tokens=process_cfg.get("summarization_model_kwargs", {}).get(
                    "max_tokens", 4096
                ),
                temperature=process_cfg.get("summarization_model_kwargs", {}).get(
                    "temperature", 0.0
                ),
                seed=process_cfg.get("summarization_model_kwargs", {}).get(
                    "seed", None
                ),
                extra_gen_kwargs=process_cfg.get("summarization_model_kwargs", {}).get(
                    "extra_gen_kwargs", {}
                ),
                max_char_to_include=process_cfg.get("max_char_to_include", 30_000),
                parallelize=process_cfg.get("parallelize_summarization", True),
                parallelization_config={
                    "n_workers": process_cfg.get(
                        "parallelize_summarization_n_workers", 32
                    )
                },
            )
            summarize_response = await summarize_search_results_async(
                request=summarize_request, context=context
            )
            for i, summary in enumerate(summarize_response.summaries):
                if summary.summary:
                    docs[i].content = summary.get_formatted_summary()
                    docs[i].extra = docs[i].extra or {}
                    docs[i].extra["summary"] = summary
    # Add more post-processing operations as needed
    else:
        logger.warning(
            "Unknown search post-processing operation: %s",
            process_cfg.get("operation"),
        )
    return docs


@agsbench_deep_research_fsm.fast_service
def search_task(
    request: SearchTaskRequest,
    context: RequestContext = None,
) -> SearchTaskResponse:
    """Search with the specified query and search API.
    This function executes a search using the provided query and search API,
    returning a response containing the merged content and individual search results.

    Args:
        request (SearchTaskRequest): The request object containing the search query,
            search API, and optional configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.
    Returns:
        SearchTaskResponse: The response object containing the merged content and
            individual search results.
    """
    logger.debug(
        "Executing search task with request: %s",
        request.get_statistics(),
    )
    search_response = search_with_queries(request=request, context=context)
    search_results = []
    if isinstance(search_response.results, SearchQueryResponse):
        # If the response is a single SearchQueryResponse, convert it to a list
        search_response.results = [search_response.results]
    for result in search_response.results:
        for doc in result.docs:
            doc.extra = doc.extra or {}
            doc.extra["query"] = result.query
            search_results.append(doc)
    unique_results = deduplicate_search_results(
        search_results=search_results,
        deduplication_strategy=request.deduplication_strategy,
    )
    if request.search_post_processing:
        logger.debug(
            "Post-processing search results with configuration: %s and results count: %d",
            request.search_post_processing,
            len(unique_results),
        )

        unique_results = post_process_search_results(
            docs=unique_results,
            process_cfg=request.search_post_processing,
            context=context,
        )
    # If the post-processing operation is not specified, we can skip it
    # and directly format the search results.
    merged_content = get_formatted_search_results(
        search_results=unique_results,
        include_raw_content=request.include_raw_content,
        max_char_to_include=request.search_post_processing.get(
            "max_char_to_include",
            (
                request.max_tokens_per_source * 4
                if request.max_tokens_per_source
                else 30_000
            ),
        ),
    )
    # Create the response object
    response = SearchTaskResponse(
        request_id=request.request_id, merged_content=merged_content
    )
    logger.debug(
        "Search task completed with %d results and merged content length: %d",
        len(unique_results),
        len(response.merged_content),
    )
    return response


@agsbench_deep_research_fsm.async_fast_service
async def search_task_async(
    request: SearchTaskRequest,
    context: RequestContext = None,
) -> SearchTaskResponse:
    """Asynchronous version of the search task function.

    Args:
        request (SearchTaskRequest): The request object containing the search query,
            search API, and optional configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.

    Returns:
        SearchTaskResponse: The response object containing the merged content and
            individual search results.
    """
    logger.debug(
        "Executing asynchronous search task with request: %s",
        request.get_statistics(),
    )
    search_response = await search_with_queries_async(request=request, context=context)
    search_results = []
    if isinstance(search_response.results, SearchQueryResponse):
        # If the response is a single SearchQueryResponse, convert it to a list
        search_response.results = [search_response.results]
    for result in search_response.results:
        for doc in result.docs:
            doc.extra = doc.extra or {}
            doc.extra["query"] = result.query
            search_results.append(doc)
    unique_results = deduplicate_search_results(
        search_results=search_results,
        deduplication_strategy=request.deduplication_strategy,
    )
    if request.search_post_processing:
        unique_results = await post_process_search_results_async(
            docs=unique_results,
            process_cfg=request.search_post_processing,
            context=context,
        )
    # If the post-processing operation is not specified, we can skip it
    # and directly format the search results.
    merged_content = get_formatted_search_results(
        search_results=unique_results,
        include_raw_content=request.include_raw_content,
        max_char_to_include=request.search_post_processing.get(
            "max_char_to_include",
            (
                request.max_tokens_per_source * 4
                if request.max_tokens_per_source
                else 30_000
            ),
        ),
    )
    # Create the response object
    response = SearchTaskResponse(
        request_id=request.request_id, merged_content=merged_content
    )
    logger.debug(
        "Asynchronous search task completed with %d results and merged content length: %d",
        len(unique_results),
        len(response.merged_content),
    )
    return response
