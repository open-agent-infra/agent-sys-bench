"""
Search service module for web and document search operations.

This module provides service implementations for interacting with
various search engines and APIs, including Tavily, Arxiv, DuckDuckGo, Wikipedia, Perplexity, and Google Search.
"""

import os
import time
import uuid
from abc import abstractmethod
from typing import List, Dict, Any, Optional, Iterable, cast, Union
from pydantic import BaseModel, Field
import logging
import asyncio
import json
import requests
import aiohttp
import hashlib
from concurrent.futures import ThreadPoolExecutor

from .base import (
    BaseServiceInstanceConfig,
    BaseServiceInstance,
    BaseService,
    BaseServiceRequest,
    BaseServiceResponse,
)
from .caching import get_cache_instance_class, CachingServiceInstanceConfig
from .scrape import scrape_with_requests, scrape_with_requests_async
from .scrape import scrape_pages_with_markitdown, scrape_pages_with_markitdown_async
from .scrape import scrap_with_firecrawl, scrap_with_firecrawl_async

search_logger = logging.getLogger(__name__)

DEFAULT_MAX_SEARCH_RESULTS = 5
IMAGE_CONTENT = Union[str, list]


class SearchServiceRequest(BaseServiceRequest):
    """Request model for Search service operations."""

    service_name: str = Field(
        default="search", description="Name of the search service."
    )
    operation_name: str = Field(
        default="search", description="Name of the operation being performed."
    )
    queries: Union[str, List[str]] = Field(
        ..., description="The search query or queries to execute."
    )
    engine: str = Field(default="tavily", description="Search engine to use.")
    max_results: int = Field(
        default=DEFAULT_MAX_SEARCH_RESULTS,
        description="Maximum number of search results to return for each query.",
    )
    kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the search operation, if needed.",
    )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search request."""
        stats = super().get_statistics()
        return {
            **stats,
            "service_name": self.service_name,
            "operation_name": self.operation_name,
            "queries": (
                self.queries if isinstance(self.queries, list) else [self.queries]
            ),
            "engine": self.engine,
            "max_results": self.max_results,
            "kwargs": self.kwargs,
        }


class RetrievedDocument(BaseModel):
    """Schema for a single search result."""

    title: Optional[str] = Field(None, description="Title of the search result.")
    url: Optional[str] = Field(None, description="URL of the search result.")
    content: Optional[str] = Field(None, description="Content of the search result.")
    score: Optional[float] = Field(
        None, description="Relevance score of the search result."
    )
    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the document, if available.",
    )
    raw_content: Optional[str] = Field(
        None, description="Raw content of the search result, if available."
    )
    extra: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata or information about the document."
    )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document."""
        return {
            "title": self.title,
            "url": self.url,
            "content_length": len(self.content) if self.content else 0,
            "score": self.score,
            "id": self.id,
            "raw_content_length": len(self.raw_content) if self.raw_content else 0,
        }


class SearchQueryResponse(BaseModel):
    """Response model for a single search query result."""

    query: str = Field(..., description="The search query that generated this result.")
    follow_up_queries: Optional[List[str]] = Field(
        None, description="List of follow-up queries related to this result."
    )
    answer: Optional[str] = Field(
        None, description="The answer or main content of the search result."
    )
    images: Optional[List[IMAGE_CONTENT]] = Field(
        None, description="List of image related to the search result."
    )
    docs: Optional[List[RetrievedDocument]] = Field(
        None, description="List of search results for the query."
    )
    elapsed_time: Optional[float] = Field(
        None, description="Elapsed time for the search operation in seconds."
    )
    from_cache: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cache data metadata if the response was served from cache",
    )
    extra: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata or information about the search result."
    )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search query response."""
        return {
            "query": self.query,
            "num_docs": len(self.docs) if self.docs else 0,
            "docs_statistics": (
                [doc.get_statistics() for doc in self.docs] if self.docs else []
            ),
            "num_images": len(self.images) if self.images else 0,
            "answer_length": len(self.answer) if self.answer else 0,
            "follow_up_queries": self.follow_up_queries or [],
            "elapsed_time": self.elapsed_time,
            "from_cache": self.from_cache is not None,
            "extra": self.extra or {},
        }


class SearchServiceResponse(BaseServiceResponse):
    """Response model for Search service operations."""

    results: Optional[Union[SearchQueryResponse, List[SearchQueryResponse]]] = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search service response."""
        if isinstance(self.results, SearchQueryResponse):
            return self.results.get_statistics()
        elif isinstance(self.results, list):
            stats = super().get_statistics()
            return {
                **stats,
                "num_queries": len(self.results),
                "queries_statistics": [
                    result.get_statistics() for result in self.results
                ],
            }
        else:
            stats = super().get_statistics()
            return {**stats, "num_queries": 0, "queries_statistics": []}


class SearchServiceInstanceConfig(BaseServiceInstanceConfig):
    """Configuration for a Search service instance."""

    engine: str = Field(
        ...,
        description="The search engine to use (e.g., 'tavily', 'arxiv', 'exa', 'duckduckgo', 'wikipedia', 'perplexity', 'googlesearch').",
    )
    max_concurrent_search: Optional[int] = Field(
        None,
        description="Maximum number of concurrent search requests to the search engine.",
    )
    caching_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration for caching search results, if applicable.",
    )
    kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the search engine, if needed.",
    )


class SearchServiceInstance(BaseServiceInstance):
    """Base class for Search service instances."""

    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__("search", instance_name, group_name, config)
        self.init_caching(config)

    @abstractmethod
    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        raise NotImplementedError

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single search query.
        This method runs the synchronous search helper in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.single_search_helper(query, max_results, **kwargs)
        )

    def launch(self):
        """Initialize any resources needed by the search instance."""
        pass

    def shutdown(self):
        """Clean up any resources used by the search instance."""
        pass

    def init_caching(self, config: SearchServiceInstanceConfig):
        self.caching_enabled = False
        self.caching_config = config.caching_config
        if self.caching_config:
            self.caching_enabled = self.caching_config.get("enabled", True)
            caching_backend = self.caching_config.get("backend", "file")
            caching_expiry = self.caching_config.get("default_ttl", None)
            caching_location = self.caching_config.get(
                "location", f"/tmp/legoagent/cache/instance_cache/{self.instance_name}"
            )
            caching_tags = set(self.caching_config.get("tags", []))
            caching_tags.add(caching_backend)
            self.caching_cls = get_cache_instance_class(caching_backend)
            config = CachingServiceInstanceConfig(
                backend=caching_backend,
                location=caching_location,
                default_ttl=caching_expiry,
                tags=frozenset(caching_tags),
            )
            self.caching_engine = self.caching_cls(
                f"instance_cache-of-{self.instance_name}",
                group_name=f"instance_cache/{self.instance_name}",
                config=config,
            )

    def get_cache_key(self, query: str, **kwargs) -> str:
        """Generate a cache key for the search query."""
        # Use a combination of query and kwargs to create a unique cache key
        key_parts = [query]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            key_parts.append(f"{k}:{v}")

        cache_key_prefix = f"{self.__class__.__name__}:{self.instance_name}"
        cache_key = f"{cache_key_prefix}:{':'.join(key_parts)}"
        # sometimes the cache key can be too long, convert to a hash
        if len(cache_key) > 128:
            cache_key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
            cache_key = f"{cache_key_prefix}:{cache_key_hash}"
            if len(cache_key) > 255:
                cache_key = cache_key_hash
        search_logger.debug(f"Generated cache key: {cache_key} for query '{query}'")
        return cache_key

    def get_cached_entry(self, query: str, **kwargs) -> Any:
        """Retrieve a cached search result if available."""
        if not self.caching_enabled:
            return None
        cache_key = self.get_cache_key(query, **kwargs)
        cached_entry = self.caching_engine.get(cache_key)
        if cached_entry:
            search_logger.debug(
                f"Cache hit for query '{query}' in {self.instance_name}"
            )
            return cast(SearchQueryResponse, cached_entry)
        search_logger.debug(f"Cache miss for query '{query}' in {self.instance_name}")
        return None

    def single_search_helper_with_cache(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Perform a single search with caching."""
        cached_entry = self.get_cached_entry(query, **kwargs)
        if cached_entry:
            return cached_entry
        result = self.single_search_helper(query, max_results, **kwargs)
        if self.caching_enabled:
            cache_key = self.get_cache_key(query, **kwargs)
            self.caching_engine.set(cache_key, result, ttl=None)
            search_logger.debug(
                f"Cached search response for query '{query}' in {self.instance_name}"
            )
        return result

    async def single_search_helper_with_cache_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single search with caching."""
        cached_entry = self.get_cached_entry(query, **kwargs)
        if cached_entry:
            return cached_entry
        result = await self.single_search_helper_async(query, max_results, **kwargs)
        if self.caching_enabled:
            cache_key = self.get_cache_key(query, **kwargs)
            self.caching_engine.set(cache_key, result, ttl=None)
            search_logger.debug(
                f"Cached async search response for query '{query}' in {self.instance_name}"
            )
        return result

    def batch_search_helper(
        self, queries: List[str], max_results: int, **kwargs
    ) -> List[SearchQueryResponse]:
        """Helper function to perform a batch search."""
        results = []
        max_concurrent_search = self.get_max_concurrent_search(default=len(queries))
        for i in range(0, len(queries), max_concurrent_search):
            batch = queries[i : i + max_concurrent_search]
            # Use ThreadPoolExecutor to run single_search_helper in parallel
            with ThreadPoolExecutor(max_workers=max_concurrent_search) as executor:
                futures = {
                    executor.submit(
                        self.single_search_helper_with_cache,
                        query,
                        max_results,
                        **kwargs,
                    ): query
                    for query in batch
                }
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        search_logger.exception(
                            f"Error processing query '{futures[future]}': {e}"
                        )
                        results.append(
                            SearchQueryResponse(
                                query=futures[future],
                                follow_up_queries=None,
                                answer=None,
                                images=None,
                                docs=[],
                            )
                        )
        return results

    async def batch_search_helper_async(
        self, queries: List[str], max_results: int, **kwargs
    ) -> List[SearchQueryResponse]:
        """Asynchronously perform a batch search."""
        results = []
        max_concurrent_search = self.get_max_concurrent_search(default=len(queries))
        for i in range(0, len(queries), max_concurrent_search):
            batch = queries[i : i + max_concurrent_search]
            tasks = [
                self.single_search_helper_with_cache_async(query, max_results, **kwargs)
                for query in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    def search(self, request: SearchServiceRequest) -> SearchServiceResponse:
        queries = (
            request.queries if isinstance(request.queries, list) else [request.queries]
        )
        try:
            kwargs = {k: v for k, v in request.kwargs.items() if k != "max_results"}
            results = self.batch_search_helper(queries, request.max_results, **kwargs)
            for result in results:
                if isinstance(result, SearchQueryResponse):
                    result.docs = self.sort_results(result.docs)
            results = results if isinstance(request.queries, list) else results[0]
            response = SearchServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=request.operation_name,
                raw_data={"results": results},
                results=results,
                success=True,
            )
            return response
        except Exception as e:
            search_logger.exception(f"{self.instance_name} search failed: {e}")
            return SearchServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=request.operation_name,
                raw_data={"error": str(e)},
                results=[],
                success=False,
                error=str(e),
            )

    async def search_async(
        self, request: SearchServiceRequest
    ) -> SearchServiceResponse:
        queries = (
            request.queries if isinstance(request.queries, list) else [request.queries]
        )
        try:
            kwargs = {k: v for k, v in request.kwargs.items() if k != "max_results"}
            results = await self.batch_search_helper_async(
                queries, request.max_results, **kwargs
            )
            for result in results:
                if isinstance(result, SearchQueryResponse):
                    result.docs = self.sort_results(result.docs)
            results = results if isinstance(request.queries, list) else results[0]
            response = SearchServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=request.operation_name,
                raw_data={"results": results},
                results=results,
                success=True,
            )
            return response
        except Exception as e:
            search_logger.exception(f"{self.instance_name} async search failed: {e}")
            return SearchServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=request.operation_name,
                raw_data={"error": str(e)},
                results=[],
                success=False,
                error=str(e),
            )

    def sort_results(self, results: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Sort search results by relevance score."""
        return sorted(
            results,
            key=lambda doc: doc.score if doc.score is not None else 0.0,
            reverse=True,
        )

    def _serve_impl(self, request: SearchServiceRequest) -> SearchServiceResponse:
        search_logger.debug(
            f"SearchServiceInstance {self.instance_name} received request: {request.get_statistics()}"
        )
        response = self.search(request)
        search_logger.debug(
            f"SearchServiceInstance {self.instance_name} returning response: {response.get_statistics()}"
        )
        return response

    async def _serve_async_impl(
        self, request: SearchServiceRequest
    ) -> SearchServiceResponse:
        search_logger.debug(
            f"SearchServiceInstance {self.instance_name} received async request: {request.get_statistics()}"
        )
        response = await self.search_async(request)
        search_logger.debug(
            f"SearchServiceInstance {self.instance_name} returning async response: {response.get_statistics()}"
        )
        return response

    def get_max_concurrent_search(self, default: int = 0) -> int:
        """Get the maximum number of concurrent search requests."""
        max_concurrent_search = getattr(self.config, "max_concurrent_search", None)
        if max_concurrent_search is None:
            return int(os.getenv("MAX_CONCURRENT_SEARCH", default))


# --- Arxiv Search Implementation ---
class ArxivSearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)

    def search_helper_with_arxiv(
        self, query: str, max_results: int, **kwargs
    ) -> List[RetrievedDocument]:
        """Helper function to perform a single actual search using arxiv."""
        import arxiv

        search = arxiv.Search(query=query, max_results=max_results)
        results: List[RetrievedDocument] = []
        for result in search.results():
            results.append(
                RetrievedDocument(
                    title=result.title,
                    url=result.entry_id,
                    content=result.summary,
                    score=None,  # arxiv doesn't provide a score, set to None
                    raw_content=None,  # arxiv doesn't provide raw content
                )
            )
        return results

    def search_helper_with_langchain(
        self, query: str, max_results: int, **kwargs
    ) -> List[RetrievedDocument]:
        """Helper function to perform a single actual search using ArxivRetriever."""
        from langchain_community.retrievers import ArxivRetriever

        get_full_documents = kwargs.get("get_full_documents", True)
        load_all_available_meta = kwargs.get("load_all_available_meta", True)
        retriever = ArxivRetriever(
            load_max_docs=max_results,
            get_full_documents=get_full_documents,
            load_all_available_meta=load_all_available_meta,
        )
        langchain_docs = retriever.get_relevant_documents(query)
        results = []
        # Assign decreasing scores based on the order
        base_score = 1.0
        score_decrement = 1.0 / (len(langchain_docs) + 1) if langchain_docs else 0

        for i, doc in enumerate(langchain_docs):
            # Extract metadata
            metadata = doc.metadata

            # Use entry_id as the URL (this is the actual arxiv link)
            url = metadata.get("entry_id", "")

            # Format content with all useful metadata
            content_parts = []

            # Primary information
            if "Summary" in metadata:
                content_parts.append(f"Summary: {metadata['Summary']}")

            if "Authors" in metadata:
                content_parts.append(f"Authors: {metadata['Authors']}")

            # Add publication information
            published = metadata.get("Published")
            published_str = (
                published.isoformat()
                if hasattr(published, "isoformat")
                else str(published) if published else ""
            )
            if published_str:
                content_parts.append(f"Published: {published_str}")

            # Add additional metadata if available
            if "primary_category" in metadata:
                content_parts.append(
                    f"Primary Category: {metadata['primary_category']}"
                )

            if "categories" in metadata and metadata["categories"]:
                content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

            if "comment" in metadata and metadata["comment"]:
                content_parts.append(f"Comment: {metadata['comment']}")

            if "journal_ref" in metadata and metadata["journal_ref"]:
                content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

            if "doi" in metadata and metadata["doi"]:
                content_parts.append(f"DOI: {metadata['doi']}")

            # Get PDF link if available in the links
            pdf_link = ""
            if "links" in metadata and metadata["links"]:
                for link in metadata["links"]:
                    if "pdf" in link:
                        pdf_link = link
                        content_parts.append(f"PDF: {pdf_link}")
                        break

            # Join all content parts with newlines
            content = "\n".join(content_parts)

            result = RetrievedDocument(
                title=metadata.get("Title", ""),
                url=url,  # Using entry_id as the URL
                content=content,
                score=base_score - (i * score_decrement),
                raw_content=doc.page_content if get_full_documents else None,
            )
            results.append(result)
        return results

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search."""
        st = time.time()
        results = self.search_helper_with_arxiv(query, max_results, **kwargs)
        et = time.time()
        elapsed_time = et - st
        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,  # arxiv doesn't provide follow-up queries
            answer=None,  # arxiv doesn't provide a direct answer
            images=None,  # arxiv doesn't provide images
            docs=results,
            elapsed_time=elapsed_time,
            extra={
                "source": "arxiv",
                "waiting_cost": elapsed_time,
            },
        )
        search_logger.debug(
            f"{self.instance_name} search results for query '{query}': {resp.get_statistics()}"
        )
        return resp


# --- Exa Search Implementation ---
class ExaSearchServiceInstance(SearchServiceInstance):
    """A search service instance for ExaSearch."""

    from exa_py.api import CostDollars, SearchResponse, ResultWithTextAndSummary

    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)
        self.api_key = config.api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ExaSearch API key not set in config or EXA_API_KEY env var."
            )

        from exa_py import Exa, AsyncExa

        self.client = Exa(api_key=self.api_key)
        self.async_client = AsyncExa(api_key=self.api_key)

    @classmethod
    def cost_dollars_to_dict(cls, cost: CostDollars) -> dict:
        if cost and isinstance(cost, cls.CostDollars):
            return {
                "total": cost.total,
                **({"search": cost.search} if cost.search is not None else {}),
                **({"contents": cost.contents} if cost.contents is not None else {}),
            }
        return {}

    def get_docs(
        self, data: SearchResponse[ResultWithTextAndSummary], **kwargs
    ) -> List[RetrievedDocument]:
        # Map the result to SearchQueryResponse
        docs = []
        seen_urls = set()  # Track URLs to avoid duplicates
        for result in data.results:
            if result.url in seen_urls:
                continue
            content = result.text or ""
            summary = result.summary or ""
            if summary:
                if content:
                    content = f"{summary}\n\n{content}"
                else:
                    content = summary
            docs.append(
                RetrievedDocument(
                    title=result.title,
                    url=result.url,
                    content=content,
                    score=result.score,
                    raw_content=result.text,
                )
            )
            seen_urls.add(result.url)

        # Helper function to safely get value regardless of if item is dict or object
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default

        if kwargs.get("subpages", None) is not None:
            for result in data.results:
                subpages_list = get_value(result, "subpages", [])
                for subpage in subpages_list:
                    # Get subpage score
                    subpage_score = get_value(subpage, "score", 0.0)

                    # Combine summary and text for subpage content
                    subpage_text = get_value(subpage, "text", "")
                    subpage_summary = get_value(subpage, "summary", "")

                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary

                    subpage_url = get_value(subpage, "url", "")

                    # Skip if we've seen this URL before
                    if subpage_url in seen_urls:
                        continue

                    seen_urls.add(subpage_url)

                    docs.append(
                        RetrievedDocument(
                            title=get_value(subpage, "title", ""),
                            url=subpage_url,
                            content=subpage_content,
                            score=subpage_score,
                            raw_content=subpage_text,
                        )
                    )
        return docs

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using ExaSearch."""
        st = time.time()
        data = self.client.search_and_contents(
            query=query,
            text=(
                kwargs.get("text", True)
                if kwargs.get("max_characters", None) is None
                else {"max_characters": kwargs["max_characters"]}
            ),
            num_results=max_results,
            summary=kwargs.get("summary", True),
            type=kwargs.get("type", "auto"),
            include_domains=kwargs.get("include_domains", None),
            exclude_domains=kwargs.get("exclude_domains", None),
            subpages=kwargs.get("subpages", None),
        )
        elapsed_time = time.time() - st

        docs = self.get_docs(data, **kwargs)

        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=[result.image for result in data.results if result.image],
            docs=docs,
            elapsed_time=elapsed_time,
            extra={
                "source": "exa",
                "autoprompt_string": data.autoprompt_string,
                "resolved_search_type": data.resolved_search_type,
                "auto_date": data.auto_date,
                "context": data.context,
                "statuses": data.statuses,
                "cost_dollars": ExaSearchServiceInstance.cost_dollars_to_dict(
                    data.cost_dollars
                ),
                "waiting_cost": elapsed_time,
            },
        )

        search_logger.debug(
            f"{self.instance_name} search results for query '{query}': {resp.get_statistics()}"
        )
        return resp

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using ExaSearch."""
        st = time.time()
        data = await self.async_client.search_and_contents(
            query=query,
            text=(
                kwargs.get("text", True)
                if kwargs.get("max_characters", None) is None
                else {"max_characters": kwargs["max_characters"]}
            ),
            num_results=max_results,
            summary=kwargs.get("summary", True),
            type=kwargs.get("type", "auto"),
            include_domains=kwargs.get("include_domains", None),
            exclude_domains=kwargs.get("exclude_domains", None),
            subpages=kwargs.get("subpages", None),
        )
        elapsed_time = time.time() - st

        docs = self.get_docs(data, **kwargs)

        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=[result.image for result in data.results if result.image],
            docs=docs,
            elapsed_time=elapsed_time,
            extra={
                "source": "exa",
                "autoprompt_string": data.autoprompt_string,
                "resolved_search_type": data.resolved_search_type,
                "auto_date": data.auto_date,
                "context": data.context,
                "statuses": data.statuses,
                "cost_dollars": ExaSearchServiceInstance.cost_dollars_to_dict(
                    data.cost_dollars
                ),
                "waiting_cost": elapsed_time,
            },
        )

        search_logger.debug(
            f"{self.instance_name} async search results for query '{query}': {resp.get_statistics()}"
        )
        return resp


# --- Dummy Search Implementation ---
class DummySearchServiceInstance(SearchServiceInstance):
    """A dummy search service instance for testing purposes."""

    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Return a dummy search response."""
        search_logger.debug(
            f"Dummy search for query: {query} with max_results: {max_results}"
        )
        st = time.time()
        # Check if we have a cached entry
        cached_entry: SearchQueryResponse = self.get_cached_entry(query, **kwargs)
        if cached_entry:
            elapsed_time = time.time() - st
            cached_entry.elapsed_time = elapsed_time
            cached_entry.from_cache = {
                "cache_key": self.get_cache_key(query, **kwargs),
                "cache_engine": self.caching_engine.instance_name,
            }
            search_logger.debug(
                f"Returning cached entry for query '{query}' in {self.instance_name}"
            )
            return cached_entry

        # If no cache hit, return dummy results
        # Simulate a delay if specified in kwargs
        sleeping_time = kwargs.get("sleep", 0.0)
        if sleeping_time > 0:
            search_logger.info(
                f"Sleeping for {sleeping_time} seconds before returning dummy results."
            )
            time.sleep(sleeping_time)
        results = [
            RetrievedDocument(
                title=f"Fake Document for {query} {i}",
                url=f"http://example.com/{hash(query)}/doc{i}",
                content="Dummy",
                score=1.0,
                raw_content="This is a dummy document content for testing purposes.",
            )
            for i in range(max_results)
        ]
        et = time.time()
        elapsed_time = et - st
        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=None,
            docs=results,
            elapsed_time=elapsed_time,
            extra={
                "source": "dummy",
                "waiting_cost": elapsed_time,
            },
        )
        # Cache the response if caching is enabled
        if self.caching_enabled:
            cache_key = self.get_cache_key(query, **kwargs)
            self.caching_engine.set(cache_key, resp, ttl=None)
            search_logger.debug(
                f"Cached dummy search response for query '{query}' in {self.instance_name}"
            )
        # Log the dummy response
        search_logger.debug(
            f"Dummy search results for query '{query}': {resp.get_statistics()}"
        )
        return resp


# --- Tavily Search Implementation ---
class TavilySearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)
        self.api_key = config.api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key not set in config or TAVILY_API_KEY env var."
            )

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using Tavily."""
        st = time.time()
        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": max_results,
            "time_range": None,
            "days": 7,
            "include_answer": True,
            "include_raw_content": True,
            "include_images": False,
            "include_image_descriptions": False,
            "include_domains": [],
            "exclude_domains": [],
            "country": None,
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        et = time.time()
        elapsed_time = et - st
        if response.status_code == 200:
            data = response.json()
            results = [
                RetrievedDocument(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    score=result.get("score", None),
                    raw_content=result.get("raw_content", None),
                )
                for result in data.get("results", [])
            ]
            resp = SearchQueryResponse(
                query=query,
                follow_up_queries=data.get("follow_up_queries", []),
                answer=data.get("answer"),
                images=data.get("images", []),
                docs=results,
                elapsed_time=data.get("response_time", elapsed_time),
                extra={
                    "source": "tavily",
                    "waiting_cost": elapsed_time,
                },
            )
            search_logger.debug(
                f"Tavily search results for query '{query}': {resp.get_statistics()}"
            )
            return resp
        else:
            search_logger.error(
                f"Failed to fetch Tavily search results: {response.text}"
            )
            return SearchQueryResponse(
                query=query,
                follow_up_queries=None,
                answer=None,
                images=None,
                docs=[],
                elapsed_time=elapsed_time,
            )

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using Tavily."""
        st = time.time()
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": max_results,
            "time_range": None,
            "days": 7,
            "include_answer": True,
            "include_raw_content": True,
            "include_images": False,
            "include_image_descriptions": False,
            "include_domains": [],
            "exclude_domains": [],
            "country": None,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    search_logger.error(f"Tavily search failed: {text}")
                    return SearchQueryResponse(
                        query=query,
                        follow_up_queries=None,
                        answer=None,
                        images=None,
                        docs=[],
                    )
                data = await resp.json()
                results = [
                    RetrievedDocument(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=result.get("content", ""),
                        score=result.get("score", None),
                        raw_content=result.get("raw_content", None),
                    )
                    for result in data.get("results", [])
                ]
                et = time.time()
                elapsed_time = et - st
                resp = SearchQueryResponse(
                    query=query,
                    follow_up_queries=data.get("follow_up_queries", []),
                    answer=data.get("answer"),
                    images=data.get("images", []),
                    docs=results,
                    elapsed_time=data.get("response_time", elapsed_time),
                    extra={
                        "source": "tavily",
                        "waiting_cost": elapsed_time,
                    },
                )
                search_logger.debug(
                    f"Tavily async search results for query '{query}': {resp.get_statistics()}"
                )

                return resp


# --- DuckDuckGo Search Implementation ---
class DuckDuckGoSearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)

    def perform_search(self, query: str, max_results: int, **kwargs) -> Dict[str, Any]:
        from duckduckgo_search import DDGS
        import time
        import random

        max_retries = 3
        retry_count = 0
        backoff_factor = 2.0
        last_exception = None

        while retry_count <= max_retries:
            try:
                results = []
                with DDGS() as ddgs:
                    # Change query slightly and add delay between retries
                    if retry_count > 0:
                        # Random delay with exponential backoff
                        delay = backoff_factor**retry_count + random.random()
                        search_logger.warning(
                            f"Retry {retry_count}/{max_retries} for query '{query}' after {delay:.2f}s delay"
                        )
                        time.sleep(delay)

                        # Add a random element to the query to bypass caching/rate limits
                        modifiers = [
                            "about",
                            "info",
                            "guide",
                            "overview",
                            "details",
                            "explained",
                        ]
                        modified_query = f"{query} {random.choice(modifiers)}"
                    else:
                        modified_query = query

                    # Execute search
                    ddg_results = list(
                        ddgs.text(modified_query, max_results=max_results)
                    )

                    # Format results
                    for i, result in enumerate(ddg_results):
                        results.append(
                            {
                                "title": result.get("title", ""),
                                "url": result.get("href", ""),
                                "content": result.get("body", ""),
                                "score": 1.0 - (i * 0.1),  # Simple scoring mechanism
                                "raw_content": result.get("body", ""),
                            }
                        )

                    # Return successful results
                    return {
                        "query": query,
                        "follow_up_questions": None,
                        "answer": None,
                        "images": [],
                        "results": results,
                    }
            except Exception as e:
                # Store the exception and retry
                last_exception = e
                retry_count += 1
                search_logger.exception(
                    f"DuckDuckGo search error: {str(e)}. Retrying {retry_count}/{max_retries}"
                )

                # If not a rate limit error, don't retry
                if "Ratelimit" not in str(e) and retry_count >= 1:
                    search_logger.error(
                        f"Non-rate limit error, stopping retries: {str(e)}"
                    )
                    break

        # If we reach here, all retries failed
        search_logger.error(
            f"All retries failed for query '{query}': {str(last_exception)}"
        )
        # Return empty results but with query info preserved
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": [],
            "error": str(last_exception),
        }

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using DuckDuckGo."""
        st = time.time()
        data = self.perform_search(query, max_results, **kwargs)
        et = time.time()
        elapsed_time = et - st
        resp = SearchQueryResponse(
            query=data["query"],
            follow_up_queries=data.get("follow_up_questions"),
            answer=data.get("answer"),
            images=data.get("images", []),
            docs=[
                RetrievedDocument(
                    title=result["title"],
                    url=result["url"],
                    content=result["content"],
                    score=result["score"],
                    raw_content=result.get("raw_content"),
                )
                for result in data["results"]
            ],
            elapsed_time=elapsed_time,
            extra={
                "source": "duckduckgo",
                "error": data.get("error"),
            },
        )
        if "error" in data:
            search_logger.error(
                f"DuckDuckGo search failed for query '{query}': {data['error']}"
            )
        else:
            search_logger.debug(
                f"DuckDuckGo search results for query '{query}': {resp.get_statistics()}"
            )

        return resp


# --- Wikipedia Search Implementation ---
class WikiSearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using Wikipedia."""
        import wikipedia

        st = time.time()
        results = []
        try:
            page_titles = wikipedia.search(query, results=max_results)
            for title in page_titles:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    results.append(
                        RetrievedDocument(
                            title=page.title,
                            url=page.url,
                            content=page.summary,
                            score=None,  # Wikipedia doesn't provide a score, set to None
                            raw_content=None,  # No raw content available
                        )
                    )
                except Exception as e:
                    search_logger.warning(f"Wiki page fetch failed for '{title}': {e}")
        except Exception as e:
            search_logger.exception(f"Wikipedia search failed for query '{query}': {e}")

        et = time.time()
        elapsed_time = et - st
        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,  # Wikipedia doesn't provide follow-up queries
            answer=None,  # Wikipedia doesn't provide a direct answer
            images=None,  # Wikipedia doesn't provide images
            docs=results,
            elapsed_time=elapsed_time,
            extra={
                "source": "wikipedia",
                "waiting_cost": elapsed_time,
            },
        )
        search_logger.debug(
            f"Wikipedia search results for query '{query}': {resp.get_statistics()}"
        )

        return resp


# --- Linkup Search Implementation ---
class LinkupSearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)
        self.api_key = config.api_key or os.getenv("LINKUP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Linkup API key not set in config or LINKUP_API_KEY env var."
            )

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using Linkup."""
        from linkup import LinkupClient

        st = time.time()
        client = LinkupClient(api_key=self.api_key)
        response = client.search(
            query=query,
            depth=kwargs.get("depth", "standard"),
            output_type=kwargs.get("output_type", "searchResults"),
        )
        docs = []
        images = []
        for result in response.results:
            if result.type == "text":
                docs.append(
                    RetrievedDocument(
                        title=result.name,
                        url=result.url,
                        content=result.content,
                        score=None,  # Linkup doesn't provide a score, set to None
                        raw_content=None,  # No raw content available
                    )
                )
            elif result.type == "image":
                images.append(result.url)
        et = time.time()
        elapsed_time = et - st

        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=images,
            docs=docs,
            elapsed_time=elapsed_time,
            extra={
                "source": "linkup",
                "waiting_cost": elapsed_time,
                "raw_results": response.results,
            },
        )
        search_logger.debug(
            f"Linkup search results for query '{query}': {resp.get_statistics()}"
        )

        return resp

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using Linkup."""
        from linkup import LinkupClient

        st = time.time()
        client = LinkupClient(api_key=self.api_key)
        response = await client.async_search(
            query=query,
            depth=kwargs.get("depth", "standard"),
            output_type=kwargs.get("output_type", "searchResults"),
        )
        docs = []
        images = []
        for result in response.results:
            if result.type == "text":
                docs.append(
                    RetrievedDocument(
                        title=result.name,
                        url=result.url,
                        content=result.content,
                        score=None,  # Linkup doesn't provide a score, set to None
                        raw_content=None,  # No raw content available
                    )
                )
            elif result.type == "image":
                images.append(result.url)
        et = time.time()
        elapsed_time = et - st

        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=images,
            docs=docs,
            elapsed_time=elapsed_time,
            extra={
                "source": "linkup",
                "waiting_cost": elapsed_time,
            },
        )
        search_logger.debug(
            f"Linkup async search results for query '{query}': {resp.get_statistics()}"
        )

        return resp


# --- Serper Search Implementation ---
class SerperSearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)
        self.api_key = config.api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Serper API key not set in config or SERPER_API_KEY env var."
            )

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using Serper."""
        url = "https://google.serper.dev/search"
        st = time.time()
        payload = json.dumps(
            {"q": query, "num": max_results, "page": kwargs.get("page", 1)}
        )
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        if response.status_code != 200:
            search_logger.error(f"Serper search failed: {response.text}")
            return SearchQueryResponse(
                query=query,
                follow_up_queries=None,
                answer=None,
                images=None,
                docs=[],
                elapsed_time=time.time() - st,
            )
        data = response.json()
        search_params = data.get("search_parameters", {})
        knowledgeGraph = data.get("knowledgeGraph", {})
        peopleAlsoAsk = data.get("peopleAlsoAsk", [])
        relatedSearches = data.get("relatedSearches", [])
        results = []
        for result in data.get("organic", []):
            results.append(
                RetrievedDocument(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    content=result.get("snippet", ""),
                    score=result.get("score", None),
                    raw_content=result.get("snippet", None),
                    extra={
                        "sitelinks": result.get("sitelinks", []),
                        "position": result.get("position", None),
                        "attributes": result.get("attributes", {}),
                        "date": result.get("date", None),
                    },
                )
            )

        et = time.time()
        elapsed_time = et - st
        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=data.get("follow_up_queries", []),
            answer=data.get("answer"),
            images=data.get("images", []),
            docs=results,
            elapsed_time=data.get("elapsed_time", elapsed_time),
            extra={
                "source": "serper",
                "waiting_cost": elapsed_time,
                "search_parameters": search_params,
                "knowledgeGraph": knowledgeGraph,
                "peopleAlsoAsk": peopleAlsoAsk,
                "relatedSearches": relatedSearches,
                "raw_results": data.get("organic", []),
            },
        )
        search_logger.debug(
            f"Serper search results for query '{query}': {resp.get_statistics()}"
        )

        return resp

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using Serper."""
        url = "https://google.serper.dev/search"
        st = time.time()
        payload = json.dumps(
            {"q": query, "num": max_results, "page": kwargs.get("page", 1)}
        )
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    search_logger.error(f"Serper search failed: {text}")
                    return SearchQueryResponse(
                        query=query,
                        follow_up_queries=None,
                        answer=None,
                        images=None,
                        docs=[],
                        elapsed_time=time.time() - st,
                    )
                data = await resp.json()
                search_params = data.get("search_parameters", {})
                knowledgeGraph = data.get("knowledgeGraph", {})
                peopleAlsoAsk = data.get("peopleAlsoAsk", [])
                relatedSearches = data.get("relatedSearches", [])
                results = []
                for result in data.get("organic", []):
                    results.append(
                        RetrievedDocument(
                            title=result.get("title", ""),
                            url=result.get("link", ""),
                            content=result.get("snippet", ""),
                            score=result.get("score", None),
                            raw_content=result.get("snippet", None),
                            extra={
                                "sitelinks": result.get("sitelinks", []),
                                "position": result.get("position", None),
                                "attributes": result.get("attributes", {}),
                                "date": result.get("date", None),
                            },
                        )
                    )

                et = time.time()
                elapsed_time = et - st
                resp = SearchQueryResponse(
                    query=query,
                    follow_up_queries=data.get("follow_up_queries", []),
                    answer=data.get("answer"),
                    images=data.get("images", []),
                    docs=results,
                    elapsed_time=data.get("elapsed_time", elapsed_time),
                    extra={
                        "source": "serper",
                        "waiting_cost": elapsed_time,
                        "search_parameters": search_params,
                        "knowledgeGraph": knowledgeGraph,
                        "peopleAlsoAsk": peopleAlsoAsk,
                        "relatedSearches": relatedSearches,
                        "raw_results": data.get("organic", []),
                    },
                )
                search_logger.debug(
                    f"Serper search results for query '{query}': {resp.get_statistics()}"
                )

                return resp

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using Serper."""
        url = "https://google.serper.dev/search"
        st = time.time()
        payload = json.dumps(
            {"q": query, "num": max_results, "page": kwargs.get("page", 1)}
        )
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    search_logger.error(f"Serper search failed: {text}")
                    return SearchQueryResponse(
                        query=query,
                        follow_up_queries=None,
                        answer=None,
                        images=None,
                        docs=[],
                        elapsed_time=time.time() - st,
                    )
                data = await resp.json()
                search_params = data.get("search_parameters", {})
                knowledgeGraph = data.get("knowledgeGraph", {})
                peopleAlsoAsk = data.get("peopleAlsoAsk", [])
                relatedSearches = data.get("relatedSearches", [])
                results = []
                for result in data.get("organic", []):
                    results.append(
                        RetrievedDocument(
                            title=result.get("title", ""),
                            url=result.get("link", ""),
                            content=result.get("snippet", ""),
                            score=result.get("score", None),
                            raw_content=result.get("snippet", None),
                            extra={
                                "sitelinks": result.get("sitelinks", []),
                                "position": result.get("position", None),
                                "attributes": result.get("attributes", {}),
                                "date": result.get("date", None),
                            },
                        )
                    )

                et = time.time()
                elapsed_time = et - st
                resp = SearchQueryResponse(
                    query=query,
                    follow_up_queries=data.get("follow_up_queries", []),
                    answer=data.get("answer"),
                    images=data.get("images", []),
                    docs=results,
                    elapsed_time=data.get("elapsed_time", elapsed_time),
                    extra={
                        "source": "serper",
                        "waiting_cost": elapsed_time,
                        "search_parameters": search_params,
                        "knowledgeGraph": knowledgeGraph,
                        "peopleAlsoAsk": peopleAlsoAsk,
                        "relatedSearches": relatedSearches,
                        "raw_results": data.get("organic", []),
                    },
                )
                search_logger.debug(
                    f"Serper search results for query '{query}': {resp.get_statistics()}"
                )

                return resp


# --- SearxNG Search Implementation ---
class SearxNGSearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)
        self.api_key = config.api_key or os.getenv("SEARXNG_API_KEY")
        self.api_url = config.location or os.getenv("SEARXNG_API_URL")
        self.auto_scraping: bool = config.kwargs.get("auto_scraping", False)
        self.scraping_kwargs: Dict = config.kwargs.get("scraping_kwargs", {})
        search_logger.debug(
            f"Initializing SearxNGSearchServiceInstance with API URL: {self.api_url}, auto_scraping: {self.auto_scraping} and scraping_kwargs: {self.scraping_kwargs}"
        )
        if "://" not in self.api_url:
            # Assume it's an IP:port format
            self.api_url = f"http://{self.api_url}"
        if not self.api_url:
            raise ValueError(
                "SearxNG API URL not set in config or SEARXNG_API_URL env var."
            )

        self.k = config.kwargs.get("k", 10)
        try:
            from langchain_community.utilities import SearxSearchWrapper

            self.client = SearxSearchWrapper(
                searx_host=self.api_url,
                headers={"Accept": "application/json"},
                k=self.k,
            )
        except ImportError:
            search_logger.exception(
                "SearxNG search requires langchain_community package. Please install it."
            )
            raise ImportError(
                "langchain_community package is required for SearxNG search."
            )
        except Exception as e:
            search_logger.exception(
                f"Failed to initialize SearxNG client: {str(e)}. Check your configuration."
            )
            raise e

    def scrape_helper(self, results: List[Dict], **kwargs) -> List[Dict]:
        scrape_engine = kwargs.get("engine", "firecrawl") or self.scraping_kwargs.get(
            "engine", "requests"
        )
        if scrape_engine == "firecrawl":
            pages = scrap_with_firecrawl(
                urls=[r.get("link", "") for r in results],
                api_url=kwargs.get("api_url", None)
                or self.scraping_kwargs.get("api_url"),
            )
        elif scrape_engine == "requests":
            extra_kwargs = {
                "max_workers": kwargs.get("max_workers", None)
                or self.scraping_kwargs.get("max_workers", None),
                "timeout": kwargs.get("timeout", None)
                or self.scraping_kwargs.get("timeout", None),
                "max_retries": kwargs.get("max_retries", None)
                or self.scraping_kwargs.get("max_retries", None),
            }
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            search_logger.debug(
                f"Using requests engine with extra kwargs: {extra_kwargs}"
            )
            pages = scrape_with_requests(
                urls=[r.get("link", "") for r in results], **extra_kwargs
            )
        elif scrape_engine == "markitdown":
            pages = scrape_pages_with_markitdown(
                urls=[r.get("link", "") for r in results],
                max_workers=kwargs.get("max_workers", None)
                or self.scraping_kwargs.get("max_workers", 32),
            )
        else:
            search_logger.error(
                f"Unsupported scraping engine: {scrape_engine}. Defaulting to requests."
            )
            pages = scrape_with_requests(
                urls=[r.get("link", "") for r in results],
            )
        assert len(pages) == len(
            results
        ), f"Incorrect number of scraped pages returned. {len(pages)} != {len(results)}"
        for i, page in enumerate(pages):
            if page.startswith("Error"):
                results[i]["raw_content"] = results[i]["snippet"]
            elif page.startswith("Content type: "):
                results[i]["raw_content"] = results[i]["snippet"]
            else:
                results[i]["raw_content"] = page  # store scraped content
        return results

    async def scrape_helper_async(self, results: List[Dict], **kwargs) -> List[Dict]:
        scrape_engine = kwargs.get("engine", "firecrawl") or self.scraping_kwargs.get(
            "engine", "requests"
        )
        if scrape_engine == "firecrawl":
            pages = await scrap_with_firecrawl_async(
                urls=[r.get("link", "") for r in results],
                api_url=kwargs.get("api_url", None)
                or self.scraping_kwargs.get("api_url"),
            )
        elif scrape_engine == "requests":
            extra_kwargs = {
                "max_workers": kwargs.get("max_workers", None)
                or self.scraping_kwargs.get("max_workers", None),
                "timeout": kwargs.get("timeout", None)
                or self.scraping_kwargs.get("timeout", None),
                "max_retries": kwargs.get("max_retries", None)
                or self.scraping_kwargs.get("max_retries", None),
            }
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
            search_logger.debug(
                f"Using requests engine with extra kwargs: {extra_kwargs}"
            )
            pages = await scrape_with_requests_async(
                urls=[r.get("link", "") for r in results],
                **extra_kwargs,
            )
        elif scrape_engine == "markitdown":
            pages = await scrape_pages_with_markitdown_async(
                urls=[r.get("link", "") for r in results],
                max_workers=kwargs.get("max_workers", None)
                or self.scraping_kwargs.get("max_workers", 32),
            )
        else:
            search_logger.error(
                f"Unsupported scraping engine: {scrape_engine}. Defaulting to requests."
            )
            pages = await scrape_with_requests_async(
                urls=[r.get("link", "") for r in results],
            )
        assert len(pages) == len(
            results
        ), f"Incorrect number of scraped pages returned. {len(pages)} != {len(results)}"
        for i, page in enumerate(pages):
            if page.startswith("Error"):
                results[i]["raw_content"] = results[i]["snippet"]
            elif page.startswith("Content type: "):
                results[i]["raw_content"] = results[i]["snippet"]
            else:
                results[i]["raw_content"] = page  # store scraped content
        return results

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using SearxNG."""
        search_logger.debug(
            f"Performing SearxNG search for query: {query} with max_results: {max_results} and kwargs: {kwargs}"
        )
        st = time.time()
        results = self.client.results(
            query=query,
            num_results=max_results,
            engines=kwargs.get("engines", None),
            categories=kwargs.get("categories", None),
            time_range=kwargs.get("time_range", None),
            query_suffix=kwargs.get("query_suffix", None),
        )
        if not results:
            search_logger.error(
                f"SearxNG search returned no results for query '{query}'"
            )
            return SearchQueryResponse(
                query=query,
                follow_up_queries=None,
                answer=None,
                images=None,
                docs=[],
                elapsed_time=time.time() - st,
            )
        # scrape raw content if auto_scraping is enabled
        if kwargs.get("auto_scraping", self.auto_scraping):
            search_logger.debug(
                f"Auto-scraping enabled, scraping {len(results)} results for query '{query}'"
            )
            results = self.scrape_helper(results, **kwargs.get("scraping_kwargs", {}))
        docs = []
        for result in results:
            docs.append(
                RetrievedDocument(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    content=result.get("snippet", ""),
                    score=result.get("score", None),
                    raw_content=result.get("raw_content", None),
                    extra={
                        "engines": result.get("engines", []),
                        "category": result.get("category", None),
                        "raw_results": result,
                    },
                )
            )
        et = time.time()
        elapsed_time = et - st
        search_logger.debug(
            f"SearxNG search completed for query '{query}' in {elapsed_time:.2f} seconds with {len(docs)} results."
        )
        return SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=None,
            docs=docs,
            elapsed_time=elapsed_time,
            extra={
                "source": "searxng",
                "waiting_cost": elapsed_time,
            },
        )

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using SearxNG."""
        search_logger.debug(
            f"Performing async SearxNG search for query: {query} with max_results: {max_results} and kwargs: {kwargs}"
        )

        st = time.time()
        # using request to get results
        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "num_results": max_results,
                "format": "json",
                # "engines": kwargs.get("engines", None),
                # "categories": kwargs.get("categories", None),
                # "time_range": kwargs.get("time_range", None),
                # "query_suffix": kwargs.get("query_suffix", None),
            }
            if kwargs.get("query_suffix", None) and len(kwargs["query_suffix"]) > 0:
                params["q"] += " " + kwargs["query_suffix"]
            if kwargs.get("engines", None):
                params["engines"] = ",".join(kwargs["engines"])
            if kwargs.get("categories", None):
                params["categories"] = ",".join(kwargs["categories"])
            if kwargs.get("time_range", None):
                params["time_range"] = kwargs["time_range"]
            headers = {"Accept": "application/json"}
            search_logger.debug(
                f"SearxNG async search params: {params} and headers: {headers}"
            )
            async with session.get(
                f"{self.api_url}/search",
                params=params,
                headers=headers,
            ) as resp:
                # Verify content type before parsing
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    error_text = await resp.text()
                    search_logger.error(
                        f"SearxNG async search returned non-JSON response: {error_text}"
                    )
                    return SearchQueryResponse(
                        query=query,
                        follow_up_queries=None,
                        answer=None,
                        images=None,
                        docs=[],
                        elapsed_time=time.time() - st,
                    )
                if resp.status != 200:
                    text = await resp.text()
                    search_logger.error(f"SearxNG async search failed: {text}")
                    return SearchQueryResponse(
                        query=query,
                        follow_up_queries=None,
                        answer=None,
                        images=None,
                        docs=[],
                        elapsed_time=time.time() - st,
                    )
                data = await resp.json()  # may error
                results = [
                    {
                        "snippet": result.get("content", ""),
                        "title": result["title"],
                        "link": result["url"],
                        "engines": result["engines"],
                        "category": result["category"],
                    }
                    for result in data.get("results", [])[:max_results]
                ]
        if not results:
            search_logger.error(
                f"SearxNG async search returned no results for query '{query}'"
            )
            return SearchQueryResponse(
                query=query,
                follow_up_queries=None,
                answer=None,
                images=None,
                docs=[],
                elapsed_time=time.time() - st,
            )
        # scrape raw content if auto_scraping is enabled
        if kwargs.get("auto_scraping", self.auto_scraping):
            search_logger.debug(
                f"Auto-scraping enabled, scraping {len(results)} results for query '{query}'"
            )
            results = await self.scrape_helper_async(
                results, **kwargs.get("scraping_kwargs", {})
            )
        docs = []
        for result in results:
            docs.append(
                RetrievedDocument(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    content=result.get("snippet", ""),
                    score=result.get("score", None),
                    raw_content=result.get("raw_content", None),
                    extra={
                        "engines": result.get("engines", []),
                        "category": result.get("category", None),
                        "raw_results": result,
                    },
                )
            )
        et = time.time()
        elapsed_time = et - st
        search_logger.debug(
            f"SearxNG async search completed for query '{query}' in {elapsed_time:.2f} seconds with {len(docs)} results."
        )
        return SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=None,
            docs=docs,
            elapsed_time=elapsed_time,
            extra={
                "source": "searxng",
                "waiting_cost": elapsed_time,
            },
        )


class FirecrawlSearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)
        self.api_key = config.api_key or os.getenv("FIRECRAWL_API_KEY")
        self.api_url = config.location or os.getenv("FIRECRAWL_API_URL")
        if not self.api_url:
            raise ValueError(
                "Firecrawl API URL not set in config or FIRECRAWL_API_URL env var."
            )

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using Firecrawl."""
        from firecrawl.firecrawl import FirecrawlApp, ScrapeOptions

        st = time.time()
        client = FirecrawlApp(api_url=self.api_url)
        content_format = kwargs.get("target_content_format", "markdown")
        response = client.search(
            query=query,
            limit=max_results,
            scrape_options=ScrapeOptions(formats=[content_format, "links"]),
        )
        if not response.success:
            search_logger.error(
                f"Firecrawl search failed for query '{query}': {response.error}"
            )
            return SearchQueryResponse(
                query=query,
                follow_up_queries=None,
                answer=None,
                images=None,
                docs=[],
                elapsed_time=time.time() - st,
            )
        docs = []
        for result in response.data:
            docs.append(
                RetrievedDocument(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("description", ""),
                    score=None,  # Firecrawl doesn't provide a score, set to None
                    raw_content=result.get(
                        content_format, ""
                    ),  # No raw content available
                    extra={
                        "raw_results": result,
                        "content_format": content_format,
                    },
                )
            )
        et = time.time()
        elapsed_time = et - st

        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=[],
            docs=docs,
            elapsed_time=elapsed_time,
            extra={
                "source": "firecrawl",
                "waiting_cost": elapsed_time,
                "raw_results": response.data,
            },
        )
        search_logger.debug(
            f"Firecrawl search completed for query '{query}' in {elapsed_time:.2f} seconds with {len(docs)} results."
        )

        return resp

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using Firecrawl."""
        from firecrawl.firecrawl import AsyncFirecrawlApp, ScrapeOptions, SearchResponse

        st = time.time()
        client = AsyncFirecrawlApp(api_url=self.api_url)
        content_format = kwargs.get("target_content_format", "markdown")
        response_json: Dict = await client.search(
            query=query,
            limit=max_results,
            scrape_options=ScrapeOptions(formats=[content_format, "links"]),
        )
        if response_json.get("success") and "data" in response_json:
            response = SearchResponse(**response_json)
        else:
            response = SearchResponse(
                success=False,
                error=response_json.get("error", "Unknown error"),
                data=[],
            )
        if not response.success:
            search_logger.error(
                f"Firecrawl async search failed for query '{query}': {response.error}"
            )
            return SearchQueryResponse(
                query=query,
                follow_up_queries=None,
                answer=None,
                images=None,
                docs=[],
                elapsed_time=time.time() - st,
            )
        docs = []
        for result in response.data:
            docs.append(
                RetrievedDocument(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("description", ""),
                    score=None,  # Firecrawl doesn't provide a score, set to None
                    raw_content=result.get(
                        content_format, ""
                    ),  # No raw content available
                    extra={
                        "raw_results": result,
                        "content_format": content_format,
                    },
                )
            )
        et = time.time()
        elapsed_time = et - st
        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=None,
            answer=None,
            images=[],
            docs=docs,
            elapsed_time=elapsed_time,
            extra={
                "source": "firecrawl",
                "waiting_cost": elapsed_time,
                "raw_results": response.data,
            },
        )
        search_logger.debug(
            f"Firecrawl async search completed for query '{query}' in {elapsed_time:.2f} seconds with {len(docs)} results."
        )
        return resp


# --- Perplexity Search Implementation ---
class PerplexitySearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)
        self.api_key = config.api_key or os.getenv("PERPLEXITY_API_KEY")

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using Perplexity."""
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not set in config or environment.")
        st = time.time()
        url = "https://api.perplexity.ai/search"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"q": query, "num_results": max_results}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            search_logger.error(f"Perplexity search failed: {response.text}")
            return SearchQueryResponse(
                query=query,
                follow_up_queries=None,
                answer=None,
                images=None,
                docs=[],
            )

        data = response.json()
        results = [
            RetrievedDocument(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", ""),
                score=result.get("score", None),
                raw_content=result.get("raw_content", None),
            )
            for result in data.get("results", [])
        ]

        et = time.time()
        elapsed_time = et - st
        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=data.get("follow_up_queries", []),
            answer=data.get("answer"),
            images=data.get("images", []),
            docs=results,
            elapsed_time=data.get("elapsed_time", elapsed_time),
            extra={
                "source": "perplexity",
                "waiting_cost": elapsed_time,
            },
        )
        search_logger.debug(
            f"Perplexity search results for query '{query}': {resp.get_statistics()}"
        )

        return resp

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using Perplexity."""
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not set in config or environment.")

        st = time.time()
        url = "https://api.perplexity.ai/search"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"q": query, "num_results": max_results}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    search_logger.error(f"Perplexity search failed: {text}")
                    return SearchQueryResponse(
                        query=query,
                        follow_up_queries=None,
                        answer=None,
                        images=None,
                        docs=[],
                    )
                data = await resp.json()
                results = [
                    RetrievedDocument(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=result.get("content", ""),
                        score=result.get("score", None),
                        raw_content=result.get("raw_content", None),
                    )
                    for result in data.get("results", [])
                ]

                et = time.time()
                elapsed_time = et - st
                resp = SearchQueryResponse(
                    query=query,
                    follow_up_queries=data.get("follow_up_queries", []),
                    answer=data.get("answer"),
                    images=data.get("images", []),
                    docs=results,
                    elapsed_time=data.get("elapsed_time", elapsed_time),
                    extra={
                        "source": "perplexity",
                        "waiting_cost": elapsed_time,
                    },
                )
                search_logger.debug(
                    f"Perplexity search results for query '{query}': {resp.get_statistics()}"
                )

                return resp


# --- Google Search Implementation ---
class GoogleSearchServiceInstance(SearchServiceInstance):
    def __init__(
        self, instance_name: str, group_name: str, config: SearchServiceInstanceConfig
    ):
        super().__init__(instance_name, group_name, config)
        self.api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        self.cx = getattr(config, "cx", os.getenv("GOOGLE_CSE_ID"))

    def single_search_helper(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Helper function to perform a single actual search using Google Custom Search."""
        if not self.api_key or not self.cx:
            raise ValueError(
                "GOOGLE_API_KEY or GOOGLE_CSE_ID not set in config or environment."
            )
        st = time.time()
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": max_results,
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            search_logger.error(f"Google search failed: {response.text}")
            return SearchQueryResponse(
                query=query,
                follow_up_queries=None,
                answer=None,
                images=None,
                docs=[],
            )

        data = response.json()
        items = data.get("items", [])
        results = [
            RetrievedDocument(
                title=item.get("title", ""),
                url=item.get("link", ""),
                content=item.get("snippet", ""),
                score=None,  # Google doesn't provide a score, set to None
                raw_content=None,  # No raw content available
            )
            for item in items
        ]

        et = time.time()
        elapsed_time = et - st
        resp = SearchQueryResponse(
            query=query,
            follow_up_queries=data.get("queries", {}).get("nextPage", []),
            answer=data.get("searchInformation", {}).get("totalResults"),
            images=data.get("items", []),  # Assuming images are part of items
            docs=results,
            elapsed_time=data.get("elapsed_time", elapsed_time),
            extra={
                "source": "google",
                "waiting_cost": elapsed_time,
            },
        )
        search_logger.debug(
            f"Google search results for query '{query}': {resp.get_statistics()}"
        )

        return resp

    async def single_search_helper_async(
        self, query: str, max_results: int, **kwargs
    ) -> SearchQueryResponse:
        """Asynchronously perform a single actual search using Google Custom Search."""
        if not self.api_key or not self.cx:
            raise ValueError(
                "GOOGLE_API_KEY or GOOGLE_CSE_ID not set in config or environment."
            )
        st = time.time()
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": max_results,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    search_logger.error(f"Google search failed: {text}")
                    return SearchQueryResponse(
                        query=query,
                        follow_up_queries=None,
                        answer=None,
                        images=None,
                        docs=[],
                    )
                data = await resp.json()
                items = data.get("items", [])
                results = [
                    RetrievedDocument(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        content=item.get("snippet", ""),
                        score=None,  # Google doesn't provide a score, set to None
                        raw_content=None,  # No raw content available
                    )
                    for item in items
                ]

                et = time.time()
                elapsed_time = et - st
                resp = SearchQueryResponse(
                    query=query,
                    follow_up_queries=data.get("queries", {}).get("nextPage", []),
                    answer=data.get("searchInformation", {}).get("totalResults"),
                    images=data.get("items", []),  # Assuming images are part of items
                    docs=results,
                    elapsed_time=data.get("elapsed_time", elapsed_time),
                    extra={
                        "source": "google",
                        "waiting_cost": elapsed_time,
                    },
                )
                search_logger.debug(
                    f"Google search results for query '{query}': {resp.get_statistics()}"
                )

                return resp


# --- SearchService Router ---
class SearchService(BaseService):
    """Search service for managing search engine instances."""

    def __init__(self, scheduling_policy: str = "round_robin"):
        super().__init__("search", scheduling_policy)

    def serve(
        self,
        request: SearchServiceRequest,
        engine: str = None,
        instance_name: str = None,
        tags=None,
    ) -> SearchServiceResponse:
        engine = engine or request.engine
        if engine is not None:
            tags = tags or set()
            if engine not in tags:
                tags.add(engine)
        return super().serve(request, instance_name, tags)

    async def serve_async(
        self,
        request: SearchServiceRequest,
        engine: str = None,
        instance_name: str = None,
        tags=None,
    ) -> SearchServiceResponse:
        engine = engine or request.engine
        if engine is not None:
            tags = tags or set()
            if engine not in tags:
                tags.add(engine)
        return await super().serve_async(request, instance_name, tags)
