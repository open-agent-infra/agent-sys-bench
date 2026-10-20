import asyncio
import time
from functools import lru_cache
from typing import Optional, Dict, List, Any, Callable, Awaitable
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiohttp
from langchain.text_splitter import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
import chromadb
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import InfinityEmbeddings
from langchain_milvus import Milvus
from langchain.retrievers import ContextualCompressionRetriever

try:
    from langchain_cohere import CohereRerank
except ImportError:
    # Fallback to deprecated import for backward compatibility
    from langchain.retrievers.document_compressors import CohereRerank
from vectordb import BaseVectorDBClient, AsyncBaseVectorDBClient
from .settings import LLMSettings, LoggingSettings
from .settings import SplitterSettings, EmbeddingSettings, VDBSettings
from .settings import get_settings
from .utils import get_vdb_client
from vectordb.async_chromadb import AsyncChromaVectorDBClient
from vectordb.async_milvusdb import AsyncMilvusVectorDBClient


async def async_run_in_executor(func: Callable, *args, **kwargs) -> Any:
    """Run a synchronous function in a thread pool executor."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


async def async_gather_with_concurrency(
    tasks: List[Awaitable], max_concurrency: int = 10
) -> List[Any]:
    """Execute tasks with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_with_semaphore(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*[_run_with_semaphore(task) for task in tasks])


@lru_cache
def get_async_lock(name: str) -> asyncio.Lock:
    """Get an async lock by name."""
    return asyncio.Lock()


def get_logger():
    """Get logger instance."""
    return logging.getLogger(__name__)


@lru_cache
def get_llm() -> BaseChatModel:
    """Get LLM instance (cached)."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        seed=settings.llm_seed,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


@lru_cache
def get_splitter(
    splitter_model: str, chunk_size: int, chunk_overlap: int
) -> TextSplitter:
    """Get text splitter instance (cached)."""
    if splitter_model == "recursive-character":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_model == "recursive-character-tiktoken":
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_model == "recursive-character-python":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        raise NotImplementedError(f"Unsupported splitter_model: {splitter_model}")


@lru_cache
def get_embedding_fn(embedding_model: str, embedding_device: str) -> Embeddings:
    """Get embedding function instance (cached)."""
    if embedding_device == "TEI":
        assert embedding_model.startswith("http"), "TEI requires a URL"
        return HuggingFaceEndpointEmbeddings(
            model=embedding_model, task="feature-extraction"
        )
    elif embedding_device == "Infinity":
        url, model = embedding_model.split(":::")
        return InfinityEmbeddings(model=model, infinity_api_url=url)
    elif embedding_device == "vLLM":
        url, model = embedding_model.split(":::")
        return OpenAIEmbeddings(
            model=model,
            base_url=url,
            check_embedding_ctx_length=False,
            chunk_size=65536,
        )
    elif embedding_device.startswith("http"):
        return OpenAIEmbeddings(
            model=embedding_model,
            base_url=embedding_device,
            check_embedding_ctx_length=False,
            chunk_size=65536,
        )
    elif embedding_model == "all-MiniLM-L6-v2":
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": embedding_device},
        )
    elif embedding_model == "text-embedding-ada-002":
        return OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=65536)
    else:
        raise NotImplementedError(f"Unsupported embedding model: {embedding_model}")


# Global cache for async VDB clients
_async_vdb_client_cache: Dict[str, AsyncBaseVectorDBClient] = {}


async def get_async_vdb_client(
    vdb_type: str,
    vdb_uri: str,
    collection_name: str,
    embedding_dimension: Optional[int] = None,
) -> AsyncBaseVectorDBClient:
    """Get async vector database client with proper async caching."""
    # Create cache key
    cache_key = f"{vdb_type}:{vdb_uri}:{collection_name}"

    # Check cache first (thread-safe)
    async with get_async_lock(cache_key):
        if cache_key in _async_vdb_client_cache:
            return _async_vdb_client_cache[cache_key]

        # Create new client
        if vdb_type == "chromadb":
            client = AsyncChromaVectorDBClient(
                uri=vdb_uri, collection_name=collection_name
            )
        elif vdb_type == "milvus":
            if embedding_dimension is None:
                raise ValueError("embedding_dimension is required for Milvus")
            client = AsyncMilvusVectorDBClient(
                uri=vdb_uri,
                collection_name=collection_name,
                embedding_dimension=embedding_dimension,
            )
        else:
            raise NotImplementedError(f"Unsupported vdb_type: {vdb_type}")

        # Cache the client
        _async_vdb_client_cache[cache_key] = client
        return client


async def close_async_vdb_client(
    vdb_type: str,
    vdb_uri: str,
    collection_name: str,
    embedding_dimension: Optional[int] = None,
):
    """Close async vector database client and remove from cache."""
    cache_key = f"{vdb_type}:{vdb_uri}:{collection_name}"
    async with get_async_lock(cache_key):
        if cache_key in _async_vdb_client_cache:
            client = _async_vdb_client_cache.pop(cache_key)
            await client.close()


@lru_cache
def get_async_reranker(model: str, engine: str = "cohere"):
    """Get async reranker instance (cached)."""
    # Use the already imported CohereRerank (with fallback handling)
    import cohere

    if engine == "cohere":
        return CohereRerank(model=model, cohere_api_key="dummy")
    elif engine.startswith("http"):
        # Clean up engine URL (remove /v1 suffix if present)
        base_url = engine
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        # Create Cohere ClientV2 with custom base URL
        try:
            # Try using ClientV2 (newer cohere versions)
            cohere_client = cohere.ClientV2(api_key="EMPTY", base_url=base_url)
        except AttributeError:
            # Fallback to older Client for backward compatibility
            cohere_client = cohere.Client(api_key="EMPTY", base_url=base_url)

        # Don't pass top_n=None to avoid HTTP 400 error
        return CohereRerank(client=cohere_client, model=model)
    else:
        raise NotImplementedError(f"Unsupported reranking engine: {engine}")


async def warm_up_async(
    splitter_settings: SplitterSettings,
    embedding_settings: EmbeddingSettings,
    vdb_settings: VDBSettings,
):
    """Async warm up function to initialize all components."""
    # Initialize cached components (these are fast)
    get_llm()
    get_splitter(
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
    )

    # Initialize embedding function
    fn = get_embedding_fn(
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
    )

    # Test embedding to get dimension - try async first
    try:
        if hasattr(fn, "aembed_query") and asyncio.iscoroutinefunction(fn.aembed_query):
            test_embedding = await fn.aembed_query("Warm Up")
        else:
            test_embedding = await async_run_in_executor(fn.embed_query, "Warm Up")
    except Exception:
        # Fallback to sync
        test_embedding = await async_run_in_executor(fn.embed_query, "Warm Up")

    # Initialize VDB client
    get_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_dimension=len(test_embedding),
    )


class AsyncFileHandler:
    """Async file operations handler."""

    @staticmethod
    async def read_file(file_path: str, encoding: str = "utf-8") -> str:
        """Read file content asynchronously."""
        async with aiofiles.open(file_path, "r", encoding=encoding) as f:
            return await f.read()

    @staticmethod
    async def write_file(file_path: str, content: str, encoding: str = "utf-8") -> None:
        """Write file content asynchronously."""
        async with aiofiles.open(file_path, "w", encoding=encoding) as f:
            await f.write(content)

    @staticmethod
    async def read_json(file_path: str) -> Dict:
        """Read JSON file asynchronously."""
        import json

        content = await AsyncFileHandler.read_file(file_path)
        return json.loads(content)

    @staticmethod
    async def write_json(file_path: str, data: Dict) -> None:
        """Write JSON file asynchronously."""
        import json

        content = json.dumps(data, ensure_ascii=False, indent=2)
        await AsyncFileHandler.write_file(file_path, content)


class AsyncHTTPClient:
    """Async HTTP client for web requests."""

    def __init__(self):
        self._session = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make async GET request."""
        if not self._session:
            raise RuntimeError(
                "HTTP client not initialized. Use as async context manager."
            )
        return await self._session.get(url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make async POST request."""
        if not self._session:
            raise RuntimeError(
                "HTTP client not initialized. Use as async context manager."
            )
        return await self._session.post(url, **kwargs)


# Async context manager for batch operations
class AsyncBatchProcessor:
    """Process items in batches asynchronously."""

    def __init__(self, batch_size: int = 32, max_concurrency: int = 10):
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    async def process_batches(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], Awaitable[List[Any]]],
    ) -> List[Any]:
        """Process items in batches with concurrency control."""
        if not items:
            return []

        # Create batches
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        # Process batches concurrently
        batch_tasks = [process_func(batch) for batch in batches]
        batch_results = await async_gather_with_concurrency(
            batch_tasks, self.max_concurrency
        )

        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results
