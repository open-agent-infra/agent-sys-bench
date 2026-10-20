"""
Optimized async document loading with native async loader support.

This module provides async document loading capabilities that use native
async methods from LangChain loaders where available, falling back to
executor-based sync methods when necessary.
"""

import os
import asyncio
from typing import Union, List
from fast_service import RequestContext
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyMuPDFLoader,
    UnstructuredHTMLLoader,
    WikipediaLoader,
)
from langchain_core.documents import Document
from .settings import get_settings
from .shared import agsbench_rag_fsm
from .shared import DocLoadingInput, DocLoadingOutput
from .async_utils import (
    get_logger,
    AsyncFileHandler,
    AsyncHTTPClient,
    async_run_in_executor,
)


class AsyncLoadingEngine:
    """Async document loading engine with native async loader support."""

    def __init__(self):
        pass

    async def execute(
        self,
        type: Union[str, List[str]],
        uri: Union[str, List[str]],
        request_id: str = None,
    ) -> list[Document]:
        """Execute document loading asynchronously."""
        get_logger().debug(
            f"AsyncLoadingEngine.load start for {request_id}: {type} {uri}"
        )

        if type == "web":
            docs = await self.load_web_docs(uri, request_id=request_id)
        elif type == "file":
            docs = await self.load_pdf_docs(uri, request_id=request_id)
        elif type == "wiki":
            docs = await self.load_wiki_docs(uri, request_id=request_id)
        elif type == "local_html":
            docs = await self.load_local_html_docs(uri, request_id=request_id)
        else:
            raise NotImplementedError(f"Unsupported type: {type}")

        get_logger().debug(
            f"AsyncLoadingEngine.load end for {request_id}: {type} {uri}"
        )
        return docs

    async def _load_with_native_async(self, loader, fallback_method: str = "load"):
        """Try to use native async loader methods, fallback to sync."""
        # Check for async load methods
        async_methods = ["aload", "aload_and_split", "load_async"]

        for method_name in async_methods:
            if hasattr(loader, method_name):
                method = getattr(loader, method_name)
                if asyncio.iscoroutinefunction(method):
                    try:
                        get_logger().debug(f"Using native async method: {method_name}")
                        return await method()
                    except Exception as e:
                        get_logger().warning(
                            f"Native async method {method_name} failed: {e}"
                        )
                        continue

        # Fallback to sync method in executor
        get_logger().debug(f"Falling back to sync method: {fallback_method}")
        sync_method = getattr(loader, fallback_method)
        return await async_run_in_executor(sync_method)

    async def load_web_docs(
        self, web_paths: Union[str, List[str]], request_id: str = None
    ) -> list[Document]:
        """Load web documents asynchronously."""
        get_logger().debug(f"load_web_docs web_paths {request_id}: {web_paths}")

        # Normalize input to list
        if isinstance(web_paths, str):
            web_paths = [web_paths]

        if get_settings().loading_persistency is not None:
            # Handle persistency per URL like sync version
            docs = []
            new_urls = []

            # Check cache for each URL individually
            for url in web_paths:
                dirs = url.replace("://", "/").split("/")
                loading_dir = os.path.join(get_settings().loading_persistency, *dirs)
                if os.path.exists(loading_dir):
                    get_logger().debug(
                        f"Loading {url} from persistency {request_id}: "
                        f"{loading_dir}"
                    )
                    # Load cached documents for this URL
                    cached_docs = await self._load_from_cache(loading_dir)
                    get_logger().debug(
                        f"Loaded {len(cached_docs)} docs from persistency "
                        f"{request_id}"
                    )
                    docs.extend(cached_docs)
                else:
                    new_urls.append(url)

            # Load only new URLs that aren't cached
            if new_urls:
                get_logger().debug(
                    f"Loading {len(new_urls)} new urls from web "
                    f"{request_id}: {new_urls}"
                )
                new_docs = await self._load_web_docs_uncached(new_urls)
                docs.extend(new_docs)

                # Save to persistency per URL
                await self._save_web_docs_to_cache(new_docs, new_urls)

            return docs
        else:
            # No persistency, load directly
            return await self._load_web_docs_uncached(web_paths)

    async def _load_web_docs_uncached(self, web_paths: List[str]) -> list[Document]:
        """Load web documents without caching."""
        # Try native async WebBaseLoader first
        try:
            loader = WebBaseLoader(
                web_paths=web_paths,
                header_template={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/34.0.1847.131 Safari/537.36"
                    )
                },
                continue_on_failure=True,
                verify_ssl=False,
                requests_kwargs={"allow_redirects": False, "timeout": 60},
            )

            # Use native async loader method
            docs = await self._load_with_native_async(loader)
            get_logger().debug(f"Loaded {len(docs)} docs from web using native async")
            return docs

        except Exception as e:
            get_logger().error(f"WebBaseLoader failed: {e}")

            # Fallback to pure async HTTP client for each URL
            all_docs = []
            for url in web_paths:
                try:
                    async with AsyncHTTPClient() as client:
                        response = await client.get(url)
                        content = await response.text()
                        doc = Document(
                            page_content=content,
                            metadata={"source": url, "type": "web"},
                        )
                        all_docs.append(doc)
                except Exception as fallback_error:
                    get_logger().error(
                        f"Fallback HTTP client failed for {url}: " f"{fallback_error}"
                    )
                    continue

            get_logger().debug(
                f"Loaded {len(all_docs)} docs using fallback HTTP client"
            )
            return all_docs

    async def _save_web_docs_to_cache(
        self, docs: list[Document], urls: List[str]
    ) -> None:
        """Save web documents to cache per URL."""
        for url in urls:
            # Find documents for this URL
            url_docs = []
            for doc in docs:
                if doc.metadata.get("source") == url:
                    url_docs.append(doc)

            # Save documents for this URL
            dirs = url.replace("://", "/").split("/")
            loading_dir = os.path.join(get_settings().loading_persistency, *dirs)
            get_logger().debug(
                f"Saving to persistency: {loading_dir} with "
                f"{len(url_docs)} docs for {url}"
            )
            await self._save_to_cache(url_docs, loading_dir)

    async def load_pdf_docs(
        self, file_path: str, request_id: str = None
    ) -> list[Document]:
        """Load PDF documents asynchronously."""

        async def _load_from_pdf(file_path):
            loader = PyMuPDFLoader(file_path=file_path)
            return await self._load_with_native_async(loader)

        file_tag = os.path.basename(file_path).replace(".", "_")
        return await self._handle_persistency(_load_from_pdf, file_path, file_tag)

    async def load_wiki_docs(
        self, query: str, request_id: str = None
    ) -> list[Document]:
        """Load Wikipedia documents asynchronously."""

        async def _load_from_wiki(query):
            try:
                loader = WikipediaLoader(query=query)
                docs = await self._load_with_native_async(loader)
                get_logger().debug(
                    f"Loaded {len(docs)} docs from Wikipedia using " f"native async"
                )
                return docs
            except Exception as e:
                get_logger().error(f"Failed to load from wiki: {e}")
                return []

        dir_tag = query.replace(" ", "_")
        return await self._handle_persistency(_load_from_wiki, query, dir_tag)

    async def load_local_html_docs(
        self, file_path: str, request_id: str = None
    ) -> list[Document]:
        """Load local HTML documents asynchronously."""
        loader = UnstructuredHTMLLoader(file_path=file_path)
        return await self._load_with_native_async(loader)

    async def _handle_persistency(self, load_func, input_data, cache_key):
        """Handle loading with persistency caching."""
        if get_settings().loading_persistency is None:
            return await load_func(input_data)

        # Check cache
        loading_dir = os.path.join(get_settings().loading_persistency, cache_key)

        if os.path.exists(loading_dir):
            get_logger().debug(f"Loading from persistency: {loading_dir}")
            return await self._load_from_cache(loading_dir)

        # Load and cache
        docs = await load_func(input_data)
        await self._save_to_cache(docs, loading_dir)
        return docs

    async def _load_from_cache(self, loading_dir):
        """Load documents from cache directory."""
        docs = []

        def has_numeric_prefix(filename):
            """Check if filename has a numeric prefix (e.g., '0.json', '1.json')."""
            try:
                int(filename.split(".")[0])
                return True
            except (ValueError, IndexError):
                return False

        # Filter for .json files with numeric prefixes and sort them
        json_files = [
            f
            for f in os.listdir(loading_dir)
            if f.endswith(".json") and has_numeric_prefix(f)
        ]
        files = sorted(json_files, key=lambda x: int(x.split(".")[0]))

        async def load_file(file):
            file_path = os.path.join(loading_dir, file)
            doc_data = await AsyncFileHandler.read_json(file_path)
            return Document.model_construct(**doc_data)

        tasks = [load_file(file) for file in files]
        return await asyncio.gather(*tasks)

    async def _save_to_cache(self, docs, loading_dir):
        """Save documents to cache directory."""
        get_logger().debug(f"Saving to persistency: {loading_dir}")
        os.makedirs(loading_dir, exist_ok=True)

        async def save_doc(i, doc):
            file_path = os.path.join(loading_dir, f"{i}.json")
            await AsyncFileHandler.write_json(file_path, doc.model_dump())

        save_tasks = [save_doc(i, doc) for i, doc in enumerate(docs)]
        await asyncio.gather(*save_tasks)


class AsyncLoadingClient:
    """Async document loading client."""

    def __init__(self, engine_type: str = "local"):
        self.engine_type = engine_type
        self.engine = AsyncLoadingEngine()

    async def execute(
        self, item: DocLoadingInput, request_id: str = None
    ) -> list[Document]:
        """Execute document loading asynchronously."""
        return await self.engine.execute(
            type=item.type, uri=item.uri, request_id=request_id
        )


# Global async loading client instance
_async_loading_client = None


async def get_async_loading_client() -> AsyncLoadingClient:
    """Get async loading client instance."""
    global _async_loading_client
    if _async_loading_client is None:
        engine_type = get_settings().loading_engine or "local"
        _async_loading_client = AsyncLoadingClient(engine_type=engine_type)
    return _async_loading_client


@agsbench_rag_fsm.fast_service
async def load_docs_async(
    item: DocLoadingInput, context: RequestContext = None
) -> DocLoadingOutput:
    """Load documents asynchronously."""
    client = await get_async_loading_client()
    docs = await client.execute(item, request_id=context.request_id)
    # sort docs by source and hash
    docs = sorted(
        docs, key=lambda x: (x.metadata.get("source", ""), hash(x.page_content))
    )
    return DocLoadingOutput(docs=docs)
