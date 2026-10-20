import asyncio
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import IndexingInput, IndexingOutput
from .shared import (
    DocLoadingInput,
    DocSplittingInput,
    EmbeddingDocsInput,
    VDBStoringInput,
)
from .shared import CollectionClearingInput, SituatingChunksInput
from .async_loading import load_docs_async
from .async_chunking import split_docs_async
from .async_embedding import embed_documents_async
from .async_vstore import vdb_store_async, get_async_vdb_client
from .async_chunk_situating import situate_chunks_async
from .async_utils import get_logger


@agsbench_rag_fsm.fast_service
async def indexing_async(
    item: IndexingInput, context: RequestContext = None
) -> IndexingOutput:
    """Index documents asynchronously."""
    get_logger().debug(f"Starting async indexing for {context.request_id}")

    # Load documents
    docs_output = await load_docs_async(
        item=DocLoadingInput(type=item.type, uri=item.uri), context=context
    )
    get_logger().debug(f"Loaded {len(docs_output.docs)} documents")

    # Split documents into chunks
    splits_output = await split_docs_async(
        item=DocSplittingInput(
            docs=docs_output.docs,
            splitter_model=item.splitter_model,
            chunk_size=item.chunk_size,
            chunk_overlap=item.chunk_overlap,
        ),
        context=context,
    )
    get_logger().debug(f"Split into {len(splits_output.docs)} chunks")

    # Generate embeddings
    embeddings_output = await embed_documents_async(
        item=EmbeddingDocsInput(
            docs=splits_output.docs,
            embedding_model=item.embedding_model,
            embedding_device=item.embedding_device,
            embedding_batch_size=item.embedding_batch_size,
        ),
        context=context,
    )
    get_logger().debug(f"Generated {len(embeddings_output.embeddings)} embeddings")

    # Store in vector database
    store_output = await vdb_store_async(
        item=VDBStoringInput(
            vdb_type=item.vdb_type,
            vdb_uri=item.vdb_uri,
            collection_name=item.collection_name,
            docs=splits_output.docs,
            embeddings=embeddings_output.embeddings,
        ),
        context=context,
    )

    get_logger().debug(f"Stored {store_output.count} documents in vector database")
    return IndexingOutput.model_construct(**store_output.model_dump())


@agsbench_rag_fsm.fast_service
async def contextual_indexing_async(
    item: IndexingInput, context: RequestContext = None
) -> IndexingOutput:
    """Index documents with contextual chunking asynchronously."""
    get_logger().debug(f"Starting async contextual indexing for {context.request_id}")

    # Load documents
    loaded_output = await load_docs_async(
        item=DocLoadingInput(type=item.type, uri=item.uri), context=context
    )

    # Add document IDs to metadata
    for i, doc in enumerate(loaded_output.docs):
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["doc_id"] = i

    get_logger().debug(f"Loaded {len(loaded_output.docs)} documents")

    # Split documents into chunks
    splits_output = await split_docs_async(
        item=DocSplittingInput(
            docs=loaded_output.docs,
            splitter_model=item.splitter_model,
            chunk_size=item.chunk_size,
            chunk_overlap=item.chunk_overlap,
        ),
        context=context,
    )
    chunks = splits_output.docs
    get_logger().debug(f"Split into {len(chunks)} chunks")

    # Situate chunks with context
    contextual_chunks_output = await situate_chunks_async(
        item=SituatingChunksInput(
            docs=loaded_output.docs,
            chunks=chunks,
        ),
        context=context,
    )
    contextual_chunks = contextual_chunks_output.chunks
    get_logger().debug(
        f"Generated contextual information for {len(contextual_chunks)} chunks"
    )

    # Generate embeddings for contextual chunks
    embeddings_output = await embed_documents_async(
        item=EmbeddingDocsInput(
            docs=contextual_chunks,
            embedding_model=item.embedding_model,
            embedding_device=item.embedding_device,
            embedding_batch_size=item.embedding_batch_size,
        ),
        context=context,
    )
    get_logger().debug(f"Generated {len(embeddings_output.embeddings)} embeddings")

    # Store in vector database
    store_output = await vdb_store_async(
        item=VDBStoringInput(
            vdb_type=item.vdb_type,
            vdb_uri=item.vdb_uri,
            collection_name=item.collection_name,
            docs=contextual_chunks,
            embeddings=embeddings_output.embeddings,
        ),
        context=context,
    )

    get_logger().debug(
        f"Stored {store_output.count} contextual documents in vector database"
    )
    return IndexingOutput.model_construct(**store_output.model_dump())


@agsbench_rag_fsm.fast_service
async def clear_collection_async(
    item: CollectionClearingInput, context: RequestContext = None
) -> IndexingOutput:
    """Clear vector database collection asynchronously."""
    get_logger().debug(f"Clearing collection {item.collection_name}")

    # For Milvus, we need embedding dimension even for collection clearing
    # Get it by testing the embedding function
    embedding_dimension = None
    if item.vdb_type == "milvus":
        from .async_utils import get_embedding_fn, async_run_in_executor

        fn = get_embedding_fn(
            embedding_model=item.embedding_model,
            embedding_device=item.embedding_device,
        )

        # Get embedding dimension by testing with a dummy query
        try:
            if hasattr(fn, "aembed_query") and asyncio.iscoroutinefunction(
                fn.aembed_query
            ):
                test_embedding = await fn.aembed_query("test")
            else:
                test_embedding = await async_run_in_executor(fn.embed_query, "test")
            embedding_dimension = len(test_embedding)
        except Exception:
            # Fallback to sync
            test_embedding = await async_run_in_executor(fn.embed_query, "test")
            embedding_dimension = len(test_embedding)

    vdb_client = await get_async_vdb_client(
        vdb_type=item.vdb_type,
        vdb_uri=item.vdb_uri,
        collection_name=item.collection_name,
        embedding_dimension=embedding_dimension,
    )

    await vdb_client.drop_collection()
    get_logger().debug(f"Cleared collection {item.collection_name}")

    return IndexingOutput(count=0)


# Advanced async indexing functions
class AsyncIndexingEngine:
    """Advanced async indexing engine with additional capabilities."""

    def __init__(self):
        pass

    async def parallel_indexing(
        self,
        items: list[IndexingInput],
        max_concurrency: int = 3,
        context: RequestContext = None,
    ) -> list[IndexingOutput]:
        """Index multiple document sets concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def index_with_semaphore(item):
            async with semaphore:
                return await indexing_async(item=item, context=context)

        tasks = [index_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    async def incremental_indexing(
        self,
        item: IndexingInput,
        batch_size: int = 100,
        context: RequestContext = None,
    ) -> IndexingOutput:
        """Index documents incrementally in batches."""
        get_logger().debug(
            f"Starting incremental indexing with batch size {batch_size}"
        )

        # Load all documents first
        docs_output = await load_docs_async(
            item=DocLoadingInput(type=item.type, uri=item.uri), context=context
        )

        total_docs = len(docs_output.docs)
        total_stored = 0

        # Process documents in batches
        for i in range(0, total_docs, batch_size):
            batch_docs = docs_output.docs[i : i + batch_size]
            get_logger().debug(
                f"Processing batch {i//batch_size + 1}: {len(batch_docs)} documents"
            )

            # Split batch
            splits_output = await split_docs_async(
                item=DocSplittingInput(
                    docs=batch_docs,
                    splitter_model=item.splitter_model,
                    chunk_size=item.chunk_size,
                    chunk_overlap=item.chunk_overlap,
                ),
                context=context,
            )

            # Generate embeddings for batch
            embeddings_output = await embed_documents_async(
                item=EmbeddingDocsInput(
                    docs=splits_output.docs,
                    embedding_model=item.embedding_model,
                    embedding_device=item.embedding_device,
                    embedding_batch_size=item.embedding_batch_size,
                ),
                context=context,
            )

            # Store batch
            store_output = await vdb_store_async(
                item=VDBStoringInput(
                    vdb_type=item.vdb_type,
                    vdb_uri=item.vdb_uri,
                    collection_name=item.collection_name,
                    docs=splits_output.docs,
                    embeddings=embeddings_output.embeddings,
                ),
                context=context,
            )

            total_stored += store_output.count
            get_logger().debug(
                f"Batch {i//batch_size + 1} complete: {store_output.count} documents stored"
            )

        return IndexingOutput(count=total_stored)

    async def multi_collection_indexing(
        self,
        item: IndexingInput,
        collections: list[str],
        context: RequestContext = None,
    ) -> dict[str, IndexingOutput]:
        """Index the same documents into multiple collections."""
        # Load and process documents once
        docs_output = await load_docs_async(
            item=DocLoadingInput(type=item.type, uri=item.uri), context=context
        )

        splits_output = await split_docs_async(
            item=DocSplittingInput(
                docs=docs_output.docs,
                splitter_model=item.splitter_model,
                chunk_size=item.chunk_size,
                chunk_overlap=item.chunk_overlap,
            ),
            context=context,
        )

        embeddings_output = await embed_documents_async(
            item=EmbeddingDocsInput(
                docs=splits_output.docs,
                embedding_model=item.embedding_model,
                embedding_device=item.embedding_device,
                embedding_batch_size=item.embedding_batch_size,
            ),
            context=context,
        )

        # Store in multiple collections concurrently
        async def store_in_collection(collection_name):
            return await vdb_store_async(
                item=VDBStoringInput(
                    vdb_type=item.vdb_type,
                    vdb_uri=item.vdb_uri,
                    collection_name=collection_name,
                    docs=splits_output.docs,
                    embeddings=embeddings_output.embeddings,
                ),
                context=context,
            )

        tasks = [store_in_collection(collection) for collection in collections]
        results = await asyncio.gather(*tasks)

        return {
            collection: IndexingOutput.model_construct(**result.model_dump())
            for collection, result in zip(collections, results)
        }

    async def health_check_before_indexing(
        self,
        item: IndexingInput,
        context: RequestContext = None,
    ) -> dict:
        """Perform health checks before indexing."""
        health_status = {
            "vdb_connection": False,
            "collection_accessible": False,
            "embedding_service": False,
            "errors": [],
        }

        try:
            # For Milvus, we need embedding dimension for health check
            embedding_dimension = None
            if item.vdb_type == "milvus":
                from .async_utils import get_embedding_fn, async_run_in_executor

                fn = get_embedding_fn(
                    embedding_model=item.embedding_model,
                    embedding_device=item.embedding_device,
                )

                # Get embedding dimension by testing with a dummy query
                try:
                    if hasattr(fn, "aembed_query") and asyncio.iscoroutinefunction(
                        fn.aembed_query
                    ):
                        test_embedding = await fn.aembed_query("test")
                    else:
                        test_embedding = await async_run_in_executor(
                            fn.embed_query, "test"
                        )
                    embedding_dimension = len(test_embedding)
                except Exception:
                    # Fallback to sync
                    test_embedding = await async_run_in_executor(fn.embed_query, "test")
                    embedding_dimension = len(test_embedding)

            # Check VDB connection
            vdb_client = await get_async_vdb_client(
                vdb_type=item.vdb_type,
                vdb_uri=item.vdb_uri,
                collection_name=item.collection_name,
                embedding_dimension=embedding_dimension,
            )
            health_status["vdb_connection"] = True

            # Check collection accessibility
            stats = await vdb_client.get_collection_stats()
            health_status["collection_accessible"] = True

        except Exception as e:
            health_status["errors"].append(f"VDB error: {str(e)}")

        try:
            # Test embedding service with a simple query
            test_embedding = await embed_documents_async(
                item=EmbeddingDocsInput(
                    docs=[type("TestDoc", (), {"page_content": "test"})()],
                    embedding_model=item.embedding_model,
                    embedding_device=item.embedding_device,
                    embedding_batch_size=1,
                ),
                context=context,
            )
            health_status["embedding_service"] = True

        except Exception as e:
            health_status["errors"].append(f"Embedding error: {str(e)}")

        return health_status


# Global async indexing engine instance
_async_indexing_engine = None


async def get_async_indexing_engine() -> AsyncIndexingEngine:
    """Get async indexing engine instance."""
    global _async_indexing_engine
    if _async_indexing_engine is None:
        _async_indexing_engine = AsyncIndexingEngine()
    return _async_indexing_engine
