from fast_service import RequestContext
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .shared import agsbench_rag_fsm
from .shared import RetrievalInput, RetrievalOutput
from .shared import EmbeddingQueryInput, EmbeddingQueryOutput
from .shared import VDBSearchingInput, VDBSearchingOutput
from .embedding import embed_query, batch_embed_query
from .vstore import vdb_search, batch_vdb_search
from .utils import get_logger, async_run
from .settings import get_settings


@agsbench_rag_fsm.fast_service
def retrieval(item: RetrievalInput, context: RequestContext = None) -> RetrievalOutput:
    get_logger().debug(f"Retrieving documents for query: {item.query}")
    embedding = embed_query(
        item=EmbeddingQueryInput(
            query=item.query,
            embedding_model=item.embedding_model,
            embedding_device=item.embedding_device,
        ),
        context=context,
    ).embedding
    docs = vdb_search(
        item=VDBSearchingInput(
            vdb_type=item.vdb_type,
            vdb_uri=item.vdb_uri,
            collection_name=item.collection_name,
            query=item.query,
            embedding=embedding,
            k=item.search_args.get("k", 4) if item.search_args else 4,
        ),
        context=context,
    ).docs
    return RetrievalOutput(docs=docs)


@agsbench_rag_fsm.fast_service
def batch_retrieval(
    items: list[RetrievalInput], context: RequestContext = None
) -> list[RetrievalOutput]:
    parallel_retrieval = get_settings().parallel_retrieval
    parallel_search = get_settings().parallel_search
    for item in items:
        get_logger().debug(f"retrieval input {context.request_id}: {item}")
    ret = []
    if parallel_retrieval:
        if parallel_search:
            embeddings = [
                out.embedding
                for out in batch_embed_query(
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
            ]
            ret = [
                RetrievalOutput(docs=out.docs)
                for out in batch_vdb_search(
                    items=[
                        VDBSearchingInput(
                            vdb_type=item.vdb_type,
                            vdb_uri=item.vdb_uri,
                            collection_name=item.collection_name,
                            query=item.query,
                            embedding=embedding,
                            k=item.search_args.get("k", 4) if item.search_args else 4,
                        )
                        for item, embedding in zip(items, embeddings)
                    ],
                    context=context,
                )
            ]
        else:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(retrieval, item=item, context=context)
                    for item in items
                ]
            ret = [future.result() for future in futures]
    else:
        for item in items:
            retrieval_output = retrieval(item=item, context=context)
            ret.append(retrieval_output)
    for rout in ret:
        get_logger().debug(f"retrieval output {context.request_id}: {rout}")
    return ret


async def _retrieval_async(
    item: RetrievalInput, context: RequestContext = None
) -> RetrievalOutput:
    return retrieval(item=item, context=context)


async def _batch_retrieval_async(
    items: list[RetrievalInput], context: RequestContext = None
) -> list[RetrievalOutput]:
    ret = await asyncio.gather(
        *[_retrieval_async(item=item, context=context) for item in items]
    )
    return ret


@agsbench_rag_fsm.fast_service
async def batch_retrieval_async(
    items: list[RetrievalInput], context: RequestContext = None
) -> list[RetrievalOutput]:
    ret = async_run(_batch_retrieval_async(items=items, context=context))
    return ret
