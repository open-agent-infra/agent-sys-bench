import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from fast_service import RequestContext, FastServiceConfig
from .settings import (
    LLMSettings,
    VDBSettings,
    SplitterSettings,
    EmbeddingSettings,
    RetrieverSettings,
    GenerationSettings,
)
from .settings import get_settings
from .shared import agsbench_rag_fsm
from .shared import RAGE2EInput, RAGE2EOutput
from .shared import CollectionClearingInput
from .shared import IndexingInput, IndexingOutput
from .shared import DocLoadingInput, DocLoadingOutput
from .shared import DocSplittingInput, DocSplittingOutput
from .shared import EmbeddingDocsInput
from .shared import VDBStoringInput, VDBStoringOutput
from .shared import QueryTranslationInput
from .shared import PostRetrievalInput, PostRetrievalOutput
from .shared import GenerationInput
from .shared import RetrievalJudgingInput
from .shared import EmbeddingQueryInput
from .shared import VDBSearchingInput
from .shared import RetrievalOutput
from .utils import get_logger

from .loading import load_docs
from .chunking import split_docs
from .embedding import embed_documents
from .embedding import embed_query, batch_embed_query
from .vstore import vdb_store
from .vstore import vdb_search, batch_vdb_search
from .rag_indexing import clear_collection
from .pre_retrieval import query_translation
from .post_retrieval import post_retrieval
from .generation import generation
from .judging import retrieval_judger


class RAGServiceGlobalContext:
    def __init__(self):
        self.ctx = {}

    def has_context(self, context: RequestContext) -> bool:
        if context is None:
            return False
        return context.request_id in self.ctx

    def add_context(self, context: RequestContext):
        self.ctx[context.request_id] = {
            "status": "Running",
            # "lock": threading.Lock(),
        }
        return self.ctx[context.request_id]

    def update_status(self, context: RequestContext, status: str):
        assert self.has_context(context)
        # with self.ctx[context.request_id]["lock"]:
        self.ctx[context.request_id]["status"] = status

    def get_status(self, context: RequestContext) -> str:
        assert self.has_context(context)
        # with self.ctx[context.request_id]["lock"]:
        return self.ctx[context.request_id]["status"]


@lru_cache
def get_global_context():
    global_ctx = RAGServiceGlobalContext()
    return global_ctx


@agsbench_rag_fsm.fast_service
def parallel_indexing(
    item: IndexingInput, context: RequestContext = None
) -> IndexingOutput:
    global_ctx = get_global_context()

    if global_ctx.get_status(context) == "Finished":
        return IndexingOutput(count=0, status=0)
    get_logger().debug(
        f"parallel_indexing load_docs: {global_ctx.ctx[context.request_id]}"
    )
    docs: DocLoadingOutput = load_docs(
        item=DocLoadingInput(type=item.type, uri=item.uri), context=context
    )

    if global_ctx.get_status(context) == "Finished":
        return IndexingOutput(count=0, status=0)
    get_logger().debug(
        f"parallel_indexing split_docs: {global_ctx.ctx[context.request_id]}"
    )
    splits: DocSplittingOutput = split_docs(
        item=DocSplittingInput(
            docs=docs.docs,
            splitter_model=item.splitter_model,
            chunk_size=item.chunk_size,
            chunk_overlap=item.chunk_overlap,
        ),
        context=context,
    )

    if global_ctx.get_status(context) == "Finished":
        return IndexingOutput(count=0, status=0)
    get_logger().debug(
        f"parallel_indexing embde_documents: {global_ctx.ctx[context.request_id]}"
    )
    embeddings = embed_documents(
        item=EmbeddingDocsInput(
            docs=splits.docs,
            embedding_model=item.embedding_model,
            embedding_device=item.embedding_device,
            embedding_batch_size=item.embedding_batch_size,
        ),
        context=context,
    ).embeddings

    if global_ctx.get_status(context) == "Finished":
        return IndexingOutput(count=0, status=0)
    get_logger().debug(
        f"parallel_indexing vdb_store: {global_ctx.ctx[context.request_id]}"
    )
    ret: VDBStoringOutput = vdb_store(
        item=VDBStoringInput(
            vdb_type=item.vdb_type,
            vdb_uri=item.vdb_uri,
            collection_name=item.collection_name,
            docs=splits.docs,
            embeddings=embeddings,
        ),
        context=context,
    )

    get_logger().debug(
        f"parallel_indexing all finished: {global_ctx.ctx[context.request_id]}"
    )

    return IndexingOutput.model_construct(**ret.model_dump())


def extract_all_settings(item: RAGE2EInput):
    if item.splitter_settings is not None:
        splitter_settings = item.splitter_settings
    else:
        splitter_settings = SplitterSettings.from_default_settings()
    if item.vdb_settings is not None:
        vdb_settings = item.vdb_settings
    else:
        vdb_settings = VDBSettings.from_default_settings()
    if item.embedding_settings is not None:
        embedding_settings = item.embedding_settings
    else:
        embedding_settings = EmbeddingSettings.from_default_settings()
    if item.retriever_settings is not None:
        retriever_settings = item.retriever_settings
    else:
        retriever_settings = RetrieverSettings.from_default_settings()
    if item.generation_settings is not None:
        generation_settings = item.generation_settings
    else:
        generation_settings = GenerationSettings.from_default_settings()
    return (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    )


@agsbench_rag_fsm.fast_service
def parallel_naive_rag(
    item: RAGE2EInput, context: RequestContext = None
) -> RAGE2EOutput:
    global_ctx = get_global_context()
    global_ctx.add_context(context)
    global_ctx.update_status(context, "Running")

    (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    ) = extract_all_settings(item=item)

    indexing_item = IndexingInput(
        uri=item.uri,
        type=item.type,
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
        embedding_batch_size=embedding_settings.embedding_batch_size,
    )
    embedding_item = EmbeddingQueryInput(
        query=item.query,
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
    )

    with ThreadPoolExecutor() as executor:
        y1 = executor.submit(parallel_indexing, item=indexing_item, context=context)
        y2 = executor.submit(
            embed_query,
            item=embedding_item,
            context=context,
        )

    y1 = y1.result()
    embedding = y2.result().embedding

    search_item = VDBSearchingInput(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        query=item.query,
        embedding=embedding,
        k=(
            retriever_settings.search_args.get("k", 4)
            if retriever_settings.search_args
            else 4
        ),
    )

    docs = vdb_search(
        item=search_item,
        context=context,
    ).docs

    post_retrieval_output: PostRetrievalOutput = post_retrieval(
        item=PostRetrievalInput(retrieved=RetrievalOutput(docs=docs)), context=context
    )

    q_ctx = "\n\n".join([doc.page_content for doc in post_retrieval_output.docs])
    generation_input = GenerationInput(
        query=item.query,
        context=q_ctx,
        method="normal",
    )
    generation_output = generation(item=generation_input, context=context)
    final_answer = generation_output.answer

    if get_settings().early_stopping:
        global_ctx.update_status(context, "Finished")
    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
        )
        y3 = clear_collection(item=clearing_item, context=context)
        get_logger().debug(f"Naive RAG clearing output {context.request_id}: {y3}")

    return RAGE2EOutput(answer=final_answer)


@agsbench_rag_fsm.fast_service
def parallel_advanced_rag(
    item: RAGE2EInput, context: RequestContext = None
) -> RAGE2EOutput:
    global_ctx = get_global_context()
    global_ctx.add_context(context)
    global_ctx.update_status(context, "Running")

    (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    ) = extract_all_settings(item=item)

    assert generation_settings.generation_method == "normal"

    indexing_item = IndexingInput(
        uri=item.uri,
        type=item.type,
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
        embedding_batch_size=embedding_settings.embedding_batch_size,
    )

    def pre_search_branch(
        original_query: str,
        embedding_settings: EmbeddingSettings,
        context: RequestContext,
    ):
        qtrans_method = "MultiQuery"
        qtrans_input = QueryTranslationInput(query=original_query, method=qtrans_method)
        qtrans_output = query_translation(item=qtrans_input, context=context)
        queries = qtrans_output.queries
        embedding_items = [
            EmbeddingQueryInput(
                query=query,
                embedding_model=embedding_settings.embedding_model,
                embedding_device=embedding_settings.embedding_device,
            )
            for query in queries
        ]
        embeddings = [
            out.embedding
            for out in batch_embed_query(
                items=embedding_items,
                context=context,
            )
        ]
        return queries, embeddings

    with ThreadPoolExecutor() as executor:
        y1 = executor.submit(parallel_indexing, item=indexing_item, context=context)
        y2 = executor.submit(
            pre_search_branch,
            original_query=item.query,
            embedding_settings=embedding_settings,
            context=context,
        )

    y1 = y1.result()
    queries, embeddings = y2.result()

    retrieved = [
        RetrievalOutput(docs=out.docs)
        for out in batch_vdb_search(
            items=[
                VDBSearchingInput(
                    vdb_type=vdb_settings.vdb_type,
                    vdb_uri=vdb_settings.vdb_uri,
                    collection_name=vdb_settings.collection_name,
                    query=query,
                    embedding=embedding,
                    k=(
                        retriever_settings.search_args.get("k", 4)
                        if retriever_settings.search_args
                        else 4
                    ),
                )
                for query, embedding in zip(queries, embeddings)
            ],
            context=context,
        )
    ]

    post_retrieval_output: PostRetrievalOutput = post_retrieval(
        item=PostRetrievalInput(
            retrieved=retrieved,
            reranking_args={
                "model": get_settings().reranker_model,
                "engine": get_settings().reranker_engine,
                "query": item.query,
            },
        ),
        context=context,
    )
    if retriever_settings.search_args is not None:
        k = retriever_settings.search_args.get("k", 4)
    else:
        k = 4
    retrieved_docs = post_retrieval_output.docs[:k]

    q_ctx = "\n\n".join([doc.page_content for doc in retrieved_docs])
    generation_input = GenerationInput(
        query=item.query,
        context=q_ctx,
        method=generation_settings.generation_method,
        method_args=generation_settings.generation_method_args,
    )
    generation_output = generation(item=generation_input, context=context)
    final_answer = generation_output.answer

    if get_settings().early_stopping:
        global_ctx.update_status(context, "Finished")

    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
        )
        y3 = clear_collection(item=clearing_item, context=context)
        get_logger().debug(f"Naive RAG clearing output {context.request_id}: {y3}")

    return RAGE2EOutput(answer=final_answer)


@agsbench_rag_fsm.fast_service
def parallel_dynamic_rag(
    item: RAGE2EInput, context: RequestContext = None
) -> RAGE2EOutput:
    global_ctx = get_global_context()
    global_ctx.add_context(context)
    global_ctx.update_status(context, "Running")

    (
        splitter_settings,
        vdb_settings,
        embedding_settings,
        retriever_settings,
        generation_settings,
    ) = extract_all_settings(item=item)

    assert generation_settings.generation_method == "normal"

    indexing_item = IndexingInput(
        uri=item.uri,
        type=item.type,
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
        embedding_batch_size=embedding_settings.embedding_batch_size,
    )

    get_logger().debug(f"parallel_dynamic_rag submit.")
    executor = ThreadPoolExecutor()
    y1 = executor.submit(parallel_indexing, item=indexing_item, context=context)
    get_logger().debug(f"parallel_dynamic_rag submit done.")

    retrieval_judger_input = RetrievalJudgingInput(question=item.query)
    retrieval_judger_output = retrieval_judger(
        item=retrieval_judger_input, context=context
    )
    need_retrieval = retrieval_judger_output.judgement

    if need_retrieval:
        qtrans_method = "MultiQuery"
        original_query = item.query
        qtrans_input = QueryTranslationInput(query=original_query, method=qtrans_method)
        qtrans_output = query_translation(item=qtrans_input, context=context)
        queries = qtrans_output.queries
        embedding_items = [
            EmbeddingQueryInput(
                query=query,
                embedding_model=embedding_settings.embedding_model,
                embedding_device=embedding_settings.embedding_device,
            )
            for query in queries
        ]
        embeddings = [
            out.embedding
            for out in batch_embed_query(
                items=embedding_items,
                context=context,
            )
        ]

        get_logger().debug(f"parallel_dynamic_rag wait indexing...")
        y1 = y1.result()
        get_logger().debug(f"parallel_dynamic_rag wait indexing done.")

        retrieved = [
            RetrievalOutput(docs=out.docs)
            for out in batch_vdb_search(
                items=[
                    VDBSearchingInput(
                        vdb_type=vdb_settings.vdb_type,
                        vdb_uri=vdb_settings.vdb_uri,
                        collection_name=vdb_settings.collection_name,
                        query=query,
                        embedding=embedding,
                        k=(
                            retriever_settings.search_args.get("k", 4)
                            if retriever_settings.search_args
                            else 4
                        ),
                    )
                    for query, embedding in zip(queries, embeddings)
                ],
                context=context,
            )
        ]

        post_retrieval_output: PostRetrievalOutput = post_retrieval(
            item=PostRetrievalInput(
                retrieved=retrieved,
                reranking_args={
                    "model": get_settings().reranker_model,
                    "engine": get_settings().reranker_engine,
                    "query": item.query,
                },
            ),
            context=context,
        )
        if retriever_settings.search_args is not None:
            k = retriever_settings.search_args.get("k", 4)
        else:
            k = 4
        retrieved_docs = post_retrieval_output.docs[:k]

        q_ctx = "\n\n".join([doc.page_content for doc in retrieved_docs])
        generation_input = GenerationInput(
            query=item.query,
            context=q_ctx,
            method=generation_settings.generation_method,
            method_args=generation_settings.generation_method_args,
        )
    else:
        if get_settings().early_stopping:
            executor.shutdown(wait=False)
        else:
            executor.shutdown(wait=True)
        generation_input = GenerationInput(
            query=item.query,
            context="",
            method="direct",
        )
    if get_settings().early_stopping:
        global_ctx.update_status(context, "Finished")
    generation_output = generation(item=generation_input, context=context)
    final_answer = generation_output.answer

    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
        )
        y3 = clear_collection(item=clearing_item, context=context)
        get_logger().debug(f"Naive RAG clearing output {context.request_id}: {y3}")

    return RAGE2EOutput(answer=final_answer)
