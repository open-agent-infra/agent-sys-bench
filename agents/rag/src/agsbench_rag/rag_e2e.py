from fast_service import RequestContext, FastServiceConfig
from .settings import (
    LLMSettings,
    VDBSettings,
    SplitterSettings,
    EmbeddingSettings,
    RetrieverSettings,
    GenerationSettings,
)
from .shared import agsbench_rag_fsm
from .shared import RAGE2EInput, RAGE2EOutput
from .shared import IndexingInput
from .shared import RAGChattingInput
from .shared import CollectionClearingInput
from .utils import get_logger, get_settings
from .utils import close_vdb_client
from .rag_indexing import indexing, contextual_indexing
from .rag_indexing import clear_collection
from .rag_chatting import naive_rag_chatting
from .rag_chatting import multiquery_rag_chatting
from .rag_chatting import dynamic_rag_chatting
from .parallel_rag import (
    parallel_naive_rag,
    parallel_advanced_rag,
    parallel_dynamic_rag,
)


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
def naive_rag(item: RAGE2EInput, context: RequestContext = None) -> RAGE2EOutput:
    if get_settings().parallel_rag:
        return parallel_naive_rag(item=item, context=context)
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
    get_logger().debug(
        f"Naive RAG indexing input {context.request_id}: {indexing_item}"
    )
    y1 = indexing(
        item=indexing_item,
        context=context,
    )
    get_logger().debug(f"Naive RAG indexing output {context.request_id}: {y1}")

    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )
    y2 = naive_rag_chatting(item=chatting_item, context=context)
    get_logger().debug(f"Naive RAG chatting output {context.request_id}: {y2}")

    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
        )
        y3 = clear_collection(item=clearing_item, context=context)
        get_logger().debug(f"Naive RAG clearing output {context.request_id}: {y3}")

    close_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    return RAGE2EOutput(answer=y2.answer)


@agsbench_rag_fsm.fast_service
def multiquery_rag(item: RAGE2EInput, context: RequestContext = None) -> RAGE2EOutput:
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
    y1 = indexing(
        item=indexing_item,
        context=context,
    )
    get_logger().debug(f"Multiquery RAG indexing output {context.request_id}: {y1}")

    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )
    y2 = multiquery_rag_chatting(item=chatting_item, context=context)
    get_logger().debug(f"Multiquery RAG chatting output {context.request_id}: {y2}")

    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
        )
        y3 = clear_collection(item=clearing_item, context=context)
        get_logger().debug(f"Naive RAG clearing output {context.request_id}: {y3}")

    close_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    return RAGE2EOutput(answer=y2.answer)


@agsbench_rag_fsm.fast_service
def advanced_rag(item: RAGE2EInput, context: RequestContext = None) -> RAGE2EOutput:
    if get_settings().parallel_rag:
        return parallel_advanced_rag(item=item, context=context)
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
    y1 = indexing(
        item=indexing_item,
        context=context,
    )
    get_logger().debug(f"Multiquery RAG indexing output {context.request_id}: {y1}")

    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )
    y2 = multiquery_rag_chatting(item=chatting_item, context=context)
    get_logger().debug(f"Multiquery RAG chatting output {context.request_id}: {y2}")

    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
        )
        y3 = clear_collection(item=clearing_item, context=context)
        get_logger().debug(f"Naive RAG clearing output {context.request_id}: {y3}")

    close_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    return RAGE2EOutput(answer=y2.answer)


@agsbench_rag_fsm.fast_service
def dynamic_rag(item: RAGE2EInput, context: RequestContext = None) -> RAGE2EOutput:
    if get_settings().parallel_rag:
        return parallel_dynamic_rag(item=item, context=context)

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
    y1 = indexing(
        item=indexing_item,
        context=context,
    )
    get_logger().debug(f"Dynamic RAG indexing output {context.request_id}: {y1}")

    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )
    y2 = dynamic_rag_chatting(item=chatting_item, context=context)
    get_logger().debug(f"Dynamic RAG chatting output {context.request_id}: {y2}")

    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
        )
        y3 = clear_collection(item=clearing_item, context=context)
        get_logger().debug(f"Naive RAG clearing output {context.request_id}: {y3}")

    close_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    return RAGE2EOutput(answer=y2.answer)


@agsbench_rag_fsm.fast_service
def contextual_rag(item: RAGE2EInput, context: RequestContext = None) -> RAGE2EOutput:
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
    get_logger().debug(
        f"Contextual RAG indexing input {context.request_id}: {indexing_item}"
    )
    y1 = contextual_indexing(
        item=indexing_item,
        context=context,
    )
    get_logger().debug(f"Contextual RAG indexing output {context.request_id}: {y1}")

    chatting_item = RAGChattingInput(
        query=item.query,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )
    y2 = naive_rag_chatting(item=chatting_item, context=context)
    get_logger().debug(f"Contextual RAG chatting output {context.request_id}: {y2}")

    if item.clear_collection:
        clearing_item = CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
        )
        y3 = clear_collection(item=clearing_item, context=context)
        get_logger().debug(f"Contextual RAG clearing output {context.request_id}: {y3}")

    close_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
    )
    return RAGE2EOutput(answer=y2.answer)


if __name__ == "__main__":
    from .utils import warm_up

    config_dict = {
        "service_impl": "fastapi",
        "client_mode": True,
        "service_list": [],
        "monitor": True,
        "monitor_storage": {"type": "csv", "store_dir": "./.cache/monitor/local_test"},
    }
    config = FastServiceConfig.load_from_dict(config_dict=config_dict)
    agsbench_rag_fsm.setup_client_mode(config=config)

    llm_settings = LLMSettings.from_default_settings()
    splitter_settings = SplitterSettings.from_default_settings()
    vdb_settings = VDBSettings.from_default_settings()
    embedding_settings = EmbeddingSettings.from_default_settings()
    retriever_settings = RetrieverSettings.from_default_settings()
    generation_settings = GenerationSettings.from_default_settings()

    # warm up
    warm_up(
        splitter_settings=splitter_settings,
        embedding_settings=embedding_settings,
        vdb_settings=vdb_settings,
    )

    clear_collection(
        item=CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
            embedding_device=embedding_settings.embedding_device,
            embedding_model=embedding_settings.embedding_model,
        )
    )

    test_item = RAGE2EInput(
        query="What is Task Decomposition?",
        uri="https://lilianweng.github.io/posts/2023-06-23-agent/",
        type="web",
        splitter_settings=splitter_settings,
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )
    print(f"Test item: {test_item}")

    print(f"---------- Test naive rag ----------")
    test_answer = naive_rag(item=test_item).answer
    print(f"Test answer: {test_answer}")

    print(f"---------- Test advanced rag ----------")
    test_answer = advanced_rag(item=test_item).answer
    print(f"Test answer: {test_answer}")

    print(f"---------- Test dynamic rag ----------")
    test_answer = dynamic_rag(item=test_item).answer
    print(f"Test answer: {test_answer}")
