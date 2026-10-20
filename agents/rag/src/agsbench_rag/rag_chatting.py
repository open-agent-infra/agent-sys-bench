from fast_service import RequestContext, FastServiceConfig
from .shared import agsbench_rag_fsm
from .shared import RAGChattingInput, RAGChattingOutput
from .shared import QueryTranslationInput
from .pre_retrieval import query_translation
from .shared import RetrievalInput
from .retrieval import retrieval, batch_retrieval
from .shared import PostRetrievalInput, PostRetrievalOutput
from .post_retrieval import post_retrieval
from .shared import GenerationInput
from .generation import generation
from .judging import retrieval_judger
from .shared import RetrievalJudgingInput
from .utils import get_logger
from .settings import get_settings


@agsbench_rag_fsm.fast_service
def naive_rag_chatting(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    retrieval_input = RetrievalInput(
        query=item.query,
        search_type=item.retriever_settings.search_type,
        search_args=item.retriever_settings.search_args,
        retriever_type=item.retriever_settings.retriever_type,
        vdb_type=item.vdb_settings.vdb_type,
        vdb_uri=item.vdb_settings.vdb_uri,
        collection_name=item.vdb_settings.collection_name,
        embedding_model=item.embedding_settings.embedding_model,
        embedding_device=item.embedding_settings.embedding_device,
        embedding_batch_size=item.embedding_settings.embedding_batch_size,
    )
    get_logger().debug(f"retrieval input {context.request_id}: {retrieval_input}")
    retrieval_output = retrieval(item=retrieval_input, context=context)
    get_logger().debug(f"retrieval output {context.request_id}: {retrieval_output}")

    post_retrieval_input = PostRetrievalInput(retrieved=retrieval_output)
    get_logger().debug(
        f"post_retrieval input {context.request_id}: {post_retrieval_input}"
    )
    post_retrieval_output: PostRetrievalOutput = post_retrieval(
        item=post_retrieval_input, context=context
    )
    get_logger().debug(
        f"post_retrieval output {context.request_id}: {post_retrieval_output}"
    )

    q_ctx = "\n\n".join([doc.page_content for doc in post_retrieval_output.docs])

    generation_input = GenerationInput(
        query=item.query,
        context=q_ctx,
        method="normal",
    )
    get_logger().debug(f"generation input {context.request_id}: {generation_input}")
    generation_output = generation(item=generation_input, context=context)
    get_logger().debug(f"generation output {context.request_id}: {generation_output}")

    return RAGChattingOutput(answer=generation_output.answer)


@agsbench_rag_fsm.fast_service
def multiquery_rag_chatting(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    """
    Based on Naive RAG
    + Pre-retrieval: MultiQuery
    + Post-retrieval: unique fusion
    """
    qtrans_method = "MultiQuery"
    reranking_args = {
        "model": get_settings().reranker_model,
        "engine": get_settings().reranker_engine,
        "query": item.query,
    }
    assert item.generation_settings.generation_method == "normal"

    qtrans_input = QueryTranslationInput(query=item.query, method=qtrans_method)
    get_logger().debug(f"query_translation input {context.request_id}: {qtrans_input}")
    qtrans_output = query_translation(item=qtrans_input, context=context)
    get_logger().debug(
        f"query_translation output {context.request_id}: {qtrans_output}"
    )
    queries = qtrans_output.queries
    qtypes = qtrans_output.qtypes

    retrieval_inputs = []
    for query, qtype in zip(queries, qtypes):
        retrieval_input = RetrievalInput(
            query=query,
            search_type=item.retriever_settings.search_type,
            search_args=item.retriever_settings.search_args,
            retriever_type=item.retriever_settings.retriever_type,
            vdb_type=item.vdb_settings.vdb_type,
            vdb_uri=item.vdb_settings.vdb_uri,
            collection_name=item.vdb_settings.collection_name,
            embedding_model=item.embedding_settings.embedding_model,
            embedding_device=item.embedding_settings.embedding_device,
            embedding_batch_size=item.embedding_settings.embedding_batch_size,
        )
        retrieval_inputs.append(retrieval_input)
    retrieval_outputs = batch_retrieval(items=retrieval_inputs, context=context)

    post_retrieval_input = PostRetrievalInput(
        retrieved=retrieval_outputs,
        reranking_args=reranking_args,
    )
    get_logger().debug(
        f"post_retrieval input {context.request_id}: {post_retrieval_input}"
    )
    post_retrieval_output: PostRetrievalOutput = post_retrieval(
        item=post_retrieval_input, context=context
    )
    get_logger().debug(
        f"post_retrieval output {context.request_id}: {post_retrieval_output}"
    )

    if item.retriever_settings.search_args is not None:
        k = item.retriever_settings.search_args.get("k", 4)
    else:
        k = 4
    retrieved_docs = post_retrieval_output.docs[:k]

    q_ctx = "\n\n".join([doc.page_content for doc in retrieved_docs])
    generation_input = GenerationInput(
        query=item.query,
        context=q_ctx,
        method=item.generation_settings.generation_method,
        method_args=item.generation_settings.generation_method_args,
    )
    get_logger().debug(f"generation input {context.request_id}: {generation_input}")
    generation_output = generation(item=generation_input, context=context)
    get_logger().debug(f"generation output {context.request_id}: {generation_output}")

    return RAGChattingOutput(answer=generation_output.answer)


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""

    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()


@agsbench_rag_fsm.fast_service
def query_decomposition_rag_chatting(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    """
    Based on Naive RAG
    + Pre-retrieval: QueryDecomposition
    + Post-retrieval: unique fusion
    + Generation: Individual
    """
    qtrans_method = "QueryDecomposition"
    reranking_args = {
        "model": get_settings().reranker_model,
        "engine": get_settings().reranker_engine,
        "query": item.query,
    }

    qtrans_input = QueryTranslationInput(query=item.query, method=qtrans_method)
    get_logger().debug(f"query_translation input {context.request_id}: {qtrans_input}")
    qtrans_output = query_translation(item=qtrans_input, context=context)
    get_logger().debug(
        f"query_translation output {context.request_id}: {qtrans_output}"
    )
    queries = qtrans_output.queries
    qtypes = qtrans_output.qtypes

    answers = []
    for query, qtype in zip(queries, qtypes):
        retrieval_input = RetrievalInput(
            query=query,
            search_type=item.retriever_settings.search_type,
            search_args=item.retriever_settings.search_args,
            retriever_type=item.retriever_settings.retriever_type,
            vdb_type=item.vdb_settings.vdb_type,
            vdb_uri=item.vdb_settings.vdb_uri,
            collection_name=item.vdb_settings.collection_name,
            embedding_model=item.embedding_settings.embedding_model,
            embedding_device=item.embedding_settings.embedding_device,
            embedding_batch_size=item.embedding_settings.embedding_batch_size,
        )
        get_logger().debug(f"retrieval input {context.request_id}: {retrieval_input}")
        retrieval_output = retrieval(item=retrieval_input, context=context)
        get_logger().debug(f"retrieval output {context.request_id}: {retrieval_output}")
        post_retrieval_input = PostRetrievalInput(
            retrieved=retrieval_output,
            reranking_args=reranking_args,
        )
        get_logger().debug(
            f"post_retrieval input {context.request_id}: {post_retrieval_input}"
        )
        post_retrieval_output: PostRetrievalOutput = post_retrieval(
            item=post_retrieval_input, context=context
        )
        get_logger().debug(
            f"post_retrieval output {context.request_id}: {post_retrieval_output}"
        )

        q_ctx = "\n\n".join([doc.page_content for doc in post_retrieval_output.docs])
        generation_input = GenerationInput(
            query=query,
            context=q_ctx,
            method="normal",
        )
        get_logger().debug(f"generation input {context.request_id}: {generation_input}")
        generation_output = generation(item=generation_input, context=context)
        get_logger().debug(
            f"generation output {context.request_id}: {generation_output}"
        )
        answers.append(generation_output.answer)

    qa_pairs = format_qa_pairs(questions=queries, answers=answers)
    generation_input = GenerationInput(
        query=query, context=qa_pairs, method="decomposition_individual"
    )
    get_logger().debug(f"generation input {context.request_id}: {generation_input}")
    generation_output = generation(item=generation_input, context=context)
    get_logger().debug(f"generation output {context.request_id}: {generation_output}")
    return RAGChattingOutput(answer=generation_output.answer)


def format_qa_pair(question, answer):
    """Format Q and A pair"""

    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


@agsbench_rag_fsm.fast_service
def query_decomposition_recursive_rag_chatting(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    """
    Based on Naive RAG
    + Pre-retrieval: QueryDecomposition
    + Post-retrieval: unique fusion
    + Generation: Recursive
    """
    qtrans_method = "QueryDecomposition"
    reranking_args = {
        "model": get_settings().reranker_model,
        "engine": get_settings().reranker_engine,
        "query": item.query,
    }

    qtrans_input = QueryTranslationInput(query=item.query, method=qtrans_method)
    get_logger().debug(f"query_translation input {context.request_id}: {qtrans_input}")
    qtrans_output = query_translation(item=qtrans_input, context=context)
    get_logger().debug(
        f"query_translation output {context.request_id}: {qtrans_output}"
    )
    queries = qtrans_output.queries
    qtypes = qtrans_output.qtypes

    answers = []
    q_a_pairs = ""
    for query, qtype in zip(queries, qtypes):
        retrieval_input = RetrievalInput(
            query=query,
            search_type=item.retriever_settings.search_type,
            search_args=item.retriever_settings.search_args,
            retriever_type=item.retriever_settings.retriever_type,
            vdb_type=item.vdb_settings.vdb_type,
            vdb_uri=item.vdb_settings.vdb_uri,
            collection_name=item.vdb_settings.collection_name,
            embedding_model=item.embedding_settings.embedding_model,
            embedding_device=item.embedding_settings.embedding_device,
            embedding_batch_size=item.embedding_settings.embedding_batch_size,
        )
        get_logger().debug(f"retrieval input {context.request_id}: {retrieval_input}")
        retrieval_output = retrieval(item=retrieval_input, context=context)
        get_logger().debug(f"retrieval output {context.request_id}: {retrieval_output}")
        post_retrieval_input = PostRetrievalInput(
            retrieved=retrieval_output,
            reranking_args=reranking_args,
        )
        get_logger().debug(
            f"post_retrieval input {context.request_id}: {post_retrieval_input}"
        )
        post_retrieval_output: PostRetrievalOutput = post_retrieval(
            item=post_retrieval_input, context=context
        )
        get_logger().debug(
            f"post_retrieval output {context.request_id}: {post_retrieval_output}"
        )

        q_ctx = "\n\n".join([doc.page_content for doc in post_retrieval_output.docs])
        generation_input = GenerationInput(
            query=query,
            context=q_ctx,
            method="decomposition_recursive",
            method_args={"q_a_pairs": q_a_pairs},
        )
        get_logger().debug(f"generation input {context.request_id}: {generation_input}")
        generation_output = generation(item=generation_input, context=context)
        get_logger().debug(
            f"generation output {context.request_id}: {generation_output}"
        )
        q_a_pair = format_qa_pair(query, generation_output.answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
        answers.append(generation_output.answer)

    return RAGChattingOutput(answer=answers[-1])


@agsbench_rag_fsm.fast_service
def HyDE_rag_chatting(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    qtrans_method = "HyDE"

    qtrans_input = QueryTranslationInput(query=item.query, method=qtrans_method)
    get_logger().debug(f"query_translation input {context.request_id}: {qtrans_input}")
    qtrans_output = query_translation(item=qtrans_input, context=context)
    get_logger().debug(
        f"query_translation output {context.request_id}: {qtrans_output}"
    )
    queries = qtrans_output.queries
    qtypes = qtrans_output.qtypes
    generated_docs_for_retrieval = queries[0]

    new_item = item.model_copy(
        update={"query": generated_docs_for_retrieval}, deep=True
    )
    return naive_rag_chatting(item=new_item, context=context)


@agsbench_rag_fsm.fast_service
def dynamic_rag_chatting(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    """
    Based on MultiQuery RAG
    + Retrieval Judger
    + Reflection Judger
    references:
    1. https://www.pinecone.io/learn/advanced-rag-techniques/
    """
    qtrans_method = "MultiQuery"
    reranking_args = {
        "model": get_settings().reranker_model,
        "engine": get_settings().reranker_engine,
        "query": item.query,
    }
    assert item.generation_settings.generation_method == "normal"

    retrieval_judger_input = RetrievalJudgingInput(question=item.query)
    get_logger().debug(
        f"retrieval_judger input {context.request_id}: {retrieval_judger_input}"
    )
    retrieval_judger_output = retrieval_judger(
        item=retrieval_judger_input, context=context
    )
    get_logger().debug(
        f"retrieval_judger output {context.request_id}: {retrieval_judger_output}"
    )
    need_retrieval = retrieval_judger_output.judgement
    if need_retrieval:
        qtrans_input = QueryTranslationInput(query=item.query, method=qtrans_method)
        get_logger().debug(
            f"query_translation input {context.request_id}: {qtrans_input}"
        )
        qtrans_output = query_translation(item=qtrans_input, context=context)
        get_logger().debug(
            f"query_translation output {context.request_id}: {qtrans_output}"
        )
        queries = qtrans_output.queries
        qtypes = qtrans_output.qtypes

        retrieval_inputs = []
        for query, qtype in zip(queries, qtypes):
            retrieval_input = RetrievalInput(
                query=query,
                search_type=item.retriever_settings.search_type,
                search_args=item.retriever_settings.search_args,
                retriever_type=item.retriever_settings.retriever_type,
                vdb_type=item.vdb_settings.vdb_type,
                vdb_uri=item.vdb_settings.vdb_uri,
                collection_name=item.vdb_settings.collection_name,
                embedding_model=item.embedding_settings.embedding_model,
                embedding_device=item.embedding_settings.embedding_device,
                embedding_batch_size=item.embedding_settings.embedding_batch_size,
            )
            retrieval_inputs.append(retrieval_input)
        retrieval_outputs = batch_retrieval(items=retrieval_inputs, context=context)

        post_retrieval_input = PostRetrievalInput(
            retrieved=retrieval_outputs,
            reranking_args=reranking_args,
        )
        get_logger().debug(
            f"post_retrieval input {context.request_id}: {post_retrieval_input}"
        )
        post_retrieval_output: PostRetrievalOutput = post_retrieval(
            item=post_retrieval_input, context=context
        )
        get_logger().debug(
            f"post_retrieval output {context.request_id}: {post_retrieval_output}"
        )

        if item.retriever_settings.search_args is not None:
            k = item.retriever_settings.search_args.get("k", 4)
        else:
            k = 4
        retrieved_docs = post_retrieval_output.docs[:k]

        q_ctx = "\n\n".join([doc.page_content for doc in retrieved_docs])
        generation_input = GenerationInput(
            query=item.query,
            context=q_ctx,
            method=item.generation_settings.generation_method,
            method_args=item.generation_settings.generation_method_args,
        )
    else:
        get_logger().info("Skip retrieval")
        generation_input = GenerationInput(
            query=item.query,
            context="",
            method="direct",
        )
    get_logger().debug(f"generation input {context.request_id}: {generation_input}")
    generation_output = generation(item=generation_input, context=context)
    get_logger().debug(f"generation output {context.request_id}: {generation_output}")
    return RAGChattingOutput(answer=generation_output.answer)


@agsbench_rag_fsm.fast_service
def self_rag_chatting(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    return RAGChattingOutput(answer="I am a chatbot.")


if __name__ == "__main__":
    from .settings import (
        SplitterSettings,
        LLMSettings,
        VDBSettings,
        EmbeddingSettings,
        RetrieverSettings,
        GenerationSettings,
    )
    from .rag_indexing import IndexingInput, IndexingOutput
    from .rag_indexing import indexing
    from .rag_indexing import CollectionClearingInput, clear_collection
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
    indexing(
        item=IndexingInput(
            uri="https://lilianweng.github.io/posts/2023-06-23-agent/",
            type="web",
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
    )

    test_item = RAGChattingInput(
        query="What is Task Decomposition?",
        vdb_settings=vdb_settings,
        embedding_settings=embedding_settings,
        retriever_settings=retriever_settings,
        generation_settings=generation_settings,
    )

    print(f"---------- Test naive_rag_chatting ----------")
    print(f"Test item: {test_item}")
    test_answer = naive_rag_chatting(item=test_item).answer
    print(f"Test answer: {test_answer}")

    print(f"---------- Test multiquery_rag_chatting ----------")
    test_answer = multiquery_rag_chatting(item=test_item).answer
    print(f"Test answer: {test_answer}")

    print(f"---------- Test query_decomposition_rag_chatting ----------")
    test_answer = query_decomposition_rag_chatting(item=test_item).answer
    print(f"Test answer: {test_answer}")

    print(f"---------- Test query_decomposition_recursive_rag_chatting ----------")
    test_answer = query_decomposition_recursive_rag_chatting(item=test_item).answer
    print(f"Test answer: {test_answer}")

    print(f"---------- Test HyDE_rag_chatting ----------")
    test_answer = HyDE_rag_chatting(item=test_item).answer
    print(f"Test answer: {test_answer}")

    print(f"---------- Test dynamic_rag_chatting ----------")
    test_answer = dynamic_rag_chatting(item=test_item).answer
    print(f"Test answer: {test_answer}")

    print(f"---------- Test self_rag_chatting ----------")
    test_answer = self_rag_chatting(item=test_item).answer
    print(f"Test answer: {test_answer}")

    clear_collection(
        item=CollectionClearingInput(
            vdb_type=vdb_settings.vdb_type,
            vdb_uri=vdb_settings.vdb_uri,
            collection_name=vdb_settings.collection_name,
            embedding_device=embedding_settings.embedding_device,
            embedding_model=embedding_settings.embedding_model,
        )
    )
