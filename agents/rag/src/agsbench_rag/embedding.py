from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import (
    EmbeddingDocsInput,
    EmbeddingDocsOutput,
    EmbeddingQueryInput,
    EmbeddingQueryOutput,
)
from .settings import get_settings
from .utils import get_embedding_fn


@agsbench_rag_fsm.fast_service
def embed_documents(
    item: EmbeddingDocsInput, context: RequestContext = None
) -> EmbeddingDocsOutput:
    # embedding_model = item.embedding_model
    # embedding_device = item.embedding_device
    item.embedding_batch_size = get_settings().embedding_batch_size
    embedding_model = get_settings().embedding_model
    embedding_device = get_settings().embedding_device
    fn = get_embedding_fn(
        embedding_model=embedding_model, embedding_device=embedding_device
    )
    docs = [doc.page_content for doc in item.docs]
    batch_size = item.embedding_batch_size
    if batch_size > 0:
        embeddings = []
        for i in range(0, len(docs), batch_size):
            embeddings.extend(fn.embed_documents(docs[i : i + batch_size]))
    else:
        embeddings = fn.embed_documents(docs)
    return EmbeddingDocsOutput(embeddings=embeddings)


@agsbench_rag_fsm.fast_service
def embed_query(
    item: EmbeddingQueryInput, context: RequestContext = None
) -> EmbeddingQueryOutput:
    # embedding_model = item.embedding_model
    # embedding_device = item.embedding_device
    embedding_model = get_settings().embedding_model
    embedding_device = get_settings().embedding_device
    fn = get_embedding_fn(
        embedding_model=embedding_model, embedding_device=embedding_device
    )
    return EmbeddingQueryOutput(embedding=fn.embed_query(item.query))


@agsbench_rag_fsm.fast_service
def batch_embed_query(
    items: list[EmbeddingQueryInput], context: RequestContext = None
) -> list[EmbeddingQueryOutput]:
    if len(items) == 0:
        return EmbeddingQueryOutput(embedding=[])
    # embedding_model = item.embedding_model
    # embedding_device = item.embedding_device
    embedding_model = get_settings().embedding_model
    embedding_device = get_settings().embedding_device
    fn = get_embedding_fn(
        embedding_model=embedding_model, embedding_device=embedding_device
    )
    queries = [item.query for item in items]
    embeddings = fn.embed_documents(queries)
    return [EmbeddingQueryOutput(embedding=embedding) for embedding in embeddings]
