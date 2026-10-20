from typing import Optional
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import VDBStoringInput, VDBStoringOutput
from .shared import VDBSearchingInput, VDBSearchingOutput
from .utils import get_vdb_client, get_logger
from .settings import get_settings


@agsbench_rag_fsm.fast_service
def vdb_store(
    item: VDBStoringInput, context: RequestContext = None
) -> VDBStoringOutput:
    item.vdb_type = get_settings().vdb_type
    item.vdb_uri = get_settings().vdb_uri
    insertion_size = get_settings().insertion_size
    embeddings = item.embeddings
    if len(embeddings) == 0:
        get_logger().warning(
            f"{context.request_id} No embeddings to store, skip {item}"
        )
        return VDBStoringOutput(count=0)
    dimension = len(embeddings[0])
    vdb_client = get_vdb_client(
        vdb_type=item.vdb_type,
        vdb_uri=item.vdb_uri,
        collection_name=item.collection_name,
        embedding_dimension=dimension,
    )
    if item.metadatas is None:
        try:
            item.metadatas = [{"source": doc.metadata["source"]} for doc in item.docs]
        except AttributeError:
            item.metadatas = [{} for _ in item.docs]

    ids = vdb_client.store(
        texts=[doc.page_content for doc in item.docs],
        embeddings=embeddings,
        metadatas=item.metadatas,
        ids=item.ids,
        insertion_size=insertion_size,
    )
    return VDBStoringOutput(count=len(ids))


@agsbench_rag_fsm.fast_service
def vdb_search(
    item: VDBSearchingInput, context: RequestContext = None
) -> VDBSearchingOutput:
    item.vdb_type = get_settings().vdb_type
    item.vdb_uri = get_settings().vdb_uri
    search_args = get_settings().search_args
    item.k = search_args.get("k", 4) if search_args else 4
    dimension = len(item.embedding)
    vdb_client = get_vdb_client(
        vdb_type=item.vdb_type,
        vdb_uri=item.vdb_uri,
        collection_name=item.collection_name,
        embedding_dimension=dimension,
    )
    documents = vdb_client.similarity_search(
        query=item.query, embedding=item.embedding, k=item.k
    )
    return VDBSearchingOutput(docs=documents)


@agsbench_rag_fsm.fast_service
def batch_vdb_search(
    items: list[VDBSearchingInput], context: RequestContext = None
) -> list[VDBSearchingOutput]:
    if len(items) == 0:
        return VDBSearchingOutput(docs=[])
    item = items[0]
    item.vdb_type = get_settings().vdb_type
    item.vdb_uri = get_settings().vdb_uri
    search_args = get_settings().search_args
    item.k = search_args.get("k", 4) if search_args else 4
    dimension = len(item.embedding)
    vdb_client = get_vdb_client(
        vdb_type=item.vdb_type,
        vdb_uri=item.vdb_uri,
        collection_name=item.collection_name,
        embedding_dimension=dimension,
    )
    all_documents = vdb_client.batch_similarity_search(
        queries=[item.query for item in items],
        embeddings=[item.embedding for item in items],
        k=item.k,
    )
    return [VDBSearchingOutput(docs=documents) for documents in all_documents]
