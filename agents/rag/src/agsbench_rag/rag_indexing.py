from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import IndexingInput, IndexingOutput
from .shared import DocLoadingInput, DocLoadingOutput
from .shared import DocSplittingInput, DocSplittingOutput
from .shared import EmbeddingDocsInput
from .shared import VDBStoringInput, VDBStoringOutput
from .shared import CollectionClearingInput
from .shared import SituatingChunksInput
from .loading import load_docs
from .chunking import split_docs
from .embedding import embed_documents
from .vstore import vdb_store
from .chunk_situating import situate_chunks
from .utils import get_vdb_client


@agsbench_rag_fsm.fast_service
def indexing(item: IndexingInput, context: RequestContext = None) -> IndexingOutput:
    docs: DocLoadingOutput = load_docs(
        item=DocLoadingInput(type=item.type, uri=item.uri), context=context
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

    embeddings = embed_documents(
        item=EmbeddingDocsInput(
            docs=splits.docs,
            embedding_model=item.embedding_model,
            embedding_device=item.embedding_device,
            embedding_batch_size=item.embedding_batch_size,
        ),
        context=context,
    ).embeddings

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
    return IndexingOutput.model_construct(**ret.model_dump())


@agsbench_rag_fsm.fast_service
def contextual_indexing(
    item: IndexingInput, context: RequestContext = None
) -> IndexingOutput:
    loaded: DocLoadingOutput = load_docs(
        item=DocLoadingInput(type=item.type, uri=item.uri), context=context
    )

    for i, doc in enumerate(loaded.docs):
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["doc_id"] = i

    splits: DocSplittingOutput = split_docs(
        item=DocSplittingInput(
            docs=loaded.docs,
            splitter_model=item.splitter_model,
            chunk_size=item.chunk_size,
            chunk_overlap=item.chunk_overlap,
        ),
        context=context,
    )
    chunks = splits.docs

    contextual_chunks = situate_chunks(
        SituatingChunksInput(
            docs=loaded.docs,
            chunks=chunks,
        ),
        context=context,
    ).chunks

    embeddings = embed_documents(
        item=EmbeddingDocsInput(
            docs=contextual_chunks,
            embedding_model=item.embedding_model,
            embedding_device=item.embedding_device,
            embedding_batch_size=item.embedding_batch_size,
        ),
        context=context,
    ).embeddings

    ret: VDBStoringOutput = vdb_store(
        item=VDBStoringInput(
            vdb_type=item.vdb_type,
            vdb_uri=item.vdb_uri,
            collection_name=item.collection_name,
            docs=contextual_chunks,
            embeddings=embeddings,
        ),
        context=context,
    )
    return IndexingOutput.model_construct(**ret.model_dump())


@agsbench_rag_fsm.fast_service
def clear_collection(
    item: CollectionClearingInput, context: RequestContext = None
) -> IndexingOutput:
    vdb_client = get_vdb_client(
        vdb_type=item.vdb_type,
        vdb_uri=item.vdb_uri,
        collection_name=item.collection_name,
    )
    vdb_client.drop_collection()

    return IndexingOutput(count=0)
