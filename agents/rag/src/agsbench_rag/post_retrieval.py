from fast_service import RequestContext
from langchain_core.documents import Document
from langchain.load import dumps, loads
import cohere
from langchain.retrievers.document_compressors import CohereRerank
from typing import Annotated
from fastapi import Body

from .settings import get_settings
from .shared import agsbench_rag_fsm
from .shared import PostRetrievalInput, PostRetrievalOutput, RetrievalOutput
from .utils import get_reranker


def reciprocal_rank_fusion(documents: list[list[Document]], k=60) -> list[Document]:
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}
    fused_docs = {}

    # Iterate through each list of ranked documents
    for docs in documents:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = doc.metadata.get("source", "") + doc.page_content
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                fused_docs[doc_str] = doc
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        fused_docs[doc_str]
        for doc_str, score in sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def get_unique_union(documents: list[list[Document]]) -> list[Document]:
    """Unique union of retrieved docs"""
    unique_docs = []
    doc_strings = []
    for sublist in documents:
        for doc in sublist:
            doc_str = doc.metadata.get("source", "") + doc.page_content
            if doc_str not in doc_strings:
                doc_strings.append(doc_str)
                unique_docs.append(doc)
    # sort docs by source and hash
    unique_docs = sorted(
        unique_docs,
        key=lambda x: (loads(x).metadata.get("source", ""), hash(x.page_content)),
    )
    # Return
    return unique_docs


@agsbench_rag_fsm.fast_service
def reranking(
    docs: list[Document],
    query: Annotated[str, Body()],
    model: Annotated[str, Body()],
    engine: Annotated[str, Body()],
    context: RequestContext = None,
) -> list[Document]:
    reranker = get_reranker(model=model, engine=engine)
    reranked_docs = reranker.compress_documents(documents=docs, query=query)
    # sort docs by relevance
    reranked_docs = sorted(
        reranked_docs, key=lambda x: x.metadata["relevance_score"], reverse=True
    )
    return reranked_docs


@agsbench_rag_fsm.fast_service
def compression(
    docs: list[Document],
    query: Annotated[str, Body()],
    method: Annotated[str, Body()],
    context: RequestContext = None,
) -> list[Document]:
    return docs


@agsbench_rag_fsm.fast_service
def selection(
    docs: list[Document],
    query: Annotated[str, Body()],
    method: Annotated[str, Body()],
    context: RequestContext = None,
) -> list[Document]:
    return docs


@agsbench_rag_fsm.fast_service
def post_retrieval(
    item: PostRetrievalInput, context: RequestContext = None
) -> PostRetrievalOutput:
    item.fusion_policy = get_settings().fusion_policy

    # unpack the input
    if isinstance(item.retrieved, RetrievalOutput):
        docs = item.retrieved.docs
    else:
        docs_list = [subitem.docs for subitem in item.retrieved]
        if item.fusion_policy == "reciprocal_rank":
            docs = reciprocal_rank_fusion(docs_list)
        elif item.fusion_policy == "unique":
            docs = get_unique_union(docs_list)
        elif item.fusion_policy == "all":
            docs = [doc for sublist in docs_list for doc in sublist]
        else:
            raise NotImplementedError(
                f"Unsupported fusion_policy: {item.fusion_policy}"
            )

    if item.reranking_args is not None:
        if (
            item.reranking_args["model"] is not None
            and item.reranking_args["engine"] is not None
        ):
            docs = reranking(
                docs,
                item.reranking_args["query"],
                model=item.reranking_args["model"],
                engine=item.reranking_args["engine"],
                context=context,
            )
    if item.compression_args is not None:
        docs = compression(
            docs,
            item.compression_args["query"],
            item.compression_args["method"],
            context=context,
        )
    if item.selection_args is not None:
        docs = selection(
            docs,
            item.selection_args["query"],
            item.selection_args["method"],
            context=context,
        )
    return PostRetrievalOutput(docs=docs)
