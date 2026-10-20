import os
from functools import lru_cache
from typing import List, Dict, Optional
from typing import Annotated
from pydantic import BaseModel
import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletion
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from fastapi import UploadFile, Form
from embedding import SentenceTransformerEmbeddingFunction
from vec_store import MilvusVectorStore
from rag_types import KnwoledgeBaseEntity
from config import get_settings, get_logger

from fast_service import fast_service_manager, RequestContext
from fast_service import RequestContext, FastServiceFile


@lru_cache
def get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=get_settings().openai_api_key,
        base_url=get_settings().openai_base_url,
    )


@lru_cache
def get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    embedding_model_name = "all-MiniLM-L6-v2"
    embedding_device = get_settings().embedding_device
    return SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name, device=embedding_device
    )


@lru_cache
def get_vstore_client() -> MilvusVectorStore:
    vdb_uri = get_settings().vdb_uri
    return MilvusVectorStore(vdb_uri)


class LLMChatInput(BaseModel):
    model: str
    messages: List[Dict]


@fast_service_manager.fast_service
def chat_generate(item: LLMChatInput, context: RequestContext = None) -> ChatCompletion:
    model = item.model
    messages = item.messages
    completion = get_llm_client().chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion


class DocIndexingOutput(BaseModel):
    count: int
    cost: float


@fast_service_manager.fast_service
def indexing(
    # collection_name: Annotated[str, Form()],
    collection_name: str,
    files: list[FastServiceFile],
    context: RequestContext = None,
) -> DocIndexingOutput:
    file_dir = get_settings().raw_files_dir
    file_dir = os.path.join(file_dir, collection_name)
    os.makedirs(file_dir, exist_ok=True)
    get_vstore_client().create_collection(
        collection_name, dimension=get_embedding_fn().dim
    )

    rets = []
    for file in files:
        filename = file.get_filename()
        file_location = os.path.join(file_dir, filename)
        if os.path.exists(file_location):
            get_logger().info(
                f"{filename} has already been indexed in {collection_name}."
            )
            rets.append(DocIndexingOutput(count=0, cost=0.0))
            continue

        with open(file_location, "wb") as f:
            content = file.get_file().read()
            f.write(content)
        loader = PyMuPDFLoader(file_location)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=get_settings().chunk_size,
            chunk_overlap=get_settings().chunk_overlap,
        )
        docs: list[Document] = loader.load_and_split(text_splitter=text_splitter)
        data = []
        vectors = get_embedding_fn()([doc.page_content for doc in docs])
        for i, doc in enumerate(docs):
            text = doc.page_content
            vector = vectors[i]
            data.append(
                KnwoledgeBaseEntity(id=i, vector=vector, text=text, subject=filename)
            )
        ret = get_vstore_client().add(data=data, collection_name=collection_name)
        rets.append(
            DocIndexingOutput(count=ret["insert_count"], cost=ret.get("cost", 0.0))
        )
    cum_count = np.sum([ret.count for ret in rets])
    cum_cost = np.sum([ret.cost for ret in rets])
    return DocIndexingOutput(count=cum_count, cost=cum_cost)


class RAGChatInput(BaseModel):
    history: List[Dict]
    question: str
    collection_name: Optional[str]
    context_limit: int = 3


class RAGChatOutput(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int
    query_context: str


@fast_service_manager.fast_service
def rag_chat(item: RAGChatInput, context: RequestContext = None) -> RAGChatOutput:
    question = item.question
    q_embedding = get_embedding_fn()([question])
    retrieval = get_vstore_client().search(
        query=q_embedding,
        collection_name=item.collection_name,
        limit=item.context_limit,
    )
    lines_w_distances = [
        [part["entity"]["text"], part["distance"]] for part in retrieval[0]
    ]
    query_context = "\n".join(
        [line_with_distance[0] for line_with_distance in lines_w_distances]
    )

    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {query_context}
    </context>
    <question>
    {question}
    </question>
    """

    messages = [{"role": "user", "content": "\n\n".join([SYSTEM_PROMPT, USER_PROMPT])}]
    chat_completions = chat_generate(
        LLMChatInput(model="google/gemma-2-2b-it", messages=messages)
    )
    answer = chat_completions.choices[0].message.content
    input_tokens = chat_completions.usage.prompt_tokens
    output_tokens = chat_completions.usage.completion_tokens
    return RAGChatOutput(
        answer=answer,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        query_context=query_context,
    )


class UndoIndexingInput(BaseModel):
    collection_name: str
    filenames: list[str]


class UndoIndexingOutput(BaseModel):
    collection_stats: Dict


@fast_service_manager.fast_service
def undo_indexing(
    item: UndoIndexingInput,
    context: RequestContext = None,
) -> UndoIndexingOutput:
    file_dir = get_settings().raw_files_dir
    file_dir = os.path.join(file_dir, item.collection_name)
    for filename in item.filenames:
        get_vstore_client().delete_subject(item.collection_name, filename)
        file_location = os.path.join(file_dir, filename)
        os.remove(file_location)
    stats = get_vstore_client().get_collection_stats(item.collection_name)
    return UndoIndexingOutput(collection_stats=stats)


class DropCollectionInput(BaseModel):
    collection_name: str


class DropCollectionOutput(BaseModel):
    collection_exists: bool


@fast_service_manager.fast_service
def drop_collection(
    item: DropCollectionInput,
    context: RequestContext = None,
) -> DropCollectionOutput:
    get_vstore_client().drop_collection(item.collection_name)
    exists = get_vstore_client().has_collection(item.collection_name)
    return DropCollectionOutput(collection_exists=exists)


class RAGE2EOutput(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int


@fast_service_manager.fast_service
def rag_e2e(
    # collection_name: Annotated[str, Form()],
    collection_name: str,
    files: list[FastServiceFile],
    # question: Annotated[str, Form()],
    # context_limit: Annotated[int, Form()] = 3,
    question: str,
    context_limit: int = 3,
    context: RequestContext = None,
) -> RAGE2EOutput:
    y1: DocIndexingOutput = indexing(
        collection_name=collection_name, files=files, context=context
    )
    get_logger().info(
        f"indexing files {[file.get_filename() for file in files]} done. {y1}"
    )
    y2: RAGChatOutput = rag_chat(
        RAGChatInput(
            history=[],
            question=question,
            collection_name=collection_name,
            context_limit=context_limit,
        ),
        context=context,
    )
    get_logger().info(f"process question {question} done. {y2}")

    y3: UndoIndexingOutput = undo_indexing(
        UndoIndexingInput(
            collection_name=collection_name,
            filenames=[file.get_filename() for file in files],
        ),
        context=context,
    )
    get_logger().info(
        f"undo indexing files {[file.get_filename() for file in files]} in {collection_name} done. {y3}"
    )

    y4: DropCollectionOutput = drop_collection(
        DropCollectionInput(collection_name=collection_name), context=context
    )
    get_logger().info(f"drop collection {collection_name} done. {y4}")

    return RAGE2EOutput(**y2.model_dump())
