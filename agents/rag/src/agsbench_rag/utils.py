import time
from functools import lru_cache
from typing import Optional, Dict
import logging
import asyncio
import threading
from langchain.text_splitter import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
import chromadb
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import InfinityEmbeddings
from langchain_milvus import Milvus
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from vectordb import BaseVectorDBClient, ChromaVectorDBClient, MilvusVectorDBClient
from .settings import LLMSettings, LoggingSettings
from .settings import SplitterSettings, EmbeddingSettings, VDBSettings
from .settings import get_settings


def async_run(task):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(task)


@lru_cache
def get_lock(name: str) -> threading.Lock:
    return threading.Lock()


@lru_cache
def get_logger(settings: Optional[LoggingSettings] = None):
    if settings is None:
        settings = LoggingSettings.from_default_settings()
    naive_rag_logger = logging.getLogger(name="rag")
    log_level = settings.log_level
    naive_rag_logger.setLevel(log_level)
    log_file = settings.log_file
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # ch = logging.StreamHandler()
    # ch.setLevel(log_level)
    # ch.setFormatter(log_format)
    # naive_rag_logger.addHandler(ch)
    # ensure the log dir exists (the FileHandler will not create it)
    import os as _os
    _log_dir = _os.path.dirname(log_file)
    if _log_dir and not _os.path.exists(_log_dir):
        _os.makedirs(_log_dir, exist_ok=True)
    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(log_level)
    fh.setFormatter(log_format)
    naive_rag_logger.addHandler(fh)
    return naive_rag_logger


@lru_cache
def get_llm(settings: Optional[LLMSettings] = None) -> BaseChatModel:
    if settings is None:
        settings = LLMSettings.from_default_settings()
    try:
        llm = ChatOpenAI(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            seed=settings.llm_seed,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    except Exception as e:
        print("Failed to initialize ChatOpenAI: ", e)
        raise NotImplementedError(f"Unsupported llm_model: {settings.llm_model}")
    return llm


@lru_cache
def get_splitter(
    splitter_model: str, chunk_size: int, chunk_overlap: int
) -> TextSplitter:
    st = time.time()
    if splitter_model == "recursive-character":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_model == "recursive-character-embedding":
        embedding_model = get_settings().embedding_model
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_model == "sentence-transformers":
        embedding_model = get_settings().embedding_model
        from langchain_text_splitters import SentenceTransformersTokenTextSplitter

        splitter = SentenceTransformersTokenTextSplitter(
            model_name=embedding_model,
            tokens_per_chunk=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif splitter_model == "recursive-character-tiktoken":
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_model == "recursive-character-html":
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_model == "recursive-character-html-tiktoken":
        separators = RecursiveCharacterTextSplitter.get_separators_for_language(
            Language.HTML
        )
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splitter._separators = separators
        splitter._is_separator_regex = True
    else:
        raise NotImplementedError(f"Unsupported splitter model: {splitter_model}")
    et = time.time()
    get_logger().debug(f"initialization cost of splitter: {et - st}")
    return splitter


@lru_cache
def get_embedding_fn(embedding_model: str, embedding_device: str) -> Embeddings:
    if embedding_device == "TEI":
        assert embedding_model.startswith("http"), "TEI requires a URL"
        return HuggingFaceEndpointEmbeddings(
            model=embedding_model, task="feature-extraction"
        )
    elif embedding_device == "Infinity":
        url, model = embedding_model.split(":::")
        return InfinityEmbeddings(model=model, infinity_api_url=url)
    elif embedding_device == "vLLM":
        url, model = embedding_model.split(":::")
        return OpenAIEmbeddings(
            model=model,
            base_url=url,
            check_embedding_ctx_length=False,
            chunk_size=65536,
        )
    elif embedding_device.startswith("http"):
        return OpenAIEmbeddings(
            model=embedding_model,
            base_url=embedding_device,
            check_embedding_ctx_length=False,
            chunk_size=65536,
        )
    elif embedding_model == "all-MiniLM-L6-v2":
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": embedding_device},
        )
    elif embedding_model == "text-embedding-ada-002":
        return OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=65536)
    else:
        raise NotImplementedError(f"Unsupported embedding model: {embedding_model}")


@lru_cache
def get_reranker(model: str, engine: str = "cohere") -> CohereRerank:
    import cohere

    if engine == "cohere":
        return CohereRerank(model=model)
    elif engine.startswith("http"):
        if engine.endswith("/v1"):
            engine = engine[:-3]

        # # Create Cohere ClientV2 with custom base URL
        # try:
        #     # Try using ClientV2 (newer cohere versions)
        #     cohere_client = cohere.ClientV2(api_key="EMPTY", base_url=engine)
        # except AttributeError:
        #     # Fallback to older Client for backward compatibility
        cohere_client = cohere.Client(api_key="EMPTY", base_url=engine)

        # Don't pass top_n=None to avoid HTTP 400 error
        return CohereRerank(client=cohere_client, model=model)
    else:
        raise NotImplementedError(f"Unsupported reranking engine: {engine}")


vdb_clients: Dict[str, BaseVectorDBClient] = {}  # type: ignore


def get_vdb_client(
    vdb_type: str,
    vdb_uri: str,
    collection_name: str,
    embedding_dimension: Optional[int] = None,
) -> BaseVectorDBClient:
    cache_key = f"{vdb_type}:{vdb_uri}:{collection_name}"
    lock = get_lock(cache_key)
    with lock:
        if cache_key in vdb_clients:
            return vdb_clients[cache_key]
    if vdb_type == "chromadb":
        client = ChromaVectorDBClient(uri=vdb_uri, collection_name=collection_name)
        vdb_clients[cache_key] = client
        return client
    elif vdb_type == "milvus":
        client = MilvusVectorDBClient(
            uri=vdb_uri,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
        )
        vdb_clients[cache_key] = client
        return client
    else:
        raise NotImplementedError(f"Unsupported vdb_type: {vdb_type}")


def close_vdb_client(
    vdb_type: str,
    vdb_uri: str,
    collection_name: str,
    embedding_dimension: Optional[int] = None,
) -> BaseVectorDBClient:
    cache_key = f"{vdb_type}:{vdb_uri}:{collection_name}"
    lock = get_lock(cache_key)
    with lock:
        if cache_key in vdb_clients:
            vdb_clients[cache_key].close()
            del vdb_clients[cache_key]


def warm_up(
    splitter_settings: SplitterSettings,
    embedding_settings: EmbeddingSettings,
    vdb_settings: VDBSettings,
):
    get_llm()
    get_splitter(
        splitter_model=splitter_settings.splitter_model,
        chunk_size=splitter_settings.chunk_size,
        chunk_overlap=splitter_settings.chunk_overlap,
    )
    fn = get_embedding_fn(
        embedding_model=embedding_settings.embedding_model,
        embedding_device=embedding_settings.embedding_device,
    )
    get_vdb_client(
        vdb_type=vdb_settings.vdb_type,
        vdb_uri=vdb_settings.vdb_uri,
        collection_name=vdb_settings.collection_name,
        embedding_dimension=len(fn.embed_query("Warm Up")),
    )
