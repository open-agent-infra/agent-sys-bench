import os
import yaml
from typing import Dict, Optional
from pydantic_settings import BaseSettings


class DefaultSettings(BaseSettings):
    app_name: str = "RAG"

    def __repr_args__(self):
        # Keep API keys out of logs.
        for k, v in super().__repr_args__():
            yield k, ("***" if k and k.endswith("api_key") else v)

    parallel_rag: bool = False
    early_stopping: bool = True

    # llm settings
    openai_api_key: str = "EMPTY"
    openai_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_temperature: float = 0.0
    llm_top_p: float = 0.9
    llm_seed: int = 0

    # loading settings
    loading_persistency: Optional[str] = None
    loading_engine: Optional[str] = None
    num_loading_engines: Optional[int] = 1
    loading_lock_name: Optional[str] = None  # None, global, loading

    # chunking settings
    splitter_model: str = "recursive-character-tiktoken"
    chunk_size: int = 256
    chunk_overlap: int = 20
    chunking_engine: Optional[str] = None
    num_chunking_engines: Optional[int] = 1
    chunking_lock_name: Optional[str] = None  # None, global, chunking

    situating_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    situating_llm_temperature: float = 0.0
    situating_llm_top_p: float = 0.9
    situating_llm_seed: int = 0
    situating_device: str = "http://localhost:8000/v1"
    situating_api_key: str = "EMPTY"
    situating_batch_size: int = 0
    parallel_situating: bool = True

    # vdb settings
    vdb_type: str = "milvus"  # or "chromadb"
    vdb_uri: str = "http://localhost:19530"
    collection_name: str = "test"
    insertion_size: int = 6400000

    # embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # retriever settings
    retriever_type: str = "VectorStore"
    search_type: str = "similarity"
    search_args: Optional[Dict] = None
    parallel_retrieval: bool = True
    parallel_search: bool = False

    # post retrieval settings

    # rank fusion setting
    fusion_policy: str = "reciprocal_rank"

    # reranker settings
    reranker_model: Optional[str] = None
    reranker_engine: Optional[str] = None
    reranker_args: Optional[Dict] = None

    # generation settings
    generation_method: str = "normal"
    generation_method_args: Optional[Dict] = None

    uvicorn_nworkers: Optional[int] = None

    log_level: str = "DEBUG"
    log_file: str = ".cache/debug.log"


DEFAULT_SETTINGS: DefaultSettings = DefaultSettings()


def init_settings(setting_file: str = None):
    global DEFAULT_SETTINGS

    if setting_file is not None and os.path.exists(setting_file):
        with open(setting_file, "r", encoding="utf-8") as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        if data is not None:
            DEFAULT_SETTINGS = DefaultSettings.model_construct(**data)
        else:
            print(f"empty file: {setting_file}, use default one.")
    else:
        print(f"File not found: {setting_file}, use default one.")
    print(f"init_settings: {DEFAULT_SETTINGS}")


def get_settings():
    global DEFAULT_SETTINGS
    return DEFAULT_SETTINGS


class LLMSettings(BaseSettings):
    llm_model: str
    llm_temperature: float
    llm_top_p: float
    llm_seed: int
    openai_api_key: str
    openai_base_url: str

    @staticmethod
    def from_default_settings():
        return LLMSettings(
            llm_model=get_settings().llm_model,
            llm_temperature=get_settings().llm_temperature,
            llm_top_p=get_settings().llm_top_p,
            llm_seed=get_settings().llm_seed,
            openai_api_key=get_settings().openai_api_key,
            openai_base_url=get_settings().openai_base_url,
        )

    # make the class hashable
    def __hash__(self):
        return hash(
            (
                self.llm_model,
                self.llm_temperature,
                self.llm_top_p,
                self.llm_seed,
                self.openai_api_key,
                self.openai_base_url,
            )
        )


class SituatingLLMSettings(LLMSettings):
    @staticmethod
    def from_default_settings():
        return LLMSettings(
            llm_model=get_settings().situating_llm_model,
            llm_temperature=get_settings().situating_llm_temperature,
            llm_top_p=get_settings().situating_llm_top_p,
            llm_seed=get_settings().situating_llm_seed,
            openai_api_key=get_settings().situating_api_key,
            openai_base_url=get_settings().situating_device,
        )


class SplitterSettings(BaseSettings):
    splitter_model: str
    chunk_size: int
    chunk_overlap: int

    @staticmethod
    def from_default_settings():
        return SplitterSettings(
            splitter_model=get_settings().splitter_model,
            chunk_size=get_settings().chunk_size,
            chunk_overlap=get_settings().chunk_overlap,
        )


class VDBSettings(BaseSettings):
    vdb_type: str
    vdb_uri: str
    collection_name: str

    @staticmethod
    def from_default_settings():
        return VDBSettings(
            vdb_type=get_settings().vdb_type,
            vdb_uri=get_settings().vdb_uri,
            collection_name=get_settings().collection_name,
        )


class EmbeddingSettings(BaseSettings):
    embedding_model: str
    embedding_device: str
    embedding_batch_size: int

    @staticmethod
    def from_default_settings():
        return EmbeddingSettings(
            embedding_model=get_settings().embedding_model,
            embedding_device=get_settings().embedding_device,
            embedding_batch_size=get_settings().embedding_batch_size,
        )


class RetrieverSettings(BaseSettings):
    retriever_type: str
    search_type: str
    search_args: Optional[Dict]

    @staticmethod
    def from_default_settings():
        return RetrieverSettings(
            retriever_type=get_settings().retriever_type,
            search_type=get_settings().search_type,
            search_args=get_settings().search_args,
        )


class GenerationSettings(BaseSettings):
    generation_method: str
    generation_method_args: Optional[Dict]

    @staticmethod
    def from_default_settings():
        return GenerationSettings(
            generation_method=get_settings().generation_method,
            generation_method_args=get_settings().generation_method_args,
        )


class LoggingSettings(BaseSettings):
    log_level: str
    log_file: str

    @staticmethod
    def from_default_settings():
        return LoggingSettings(
            log_level=get_settings().log_level,
            log_file=get_settings().log_file,
        )
