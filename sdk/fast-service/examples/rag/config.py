from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging


class Settings(BaseSettings):
    app_name: str = "Naive RAG"

    vdb_uri: str = "http://localhost:19530"
    vdb_collection: str = "test"

    embedding_device: str = "cuda"

    raw_files_dir: str = ".cache/files"

    chunk_size: int = 256
    chunk_overlap: int = 128

    openai_api_key: str = "EMPTY"
    openai_base_url: str = "http://localhost:8000/v1"

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings():
    return Settings()


@lru_cache
def get_logger():
    naive_rag_logger = logging.getLogger(name="naive_rag")
    naive_rag_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=".cache/debug.log")
    fh.setLevel(logging.DEBUG)
    naive_rag_logger.addHandler(fh)
    return naive_rag_logger
