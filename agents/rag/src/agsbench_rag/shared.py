from typing import Dict, Optional, Union, List
from pydantic import BaseModel
from fast_service import FastServiceManager
from langchain_core.documents import Document
from .settings import (
    EmbeddingSettings,
    GenerationSettings,
    RetrieverSettings,
    SplitterSettings,
    VDBSettings,
)

agsbench_rag_fsm = FastServiceManager()


class DocLoadingInput(BaseModel):
    type: str
    uri: Union[str, List[str]]


class DocLoadingOutput(BaseModel):
    docs: list[Document]


class DocSplittingInput(BaseModel):
    docs: list[Document]
    splitter_model: str = "recursive-character"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class DocSplittingOutput(BaseModel):
    docs: list[Document]


class SituatingChunksInput(BaseModel):
    docs: list[Document]
    chunks: list[Document]


class SituatingChunksOutput(BaseModel):
    chunks: list[Document]


class EmbeddingDocsInput(BaseModel):
    docs: list[Document]
    embedding_model: str
    embedding_device: str
    embedding_batch_size: int = 32


class EmbeddingDocsOutput(BaseModel):
    embeddings: list[list[float]]


class EmbeddingQueryInput(BaseModel):
    query: str
    embedding_model: str
    embedding_device: str


class EmbeddingQueryOutput(BaseModel):
    embedding: list[float]


class VDBStoringInput(BaseModel):
    vdb_type: str
    vdb_uri: str
    collection_name: str
    docs: list[Document]
    embeddings: list[list[float]]
    metadatas: Optional[list[dict]] = None
    ids: Optional[list[str]] = None


class VDBStoringOutput(BaseModel):
    count: int
    status: int = 0


class VDBSearchingInput(BaseModel):
    vdb_type: str
    vdb_uri: str
    collection_name: str
    query: str
    embedding: list[float]
    k: int = 4
    filter: Optional[Dict[str, str]] = None


class VDBSearchingOutput(BaseModel):
    docs: list[Document]


class IndexingInput(DocLoadingInput):
    uri: Union[str, List[str]]
    type: str
    splitter_model: str = "recursive-character"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vdb_type: str = "chromadb"
    vdb_uri: str = "http://localhost:8800"
    collection_name: str = "rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32


class IndexingOutput(BaseModel):
    count: int
    status: int = 0


class CollectionClearingInput(BaseModel):
    vdb_type: str = "chromadb"
    vdb_uri: str = "http://localhost:8800"
    collection_name: str = "rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"


class QueryTranslationInput(BaseModel):
    query: str
    method: str = "MultiQuery"


class QueryTranslationOutput(BaseModel):
    queries: list[str]
    qtypes: list[str]
    usage: Optional[dict] = None


class PreRetrievalInput(QueryTranslationInput):
    pass


class PreRetrievalOutput(QueryTranslationOutput):
    pass


class RetrievalInput(BaseModel):
    query: str
    search_type: str = "similarity"
    search_args: Optional[Dict] = None
    retriever_type: str = "VectorStore"
    vdb_type: str = "chromadb"
    vdb_uri: str = "http://localhost:8800"
    collection_name: str = "rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32


class RetrievalOutput(BaseModel):
    docs: list[Document]


class PostRetrievalInput(BaseModel):
    retrieved: Union[list[RetrievalOutput], RetrievalOutput]
    fusion_policy: str = "reciprocal_rank"  # [unique, all]

    reranking_args: Optional[Dict] = None
    compression_args: Optional[Dict] = None
    selection_args: Optional[Dict] = None


class PostRetrievalOutput(BaseModel):
    docs: list[Document]


class GenerationInput(BaseModel):
    query: str
    context: str
    method: str = "normal"
    method_args: Optional[Dict] = None


class GenerationOutput(BaseModel):
    answer: str
    usage: Optional[dict] = None


class RAGChattingInput(BaseModel):
    query: str
    vdb_settings: Optional[VDBSettings] = None
    embedding_settings: Optional[EmbeddingSettings] = None
    retriever_settings: Optional[RetrieverSettings] = None
    generation_settings: Optional[GenerationSettings] = None


class RAGChattingOutput(BaseModel):
    answer: str


class JudgingOutput(BaseModel):
    judgement: bool
    usage: Optional[dict] = None


class RetrievalJudgingInput(BaseModel):
    question: str


class DocRelevanceJudgingInput(BaseModel):
    question: str
    context: str


class HallucinationJudgingInput(BaseModel):
    answer: str
    context: str


class AnswerJudgingInput(BaseModel):
    question: str
    answer: str


class RAGE2EInput(BaseModel):
    query: str
    uri: Union[str, List[str]]
    type: str
    splitter_settings: Optional[SplitterSettings] = None
    vdb_settings: Optional[VDBSettings] = None
    embedding_settings: Optional[EmbeddingSettings] = None
    retriever_settings: Optional[RetrieverSettings] = None
    generation_settings: Optional[GenerationSettings] = None
    clear_collection: bool = True


class RAGE2EOutput(BaseModel):
    answer: str
