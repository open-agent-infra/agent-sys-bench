"""Shared components for LangChain Deep Research System."""

from fast_service import FastServiceManager, RequestContext

agsbench_deep_research_fsm = FastServiceManager()

DEFAULT_LLM_MODEL = "Qwen3/Qwen-8B"
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
DEFAULT_SPLITTING_MODEL = "langchain-recursive-character"
DEFAULT_VDB_COLLECTION = "langchain_deep_research"
DEFAULT_SEARCH_POST_PROCESSING_CONFIG = {
    "operation": "summarize",
    "max_char_to_include": 30_000,
    "splitting_model": DEFAULT_SPLITTING_MODEL,
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "vdb_collection": DEFAULT_VDB_COLLECTION,
    "vdb_topk": 5,
    "summarize_after_rerank": False,
    "parallelize_summarization": True,
    "parallelize_summarization_n_workers": 32,
    "parallelize_split_and_rerank": True,
    "parallelize_split_and_rerank_n_workers": 32,
}
