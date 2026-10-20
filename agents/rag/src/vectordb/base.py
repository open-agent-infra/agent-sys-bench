from typing import Dict, Optional, List, Union, Iterable, Any
from abc import ABC, abstractmethod
import uuid
import chromadb
from chromadb.api import ClientAPI
from langchain_core.documents import Document


class BaseVectorDBClient(ABC):
    def store(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        pass

    def add_documents(
        self, documents: list[Document], embeddings: list[list[float]], **kwargs: Any
    ) -> list[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return self.store(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            **kwargs,
        )

    def search(
        self, query: str, embedding: list[float], search_type: str, **kwargs: Any
    ) -> list[Document]:
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = self.similarity_search_with_relevance_scores(
                query=query, embedding=embedding, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        elif search_type == "mmr":
            return self.max_marginal_relevance_search(
                query=query, embedding=embedding, **kwargs
            )
        else:
            msg = (
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity', 'similarity_score_threshold'"
                " or 'mmr'."
            )
            raise ValueError(msg)

    def similarity_search(
        self, query: str, embedding: list[float], **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError

    def batch_similarity_search(
        self, queries: list[str], embeddings: list[list[float]], **kwargs: Any
    ) -> list[list[Document]]:
        raise NotImplementedError

    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError

    def similarity_search_with_relevance_scores(
        self, query: str, embedding: list[float], **kwargs: Any
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError

    def max_marginal_relevance_search(
        self, query: str, embedding: list[float], **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError

    def drop_collection(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
