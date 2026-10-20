from typing import Dict, Optional, List, Union, Iterable, Any
from abc import ABC, abstractmethod
import uuid
from pymilvus import MilvusClient
from pymilvus.client.constants import ConsistencyLevel
from langchain_core.documents import Document
from langchain_core.utils import xor_args
from .base import BaseVectorDBClient


class MilvusVectorDBClient(BaseVectorDBClient):
    def __init__(
        self,
        uri: str,
        collection_name: str,
        embedding_dimension: int,
        collection_metadata: Optional[Dict] = None,
    ):
        self._client = MilvusClient(uri)
        self._collection_name = collection_name

        self._dimension = embedding_dimension
        self._metric_type = "COSINE"
        self._collection_metadata = collection_metadata

    def __ensure_collection(self) -> None:
        if not self._client.has_collection(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=self._dimension,
                primary_field_name="id",
                id_type="str",
                auto_id=False,
                consistency_level="Strong",
                metric_type=self._metric_type,
                max_length=len(str(uuid.uuid4())) + 1,
                **(self._collection_metadata or {}),
            )
        self._client.load_collection(self._collection_name)

    def store(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        self.__ensure_collection()
        if len(embeddings) == 0:
            return []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        texts = list(texts)
        data = [
            {
                "id": ids[idx],
                "vector": embedding,
                "text": texts[idx],
                "metadata": metadata,
            }
            for idx, (embedding, metadata) in enumerate(
                zip(embeddings, metadatas or [{}] * len(embeddings))
            )
        ]
        insertion_size = kwargs.get("insertion_size", 6400000)
        embedding_dim = len(embeddings[0])
        batch_size = insertion_size // embedding_dim
        for i in range(0, len(data), batch_size):
            self._client.upsert(
                collection_name=self._collection_name,
                data=data[i : i + batch_size],
            )
        return ids

    def similarity_search(
        self,
        query: str,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        if not self._client.has_collection(self._collection_name):
            return []
        ret = self._client.search(
            collection_name=self._collection_name,
            data=[embedding],
            limit=k,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["text", "metadata"],
        )
        docs = [
            Document(
                page_content=doc["entity"]["text"],
                metadata={**doc["entity"]["metadata"], "distance": doc["distance"]},
            )
            for doc in ret[0]
        ]
        # sort by distance
        docs.sort(key=lambda doc: doc.metadata["distance"], reverse=True)
        return docs

    def batch_similarity_search(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[list[Document]]:
        if not self._client.has_collection(self._collection_name):
            return [[]]
        ret = self._client.search(
            collection_name=self._collection_name,
            data=embeddings,
            limit=k,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["text", "metadata"],
        )
        all_docs = [
            [
                Document(
                    page_content=doc["entity"]["text"],
                    metadata={**doc["entity"]["metadata"], "distance": doc["distance"]},
                )
                for doc in docs
            ]
            for docs in ret
        ]
        # sort by distance
        for docs in all_docs:
            docs.sort(key=lambda doc: doc.metadata["distance"], reverse=True)
        return all_docs

    def drop_collection(self):
        if self._client.has_collection(self._collection_name):
            self._client.drop_collection(self._collection_name)

    def close(self) -> None:
        self._client.close()
