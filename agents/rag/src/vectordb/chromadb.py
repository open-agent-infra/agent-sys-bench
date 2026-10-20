from typing import Dict, Optional, List, Union, Iterable, Any
import uuid
import chromadb
from langchain_core.documents import Document
from .base import BaseVectorDBClient


def _results_to_docs_and_scores(results: Any) -> List[tuple[Document, float]]:
    return [
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class ChromaVectorDBClient(BaseVectorDBClient):
    def __init__(
        self,
        uri: str,
        collection_name: str,
        collection_metadata: Optional[Dict] = None,
    ):
        host, port = uri.split("://")[1].split(":")
        self._client = chromadb.HttpClient(host=host, port=int(port))
        self._collection_name = collection_name
        self._collection_metadata = collection_metadata
        self.__ensure_collection()

    def __ensure_collection(self) -> None:
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=None,
            metadata=self._collection_metadata,
        )

    def store(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        self.__ensure_collection()
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        texts = list(texts)
        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,  # type: ignore
                        embeddings=embeddings_with_metadatas,  # type: ignore
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,  # type: ignore
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,  # type: ignore
                documents=texts,
                ids=ids,
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
        docs_and_scores = self.similarity_search_with_score(
            query, embedding, k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def __query_collection(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[List[Document], chromadb.QueryResult]:
        self.__ensure_collection()
        return self._collection.query(
            query_embeddings=query_embeddings,  # type: ignore
            n_results=n_results,
            where=where,  # type: ignore
            where_document=where_document,  # type: ignore
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        results = self.__query_collection(
            query_embeddings=[embedding],
            n_results=k,
            where=filter,
            where_document=where_document,
            **kwargs,
        )

        return _results_to_docs_and_scores(results)

    def drop_collection(self):
        self.__ensure_collection()
        self._client.delete_collection(self._collection_name)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    vdb = ChromaVectorDBClient(
        uri="http://localhost:8800",
        collection_name="rag",
    )
    vdb.drop_collection()
    vdb.store(
        texts=["hello", "world"],
        embeddings=[[1, 2], [3, 4]],
    )
    print(vdb.similarity_search("hello", [1, 2]))
    vdb.drop_collection()
    print("done")
