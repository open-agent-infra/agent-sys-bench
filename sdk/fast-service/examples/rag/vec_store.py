from typing import List, Dict, Union
from abc import abstractmethod
from rag_types import KnwoledgeBaseEntity
from pymilvus import MilvusClient


class BaseVectorStore:
    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int, exists_ok=True):
        """ """

    @abstractmethod
    def get_collection_stats(self, collection_name: str):
        """ """

    @abstractmethod
    def drop_collection(self, collection_name: str):
        """ """

    @abstractmethod
    def add(
        self, data: Union[KnwoledgeBaseEntity, List[KnwoledgeBaseEntity]], **kwargs
    ):
        """ """

    @abstractmethod
    def delete_subject(self, collection_name: str, subject: str, **kwargs):
        """ """

    @abstractmethod
    def search(
        self,
        query: Union[List[list], list],
        limit: int = 10,
        output_fields: List[str] = None,
        **kwargs,
    ) -> List[List[Dict]]:
        """ """


class MilvusVectorStore(BaseVectorStore):
    def __init__(self, db_uri: str, **kwargs) -> None:
        self.db_uri = db_uri
        self.client = MilvusClient(db_uri)

    def create_collection(self, collection_name: str, dimension: int, exists_ok=True):
        if self.client.has_collection(collection_name):
            if not exists_ok:
                collection_stats = self.client.get_collection_stats(collection_name)
                raise Exception(
                    f"collection {collection_name} already exists, stats: {collection_stats}"
                )
        else:
            self.client.create_collection(collection_name, dimension)

    def has_collection(self, collection_name: str):
        return self.client.has_collection(collection_name)

    def get_collection_stats(self, collection_name: str) -> Dict:
        return self.client.get_collection_stats(collection_name)

    def drop_collection(self, collection_name: str):
        self.client.drop_collection(collection_name)

    def add(
        self,
        data: Union[KnwoledgeBaseEntity, List[KnwoledgeBaseEntity]],
        collection_name: str,
        **kwargs,
    ) -> Dict:
        ret = self.client.insert(collection_name=collection_name, data=data)
        self.client.refresh_load(collection_name=collection_name)
        return ret

    def delete_subject(self, collection_name: str, subject: str, **kwargs):
        filter = f'subject like "{subject}"'
        self.client.delete(collection_name, filter=filter)

    def search(
        self,
        query: Union[List[list], list],
        collection_name: str,
        limit: int = 10,
        output_fields: List[str] = None,
        **kwargs,
    ) -> List[List[Dict]]:
        if output_fields is None:
            output_fields = ["text"]
        ret = self.client.search(
            collection_name=collection_name,
            data=query,
            limit=limit,
            output_fields=output_fields,
            **kwargs,
        )
        return ret
