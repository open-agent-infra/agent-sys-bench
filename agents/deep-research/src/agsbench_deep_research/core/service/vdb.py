"""
VectorDB service module for vector database interactions.

This module provides service implementations for interacting with
various vector databases, including remote Milvus and local Milvus Lite.
"""

import logging
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Union,
    Literal,
    Tuple,
    cast,
    Iterable,
    Awaitable,
    override,
)
from pydantic import BaseModel, Field
import numpy as np
import uuid
import os
import asyncio
from abc import abstractmethod
from pymilvus import MilvusClient, utility, AsyncMilvusClient
from ..utils import compute_cosine_similarity

# Configure logger
logger = logging.getLogger(__name__)

from .base import (
    BaseServiceInstanceConfig,
    BaseServiceInstance,
    BaseService,
    BaseServiceRequest,
    BaseServiceResponse,
)


class VectorDBServiceRequest(BaseServiceRequest):
    """Request model for VectorDB service operations."""

    service_name: str = "vdb"

    def get_statistics(self):
        return super().get_statistics()


class VectorDBServiceResponse(BaseServiceResponse):
    """Response model for VectorDB service operations."""

    service_name: str = "vdb"

    def get_statistics(self):
        return super().get_statistics()


class VectorDBServiceCreateCollectionRequest(VectorDBServiceRequest):
    """Request model for vector database create collection operations.

    Attributes:
        collection_name: Name of the collection to create
        dim: Dimensionality of the vectors
        kwargs: Additional keyword arguments for the create collection
        raw_data: Additional raw data for the request
    """

    operation_name: str = "create_collection"
    collection_name: str = Field(..., description="Name of the collection to create")
    dim: int = Field(default=1024, description="Dimensionality of the vectors")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the create collection",
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "collection_name": self.collection_name,
                "dim": self.dim,
                "kwargs": self.kwargs,
            }
        )
        return stats


class VectorDBServiceDropCollectionRequest(VectorDBServiceRequest):
    """Request model for vector database drop collection operations.

    Attributes:
        collection_name: Name of the collection to drop
        kwargs: Additional keyword arguments for the drop collection
        raw_data: Additional raw data for the request
    """

    operation_name: str = "drop_collection"
    collection_name: str = Field(..., description="Name of the collection to drop")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the drop collection",
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "collection_name": self.collection_name,
                "kwargs": self.kwargs,
            }
        )
        return stats


class VectorDBServiceListCollectionsRequest(VectorDBServiceRequest):
    """Request model for vector database list collections operations.

    Attributes:
        raw_data: Additional raw data for the request
    """

    operation_name: str = "list_collections"
    raw_data: Dict[str, Any] = Field(
        default_factory=dict, description="Additional raw data for the request"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "raw_data_keys": (
                    list(self.raw_data.keys())
                    if isinstance(self.raw_data, dict)
                    else []
                ),
            }
        )
        return stats


class VectorDBServiceStoreRequest(VectorDBServiceRequest):
    """Request model for vector database store operations.

    Attributes:
        texts: List of texts to store
        text_embeddings: List of embeddings to store
        text_metadatas: Optional list of metadata dictionaries
        ids: Optional list of IDs (if not provided, UUIDs will be generated)
        collection_name: Optional collection name (uses instance collection_name if not provided)
        kwargs: Additional keyword arguments for the store operation
        raw_data: Additional raw data for the request
    """

    operation_name: str = "store"
    texts: List[str] = Field(..., description="List of texts to store")
    text_embeddings: List[List[float]] = Field(
        ..., description="List of embeddings to store"
    )
    text_metadatas: Optional[List[dict]] = Field(
        default=None, description="Optional metadata for each text"
    )
    ids: Optional[List[str]] = Field(default=None, description="Optional list of IDs")
    collection_name: Optional[str] = Field(
        default=None, description="Collection name to store in"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments for the store"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "texts_count": len(self.texts),
                "text_embeddings_count": len(self.text_embeddings),
                "text_metadatas_count": (
                    len(self.text_metadatas) if self.text_metadatas else 0
                ),
                "ids_count": len(self.ids) if self.ids else 0,
                "text_embedding_dim": (
                    len(self.text_embeddings[0])
                    if (self.text_embeddings and len(self.text_embeddings[0]) > 0)
                    else None
                ),
                "text_metadata_keys": (
                    list(self.text_metadatas[0].keys()) if self.text_metadatas else []
                ),
                "collection_name": self.collection_name,
                "kwargs": self.kwargs,
            }
        )
        return stats


class VectorDBServiceSearchRequest(VectorDBServiceRequest):
    """Request model for vector database search operations.

    Attributes:
        texts: List of query texts
        text_embeddings: List of query embeddings
        k: Number of results to return per query
        collection_name: Optional collection name (uses instance collection_name if not provided)
        kwargs: Additional keyword arguments for the search
        raw_data: Additional raw data for the request
    """

    operation_name: str = "search"
    texts: Optional[List[str]] = Field(default=None, description="List of query texts")
    text_embeddings: List[list[float]] = Field(
        ..., description="List of query embeddings"
    )
    k: int = Field(default=5, description="Number of results to return per query")
    collection_name: Optional[str] = Field(
        default=None, description="Collection name to search in"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments for the search"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "texts_count": len(self.texts),
                "text_embeddings_count": len(self.text_embeddings),
                "k": self.k,
                "collection_name": self.collection_name,
                "text_embedding_dim": (
                    len(self.text_embeddings[0])
                    if (self.text_embeddings and len(self.text_embeddings[0]) > 0)
                    else None
                ),
                "kwargs": self.kwargs,
            }
        )
        return stats


class VectorDBServiceCreateCollectionResponse(VectorDBServiceResponse):
    """Response model for vector database create collection operations.

    Attributes:
        success: Whether the collection creation was successful
        error: Error message if collection creation failed
        raw_data: Additional raw data for the response
    """

    operation_name: str = "create_collection"
    success: bool = Field(
        default=True, description="Whether the collection was created"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if collection creation failed"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "operation_name": self.operation_name,
                "success": self.success,
                "error": self.error,
            }
        )
        return stats


class VectorDBServiceDropCollectionResponse(VectorDBServiceResponse):
    """Response model for vector database drop collection operations.

    Attributes:
        success: Whether the collection drop was successful
        error: Error message if collection drop failed
        raw_data: Additional raw data for the response
    """

    operation_name: str = "drop_collection"
    success: bool = Field(
        default=True, description="Whether the collection was dropped"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if collection drop failed"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "operation_name": self.operation_name,
                "success": self.success,
                "error": self.error,
            }
        )
        return stats


class VectorDBServiceListCollectionsResponse(VectorDBServiceResponse):
    """Response model for vector database list collections operations.

    Attributes:
        collections: List of collection names
        raw_data: Additional raw data for the response
    """

    operation_name: str = "list_collections"
    collections: List[str] = Field(
        default_factory=list, description="List of collection names"
    )
    raw_data: Dict[str, Any] = Field(
        default_factory=dict, description="Additional raw data for the response"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "operation_name": self.operation_name,
                "collections_count": len(self.collections),
                "collections": self.collections,
                "raw_data_keys": (
                    list(self.raw_data.keys())
                    if isinstance(self.raw_data, dict)
                    else []
                ),
            }
        )
        return stats


class VectorDBServiceStoreResponse(VectorDBServiceResponse):
    """Response model for vector database store operations.

    Attributes:
        ids: List of IDs of the stored vectors
        success: Whether the store operation was successful
        error: Error message if store operation failed
        raw_data: Additional raw data for the response
    """

    operation_name: str = "store"
    ids: List[str] = Field(default_factory=list, description="List of stored IDs")
    success: bool = Field(default=True, description="Whether the store was successful")
    error: Optional[str] = Field(
        default=None, description="Error message if store failed"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "operation_name": self.operation_name,
                "ids_count": len(self.ids),
                "success": self.success,
                "error": self.error,
            }
        )
        return stats


class VDBSearchResult(BaseModel):
    """Model for a single search result in vector database search operations.

    Attributes:
        id: ID of the document
        text: Text content of the document
        metadata: Optional Metadata associated with the document
        score: Similarity score or distance of the document from the query vector
    """

    id: str = Field(..., description="ID of the document")
    text: str = Field(..., description="Text content of the document")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata associated with the document"
    )
    score: float = Field(
        ...,
        description="Similarity score or distance of the document from the query vector",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the search result."""
        return {
            "id": self.id,
            "text_length": len(self.text),
            "metadata_keys": (list(self.metadata.keys()) if self.metadata else []),
            "score": self.score,
        }


class VectorDBServiceSearchResponse(VectorDBServiceResponse):
    """Response model for vector database search operations.

    Attributes:
        results: List of List of search results, each result is an instance of VDBSearchResult
        success: Whether the search operation was successful
        error: Error message if search operation failed
        raw_data: Additional raw data for the response
    """

    operation_name: str = "search"
    results: List[List[VDBSearchResult]] = Field(
        default_factory=list, description="List of search results"
    )
    success: bool = Field(default=True, description="Whether the search was successful")
    error: Optional[str] = Field(
        default=None, description="Error message if search failed"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "operation_name": self.operation_name,
                "results_count": len(self.results),
                "results": [
                    [result.get_statistics() for result in result_list]
                    for result_list in self.results
                ],
                "success": self.success,
                "error": self.error,
            }
        )
        return stats


class VectorDBServiceInstanceConfig(BaseServiceInstanceConfig):
    """Configuration for a VectorDB service instance.

    Attributes:
        model: Model name or identifier
        db_type: Type of the vector database ('milvus' or 'milvus_lite')
        collection_name: Name of the collection to work with
        dim: Dimensionality of the vectors
        index_params: Parameters for vector index
        search_params: Parameters for vector search
        metric_type: Distance metric type ('L2', 'IP', 'JACCARD', etc.)
        location: Connection string for the database (e.g., 'localhost:19530')
    """

    model: str = "default_model"
    db_type: Literal["milvus", "milvus_lite", "simple"] = "milvus"
    collection_name: str = "default_collection"
    dim: int = 1024
    index_params: Dict[str, Any] = Field(default_factory=dict)
    search_params: Dict[str, Any] = Field(default_factory=dict)
    metric_type: str = "L2"
    location: str = "localhost:19530"


class VectorDBServiceInstance(BaseServiceInstance):
    """Base class for VectorDB service instances.

    This abstract class defines the interface that all VectorDB service
    implementations must provide for vector database operations.
    """

    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: VectorDBServiceInstanceConfig,
    ):
        super().__init__("vdb", instance_name, group_name, config)
        self.collection = None
        self.collection_name = getattr(config, "collection_name", "default_collection")
        self.dim = getattr(config, "dim", 1024)
        self.metric_type = getattr(config, "metric_type", "L2")

        # Initialize index parameters with defaults if not provided
        self.index_params = getattr(config, "index_params", None) or {
            "index_type": "IVF_FLAT",
            "metric_type": self.metric_type,
            "params": {"nlist": 128},
        }

        # Initialize search parameters with defaults if not provided
        self.search_params = getattr(config, "search_params", None) or {
            "metric_type": self.metric_type,
            "params": {"nprobe": 10},
        }

    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            bool: True if the collection exists, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections.

        Returns:
            List of collection names
        """
        raise NotImplementedError

    @abstractmethod
    def create_collection(self, collection_name: str, dim: int, **kwargs) -> bool:
        """Create a new collection.

        Args:
            collection_name: Name of the collection to create
            dim: Dimensionality of the vectors
            **kwargs: Additional collection parameters

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def store(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = None,
        **kwargs,
    ) -> List[str]:
        """Insert vectors into the collection.

        Args:
            texts: List of texts to insert
            embeddings: List of embeddings to insert
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional arguments

        Returns:
            List of inserted IDs
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_texts: List[str],
        query_embeddings: List[List[float]],
        k: int = 5,
        collection_name: str = None,
        **kwargs,
    ) -> List[List[VDBSearchResult]]:
        """Search for similar vectors in the collection.

        Args:
            query_texts: List of query texts
            query_embeddings: List of query embeddings
            k: Number of results to return per query
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional search parameters

        Returns:
            List of lists of search results (one list per query)
        """
        raise NotImplementedError

    @abstractmethod
    def drop_collection(self, collection_name: str, **kwargs) -> bool:
        """Drop a collection.

        Args:
            collection_name: Name of the collection to drop
            **kwargs: Additional parameters

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    async def create_collection_async(
        self, collection_name: str, dim: int, **kwargs
    ) -> bool:
        """Create a new collection asynchronously.

        Args:
            collection_name: Name of the collection to create
            dim: Dimensionality of the vectors
            **kwargs: Additional collection parameters

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    async def store_async(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = None,
        **kwargs,
    ) -> List[str]:
        """Insert vectors into the collection asynchronously.

        Args:
            texts: List of texts to insert
            embeddings: List of embeddings to insert
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional arguments

        Returns:
            List of inserted IDs
        """
        raise NotImplementedError

    @abstractmethod
    async def search_async(
        self,
        query_texts: List[str],
        query_embeddings: List[List[float]],
        k: int = 5,
        collection_name: str = None,
        **kwargs,
    ) -> List[List[VDBSearchResult]]:
        """Search for similar vectors in the collection asynchronously.

        Args:
            query_texts: List of query texts
            query_embeddings: List of query embeddings
            k: Number of results to return per query
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional search parameters

        Returns:
            List of lists of search results (one list per query)
        """
        raise NotImplementedError

    @abstractmethod
    async def drop_collection_async(self, collection_name: str, **kwargs) -> bool:
        """Drop a collection asynchronously.

        Args:
            collection_name: Name of the collection to drop
            **kwargs: Additional parameters

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    def search_single(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 5,
        collection_name: str = None,
        **kwargs,
    ) -> List[VDBSearchResult]:
        """Search for similar vectors in the collection.

        Args:
            query_text: Query text
            query_embedding: Query embedding
            k: Number of results to return
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """

        return self.search(
            [query_text], [query_embedding], k, collection_name, **kwargs
        )[0]

    def similarity_search(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 5,
        **kwargs,
    ) -> List[VDBSearchResult]:
        """Deprecated interface for search_single"""

        return self.search_single(query_text, query_embedding, k, **kwargs)

    def batch_similarity_search(
        self,
        queries: list[str],
        embeddings: List[List[float]],
        k: int = 4,
        **kwargs: Any,
    ) -> List[List[VDBSearchResult]]:
        """Deprecated interface for search."""

        return self.search(queries, embeddings, k, **kwargs)

    async def search_single_async(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 5,
        collection_name: str = None,
        **kwargs,
    ) -> List[VDBSearchResult]:
        """Search for similar vectors in the collection asynchronously.

        Args:
            query_text: Query text
            query_embedding: Query embedding
            k: Number of results to return
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """

        results = await self.search_async(
            [query_text], [query_embedding], k, collection_name, **kwargs
        )
        return results[0]

    async def similarity_search_async(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 5,
        **kwargs,
    ) -> List[VDBSearchResult]:
        """Deprecated interface for search_single_async"""

        return await self.search_single_async(query_text, query_embedding, k, **kwargs)

    async def batch_similarity_search_async(
        self,
        queries: list[str],
        embeddings: List[List[float]],
        k: int = 4,
        **kwargs: Any,
    ) -> List[List[VDBSearchResult]]:
        """Deprecated interface for search_async."""

        return await self.search_async(queries, embeddings, k, **kwargs)

    def _serve_impl(self, request: VectorDBServiceRequest) -> VectorDBServiceResponse:
        """Handle incoming service requests."""
        operation_name = request.operation_name
        raw_data = request.raw_data

        try:
            if operation_name == "create_collection":
                if isinstance(request, VectorDBServiceCreateCollectionRequest):
                    result = self.create_collection(
                        collection_name=request.collection_name,
                        dim=request.dim,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = self.create_collection(**raw_data)
                return VectorDBServiceCreateCollectionResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    success=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            elif operation_name == "store" or operation_name == "insert":
                if isinstance(request, VectorDBServiceStoreRequest):
                    result = self.store(
                        texts=request.texts,
                        embeddings=request.text_embeddings,
                        metadatas=request.text_metadatas,
                        ids=request.ids,
                        collection_name=request.collection_name,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = self.store(**raw_data)
                return VectorDBServiceStoreResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    ids=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            elif operation_name == "search":
                if isinstance(request, VectorDBServiceSearchRequest):
                    result = self.search(
                        query_texts=request.texts,
                        query_embeddings=request.text_embeddings,
                        k=request.k,
                        collection_name=request.collection_name,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = self.search(**raw_data)
                return VectorDBServiceSearchResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    results=result,
                    raw_data={"result": result},  # for backward compatibility
                )

            elif operation_name == "search_single":
                if isinstance(request, VectorDBServiceSearchRequest):
                    result = self.search_single(
                        query_text=request.texts[0],
                        query_embedding=request.text_embeddings[0],
                        k=request.k,
                        collection_name=request.collection_name,
                        **request.kwargs,
                    )
                else:
                    result = self.search_single(**raw_data)
                return VectorDBServiceSearchResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    results=[result],
                    raw_data={"result": result},  # for backward compatibility
                )
            elif operation_name == "drop_collection":
                if isinstance(request, VectorDBServiceDropCollectionRequest):
                    result = self.drop_collection(
                        collection_name=request.collection_name, **request.kwargs
                    )
                else:
                    result = self.drop_collection(**raw_data)
                return VectorDBServiceDropCollectionResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    success=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            elif operation_name == "similarity_search":
                if isinstance(request, VectorDBServiceSearchRequest):
                    result = self.similarity_search(
                        query_text=request.texts[0],
                        query_embedding=request.text_embeddings[0],
                        k=request.k,
                        **request.kwargs,
                    )
                else:
                    result = self.similarity_search(**raw_data)
                return VectorDBServiceSearchResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    results=[result],
                    raw_data={"result": result},  # for backward compatibility
                )
            elif operation_name == "batch_similarity_search":
                if isinstance(request, VectorDBServiceSearchRequest):
                    result = self.batch_similarity_search(
                        queries=request.texts,
                        embeddings=request.text_embeddings,
                        k=request.k,
                        **request.kwargs,
                    )
                else:
                    result = self.batch_similarity_search(**raw_data)
                return VectorDBServiceSearchResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    results=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            else:
                raise ValueError(f"Unknown operation: {operation_name}")
        except Exception as e:
            return VectorDBServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=operation_name,
                success=False,
                error=str(e),
                raw_data={"error": str(e)},
            )

    async def _serve_async_impl(
        self, request: VectorDBServiceRequest
    ) -> VectorDBServiceResponse:
        """Handle incoming async service requests."""
        operation_name = request.operation_name
        raw_data = request.raw_data

        try:
            if (
                operation_name == "create_collection"
                or operation_name == "create_collection_async"
            ):
                if isinstance(request, VectorDBServiceCreateCollectionRequest):
                    result = await self.create_collection_async(
                        collection_name=request.collection_name,
                        dim=request.dim,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = await self.create_collection_async(**raw_data)
                return VectorDBServiceCreateCollectionResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    success=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            elif operation_name in ["store", "insert"] or operation_name in [
                "store_async",
                "insert_async",
            ]:
                if isinstance(request, VectorDBServiceStoreRequest):
                    result = await self.store_async(
                        texts=request.texts,
                        embeddings=request.text_embeddings,
                        metadatas=request.text_metadatas,
                        ids=request.ids,
                        collection_name=request.collection_name,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = await self.store_async(**raw_data)
                return VectorDBServiceStoreResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    ids=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            elif operation_name == "search" or operation_name == "search_async":
                if isinstance(request, VectorDBServiceSearchRequest):
                    result = await self.search_async(
                        query_texts=request.texts,
                        query_embeddings=request.text_embeddings,
                        k=request.k,
                        collection_name=request.collection_name,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = await self.search_async(**raw_data)
                return VectorDBServiceSearchResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    results=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            elif (
                operation_name == "search_single"
                or operation_name == "search_single_async"
            ):
                if isinstance(request, VectorDBServiceSearchRequest):
                    result = await self.search_single_async(
                        query_text=request.texts[0],
                        query_embedding=request.text_embeddings[0],
                        k=request.k,
                        collection_name=request.collection_name,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = await self.search_single_async(**raw_data)
                return VectorDBServiceSearchResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    results=[result],
                    raw_data={"result": result},  # for backward compatibility
                )
            elif (
                operation_name == "drop_collection"
                or operation_name == "drop_collection_async"
            ):
                if isinstance(request, VectorDBServiceDropCollectionRequest):
                    result = await self.drop_collection_async(
                        collection_name=request.collection_name, **request.kwargs
                    )
                else:
                    # for backward compatibility
                    result = await self.drop_collection_async(**raw_data)
                return VectorDBServiceDropCollectionResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    success=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            elif (
                operation_name == "similarity_search"
                or operation_name == "similarity_search_async"
            ):
                if isinstance(request, VectorDBServiceSearchRequest):
                    result = await self.similarity_search_async(
                        query_text=request.texts[0],
                        query_embedding=request.text_embeddings[0],
                        k=request.k,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = await self.similarity_search_async(**raw_data)
                return VectorDBServiceSearchResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    results=[result],
                    raw_data={"result": result},  # for backward compatibility
                )
            elif (
                operation_name == "batch_similarity_search"
                or operation_name == "batch_similarity_search_async"
            ):
                if isinstance(request, VectorDBServiceSearchRequest):
                    result = await self.batch_similarity_search_async(
                        queries=request.texts,
                        embeddings=request.text_embeddings,
                        k=request.k,
                        **request.kwargs,
                    )
                else:
                    # for backward compatibility
                    result = await self.batch_similarity_search_async(**raw_data)
                return VectorDBServiceSearchResponse(
                    request_id=request.request_id,
                    instance_name=self.instance_name,
                    operation_name=operation_name,
                    results=result,
                    raw_data={"result": result},  # for backward compatibility
                )
            else:
                # Fallback to sync implementation for unknown operations
                return self._serve_impl(request)
        except Exception as e:
            return VectorDBServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=operation_name,
                success=False,
                error=str(e),
                raw_data={"error": str(e)},
            )


class MilvusServiceInstance(VectorDBServiceInstance):
    """VectorDB service implementation for remote Milvus."""

    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: VectorDBServiceInstanceConfig,
    ):
        # Initialize base class first which will set up all the common attributes
        super().__init__(instance_name, group_name, config)

        # Initialize Milvus-specific attributes
        self.client = None
        self.async_client = None
        self.collection = None
        self.launch()

    def launch(self):
        """Initialize the Milvus client using MilvusClient."""
        self.client = None
        self.async_client = None

        try:
            # Parse connection string (format: "host:port")
            if hasattr(self.config, "location") and ":" in self.config.location:
                if "://" in self.config.location:
                    uri = self.config.location
                else:
                    uri = f"http://{self.config.location}"
            elif hasattr(self.config, "location"):
                uri = self.config.location
            else:
                uri = "http://localhost:19530"

            logger.info(f"Connecting to Milvus at {uri}")

            # Ensure collection_name is set
            if not hasattr(self, "collection_name") or not self.collection_name:
                self.collection_name = getattr(
                    self.config, "collection_name", "default_collection"
                )

            # Ensure metric_type is set
            if not hasattr(self, "metric_type") or not self.metric_type:
                self.metric_type = getattr(self.config, "metric_type", "L2")

            # Ensure dim is set
            if not hasattr(self, "dim") or not self.dim:
                self.dim = getattr(self.config, "dim", 1024)  # Default dimension

            # Set default index params if not provided
            if not hasattr(self, "index_params") or not self.index_params:
                self.index_params = {
                    "metric_type": self.metric_type,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                }

            # Ensure metric_type is set in index_params
            if "metric_type" not in self.index_params:
                self.index_params["metric_type"] = self.metric_type

            logger.info(f"Checking collection: {self.collection_name}")

            # Initialize synchronous client
            self.init_sync_client(uri=uri)
            # Initialize asynchronous client
            asyncio.run(self.init_async_client(uri=uri))

            # Check if collection exists, create if not
            self.create_collection()

            logger.info("Milvus client initialized successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize Milvus client: {str(e)}")
            raise

    def init_sync_client(self, uri):
        """Initialize the synchronous Milvus client."""
        if self.client is None:
            try:
                self.client = MilvusClient(uri=uri)
                logger.info("Synchronous Milvus client initialized successfully")
            except Exception as e:
                logger.exception(
                    f"Failed to initialize synchronous Milvus client: {str(e)}"
                )
                raise

    async def init_async_client(self, uri):
        """Initialize the asynchronous Milvus client."""
        if self.async_client is None:
            try:
                self.async_client = AsyncMilvusClient(uri=uri)
                logger.info("Asynchronous Milvus client initialized successfully")
            except Exception as e:
                logger.exception(
                    f"Failed to initialize asynchronous Milvus client: {str(e)}"
                )
                raise

    def shutdown(self):
        """Clean up resources."""
        if self.client:
            self.client.close()
        self.client = None
        if self.async_client:
            asyncio.run(self.async_client.close())
        self.async_client = None

    def has_collection(self, collection_name: str) -> bool:
        return self.client.has_collection(collection_name)

    def create_collection(
        self,
        collection_name: str = None,
        dim: int = None,
        metric_type: str = None,
        index_params: dict = None,
        **kwargs,
    ) -> bool:
        """Create a new collection in the vector database."""
        try:
            collection_name = collection_name or self.collection_name
            metric_type = metric_type or self.metric_type
            index_params = index_params or self.index_params
            dim = dim or self.dim
            if not self.client.has_collection(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    dimension=dim,
                    primary_field_name="id",
                    id_type="str",
                    auto_id=False,
                    consistency_level="Strong",
                    metric_type=metric_type,
                    max_length=len(str(uuid.uuid4())) + 1,
                    index_params=index_params,
                    **kwargs,
                )
                logger.info(f"Created collection {collection_name}")
            self.client.load_collection(collection_name)
            return True

        except Exception as e:
            logger.exception(f"Error creating collection {collection_name}: {str(e)}")
            raise ValueError("Collection not initialized") from e

    def drop_collection(self, collection_name: str) -> bool:
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                return True
        except Exception as e:
            logger.exception(f"Error dropping collection {collection_name}: {str(e)}")
            raise ValueError("Collection not dropped") from e

    def store(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = None,
        **kwargs,
    ) -> List[str]:
        """Insert vectors into the collection.

        Args:
            texts: List of texts to insert
            embeddings: List of embeddings to insert
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional arguments

        Returns:
            List of inserted IDs
        """
        try:
            collection_name = collection_name or self.collection_name
            self.create_collection(collection_name)
            if len(embeddings) == 0:
                return []
            if ids is None or ids == [] or ids[0] is None:
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
                self.client.insert(
                    collection_name=collection_name,
                    data=data[i : i + batch_size],
                )
            return ids
        except Exception as e:
            logger.exception(
                f"Error storing data in collection {collection_name}: {str(e)}"
            )
            raise ValueError("Data not stored") from e

    def search(
        self,
        query_texts: List[str],
        query_embeddings: List[List[float]],
        k: int = 5,
        collection_name: str = None,
        **kwargs,
    ) -> List[List[VDBSearchResult]]:
        try:
            collection_name = collection_name or self.collection_name
            if not self.client.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} does not exist")
            results = self.client.search(
                collection_name=collection_name,
                data=query_embeddings,
                limit=k,
                search_params={"metric_type": self.metric_type, "params": {}},
                output_fields=["text", "metadata"],
                **kwargs,
            )
            # sort by distance in ascending order
            for result in results:
                result.sort(key=lambda x: x["distance"], reverse=True)
            return [
                [
                    VDBSearchResult(
                        id=item["id"],
                        text=item["entity"]["text"],
                        metadata=item["entity"]["metadata"],
                        score=1.0 - item["distance"],
                    )
                    for item in result
                ]
                for result in results
            ]
        except Exception as e:
            logger.exception(f"Error searching collection {collection_name}: {str(e)}")
            raise ValueError("Search failed") from e

    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        try:
            # MilvusClient uses list_collections() to get all collections
            return self.client.list_collections()
        except Exception as e:
            logger.exception(f"Error listing collections: {str(e)}")
            return []

    async def list_collections_async(self) -> List[str]:
        """List all collections in the database asynchronously."""
        try:
            # AsyncMilvusClient do not has no list_collections method
            return self.client.list_collections()
        except Exception as e:
            logger.exception(f"Error listing collections asynchronously: {str(e)}")
            return []

    async def create_collection_async(
        self,
        collection_name: str = None,
        dim: int = None,
        metric_type: str = None,
        index_params: dict = None,
        **kwargs,
    ) -> bool:
        """Create a new collection in the vector database asynchronously."""
        try:
            collection_name = collection_name or self.collection_name
            metric_type = metric_type or self.metric_type
            index_params = index_params or self.index_params
            dim = dim or self.dim
            has_collection = self.client.has_collection(collection_name)
            if not has_collection:
                await self.async_client.create_collection(
                    collection_name=collection_name,
                    dimension=dim,
                    primary_field_name="id",
                    id_type="str",
                    auto_id=False,
                    consistency_level="Strong",
                    metric_type=metric_type,
                    max_length=len(str(uuid.uuid4())) + 1,
                    index_params=index_params,
                    **kwargs,
                )
            await self.async_client.load_collection(collection_name)
            return True

        except Exception as e:
            logger.exception(
                f"Error creating collection asynchronously {collection_name}: {str(e)}"
            )
            raise ValueError("Collection not initialized") from e

    async def drop_collection_async(self, collection_name: str) -> bool:
        """Drop a collection asynchronously."""
        try:
            has_collection = self.client.has_collection(collection_name)
            if has_collection:
                await self.async_client.drop_collection(collection_name)
            return True
        except Exception as e:
            logger.exception(
                f"Error dropping collection asynchronously {collection_name}: {str(e)}"
            )
            raise ValueError("Collection not dropped") from e

    async def store_async(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = None,
        **kwargs,
    ) -> List[str]:
        """Insert vectors into the collection asynchronously.

        Args:
            texts: List of texts to insert
            embeddings: List of embeddings to insert
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional arguments

        Returns:
            List of inserted IDs
        """
        try:
            collection_name = collection_name or self.collection_name
            await self.create_collection_async(collection_name)
            if len(embeddings) == 0:
                return []
            if ids is None or ids == [] or ids[0] is None:
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
            tasks = []
            for i in range(0, len(data), batch_size):
                tasks.append(
                    self.async_client.insert(
                        collection_name=collection_name,
                        data=data[i : i + batch_size],
                    )
                )
            await asyncio.gather(*tasks)
            return ids
        except Exception as e:
            logger.exception(
                f"Error storing data asynchronously in collection {collection_name}: {str(e)}"
            )
            raise ValueError("Data not stored") from e

    async def search_async(
        self,
        query_texts: List[str],
        query_embeddings: List[List[float]],
        k: int = 5,
        collection_name: str = None,
        **kwargs,
    ) -> List[List[VDBSearchResult]]:
        """Search for similar vectors in the collection asynchronously.

        Args:
            query_texts: List of query texts
            query_embeddings: List of query embeddings
            k: Number of results to return per query
            collection_name: Optional collection name (uses instance collection_name if not provided)
            **kwargs: Additional search parameters

        Returns:
            List of lists of search results (one list per query)
        """
        try:
            collection_name = collection_name or self.collection_name
            has_collection = self.client.has_collection(collection_name)
            if not has_collection:
                raise ValueError(f"Collection {collection_name} does not exist")
            results = await self.async_client.search(
                collection_name=collection_name,
                data=query_embeddings,
                limit=k,
                search_params={"metric_type": self.metric_type, "params": {}},
                output_fields=["text", "metadata"],
                **kwargs,
            )
            # sort by distance in ascending order
            for result in results:
                result.sort(key=lambda x: x["distance"], reverse=True)
            return [
                [
                    VDBSearchResult(
                        id=item["id"],
                        text=item["entity"]["text"],
                        metadata=item["entity"]["metadata"],
                        score=1.0 - item["distance"],
                    )
                    for item in result
                ]
                for result in results
            ]
        except Exception as e:
            logger.exception(
                f"Error searching collection asynchronously {collection_name}: {str(e)}"
            )
            raise ValueError("Search failed") from e


class MilvusLiteServiceInstance(MilvusServiceInstance):
    """VectorDB service implementation for Milvus Lite.

    This is a thin wrapper around MilvusServiceInstance.
    The difference between MilvusServiceInstance and MilvusLiteServiceInstance is that
    MilvusLiteServiceInstance uses a local Milvus server.
    """

    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: VectorDBServiceInstanceConfig,
    ):
        # Initialize base class first
        super().__init__(instance_name, group_name, config)
        self.server = None

    def launch(self):
        """Initialize Milvus Lite. The uri should be a filename."""
        assert self.config.location is not None
        assert not self.config.location.startswith("http://")
        assert ":" not in self.config.location
        super().launch()

    # The rest of the methods are the same as MilvusServiceInstance
    # since they both use the same pymilvus API


class SimpleVectorDB:
    """A simple in-memory vector database."""

    def __init__(
        self,
        name: str,
        dim: int,
        texts: Optional[List[str]] = None,
        vectors: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        self.name = name
        self.dim = dim
        self.texts = texts or []
        self.vectors = vectors or []
        self.metadata = metadata or []
        self.ids = ids or []

    def generate_id(self) -> str:
        """Generate a unique ID for a new entry."""
        return str(uuid.uuid4())

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """Add examples to the vector database.

        Args:
            texts: List of text examples
            embeddings: List of embeddings for the texts
            ids: Optional list of IDs for each example
            metadatas: Optional list of metadata for each example
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        if metadatas is None:
            metadatas = [{}] * len(texts)
        self.texts.extend(texts)
        self.vectors.extend(embeddings)
        self.metadata.extend(metadatas)
        if ids is None or ids == [] or ids[0] is None:
            ids = [self.generate_id() for _ in range(len(texts))]
        self.ids.extend(ids)
        return ids

    def search(
        self, query_embeddings: List[List[float]], k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Search for similar examples in the vector database.

        Args:
            query_embeddings: The query embeddings
            k: Number of results to return

        Returns:
            List of similar examples with metadata
        """
        if not self.vectors:
            logger.warning(f"Vector database {self.name} is empty")
            return []

        # Convert to numpy arrays for efficient computation
        query_embeddings = np.array(query_embeddings)
        vectors = np.array(self.vectors)

        logger.debug(f"{self.name} Vectors shape: {vectors.shape}")
        logger.debug(f"{self.name} Query embeddings shape: {query_embeddings.shape}")

        # Compute cosine similarity
        similarities = compute_cosine_similarity(query_embeddings, vectors)

        # Get the indices of the top k results
        top_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
        logger.debug(f"{self.name} Top indices: {top_indices}")

        # Return the top k results with metadata
        return [
            [
                {
                    "text": self.texts[idx],
                    "embedding": self.vectors[idx],
                    "metadata": self.metadata[idx],
                    "id": self.ids[idx],
                    "score": float(similarities[qid][idx]),
                }
                for idx in top_indices
            ]
            for qid, top_indices in enumerate(top_indices)
        ]

    async def add_async(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """Add examples to the vector database asynchronously."""
        return self.add(texts, embeddings, metadatas, ids)

    async def search_async(
        self,
        query_embeddings: List[List[float]],
        k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Search for similar examples in the vector database asynchronously."""
        return self.search(query_embeddings, k)


class SimpleVectorDBServiceInstance(VectorDBServiceInstance):
    """VectorDB service implementation for a simple in-memory vector database."""

    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: VectorDBServiceInstanceConfig,
    ):
        super().__init__(instance_name, group_name, config)
        self.collections: Dict[str, SimpleVectorDB] = {}

    @override
    def launch(self):
        try:
            # Ensure collection_name is set
            if not hasattr(self, "collection_name") or not self.collection_name:
                self.collection_name = getattr(
                    self.config, "collection_name", "default_collection"
                )

            # Ensure metric_type is set
            if not hasattr(self, "metric_type") or not self.metric_type:
                self.metric_type = getattr(self.config, "metric_type", "L2")

            # Ensure dim is set
            if not hasattr(self, "dim") or not self.dim:
                self.dim = getattr(self.config, "dim", 1024)  # Default dimension

            self.create_collection()

        except Exception as e:
            logger.exception(f"Failed to initialize SimpleVectorDB: {str(e)}")
            raise ValueError("SimpleVectorDB not initialized") from e

    def shutdown(self):
        self.collections.clear()

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str = None, dim: int = None) -> bool:
        try:
            collection_name = collection_name or self.collection_name
            dim = dim or self.dim
            if self.has_collection(collection_name):
                return True
            self.collections[collection_name] = SimpleVectorDB(
                name=collection_name, dim=dim
            )
            logger.info(f"Created collection {collection_name} with dimension {dim}")
            return True
        except Exception as e:
            logger.exception(f"Failed to create collection {collection_name}: {str(e)}")
            raise ValueError("Collection not created") from e

    def drop_collection(self, collection_name: str) -> bool:
        try:
            collection_name = collection_name or self.collection_name
            if self.has_collection(collection_name):
                self.collections.pop(collection_name, None)
                logger.info(f"Dropped collection {collection_name}")
            else:
                logger.warning(
                    f"Collection {collection_name} does not exist, nothing to drop"
                )
            return True
        except Exception as e:
            logger.exception(f"Failed to drop collection {collection_name}: {str(e)}")
            raise ValueError("Collection not dropped") from e

    def store(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = None,
        **kwargs,
    ) -> List[str]:
        try:
            collection_name = collection_name or self.collection_name
            self.create_collection(collection_name)
            if not isinstance(texts, list):
                texts = list(texts)
            logger.debug(
                f"Storing {len(texts)} texts in collection {collection_name} with {len(embeddings)} embeddings"
            )
            return self.collections[collection_name].add(
                texts, embeddings, metadatas, ids
            )
        except Exception as e:
            logger.exception(
                f"Failed to store texts in collection {collection_name}: {str(e)}"
            )
            raise ValueError("Texts not stored") from e

    def search(
        self,
        query_texts: List[str],
        query_embeddings: List[List[float]],
        k: int = 5,
        collection_name: str = None,
        **kwargs,
    ) -> List[List[VDBSearchResult]]:
        try:
            collection_name = collection_name or self.collection_name
            self.create_collection(collection_name)
            results = self.collections[collection_name].search(query_embeddings, k)
            logger.debug(f"Got {len(results)} results for collection {collection_name}")
            return [
                [
                    VDBSearchResult(
                        id=item["id"],
                        text=item["text"],
                        metadata=item["metadata"],
                        score=item["score"],
                    )
                    for item in result
                ]
                for result in results
            ]
        except Exception as e:
            logger.exception(
                f"Failed to search in collection {collection_name}: {str(e)}"
            )
            raise ValueError("Search failed") from e

    async def create_collection_async(
        self, collection_name: str, dim: int, **kwargs
    ) -> bool:
        try:
            return self.create_collection(collection_name, dim, **kwargs)
        except Exception as e:
            logger.exception(f"Failed to create collection {collection_name}: {str(e)}")
            raise ValueError("Collection not created") from e

    async def drop_collection_async(self, collection_name: str, **kwargs) -> bool:
        try:
            return self.drop_collection(collection_name, **kwargs)
        except Exception as e:
            logger.exception(f"Failed to drop collection {collection_name}: {str(e)}")
            raise ValueError("Collection not dropped") from e

    async def store_async(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = None,
        **kwargs,
    ) -> List[str]:
        try:
            collection_name = collection_name or self.collection_name
            self.create_collection(collection_name)
            if not isinstance(texts, list):
                texts = list(texts)
            logger.debug(
                f"Storing {len(texts)} texts in collection {collection_name} with {len(embeddings)} embeddings"
            )
            return await self.collections[collection_name].add_async(
                texts, embeddings, metadatas, ids
            )
        except Exception as e:
            logger.exception(
                f"Failed to store texts in collection {collection_name}: {str(e)}"
            )
            raise ValueError("Texts not stored") from e

    async def search_async(
        self,
        query_texts: List[str],
        query_embeddings: List[List[float]],
        k: int = 5,
        collection_name: str = None,
        **kwargs,
    ) -> List[List[VDBSearchResult]]:
        try:
            collection_name = collection_name or self.collection_name
            self.create_collection(collection_name)
            results = await self.collections[collection_name].search_async(
                query_embeddings, k
            )
            logger.debug(f"Got {len(results)} results for collection {collection_name}")
            return [
                [
                    VDBSearchResult(
                        id=item["id"],
                        text=item["text"],
                        metadata=item["metadata"],
                        score=item["score"],
                    )
                    for item in result
                ]
                for result in results
            ]
        except Exception as e:
            logger.exception(
                f"Failed to search in collection {collection_name}: {str(e)}"
            )
            raise ValueError("Search failed") from e

    def list_collections(self):
        return self.collections.keys()


class VectorDBService(BaseService):
    """VectorDB service for managing vector database instances with collection-aware routing.

    This service ensures that all operations on the same collection are consistently
    routed to the same VDB instance, maintaining data consistency across insert,
    search, and other collection operations.
    """

    def __init__(
        self,
        scheduling_policy: str = "round_robin",
        collection_assignment_strategy: str = "consistent_hashing",
    ):
        """Initialize VectorDB service with collection-aware routing.

        Args:
            scheduling_policy: Default scheduling policy for new collections
            collection_assignment_strategy: Strategy for assigning collections to instances
                - "consistent_hashing": Use consistent hashing for deterministic assignment
                - "explicit": Allow manual collection-to-instance mapping
                - "load_balanced": Consider instance load when assigning new collections
        """
        # Enable stateful routing with collection-aware strategy
        super().__init__(
            service_name="vdb",
            scheduling_policy=scheduling_policy,
            stateful_routing=True,
            stateful_assignment_strategy=collection_assignment_strategy,
        )

    def extract_hash_key(self, request: VectorDBServiceRequest) -> Optional[str]:
        """Extract collection name from VDB request for stateful routing.

        Args:
            request: The VDB service request

        Returns:
            Collection name or None if not found
        """
        # First check if hash_key is explicitly set
        if request.hash_key:
            return request.hash_key

        # Extract collection_name from raw_data for VDB requests
        if isinstance(request, VectorDBServiceRequest):
            raw_data = request.raw_data or {}
            return raw_data.get("collection_name")

        return None

    def clear_collection(self, collection_name: str):
        """Clear a specific collection from all instances.

        Args:
            collection_name: Name of the collection to clear
        """
        for instance in self.instances.values():
            instance = cast(VectorDBServiceInstance, instance)
            if instance.has_collection(collection_name):
                instance.drop_collection(collection_name)
                logger.info(
                    "Dropped collection %s from instance %s",
                    collection_name,
                    instance.instance_name,
                )

        # Clear the hash key mapping for this collection
        self.clear_hash_key_mapping(collection_name)
        logger.info("Cleared collection mapping for %s", collection_name)

    def clear_all(self):
        """Clear all collections in all instances and reset mappings."""
        for instance in self.instances.values():
            instance = cast(VectorDBServiceInstance, instance)
            collections = list(instance.list_collections())
            for collection in collections:
                instance.drop_collection(collection)
                logger.info("Dropped collection %s", collection)

        # Clear all hash key mappings (collections)
        self.hash_key_to_instance.clear()
        self.instance_hash_key_count = {name: 0 for name in self.instance_names}

    # Backward compatibility methods for collection-specific API
    def get_collection_mapping(self) -> Dict[str, str]:
        """Get the current collection-to-instance mapping.

        Returns:
            Dictionary mapping collection names to instance names
        """
        return self.get_hash_key_mapping()

    def set_collection_mapping(self, collection_name: str, instance_name: str) -> None:
        """Explicitly set a collection-to-instance mapping.

        Args:
            collection_name: Name of the collection
            instance_name: Name of the instance to assign the collection to

        Raises:
            ValueError: If the instance doesn't exist
        """
        self.set_hash_key_mapping(collection_name, instance_name)

    def clear_collection_mapping(self, collection_name: str) -> None:
        """Clear the mapping for a specific collection.

        Args:
            collection_name: Name of the collection to clear mapping for
        """
        self.clear_hash_key_mapping(collection_name)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)

    vdb_service = SimpleVectorDBServiceInstance(
        instance_name="simple_vdb",
        group_name="default",
        config=VectorDBServiceInstanceConfig(
            collection_name="example_collection",
            dim=128,
            metric_type="L2",
            index_params={"index_type": "IVF_FLAT", "nlist": 100},
        ),
    )
    vdb_service.launch()

    async def async_test(collection_name: str = "example_collection", n: int = 1):
        collection_name = "deep_research_langchain_drs_ab5f3bae5228ade27687b8cfa2d09df4_3fddf624-023b-4fa6-bd8a-a68f6b1388e1"
        texts = [f"example text {1*n}", f"example text {2*n}"]
        embeddings = [[0.1 * n] * 128, [0.2 * n] * 128]
        ids = await vdb_service.store_async(
            texts, embeddings, collection_name=collection_name
        )
        print(f"Stored IDs: {ids}")
        search_results = await vdb_service.search_async(
            query_texts=[f"example text {1*n}"],
            query_embeddings=[[0.1 * n] * 128],
            k=1,
            collection_name=collection_name,
        )
        print(f"Search results: {search_results}")

        # drop collection
        await vdb_service.drop_collection_async(collection_name)
        print(f"Collection {collection_name} dropped.")

    async def run_async_tests():
        """Run asynchronous tests for the VectorDB service."""
        tasks = [
            async_test(collection_name="example_collection", n=1),
            async_test(collection_name="example_collection", n=10),
            async_test(collection_name="example_collection", n=100),
        ]
        return await asyncio.gather(*tasks)

    asyncio.run(run_async_tests())
    vdb_service.shutdown()
    logger.info("VectorDB service instance shutdown complete.")
