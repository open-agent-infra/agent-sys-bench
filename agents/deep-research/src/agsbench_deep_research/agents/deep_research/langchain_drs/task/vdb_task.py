"""
Vector database task module for deep research operations.

This module provides task implementations for vector database operations.
"""

from typing import List, Dict, Any, Optional, cast
import logging
from pydantic import Field

from agsbench_deep_research.core.schema.io import TaskRequest, TaskResponse
from agsbench_deep_research.core.registry_setup import get_service_manager
from agsbench_deep_research.core.service.vdb import (
    VectorDBServiceRequest,
    VectorDBServiceStoreRequest,
    VectorDBServiceSearchRequest,
)
from agsbench_deep_research.core.service.vdb import (
    VectorDBServiceResponse,
    VectorDBServiceStoreResponse,
    VectorDBServiceSearchResponse,
    VectorDBServiceDropCollectionRequest,
    VectorDBServiceDropCollectionResponse,
)
from agsbench_deep_research.core.service.vdb import VDBSearchResult
from ..shared import agsbench_deep_research_fsm, RequestContext

logger = logging.getLogger(__name__)


class VDBSearchRequest(TaskRequest):
    """Request model for vector database search task.

    Attributes:
        query_texts: List of query texts
        query_embeddings: List of query embeddings
        collection_name: Name of the vector database collection
        k: Number of results to return
    """

    task_name: str = "vdb_search_task"

    query_texts: List[str] = Field(
        default_factory=list, description="List of query texts"
    )
    query_embeddings: List[List[float]] = Field(
        ..., description="List of query embeddings"
    )
    collection_name: str = Field(
        default="simple", description="Name of the vector database collection"
    )
    k: int = Field(default=5, description="Number of results to return")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments for search"
    )


class VDBSearchResponse(TaskResponse):
    """Response model for vector database search task.

    Attributes:
        results: List of search results
        success: Whether the search was successful
        error: Error message if search failed
    """

    task_name: str = "vdb_search_task"

    results: List[List[VDBSearchResult]] = Field(
        default_factory=list, description="List of search results"
    )
    success: bool = Field(default=True, description="Whether the search was successful")
    error: Optional[str] = Field(
        default=None, description="Error message if search failed"
    )


class VDBStoreRequest(TaskRequest):
    """Request model for vector database store task.

    Attributes:
        texts: List of texts to store
        embeddings: List of embeddings to store
        metadatas: List of metadata for each text
        collection_name: Name of the vector database collection
    """

    task_name: str = "vdb_store_task"

    texts: List[str] = Field(..., description="List of texts to store")
    embeddings: List[List[float]] = Field(
        ..., description="List of embeddings to store"
    )
    metadatas: List[Dict[str, Any]] = Field(
        ..., description="List of metadata for each text"
    )
    ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of IDs for each text, if not provided, IDs will be generated",
    )
    collection_name: str = Field(
        default="simple", description="Name of the vector database collection"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments for store"
    )


class VDBClearRequest(TaskRequest):
    """Request model for vector database clear task.

    Attributes:
        collection_name: Name of the vector database collection to clear
    """

    task_name: str = "vdb_clear_task"

    collection_name: str = Field(
        default="simple", description="Name of the vector database collection to clear"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments for clear"
    )


class VDBClearResponse(TaskResponse):
    """Response model for vector database clear task.

    Attributes:
        success: Whether the clear operation was successful
        error: Error message if clear operation failed
    """

    task_name: str = "vdb_clear_task"

    success: bool = Field(
        default=True, description="Whether the clear operation was successful"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if clear operation failed"
    )


class VDBStoreResponse(TaskResponse):
    """Response model for vector database store task.

    Attributes:
        success: Whether the store operation was successful
        error: Error message if store operation failed
    """

    task_name: str = "vdb_store_task"

    success: bool = Field(
        default=True, description="Whether the store operation was successful"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if store operation failed"
    )


@agsbench_deep_research_fsm.fast_service
def search_vdb(
    request: VDBSearchRequest, context: RequestContext = None
) -> VDBSearchResponse:
    """Search the vector database for similar examples.

    Args:
        request: The vector database search request

    Returns:
        Vector database search response
    """
    logger.info(
        f"Searching vector database for {len(request.query_embeddings)} queries"
    )

    # Get service manager
    service_manager = get_service_manager()

    # Get vector database service
    vdb_service = service_manager.get_service("vdb")

    # Create vector database search request
    vdb_request = VectorDBServiceSearchRequest(
        service_name="vdb",
        operation_name="search",
        texts=request.query_texts,
        text_embeddings=request.query_embeddings,
        k=request.k,
        collection_name=request.collection_name,
        kwargs=request.kwargs,
    )

    try:
        # Search vector database
        vdb_response: VectorDBServiceSearchResponse = vdb_service.serve(vdb_request)
        search_results = vdb_response.results

        if not search_results or not search_results[0]:
            logger.warning("No search results found")
            return VDBSearchResponse(
                request_id=request.request_id,
                results=[[]],
                success=True,
            )

        logger.info(
            f"Found {[len(result) for result in search_results]} search results"
        )
        return VDBSearchResponse(
            request_id=request.request_id,
            results=search_results,
            success=True,
        )
    except Exception as e:
        logger.exception(f"Error searching vector database: {str(e)}")
        return VDBSearchResponse(
            request_id=request.request_id,
            results=[[]],
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.async_fast_service
async def search_vdb_async(
    request: VDBSearchRequest, context: RequestContext = None
) -> VDBSearchResponse:
    """Search the vector database for similar examples asynchronously.

    Args:
        request: The vector database search request

    Returns:
        Vector database search response
    """
    logger.info(
        f"Searching vector database for {len(request.query_embeddings)} queries asynchronously"
    )

    # Get service manager
    service_manager = get_service_manager()

    # Get vector database service
    vdb_service = service_manager.get_service("vdb")

    # Create vector database search request
    vdb_request = VectorDBServiceSearchRequest(
        service_name="vdb",
        operation_name="search_async",
        texts=request.query_texts,
        text_embeddings=request.query_embeddings,
        k=request.k,
        collection_name=request.collection_name,
        kwargs=request.kwargs,
    )

    try:
        # Search vector database
        vdb_response: VectorDBServiceSearchResponse = await vdb_service.serve_async(
            vdb_request
        )
        search_results = vdb_response.results

        if not search_results or not search_results[0]:
            logger.warning("No search results found")
            return VDBSearchResponse(
                request_id=request.request_id,
                results=[[]],
                success=True,
            )

        logger.info(
            f"Found {[len(result) for result in search_results]} search results asynchronously"
        )
        return VDBSearchResponse(
            request_id=request.request_id,
            results=search_results,
            success=True,
        )
    except Exception as e:
        logger.exception(f"Error searching vector database asynchronously: {str(e)}")
        return VDBSearchResponse(
            request_id=request.request_id,
            results=[[]],
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.fast_service
def store_vdb(
    request: VDBStoreRequest, context: RequestContext = None
) -> VDBStoreResponse:
    """Store texts and embeddings in the vector database.

    Args:
        request: The vector database store request

    Returns:
        Vector database store response
    """
    logger.info(f"Storing {len(request.texts)} texts in vector database")

    # Get service manager
    service_manager = get_service_manager()

    # Get vector database service
    vdb_service = service_manager.get_service("vdb")

    # Create vector database store request
    vdb_request = VectorDBServiceStoreRequest(
        service_name="vdb",
        operation_name="store",
        texts=request.texts,
        text_embeddings=request.embeddings,
        text_metadatas=request.metadatas,
        ids=request.ids,
        collection_name=request.collection_name,
        kwargs=request.kwargs,
    )

    try:
        # Store in vector database
        vdb_response: VectorDBServiceStoreResponse = vdb_service.serve(vdb_request)
        logger.info(
            f"Successfully stored {len(vdb_response.ids)} texts in vector database"
        )
        return VDBStoreResponse(
            request_id=request.request_id,
            success=True,
        )
    except Exception as e:
        logger.exception(f"Error storing in vector database: {str(e)}")
        return VDBStoreResponse(
            request_id=request.request_id,
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.async_fast_service
async def store_vdb_async(
    request: VDBStoreRequest, context: RequestContext = None
) -> VDBStoreResponse:
    """Store texts and embeddings in the vector database asynchronously.

    Args:
        request: The vector database store request

    Returns:
        Vector database store response
    """
    logger.info(f"Storing {len(request.texts)} texts in vector database asynchronously")

    # Get service manager
    service_manager = get_service_manager()

    # Get vector database service
    vdb_service = service_manager.get_service("vdb")

    # Create vector database store request
    vdb_request = VectorDBServiceStoreRequest(
        service_name="vdb",
        operation_name="store_async",
        texts=request.texts,
        text_embeddings=request.embeddings,
        text_metadatas=request.metadatas,
        ids=request.ids,
        collection_name=request.collection_name,
        kwargs=request.kwargs,
    )

    try:
        # Store in vector database
        vdb_response: VectorDBServiceStoreResponse = await vdb_service.serve_async(
            vdb_request
        )

        logger.info(
            f"Successfully stored {len(vdb_response.ids)} texts in vector database asynchronously"
        )
        return VDBStoreResponse(
            request_id=request.request_id,
            success=True,
        )
    except Exception as e:
        logger.exception(f"Error storing in vector database asynchronously: {str(e)}")
        return VDBStoreResponse(
            request_id=request.request_id,
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.fast_service
def clear_vdb(
    request: VDBClearRequest, context: RequestContext = None
) -> VDBClearResponse:
    """Clear the vector database collection.

    Args:
        request: The vector database clear request

    Returns:
        Vector database clear response
    """
    logger.info(f"Clearing vector database collection: {request.collection_name}")

    # Get service manager
    service_manager = get_service_manager()

    # Get vector database service
    vdb_service = service_manager.get_service("vdb")

    # Create vector database clear request
    vdb_request = VectorDBServiceDropCollectionRequest(
        service_name="vdb",
        operation_name="drop_collection",
        collection_name=request.collection_name,
        kwargs=request.kwargs,
    )

    try:
        # Clear vector database collection
        vdb_response: VectorDBServiceDropCollectionResponse = vdb_service.serve(
            vdb_request
        )

        if not vdb_response.success:
            logger.error(
                f"Failed to clear vector database collection: {vdb_response.error}"
            )
            return VDBClearResponse(
                request_id=request.request_id,
                success=False,
                error=vdb_response.error,
            )
        logger.info(
            f"Successfully cleared vector database collection: {request.collection_name}"
        )
        return VDBClearResponse(
            request_id=request.request_id,
            success=True,
        )
    except Exception as e:
        logger.exception(f"Error clearing vector database collection: {str(e)}")
        return VDBClearResponse(
            request_id=request.request_id,
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.async_fast_service
async def clear_vdb_async(
    request: VDBClearRequest, context: RequestContext = None
) -> VDBClearResponse:
    """Clear the vector database collection asynchronously.

    Args:
        request: The vector database clear request

    Returns:
        Vector database clear response
    """
    logger.info(
        f"Clearing vector database collection: {request.collection_name} asynchronously"
    )

    # Get service manager
    service_manager = get_service_manager()

    # Get vector database service
    vdb_service = service_manager.get_service("vdb")

    # Create vector database clear request
    vdb_request = VectorDBServiceDropCollectionRequest(
        service_name="vdb",
        operation_name="drop_collection_async",
        collection_name=request.collection_name,
        kwargs=request.kwargs,
    )

    try:
        # Clear vector database collection asynchronously
        vdb_response: VectorDBServiceDropCollectionResponse = (
            await vdb_service.serve_async(vdb_request)
        )

        if not vdb_response.success:
            logger.error(
                f"Failed to clear vector database collection asynchronously: {vdb_response.error}"
            )
            return VDBClearResponse(
                request_id=request.request_id,
                success=False,
                error=vdb_response.error,
            )
        logger.info(
            f"Successfully cleared vector database collection: {request.collection_name} asynchronously"
        )
        return VDBClearResponse(
            request_id=request.request_id,
            success=True,
        )
    except Exception as e:
        logger.exception(
            f"Error clearing vector database collection asynchronously: {str(e)}"
        )
        return VDBClearResponse(
            request_id=request.request_id,
            success=False,
            error=str(e),
        )
