"""
Embedding task module for deep research operations.

This module provides task implementations for embedding operations.
"""

from typing import List, Optional
import logging
from pydantic import Field

from agsbench_deep_research.core.schema.io import TaskRequest, TaskResponse
from agsbench_deep_research.core.registry_setup import get_service_manager
from agsbench_deep_research.core.service.embedding import (
    EmbeddingServiceRequest,
    EmbeddingServiceResponse,
    EmbeddingService,
)
from ..shared import agsbench_deep_research_fsm, RequestContext

logger = logging.getLogger(__name__)


class EmbeddingTaskRequest(TaskRequest):
    """Request model for embedding task.

    Attributes:
        texts: List of texts to embed
        model: Embedding model to use
    """

    task_name: str = "embedding_task"

    texts: List[str] = Field(..., description="List of texts to embed")
    model: Optional[str] = Field(default=None, description="Embedding model to use")


class EmbeddingTaskResponse(TaskResponse):
    """Response model for embedding task.

    Attributes:
        embeddings: List of embeddings
        success: Whether the embedding was successful
        error: Error message if embedding failed
    """

    task_name: str = "embedding_task"

    embeddings: List[List[float]] = Field(
        default_factory=list, description="List of embeddings"
    )
    success: bool = Field(
        default=True, description="Whether the embedding was successful"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if embedding failed"
    )


@agsbench_deep_research_fsm.fast_service
def inner_embed_texts(
    request: EmbeddingTaskRequest, context: RequestContext = None
) -> EmbeddingTaskResponse:
    """Embed texts using the embedding service.

    Args:
        request: The embedding request

    Returns:
        Embedding task response
    """
    logger.info(f"Embedding {len(request.texts)} texts")

    # Get service manager
    service_manager = get_service_manager()

    # Get embedding service
    embedding_service: EmbeddingService = service_manager.get_service("embedding")

    # Create embedding request
    embedding_request = EmbeddingServiceRequest(
        request_id=request.request_id,
        service_name="embedding",
        operation_name="embed",
        model=request.model,
        texts=request.texts,
    )

    try:
        # Get embeddings
        embedding_response: EmbeddingServiceResponse = embedding_service.serve(
            embedding_request, request.model
        )

        if (not embedding_response.success) or (not embedding_response.embeddings):
            logger.error("Failed to generate embeddings")
            return EmbeddingTaskResponse(
                request_id=request.request_id,
                embeddings=[],
                success=False,
                error="Failed to generate embeddings",
            )
        embeddings = embedding_response.embeddings

        logger.info(f"Successfully embedded {len(embeddings)} texts")
        return EmbeddingTaskResponse(
            request_id=request.request_id,
            embeddings=embeddings,
            success=True,
        )
    except Exception as e:
        logger.exception(f"Error embedding texts: {str(e)}")
        return EmbeddingTaskResponse(
            request_id=request.request_id,
            embeddings=[],
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.async_fast_service
async def inner_embed_texts_async(
    request: EmbeddingTaskRequest, context: RequestContext = None
) -> EmbeddingTaskResponse:
    """Embed texts using the embedding service asynchronously.

    Args:
        request: The embedding request

    Returns:
        Embedding task response
    """
    logger.info(f"Embedding {len(request.texts)} texts asynchronously")

    # Get service manager
    service_manager = get_service_manager()

    # Get embedding service
    embedding_service: EmbeddingService = service_manager.get_service("embedding")

    # Create embedding request
    embedding_request = EmbeddingServiceRequest(
        request_id=request.request_id,
        service_name="embedding",
        operation_name="embed",
        model=request.model,
        texts=request.texts,
    )

    try:
        # Get embeddings
        embedding_response: EmbeddingServiceResponse = (
            await embedding_service.serve_async(embedding_request, request.model)
        )

        if (not embedding_response.success) or (not embedding_response.embeddings):
            logger.error("Failed to generate embeddings")
            return EmbeddingTaskResponse(
                request_id=request.request_id,
                embeddings=[],
                success=False,
                error="Failed to generate embeddings",
            )
        embeddings = embedding_response.embeddings

        logger.info(f"Successfully embedded {len(embeddings)} texts asynchronously")
        return EmbeddingTaskResponse(
            request_id=request.request_id,
            embeddings=embeddings,
            success=True,
        )
    except Exception as e:
        logger.exception(f"Error embedding texts asynchronously: {str(e)}")
        return EmbeddingTaskResponse(
            request_id=request.request_id,
            embeddings=[],
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.fast_service
def embed_texts(
    request: EmbeddingTaskRequest, context: RequestContext = None
) -> EmbeddingTaskResponse:
    return inner_embed_texts(request, context=context)


@agsbench_deep_research_fsm.async_fast_service
async def embed_texts_async(
    request: EmbeddingTaskRequest, context: RequestContext = None
) -> EmbeddingTaskResponse:
    return await inner_embed_texts_async(request, context=context)


@agsbench_deep_research_fsm.fast_service
def embed_docs(
    request: EmbeddingTaskRequest, context: RequestContext = None
) -> EmbeddingTaskResponse:
    return inner_embed_texts(request, context=context)


@agsbench_deep_research_fsm.async_fast_service
async def embed_docs_async(
    request: EmbeddingTaskRequest, context: RequestContext = None
) -> EmbeddingTaskResponse:
    return await inner_embed_texts_async(request, context=context)
