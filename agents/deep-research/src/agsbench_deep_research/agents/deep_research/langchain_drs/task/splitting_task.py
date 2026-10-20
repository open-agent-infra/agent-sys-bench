"""
Splitting (Chunking) task module for deep research operations.

This module provides task implementations for splitting (chunking) operations.
"""

from typing import List, Optional
import logging
from pydantic import Field

from agsbench_deep_research.core.schema.io import TaskRequest, TaskResponse
from agsbench_deep_research.core.registry_setup import get_service_manager
from ..shared import agsbench_deep_research_fsm, RequestContext

logger = logging.getLogger(__name__)


class SplittingTaskRequest(TaskRequest):
    """Request model for splitting task.

    Attributes:
        texts: List of texts to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        model: Optional model to use for splitting
    """

    task_name: str = "splitting_task"

    texts: List[str] = Field(..., description="List of texts to split")
    metadatas: Optional[List[dict]] = Field(
        default=None, description="Optional metadata for each text"
    )
    model: Optional[str] = Field(
        default="langchain-recursive-character",
        description="Model to use for splitting",
    )
    chunk_size: int = Field(..., description="Size of each chunk")
    chunk_overlap: int = Field(default=0, description="Overlap between chunks")

    def get_statistics(self) -> dict:
        """Get statistics for the splitting task."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "texts_count": len(self.texts),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "model": self.model,
                "text_lengths": [len(text) for text in self.texts],
                "metadata_keys": (
                    list(self.metadatas[0].keys())
                    if (self.metadatas and len(self.metadatas) > 0)
                    else []
                ),
            }
        )
        return stats


class SplittingTaskResponse(TaskResponse):
    """Response model for splitting task.

    Attributes:
        chunks: List of text chunks
        success: Whether the splitting was successful
        error: Error message if splitting failed
    """

    task_name: str = "splitting_task"

    chunks: List[str] = Field(default_factory=list, description="List of text chunks")
    metadatas: Optional[List[dict]] = Field(
        default=None, description="Optional metadata for each chunk"
    )
    # Note: metadatas should match the chunks in order, if provided
    # This is useful for keeping track of which metadata corresponds to which chunk
    success: bool = Field(
        default=True, description="Whether the splitting was successful"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if splitting failed"
    )

    def get_statistics(self) -> dict:
        """Get statistics for the splitting task response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "chunks_count": len(self.chunks),
                "metadatas_count": len(self.metadatas) if self.metadatas else 0,
                "metadata_keys": (
                    list(self.metadatas[0].keys())
                    if (self.metadatas and len(self.metadatas) > 0)
                    else []
                ),
                "chunk_lengths": [len(chunk) for chunk in self.chunks],
                "success": self.success,
                "error": self.error,
            }
        )
        return stats


@agsbench_deep_research_fsm.fast_service
def split_texts(
    request: SplittingTaskRequest, context: RequestContext = None
) -> SplittingTaskResponse:
    """Split texts into chunks using the splitting service.
    TODO: splitting service is not implemented yet, we use langchain's splitting tools for now.

    Args:
        request: SplittingTaskRequest containing texts and parameters
        context: Optional RequestContext for additional context

    Returns:
        SplittingTaskResponse with chunks and success status
    """

    logger.debug("Splitting texts with request: %s", request.get_statistics())
    try:
        from langchain_core.documents import Document as LC_Document

        docs = [
            LC_Document(page_content=text, metadata=metadata)
            for text, metadata in zip(request.texts, request.metadatas)
        ]

        if request.model == "langchain-recursive-character":
            # Use langchain's recursive character text splitter
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                add_start_index=True,
            )
        else:
            # not implemented yet, raise an error or use a default splitter
            raise NotImplementedError(
                f"Splitting model '{request.model}' is not implemented yet."
            )
        logger.debug("Splitting %d docs with %s splitter.", len(docs), request.model)
        chunked_docs = splitter.split_documents(docs)
        chunks = [doc.page_content for doc in chunked_docs]
        metadatas = [doc.metadata for doc in chunked_docs]
        logger.debug("Splitting completed successfully with %d chunks", len(chunks))
        return SplittingTaskResponse(
            request_id=request.request_id,
            chunks=chunks,
            metadatas=metadatas,
            success=True,
            error=None,
        )
    except Exception as e:
        logger.exception("Error splitting texts: %s", e)
        return SplittingTaskResponse(
            request_id=request.request_id, success=False, error=str(e)
        )


@agsbench_deep_research_fsm.async_fast_service
async def split_texts_async(
    request: SplittingTaskRequest, context: RequestContext = None
) -> SplittingTaskResponse:
    """Asynchronously split texts into chunks using the splitting service."""
    logger.debug(
        "Asynchronously splitting texts with request: %s", request.get_statistics()
    )
    try:
        from langchain_core.documents import Document as LC_Document

        docs = [
            LC_Document(page_content=text, metadata=metadata)
            for text, metadata in zip(request.texts, request.metadatas)
        ]

        if request.model == "langchain-recursive-character":
            # Use langchain's recursive character text splitter
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                add_start_index=True,
            )
        else:
            # not implemented yet, raise an error or use a default splitter
            raise NotImplementedError(
                f"Splitting model '{request.model}' is not implemented yet."
            )
        logger.debug("Splitting %d docs with %s splitter.", len(docs), request.model)
        chunked_docs = await splitter.atransform_documents(docs)
        chunks = [doc.page_content for doc in chunked_docs]
        metadatas = [doc.metadata for doc in chunked_docs]
        logger.debug(
            "Asynchronous splitting completed successfully with %d chunks", len(chunks)
        )
        return SplittingTaskResponse(
            request_id=request.request_id,
            chunks=chunks,
            metadatas=metadatas,
            success=True,
            error=None,
        )
    except Exception as e:
        logger.exception("Error asynchronously splitting texts: %s", e)
        return SplittingTaskResponse(
            request_id=request.request_id, success=False, error=str(e)
        )
