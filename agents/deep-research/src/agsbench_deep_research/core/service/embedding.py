"""
Embedding service module for text embedding operations.

This module provides service implementations for generating embeddings
from text using various embedding models, including remote API-based models
and local models running in containers.
"""

from .base import (
    BaseServiceInstanceConfig,
    BaseServiceInstance,
    BaseService,
    BaseServiceRequest,
    BaseServiceResponse,
    BaseModel,
)
import os
from abc import abstractmethod
from typing import Iterable, List, Union, cast, Optional, Dict, Any, Type
from pydantic import BaseModel, Field, model_validator
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


class EmbeddingServiceRequest(BaseServiceRequest):
    """Request model for Embedding service operations."""

    model: str = Field(..., description="Model name or identifier to use for embedding")
    texts: List[str] = Field(..., description="List of texts to embed")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific parameters"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "model": self.model,
                "texts_count": len(self.texts),
                "text_lengths": [len(text) for text in self.texts],
                "kwargs": self.kwargs,
            }
        )
        return stats


class EmbeddingServiceResponse(BaseServiceResponse):
    """Response model for Embedding service operations."""

    embeddings: List[List[float]] = Field(
        ..., description="List of embeddings, where each embedding is a list of floats"
    )

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "embeddings_count": len(self.embeddings),
                "embedding_dimensions": (
                    len(self.embeddings[0]) if self.embeddings else 0
                ),
            }
        )
        return stats


class EmbeddingServiceInstanceConfig(BaseServiceInstanceConfig):
    """Configuration for an Embedding service instance.

    Attributes:
        model: Model name or identifier
    """

    model: str = Field(..., description="Model name or identifier")


class EmbeddingServiceInstance(BaseServiceInstance):
    """Base class for Embedding service instances.

    This abstract class defines the interface that all Embedding service
    implementations must provide, including methods for generating embeddings
    in both synchronous and asynchronous forms.
    """

    @abstractmethod
    def embed(self, request: EmbeddingServiceRequest) -> EmbeddingServiceResponse:
        """Generate embeddings for a list of texts.
        Args:
            request: The embedding request containing model, texts, and additional parameters
        Returns:
            EmbeddingServiceResponse containing the generated embeddings
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_async(
        self, request: EmbeddingServiceRequest
    ) -> EmbeddingServiceResponse:
        """Generate embeddings for a list of texts asynchronously.

        Args:
            request: The embedding request containing model, texts, and additional parameters

        Returns:
            EmbeddingServiceResponse containing the generated embeddings
        """
        raise NotImplementedError

    def _serve_impl(self, request: EmbeddingServiceRequest) -> EmbeddingServiceResponse:
        """Serve a request."""
        logger.debug(f"Serving embedding request: {request.get_statistics()}")
        operation_name = request.operation_name
        try:
            if operation_name == "embed":
                response = self.embed(request)
            else:
                raise ValueError(f"Unknown operation name: {operation_name}")

            response.request_id = request.request_id
        except Exception as e:
            # Handle exceptions and return an error response
            logger.exception(
                f"Error serving embedding request {request.request_id}: {str(e)}"
            )
            response = EmbeddingServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=operation_name,
                raw_data={"error": str(e)},
            )

        return response

    async def _serve_async_impl(
        self, request: EmbeddingServiceRequest
    ) -> EmbeddingServiceResponse:
        """Serve a request asynchronously."""
        logger.debug(
            f"Serving embedding request asynchronously: {request.get_statistics()}"
        )
        operation_name = request.operation_name
        try:
            if operation_name == "embed" or operation_name == "embed_async":
                response = await self.embed_async(request=request)
            else:
                raise ValueError(f"Unknown operation name: {operation_name}")
            response.request_id = request.request_id
        except Exception as e:
            # Handle exceptions and return an error response
            logger.exception(
                f"Error serving embedding request {request.request_id} asynchronously: {str(e)}"
            )
            response = EmbeddingServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=operation_name,
                raw_data={"error": str(e)},
            )

        return response


class EmbeddingServiceRemoteInstance(EmbeddingServiceInstance):
    """Embedding service implementation for remote OpenAI-compatible API."""

    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: EmbeddingServiceInstanceConfig,
    ):
        super().__init__("embedding", instance_name, group_name, config)
        self.launch()

    def get_api_key(self):
        """Get the API key from environment variables."""
        api_key = self.config.api_key
        if api_key is not None:
            return api_key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")
        return api_key

    def get_base_url(self):
        """Get the base URL for the API."""
        if "://" in self.config.location:
            return self.config.location
        ip, port = self.config.location.split(":")
        return f"http://{ip}:{port}/v1"

    def launch(self):
        """Initialize the OpenAI client."""
        from openai import OpenAI, AsyncOpenAI

        self.client = OpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
        )
        self.async_client = AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
        )

    def shutdown(self):
        """Clean up resources."""
        pass

    def embed(self, request: EmbeddingServiceRequest) -> EmbeddingServiceResponse:
        """Generate embeddings using the OpenAI API (with batching for APIs with limits)."""
        x_request_id = request.kwargs.get("X-Request-Id", None) or request.request_id
        texts = list(request.texts)
        batch_size = 10  # Some APIs (e.g. DashScope) limit batch size
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.embeddings.create(
                model=request.model,
                input=batch,
                extra_headers={"X-Request-Id": x_request_id},
            )
            all_embeddings.extend([item.embedding for item in result.data])

        return EmbeddingServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            embeddings=all_embeddings,
            raw_data={},
        )

    async def embed_async(
        self, request: EmbeddingServiceRequest
    ) -> EmbeddingServiceResponse:
        """Generate embeddings asynchronously using the OpenAI API (with batching)."""
        x_request_id = request.kwargs.get("X-Request-Id", None) or request.request_id
        texts = list(request.texts)
        batch_size = 10  # Some APIs (e.g. DashScope) limit batch size
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = await self.async_client.embeddings.create(
                model=request.model,
                input=batch,
                extra_headers={"X-Request-Id": x_request_id},
            )
            all_embeddings.extend([item.embedding for item in result.data])

        return EmbeddingServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            embeddings=all_embeddings,
            raw_data={},
        )


class EmbeddingServiceLocalInstance(EmbeddingServiceInstance):
    """Embedding service implementation for local models."""

    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: EmbeddingServiceInstanceConfig,
    ):
        super().__init__("embedding", instance_name, group_name, config)
        self.device = config.location or "cpu"
        self.model = config.model
        self.async_executor = ThreadPoolExecutor(
            max_workers=os.cpu_count() or 1
        )  # Use a thread pool for async execution
        self.launch()

    def launch(self):
        """Initialize the local embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model, device=self.device)
        except ImportError:
            raise ImportError(
                "SentenceTransformer package is required for local embedding models. "
                "Install it with: pip install sentence-transformers"
            )

    def shutdown(self):
        """Clean up resources."""
        pass

    def embed(self, request: EmbeddingServiceRequest) -> EmbeddingServiceResponse:
        """Generate embeddings using the local model."""
        import numpy as np
        from sentence_transformers.SentenceTransformer import Tensor

        results = self.model.encode(list(request.texts), **request.kwargs)
        if isinstance(results, list):
            embeddings = results
        elif isinstance(results, dict) and "embeddings" in results:
            embeddings = results["embeddings"]
        elif isinstance(results, Tensor):
            embeddings = results.cpu().numpy().tolist()
        elif isinstance(results, np.ndarray):
            embeddings = results.tolist()
        else:
            raise ValueError(
                f"Unexpected result type from model.encode: {type(results)}"
            )
        return EmbeddingServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            embeddings=embeddings,
            raw_data={"response": results},
        )

    async def embed_async(
        self, request: EmbeddingServiceRequest
    ) -> EmbeddingServiceResponse:
        """Generate embeddings asynchronously using the local model."""
        # Run the synchronous embedding in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.async_executor, self.embed, request)


class EmbeddingService(BaseService):
    """Embedding service for managing embedding service instances."""

    def __init__(self, scheduling_policy: str = "round_robin"):
        super().__init__("embedding", scheduling_policy)

    def serve(
        self,
        request: EmbeddingServiceRequest,
        model: str = None,
        instance_name: str = None,
        tags=None,
    ) -> EmbeddingServiceResponse:
        if model is not None:
            tags = tags or set()
            if model not in tags:
                tags.add(model)
        return super().serve(request, instance_name, tags)

    async def serve_async(
        self,
        request: EmbeddingServiceRequest,
        model: str = None,
        instance_name: str = None,
        tags=None,
    ) -> EmbeddingServiceResponse:
        if model is not None:
            tags = tags or set()
            if model not in tags:
                tags.add(model)
        return await super().serve_async(request, instance_name, tags)
