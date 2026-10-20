from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_serializer
import random
import uuid
from datetime import datetime, timezone


class BaseRequest(BaseModel):
    """Base class for all requests.

    Attributes:
        request_id (str): Unique identifier for the request (auto-generated if not provided)
        timestamp (datetime): When the request was created
        metadata (Dict[str, Any]): Additional metadata for the request
        raw_data (Dict): Operation-specific request data
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_data: Dict = Field(default_factory=dict)
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout for the request in seconds. If None, no timeout is applied.",
    )
    max_retries: int = Field(
        default=0,
        description="Maximum number of retries for the request. If 0, no retries are attempted.",
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay in seconds between retries",
    )
    on_retry: Optional[Callable[[int, Exception], None]] = Field(
        default=None,
        description="Optional callback function to execute on each retry. "
        "It receives the retry count and the exception raised.",
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def model_dump(self, *, mode: str = "python", **kwargs):
        if mode == "json":
            return super().model_dump(mode="json", **kwargs)
        data = super().model_dump(**kwargs)
        if isinstance(self.timestamp, datetime):
            data["timestamp"] = self.timestamp.isoformat()
        return data

    def get_statistics(self) -> dict:
        """Get statistics for the request."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "raw_data_keys": list(self.raw_data.keys()),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }


class BaseResponse(BaseModel):
    """Base class for all responses.

    Attributes:
        request_id (str): ID of the original request
        timestamp (datetime): When the response was created
        metadata (Dict[str, Any]): Additional metadata for the response
        raw_data (Dict): Operation-specific response data
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_data: Dict = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def model_dump(self, *, mode: str = "python", **kwargs):
        if mode == "json":
            return super().model_dump(mode="json", **kwargs)
        data = super().model_dump(**kwargs)
        if isinstance(self.timestamp, datetime):
            data["timestamp"] = self.timestamp.isoformat()
        return data

    def get_statistics(self) -> dict:
        """Get statistics for the response."""
        return {
            "request_id": self.request_id,
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else self.timestamp
            ),
            "metadata": self.metadata,
            "raw_data_keys": (
                list(self.raw_data.keys()) if isinstance(self.raw_data, dict) else []
            ),
        }


class ServiceRequest(BaseRequest):
    """Request to a service.

    Attributes:
        service_name (str): Name of the target service
        operation_name (str): Name of the operation to perform
        raw_data (Dict): Operation-specific request data
    """

    service_name: str
    operation_name: str

    def get_statistics(self) -> dict:
        """Get statistics for the service request."""
        stats = super().get_statistics()
        stats.update(
            {
                "service_name": self.service_name,
                "operation_name": self.operation_name,
            }
        )
        return stats


class ServiceResponse(BaseResponse):
    """Response from a service.

    Attributes:
        service_name (str): Name of the responding service
        operation_name (str): Name of the performed operation
        instance_name (str): Name of the specific service instance
        raw_data (Dict): Operation-specific response data
    """

    service_name: str
    operation_name: str
    instance_name: str

    def get_statistics(self) -> dict:
        """Get statistics for the service response."""
        stats = super().get_statistics()
        stats.update(
            {
                "service_name": self.service_name,
                "operation_name": self.operation_name,
                "instance_name": self.instance_name,
            }
        )
        return stats


class TaskRequest(BaseRequest):
    """Request for a task.

    Attributes:
        task_name (str): Name of the task
        raw_data (Dict): Task-specific request data
    """

    task_name: str = ""

    def get_statistics(self) -> dict:
        """Get statistics for the task request."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "raw_data_keys": (
                    list(self.raw_data.keys())
                    if isinstance(self.raw_data, dict)
                    else []
                ),
            }
        )
        return stats


class TaskResponse(BaseResponse):
    """Response from a task.

    Attributes:
        task_name (str): Name of the task
        raw_data (Dict): Task-specific response data
        usage (dict): Token usage from the LLM API (if applicable)
    """

    task_name: str = ""
    success: bool = Field(default=True, description="Whether the task was successful")
    error: str = Field(default=None, description="Error message if the task failed")
    usage: Optional[dict] = Field(default=None, description="Token usage from LLM API")

    def get_statistics(self) -> dict:
        """Get statistics for the task response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "success": self.success,
                "error": self.error,
            }
        )
        return stats
