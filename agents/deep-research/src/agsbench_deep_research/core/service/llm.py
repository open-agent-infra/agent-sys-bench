"""
LLM service module for language model interactions.

This module provides service implementations for interacting with
various language models, including remote API-based models and
local models running in containers.
"""

import os
import json
from abc import abstractmethod
import logging
from typing import Iterable, List, Union, cast, Optional, Dict, Any, Type
from pydantic import BaseModel, Field, model_validator
from .base import (
    BaseServiceInstanceConfig,
    BaseServiceInstance,
    BaseService,
    BaseServiceRequest,
    BaseServiceResponse,
    BaseModel,
)
from ..utils import StructuredOutputParser

logger = logging.getLogger(__name__)


class LLMServiceRequest(BaseServiceRequest):
    """LLM-specific service request with convenience attributes."""

    model: str = None
    messages: Optional[List[dict]] = None
    prompt: Optional[Union[str, List[str], List[int], List[List[int]]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    response_format: Optional[Union[Dict[str, Any], Type]] = Field(
        default=None,
        description="If provided, this will be used to specify the expected output format.",
    )
    structured_output_prompting: Optional[bool] = Field(
        default=None,
        description="If True, use prompting to guide the model's output towards structured data.",
    )
    guided_decoding: Optional[Dict[str, Any]] = Field(
        default=None,
        description="If provided, this will be used to guide the model's decoding process.",
    )
    kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the model API.",
    )

    @model_validator(mode="after")
    def populate_fields_from_raw_data(self) -> "LLMServiceRequest":
        """Auto-populate fields from raw_data if they're not explicitly set."""
        for field in [
            "model",
            "messages",
            "prompt",
            "max_tokens",
            "temperature",
            "seed",
            "response_format",
            "structured_output_prompting",
            "guided_decoding",
        ]:
            if getattr(self, field) is None and field in self.raw_data:
                setattr(self, field, self.raw_data[field])

        # Validation checks
        if not self.model:
            raise ValueError("Model must be specified in the request.")
        if not self.messages and not self.prompt:
            raise ValueError(
                "Either messages or prompt must be provided in the request."
            )

        return self

    def get_statistics(self) -> dict:
        """Get statistics for the LLM request."""
        stats = super().get_statistics()
        return {
            **stats,
            "model": self.model,
            "messages_count": len(list(self.messages)) if list(self.messages) else 0,
            "messages_statistics": [
                {
                    "role": msg.get("role", "unknown"),
                    "content_length": len(msg.get("content", "")),
                }
                for msg in list(self.messages)
                if isinstance(msg, dict)
            ],
            "prompt_length": len(self.prompt) if self.prompt else 0,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "response_format": self.response_format,
            "structured_output_prompting": self.structured_output_prompting,
            "guided_decoding": self.guided_decoding,
            "kwargs": self.kwargs,
        }


class LLMServiceResponse(BaseServiceResponse):
    """LLM-specific service response with convenience attributes."""

    text: Optional[str] = None  # The raw text response from the model
    structured_output: Optional[Union[BaseModel, Dict[str, Any], str, list]] = (
        None  # The structured output from the model, if any
    )
    usage: Optional[dict] = None  # Token usage from the LLM API response

    def get_statistics(self) -> dict:
        """Get statistics for the LLM response."""
        stats = super().get_statistics()
        return {
            **stats,
            "text_length": len(self.text) if self.text else 0,
            "structured_output_type": (
                type(self.structured_output).__name__
                if self.structured_output is not None
                else None
            ),
            "structured_output": (
                self.structured_output.model_dump_json()
                if isinstance(self.structured_output, BaseModel)
                else self.structured_output
            ),
            "success": self.success,
            "error": self.error,
        }


class LLMServiceInstanceConfig(BaseServiceInstanceConfig):
    model: str


class LLMServiceInstance(BaseServiceInstance):
    """Base class for LLM service instances.

    This abstract class defines the interface that all LLM service
    implementations must provide, including methods for chat completions
    and text completions in both synchronous and asynchronous forms.
    """

    @abstractmethod
    def chat(self, request: LLMServiceRequest) -> LLMServiceResponse:
        """Generate a chat completion, optionally returning structured output.

        Args:
            request: The LLMServiceRequest object.

        Returns:
            The model's response as a LLMServiceResponse object.
        """
        raise NotImplementedError

    @abstractmethod
    async def chat_async(self, request: LLMServiceRequest) -> LLMServiceResponse:
        """Generate a chat completion asynchronously, optionally returning structured output.

        Args:
            request: The LLMServiceRequest object.

        Returns:
            The model's response as a LLMServiceResponse object.
        """
        raise NotImplementedError

    @abstractmethod
    def text_completion(self, request: LLMServiceRequest) -> LLMServiceResponse:
        """Generate a text completion, optionally returning structured output.

        Args:
            request: The LLMServiceRequest object.

        Returns:
            The model's response as a LLMServiceResponse object.
        """
        raise NotImplementedError

    @abstractmethod
    async def text_completion_async(
        self, request: LLMServiceRequest
    ) -> LLMServiceResponse:
        """Generate a text completion asynchronously, optionally returning structured output.

        Args:
            request: The LLMServiceRequest object.

        Returns:
            The model's response as a LLMServiceResponse object.
        """
        raise NotImplementedError

    def _add_structured_output_instructions(
        self,
        original: Union[List[dict], str],
        response_format: Optional[Union[Dict[str, Any], Type]],
    ) -> Union[List[dict], str]:
        """Add structured output instructions to the original messages or prompt."""
        schema = StructuredOutputParser.extract_schema_from_response_format(
            response_format
        )
        if isinstance(original, list):
            return self._add_structured_output_to_messages(original, schema)
        elif isinstance(original, str):
            return self._add_structured_output_to_prompt(original, schema)
        else:
            raise ValueError(
                "Original must be either a list of messages or a string prompt."
            )

    def _add_structured_output_to_messages(
        self, messages: List[dict], schema: Optional[Dict[str, Any]]
    ) -> List[dict]:
        """Add structured output instructions to messages."""
        if not schema:
            return messages

        # Create instruction message
        schema_str = json.dumps(schema, indent=2)
        instruction = (
            f"Please respond with a valid JSON object that follows this exact schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Your response should be valid JSON that can be parsed directly. "
            f"Do not include any explanation or additional text outside the JSON."
        )

        # Add system message or modify existing one
        modified_messages = list(messages)

        # Check if there's already a system message
        if modified_messages and modified_messages[0].get("role") == "system":
            # Append to existing system message
            modified_messages[0]["content"] += f"\n\n{instruction}"
        else:
            # Add new system message at the beginning
            modified_messages.insert(0, {"role": "system", "content": instruction})

        return modified_messages

    def _add_structured_output_to_prompt(
        self,
        prompt: Union[str, List[str], List[int], List[List[int]], None],
        schema: Optional[Dict[str, Any]],
    ) -> Union[str, List[str], List[int], List[List[int]], None]:
        """Add structured output instructions to prompt."""
        if not schema or not isinstance(prompt, str):
            return prompt

        schema_str = json.dumps(schema, indent=2)
        instruction = (
            f"\n\nPlease respond with a valid JSON object that follows this exact schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Your response should be valid JSON that can be parsed directly."
        )

        return prompt + instruction

    def _extract_text_from_response(self, result) -> Optional[str]:
        """Extract text content from various response formats."""
        text = None
        if hasattr(result, "choices"):
            if hasattr(result.choices[0], "message"):
                text = result.choices[0].message.content
            elif hasattr(result.choices[0], "text"):
                text = result.choices[0].text
        elif isinstance(result, dict):
            if "text" in result:
                text = result["text"]
            elif "content" in result:
                text = result["content"]
            elif "message" in result and isinstance(result["message"], dict):
                text = result["message"].get("content")
        elif hasattr(result, "content"):
            # For Anthropic responses
            if hasattr(result.content, "__iter__") and not isinstance(
                result.content, str
            ):
                # Handle list of content blocks
                text_parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                text = "".join(text_parts)
            else:
                text = str(result.content)

        return text

    def _preprocess_request(self, request: LLMServiceRequest) -> LLMServiceRequest:
        """Preprocess the request before serving."""
        # Ensure model is set
        if not request.model:
            raise ValueError("Model must be specified in the request.")

        # Ensure messages or prompt is provided
        if not request.messages and not request.prompt:
            raise ValueError(
                "Either messages or prompt must be provided in the request."
            )

        # Modify messages and prompt to include structured output instructions
        if request.response_format and request.structured_output_prompting:
            if request.messages:
                request.messages = self._add_structured_output_instructions(
                    list(request.messages), request.response_format
                )
            if request.prompt:
                request.prompt = self._add_structured_output_instructions(
                    request.prompt, request.response_format
                )
        return request

    def _postprocess_response(
        self, response: LLMServiceResponse, request: LLMServiceRequest
    ) -> LLMServiceResponse:
        """Postprocess the response after serving."""
        # Extract text from the response
        text = self._extract_text_from_response(response.raw_data.get("response"))

        # Parse structured output if response format is provided
        structured_output = (
            StructuredOutputParser.parse_structured_output(
                text, request.response_format
            )
            if request.response_format and text
            else None
        )

        # Extract token usage from the raw LLM response
        raw_resp = response.raw_data.get("response")
        usage = None
        if raw_resp is not None:
            if hasattr(raw_resp, "usage") and raw_resp.usage:
                u = raw_resp.usage
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                    "total_tokens": getattr(u, "total_tokens", 0),
                }
            elif isinstance(raw_resp, dict) and "usage" in raw_resp:
                usage = raw_resp["usage"]

        # Update the response with extracted text, structured output, and usage
        response.text = text
        response.structured_output = structured_output
        response.usage = usage

        return response

    def _serve_impl(self, request: LLMServiceRequest) -> LLMServiceResponse:
        """Serve a request."""
        # Cast to the specific request type for type checking
        logger.debug(f"Serving LLM request: {request.get_statistics()}")
        request = self._preprocess_request(request)

        try:
            if request.operation_name == "chat":
                response = self.chat(request)
            elif request.operation_name == "text_completion":
                response = self.text_completion(request)
            else:
                raise ValueError(f"Unknown operation name: {request.operation_name}")

            response = self._postprocess_response(response, request)
            logger.debug(f"LLM response: {response.get_statistics()}")
            return response
        except Exception as e:
            logger.exception(f"Error serving LLM request: {e}")
            return LLMServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=request.operation_name,
                raw_data={},
                text=None,
                structured_output=None,
                success=False,
                error=str(e),
            )

    async def _serve_async_impl(self, request: LLMServiceRequest) -> LLMServiceResponse:
        """Serve a request asynchronously."""
        logger.debug(f"Serving LLM async request: {request.get_statistics()}")
        request = self._preprocess_request(request)

        try:
            if request.operation_name in ["chat", "chat_async"]:
                response = await self.chat_async(request=request)
            elif request.operation_name in ["text_completion", "text_completion_async"]:
                response = await self.text_completion_async(request=request)
            else:
                raise ValueError(f"Unknown operation name: {request.operation_name}")

            response = self._postprocess_response(response, request)
            logger.debug(f"Async LLM response: {response.get_statistics()}")
            return response
        except Exception as e:
            logger.exception(f"Error serving LLM async request: {e}")
            return LLMServiceResponse(
                request_id=request.request_id,
                service_name=self.service_name,
                instance_name=self.instance_name,
                operation_name=request.operation_name,
                raw_data={},
                text=None,
                structured_output=None,
                success=False,
                error=str(e),
            )


class LLMServiceRemoteInstance(LLMServiceInstance):
    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: LLMServiceInstanceConfig,
    ):
        super().__init__("llm", instance_name, group_name, config)
        self.launch()

    def get_api_key(self):
        api_key = self.config.api_key
        if api_key is not None:
            return api_key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")
        return api_key

    def get_base_url(self):
        if "://" in self.config.location:
            return self.config.location
        ip, port = self.config.location.split(":")
        return f"http://{ip}:{port}/v1"

    def get_openai_proxy(self):
        """Get the OpenAI proxy URL from the configuration."""
        return getattr(self.config, "openai_proxy", None)

    def launch(self):
        from openai import OpenAI, AsyncOpenAI
        import httpx
        import certifi
        import ssl

        global_ssl_context = ssl.create_default_context(cafile=certifi.where())

        self.http_client = httpx.Client(
            proxy=self.get_openai_proxy(), verify=global_ssl_context
        )
        self.client = OpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
            http_client=self.http_client,
        )
        self.http_async_client = httpx.AsyncClient(
            proxy=self.get_openai_proxy(), verify=global_ssl_context
        )
        self.async_client = AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
            http_client=self.http_async_client,
        )

    def _prepare_openai_kwargs(self, request: LLMServiceRequest):
        """Prepare kwargs for OpenAI API calls, including structured output support."""
        openai_kwargs = {
            "temperature": request.temperature,
            "seed": request.seed,
            "max_tokens": request.max_tokens,
            "top_p": request.kwargs.get("top_p", None),
        }

        extra_body = {
            "top_k": request.kwargs.get("top_k", None),
            "min_p": request.kwargs.get("min_p", None),
            "repetition_penalty": request.kwargs.get("repetition_penalty", None),
            "length_penalty": request.kwargs.get("length_penalty", None),
            "stop_token_ids": request.kwargs.get("stop_token_ids", None),
            "stop": request.kwargs.get("stop", None),
        }
        extra_body = {k: v for k, v in extra_body.items() if v is not None}

        if extra_body:
            # Add extra body parameters if provided
            openai_kwargs["extra_body"] = extra_body

        # Add request ID header if provided
        x_request_id = request.kwargs.get("X-Request-Id", None) or request.request_id
        if x_request_id:
            openai_kwargs["extra_headers"] = {"X-Request-Id": x_request_id}

        # Handle native structured output for compatible models
        if request.response_format:
            try:
                # For newer OpenAI models that support native structured output
                openai_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "schema": StructuredOutputParser.extract_schema_from_response_format(
                            request.response_format
                        ),
                    },
                }
            except Exception as e:
                logger.debug(
                    f"Native structured output not supported, falling back to prompt engineering: {e}"
                )
        # remove None values from kwargs
        openai_kwargs = {k: v for k, v in openai_kwargs.items() if v is not None}
        return openai_kwargs

    def chat(self, request: LLMServiceRequest) -> LLMServiceResponse:
        openai_kwargs = self._prepare_openai_kwargs(request)

        result = self.client.chat.completions.create(
            messages=request.messages, model=request.model, **openai_kwargs
        )
        return LLMServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            raw_data={"response": result},
            success=True,
        )

    async def chat_async(self, request: LLMServiceRequest) -> LLMServiceResponse:
        openai_kwargs = self._prepare_openai_kwargs(request)

        result = await self.async_client.chat.completions.create(
            messages=request.messages, model=request.model, **openai_kwargs
        )
        return LLMServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            raw_data={"response": result},
            success=True,
        )

    def text_completion(self, request: LLMServiceRequest) -> LLMServiceResponse:
        openai_kwargs = self._prepare_openai_kwargs(request)
        openai_kwargs.pop("response_format", None)

        result = self.client.completions.create(
            model=request.model, prompt=request.prompt, **openai_kwargs
        )
        return LLMServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            raw_data={"response": result},
            success=True,
        )

    async def text_completion_async(
        self, request: LLMServiceRequest
    ) -> LLMServiceResponse:
        openai_kwargs = self._prepare_openai_kwargs(request)
        # Remove response_format for completions endpoint
        openai_kwargs.pop("response_format", None)

        result = await self.async_client.completions.create(
            model=request.model, prompt=request.prompt, **openai_kwargs
        )
        return LLMServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            raw_data={"response": result},
            success=True,
        )

    def shutdown(self):
        pass

    def check_health(self) -> Dict[str, Any]:
        """Perform a health check on the service instance.

        Returns:
            Dictionary with health status and instance information
        """
        # get health status from the client
        try:
            base_url = self.get_base_url()
            health_check_url = base_url.replace("/v1", "/health")
            response = self.http_client.get(health_check_url)
            if response.status_code == 200:
                health_status = "healthy"
            else:
                health_status = "unhealthy"
        except Exception as e:
            logger.error(f"Health check for {self.instance_name} failed: {e}")
            return {
                "healthy": False,
                "status": "unhealthy",
                "service_name": self.service_name,
                "instance_name": self.instance_name,
                "group_name": self.group_name,
                "error": str(e),
            }
        return {
            "healthy": health_status == "healthy",
            "status": health_status,
            "service_name": self.service_name,
            "instance_name": self.instance_name,
            "group_name": self.group_name,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "tags": list(self.tags),
        }


class LLMServiceAnthropicInstance(LLMServiceInstance):
    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: LLMServiceInstanceConfig,
    ):
        super().__init__("llm", instance_name, group_name, config)
        self.launch()

    def get_api_key(self):
        api_key = self.config.api_key if hasattr(self.config, "api_key") else None
        if api_key is not None:
            return api_key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return api_key

    def launch(self):
        from anthropic import Anthropic, AsyncAnthropic

        self.client = Anthropic(api_key=self.get_api_key())
        self.async_client = AsyncAnthropic(api_key=self.get_api_key())

    def _prepare_anthropic_messages(self, messages: List[dict]) -> tuple:
        """Prepare messages for Anthropic API format."""
        messages_list = list(messages)
        system_message = None

        # Extract system message if present
        if messages_list and messages_list[0].get("role") == "system":
            system_message = messages_list[0]["content"]
            messages_list = messages_list[1:]

        return system_message, messages_list

    def _prepare_anthropic_kwargs(self, request: LLMServiceRequest) -> dict:
        """Prepare kwargs for Anthropic API calls, including structured output support."""
        anthropic_kwargs = {
            "temperature": request.temperature,
            # "seed": request.seed,
            "max_tokens": request.max_tokens,
        }

        # Add request ID header if provided
        request_id = request.request_id
        if request_id:
            anthropic_kwargs["extra_headers"] = {"X-Request-Id": request_id}

        # TODO: Implement structured output handling for Anthropic

        # remove None values from kwargs
        anthropic_kwargs = {k: v for k, v in anthropic_kwargs.items() if v is not None}
        return anthropic_kwargs

    def chat(self, request: LLMServiceRequest) -> LLMServiceResponse:
        system_message, messages = self._prepare_anthropic_messages(request.messages)
        anthropic_kwargs = self._prepare_anthropic_kwargs(request)

        result = self.client.messages.create(
            model=request.model,
            messages=messages,
            system=system_message,
            **anthropic_kwargs,
        )
        return LLMServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            raw_data={"response": result},
            success=True,
        )

    async def chat_async(self, request: LLMServiceRequest) -> LLMServiceResponse:
        system_message, messages = self._prepare_anthropic_messages(request.messages)
        anthropic_kwargs = self._prepare_anthropic_kwargs(request)

        result = await self.async_client.messages.create(
            model=request.model,
            messages=messages,
            system=system_message,
            **anthropic_kwargs,
        )
        return LLMServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            raw_data={"response": result},
            success=True,
        )

    def text_completion(self, request: LLMServiceRequest) -> LLMServiceResponse:
        anthropic_kwargs = self._prepare_anthropic_kwargs(request)
        anthropic_kwargs["max_tokens_to_sample"] = anthropic_kwargs["max_tokens"]
        anthropic_kwargs = {
            k: v for k, v in anthropic_kwargs.items() if k != "max_tokens"
        }
        result = self.client.completions.create(
            model=request.model, prompt=request.prompt, **anthropic_kwargs
        )
        return LLMServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            raw_data={"response": result},
            success=True,
        )

    async def text_completion_async(
        self, request: LLMServiceRequest
    ) -> LLMServiceResponse:
        anthropic_kwargs = self._prepare_anthropic_kwargs(request)
        anthropic_kwargs["max_tokens_to_sample"] = anthropic_kwargs["max_tokens"]
        anthropic_kwargs = {
            k: v for k, v in anthropic_kwargs.items() if k != "max_tokens"
        }

        result = await self.async_client.completions.create(
            model=request.model, prompt=request.prompt, **anthropic_kwargs
        )
        return LLMServiceResponse(
            request_id=request.request_id,
            service_name=self.service_name,
            instance_name=self.instance_name,
            operation_name=request.operation_name,
            raw_data={"response": result},
            success=True,
        )


class LLMServiceLocalVLLMInstance(LLMServiceInstance):
    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: LLMServiceInstanceConfig,
    ):
        super().__init__("llm", instance_name, group_name, config)
        self.launch()


class LLMServiceLocalSGLangInstance(LLMServiceInstance):
    def __init__(
        self,
        instance_name: str,
        group_name: str,
        config: LLMServiceInstanceConfig,
    ):
        super().__init__("llm", instance_name, group_name, config)
        self.launch()


class LLMService(BaseService):
    def __init__(self, scheduling_policy: str = "round_robin"):
        super().__init__("llm", scheduling_policy=scheduling_policy)

    def serve(
        self,
        request: LLMServiceRequest,
        model: str = None,
        instance_name: str = None,
        tags=None,
    ) -> LLMServiceResponse:
        model = model or request.model
        if model is not None:
            tags = tags or set()
            if model not in tags:
                tags.add(model)
        return super().serve(request=request, instance_name=instance_name, tags=tags)

    async def serve_async(
        self,
        request: LLMServiceRequest,
        model: str = None,
        instance_name: str = None,
        tags=None,
    ) -> LLMServiceResponse:
        model = model or request.model
        if model is not None:
            tags = tags or set()
            if model not in tags:
                tags.add(model)
        return await super().serve_async(
            request=request, instance_name=instance_name, tags=tags
        )
