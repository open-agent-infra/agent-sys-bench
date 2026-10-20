import os

from abc import ABCMeta, abstractmethod
from typing import Optional

from openai import OpenAI, ChatCompletion, AsyncOpenAI

TRACE_LLM_INFERENCE = os.environ.get("TRACE_LLM_INFERENCE")
ENABLE_TRACE_LLM = (
    TRACE_LLM_INFERENCE.lower() == "true" if TRACE_LLM_INFERENCE is not None else False
)
print(
    f"Is TRACE_LLM_INFERENCE enabled: {ENABLE_TRACE_LLM}; {TRACE_LLM_INFERENCE}",
    flush=True,
)


class LLMResponse:

    content: Optional[str | None]

    completion_tokens: int
    """Number of tokens in the generated completion."""

    prompt_tokens: int
    """Number of tokens in the prompt."""

    def __init__(
        self, content: Optional[str | None], completion_tokens: int, prompt_tokens: int
    ):
        self.content = content
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens


class LLMClient(metaclass=ABCMeta):

    @abstractmethod
    def call_llm(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        top_p: float | None = None,
        call_id: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> LLMResponse:
        pass


class AsyncLLMClient(metaclass=ABCMeta):

    @abstractmethod
    async def call_llm(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        top_p: float | None = None,
        call_id: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> LLMResponse:
        pass


class OpenAIClient(LLMClient):

    openai: OpenAI

    def __init__(
        self,
        api_key: str,
        base_url: str,
    ):
        super().__init__()

        self.openai = OpenAI(api_key=api_key, base_url=base_url)

    def call_llm(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        top_p: float | None = None,
        call_id: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> LLMResponse:

        extra_headers = (
            {"X-Request-Id": f"{call_id}"}
            if ENABLE_TRACE_LLM and call_id is not None
            else None
        )
        response: ChatCompletion = self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
            max_completion_tokens=max_completion_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.prompt_tokens,
        )


class AsyncOpenAIClient(AsyncLLMClient):

    openai: AsyncOpenAI

    def __init__(
        self,
        api_key: str,
        base_url: str,
    ):
        super().__init__()

        self.openai = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def call_llm(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        top_p: float | None = None,
        call_id: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> LLMResponse:
        extra_headers = (
            {"X-Request-Id": f"{call_id}"}
            if ENABLE_TRACE_LLM and call_id is not None
            else None
        )
        response: ChatCompletion = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
            max_completion_tokens=max_completion_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.prompt_tokens,
        )


class LLMClientFactory:

    @staticmethod
    def create_client(
        api_key: str,
        base_url: str,
    ) -> LLMClient:
        return OpenAIClient(api_key=api_key, base_url=base_url)

    @staticmethod
    def create_async_client(
        api_key: str,
        base_url: str,
    ) -> AsyncLLMClient:
        return AsyncOpenAIClient(api_key=api_key, base_url=base_url)
