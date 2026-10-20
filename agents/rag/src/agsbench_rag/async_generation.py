import asyncio
from fast_service import RequestContext
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
from fastapi import Body
from .shared import agsbench_rag_fsm
from .shared import GenerationInput, GenerationOutput
from .prompt_templates import ALL_DEFAULT_TEMPLATES
from .async_utils import get_llm, get_logger


class AsyncGenerationEngine:
    """Async generation engine for LLM-based text generation."""

    def __init__(self):
        self._llm = None

    async def get_llm(self):
        """Get LLM instance asynchronously."""
        if self._llm is None:
            # get_llm is a cached function, so it's fast and doesn't need executor
            self._llm = get_llm()
        return self._llm

    async def generate_with_template(
        self,
        template_name: str,
        template_vars: dict,
        context: RequestContext = None,
    ) -> str:
        """Generate text using a template asynchronously."""
        llm = await self.get_llm()

        if template_name not in ALL_DEFAULT_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        prompt_template = ALL_DEFAULT_TEMPLATES[template_name]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)

        # Prepare LLM input (this is fast, no need for executor)
        llm_input = prompt.invoke(template_vars)
        get_logger().debug(
            f"{template_name} llm input {context.request_id if context else 'N/A'}: {llm_input}"
        )

        # Use native async LLM call
        try:
            # Try native async method first
            llm_output = await llm.ainvoke(
                llm_input,
                extra_headers={"X-Request-Id": context.call_id} if context else None,
            )
        except AttributeError:
            # Fallback to sync method in executor if async not available
            get_logger().warning(
                f"LLM {type(llm).__name__} doesn't support ainvoke, falling back to sync"
            )
            from .async_utils import async_run_in_executor

            llm_output = await async_run_in_executor(
                llm.invoke,
                llm_input,
                extra_headers={"X-Request-Id": context.call_id} if context else None,
            )

        get_logger().debug(
            f"{template_name} llm output {context.request_id if context else 'N/A'}: {llm_output}"
        )

        return llm_output.content


# Global async generation engine instance
_async_generation_engine = None


async def get_async_generation_engine() -> AsyncGenerationEngine:
    """Get async generation engine instance."""
    global _async_generation_engine
    if _async_generation_engine is None:
        _async_generation_engine = AsyncGenerationEngine()
    return _async_generation_engine


@agsbench_rag_fsm.fast_service
async def normal_generation_async(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    """Normal generation asynchronously."""
    engine = await get_async_generation_engine()

    answer = await engine.generate_with_template(
        template_name="normal_generation",
        template_vars={"context": query_context, "question": question},
        context=context,
    )

    return GenerationOutput(answer=answer)


@agsbench_rag_fsm.fast_service
async def direct_generation_async(
    question: Annotated[str, Body()], context: RequestContext = None
) -> GenerationOutput:
    """Direct generation without context asynchronously."""
    engine = await get_async_generation_engine()

    answer = await engine.generate_with_template(
        template_name="direct_generation",
        template_vars={"question": question},
        context=context,
    )

    return GenerationOutput(answer=answer)


@agsbench_rag_fsm.fast_service
async def decomposition_recursive_generation_async(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    q_a_pairs: Annotated[list[tuple[str, str]], Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    """Decomposition recursive generation asynchronously."""
    engine = await get_async_generation_engine()

    # Format Q&A pairs for the template
    formatted_qa_pairs = "\n".join([f"Q: {q}\nA: {a}" for q, a in q_a_pairs])

    answer = await engine.generate_with_template(
        template_name="decomposition_recursive_generation",
        template_vars={
            "context": query_context,
            "question": question,
            "q_a_pairs": formatted_qa_pairs,
        },
        context=context,
    )

    return GenerationOutput(answer=answer)


@agsbench_rag_fsm.fast_service
async def decomposition_individual_final_generation_async(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    """Decomposition individual final generation asynchronously."""
    engine = await get_async_generation_engine()

    answer = await engine.generate_with_template(
        template_name="decomposition_individual_final_generation",
        template_vars={"context": query_context, "question": question},
        context=context,
    )

    return GenerationOutput(answer=answer)


@agsbench_rag_fsm.fast_service
async def step_back_generation_async(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    step_back_context: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    """Step back generation asynchronously."""
    engine = await get_async_generation_engine()

    answer = await engine.generate_with_template(
        template_name="step_back_generation",
        template_vars={
            "context": query_context,
            "question": question,
            "step_back_context": step_back_context,
        },
        context=context,
    )

    return GenerationOutput(answer=answer)


@agsbench_rag_fsm.fast_service
async def HyDE_generation_async(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    """HyDE generation asynchronously."""
    engine = await get_async_generation_engine()

    answer = await engine.generate_with_template(
        template_name="hyde_generation",
        template_vars={"context": query_context, "question": question},
        context=context,
    )

    return GenerationOutput(answer=answer)


@agsbench_rag_fsm.fast_service
async def generation_async(
    item: GenerationInput, context: RequestContext = None
) -> GenerationOutput:
    """Main generation function that routes to specific generation methods asynchronously."""
    if item.method == "normal":
        return await normal_generation_async(
            question=item.query,
            query_context=item.context,
            context=context,
        )
    elif item.method == "direct":
        return await direct_generation_async(
            question=item.query,
            context=context,
        )
    elif item.method == "decomposition_recursive":
        return await decomposition_recursive_generation_async(
            question=item.query,
            query_context=item.context,
            q_a_pairs=item.method_args["q_a_pairs"],
            context=context,
        )
    elif item.method == "decomposition_individual":
        return await decomposition_individual_final_generation_async(
            question=item.query,
            query_context=item.context,
            context=context,
        )
    elif item.method == "step_back":
        return await step_back_generation_async(
            question=item.query,
            query_context=item.context,
            step_back_context=item.method_args["step_back_context"],
            context=context,
        )
    elif item.method == "HyDE":
        return await HyDE_generation_async(
            question=item.query,
            query_context=item.context,
            context=context,
        )
    else:
        raise NotImplementedError(f"Unsupported generation method: {item.method}")


# Advanced async generation functions
class AdvancedAsyncGenerationEngine:
    """Advanced async generation engine with additional capabilities."""

    def __init__(self):
        self.base_engine = None

    async def get_base_engine(self):
        """Get base generation engine."""
        if self.base_engine is None:
            self.base_engine = await get_async_generation_engine()
        return self.base_engine

    async def batch_generation(
        self,
        items: list[GenerationInput],
        max_concurrency: int = 5,
        context: RequestContext = None,
    ) -> list[GenerationOutput]:
        """Generate responses for multiple inputs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def generate_with_semaphore(item):
            async with semaphore:
                return await generation_async(item=item, context=context)

        tasks = [generate_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    async def streaming_generation(
        self,
        item: GenerationInput,
        context: RequestContext = None,
    ):
        """Generate response with streaming (placeholder for future implementation)."""
        # This would require streaming-capable LLM clients
        # For now, return the regular generation result
        result = await generation_async(item=item, context=context)
        yield result.answer

    async def multi_model_generation(
        self,
        item: GenerationInput,
        model_configs: list[dict],
        voting_strategy: str = "majority",
        context: RequestContext = None,
    ) -> GenerationOutput:
        """Generate using multiple models and combine results."""
        # This is a placeholder for multi-model generation
        # In practice, you'd need to configure multiple LLM instances

        # For now, just use the default model
        return await generation_async(item=item, context=context)

    async def adaptive_generation(
        self,
        item: GenerationInput,
        quality_threshold: float = 0.8,
        max_attempts: int = 3,
        context: RequestContext = None,
    ) -> GenerationOutput:
        """Generate with quality checking and retry logic."""
        for attempt in range(max_attempts):
            try:
                result = await generation_async(item=item, context=context)

                # Simple quality check (length-based)
                if len(result.answer.strip()) > 10:  # Basic quality check
                    return result

                get_logger().warning(
                    f"Generation attempt {attempt + 1} produced low-quality result"
                )

            except Exception as e:
                get_logger().warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise e

        # Return a fallback response if all attempts failed
        return GenerationOutput(
            answer="I apologize, but I'm unable to provide a satisfactory answer at this time."
        )


# Global advanced async generation engine instance
_advanced_async_generation_engine = None


async def get_advanced_async_generation_engine() -> AdvancedAsyncGenerationEngine:
    """Get advanced async generation engine instance."""
    global _advanced_async_generation_engine
    if _advanced_async_generation_engine is None:
        _advanced_async_generation_engine = AdvancedAsyncGenerationEngine()
    return _advanced_async_generation_engine
