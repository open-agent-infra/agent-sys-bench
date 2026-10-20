from fast_service import RequestContext
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
from fastapi import Body

from .prompt_templates import ALL_DEFAULT_TEMPLATES
from .shared import agsbench_rag_fsm
from .shared import GenerationInput, GenerationOutput
from .utils import get_llm, get_logger


@agsbench_rag_fsm.fast_service
def direct_generation(
    question: Annotated[str, Body()], context: RequestContext = None
) -> GenerationOutput:
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["direct_generation"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(input={"question": question})
    get_logger().debug(f"direct_generation llm input {context.request_id}: {llm_input}")
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    get_logger().debug(
        f"direct_generation llm output {context.request_id}: {llm_output}"
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return GenerationOutput(answer=answer, usage=usage)


@agsbench_rag_fsm.fast_service
def normal_generation(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["normal_generation"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(input={"context": query_context, "question": question})
    get_logger().debug(f"normal_generation llm input {context.request_id}: {llm_input}")
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    get_logger().debug(
        f"normal_generation llm output {context.request_id}: {llm_output}"
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return GenerationOutput(answer=answer, usage=usage)


@agsbench_rag_fsm.fast_service
def decomposition_recursive_generation(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    q_a_pairs: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["decomposition_recursive_generation"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(
        input={"context": query_context, "question": question, "q_a_pairs": q_a_pairs}
    )
    get_logger().debug(
        f"decomposition_recursive_generation llm input {context.request_id}: {llm_input}"
    )
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    get_logger().debug(
        f"decomposition_recursive_generation llm output {context.request_id}: {llm_output}"
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return GenerationOutput(answer=answer, usage=usage)


@agsbench_rag_fsm.fast_service
def decomposition_individual_final_generation(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["decomposition_individual_generation"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(input={"context": query_context, "question": question})
    get_logger().debug(
        f"decomposition_individual_final_generation llm input {context.request_id}: {llm_input}"
    )
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    get_logger().debug(
        f"decomposition_individual_final_generation llm output {context.request_id}: {llm_output}"
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return GenerationOutput(answer=answer, usage=usage)


@agsbench_rag_fsm.fast_service
def step_back_generation(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    step_back_context: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["step_back_generation"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(
        input={
            "context": query_context,
            "question": question,
            "step_back_context": step_back_context,
        }
    )
    get_logger().debug(
        f"step_back_generation llm input {context.request_id}: {llm_input}"
    )
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    get_logger().debug(
        f"step_back_generation llm output {context.request_id}: {llm_output}"
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return GenerationOutput(answer=answer, usage=usage)


@agsbench_rag_fsm.fast_service
def HyDE_generation(
    question: Annotated[str, Body()],
    query_context: Annotated[str, Body()],
    context: RequestContext = None,
) -> GenerationOutput:
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["hyde_generation"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(input={"context": query_context, "question": question})
    get_logger().debug(f"HyDE_generation llm input {context.request_id}: {llm_input}")
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    get_logger().debug(f"HyDE_generation llm output {context.request_id}: {llm_output}")
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return GenerationOutput(answer=answer, usage=usage)


@agsbench_rag_fsm.fast_service
def generation(
    item: GenerationInput, context: RequestContext = None
) -> GenerationOutput:
    if item.method == "normal":
        ret = normal_generation(
            question=item.query,
            query_context=item.context,
            context=context,
        )
    elif item.method == "direct":
        ret = direct_generation(
            question=item.query,
            context=context,
        )
    elif item.method == "decomposition_recursive":
        ret = decomposition_recursive_generation(
            question=item.query,
            query_context=item.context,
            q_a_pairs=item.method_args["q_a_pairs"],
            context=context,
        )
    elif item.method == "decomposition_individual":
        ret = decomposition_individual_final_generation(
            question=item.query,
            query_context=item.context,
            context=context,
        )
    elif item.method == "step_back":
        ret = step_back_generation(
            question=item.query,
            query_context=item.context,
            step_back_context=item.method_args["step_back_context"],
            context=context,
        )
    elif item.method == "HyDE":
        ret = HyDE_generation(
            question=item.query,
            query_context=item.context,
            context=context,
        )
    else:
        raise NotImplementedError(f"Unsupported generation method: {item.method}")
    return ret
