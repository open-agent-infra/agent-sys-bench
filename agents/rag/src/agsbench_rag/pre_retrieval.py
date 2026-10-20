from fast_service import RequestContext
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from .prompt_templates import ALL_DEFAULT_TEMPLATES
from .utils import get_logger
from .shared import agsbench_rag_fsm
from .shared import QueryTranslationInput, QueryTranslationOutput
from .utils import get_llm


def step_back_query_transformation(query: str, call_id: str = None) -> str:
    prompt_template = ALL_DEFAULT_TEMPLATES["query_step_back"]
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt_template,
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )
    llm = get_llm()
    llm_input = prompt.invoke(input={"question": query})
    llm_output = llm.invoke(
        input=llm_input, extra_headers={"X-Request-Id": call_id} if call_id else None
    )
    queries = llm_output.content.split("\n")
    new_query = queries[0]
    usage = llm_output.usage_metadata
    get_logger().debug(f"Generated query with QueryStepBack: {new_query}")
    return new_query, usage


def HyDE_query_transformation(query: str, call_id: str = None) -> str:
    query_template = ALL_DEFAULT_TEMPLATES["hyde"]
    prompt = ChatPromptTemplate.from_template(template=query_template)
    llm = get_llm()
    llm_input = prompt.invoke(input={"question": query})
    llm_output = llm.invoke(
        input=llm_input, extra_headers={"X-Request-Id": call_id} if call_id else None
    )
    queries = llm_output.content.split("\n")
    new_query = "".join(queries)
    usage = llm_output.usage_metadata
    get_logger().debug(f"Generated query with HyDE: {new_query}")
    return new_query, usage


def query_rewrite(query: str, call_id: str = None) -> str:
    query_template = ALL_DEFAULT_TEMPLATES["query_rewriting"]
    prompt = ChatPromptTemplate.from_template(template=query_template)
    llm = get_llm()
    llm_input = prompt.invoke(input={"question": query})
    llm_output = llm.invoke(
        input=llm_input, extra_headers={"X-Request-Id": call_id} if call_id else None
    )
    queries = llm_output.content.split("\n")
    new_query = "".join(queries)
    usage = llm_output.usage_metadata
    get_logger().debug(f"Generated query with HyDE: {new_query}")
    return new_query, usage


def query_transformation(query: str, method: str, call_id: str = None) -> str:
    if method == "QueryStepBack":
        new_query, usage = step_back_query_transformation(query, call_id)
    elif method == "HyDE":
        new_query, usage = HyDE_query_transformation(query, call_id)
    elif method == "Rewrite":
        new_query, usage = query_rewrite(query, call_id)
    else:
        raise NotImplementedError(f"Unsupported query_translation_type: {method}")
    return new_query, usage


def multi_queries_expansion(query: str, call_id: str = None) -> list[str]:
    query_template = ALL_DEFAULT_TEMPLATES["multi_query"]
    prompt = ChatPromptTemplate.from_template(template=query_template)
    llm = get_llm()
    llm_input = prompt.invoke(input={"question": query})
    llm_output = llm.invoke(
        input=llm_input, extra_headers={"X-Request-Id": call_id} if call_id else None
    )
    queries = llm_output.content.split("\n")
    usage = llm_output.usage_metadata
    get_logger().debug(f"Generated queries with MultiQuery: {queries}")
    return queries, usage


def multi_queries_with_fusion_expansion(query: str, call_id: str = None) -> list[str]:
    query_template = ALL_DEFAULT_TEMPLATES["multi_query_with_fusion"]
    prompt = ChatPromptTemplate.from_template(template=query_template)
    llm = get_llm()
    llm_input = prompt.invoke(input={"question": query})
    llm_output = llm.invoke(
        input=llm_input, extra_headers={"X-Request-Id": call_id} if call_id else None
    )
    queries = llm_output.content.split("\n")
    usage = llm_output.usage_metadata
    get_logger().debug(f"Generated queries with MultiQueryWithFusion: {queries}")
    return queries, usage


def query_decomposition_expansion(query: str, call_id: str = None) -> list[str]:
    query_template = ALL_DEFAULT_TEMPLATES["query_decomposition"]
    prompt = ChatPromptTemplate.from_template(template=query_template)
    llm = get_llm()
    llm_input = prompt.invoke(input={"question": query})
    llm_output = llm.invoke(
        input=llm_input, extra_headers={"X-Request-Id": call_id} if call_id else None
    )
    queries = llm_output.content.split("\n")
    usage = llm_output.usage_metadata
    get_logger().debug(f"Generated queries with QueryDecomposition: {queries}")
    return queries, usage


def query_expansion(query: str, method: str, call_id: str = None) -> list[str]:
    if method == "MultiQuery":
        queries = multi_queries_expansion(query, call_id)
    elif method == "MultiQueryWithFusion":
        queries = multi_queries_with_fusion_expansion(query, call_id)
    elif method == "QueryDecomposition":
        queries = query_decomposition_expansion(query, call_id)
    else:
        raise NotImplementedError(f"Unsupported query_translation_type: {method}")
    return queries


@agsbench_rag_fsm.fast_service
def query_translation(
    item: QueryTranslationInput, context: RequestContext = None
) -> QueryTranslationOutput:
    method = item.method
    transformation_types = [
        "QueryStepBack",
        "HyDE",
    ]
    expansion_types = [
        "MultiQuery",
        "MultiQueryWithFusion",
        "QueryDecomposition",
    ]
    if method in transformation_types:
        query, usage = query_transformation(item.query, method, context.call_id)
        ret = QueryTranslationOutput(queries=[query], qtypes=["transformation"], usage=usage)
    elif method in expansion_types:
        queries, usage = query_expansion(item.query, method, context.call_id)
        ret = QueryTranslationOutput(
            queries=queries, qtypes=["expansion"] * len(queries), usage=usage
        )
    else:
        raise NotImplementedError(f"Unsupported query_translation_type: {method}")
    return ret
