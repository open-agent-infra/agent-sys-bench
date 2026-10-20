from typing import Literal, Dict, List
from pydantic import Field, BaseModel
from functools import lru_cache
from langchain_community.utils.math import cosine_similarity
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import RAGChattingInput, RAGChattingOutput
from .utils import get_llm, get_embedding_fn


class RoutingInput(BaseModel):
    query: str


class DSRoutingOutput(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


@lru_cache
def get_structured_llm(schema: Dict | type):
    llm = get_llm()
    structured_llm = llm.with_structured_output(schema)
    return structured_llm


@agsbench_rag_fsm.fast_service
def datasource_routing(
    item: RoutingInput, context: RequestContext = None
) -> DSRoutingOutput:
    """
    Logical routing to the correct service
    """
    structured_llm = get_structured_llm(DSRoutingOutput)
    system = """You are an expert at routing a user question to the appropriate data source.

            Based on the programming language the question is referring to, route it to the relevant data source."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm_input = prompt.invoke(input={"question": item.query})
    llm_output = structured_llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    return llm_output


@lru_cache
def get_prompt_templates() -> List[str]:
    # Two prompts
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise and easy to understand manner. \
    When you don't know the answer to a question you admit that you don't know.

    Here is a question:
    {query}"""

    math_template = """You are a very good mathematician. You are great at answering math questions. \
    You are so good because you are able to break down hard problems into their component parts, \
    answer the component parts, and then put them together to answer the broader question.

    Here is a question:
    {query}"""
    prompt_templates = [physics_template, math_template]
    return prompt_templates


@lru_cache
def get_prompt_embeddings() -> List[List[float]]:
    embedding_fn = get_embedding_fn(
        embedding_model="all-MiniLM-L6-v2", embedding_device="cpu"
    )
    prompt_templates = get_prompt_templates()
    return embedding_fn.embed_documents(prompt_templates)


class PTRoutingOutput(BaseModel):
    prompt_template: str


@agsbench_rag_fsm.fast_service
def prompt_routing(
    item: RoutingInput, context: RequestContext = None
) -> PromptTemplate:
    embedding_fn = get_embedding_fn(
        embedding_model="all-MiniLM-L6-v2", embedding_device="cpu"
    )
    prompt_templates = get_prompt_templates()
    prompt_embeddings = get_prompt_embeddings()
    # Embed question
    query_embedding = embedding_fn.embed_query(item.query)
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    return PromptTemplate.from_template(most_similar)


@agsbench_rag_fsm.fast_service
def chatting_with_routing(
    item: RAGChattingInput, context: RequestContext = None
) -> RAGChattingOutput:
    llm = get_llm()
    prompt = prompt_routing(item=RoutingInput(query=item.query), context=context)
    llm_input = prompt.invoke(input={"query": item.query})
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    answer = llm_output.content
    return RAGChattingOutput(answer=answer)


# %% Test
if __name__ == "__main__":
    from fast_service import FastServiceConfig

    config_dict = {
        "service_impl": "fastapi",
        "client_mode": True,
        "service_list": [],
        "monitor": True,
        "monitor_storage": {"type": "csv", "store_dir": "./.cache/monitor/local_test"},
    }
    config = FastServiceConfig.load_from_dict(config_dict=config_dict)
    agsbench_rag_fsm.setup_client_mode(config=config)

    question = """Why doesn't the following code work:

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
        prompt.invoke("french")
        """
    datasource = datasource_routing(item=RoutingInput(query=question))
    print(f"datasource => {datasource}")

    question = "What's a black hole"
    ans = chatting_with_routing(item=RAGChattingInput(query=question))
    print(f"ans => {ans}")
