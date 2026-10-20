from fast_service import RequestContext
from langchain_core.prompts import ChatPromptTemplate
from .prompt_templates import ALL_DEFAULT_TEMPLATES
from .shared import agsbench_rag_fsm
from .shared import JudgingOutput
from .shared import RetrievalJudgingInput
from .shared import DocRelevanceJudgingInput
from .shared import HallucinationJudgingInput
from .shared import AnswerJudgingInput
from .utils import get_llm


@agsbench_rag_fsm.fast_service
def retrieval_judger(
    item: RetrievalJudgingInput, context: RequestContext = None
) -> JudgingOutput:
    question = item.question
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["retrieval_judging"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(
        input={
            "question": question,
        }
    )
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return JudgingOutput(judgement="yes" in answer.lower(), usage=usage)


@agsbench_rag_fsm.fast_service
def doc_relevance_judger(
    item: DocRelevanceJudgingInput, context: RequestContext = None
) -> JudgingOutput:
    question = item.question
    q_ctx = item.context
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["doc_relevance_judging"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(
        input={
            "context": q_ctx,
            "question": question,
        }
    )
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return JudgingOutput(judgement="yes" in answer.lower(), usage=usage)


@agsbench_rag_fsm.fast_service
def hallucination_judger(
    item: HallucinationJudgingInput, context: RequestContext = None
) -> JudgingOutput:
    answer = item.answer
    q_ctx = item.context
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["hallucination_judging"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(
        input={
            "context": q_ctx,
            "answer": answer,
        }
    )
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return JudgingOutput(judgement="yes" in answer.lower(), usage=usage)


@agsbench_rag_fsm.fast_service
def answer_judger(
    item: AnswerJudgingInput, context: RequestContext = None
) -> JudgingOutput:
    question = item.question
    answer = item.answer
    llm = get_llm()
    prompt_template = ALL_DEFAULT_TEMPLATES["answer_judging"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(
        input={
            "question": question,
            "answer": answer,
        }
    )
    llm_output = llm.invoke(
        input=llm_input,
        extra_headers={"X-Request-Id": context.call_id} if context else None,
    )
    answer = llm_output.content
    usage = llm_output.usage_metadata
    return JudgingOutput(judgement="yes" in answer.lower(), usage=usage)
