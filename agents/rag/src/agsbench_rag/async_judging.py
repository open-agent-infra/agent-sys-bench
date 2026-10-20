import asyncio
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import (
    JudgingOutput,
    RetrievalJudgingInput,
    DocRelevanceJudgingInput,
    HallucinationJudgingInput,
    AnswerJudgingInput,
)
from .async_generation import get_async_generation_engine
from .async_utils import get_logger


@agsbench_rag_fsm.fast_service
async def retrieval_judger_async(
    item: RetrievalJudgingInput, context: RequestContext = None
) -> JudgingOutput:
    """Judge whether retrieval is needed for a question asynchronously."""
    engine = await get_async_generation_engine()

    response = await engine.generate_with_template(
        template_name="retrieval_judging",
        template_vars={"question": item.question},
        context=context,
    )

    # Parse the response
    response = response.strip().upper()
    judgement = "YES" in response

    get_logger().debug(
        f"retrieval_judger_async output {context.request_id}: {judgement}"
    )

    return JudgingOutput(judgement=judgement)


@agsbench_rag_fsm.fast_service
async def doc_relevance_judger_async(
    item: DocRelevanceJudgingInput, context: RequestContext = None
) -> JudgingOutput:
    """Judge document relevance to a question asynchronously."""
    engine = await get_async_generation_engine()

    response = await engine.generate_with_template(
        template_name="doc_relevance_judging",
        template_vars={"question": item.question, "context": item.context},
        context=context,
    )

    # Parse the response
    response = response.strip().upper()
    judgement = "YES" in response

    get_logger().debug(
        f"doc_relevance_judger_async output {context.request_id}: {judgement}"
    )

    return JudgingOutput(judgement=judgement)


@agsbench_rag_fsm.fast_service
async def hallucination_judger_async(
    item: HallucinationJudgingInput, context: RequestContext = None
) -> JudgingOutput:
    """Judge whether an answer contains hallucinations asynchronously."""
    engine = await get_async_generation_engine()

    response = await engine.generate_with_template(
        template_name="hallucination_judging",
        template_vars={"context": item.context, "answer": item.answer},
        context=context,
    )

    # Parse the response
    response = response.strip().upper()
    judgement = "YES" in response  # YES means no hallucination

    get_logger().debug(
        f"hallucination_judger_async output {context.request_id}: {judgement}"
    )

    return JudgingOutput(judgement=judgement)


@agsbench_rag_fsm.fast_service
async def answer_judger_async(
    item: AnswerJudgingInput, context: RequestContext = None
) -> JudgingOutput:
    """Judge answer quality asynchronously."""
    engine = await get_async_generation_engine()

    response = await engine.generate_with_template(
        template_name="answer_judging",
        template_vars={"question": item.question, "answer": item.answer},
        context=context,
    )

    # Parse the response
    response = response.strip().upper()
    judgement = "YES" in response

    get_logger().debug(f"answer_judger_async output {context.request_id}: {judgement}")

    return JudgingOutput(judgement=judgement)


# Advanced async judging functions
class AdvancedAsyncJudgingEngine:
    """Advanced async judging engine with additional capabilities."""

    def __init__(self):
        pass

    async def multi_criteria_judging(
        self,
        question: str,
        answer: str,
        context: str = None,
        criteria: list[str] = None,
        context_obj: RequestContext = None,
    ) -> dict[str, bool]:
        """Judge answer against multiple criteria concurrently."""
        if criteria is None:
            criteria = ["relevance", "completeness", "accuracy", "clarity"]

        tasks = []

        if "relevance" in criteria:
            tasks.append(
                ("relevance", self._judge_relevance(question, answer, context_obj))
            )

        if "completeness" in criteria:
            tasks.append(
                (
                    "completeness",
                    self._judge_completeness(question, answer, context_obj),
                )
            )

        if "accuracy" in criteria and context:
            tasks.append(
                ("accuracy", self._judge_accuracy(answer, context, context_obj))
            )

        if "clarity" in criteria:
            tasks.append(("clarity", self._judge_clarity(answer, context_obj)))

        if "hallucination" in criteria and context:
            tasks.append(
                (
                    "hallucination",
                    hallucination_judger_async(
                        HallucinationJudgingInput(answer=answer, context=context),
                        context_obj,
                    ),
                )
            )

        # Execute all judgments concurrently
        criterion_names = [name for name, _ in tasks]
        judgment_tasks = [task for _, task in tasks]

        results = await asyncio.gather(*judgment_tasks, return_exceptions=True)

        # Combine results
        judgments = {}
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                if hasattr(result, "judgement"):
                    judgments[criterion_names[i]] = result.judgement
                else:
                    judgments[criterion_names[i]] = result
            else:
                get_logger().warning(f"Judgment {criterion_names[i]} failed: {result}")
                judgments[criterion_names[i]] = False

        return judgments

    async def _judge_relevance(
        self, question: str, answer: str, context: RequestContext = None
    ) -> bool:
        """Judge answer relevance to question."""
        result = await answer_judger_async(
            AnswerJudgingInput(question=question, answer=answer), context
        )
        return result.judgement

    async def _judge_completeness(
        self, question: str, answer: str, context: RequestContext = None
    ) -> bool:
        """Judge answer completeness."""
        engine = await get_async_generation_engine()

        response = await engine.generate_with_template(
            template_name="completeness_judging",
            template_vars={"question": question, "answer": answer},
            context=context,
        )

        return "YES" in response.strip().upper()

    async def _judge_accuracy(
        self, answer: str, context: str, context_obj: RequestContext = None
    ) -> bool:
        """Judge answer accuracy against context."""
        result = await hallucination_judger_async(
            HallucinationJudgingInput(answer=answer, context=context), context_obj
        )
        return result.judgement

    async def _judge_clarity(self, answer: str, context: RequestContext = None) -> bool:
        """Judge answer clarity."""
        engine = await get_async_generation_engine()

        response = await engine.generate_with_template(
            template_name="clarity_judging",
            template_vars={"answer": answer},
            context=context,
        )

        return "YES" in response.strip().upper()

    async def adaptive_judging(
        self,
        question: str,
        answer: str,
        context: str = None,
        question_type: str = "general",
        context_obj: RequestContext = None,
    ) -> dict:
        """Adaptively choose judging criteria based on question type."""
        if question_type == "factual":
            criteria = ["accuracy", "completeness", "hallucination"]
        elif question_type == "analytical":
            criteria = ["relevance", "completeness", "clarity"]
        elif question_type == "creative":
            criteria = ["relevance", "clarity"]
        else:
            criteria = ["relevance", "completeness", "clarity"]

        # Add hallucination check if context is available
        if context and "hallucination" not in criteria:
            criteria.append("hallucination")

        judgments = await self.multi_criteria_judging(
            question=question,
            answer=answer,
            context=context,
            criteria=criteria,
            context_obj=context_obj,
        )

        # Calculate overall score
        overall_score = sum(judgments.values()) / len(judgments) if judgments else 0.0

        return {
            "criteria_judgments": judgments,
            "overall_score": overall_score,
            "question_type": question_type,
        }


# Global advanced async judging engine instance
_advanced_async_judging_engine = None


async def get_advanced_async_judging_engine() -> AdvancedAsyncJudgingEngine:
    """Get advanced async judging engine instance."""
    global _advanced_async_judging_engine
    if _advanced_async_judging_engine is None:
        _advanced_async_judging_engine = AdvancedAsyncJudgingEngine()
    return _advanced_async_judging_engine
