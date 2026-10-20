"""LLM task module for deep research operations."""

from typing import List, Optional
import logging
from pydantic import Field

from agsbench_deep_research.core.schema.io import TaskRequest, TaskResponse
from agsbench_deep_research.core.registry_setup import get_service_manager
from agsbench_deep_research.core.service.llm import (
    LLMServiceRequest,
    LLMServiceResponse,
    LLMService,
)
from agsbench_deep_research.core.service.search import RetrievedDocument
from ..shared import agsbench_deep_research_fsm, RequestContext
from ..state import Section, Sections, Queries, Feedback, SearchResultSummary

logger = logging.getLogger(__name__)


class ReportPlanningRequest(TaskRequest):
    """Request model for report planning LLM task."""

    task_name: str = "report_planning_llm_task"

    topic: str = Field(description="The topic to create a report plan for")
    report_structure: str = Field(description="The report structure template")
    context: str = Field(description="Context from web search to inform planning")
    feedback: str = Field(default="", description="Feedback on previous plans")
    model: Optional[str] = Field(default=None, description="Language model to use")
    max_tokens: Optional[int] = Field(
        default=4096, description="Maximum tokens for generation"
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Temperature for generation"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for llm generation (if needed)"
    )
    extra_gen_kwargs: Optional[dict] = Field(
        default_factory=dict,
        description="Additional generation parameters (top_p, top_k, min_p, etc) for the LLM, if needed",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the report planning request."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "topic": self.topic,
                "report_structure": self.report_structure,
                "context_length": len(self.context),
                "feedback_length": len(self.feedback),
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
                "extra_gen_kwargs": self.extra_gen_kwargs,
            }
        )
        return stats


class ReportPlanningResponse(TaskResponse):
    """Response model for report planning LLM task."""

    task_name: str = "report_planning_llm_task"

    sections: List[Section] = Field(description="The planned report sections")
    success: bool = Field(description="Whether the LLM call was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def get_statistics(self) -> dict:
        """Get statistics for the report planning response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "sections_count": len(self.sections),
                "success": self.success,
                "error": self.error,
                "sections": [section.model_dump() for section in self.sections],
            }
        )
        return stats


class PlannerQueryWritingRequest(TaskRequest):
    """Request model for planner query writing LLM task."""

    task_name: str = "planner_query_writing_llm_task"

    topic: str = Field(description="The main topic")
    report_structure: str = Field(description="The report structure template")
    number_of_queries: int = Field(
        default=2, description="Number of queries to generate"
    )
    model: Optional[str] = Field(default=None, description="Language model to use")
    max_tokens: Optional[int] = Field(
        default=4096, description="Maximum tokens for generation"
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Temperature for generation"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for llm generation (if needed)"
    )
    extra_gen_kwargs: Optional[dict] = Field(
        default_factory=dict,
        description="Additional generation parameters (top_p, top_k, min_p, etc) for the LLM, if needed",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the planner query writing request."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "topic": self.topic,
                "report_structure": self.report_structure,
                "number_of_queries": self.number_of_queries,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
                "extra_gen_kwargs": self.extra_gen_kwargs,
            }
        )
        return stats


class PlannerQueryWritingResponse(TaskResponse):
    """Response model for planner query writing LLM task."""

    task_name: str = "planner_query_writing_llm_task"

    queries: List[str] = Field(description="The generated search queries")
    success: bool = Field(description="Whether the LLM call was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def get_statistics(self) -> dict:
        """Get statistics for the planner query writing response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "queries_count": len(self.queries),
                "success": self.success,
                "error": self.error,
                "queries": self.queries,
            }
        )
        return stats


class QueryGenerationRequest(TaskRequest):
    """Request model for query generation LLM task."""

    task_name: str = "query_generation_llm_task"

    topic: str = Field(description="The main topic")
    section_topic: str = Field(description="The specific section topic")
    number_of_queries: int = Field(
        default=2, description="Number of queries to generate"
    )
    model: Optional[str] = Field(default=None, description="Language model to use")
    max_tokens: Optional[int] = Field(
        default=4096, description="Maximum tokens for generation"
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Temperature for generation"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for llm generation (if needed)"
    )
    extra_gen_kwargs: Optional[dict] = Field(
        default_factory=dict,
        description="Additional generation parameters (top_p, top_k, min_p, etc) for the LLM, if needed",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the query generation request."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "topic": self.topic,
                "section_topic": self.section_topic,
                "number_of_queries": self.number_of_queries,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
                "extra_gen_kwargs": self.extra_gen_kwargs,
            }
        )
        return stats


class QueryGenerationResponse(TaskResponse):
    """Response model for query generation LLM task."""

    task_name: str = "query_generation_llm_task"

    queries: List[str] = Field(description="The generated search queries")
    success: bool = Field(description="Whether the LLM call was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def get_statistics(self) -> dict:
        """Get statistics for the query generation response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "queries_count": len(self.queries),
                "success": self.success,
                "error": self.error,
                "queries": self.queries,
            }
        )
        return stats


class SectionWritingRequest(TaskRequest):
    """Request model for section writing LLM task."""

    task_name: str = "section_writing_llm_task"

    topic: str = Field(description="The main report topic")
    section_name: str = Field(description="Name of the section to write")
    section_topic: str = Field(description="Topic/description of the section")
    section_content: str = Field(default="", description="Existing section content")
    context: str = Field(description="Source material from web search")
    model: Optional[str] = Field(default=None, description="Language model to use")
    max_tokens: Optional[int] = Field(
        default=4096, description="Maximum tokens for generation"
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Temperature for generation"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for llm generation (if needed)"
    )
    extra_gen_kwargs: Optional[dict] = Field(
        default_factory=dict,
        description="Additional generation parameters (top_p, top_k, min_p, etc) for the LLM, if needed",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the section writing request."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "topic": self.topic,
                "section_name": self.section_name,
                "section_topic": self.section_topic,
                "section_content_length": len(self.section_content),
                "context_length": len(self.context),
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
                "extra_gen_kwargs": self.extra_gen_kwargs,
            }
        )
        return stats


class SectionWritingResponse(TaskResponse):
    """Response model for section writing LLM task."""

    task_name: str = "section_writing_llm_task"

    content: str = Field(description="The written section content")
    success: bool = Field(description="Whether the LLM call was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def get_statistics(self) -> dict:
        """Get statistics for the section writing response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "content_length": len(self.content),
                "success": self.success,
                "error": self.error,
            }
        )
        return stats


class SectionGradingRequest(TaskRequest):
    """Request model for section grading LLM task."""

    task_name: str = "section_grading_llm_task"

    topic: str = Field(description="The main report topic")
    section_topic: str = Field(description="The section topic")
    section_content: str = Field(description="The section content to grade")
    number_of_follow_up_queries: int = Field(
        default=2, description="Number of follow-up queries if needed"
    )
    model: Optional[str] = Field(default=None, description="Language model to use")
    max_tokens: Optional[int] = Field(
        default=4096, description="Maximum tokens for generation"
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Temperature for generation"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for llm generation (if needed)"
    )
    extra_gen_kwargs: Optional[dict] = Field(
        default_factory=dict,
        description="Additional generation parameters (top_p, top_k, min_p, etc) for the LLM, if needed",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the section grading request."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "topic": self.topic,
                "section_topic": self.section_topic,
                "section_content_length": len(self.section_content),
                "number_of_follow_up_queries": self.number_of_follow_up_queries,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
                "extra_gen_kwargs": self.extra_gen_kwargs,
            }
        )
        return stats


class SectionGradingResponse(TaskResponse):
    """Response model for section grading LLM task."""

    task_name: str = "section_grading_llm_task"

    grade: str = Field(description="Grade: 'pass' or 'fail'")
    follow_up_queries: List[str] = Field(
        default_factory=list, description="Follow-up queries if grade is 'fail'"
    )
    success: bool = Field(description="Whether the LLM call was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def get_statistics(self) -> dict:
        """Get statistics for the section grading response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "grade": self.grade,
                "follow_up_queries_count": len(self.follow_up_queries),
                "success": self.success,
                "error": self.error,
                "follow_up_queries": self.follow_up_queries,
            }
        )
        return stats


class FinalSectionWritingRequest(TaskRequest):
    """Request model for final section writing LLM task."""

    task_name: str = "final_section_writing_llm_task"

    topic: str = Field(description="The main report topic")
    section_name: str = Field(description="Name of the final section to write")
    section_topic: str = Field(description="Topic/description of the final section")
    completed_report_sections: str = Field(
        description="Content of all completed report sections"
    )
    model: Optional[str] = Field(default=None, description="Language model to use")
    max_tokens: Optional[int] = Field(
        default=4096, description="Maximum tokens for generation"
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Temperature for generation"
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for llm generation (if needed)"
    )
    extra_gen_kwargs: Optional[dict] = Field(
        default_factory=dict,
        description="Additional generation parameters (top_p, top_k, min_p, etc) for the LLM, if needed",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the final section writing request."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "topic": self.topic,
                "section_name": self.section_name,
                "section_topic": self.section_topic,
                "completed_report_sections_length": len(self.completed_report_sections),
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
                "extra_gen_kwargs": self.extra_gen_kwargs,
            }
        )
        return stats


class FinalSectionWritingResponse(TaskResponse):
    """Response model for final section writing LLM task."""

    task_name: str = "final_section_writing_llm_task"

    content: str = Field(description="The written final section content")
    success: bool = Field(description="Whether the LLM call was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def get_statistics(self) -> dict:
        """Get statistics for the final section writing response."""
        stats = super().get_statistics()
        stats.update(
            {
                "task_name": self.task_name,
                "content_length": len(self.content),
                "success": self.success,
                "error": self.error,
            }
        )
        return stats


class SummarizeRetrievedDocumentRequest(TaskRequest):
    """Request schema for summarizing a retrieved document."""

    task_name: str = "summarize_retrieved_document_with_llm"
    """Name of the task, used for identification."""

    document: RetrievedDocument = Field(
        ...,
        description="The retrieved document to summarize.",
    )
    model: str = Field(..., description="Model to use for summarization.")
    max_tokens: Optional[int] = Field(
        4096, description="Maximum number of tokens for the summary."
    )
    temperature: Optional[float] = Field(
        0.0, description="Temperature for the summarization model."
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for llm generation (if needed)"
    )
    extra_gen_kwargs: Optional[dict] = Field(
        default_factory=dict,
        description="Additional generation parameters (top_p, top_k, min_p, etc) for the LLM, if needed",
    )
    max_char_to_include: Optional[int] = Field(
        30_000,
        description="Maximum number of characters to include from the document for summarization.",
    )

    def get_statistics(self) -> dict:
        """Get statistics for the document summarization request."""
        stats = super().get_statistics()
        stats.update(
            {
                "document_id": (
                    self.document.id
                    if isinstance(self.document, RetrievedDocument)
                    else None
                ),
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
                "extra_gen_kwargs": self.extra_gen_kwargs,
                "max_char_to_include": self.max_char_to_include,
            }
        )
        return stats


class SummarizeRetrievedDocumentResponse(TaskResponse):
    """Response schema for summarizing a retrieved document."""

    task_name: str = "summarize_retrieved_document_with_llm"
    """Name of the task, used for identification."""

    summary: str = Field(
        ...,
        description="Summary of the retrieved document.",
    )
    key_excerpts: List[str] = Field(
        default_factory=list,
        description="Key excerpts from the document that are relevant to the summary.",
    )

    success: bool = Field(
        default=False, description="Whether the summarization was successful."
    )
    error: Optional[str] = Field(
        default=None, description="Error message if summarization failed."
    )

    def get_statistics(self) -> dict:
        """Get statistics for the document summarization response."""
        stats = super().get_statistics()
        stats.update(
            {
                "summary_length": len(self.summary),
                "key_excerpts_count": len(self.key_excerpts),
                "success": self.success,
                "error": self.error,
            }
        )
        return stats


@agsbench_deep_research_fsm.fast_service
def plan_report_with_llm(
    request: ReportPlanningRequest, context: RequestContext = None
) -> ReportPlanningResponse:
    """Plan a report using LLM.

    Args:
        request: The report planning request
        context: Request context

    Returns:
        Report planning response with sections
    """
    try:
        from ..prompts import REPORT_PLANNER_INSTRUCTIONS

        logger.debug(
            "Starting report planning with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = REPORT_PLANNER_INSTRUCTIONS.format(
            topic=request.topic,
            report_organization=request.report_structure,
            context=request.context,
            feedback=request.feedback,
        )
        user_prompt = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections.
                        Each section must have: name, description, research, and content fields."""

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=Sections.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = llm_service.call(llm_request)

        if not llm_response.success:
            return ReportPlanningResponse(
                request_id=request.request_id,
                sections=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            sections_data = llm_response.structured_output
            sections_obj = Sections.model_validate(sections_data)
            sections = sections_obj.sections
        except Exception as e:
            logger.exception(f"Failed to parse sections from LLM response: {e}")
            return ReportPlanningResponse(
                request_id=request.request_id,
                sections=[],
                success=False,
                error=f"Failed to parse sections: {e}",
            )
        logger.debug(f"Planned {len(sections)} sections for the report")
        return ReportPlanningResponse(
            request_id=request.request_id, sections=sections, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in plan_report_with_llm: {e}")
        return ReportPlanningResponse(
            request_id=request.request_id, sections=[], success=False, error=str(e)
        )


@agsbench_deep_research_fsm.fast_service
def generate_queries_for_planner_with_llm(
    request: PlannerQueryWritingRequest, context: RequestContext = None
) -> PlannerQueryWritingResponse:
    """Generate search queries for planner using LLM.

    Args:
        request: The query generation request
        context: Request context

    Returns:
        Query generation response with queries
    """
    try:
        from ..prompts import REPORT_PLANNER_QUERY_WRITER_INSTRUCTIONS
        from ..utils import get_today_str

        logger.debug(
            "Starting query generation for planner with request: %s",
            request.get_statistics(),
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = REPORT_PLANNER_QUERY_WRITER_INSTRUCTIONS.format(
            topic=request.topic,
            report_organization=request.report_structure,
            number_of_queries=request.number_of_queries,
            today=get_today_str(),
        )
        user_prompt = "Generate search queries that will help with planning the sections of the report."

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=Queries.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = llm_service.call(llm_request)

        if not llm_response.success:
            return PlannerQueryWritingResponse(
                request_id=request.request_id,
                queries=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            queries_data = llm_response.structured_output
            queries_obj = Queries.model_validate(queries_data)
            queries = [q.search_query for q in queries_obj.queries]
        except Exception as e:
            logger.exception(f"Failed to parse queries from LLM response: {e}")
            return PlannerQueryWritingResponse(
                request_id=request.request_id,
                queries=[],
                success=False,
                error=f"Failed to parse queries: {e}",
            )
        logger.debug(f"Generated {len(queries)} queries for the planner")
        return PlannerQueryWritingResponse(
            request_id=request.request_id, queries=queries, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in generate_queries_for_planner_llm: {e}")
        return PlannerQueryWritingResponse(
            request_id=request.request_id, queries=[], success=False, error=str(e)
        )


@agsbench_deep_research_fsm.fast_service
def generate_queries_with_llm(
    request: QueryGenerationRequest, context: RequestContext = None
) -> QueryGenerationResponse:
    """Generate search queries using LLM.

    Args:
        request: The query generation request
        context: Request context

    Returns:
        Query generation response with queries
    """
    try:
        from ..prompts import QUERY_WRITER_INSTRUCTIONS
        from ..utils import get_today_str

        logger.debug(
            "Starting query generation with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = QUERY_WRITER_INSTRUCTIONS.format(
            topic=request.topic,
            section_topic=request.section_topic,
            number_of_queries=request.number_of_queries,
            today=get_today_str(),
        )
        user_prompt = "Generate search queries on the provided topic."

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=Queries.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = llm_service.serve(llm_request)

        if not llm_response.success:
            return QueryGenerationResponse(
                request_id=request.request_id,
                queries=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            queries_data = llm_response.structured_output
            queries_obj = Queries.model_validate(queries_data)
            queries = [q.search_query for q in queries_obj.queries]
        except Exception as e:
            logger.exception(f"Failed to parse queries from LLM response: {e}")
            return QueryGenerationResponse(
                request_id=request.request_id,
                queries=[],
                success=False,
                error=f"Failed to parse queries: {e}",
            )
        logger.debug(f"Generated {len(queries)} queries for the section")
        return QueryGenerationResponse(
            request_id=request.request_id, queries=queries, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in generate_queries_with_llm: {e}")
        return QueryGenerationResponse(
            request_id=request.request_id, queries=[], success=False, error=str(e)
        )


@agsbench_deep_research_fsm.fast_service
def write_section_with_llm(
    request: SectionWritingRequest, context: RequestContext = None
) -> SectionWritingResponse:
    """Write a section using LLM.

    Args:
        request: The section writing request
        context: Request context

    Returns:
        Section writing response with content
    """
    try:
        from ..prompts import SECTION_WRITER_INSTRUCTIONS, SECTION_WRITER_INPUTS

        logger.debug(
            "Starting section writing with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = SECTION_WRITER_INSTRUCTIONS
        user_prompt = SECTION_WRITER_INPUTS.format(
            topic=request.topic,
            section_name=request.section_name,
            section_topic=request.section_topic,
            section_content=request.section_content,
            context=request.context,
        )

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
        )

        # Call LLM service
        llm_response: LLMServiceResponse = llm_service.call(llm_request)

        if not llm_response.success:
            return SectionWritingResponse(
                request_id=request.request_id,
                content="",
                success=False,
                error=llm_response.error,
            )
        logger.debug(
            "Section writing completed successfully with %d characters.",
            len(llm_response.text),
        )
        return SectionWritingResponse(
            request_id=request.request_id, content=llm_response.text, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in write_section_with_llm: {e}")
        return SectionWritingResponse(
            request_id=request.request_id, content="", success=False, error=str(e)
        )


@agsbench_deep_research_fsm.fast_service
def grade_section_with_llm(
    request: SectionGradingRequest, context: RequestContext = None
) -> SectionGradingResponse:
    """Grade a section using LLM.

    Args:
        request: The section grading request
        context: Request context

    Returns:
        Section grading response with grade and follow-up queries
    """
    try:
        from ..prompts import SECTION_GRADER_INSTRUCTIONS

        logger.debug(
            "Starting section grading with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = SECTION_GRADER_INSTRUCTIONS.format(
            topic=request.topic,
            section_topic=request.section_topic,
            section=request.section_content,
            number_of_follow_up_queries=request.number_of_follow_up_queries,
        )
        user_prompt = (
            "Grade the report and consider follow-up questions for missing information. "
            "If the grade is 'pass', return empty strings for all follow-up queries. "
            "If the grade is 'fail', provide specific search queries to gather missing information."
        )

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=Feedback.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = llm_service.call(llm_request)

        if not llm_response.success:
            return SectionGradingResponse(
                request_id=request.request_id,
                grade="fail",
                follow_up_queries=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            feedback_data = llm_response.structured_output
            feedback_obj = Feedback.model_validate(feedback_data)
            follow_up_queries = [q.search_query for q in feedback_obj.follow_up_queries]
        except Exception as e:
            logger.exception(f"Failed to parse feedback from LLM response: {e}")
            return SectionGradingResponse(
                request_id=request.request_id,
                grade="fail",
                follow_up_queries=[],
                success=False,
                error=f"Failed to parse feedback: {e}",
            )
        logger.debug(
            "Graded section with grade: %s and %d follow-up queries.",
            feedback_obj.grade,
            len(follow_up_queries),
        )
        return SectionGradingResponse(
            request_id=request.request_id,
            grade=feedback_obj.grade,
            follow_up_queries=follow_up_queries,
            success=True, usage=llm_response.usage,
        )

    except Exception as e:
        logger.exception(f"Error in grade_section_with_llm: {e}")
        return SectionGradingResponse(
            request_id=request.request_id,
            grade="fail",
            follow_up_queries=[],
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.fast_service
def write_final_section_with_llm(
    request: FinalSectionWritingRequest, context: RequestContext = None
) -> FinalSectionWritingResponse:
    """Write a final section using LLM.

    Args:
        request: The section writing request
        context: Request context

    Returns:
        Section writing response with content
    """
    try:
        from ..prompts import FINAL_SECTION_WRITER_INSTRUCTIONS

        logger.debug(
            "Starting final section writing with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = FINAL_SECTION_WRITER_INSTRUCTIONS.format(
            topic=request.topic,
            section_name=request.section_name,
            section_topic=request.section_topic,
            context=request.completed_report_sections,
        )
        user_prompt = "Generate a report section based on the provided sources."

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
        )

        # Call LLM service
        llm_response: LLMServiceResponse = llm_service.call(llm_request)

        if not llm_response.success:
            return SectionWritingResponse(
                request_id=request.request_id,
                content="",
                success=False,
                error=llm_response.error,
            )
        logger.debug(
            "Final section writing completed successfully with %d characters.",
            len(llm_response.text),
        )
        return SectionWritingResponse(
            request_id=request.request_id, content=llm_response.text, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in write_final_section_with_llm: {e}")
        return SectionWritingResponse(
            request_id=request.request_id, content="", success=False, error=str(e)
        )


@agsbench_deep_research_fsm.fast_service
def summarize_retrieved_document_with_llm(
    request: SummarizeRetrievedDocumentRequest, context: RequestContext = None
) -> SummarizeRetrievedDocumentResponse:
    """Summarize a retrieved document (raw_content) using the specified model.
    This function takes a retrieved document and generates a summary using the specified model.

    Args:
        request (SummarizeRetrievedDocumentRequest): The request object containing the document,
            model, and configuration parameters.
        context (RequestContext, optional): The request context, if available.
            Defaults to None.

    Returns:
        SummarizeRetrievedDocumentResponse: The response object containing the summary of the document.
    """
    try:
        from ..prompts import SUMMARIZATION_PROMPT

        logger.debug(
            "Starting summarization for document with raw content length: %d",
            (
                len(request.document.raw_content)
                if request.document and request.document.raw_content
                else 0
            ),
        )
        if not request.document or not request.document.raw_content:
            return SummarizeRetrievedDocumentResponse(
                request_id=request.request_id,
                summary="",
                key_excerpts=[],
                success=False,
                error="No document content provided",
            )

        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = SUMMARIZATION_PROMPT.format(
            webpage_content=request.document.raw_content
        )
        user_prompt = "Please summarize the article"

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=SearchResultSummary.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = llm_service.call(llm_request)

        if not llm_response.success:
            return SummarizeRetrievedDocumentResponse(
                request_id=request.request_id,
                summary="",
                key_excerpts=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            summary_data = llm_response.structured_output
            summary_obj = SearchResultSummary.model_validate(summary_data)
            summary = summary_obj.summary
            key_excerpts = summary_obj.key_excerpts

            logger.debug(
                "Summarization completed successfully with summary length: %d",
                len(summary),
            )
            return SummarizeRetrievedDocumentResponse(
                request_id=request.request_id,
                summary=summary,
                key_excerpts=key_excerpts,
                success=True, usage=llm_response.usage,
            )

        except Exception as e:
            logger.exception(f"Error in parsing LLM response: {e}")
            return SummarizeRetrievedDocumentResponse(
                request_id=request.request_id,
                summary="",
                key_excerpts=[],
                success=False,
                error=str(e),
            )

    except Exception as e:
        logger.exception(f"Error in summarize_retrieved_document_with_llm: {e}")
        return SummarizeRetrievedDocumentResponse(
            request_id=request.request_id,
            summary="",
            key_excerpts=[],
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.async_fast_service
async def plan_report_with_llm_async(
    request: ReportPlanningRequest, context: RequestContext = None
) -> ReportPlanningResponse:
    """Asynchronous version of plan_report_with_llm."""
    try:
        from ..prompts import REPORT_PLANNER_INSTRUCTIONS

        logger.debug(
            "Starting async report planning with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = REPORT_PLANNER_INSTRUCTIONS.format(
            topic=request.topic,
            report_organization=request.report_structure,
            context=request.context,
            feedback=request.feedback,
        )
        user_prompt = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections.
                        Each section must have: name, description, research, and content fields."""

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=Sections.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = await llm_service.call_async(llm_request)

        if not llm_response.success:
            return ReportPlanningResponse(
                request_id=request.request_id,
                sections=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            sections_data = llm_response.structured_output
            sections_obj = Sections.model_validate(sections_data)
            sections = sections_obj.sections
        except Exception as e:
            logger.exception(f"Failed to parse sections from LLM response: {e}")
            return ReportPlanningResponse(
                request_id=request.request_id,
                sections=[],
                success=False,
                error=f"Failed to parse sections: {e}",
            )
        logger.debug(f"Planned {len(sections)} sections for the report")
        return ReportPlanningResponse(
            request_id=request.request_id, sections=sections, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in plan_report_with_llm_async: {e}")
        return ReportPlanningResponse(
            request_id=request.request_id, sections=[], success=False, error=str(e)
        )


@agsbench_deep_research_fsm.async_fast_service
async def generate_queries_for_planner_with_llm_async(
    request: PlannerQueryWritingRequest, context: RequestContext = None
) -> PlannerQueryWritingResponse:
    """Asynchronous version of generate_queries_for_planner_with_llm."""
    try:
        from ..prompts import REPORT_PLANNER_QUERY_WRITER_INSTRUCTIONS
        from ..utils import get_today_str

        logger.debug(
            "Starting async query generation for planner with request: %s",
            request.get_statistics(),
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = REPORT_PLANNER_QUERY_WRITER_INSTRUCTIONS.format(
            topic=request.topic,
            report_organization=request.report_structure,
            number_of_queries=request.number_of_queries,
            today=get_today_str(),
        )
        user_prompt = "Generate search queries that will help with planning the sections of the report."

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=Queries.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = await llm_service.call_async(llm_request)

        if not llm_response.success:
            return PlannerQueryWritingResponse(
                request_id=request.request_id,
                queries=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            queries_data = llm_response.structured_output
            queries_obj = Queries.model_validate(queries_data)
            queries = [q.search_query for q in queries_obj.queries]
        except Exception as e:
            logger.exception(f"Failed to parse queries from LLM response: {e}")
            return PlannerQueryWritingResponse(
                request_id=request.request_id,
                queries=[],
                success=False,
                error=f"Failed to parse queries: {e}",
            )
        logger.debug(f"Generated {len(queries)} queries for the planner")
        return PlannerQueryWritingResponse(
            request_id=request.request_id, queries=queries, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in generate_queries_for_planner_llm_async: {e}")
        return PlannerQueryWritingResponse(
            request_id=request.request_id, queries=[], success=False, error=str(e)
        )


@agsbench_deep_research_fsm.async_fast_service
async def generate_queries_with_llm_async(
    request: QueryGenerationRequest, context: RequestContext = None
) -> QueryGenerationResponse:
    """Asynchronous version of generate_queries_with_llm."""
    try:
        from ..prompts import QUERY_WRITER_INSTRUCTIONS
        from ..utils import get_today_str

        logger.debug(
            "Starting async query generation with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = QUERY_WRITER_INSTRUCTIONS.format(
            topic=request.topic,
            section_topic=request.section_topic,
            number_of_queries=request.number_of_queries,
            today=get_today_str(),
        )
        user_prompt = "Generate search queries on the provided topic."

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=Queries.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = await llm_service.call_async(llm_request)

        if not llm_response.success:
            return QueryGenerationResponse(
                request_id=request.request_id,
                queries=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            queries_data = llm_response.structured_output
            queries_obj = Queries.model_validate(queries_data)
            queries = [q.search_query for q in queries_obj.queries]
        except Exception as e:
            logger.exception(f"Failed to parse queries from LLM response: {e}")
            return QueryGenerationResponse(
                request_id=request.request_id,
                queries=[],
                success=False,
                error=f"Failed to parse queries: {e}",
            )
        logger.debug(f"Generated {len(queries)} queries for the section")
        return QueryGenerationResponse(
            request_id=request.request_id, queries=queries, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in generate_queries_with_llm_async: {e}")
        return QueryGenerationResponse(
            request_id=request.request_id, queries=[], success=False, error=str(e)
        )


@agsbench_deep_research_fsm.async_fast_service
async def write_section_with_llm_async(
    request: SectionWritingRequest, context: RequestContext = None
) -> SectionWritingResponse:
    """Asynchronous version of write_section_with_llm."""
    try:
        from ..prompts import SECTION_WRITER_INSTRUCTIONS, SECTION_WRITER_INPUTS

        logger.debug(
            "Starting async section writing with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = SECTION_WRITER_INSTRUCTIONS
        user_prompt = SECTION_WRITER_INPUTS.format(
            topic=request.topic,
            section_name=request.section_name,
            section_topic=request.section_topic,
            section_content=request.section_content,
            context=request.context,
        )

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
        )

        # Call LLM service
        llm_response: LLMServiceResponse = await llm_service.call_async(llm_request)

        if not llm_response.success:
            return SectionWritingResponse(
                request_id=request.request_id,
                content="",
                success=False,
                error=llm_response.error,
            )
        logger.debug(
            "Async section writing completed successfully with %d characters.",
            len(llm_response.text),
        )
        return SectionWritingResponse(
            request_id=request.request_id, content=llm_response.text, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in write_section_with_llm_async: {e}")
        return SectionWritingResponse(
            request_id=request.request_id, content="", success=False, error=str(e)
        )


@agsbench_deep_research_fsm.async_fast_service
async def grade_section_with_llm_async(
    request: SectionGradingRequest, context: RequestContext = None
) -> SectionGradingResponse:
    """Asynchronous version of grade_section_with_llm."""
    try:
        from ..prompts import SECTION_GRADER_INSTRUCTIONS

        logger.debug(
            "Starting async section grading with request: %s", request.get_statistics()
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = SECTION_GRADER_INSTRUCTIONS.format(
            topic=request.topic,
            section_topic=request.section_topic,
            section=request.section_content,
            number_of_follow_up_queries=request.number_of_follow_up_queries,
        )
        user_prompt = (
            "Grade the report and consider follow-up questions for missing information. "
            "If the grade is 'pass', return empty strings for all follow-up queries. "
            "If the grade is 'fail', provide specific search queries to gather missing information."
        )

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=Feedback.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = await llm_service.call_async(llm_request)

        if not llm_response.success:
            return SectionGradingResponse(
                request_id=request.request_id,
                grade="fail",
                follow_up_queries=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            feedback_data = llm_response.structured_output
            feedback_obj = Feedback.model_validate(feedback_data)
            follow_up_queries = [q.search_query for q in feedback_obj.follow_up_queries]
        except Exception as e:
            logger.exception(f"Failed to parse feedback from LLM response: {e}")
            return SectionGradingResponse(
                request_id=request.request_id,
                grade="fail",
                follow_up_queries=[],
                success=False,
                error=f"Failed to parse feedback: {e}",
            )
        logger.debug(
            "Graded section with grade: %s and %d follow-up queries.",
            feedback_obj.grade,
            len(follow_up_queries),
        )
        return SectionGradingResponse(
            request_id=request.request_id,
            grade=feedback_obj.grade,
            follow_up_queries=follow_up_queries,
            success=True, usage=llm_response.usage,
        )
    except Exception as e:
        logger.exception(f"Error in grade_section_with_llm_async: {e}")
        return SectionGradingResponse(
            request_id=request.request_id,
            grade="fail",
            follow_up_queries=[],
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.async_fast_service
async def write_final_section_with_llm_async(
    request: FinalSectionWritingRequest, context: RequestContext = None
) -> FinalSectionWritingResponse:
    """Asynchronous version of write_final_section_with_llm."""
    try:
        from ..prompts import FINAL_SECTION_WRITER_INSTRUCTIONS

        logger.debug(
            "Starting async final section writing with request: %s",
            request.get_statistics(),
        )
        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = FINAL_SECTION_WRITER_INSTRUCTIONS.format(
            topic=request.topic,
            section_name=request.section_name,
            section_topic=request.section_topic,
            context=request.completed_report_sections,
        )
        user_prompt = "Generate a report section based on the provided sources."

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
        )

        # Call LLM service
        llm_response: LLMServiceResponse = await llm_service.call_async(llm_request)

        if not llm_response.success:
            return SectionWritingResponse(
                request_id=request.request_id,
                content="",
                success=False,
                error=llm_response.error,
            )
        logger.debug(
            "Async final section writing completed successfully with %d characters.",
            len(llm_response.text),
        )
        return SectionWritingResponse(
            request_id=request.request_id, content=llm_response.text, success=True, usage=llm_response.usage
        )

    except Exception as e:
        logger.exception(f"Error in write_final_section_with_llm_async: {e}")
        return SectionWritingResponse(
            request_id=request.request_id, content="", success=False, error=str(e)
        )


@agsbench_deep_research_fsm.async_fast_service
async def summarize_retrieved_document_with_llm_async(
    request: SummarizeRetrievedDocumentRequest, context: RequestContext = None
) -> SummarizeRetrievedDocumentResponse:
    """Asynchronous version of summarize_retrieved_document_with_llm."""
    try:
        from ..prompts import SUMMARIZATION_PROMPT

        logger.debug(
            "Starting async summarization for document with request: %s",
            request.get_statistics(),
        )
        if not request.document or not request.document.raw_content:
            return SummarizeRetrievedDocumentResponse(
                request_id=request.request_id,
                summary="",
                key_excerpts=[],
                success=False,
                error="No document content provided",
            )

        # Get LLM service
        service_manager = get_service_manager()
        llm_service: LLMService = service_manager.get_service("llm")

        # Format the prompt
        system_prompt = SUMMARIZATION_PROMPT.format(
            webpage_content=request.document.raw_content
        )
        user_prompt = "Please summarize the article"

        # Create LLM request
        llm_request = LLMServiceRequest(
            request_id=context.call_id if context else request.request_id,
            service_name="llm",
            operation_name="chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            kwargs=request.extra_gen_kwargs,
            response_format=SearchResultSummary.model_json_schema(),
        )

        # Call LLM service
        llm_response: LLMServiceResponse = await llm_service.call_async(llm_request)

        if not llm_response.success:
            return SummarizeRetrievedDocumentResponse(
                request_id=request.request_id,
                summary="",
                key_excerpts=[],
                success=False,
                error=llm_response.error,
            )

        # Parse the structured output
        try:
            summary_data = llm_response.structured_output
            summary_obj = SearchResultSummary.model_validate(summary_data)
            summary = summary_obj.summary
            key_excerpts = summary_obj.key_excerpts

            logger.debug(
                "Async summarization completed successfully with summary length: %d",
                len(summary),
            )
            return SummarizeRetrievedDocumentResponse(
                request_id=request.request_id,
                summary=summary,
                key_excerpts=key_excerpts,
                success=True, usage=llm_response.usage,
            )

        except Exception as e:
            logger.exception(f"Error in parsing LLM response: {e}")
            return SummarizeRetrievedDocumentResponse(
                request_id=request.request_id,
                summary="",
                key_excerpts=[],
                success=False,
                error=str(e),
            )

    except Exception as e:
        logger.exception(f"Error in summarize_retrieved_document_with_llm: {e}")
        return SummarizeRetrievedDocumentResponse(
            request_id=request.request_id,
            summary="",
            key_excerpts=[],
            success=False,
            error=str(e),
        )
