"""
Deep Research workflow implementation.

This module provides the complete deep research workflow that orchestrates
the research process from planning to final report generation.
"""

from typing import List, Dict, Any, Optional, Annotated, Literal
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import Field, model_validator, field_validator
from fastapi import Body

from agsbench_deep_research.core.schema.io import TaskRequest, TaskResponse
from agsbench_deep_research.core.utils import with_timeout_and_retry
from ..shared import agsbench_deep_research_fsm, RequestContext
from ..shared import DEFAULT_SEARCH_POST_PROCESSING_CONFIG
from ..state import (
    ReportState,
    Section,
)
from ..task.llm_task import (
    ReportPlanningRequest,
    PlannerQueryWritingRequest,
    QueryGenerationRequest,
    SectionWritingRequest,
    SectionGradingRequest,
    FinalSectionWritingRequest,
    PlannerQueryWritingResponse,
    FinalSectionWritingResponse,
    plan_report_with_llm,
    generate_queries_for_planner_with_llm,
    generate_queries_with_llm,
    write_section_with_llm,
    grade_section_with_llm,
    write_final_section_with_llm,
    plan_report_with_llm_async,
    generate_queries_for_planner_with_llm_async,
    generate_queries_with_llm_async,
    write_section_with_llm_async,
    grade_section_with_llm_async,
    write_final_section_with_llm_async,
)
from ..task.search_task import search_task, search_task_async
from ..task.search_task import SearchTaskRequest, SearchTaskResponse
from ..utils import (
    format_sections,
    get_drs_config_value,
)
from ..prompts import DEFAULT_REPORT_STRUCTURE

logger = logging.getLogger(__name__)


class DeepResearchWorkflowRequest(TaskRequest):
    """Request model for deep research workflow.

    Attributes:
        topic: The research topic/question
        report_structure: Description of desired report organization
        max_search_depth: Maximum iterations of search per section
        number_of_queries: Number of search queries to generate
        include_source_str: Whether to include source strings in output
        search_api: Search API to use (e.g., 'tavily', 'bing')
        search_api_config: Configuration for the search API
        max_tokens_per_source: Maximum number of tokens to return per source in the search results
        include_raw_content: Whether to include raw content in the search results
        deduplication_strategy: Strategy for deduplicating search results with the same URL
        writer_model: Model for content writing
        planner_model: Model for planning
        summarization_model: Model for summarization (if needed)
        writer_model_kwargs: Additional kwargs for the writer model
        planner_model_kwargs: Additional kwargs for the planner model
        summarization_model_kwargs: Additional kwargs for the summary model
        search_post_processing: Optional post-processing parameters for the search results
        parallelize_sections: Whether to parallelize section research and writing
        parallelize_sections_n_workers: Number of parallel workers to use for section processing
        max_tokens: Maximum tokens for generation
        temperature: Temperature for generation
        seed: Seed for random number generation (if needed)
        extra_gen_kwargs: Additional generation kwargs, e.g., {'top_p': 0.95, 'top_k': 20}
        max_structured_output_retries: Maximum retries for structured output generation
    This request model is used to configure the deep research workflow.
    It allows users to specify various parameters for the research process,
    including the topic, report structure, search parameters, and models to use.
    """

    task_name: str = "deep_research_workflow"

    topic: str = Field(..., description="The research topic/question")
    report_structure: str = Field(
        default=get_drs_config_value("report_structure", DEFAULT_REPORT_STRUCTURE),
        description="Description of desired report organization",
    )
    max_search_depth: int = Field(
        default=get_drs_config_value("max_search_depth", 2),
        description="Maximum iterations of search per section",
    )
    number_of_queries: int = Field(
        default=get_drs_config_value("number_of_queries", 2),
        description="Number of search queries to generate",
    )
    include_source_str: bool = Field(
        default=get_drs_config_value("include_source_str", False),
        description="Whether to include source strings in output",
    )
    search_api: Optional[str] = Field(
        default=get_drs_config_value("search_api", "dummy"),
        description="Search API to use (e.g., 'tavily', 'bing')",
    )
    search_api_config: Optional[Dict[str, Any]] = Field(
        default=get_drs_config_value("search_api_config", {}),
        description="Configuration for the search API. Example: {'max_results': 10, 'language': 'en'}",
    )
    max_tokens_per_source: Optional[int] = Field(
        default=get_drs_config_value("max_tokens_per_source", 5000),
        description="Maximum number of tokens to return per source in the search results",
    )
    include_raw_content: bool = Field(
        default=get_drs_config_value("include_raw_content", True),
        description="Whether to include raw content in the search results",
    )
    deduplication_strategy: Literal["keep_first", "keep_last"] = Field(
        default=get_drs_config_value("deduplication_strategy", "keep_first"),
        description="Strategy for deduplicating search results with the same URL. "
        "'keep_first' retains the first occurrence, while 'keep_last' retains the last occurrence.",
    )
    writer_model: Optional[str] = Field(
        default=get_drs_config_value("writer_model"),
        description="Model for content writing",
    )
    planner_model: Optional[str] = Field(
        default=get_drs_config_value("planner_model"),
        description="Model for planning",
    )
    summarization_model: Optional[str] = Field(
        default=get_drs_config_value("summarization_model"),
        description="Model for summarization (if needed)",
    )
    writer_model_kwargs: Optional[Dict[str, Any]] = Field(
        default=get_drs_config_value("writer_model_kwargs", {}),
        description="Additional kwargs for the writer model",
    )
    planner_model_kwargs: Optional[Dict[str, Any]] = Field(
        default=get_drs_config_value("planner_model_kwargs", {}),
        description="Additional kwargs for the planner model",
    )
    summarization_model_kwargs: Optional[Dict[str, Any]] = Field(
        default=get_drs_config_value("summarization_model_kwargs", {}),
        description="Additional kwargs for the summary model",
    )
    search_post_processing: Optional[Dict[str, Any]] = Field(
        default=get_drs_config_value(
            "search_post_processing", DEFAULT_SEARCH_POST_PROCESSING_CONFIG
        ),
        description="Optional post-processing parameters for the search results. "
        "Example1: {'operation': 'summarize', ...}"
        "Example2: {'operation': 'split_and_rerank', ...}",
    )
    parallelize_sections: bool = Field(
        default=get_drs_config_value("parallelize_sections", True),
        description="Whether to parallelize section research and writing",
    )
    parallelize_sections_n_workers: int = Field(
        default=get_drs_config_value("parallelize_sections_n_workers", 32),
        description="Number of parallel workers to use for section processing",
    )
    max_tokens: Optional[int] = Field(
        default=get_drs_config_value("max_tokens", 4096),
        description="Maximum tokens for generation",
    )  # deprecated, use writer_model_kwargs and planner_model_kwargs instead
    temperature: Optional[float] = Field(
        default=get_drs_config_value("temperature", 0.0),
        description="Temperature for generation",
    )  # deprecated
    seed: Optional[int] = Field(
        default=get_drs_config_value("seed", None),
        description="Seed for random number generation (if needed)",
    )  # deprecated, not used in the workflow
    extra_gen_kwargs: Optional[Dict[str, Any]] = Field(
        default=get_drs_config_value("extra_gen_kwargs", {}),
        description="Additional generation kwargs, e.g., {'top_p': 0.95, 'top_k': 20}",
    )  # deprecated, not used in the workflow
    max_structured_output_retries: int = Field(
        default=get_drs_config_value("max_structured_output_retries", 3),
        description="Maximum retries for structured output generation",
    )  # unused know since we haven't implemented customized retry logic yet, OpenAI's retry logic is used by default

    @model_validator(mode="before")
    @classmethod
    def check_topic_present(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "topic" not in data or not data["topic"]:
                raise ValueError("Topic must be provided")
            if not isinstance(data["topic"], str) or data["topic"].strip() == "":
                raise ValueError("Topic cannot be empty")
        return data

    @model_validator(mode="after")
    def init_search_post_processing(self) -> "DeepResearchWorkflowRequest":
        """Initialize post-processing parameters for search if not provided."""
        if not self.search_post_processing:
            self.search_post_processing = DEFAULT_SEARCH_POST_PROCESSING_CONFIG
        if "operation" in self.search_post_processing:
            # init summarization_model and summarization_model_kwargs if not provided
            if not self.search_post_processing.get("summarization_model", None):
                self.search_post_processing["summarization_model"] = (
                    self.summarization_model
                    or get_drs_config_value("summarization_model", None)
                )
            if not self.search_post_processing.get("summarization_model_kwargs", None):
                self.search_post_processing["summarization_model_kwargs"] = (
                    self.summarization_model_kwargs
                    or get_drs_config_value("summarization_model_kwargs", {})
                )
        return self

    @field_validator("report_structure", mode="before")
    @classmethod
    def validate_report_structure(cls, value: str) -> str:
        """Validate the report structure format."""
        if value == "default":
            # If 'default' is specified, use the default structure
            value = get_drs_config_value("report_structure", DEFAULT_REPORT_STRUCTURE)
        if isinstance(value, dict):
            # If it's a dict, convert to string representation
            value = str(value)
        # Ensure it's a non-empty string
        if not value or not isinstance(value, str):
            raise ValueError("Report structure must be a non-empty string")
        value = value.strip()
        return value


class DeepResearchWorkflowResponse(TaskResponse):
    """Response model for deep research workflow.

    Attributes:
        final_report: The complete research report
        sections: List of sections that were created
        source_str: Source information (if include_source_str is True)
    """

    task_name: str = "deep_research_workflow"

    final_report: str = Field(..., description="The complete research report")
    sections: List[Section] = Field(
        ..., description="List of sections that were created"
    )
    source_str: Optional[str] = Field(
        default=None,
        description="Source information (if include_source_str is True)",
    )
    success: bool = Field(
        default=True, description="Whether the workflow was successful"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the workflow failed"
    )


@agsbench_deep_research_fsm.fast_service
def generate_report_plan_step(
    state: ReportState,
    request: DeepResearchWorkflowRequest,
    context: RequestContext = None,
) -> Dict[str, Any]:
    """Generate the initial report plan with sections.

    This step:
    1. Generates search queries to gather context for planning
    2. Performs web searches using those queries
    3. Uses an LLM to generate a structured plan with sections

    Args:
        state: Current workflow state containing the report topic
        request: Original workflow request with configuration

    Returns:
        Dict containing the generated sections
    """
    logger.info(
        "Generating report plan for topic: %s with request: %s",
        state.topic,
        request,
    )
    topic = state.topic
    feedback_list = state.feedback_on_report_plan or []
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    # Generate planning queries using the LLM task
    query_req = PlannerQueryWritingRequest(
        topic=topic,
        report_structure=request.report_structure,
        number_of_queries=request.number_of_queries,
        model=request.writer_model,
        max_tokens=request.writer_model_kwargs.get("max_tokens", request.max_tokens),
        temperature=request.writer_model_kwargs.get("temperature", request.temperature),
        seed=request.writer_model_kwargs.get("seed", request.seed),
        extra_gen_kwargs=request.writer_model_kwargs.get(
            "extra_gen_kwargs", request.extra_gen_kwargs
        ),
    )

    queries_response = generate_queries_for_planner_with_llm(query_req, context=context)
    if not queries_response.success:
        raise RuntimeError(f"Query generation failed: {queries_response.error}")

    query_list = queries_response.queries

    search_api_config = request.search_api_config or {}

    # Use new async search task
    search_request = SearchTaskRequest(
        queries=query_list,
        search_api=request.search_api,
        search_api_config=search_api_config,
        max_tokens_per_source=request.max_tokens_per_source,
        include_raw_content=request.include_raw_content,
        deduplication_strategy=request.deduplication_strategy,
        search_post_processing=request.search_post_processing,
    )
    search_response = search_task(request=search_request, context=context)
    source_str = search_response.merged_content

    # Generate report plan using the LLM task
    plan_req = ReportPlanningRequest(
        topic=topic,
        report_structure=request.report_structure,
        context=source_str,
        feedback=feedback,
        model=request.planner_model,
        max_tokens=request.planner_model_kwargs.get("max_tokens", request.max_tokens),
        temperature=request.planner_model_kwargs.get(
            "temperature", request.temperature
        ),
        seed=request.planner_model_kwargs.get("seed", request.seed),
        extra_gen_kwargs=request.planner_model_kwargs.get(
            "extra_gen_kwargs", request.extra_gen_kwargs
        ),
    )

    plan_response = plan_report_with_llm(plan_req, context=context)
    if not plan_response.success:
        raise RuntimeError(f"Report planning failed: {plan_response.error}")

    logger.info("Generated report plan with sections: %s", plan_response.sections)
    return {
        "sections": plan_response.sections,
        "source_str": source_str if request.include_source_str else None,
    }


@agsbench_deep_research_fsm.fast_service
def build_section_with_research(
    section: Section,
    topic: Annotated[str, Body(..., description="The overall research topic")],
    request: DeepResearchWorkflowRequest,
    context: RequestContext = None,
) -> Section:
    """Build a section with iterative research and writing.

    Args:
        section: The section to research and write
        topic: The overall research topic
        request: Configuration for the workflow

    Returns:
        The completed section with content
    """
    logger.info("Building section: %s for topic: %s", section.name, topic)
    search_iterations = 0
    current_section = section

    # Generate queries for this section
    query_req = QueryGenerationRequest(
        topic=topic,
        section_topic=current_section.description,
        number_of_queries=request.number_of_queries,
        model=request.writer_model,
        max_tokens=request.writer_model_kwargs.get("max_tokens", request.max_tokens),
        temperature=request.writer_model_kwargs.get("temperature", request.temperature),
        seed=request.writer_model_kwargs.get("seed", request.seed),
        extra_gen_kwargs=request.writer_model_kwargs.get(
            "extra_gen_kwargs", request.extra_gen_kwargs
        ),
    )

    queries_response = generate_queries_with_llm(query_req, context=context)
    if not queries_response.success:
        logger.warning("Query generation failed: %s", queries_response.error)
        return current_section  # Return section without content

    # Extract queries from response
    query_list = queries_response.queries

    while search_iterations < request.max_search_depth:
        search_api_config = request.search_api_config or {}
        # params_to_pass = get_search_params(request.search_api, search_api_config)

        # Use new async search task
        search_request = SearchTaskRequest(
            queries=query_list,
            search_api=request.search_api,
            search_api_config=search_api_config,
            max_tokens_per_source=request.max_tokens_per_source,
            include_raw_content=request.include_raw_content,
            deduplication_strategy=request.deduplication_strategy,
            search_post_processing=request.search_post_processing,
        )
        search_response = search_task(request=search_request, context=context)
        source_str = search_response.merged_content

        # Write section content
        write_req = SectionWritingRequest(
            topic=topic,
            section_name=current_section.name,
            section_topic=current_section.description,
            context=source_str,
            section_content=current_section.content or "",
            model=request.writer_model,
            max_tokens=request.writer_model_kwargs.get(
                "max_tokens", request.max_tokens
            ),
            temperature=request.writer_model_kwargs.get(
                "temperature", request.temperature
            ),
            seed=request.writer_model_kwargs.get("seed", request.seed),
            extra_gen_kwargs=request.writer_model_kwargs.get(
                "extra_gen_kwargs", request.extra_gen_kwargs
            ),
        )

        write_response = write_section_with_llm(write_req, context=context)
        if not write_response.success:
            logger.warning("Section writing failed: %s", write_response.error)
            break

        current_section.content = write_response.content

        # Grade the section
        grade_req = SectionGradingRequest(
            topic=topic,
            section_topic=current_section.description,
            section_content=current_section.content,
            number_of_follow_up_queries=request.number_of_queries,
            model=request.planner_model,
            max_tokens=request.planner_model_kwargs.get(
                "max_tokens", request.max_tokens
            ),
            temperature=request.planner_model_kwargs.get(
                "temperature", request.temperature
            ),
            seed=request.planner_model_kwargs.get("seed", request.seed),
            extra_gen_kwargs=request.planner_model_kwargs.get(
                "extra_gen_kwargs", request.extra_gen_kwargs
            ),
        )

        grade_response = grade_section_with_llm(grade_req, context=context)
        if not grade_response.success:
            logger.warning("Section grading failed: %s", grade_response.error)
            break

        # If passing or max depth reached, we're done
        if (
            grade_response.grade == "pass"
            or search_iterations >= request.max_search_depth - 1
        ):
            if grade_response.grade == "pass":
                logger.info(
                    "Section '%s' completed with grade: %s",
                    current_section.name,
                    grade_response.grade,
                )
            else:
                logger.warning(
                    "Section '%s' failed with grade: %s, but max depth reached",
                    current_section.name,
                    grade_response.grade,
                )
            break

        search_iterations += 1
        query_list = grade_response.follow_up_queries
        if not query_list:
            logger.warning(
                "Grade failed, but No follow-up queries generated, stopping research for section: %s",
                current_section.name,
            )
            break

    logger.info(
        "Completed section '%s' after %d search iterations",
        current_section.name,
        search_iterations,
    )
    return current_section


@agsbench_deep_research_fsm.fast_service
def write_final_section(
    section: Section,
    topic: Annotated[str, Body(..., description="The overall research topic")],
    completed_sections_str: Annotated[
        str, Body(..., description="Context from completed research sections")
    ],
    request: DeepResearchWorkflowRequest,
    context: RequestContext = None,
) -> Section:
    """Write a section that doesn't require research.

    Args:
        section: The section to write
        topic: The overall research topic
        completed_sections_str: Context from completed research sections
        request: Configuration for the workflow

    Returns:
        The completed section with content
    """
    logger.info("Writing final section: %s for topic: %s", section.name, topic)
    write_req = FinalSectionWritingRequest(
        topic=topic,
        section_name=section.name,
        section_topic=section.description,
        completed_report_sections=completed_sections_str,
        model=request.writer_model,
        max_tokens=request.writer_model_kwargs.get("max_tokens", request.max_tokens),
        temperature=request.writer_model_kwargs.get("temperature", request.temperature),
        seed=request.writer_model_kwargs.get("seed", request.seed),
        extra_gen_kwargs=request.writer_model_kwargs.get(
            "extra_gen_kwargs", request.extra_gen_kwargs
        ),
    )

    write_response = write_final_section_with_llm(write_req, context=context)
    if not write_response.success:
        logger.warning("Final section writing failed: %s", write_response.error)
        section.content = f"Error writing section: {write_response.error}"
    else:
        section.content = write_response.content

    logger.info(
        "Completed final section '%s' with content: %s",
        section.name,
        section.content[:10] + "...",  # Log first 10 chars of content
    )
    return section


@agsbench_deep_research_fsm.fast_service
@with_timeout_and_retry(
    timeout=get_drs_config_value("timeout", 60 * 60),  # 1 hour default timeout
    max_retries=get_drs_config_value("max_retries", 0),  # default 0 retries
    timeout_method="thread",  # Use thread-based timeout, the default "signal" is problematic when using together with fastapi.
)
def deep_research_workflow(
    request: DeepResearchWorkflowRequest,
    context: RequestContext = None,
) -> DeepResearchWorkflowResponse:
    """Execute the deep research workflow.

    Args:
        request: The workflow request with topic and configuration

    Returns:
        The workflow response with the completed report
    """
    logger.info(
        "Starting deep research workflow for topic: %s with request %s",
        request.topic,
        request,
    )

    # Initialize state
    state = ReportState(
        topic=request.topic,
        sections=[],
        completed_sections=[],
        feedback_on_report_plan=[],
        report_sections_from_research="",
        final_report="",
        source_str="",
    )

    try:
        # Step 1: Generate report plan
        logger.info("Generating report plan...")
        plan_result = generate_report_plan_step(state, request, context=context)
        state.sections = plan_result["sections"]
        if request.include_source_str and plan_result.get("source_str"):
            state.source_str = plan_result["source_str"]

        # Step 2: Research and write sections that need research
        logger.info("Building sections with research...")
        research_sections = [s for s in state.sections if s.research]
        completed_research_sections = []

        if request.parallelize_sections:
            # Use ThreadPoolExecutor to parallelize section research
            with ThreadPoolExecutor(
                max_workers=request.parallelize_sections_n_workers
            ) as executor:
                completed_research_sections.extend(
                    list(
                        executor.map(
                            lambda sec: build_section_with_research(
                                sec, state.topic, request, context=context
                            ),
                            research_sections,
                        )
                    )
                )
        else:
            for section in research_sections:
                logger.info("Researching section: %s", section.name)
                completed_section = build_section_with_research(
                    section, state.topic, request, context=context
                )
                completed_research_sections.append(completed_section)

        # Step 3: Format completed research sections as context
        completed_sections_str = format_sections(completed_research_sections)
        state.report_sections_from_research = completed_sections_str
        state.completed_sections = completed_research_sections.copy()

        # Step 4: Write final sections that don't need research
        logger.info("Writing final sections...")
        final_sections = [s for s in state.sections if not s.research]

        if len(final_sections) > 0:
            if request.parallelize_sections:
                # Use ThreadPoolExecutor to parallelize final section writing
                with ThreadPoolExecutor(
                    max_workers=request.parallelize_sections_n_workers
                ) as executor:
                    state.completed_sections.extend(
                        list(
                            executor.map(
                                lambda sec: write_final_section(
                                    sec,
                                    state.topic,
                                    completed_sections_str,
                                    request,
                                    context=context,
                                ),
                                final_sections,
                            )
                        )
                    )
            else:
                for section in final_sections:
                    logger.info("Writing final section: %s", section.name)
                    completed_section = write_final_section(
                        section,
                        state.topic,
                        completed_sections_str,
                        request,
                        context=context,
                    )
                    state.completed_sections.append(completed_section)
        else:
            logger.info("No final sections to write, skipping this step.")

        # Step 5: Compile final report
        logger.info("Compiling final report...")
        completed_sections_dict = {s.name: s.content for s in state.completed_sections}

        # Update sections with completed content in original order
        for section in state.sections:
            section.content = completed_sections_dict[section.name]

        # Compile final report
        final_report = "\n\n".join([s.content for s in state.sections])
        state.final_report = final_report

        logger.info("Deep research workflow completed successfully")
        logger.debug("Final report content: %s", final_report)

        return DeepResearchWorkflowResponse(
            request_id=request.request_id,
            final_report=state.final_report,
            sections=state.sections,
            source_str=(state.source_str if request.include_source_str else None),
        )

    except Exception as e:
        logger.exception("Error in deep research workflow: %s", str(e))
        return DeepResearchWorkflowResponse(
            request_id=request.request_id,
            final_report="",
            sections=[],
            source_str=None,
            success=False,
            error=str(e),
        )


@agsbench_deep_research_fsm.async_fast_service
async def generate_report_plan_step_async(
    state: ReportState,
    request: DeepResearchWorkflowRequest,
    context: RequestContext = None,
) -> Dict[str, Any]:
    """Generate a report plan step asynchronously.

    Args:
        state: The current report state.
        request: The workflow request.
        context: The request context.

    Returns:
        A dictionary containing the report plan.
    """
    logger.info(
        "Asynchronously Generating report plan for topic: %s with request: %s",
        state.topic,
        request,
    )
    # Generate planning queries using the LLM task
    topic = state.topic
    feedback_list = state.feedback_on_report_plan or []
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    # Generate planning queries using the LLM task
    query_req = PlannerQueryWritingRequest(
        topic=topic,
        report_structure=request.report_structure,
        number_of_queries=request.number_of_queries,
        model=request.writer_model,
        max_tokens=request.writer_model_kwargs.get("max_tokens", request.max_tokens),
        temperature=request.writer_model_kwargs.get("temperature", request.temperature),
        seed=request.writer_model_kwargs.get("seed", request.seed),
        extra_gen_kwargs=request.writer_model_kwargs.get(
            "extra_gen_kwargs", request.extra_gen_kwargs
        ),
    )

    queries_response = await generate_queries_for_planner_with_llm_async(
        query_req, context=context
    )
    if not queries_response.success:
        raise RuntimeError(f"Query generation failed: {queries_response.error}")

    query_list = queries_response.queries

    search_api_config = request.search_api_config or {}

    # Use new async search task
    search_request = SearchTaskRequest(
        queries=query_list,
        search_api=request.search_api,
        search_api_config=search_api_config,
        max_tokens_per_source=request.max_tokens_per_source,
        include_raw_content=request.include_raw_content,
        deduplication_strategy=request.deduplication_strategy,
        search_post_processing=request.search_post_processing,
    )
    search_response = await search_task_async(request=search_request, context=context)
    source_str = search_response.merged_content

    # Generate report plan using the LLM task
    plan_req = ReportPlanningRequest(
        topic=topic,
        report_structure=request.report_structure,
        context=source_str,
        feedback=feedback,
        model=request.planner_model,
        max_tokens=request.planner_model_kwargs.get("max_tokens", request.max_tokens),
        temperature=request.planner_model_kwargs.get(
            "temperature", request.temperature
        ),
        seed=request.planner_model_kwargs.get("seed", request.seed),
        extra_gen_kwargs=request.planner_model_kwargs.get(
            "extra_gen_kwargs", request.extra_gen_kwargs
        ),
    )

    plan_response = await plan_report_with_llm_async(plan_req, context=context)
    if not plan_response.success:
        raise RuntimeError(f"Report planning failed: {plan_response.error}")

    logger.info("Generated report plan with sections: %s", plan_response.sections)
    return {
        "sections": plan_response.sections,
        "source_str": source_str if request.include_source_str else None,
    }


@agsbench_deep_research_fsm.async_fast_service
async def build_section_with_research_async(
    section: Section,
    topic: Annotated[str, Body(..., description="The overall research topic")],
    request: DeepResearchWorkflowRequest,
    context: RequestContext = None,
) -> Section:
    """Build a section with iterative research and writing asynchronously.

    Args:
        section: The section to research and write
        topic: The overall research topic
        request: Configuration for the workflow
        context: The request context

    Returns:
        The completed section with content
    """
    logger.info(
        "Asynchronously Building section: %s for topic: %s", section.name, topic
    )
    search_iterations = 0
    current_section = section

    # Generate queries for this section
    query_req = QueryGenerationRequest(
        topic=topic,
        section_topic=current_section.description,
        number_of_queries=request.number_of_queries,
        model=request.writer_model,
        max_tokens=request.writer_model_kwargs.get("max_tokens", request.max_tokens),
        temperature=request.writer_model_kwargs.get("temperature", request.temperature),
        seed=request.writer_model_kwargs.get("seed", request.seed),
        extra_gen_kwargs=request.writer_model_kwargs.get(
            "extra_gen_kwargs", request.extra_gen_kwargs
        ),
    )

    queries_response = await generate_queries_with_llm_async(query_req, context=context)
    if not queries_response.success:
        logger.warning("Query generation failed: %s", queries_response.error)
        return current_section  # Return section without content

    # Extract queries from response
    query_list = queries_response.queries

    while search_iterations < request.max_search_depth:
        # Search the web (async operation)
        search_api_config = request.search_api_config or {}

        # Use new async search task
        search_request = SearchTaskRequest(
            queries=query_list,
            search_api=request.search_api,
            search_api_config=search_api_config,
            max_tokens_per_source=request.max_tokens_per_source,
            include_raw_content=request.include_raw_content,
            deduplication_strategy=request.deduplication_strategy,
            search_post_processing=request.search_post_processing,
        )
        search_response = await search_task_async(
            request=search_request, context=context
        )
        source_str = search_response.merged_content

        # Write section content
        write_req = SectionWritingRequest(
            topic=topic,
            section_name=current_section.name,
            section_topic=current_section.description,
            context=source_str,
            section_content=current_section.content or "",
            model=request.writer_model,
            max_tokens=request.writer_model_kwargs.get(
                "max_tokens", request.max_tokens
            ),
            temperature=request.writer_model_kwargs.get(
                "temperature", request.temperature
            ),
            seed=request.writer_model_kwargs.get("seed", request.seed),
            extra_gen_kwargs=request.writer_model_kwargs.get(
                "extra_gen_kwargs", request.extra_gen_kwargs
            ),
        )

        write_response = await write_section_with_llm_async(write_req, context=context)
        if not write_response.success:
            logger.warning("Section writing failed: %s", write_response.error)
            break

        current_section.content = write_response.content

        # Grade the section
        grade_req = SectionGradingRequest(
            topic=topic,
            section_topic=current_section.description,
            section_content=current_section.content,
            number_of_follow_up_queries=request.number_of_queries,
            model=request.planner_model,
            max_tokens=request.planner_model_kwargs.get(
                "max_tokens", request.max_tokens
            ),
            temperature=request.planner_model_kwargs.get(
                "temperature", request.temperature
            ),
            seed=request.planner_model_kwargs.get("seed", request.seed),
            extra_gen_kwargs=request.planner_model_kwargs.get(
                "extra_gen_kwargs", request.extra_gen_kwargs
            ),
        )

        grade_response = await grade_section_with_llm_async(grade_req, context=context)
        if not grade_response.success:
            logger.warning("Section grading failed: %s", grade_response.error)
            break

        # If passing or max depth reached, we're done
        if (
            grade_response.grade == "pass"
            or search_iterations >= request.max_search_depth - 1
        ):
            if grade_response.grade == "pass":
                logger.info(
                    "Section '%s' completed with grade: %s",
                    current_section.name,
                    grade_response.grade,
                )
            else:
                logger.warning(
                    "Section '%s' failed with grade: %s, but max depth reached",
                    current_section.name,
                    grade_response.grade,
                )
            break

        search_iterations += 1
        query_list = grade_response.follow_up_queries
        if not query_list:
            logger.warning(
                "Grade failed, but No follow-up queries generated, stopping research for section: %s",
                current_section.name,
            )
            break

    logger.info(
        "Completed section '%s' after %d search iterations",
        current_section.name,
        search_iterations,
    )
    return current_section


@agsbench_deep_research_fsm.async_fast_service
async def write_final_section_async(
    section: Section,
    topic: Annotated[str, Body(..., description="The overall research topic")],
    completed_sections_str: Annotated[
        str, Body(..., description="Context from completed research sections")
    ],
    request: DeepResearchWorkflowRequest,
    context: RequestContext = None,
) -> Section:
    """Write a section that doesn't require research asynchronously.

    Args:
        section: The section to write
        topic: The overall research topic
        completed_sections_str: Context from completed research sections
        request: Configuration for the workflow
        context: The request context

    Returns:
        The completed section with content
    """
    logger.info(
        "Asynchronously Writing final section: %s for topic: %s", section.name, topic
    )
    write_req = FinalSectionWritingRequest(
        topic=topic,
        section_name=section.name,
        section_topic=section.description,
        completed_report_sections=completed_sections_str,
        model=request.writer_model,
        max_tokens=request.writer_model_kwargs.get("max_tokens", request.max_tokens),
        temperature=request.writer_model_kwargs.get("temperature", request.temperature),
        seed=request.writer_model_kwargs.get("seed", request.seed),
        extra_gen_kwargs=request.writer_model_kwargs.get(
            "extra_gen_kwargs", request.extra_gen_kwargs
        ),
    )

    write_response = await write_final_section_with_llm_async(
        write_req, context=context
    )
    if not write_response.success:
        logger.warning("Final section writing failed: %s", write_response.error)
        section.content = f"Error writing section: {write_response.error}"
    else:
        section.content = write_response.content

    logger.info(
        "Completed final section '%s' with content: %s",
        section.name,
        section.content[:10] + "...",  # Log first 10 chars of content
    )
    return section


@agsbench_deep_research_fsm.async_fast_service
@with_timeout_and_retry(
    timeout=get_drs_config_value("timeout", 60 * 60),  # 1 hour default timeout
    max_retries=get_drs_config_value("max_retries", 0),  # default 0 retries
)
async def deep_research_workflow_async(
    request: DeepResearchWorkflowRequest,
    context: RequestContext = None,
) -> DeepResearchWorkflowResponse:
    """Execute the deep research workflow asynchronously.

    Args:
        request: The workflow request with topic and configuration

    Returns:
        The workflow response with the completed report
    """
    logger.info(
        "Starting deep research workflow for topic: %s with request %s",
        request.topic,
        request,
    )

    # Initialize state
    state = ReportState(
        topic=request.topic,
        sections=[],
        completed_sections=[],
        feedback_on_report_plan=[],
        report_sections_from_research="",
        final_report="",
        source_str="",
    )

    try:
        # Step 1: Generate report plan
        logger.info("Generating report plan...")
        plan_result = await generate_report_plan_step_async(
            state, request, context=context
        )
        state.sections = plan_result["sections"]
        if request.include_source_str and plan_result.get("source_str"):
            state.source_str = plan_result["source_str"]

        # Step 2: Research and write sections that need research
        logger.info("Building sections with research...")
        research_sections = [s for s in state.sections if s.research]

        # for async processing, we enable parallelization with asyncio always
        completed_research_sections = await asyncio.gather(
            *[
                build_section_with_research_async(
                    section, state.topic, request, context=context
                )
                for section in research_sections
            ]
        )

        # Step 3: Format completed research sections as context
        completed_sections_str = format_sections(completed_research_sections)
        state.report_sections_from_research = completed_sections_str
        state.completed_sections = completed_research_sections.copy()

        # Step 4: Write final sections that don't need research
        logger.info("Writing final sections...")
        final_sections = [s for s in state.sections if not s.research]
        if len(final_sections) > 0:
            state.completed_sections.extend(
                await asyncio.gather(
                    *[
                        write_final_section_async(
                            section,
                            state.topic,
                            completed_sections_str,
                            request,
                            context=context,
                        )
                        for section in final_sections
                    ]
                )
            )
        else:
            logger.info("No final sections to write, skipping this step.")

        # Step 5: Compile final report
        logger.info("Compiling final report...")
        completed_sections_dict = {s.name: s.content for s in state.completed_sections}

        # Update sections with completed content in original order
        for section in state.sections:
            section.content = completed_sections_dict[section.name]

        # Compile final report
        final_report = "\n\n".join([s.content for s in state.sections])
        state.final_report = final_report

        logger.info("Deep research workflow completed successfully")
        logger.debug("Final report content: %s", final_report)

        return DeepResearchWorkflowResponse(
            request_id=request.request_id,
            final_report=state.final_report,
            sections=state.sections,
            source_str=(state.source_str if request.include_source_str else None),
        )

    except Exception as e:
        logger.exception("Error in deep research workflow: %s", str(e))
        return DeepResearchWorkflowResponse(
            request_id=request.request_id,
            final_report="",
            sections=[],
            source_str=None,
            success=False,
            error=str(e),
        )
