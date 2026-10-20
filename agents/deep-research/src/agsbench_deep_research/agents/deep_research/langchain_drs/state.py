"""State models for Deep Research System."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Section(BaseModel):
    """Represents a section of the research report."""

    name: str = Field(description="Name for this section of the report")
    description: str = Field(
        description="Brief overview of main topics and concepts to be covered"
    )
    research: bool = Field(
        description="Whether to perform web research for this section"
    )
    content: str = Field(description="The content of the section")


class Sections(BaseModel):
    """Container for report sections."""

    sections: List[Section] = Field(description="Sections of the report")


class SearchQuery(BaseModel):
    """Represents a search query."""

    search_query: str = Field(description="Query for web search")


class Queries(BaseModel):
    """Container for search queries."""

    queries: List[SearchQuery] = Field(description="List of search queries")


class Feedback(BaseModel):
    """Feedback on section quality."""

    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result indicating pass or fail"
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries"
    )


class ReportState(BaseModel):
    """State for the overall report generation."""

    topic: str = Field(description="Report topic")
    feedback_on_report_plan: List[str] = Field(
        default_factory=list, description="List of feedback on the report plan"
    )
    sections: List[Section] = Field(
        default_factory=list, description="List of report sections"
    )
    completed_sections: List[Section] = Field(
        default_factory=list, description="Completed sections"
    )
    report_sections_from_research: str = Field(
        default="", description="String of completed sections from research"
    )
    final_report: str = Field(default="", description="Final report")
    source_str: str = Field(
        default="", description="String of formatted source content"
    )


class SectionState(BaseModel):
    """State for individual section processing."""

    topic: str = Field(description="Report topic")
    section: Section = Field(description="Report section")
    search_iterations: int = Field(
        default=0, description="Number of search iterations done"
    )
    search_queries: List[SearchQuery] = Field(
        default_factory=list, description="List of search queries"
    )
    source_str: str = Field(
        default="", description="String of formatted source content"
    )
    report_sections_from_research: str = Field(
        default="", description="String of completed sections from research"
    )
    completed_sections: List[Section] = Field(
        default_factory=list, description="Final key for completed sections"
    )


class ReportStateInput(BaseModel):
    """Input state for report generation."""

    topic: str = Field(description="Report topic")


class ReportStateOutput(BaseModel):
    """Output state for report generation."""

    final_report: str = Field(description="Final report")
    source_str: Optional[str] = Field(
        default=None, description="String of formatted source content"
    )


class SectionOutputState(BaseModel):
    """Output state for section processing."""

    completed_sections: List[Section] = Field(description="Completed sections")
    source_str: Optional[str] = Field(
        default=None, description="String of formatted source content"
    )


class SearchResultSummary(BaseModel):
    summary: str
    key_excerpts: list[str]

    def get_formatted_summary(self) -> str:
        excerpts_str = "\n".join(f"- {e}" for e in self.key_excerpts)
        return f"""<summary>\n{self.summary}\n</summary>\n\n<key_excerpts>\n{excerpts_str}\n</key_excerpts>"""
