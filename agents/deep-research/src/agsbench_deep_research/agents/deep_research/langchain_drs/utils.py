"""Utilities for Deep Research System."""

import datetime
import textwrap
import logging
import asyncio
from typing import List, Optional, Dict, Any
from agsbench_deep_research.core.config import config_manager
from agsbench_deep_research.core.service.search import RetrievedDocument
from .state import Section

logger = logging.getLogger(__name__)


def get_drs_config_value(key, default=None):
    """Get a configuration value from the deep research config."""
    return config_manager.get_config_value("deep_research.langchain_drs", key, default)


def get_search_params(
    search_api: str, search_api_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Filter search_api_config to include only parameters accepted by the API.

    Args:
        search_api: The search API identifier (e.g., "tavily", "perplexity").
        search_api_config: The configuration dictionary for the search API.

    Returns:
        A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    search_api_params = {
        "tavily": ["max_results", "topic"],
        "perplexity": [],
        "duckduckgo": ["max_results"],
        "googlesearch": ["max_results"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = search_api_params.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}


def deduplicate_search_results(
    search_results: List[RetrievedDocument],
    deduplication_strategy: str = "keep_first",
) -> List[RetrievedDocument]:
    """Deduplicate search results based on URL.

    Args:
        search_results: List of RetrievedDocument objects.
        deduplication_strategy: Whether to keep the first or last result.

    Returns:
        A list of unique RetrievedDocument objects, deduplicated by URL.
        If deduplication_strategy is "keep_first", the first occurrence is kept.
        If "keep_last", the last occurrence is kept.
    """
    logger.debug(
        f"Deduplicating {len(search_results)} search results using strategy: {deduplication_strategy}"
    )
    if deduplication_strategy == "keep_first":
        seen_urls = set()
        unique_results = []
        for result in search_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
    elif deduplication_strategy == "keep_last":
        seen_urls = {}
        for result in search_results:
            seen_urls[result.url] = result
        unique_results = list(seen_urls.values())
    else:
        raise ValueError(f"Invalid deduplication strategy: {deduplication_strategy}")
    logger.debug(f"Deduplication complete. {len(unique_results)} unique results found.")
    return unique_results


def get_formatted_search_results(
    search_results: List[RetrievedDocument],
    include_raw_content: bool = True,
    max_char_to_include: Optional[int] = 30_000,
) -> str:
    """Format search results into a readable string.

    Args:
        search_results: List of RetrievedDocument objects.
        include_raw_content: Whether to include raw content in the output.
        max_char_to_include: Maximum number of characters to include from raw content.

    Returns:
        Formatted string with search results.
    """
    logger.debug(
        f"Formatting {len(search_results)} search results with include_raw_content={include_raw_content} and max_char_to_include={max_char_to_include}"
    )

    if not search_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    # # v1
    # formatted_text = "Content from sources:\n"
    # for i, result in enumerate(search_results, 1):
    #     formatted_text += f"{'='*80}\n"  # Clear section separator
    #     formatted_text += f"Source {i}: {result.title}\n"
    #     formatted_text += f"{'-'*80}\n"  # Subsection separator
    #     formatted_text += f"URL: {result.url}\n===\n"
    #     formatted_text += f"Most relevant content from source: {result.content}\n===\n"
    #     if (
    #         include_raw_content
    #         and result.raw_content
    #         and result.raw_content != result.content
    #     ):
    #         # Using rough estimate of 4 characters per token
    #         char_limit = max_char_to_include
    #         if len(raw_content) > char_limit:
    #             raw_content = raw_content[:char_limit] + "... [truncated]"
    #         formatted_text += (
    #             f"Full source content limited to {max_char_to_include} "
    #             f"characters: {raw_content}\n\n"
    #         )
    #     formatted_text += f"{'='*80}\n\n"  # End section separator

    formatted_text = f"Search results: \n\n"
    for i, result in enumerate(search_results):
        formatted_text += f"\n\n--- SOURCE {i+1}: {result.title} ---\n"
        formatted_text += f"URL: {result.url}\n\n"
        formatted_text += f"SUMMARY:\n{result.content}\n\n"
        if (
            include_raw_content
            and result.raw_content
            and result.raw_content != result.content
        ):
            if max_char_to_include is None:
                formatted_text += (
                    f"FULL CONTENT:\n{result.raw_content}"  # Limit content size
                )
            else:
                formatted_text += f"FULL CONTENT:\n{result.raw_content[:max_char_to_include]}"  # Limit content size
        formatted_text += "\n\n" + "-" * 80 + "\n"

    formatted_text = formatted_text.strip()
    logger.debug(
        f"Formatted search results:\n{formatted_text[:50]}..."
    )  # Log first 50 chars
    return formatted_text


def format_sections(sections: List[Section]) -> str:
    """Format a list of sections into a string"""
    logger.debug(f"Formatting {len(sections)} sections for report")
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += textwrap.dedent(
            f"""
            {'='*60}
            Section {idx}: {section.name}
            {'='*60}
            Description:
            {section.description}
            Requires Research:
            {section.research}

            Content:
            {section.content if section.content else '[Not yet written]'}

            """
        )
    return formatted_str


async def scrape_pages_async(titles: List[str], urls: List[str]) -> List[str]:
    """
    Scrapes content from a list of URLs and formats it into a readable markdown document.

    This function:
    1. Takes a list of page titles and URLs
    2. Makes asynchronous HTTP requests to each URL
    3. Converts HTML content to markdown
    4. Formats all content with clear source attribution

    Args:
        titles (List[str]): A list of page titles corresponding to each URL
        urls (List[str]): A list of URLs to scrape content from

    Returns:
        str: A formatted string containing the full content of each page in markdown format,
             with clear section dividers and source attribution
    """
    from markdownify import markdownify
    import httpx

    # Create an async HTTP client
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        pages = []

        # Fetch each URL and convert to markdown
        for url in urls:
            try:
                # Fetch the content
                response = await client.get(url)
                response.raise_for_status()

                # Convert HTML to markdown if successful
                if response.status_code == 200:
                    # Handle different content types
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type:
                        # Convert HTML to markdown
                        markdown_content = markdownify(response.text)
                        pages.append(markdown_content)
                    else:
                        # For non-HTML content, just mention the content type
                        pages.append(
                            f"Content type: {content_type} (not converted to markdown)"
                        )
                else:
                    pages.append(f"Error: Received status code {response.status_code}")

            except Exception as e:
                # Handle any exceptions during fetch
                pages.append(f"Error fetching URL: {str(e)}")

    return pages


def scrape_pages(titles: List[str], urls: List[str]) -> List[str]:
    """
    Synchronously scrapes content from a list of URLs and formats it into a readable markdown document.

    This function:
    1. Takes a list of page titles and URLs
    2. Makes synchronous HTTP requests to each URL
    3. Converts HTML content to markdown
    4. Formats all content with clear source attribution

    Args:
        titles (List[str]): A list of page titles corresponding to each URL
        urls (List[str]): A list of URLs to scrape content from

    Returns:
        str: A formatted string containing the full content of each page in markdown format,
             with clear section dividers and source attribution
    """
    # Create a new event loop for synchronous execution
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run the async scraping function in the event loop
    return loop.run_until_complete(scrape_pages_async(titles, urls))


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.datetime.now().strftime("%a %b %-d, %Y")


def sync_caller(async_function, *args, **kwargs):
    # Create or get an event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the async function to completion
    return loop.run_until_complete(async_function(*args, **kwargs))
    # return asyncio.run(async_function(*args, **kwargs))
