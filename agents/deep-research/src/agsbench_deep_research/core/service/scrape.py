"""
Scrape service for handling web scraping tasks.
Currently it is only a tool sets for the deep_research agent.
TODO: Implement a more generic scrape service that can be used by other agents.
"""

import os
import time
import uuid
import random
from abc import abstractmethod
from typing import List, Dict, Any, Optional, Iterable, cast, Union
from pydantic import BaseModel, Field
import logging
import asyncio
import json
import requests
import aiohttp
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
from markitdown import MarkItDown
import httpx
from io import BytesIO

scrape_logger = logging.getLogger(__name__)

md_client = MarkItDown(enable_plugins=True)  # Set to True to enable plugins

default_scrape_timeout = int(os.getenv("DEFAULT_SCRAPE_TIMEOUT", "30"))
default_scrape_max_retries = int(os.getenv("DEFAULT_SCRAPE_MAX_RETRIES", "3"))
scrape_max_concurrent = int(os.getenv("SCRAPE_MAX_CONCURRENT", "65536"))
max_concurrent = scrape_max_concurrent if scrape_max_concurrent > 0 else 65536

async_scrape_semaphore = asyncio.Semaphore(max_concurrent)
sync_scrape_semaphore = threading.Semaphore(max_concurrent)

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
]

SUPPORT_CONTENT_TYPE = [
    "text/html",
    "text/plain",
    "text/csv",
    "application/xhtml+xml",
    "application/xml",
    "application/pdf",
    "application/json",
]


def md_convert(url: str, data=None) -> str:
    data = data or url
    try:
        md_content = md_client.convert(data)
        return md_content.text_content
    except Exception as e:
        scrape_logger.exception(f"failed fetching URL {url}: {str(e)}")
        return f"Error: failed fetching URL {url}: {str(e)}"


def scrape_with_requests(
    urls: List[str],
    max_workers: int = scrape_max_concurrent,
    *,
    max_retries: int = default_scrape_max_retries,
    timeout: float = default_scrape_timeout,
    user_agents: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
    semaphore: Optional[threading.Semaphore] = sync_scrape_semaphore,
) -> List[str]:
    """Scrape multiple URLs synchronously with thread pooling and return their content as markdown.

    Args:
        urls: List of URLs to scrape
        max_workers: Maximum number of concurrent workers (default: 32)
        max_retries: Maximum retry attempts for failed requests (default: 3)
        timeout: Request timeout in seconds (default: 30.0)
        user_agents: List of user agents to rotate (default: common browsers)
        headers: Additional headers to include

    Returns:
        List of markdown strings (or error messages) for each URL
    """
    # Configure headers
    final_headers = {
        "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        **(headers or {}),
    }

    if user_agents:
        final_headers["User-Agent"] = random.choice(user_agents)
    else:
        final_headers["User-Agent"] = random.choice(DEFAULT_USER_AGENTS)

    counters = {"success": 0, "error": 0, "retries": 0}

    def scrape_single_url(url: str) -> str:
        """Scrape a single URL with an iterative retry loop."""
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                # The semaphore is now acquired and released within each loop iteration
                with semaphore:
                    with httpx.Client(
                        headers=final_headers, follow_redirects=True, timeout=timeout
                    ) as client:
                        response = client.get(url)
                        response.raise_for_status()

                        content_type = response.headers.get("Content-Type", "").lower()
                        if not any(ct in content_type for ct in SUPPORT_CONTENT_TYPE):
                            scrape_logger.warning(
                                f"Unsupported content type {content_type} for {url}"
                            )
                            counters["error"] += 1
                            counters[content_type] = counters.get(content_type, 0) + 1
                            return f"Error: unsupported content type {content_type} for {url}"

                        result = md_convert(url, BytesIO(response.content))
                        scrape_logger.debug(
                            f"Scraped {url} successfully with content type {content_type} and length {len(result)}"
                        )
                        counters["success"] += 1
                        return result  # Success, exit the loop and function

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                is_retryable = isinstance(e, httpx.RequestError) or (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code in {429, 500, 502, 503, 504}
                )

                if attempt < max_retries and is_retryable:
                    wait = min(2**attempt, 10)  # Exponential backoff
                    time.sleep(wait + random.uniform(0, 1))  # Jitter
                    counters["retries"] += 1
                    continue  # Go to the next attempt
                else:
                    break  # Exit loop if not retryable or max retries reached

            except Exception as e:
                last_exception = e
                scrape_logger.exception(f"Unexpected error processing {url}")
                break  # Exit loop on other unexpected errors

        # This block runs if the loop finishes without a successful return
        counters["error"] += 1
        status_code = getattr(
            getattr(last_exception, "response", None), "status_code", "N/A"
        )
        scrape_logger.error(
            f"Failed {url} (attempt {max_retries}): {status_code} - {str(last_exception)}"
        )
        return f"Error: failed fetching URL {url} due to {str(last_exception)} (after {max_retries} attempts)"

    try:
        if max_workers <= 1:
            if max_workers <= 0:
                scrape_logger.warning(
                    f"max_workers is set to {max_workers}, using single-threaded scraping."
                )
            results = [scrape_single_url(url) for url in urls]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(scrape_single_url, urls))

        scrape_logger.info(
            f"Scraped {counters['success']}/{len(urls)} pages successfully with requests, "
            f"{counters['error']} errors, {counters['retries']} retries, "
            f"counter for each content type: {counters}"
        )

        return results

    except Exception as e:
        scrape_logger.error(f"Unexpected error in scraping batch: {str(e)}")
        return [
            f"Error: failed fetching URL {url} due to unexpected error: {str(e)}"
            for url in urls
        ]


async def scrape_with_requests_async(
    urls: List[str],
    *,
    max_retries: int = default_scrape_max_retries,
    timeout: float = default_scrape_timeout,
    user_agents: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
    semaphore: Optional[asyncio.Semaphore] = async_scrape_semaphore,
) -> List[str]:
    """Scrape multiple URLs asynchronously and return their content as markdown.

    Args:
        urls: List of URLs to scrape
        max_retries: Maximum retry attempts for failed requests (default: 3)
        timeout: Request timeout in seconds (default: 30.0)
        user_agents: List of user agents to rotate (default: common browsers)
        headers: Additional headers to include
        semaphore: Optional existing semaphore to share with other tasks

    Returns:
        List of markdown strings (or error messages) for each URL
    """
    # Configure headers
    final_headers = {
        "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        **(headers or {}),
    }

    if user_agents:
        final_headers["User-Agent"] = random.choice(user_agents)
    else:
        final_headers["User-Agent"] = random.choice(DEFAULT_USER_AGENTS)

    counters = {"success": 0, "error": 0, "retries": 0}

    async def scrape_url_async(url: str) -> str:
        """Scrape a single URL with an iterative retry loop."""
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                # The semaphore is now acquired and released within each loop iteration
                async with semaphore:
                    async with httpx.AsyncClient(
                        headers=final_headers, follow_redirects=True, timeout=timeout
                    ) as client:
                        response = await client.get(url)
                        response.raise_for_status()

                        content_type = response.headers.get("Content-Type", "").lower()
                        if not any(ct in content_type for ct in SUPPORT_CONTENT_TYPE):
                            scrape_logger.warning(
                                f"Unsupported content type {content_type} for {url}"
                            )
                            counters["error"] += 1
                            counters[content_type] = counters.get(content_type, 0) + 1
                            return f"Error: unsupported content type {content_type} for {url}"
                        else:
                            result = md_convert(url, BytesIO(response.content))
                            scrape_logger.debug(
                                f"Scraped {url} successfully with content type {content_type} and length {len(result)}"
                            )
                            counters["success"] += 1
                            return result

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                is_retryable = isinstance(e, httpx.RequestError) or (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code in {429, 500, 502, 503, 504}
                )

                if attempt < max_retries and is_retryable:
                    wait = min(2**attempt, 10)  # Exponential backoff
                    # Sleep happens *outside* the semaphore's scope
                    await asyncio.sleep(wait + random.uniform(0, 1))
                    counters["retries"] += 1
                    continue  # Go to the next attempt
                else:
                    break  # Exit loop if not retryable or max retries reached

            except Exception as e:
                last_exception = e
                scrape_logger.exception(f"Unexpected error processing {url}")
                break  # Exit loop on other unexpected errors

        # This block runs if the loop finishes without a successful return
        counters["error"] += 1
        status_code = getattr(
            getattr(last_exception, "response", None), "status_code", "N/A"
        )
        scrape_logger.error(
            f"Failed {url} (attempt {max_retries}): {status_code} - {str(last_exception)}"
        )
        return f"Error: failed fetching URL {url} due to {str(last_exception)} (after {max_retries} attempts)"

    try:
        tasks = [scrape_url_async(url) for url in urls]
        pages = await asyncio.gather(*tasks, return_exceptions=False)

        scrape_logger.info(
            f"Scraped {counters['success']}/{len(urls)} pages successfully with requests async, "
            f"{counters['error']} errors, {counters['retries']} retries, "
            f"counter for each content type: {counters}"
        )

        return pages

    except Exception as e:
        scrape_logger.error(
            f"Unexpected error in scraping batch: {str(e)} with requests async"
        )
        return [
            f"Error: failed fetching URL {url} due to unexpected error: {str(e)}"
            for url in urls
        ]


def scrape_pages_with_markitdown(urls: List[str], max_workers: int = 32) -> List[str]:
    if max_workers <= 1:
        if max_workers <= 0:
            scrape_logger.warning(
                f"max_workers is set to {max_workers}, using single-threaded scraping."
            )
        pages = [md_convert(url) for url in urls]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use map to apply the function to each URL in parallel
            pages = list(executor.map(md_convert, urls))

    success_count = sum(1 for page in pages if not page.startswith("Error:"))
    error_count = len(pages) - success_count
    scrape_logger.info(
        f"Scraped {success_count} / {len(urls)} pages successfully with markitdown, "
        f"{error_count} / {len(urls)} errors."
    )
    return pages


async def scrape_pages_with_markitdown_async(
    urls: List[str], max_workers: int = 32
) -> List[str]:
    loop = asyncio.get_event_loop()
    if max_workers <= 1:
        if max_workers <= 0:
            scrape_logger.warning(
                f"max_workers is set to {max_workers}, using single-threaded scraping."
            )
        pages = [md_convert(url) for url in urls]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [loop.run_in_executor(executor, md_convert, url) for url in urls]
            pages = await asyncio.gather(*tasks)
    success_count = sum(1 for page in pages if not page.startswith("Error:"))
    error_count = len(pages) - success_count
    scrape_logger.info(
        f"Scraped {success_count} / {len(urls)} pages successfully with markitdown async, "
        f"{error_count} / {len(urls)} errors."
    )
    return pages


def scrap_with_firecrawl(urls: List[str], api_url: str, **kwargs) -> List[str]:
    from firecrawl.firecrawl import FirecrawlApp
    from firecrawl.firecrawl import ScrapeOptions

    app = FirecrawlApp(api_url=api_url)
    method = kwargs.get("method", "scrape")  # Default to 'scrape'
    pages = []
    for url in urls:
        if method == "scrape":
            try:
                scrape_status = app.scrape_url(url, formats=["markdown", "links"])
            except Exception as e:
                scrape_logger.exception(f"Failed to scrape {url}: {str(e)}")
                page = f"Error: failed fetching URL {url}: {str(e)}"
                pages.append(page)
                continue
            if scrape_status.success:
                page = scrape_status.markdown
                scrape_logger.info(f"Scraped {url} successfully.")
            else:
                page = f"Error: failed fetching URL {url}: {scrape_status.error or 'Unknown error'}"
                scrape_logger.error(
                    f"failed fetching URL {url}: {scrape_status.error or 'Unknown error'}"
                )
        elif method == "crawl":
            try:
                crawl_status = app.crawl_url(
                    url,
                    limit=100,
                    scrape_options=ScrapeOptions(
                        formats=["markdown"],
                    ),
                    poll_interval=30,
                )
            except Exception as e:
                scrape_logger.exception(f"Failed to crawl {url}: {str(e)}")
                page = f"Error: failed to crawl {url}: {str(e)}"
                pages.append(page)
                continue
            if crawl_status.success:
                if not crawl_status.data:
                    scrape_logger.warning(f"No pages crawled from {url}.")
                    page = f"Error: No pages crawled from {url}."
                else:
                    page = "\n\n".join(
                        page.markdown for page in crawl_status.data if page.markdown
                    )
                    scrape_logger.info(
                        f"Crawled {len(crawl_status.data)} pages from {url}."
                    )
            else:
                scrape_logger.error(
                    f"Failed to crawl {url}: {crawl_status or 'Unknown error'}"
                )
                page = (
                    f"Error: failed to crawl {url}: {crawl_status or 'Unknown error'}"
                )
        else:
            raise ValueError("Method must be 'scrape' or 'crawl'.")
        pages.append(page)
    success_count = sum(1 for page in pages if not page.startswith("Error:"))
    error_count = len(pages) - success_count
    scrape_logger.info(
        f"Scraped {success_count} / {len(urls)} pages successfully with firecrawl, "
        f"{error_count} / {len(urls)} errors."
    )
    return pages


async def scrap_with_firecrawl_async(
    urls: List[str], api_url: str, **kwargs
) -> List[str]:
    from firecrawl.firecrawl import AsyncFirecrawlApp
    from firecrawl.firecrawl import ScrapeOptions

    app = AsyncFirecrawlApp(api_url=api_url)
    method = kwargs.get("method", "scrape")  # Default to 'scrape'

    async def scrape_url(url: str) -> str:
        if method == "scrape":
            try:
                scrape_status = await app.scrape_url(url, formats=["markdown", "links"])
            except Exception as e:
                scrape_logger.exception(f"Failed to scrape {url}: {str(e)}")
                return f"Error: failed fetching URL {url}: {str(e)}"
            if scrape_status.success:
                page = scrape_status.markdown
            else:
                page = f"Error: failed fetching URL {url}: {scrape_status.error or 'Unknown error'}"
                scrape_logger.error(
                    f"failed fetching URL {url}: {scrape_status.error or 'Unknown error'}"
                )
        elif method == "crawl":
            try:
                crawl_status = await app.crawl_url(
                    url,
                    limit=100,
                    scrape_options=ScrapeOptions(
                        formats=["markdown"],
                    ),
                    poll_interval=30,
                )
            except Exception as e:
                scrape_logger.exception(f"Failed to crawl {url}: {str(e)}")
                return f"Error: failed to crawl {url}: {str(e)}"
            if crawl_status.success:
                if not crawl_status.data:
                    scrape_logger.warning(f"No pages crawled from {url}.")
                    return f"Error: No pages crawled from {url}."

                page = "\n\n".join(
                    page.markdown for page in crawl_status.data if page.markdown
                )
                scrape_logger.info(
                    f"Crawled {len(crawl_status.data)} pages from {url}."
                )
            else:
                scrape_logger.error(
                    f"failed crawling {url}: {crawl_status.error or 'Unknown error'}"
                )
                page = f"Error: failed crawling {url}: {crawl_status.error or 'Unknown error'}"
        else:
            raise ValueError("Method must be 'scrape' or 'crawl'.")
        return page

    tasks = [scrape_url(url) for url in urls]
    pages = await asyncio.gather(*tasks)

    success_count = sum(1 for page in pages if not page.startswith("Error:"))
    error_count = len(pages) - success_count
    scrape_logger.info(
        f"Scraped {success_count} / {len(urls)} pages successfully with firecrawl async, "
        f"{error_count} / {len(urls)} errors."
    )
    return pages


if __name__ == "__main__":
    # Example usage
    # logging.basicConfig(level=logging.DEBUG)
    # set DEBUG logging level for scrape_logger
    scrape_logger.setLevel(logging.DEBUG)
    urls = [
        "http://httpbin.org",
        "http://httpbin.org/headers",
        "http://httpbin.org/user-agent",
        "http://httpbin.org/robots.txt",
        "http://httpbin.org/html",
        "http://httpbin.org/json",
        "http://httpbin.org/xml",
        "http://httpbin.org/image/png",
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "http://httpbin.org/response-headers?Content-Type=text/csv",
        "https://code.jquery.com/jquery-3.6.0.min.js",
    ]
    results = scrape_with_requests(urls)
    for i, (url, result) in enumerate(zip(urls, results)):
        print(f"({i}) {url}: {result[:10]}")  # Print first 10 characters of each result
    time.sleep(1)  # Give some time for logging to flush

    async def main():
        async_results = await scrape_with_requests_async(urls)
        for i, (url, result) in enumerate(zip(urls, async_results)):
            print(f"({i}) {url}: {result[:10]}")

    asyncio.run(main())
