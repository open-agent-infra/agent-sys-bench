"""
Utility functions for the LLM service.
"""

import json
import re
import logging
from typing import Dict, Any, Union, Type, Callable, Optional, Tuple, List
from pydantic import BaseModel
import numpy as np
import asyncio
import functools
import time
import inspect
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


Matrix = Union[list[list[float]], list[np.ndarray], np.ndarray]

logger = logging.getLogger(__name__)


class StructuredOutputParser:
    """Helper class for parsing structured output from LLM responses."""

    @staticmethod
    def extract_schema_from_response_format(
        response_format: Union[Dict[str, Any], Type],
    ) -> Dict[str, Any]:
        """Extract schema from the response format."""
        if isinstance(response_format, Type):
            # If response_format is a Pydantic model, convert it to a schema dict
            if not issubclass(response_format, BaseModel):
                raise ValueError("response_format must be a Pydantic model or a dict.")
            return response_format.model_json_schema()
        elif isinstance(response_format, dict):
            # If response_format is a dict, use it directly as the schema
            return response_format
        else:
            # If no response_format is provided, return None
            return None

    @staticmethod
    def extract_json_from_text(text: str) -> str:
        """Extract JSON content from text, handling code blocks and markdown."""
        if not text:
            return text

        # First try to find JSON in code blocks
        json_block_patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"`(.*?)`",
        ]

        for pattern in json_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match.strip())
                    return match.strip()
                except json.JSONDecodeError:
                    continue

        # Try to find JSON-like content in the text
        brace_pattern = r"\{.*\}"
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

        return text.strip()

    @staticmethod
    def parse_structured_output(
        text: str, response_format: Union[Dict[str, Any], Type]
    ) -> Any:
        """Parse structured output from text using the provided response_format."""
        logger.debug(f"Parsing structured output from text: {text}")
        schema = StructuredOutputParser.extract_schema_from_response_format(
            response_format
        )
        if not schema or not text:
            return text

        try:
            json_text = StructuredOutputParser.extract_json_from_text(text)
            parsed_data = json.loads(json_text)

            if isinstance(parsed_data, dict) and "properties" in schema:
                validated_data = {}
                properties = schema.get("properties", {})

                for key, value in parsed_data.items():
                    if key in properties:
                        validated_data[key] = value
                    else:
                        validated_data[key] = value

                # return instance of BaseModel if schema is a Pydantic model
                if isinstance(response_format, type) and issubclass(
                    response_format, BaseModel
                ):
                    try:
                        result = response_format.model_validate(validated_data)
                        logger.debug(f"Validated structured output: {result}")
                        return result
                    except Exception as e:
                        logger.exception(f"Failed to validate structured output: {e}")
                        return {"error": "Validation failed", "raw_text": text}
                else:
                    result = validated_data if validated_data else parsed_data
                    logger.debug(f"Parsed structured output: {result}")
                    return result
            logger.debug(f"Parsed JSON data: {parsed_data}")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Original text: {text}")

            try:
                return StructuredOutputParser._extract_key_value_pairs(text, schema)
            except Exception as e2:
                logger.exception(f"Failed to extract structured data: {e2}")
                return {"error": "Failed to parse structured output", "raw_text": text}

    @staticmethod
    def _extract_key_value_pairs(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key-value pairs from unstructured text based on schema."""
        result = {}
        properties = schema.get("properties", {})

        for key in properties.keys():
            patterns = [
                rf'"{key}"\s*:\s*"([^"]*)"',
                rf'"{key}"\s*:\s*([^,\n\}}]+)',
                rf'{key}\s*:\s*"([^"]*)"',
                rf"{key}\s*:\s*([^,\n\}}]+)",
                rf'{key}\s*=\s*"([^"]*)"',
                rf"{key}\s*=\s*([^,\n\}}]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    prop_type = properties[key].get("type")
                    if prop_type == "integer":
                        try:
                            result[key] = int(value)
                        except ValueError:
                            result[key] = value
                    elif prop_type == "number":
                        try:
                            result[key] = float(value)
                        except ValueError:
                            result[key] = value
                    elif prop_type == "boolean":
                        result[key] = value.lower() in ("true", "yes", "1")
                    else:
                        result[key] = value
                    break

        return result if result else {"raw_text": text}


def compute_cosine_similarity(x: Matrix, y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices.
    (referencing the implementation of vectorstores in langchain_core)

    Args:
        x: A matrix of shape (n, m).
        y: A matrix of shape (k, m).

    Returns:
        A matrix of shape (n, k) where each element (i, j) is the cosine similarity
        between the ith row of X and the jth row of Y.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        msg = (
            "cosine_similarity requires numpy to be installed. "
            "Please install numpy with `pip install numpy`."
        )
        raise ImportError(msg) from e

    if len(x) == 0 or len(y) == 0:
        return np.array([])

    x = np.array(x)
    y = np.array(y)
    if x.shape[1] != y.shape[1]:
        msg = (
            f"Number of columns in X and Y must be the same. X has shape {x.shape} "
            f"and Y has shape {y.shape}."
        )
        raise ValueError(msg)
    try:
        import simsimd as simd  # type: ignore[import-not-found]
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(x, y.T) / np.outer(x_norm, y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return 1 - np.array(simd.cdist(x, y, metric="cosine"))


def compute_maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """Calculate maximal marginal relevance.
    (referencing the implementation of vectorstores in langchain_core)

    Args:
        query_embedding: The query embedding.
        embedding_list: A list of embeddings.
        lambda_mult: The lambda parameter for MMR. Default is 0.5.
        k: The number of embeddings to return. Default is 4.

    Returns:
        A list of indices of the embeddings to return.

    Raises:
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        msg = (
            "maximal_marginal_relevance requires numpy to be installed. "
            "Please install numpy with `pip install numpy`."
        )
        raise ImportError(msg) from e

    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = compute_cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = compute_cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


class TimeoutError(Exception):
    """Raised when function execution times out."""

    pass


def with_timeout_and_retry(
    timeout: Optional[float] = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    backoff_factor: float = 1.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    on_retry: Optional[Callable[[int, Exception, tuple, dict], None]] = None,
    raise_on_timeout: bool = True,
    timeout_method: str = "signal",  # "signal", "thread", or "cooperative"
):
    """
    Decorator that adds timeout and retry functionality to both sync and async functions.

    Args:
        timeout: Maximum time in seconds to wait for function completion
        max_retries: Maximum number of retry attempts (0 means no retries)
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for retry delay after each attempt
        exceptions: Exception types to catch and retry on
        on_retry: Optional callback function called on each retry attempt
                 Signature: on_retry(attempt_number, exception, args, kwargs)
                 You can modify args (tuple) and kwargs (dict) for next retry
        raise_on_timeout: Whether to raise TimeoutError or return None on timeout
        timeout_method: Method for sync timeout ("signal", "thread", or "cooperative")

    Returns:
        Decorated function with timeout and retry capabilities
    """

    def decorator(func: Callable) -> Callable:
        logger.debug(
            "Creating decorator for function: %s with timeout %s and max retries %d",
            func.__name__,
            timeout,
            max_retries,
        )
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _execute_async_with_timeout_and_retry(
                    func,
                    args,
                    kwargs,
                    timeout,
                    max_retries,
                    retry_delay,
                    backoff_factor,
                    exceptions,
                    on_retry,
                    raise_on_timeout,
                )

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return _execute_sync_with_timeout_and_retry(
                    func,
                    args,
                    kwargs,
                    timeout,
                    max_retries,
                    retry_delay,
                    backoff_factor,
                    exceptions,
                    on_retry,
                    raise_on_timeout,
                    timeout_method,
                )

            return sync_wrapper

    return decorator


async def _execute_async_with_timeout_and_retry(
    func: Callable,
    args: tuple,
    kwargs: dict,
    timeout: Optional[float],
    max_retries: int,
    retry_delay: float,
    backoff_factor: float,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
    on_retry: Optional[Callable[[int, Exception, tuple, dict], None]],
    raise_on_timeout: bool,
) -> Any:
    """Execute async function with timeout and retry logic."""
    last_exception = None
    current_delay = retry_delay
    # Convert to lists to allow modification
    args_list = list(args)

    for attempt in range(max_retries + 1):
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    func(*args_list, **kwargs), timeout=timeout
                )
            else:
                return await func(*args_list, **kwargs)

        except asyncio.TimeoutError as e:
            last_exception = TimeoutError(f"Function timed out after {timeout} seconds")
            if raise_on_timeout:
                if attempt == max_retries:
                    raise last_exception
            else:
                return None

        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break

        # Don't sleep after the last attempt
        if attempt < max_retries:
            if on_retry:
                # Call on_retry with current attempt number (1-based)
                on_retry(attempt + 1, last_exception, args_list, kwargs)
                # Convert back to tuple for next iteration
                args_list = list(args_list)

            logger.debug(
                f"Retrying in {current_delay} seconds (attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(current_delay)
            current_delay *= backoff_factor

    raise RetryError(
        f"Function failed after {max_retries + 1} attempts",
        last_exception,
        max_retries + 1,
    )


def _execute_sync_with_timeout_and_retry(
    func: Callable,
    args: tuple,
    kwargs: dict,
    timeout: Optional[float],
    max_retries: int,
    retry_delay: float,
    backoff_factor: float,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
    on_retry: Optional[Callable[[int, Exception, tuple, dict], None]],
    raise_on_timeout: bool,
    timeout_method: str = "signal",
) -> Any:
    """Execute sync function with timeout and retry logic."""
    last_exception = None
    current_delay = retry_delay
    # Convert to lists to allow modification
    args_list = list(args)

    for attempt in range(max_retries + 1):
        try:
            if timeout is not None:
                if timeout_method == "signal":
                    return _execute_with_signal_timeout(
                        func, tuple(args_list), kwargs, timeout
                    )
                elif timeout_method == "thread":
                    return _execute_with_thread_timeout(
                        func, tuple(args_list), kwargs, timeout
                    )
                elif timeout_method == "cooperative":
                    return _execute_with_cooperative_timeout(
                        func, tuple(args_list), kwargs, timeout
                    )
                else:
                    raise ValueError(f"Unknown timeout method: {timeout_method}")
            else:
                return func(*args_list, **kwargs)

        except TimeoutError as e:
            last_exception = e
            logger.warning(
                f"Function timed out on attempt {attempt + 1} after {timeout} seconds: {e}"
            )
            if raise_on_timeout:
                if attempt == max_retries:
                    raise last_exception
            else:
                return None

        except exceptions as e:
            last_exception = e
            logger.warning(f"Function raised exception on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                break

        # Don't sleep after the last attempt
        if attempt < max_retries:
            if on_retry:
                # Call on_retry with current attempt number (1-based)
                on_retry(attempt + 1, last_exception, args_list, kwargs)
                # Convert back to tuple for next iteration
                args_list = list(args_list)

            logger.debug(
                f"Retrying in {current_delay} seconds (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(current_delay)
            current_delay *= backoff_factor

    raise RetryError(
        f"Function failed after {max_retries + 1} attempts",
        last_exception,
        max_retries + 1,
    )


def _execute_with_signal_timeout(
    func: Callable, args: tuple, kwargs: dict, timeout: float
) -> Any:
    """Execute function with signal-based timeout (Unix only)."""
    import platform

    if platform.system() == "Windows":
        # Fall back to thread-based timeout on Windows
        return _execute_with_thread_timeout(func, args, kwargs, timeout)

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function timed out after {timeout} seconds")

    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        raise
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _execute_with_thread_timeout(
    func: Callable, args: tuple, kwargs: dict, timeout: float
) -> Any:
    """Execute function with thread-based timeout (fallback method)."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(f"Function timed out after {timeout} seconds")


def _execute_with_cooperative_timeout(
    func: Callable, args: tuple, kwargs: dict, timeout: float
) -> Any:
    """
    Execute function with cooperative timeout using a timeout context manager.
    This requires the function to check for timeout periodically.
    """
    timeout_context = TimeoutContext(timeout)

    # If function accepts a timeout_context parameter, pass it
    sig = inspect.signature(func)
    if "timeout_context" in sig.parameters:
        kwargs["timeout_context"] = timeout_context

    start_time = time.time()

    try:
        result = func(*args, **kwargs)
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Function timed out after {timeout} seconds")
        return result
    except TimeoutError:
        raise
    except Exception as e:
        # Check if we timed out during execution
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Function timed out after {timeout} seconds")
        raise


class TimeoutContext:
    """Context manager for cooperative timeout checking."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self.start_time = time.time()

    def check_timeout(self):
        """Check if timeout has been exceeded."""
        if time.time() - self.start_time > self.timeout:
            raise TimeoutError(f"Function timed out after {self.timeout} seconds")

    @property
    def remaining_time(self) -> float:
        """Get remaining time before timeout."""
        return max(0, self.timeout - (time.time() - self.start_time))

    @property
    def is_timeout(self) -> bool:
        """Check if timeout has been exceeded without raising exception."""
        return time.time() - self.start_time > self.timeout


# Convenience functions for direct wrapping without decorators
def wrap_with_timeout_and_retry(
    func: Callable,
    timeout: Optional[float] = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    backoff_factor: float = 1.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    on_retry: Optional[Callable[[int, Exception, tuple, dict], None]] = None,
    raise_on_timeout: bool = True,
    timeout_method: str = "signal",
) -> Callable:
    """
    Wrap a function with timeout and retry capabilities without using as decorator.

    Args:
        func: Function to wrap
        timeout: Maximum time in seconds to wait for function completion
        max_retries: Maximum number of retry attempts (0 means no retries)
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for retry delay after each attempt
        exceptions: Exception types to catch and retry on
        on_retry: Optional callback function called on each retry attempt
        raise_on_timeout: Whether to raise TimeoutError or return None on timeout
        timeout_method: Method for sync timeout ("signal", "thread", or "cooperative")

    Returns:
        Wrapped function with timeout and retry capabilities
    """
    return with_timeout_and_retry(
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        backoff_factor=backoff_factor,
        exceptions=exceptions,
        on_retry=on_retry,
        raise_on_timeout=raise_on_timeout,
        timeout_method=timeout_method,
    )(func)


# Example usage and tests
if __name__ == "__main__":
    import random

    # Example sync function that might fail
    @with_timeout_and_retry(
        timeout=2.0,
        max_retries=3,
        retry_delay=0.5,
        backoff_factor=2.0,
        exceptions=(ValueError, ConnectionError),
        on_retry=lambda attempt, exc, args, kwargs: print(f"Retry {attempt}: {exc}"),
    )
    def unreliable_sync_function(success_rate: float = 0.3):
        """Simulate an unreliable function."""
        time.sleep(random.uniform(0.1, 0.5))  # Simulate work
        if random.random() > success_rate:
            raise ValueError("Random failure occurred")
        return "Success!"

    # Example async function that might fail
    @with_timeout_and_retry(
        timeout=3.0,
        max_retries=2,
        retry_delay=1.0,
        exceptions=(asyncio.TimeoutError, ValueError),
    )
    async def unreliable_async_function(success_rate: float = 0.3):
        """Simulate an unreliable async function."""
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate async work
        if random.random() > success_rate:
            raise ValueError("Random async failure occurred")
        return "Async Success!"

    # Test sync function
    print("Testing sync function:")
    try:
        result = unreliable_sync_function(0.8)
        print(f"Sync result: {result}")
    except RetryError as e:
        print(f"Sync failed after retries: {e}")
    except TimeoutError as e:
        print(f"Sync timed out: {e}")

    # Test async function
    print("\nTesting async function:")

    async def test_async():
        try:
            result = await unreliable_async_function(0.8)
            print(f"Async result: {result}")
        except RetryError as e:
            print(f"Async failed after retries: {e}")
        except TimeoutError as e:
            print(f"Async timed out: {e}")

    asyncio.run(test_async())

    # Example 1: Database connection with changing connection parameters
    print("Example 1: Database connection with failover servers")

    def connect_to_database(host: str, port: int, database: str):
        """Simulate database connection that might fail."""
        print(f"Attempting to connect to {host}:{port}/{database}")
        # Simulate connection failure for certain hosts
        if host in ["primary-db.com", "backup-db.com"]:
            raise ConnectionError(f"Could not connect to {host}")
        return f"Connected to {host}:{port}/{database}"

    # List of fallback servers
    fallback_servers = [
        ("primary-db.com", 5432),
        ("backup-db.com", 5432),
        ("emergency-db.com", 5433),
        ("final-db.com", 5434),
    ]

    def database_retry_callback(
        attempt: int, exception: Exception, args: list, kwargs: dict
    ):
        """Update connection parameters on retry."""
        if attempt <= len(fallback_servers):
            new_host, new_port = fallback_servers[attempt - 1]
            # Modify kwargs for next attempt
            kwargs["host"] = new_host
            kwargs["port"] = new_port
            print(f"Retry {attempt}: Switching to {new_host}:{new_port}")

    @with_timeout_and_retry(
        max_retries=3,
        retry_delay=0.5,
        exceptions=ConnectionError,
        on_retry=database_retry_callback,
    )
    def connect_with_failover(**kwargs):
        return connect_to_database(**kwargs)

    try:
        result = connect_with_failover(
            host="primary-db.com", port=5432, database="myapp"
        )
        print(f"Database result: {result}")
    except Exception as e:
        print(f"Database connection failed: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: HTTP request with exponential backoff and jitter
    print("Example 2: HTTP request with dynamic timeout adjustment")

    def make_http_request(url: str, timeout: float = 5.0, headers: dict = None):
        """Simulate HTTP request that might fail."""
        print(f"Making request to {url} with timeout {timeout}s")
        if headers:
            print(f"Headers: {headers}")

        # Simulate different failure request
        if "slow-api.com" in url:
            time.sleep(timeout + 0.1)  # Always timeout
            raise TimeoutError("Request timed out")
        elif "rate-limited.com" in url:
            raise Exception("Rate limit exceeded")

        return f"Response from {url}"

    def http_retry_callback(
        attempt: int, exception: Exception, args: list, kwargs: dict
    ):
        """Adjust parameters based on the type of failure."""
        print(f"Retry {attempt}: {exception}")

        if "timeout" in str(exception).lower():
            # Increase timeout for next attempt
            kwargs["timeout"] = kwargs.get("timeout", 5.0) * 1.5
            print(f"Increasing timeout to {kwargs['timeout']}s")

        elif "rate limit" in str(exception).lower():
            # Add delay header and change URL path
            if "headers" not in kwargs:
                kwargs["headers"] = {}
            kwargs["headers"]["X-Retry-After"] = str(attempt * 30)

            # Try alternative endpoint
            if "url" in kwargs:
                kwargs["url"] = kwargs["url"].replace("/api/v1/", "/api/v2/")
                print(f"Switching to alternative endpoint: {kwargs['url']}")

    @with_timeout_and_retry(
        max_retries=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        exceptions=(TimeoutError, Exception),
        on_retry=http_retry_callback,
    )
    def make_resilient_request(**kwargs):
        return make_http_request(**kwargs)

    try:
        result = make_resilient_request(
            url="https://rate-limited.com/api/v1/data",
            timeout=1.0,
            headers={"User-Agent": "MyApp/1.0"},
        )
        print(f"HTTP result: {result}")
    except Exception as e:
        print(f"HTTP request failed: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 3: File processing with dynamic batch size
    print("Example 3: File processing with dynamic batch size")

    def process_file_batch(filename: str, batch_size: int = 100, offset: int = 0):
        """Simulate file processing that might fail with large batches."""
        print(f"Processing {filename}: batch_size={batch_size}, offset={offset}")

        if batch_size > 50:
            raise MemoryError(f"Batch size {batch_size} too large")

        # Simulate successful processing
        return f"Processed {batch_size} records from {filename} starting at {offset}"

    def file_processing_retry_callback(
        attempt: int, exception: Exception, args: list, kwargs: dict
    ):
        """Reduce batch size on memory errors."""
        if "memory" in str(exception).lower() or "batch size" in str(exception).lower():
            current_batch_size = kwargs.get("batch_size", 100)
            new_batch_size = max(10, current_batch_size // 2)
            kwargs["batch_size"] = new_batch_size
            print(f"Retry {attempt}: Reducing batch size to {new_batch_size}")

    @with_timeout_and_retry(
        max_retries=3,
        retry_delay=0.5,
        exceptions=(MemoryError, Exception),
        on_retry=file_processing_retry_callback,
    )
    def process_with_adaptive_batch_size(**kwargs):
        return process_file_batch(**kwargs)

    try:
        result = process_with_adaptive_batch_size(
            filename="large_dataset.csv", batch_size=200, offset=0
        )
        print(f"File processing result: {result}")
    except Exception as e:
        print(f"File processing failed: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 4: API call with credential rotation
    print("Example 4: API call with credential rotation")

    def call_api(endpoint: str, api_key: str, user_id: str = None):
        """Simulate API call that might fail due to invalid credentials."""
        print(f"Calling API {endpoint} with key {api_key[:8]}...")

        invalid_keys = ["key12345", "backup_key123", "old_key456"]
        if api_key in invalid_keys:
            raise PermissionError(f"Invalid API key: {api_key}")

        return f"API response from {endpoint}"

    # Available API keys (in priority order)
    api_keys = ["key12345", "backup_key123", "old_key456", "valid_key789"]

    def api_retry_callback(
        attempt: int, exception: Exception, args: list, kwargs: dict
    ):
        """Rotate API keys on authentication failure."""
        if (
            "invalid api key" in str(exception).lower()
            or "permission" in str(exception).lower()
        ):
            if attempt < len(api_keys):
                new_key = api_keys[attempt]
                kwargs["api_key"] = new_key
                print(f"Retry {attempt}: Rotating to API key {new_key[:8]}...")

    @with_timeout_and_retry(
        max_retries=len(api_keys) - 1,
        retry_delay=0.5,
        exceptions=PermissionError,
        on_retry=api_retry_callback,
    )
    def call_api_with_key_rotation(**kwargs):
        return call_api(**kwargs)

    try:
        result = call_api_with_key_rotation(
            endpoint="https://api.example.com/users",
            api_key="key12345",
            user_id="user123",
        )
        print(f"API result: {result}")
    except Exception as e:
        print(f"API call failed: {e}")

    # Example of parsing structured output
    print("=============== Testing structured output parsing ===============")
    text = """
    Here is some text with a JSON block:
    ```json
    {
        "name": "John Doe",
        "age": 30,
        "is_student": false
    }
    ```
    And some additional text.
    """

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_student": {"type": "boolean"},
        },
    }

    parser = StructuredOutputParser()
    parsed_output = parser.parse_structured_output(text, schema)
    print(parsed_output)
