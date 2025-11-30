"""
Utility functions for LiteAds.
"""

import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any, TypeVar

import orjson

T = TypeVar("T")


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def generate_id() -> int:
    """Generate a unique integer ID using timestamp and random bits."""
    timestamp = int(time.time() * 1000)
    random_bits = uuid.uuid4().int & 0xFFFF
    return (timestamp << 16) | random_bits


def current_timestamp() -> int:
    """Get current Unix timestamp in seconds."""
    return int(time.time())


def current_timestamp_ms() -> int:
    """Get current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


def current_datetime() -> datetime:
    """Get current datetime in UTC."""
    return datetime.now(timezone.utc)


def current_hour() -> str:
    """Get current hour as string (YYYYMMDDHH)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H")


def current_date() -> str:
    """Get current date as string (YYYYMMDD)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def hash_string(s: str) -> str:
    """Hash a string using MD5."""
    return hashlib.md5(s.encode()).hexdigest()


def hash_user_id(user_id: str) -> int:
    """Hash user ID to integer for consistent bucketing."""
    return int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)


def json_dumps(obj: Any) -> str:
    """Fast JSON serialization using orjson."""
    return orjson.dumps(obj).decode("utf-8")


def json_loads(s: str | bytes) -> Any:
    """Fast JSON deserialization using orjson."""
    return orjson.loads(s)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero division."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    import math

    return 1 / (1 + math.exp(-x))


def chunks(lst: list[T], n: int) -> list[list[T]]:
    """Split a list into chunks of size n."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def flatten(nested: list[list[T]]) -> list[T]:
    """Flatten a nested list."""
    return [item for sublist in nested for item in sublist]


def dedupe(lst: list[T]) -> list[T]:
    """Remove duplicates while preserving order."""
    seen: set[Any] = set()
    result: list[T] = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time


def retry(
    max_attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """
    Decorator for retrying a function on failure.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay on each retry.
        exceptions: Tuple of exceptions to catch.
    """
    import functools

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception  # type: ignore

        return wrapper

    return decorator
