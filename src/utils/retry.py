"""Retry utilities with exponential backoff"""

import asyncio
import time
from functools import wraps
from typing import Callable, Any, Optional, Type, Union, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    max_wait_time: float = 60.0,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        max_wait_time: Maximum wait time between retries
        retry_exceptions: Tuple of exception types that should trigger a retry
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=max_wait_time),
        retry=retry_if_exception_type(retry_exceptions),
        reraise=True
    )


class RetryableClient:
    """Base class for clients that need retry functionality"""
    
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 2.0, max_wait_time: float = 60.0):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.max_wait_time = max_wait_time
    
    def _get_retry_decorator(self, retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)):
        """Get a retry decorator with the configured parameters"""
        return retry_with_backoff(
            max_attempts=self.max_attempts,
            backoff_factor=self.backoff_factor,
            max_wait_time=self.max_wait_time,
            retry_exceptions=retry_exceptions
        )


def async_retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    max_wait_time: float = 60.0,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Async version of retry decorator
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            wait_time = 1.0
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)
                        wait_time = min(wait_time * backoff_factor, max_wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
                        raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator
