from tenacity import retry, stop_after_attempt, wait_exponential
from functools import wraps
from core.gcp_client import get_retry_config

retry_config = get_retry_config()

def default_retry_decorator():
    """
    A default retry decorator using settings from the main_config.yaml.
    """
    return retry(
        stop=stop_after_attempt(retry_config.get('attempts', 3)),
        wait=wait_exponential(
            multiplier=retry_config.get('wait_multiplier', 1),
            min=retry_config.get('wait_initial', 1),
            max=retry_config.get('wait_max', 10)
        )
    )

def async_retry_decorator(
        attempts=retry_config.get('attempts', 3),
        wait_initial=retry_config.get('wait_initial', 1),
        wait_multiplier=retry_config.get('wait_multiplier', 2),
        wait_max=retry_config.get('wait_max', 10)):
    """
    A retry decorator for asynchronous functions.
    Uses tenacity's retry, stop_after_attempt, and wait_exponential.
    """
    def decorator(async_func):
        @wraps(async_func)
        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=wait_multiplier, min=wait_initial, max=wait_max)
        )
        async def wrapped_function(*args, **kwargs):
            return await async_func(*args, **kwargs)
        return wrapped_function
    return decorator