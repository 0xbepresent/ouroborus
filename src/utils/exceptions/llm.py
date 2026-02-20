"""LLM-related exceptions."""

from src.utils.exceptions.base import OuroborusError


class LLMError(OuroborusError):
    """Base class for all LLM-related errors."""
    pass


class LLMConfigError(LLMError):
    """LLM configuration errors (missing keys, invalid provider, etc.)."""
    pass


class LLMApiError(LLMError):
    """LLM API call failures (timeouts, rate limits, 5xx, etc.)."""
    pass


