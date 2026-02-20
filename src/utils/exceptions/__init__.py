"""Ouroborus exception hierarchy."""

from src.utils.exceptions.base import OuroborusError
from src.utils.exceptions.codeql import (
    CodeQLError,
    CodeQLConfigError,
    CodeQLExecutionError,
)
from src.utils.exceptions.llm import (
    LLMError,
    LLMConfigError,
    LLMApiError,
)

__all__ = [
    "OuroborusError",
    "CodeQLError",
    "CodeQLConfigError",
    "CodeQLExecutionError",
    "LLMError",
    "LLMConfigError",
    "LLMApiError",
]


