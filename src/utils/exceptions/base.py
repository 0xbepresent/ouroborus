"""Base exception class for all Ouroborus-specific errors."""


class OuroborusError(Exception):
    """
    Base exception for all Ouroborus-specific errors.

    Args:
        message: Human-readable error message.
        cause: Optional underlying exception that caused this error.
    """
    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause
        if cause is not None:
            # Enables chained traceback: OuroborusError <- cause
            self.__cause__ = cause

