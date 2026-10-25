"""User-facing error definitions for bot handlers."""


class UserFacingError(Exception):
    """Exception carrying a safe, sanitized message intended to be shown directly to users."""

    pass
