"""Errors intentionally exposed by the module."""


class LayoutDetectionError(Exception):
    """Base class for errors raised by this package."""


class InputValidationError(LayoutDetectionError):
    """The supplied PDF is unsupported, unsafe, or cannot be opened."""


class RenderingError(LayoutDetectionError):
    """A validated PDF could not be rendered."""


class ModelUnavailableError(LayoutDetectionError):
    """A requested optional detector runtime or checkpoint is unavailable."""


class ProcessingError(LayoutDetectionError):
    """A detector or processing stage failed in strict processing mode."""


class ResultValidationError(LayoutDetectionError):
    """A result violates the public result contract."""
