"""Custom exceptions for the Grounded RAG project."""


class GroundedRAGError(Exception):
    """Base exception for project-specific failures."""


class ConfigurationError(GroundedRAGError):
    """Raised when required settings are missing or invalid."""


class IngestionError(GroundedRAGError):
    """Raised when document ingestion fails."""


class IndexNotReadyError(GroundedRAGError):
    """Raised when a persisted index cannot be loaded."""


class GenerationError(GroundedRAGError):
    """Raised when answer generation fails."""

