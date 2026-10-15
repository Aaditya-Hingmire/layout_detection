"""Public API for the answer-sheet layout detection foundation."""

from .analysis import analyze_pdf
from .api import ProcessOptions, process_pdf
from .association import associate_question_regions
from .benchmark import evaluate_benchmark
from .config import AssociationConfig, LayoutConfig, PaddleConfig, PreprocessingConfig, QuestionAnchorConfig, RenderingConfig, YoloDiagramConfig
from .errors import InputValidationError, LayoutDetectionError, ModelUnavailableError, ProcessingError, RenderingError, ResultValidationError
from .models import (BoundingBox, DocumentResult, OcrLine, PageResult, QuestionAnchorCandidate, Region,
                     RegionClass, TransformOperation)
from .rendering import render_pdf
from .serialization import serialize_document_result, validate_document_result, write_document_result
from .yolo_diagram import YoloDiagramDetector

__all__ = [
    "BoundingBox",
    "AssociationConfig",
    "DocumentResult",
    "InputValidationError",
    "LayoutConfig",
    "LayoutDetectionError",
    "ModelUnavailableError",
    "OcrLine",
    "PaddleConfig",
    "PageResult",
    "PreprocessingConfig",
    "ProcessOptions",
    "ProcessingError",
    "QuestionAnchorCandidate",
    "QuestionAnchorConfig",
    "Region",
    "RegionClass",
    "RenderingConfig",
    "RenderingError",
    "ResultValidationError",
    "TransformOperation",
    "YoloDiagramConfig",
    "YoloDiagramDetector",
    "analyze_pdf",
    "associate_question_regions",
    "evaluate_benchmark",
    "process_pdf",
    "render_pdf",
    "serialize_document_result",
    "validate_document_result",
    "write_document_result",
]
