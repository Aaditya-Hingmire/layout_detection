"""Configuration for rendering and conservative page preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Pattern
import re
from pathlib import Path


@dataclass(frozen=True)
class RenderingConfig:
    dpi: int = 200
    max_pages: int = 500
    max_pixels_per_page: int = 40_000_000

    def __post_init__(self) -> None:
        if self.dpi < 72 or self.dpi > 600:
            raise ValueError("dpi must be between 72 and 600")
        if self.max_pages < 1 or self.max_pixels_per_page < 1:
            raise ValueError("rendering limits must be positive")


@dataclass(frozen=True)
class PreprocessingConfig:
    deskew: bool = False
    enhance_contrast: bool = False
    denoise: bool = False
    # Ruled lines are intentionally retained. Their removal belongs to a later,
    # data-validated preprocessing option rather than the default pipeline.
    preserve_ruled_lines: bool = True
    blur_threshold: float = 80.0
    low_contrast_threshold: float = 20.0


@dataclass(frozen=True)
class PaddleConfig:
    """Runtime options for the optional PP-StructureV3 adapter."""

    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    device: str | None = None


@dataclass(frozen=True)
class QuestionAnchorConfig:
    """Conservative OCR-text rules; matches are candidates, never associations."""

    patterns: tuple[str, ...] = (
        r"^\s*(?:q(?:uestion)?\s*\.?\s*\d+[a-z]?|\d+\s*\([a-z]\))\s*[.:)]?\s*$",
    )
    min_confidence: float = 0.0

    def compiled_patterns(self) -> tuple[Pattern[str], ...]:
        return tuple(re.compile(pattern, re.IGNORECASE) for pattern in self.patterns)


@dataclass(frozen=True)
class YoloDiagramConfig:
    """Explicit opt-in configuration for a task-trained one-class checkpoint."""

    enabled: bool = False
    model_path: Path | None = None
    model_version: str | None = None
    trained_for_diagrams: bool = False
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1 or not 0 <= self.iou_threshold <= 1:
            raise ValueError("YOLO confidence and IoU thresholds must be between 0 and 1")
        if self.enabled and (self.model_path is None or not self.model_version or not self.trained_for_diagrams):
            raise ValueError("enabled YOLO requires a diagram-trained model_path and model_version")


@dataclass(frozen=True)
class AssociationConfig:
    """Deterministic, intentionally conservative question-region association rules."""

    continuation_confidence: float = 0.55
    same_page_confidence: float = 0.85
    diagram_same_segment_confidence: float = 0.70


@dataclass(frozen=True)
class LayoutConfig:
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    paddle: PaddleConfig = field(default_factory=PaddleConfig)
    question_anchors: QuestionAnchorConfig = field(default_factory=QuestionAnchorConfig)
    yolo_diagrams: YoloDiagramConfig = field(default_factory=YoloDiagramConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
