"""Versioned, JSON-friendly schemas shared by detectors and callers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"


class RegionClass(str, Enum):
    UNKNOWN = "unknown"
    HANDWRITTEN_TEXT = "handwritten_text"
    PRINTED_TEXT = "printed_text"
    QUESTION_LABEL = "question_label"
    EQUATION = "equation"
    TABLE = "table"
    DIAGRAM = "diagram"
    HEADER = "header"
    FOOTER = "footer"
    IMAGE = "image"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space rectangle using a top-left origin and exclusive right/bottom edges."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("bounding-box width and height cannot be negative")

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @classmethod
    def from_xyxy(cls, left: float, top: float, right: float, bottom: float) -> "BoundingBox":
        return cls(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class TransformOperation:
    """One recorded image operation, including non-geometric preprocessing."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AffineTransform:
    """A homogeneous 3x3 matrix mapping original render coordinates to current coordinates."""

    name: str
    matrix: tuple[tuple[float, float, float], ...]
    operations: tuple[TransformOperation, ...] = ()

    @classmethod
    def identity(cls) -> "AffineTransform":
        return cls("identity", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    def as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def map_point_to_original(self, x: float, y: float) -> tuple[float, float]:
        point = np.array([x, y, 1.0], dtype=np.float64)
        mapped = np.linalg.inv(self.as_array()) @ point
        return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))

    def map_box_to_original(self, box: BoundingBox) -> BoundingBox:
        points = [self.map_point_to_original(x, y) for x, y in (
            (box.x, box.y), (box.x + box.width, box.y),
            (box.x, box.y + box.height), (box.x, box.y + box.height),
        )]
        xs, ys = zip(*points)
        return BoundingBox.from_xyxy(min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class PageQuality:
    blur_variance: float
    contrast_stddev: float
    estimated_skew_degrees: float | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Region:
    id: str
    page_id: str
    region_class: RegionClass
    bounding_box: BoundingBox
    source_detector: str
    confidence: float | None = None
    crop_reference: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class OcrLine:
    """Raw OCR text plus the detector polygon and its enclosing page-space box."""

    id: str
    page_id: str
    text: str
    bounding_box: BoundingBox
    polygon: tuple[tuple[float, float], ...]
    source_detector: str
    confidence: float | None = None
    reading_order: int | None = None
    raw_reference: str | None = None


@dataclass(frozen=True)
class QuestionAnchorCandidate:
    """A text-pattern match only; it deliberately contains no answer association."""

    id: str
    page_id: str
    ocr_line_id: str
    text: str
    bounding_box: BoundingBox
    rule: str
    confidence: float | None
    status: str = "candidate"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssociatedContent:
    """Reference to one original detector item, with deterministic association evidence."""

    content_id: str
    content_type: str
    page_id: str
    bounding_box: BoundingBox
    source_detector: str
    crop_reference: str | None
    detector_confidence: float | None
    association_confidence: float
    evidence: tuple[str, ...]


@dataclass
class QuestionWiseLayout:
    """A candidate question grouping; it does not confirm question semantics."""

    candidate_question_id: str
    anchor_evidence: QuestionAnchorCandidate
    associated_content: list[AssociatedContent] = field(default_factory=list)
    page_ids: list[str] = field(default_factory=list)
    confidence_indicator: str = "candidate"
    continuation_from_previous_page: bool = False
    ambiguities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UnresolvedContent:
    content: AssociatedContent
    status: str
    reason: str
    plausible_candidate_question_ids: tuple[str, ...] = ()


@dataclass
class QuestionAssociationResult:
    question_layouts: list[QuestionWiseLayout] = field(default_factory=list)
    unresolved_content: list[UnresolvedContent] = field(default_factory=list)
    total_content_items: int = 0
    accounted_content_items: int = 0
    unaccounted_content_ids: list[str] = field(default_factory=list)


@dataclass
class PageResult:
    page_id: str
    page_index: int
    width_px: int
    height_px: int
    original_image_path: Path
    processed_image_path: Path | None
    transform_from_original: AffineTransform
    quality: PageQuality
    regions: list[Region] = field(default_factory=list)
    ocr_lines: list[OcrLine] = field(default_factory=list)
    question_anchor_candidates: list[QuestionAnchorCandidate] = field(default_factory=list)
    raw_result_paths: list[Path] = field(default_factory=list)
    detector_warnings: list[str] = field(default_factory=list)
    diagram_regions: list[Region] = field(default_factory=list)


@dataclass
class DocumentResult:
    source_pdf: Path
    pages: list[PageResult]
    schema_version: str = SCHEMA_VERSION
    question_association: QuestionAssociationResult | None = None
    processing_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))
