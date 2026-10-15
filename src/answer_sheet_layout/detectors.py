"""Stable extension interfaces; concrete OCR/layout detectors are intentionally absent."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .models import OcrLine, PageResult, QuestionAnchorCandidate, Region


class RegionDetector(Protocol):
    """A replaceable page-level detector consuming a processed page image."""

    name: str

    def detect(self, image: NDArray[np.uint8], page: PageResult) -> list[Region]: ...


class PageAnalysisDetector(Protocol):
    """A detector capable of returning layout and OCR evidence for one page."""

    name: str

    def analyze(self, image: NDArray[np.uint8], page: PageResult) -> "PageAnalysis": ...


class DiagramDetector(Protocol):
    """Replaceable visual-region detector; coordinates must be page-image pixels."""

    name: str

    def detect_diagrams(self, image: NDArray[np.uint8], page: PageResult) -> "DiagramDetection": ...


class PageAnalysis:
    """Normalized detector result; failures are evidence, not silently dropped."""

    def __init__(
        self,
        regions: list[Region] | None = None,
        ocr_lines: list[OcrLine] | None = None,
        question_anchor_candidates: list[QuestionAnchorCandidate] | None = None,
        raw_result: object | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.regions = regions or []
        self.ocr_lines = ocr_lines or []
        self.question_anchor_candidates = question_anchor_candidates or []
        self.raw_result = raw_result
        self.warnings = warnings or []


class DiagramDetection:
    def __init__(self, regions: list[Region] | None = None, warnings: list[str] | None = None) -> None:
        self.regions = regions or []
        self.warnings = warnings or []


class CropStore(Protocol):
    """Optional artifact store used by later detectors for durable region crops."""

    def save_crop(self, region: Region, image: NDArray[np.uint8]) -> Path: ...
