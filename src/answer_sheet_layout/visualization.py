"""OpenCV debug overlays using original-page coordinate space."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import PageResult


def render_overlay(image: NDArray[np.uint8], page: PageResult) -> NDArray[np.uint8]:
    """Draw regions in green, OCR lines in blue, and question candidates in orange."""
    output = image.copy()
    superseded = {
        region_id for diagram in page.diagram_regions
        for region_id in diagram.attributes.get("overlaps_baseline_region_ids", [])
    }
    for region in page.regions:
        if region.id in superseded:
            continue
        _draw(output, region.bounding_box.as_xyxy(), (0, 180, 0), f"{region.region_class.value}:{region.id}")
    for line in page.ocr_lines:
        _draw(output, line.bounding_box.as_xyxy(), (255, 0, 0), line.text[:40])
    for anchor in page.question_anchor_candidates:
        _draw(output, anchor.bounding_box.as_xyxy(), (0, 140, 255), f"CANDIDATE {anchor.text}")
    for diagram in page.diagram_regions:
        _draw(output, diagram.bounding_box.as_xyxy(), (180, 0, 180), f"DIAGRAM {diagram.source_detector}")
    return output


def render_comparison(image: NDArray[np.uint8], page: PageResult) -> NDArray[np.uint8]:
    """Return an original-left, annotated-right inspection image."""
    overlay = render_overlay(image, page)
    return np.concatenate((image, overlay), axis=1)


def _draw(image: NDArray[np.uint8], box: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    left, top, right, bottom = (round(value) for value in box)
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    cv2.putText(image, label, (left, max(14, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
