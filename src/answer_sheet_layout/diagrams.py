"""Diagram-region de-duplication, crops, and baseline/YOLO comparison reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import BoundingBox, DocumentResult, PageResult, Region, RegionClass


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def merge_diagram_regions(page: PageResult, yolo_regions: list[Region], iou_threshold: float) -> list[Region]:
    """Create one canonical diagram list, preferring a YOLO box on overlap.

    Baseline layout regions remain in ``page.regions`` for provenance, while
    ``page.diagram_regions`` is the de-duplicated diagram-specific output.
    """
    baseline = [region for region in page.regions if region.region_class in {RegionClass.DIAGRAM, RegionClass.IMAGE}]
    canonical: list[Region] = list(baseline)
    for yolo in yolo_regions:
        overlaps = [region for region in canonical if intersection_over_union(region.bounding_box, yolo.bounding_box) >= iou_threshold]
        if overlaps:
            for region in overlaps:
                canonical.remove(region)
            yolo.attributes["overlaps_baseline_region_ids"] = [region.id for region in overlaps]
        canonical.append(yolo)
    page.diagram_regions = canonical
    return canonical


def save_diagram_crops(image: NDArray[np.uint8], page: PageResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    saved: list[Region] = []
    for region in page.diagram_regions:
        x1, y1, x2, y2 = region.bounding_box.as_xyxy()
        # Context preserves nearby labels/legend text without changing the detected box.
        margin = 16
        left, top = max(0, int(np.floor(x1)) - margin), max(0, int(np.floor(y1)) - margin)
        right, bottom = min(width, int(np.ceil(x2)) + margin), min(height, int(np.ceil(y2)) + margin)
        if right <= left or bottom <= top:
            page.detector_warnings.append(f"invalid_diagram_crop: {region.id}")
            saved.append(region)
            continue
        crop_path = output_dir / f"{region.id}.png"
        if cv2.imwrite(str(crop_path), image[top:bottom, left:right]):
            saved.append(replace(region, crop_reference=str(crop_path)))
        else:
            page.detector_warnings.append(f"diagram_crop_write_failed: {region.id}")
            saved.append(region)
    page.diagram_regions = saved


def attach_diagram_context(page: PageResult, margin_px: float = 32.0) -> None:
    """Attach nearby OCR evidence IDs; no semantic interpretation is performed."""
    for region in page.diagram_regions:
        expanded = BoundingBox(region.bounding_box.x - margin_px, region.bounding_box.y - margin_px,
                               region.bounding_box.width + margin_px * 2, region.bounding_box.height + margin_px * 2)
        region.attributes["nearby_ocr_line_ids"] = [
            line.id for line in page.ocr_lines if intersection_over_union(expanded, line.bounding_box) > 0
        ]


def write_comparison_report(result: DocumentResult, output_path: Path, yolo_enabled: bool) -> dict:
    """Report observed branch counts; accuracy metrics remain unavailable without labels."""
    pages: list[dict] = []
    total_baseline = total_yolo = total_canonical = 0
    for page in result.pages:
        baseline = [region for region in page.regions if region.region_class in {RegionClass.DIAGRAM, RegionClass.IMAGE}]
        yolo = [region for region in page.diagram_regions if region.source_detector == "ultralytics_yolo_diagram"]
        total_baseline += len(baseline)
        total_yolo += len(yolo)
        total_canonical += len(page.diagram_regions)
        pages.append({"page_id": page.page_id, "baseline_visual_regions": len(baseline),
                      "yolo_diagram_regions": len(yolo), "canonical_diagram_regions": len(page.diagram_regions),
                      "warnings": page.detector_warnings})
    report = {
        "report_type": "baseline_vs_yolo_detection_comparison",
        "yolo_enabled": yolo_enabled,
        "measured_counts": {"baseline_visual_regions": total_baseline, "yolo_diagram_regions": total_yolo,
                            "canonical_diagram_regions": total_canonical},
        "accuracy_metrics": {"precision": None, "recall": None, "map50": None, "map50_95": None,
                             "false_negative_examples": [], "incorrect_detection_examples": [],
                             "reason": "Ground-truth labels and a held-out evaluation split are required."},
        "pages": pages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
