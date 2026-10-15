"""Orchestration for baseline layout/OCR analysis and durable debug artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2

from .detectors import PageAnalysisDetector
from .detectors import DiagramDetector
from .diagrams import attach_diagram_context, merge_diagram_regions, save_diagram_crops, write_comparison_report
from .association import associate_question_regions
from .models import DocumentResult
from .paddle_adapter import _json_safe
from .rendering import render_pdf
from .visualization import render_comparison, render_overlay


def analyze_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    detector: PageAnalysisDetector,
    config=None,
    create_visualizations: bool = True,
    diagram_detector: DiagramDetector | None = None,
    export_crops: bool = True,
) -> DocumentResult:
    """Render then analyze every page, preserving raw response and failure evidence."""
    started = perf_counter()
    result = render_pdf(pdf_path, output_dir, config)
    root = Path(output_dir).resolve()
    raw_dir = root / "raw" / detector.name
    visual_dir = root / "visualizations"
    for page in result.pages:
        image_path = page.processed_image_path or page.original_image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            page.detector_warnings.append(f"analysis_image_unreadable: {image_path}")
            continue
        analysis = detector.analyze(image, page)
        page.regions.extend(analysis.regions)
        page.ocr_lines.extend(analysis.ocr_lines)
        page.question_anchor_candidates.extend(analysis.question_anchor_candidates)
        page.detector_warnings.extend(analysis.warnings)
        yolo_regions = []
        if diagram_detector is not None:
            diagrams = diagram_detector.detect_diagrams(image, page)
            yolo_regions = diagrams.regions
            page.detector_warnings.extend(diagrams.warnings)
        merge_diagram_regions(page, yolo_regions, getattr(getattr(config, "yolo_diagrams", None), "iou_threshold", 0.5))
        attach_diagram_context(page)
        original = cv2.imread(str(page.original_image_path), cv2.IMREAD_COLOR)
        if original is not None and export_crops:
            save_diagram_crops(original, page, root / "crops" / "diagrams" / page.page_id)
        if analysis.raw_result is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{page.page_id}.json"
            raw_path.write_text(json.dumps(_json_safe(analysis.raw_result), indent=2), encoding="utf-8")
            page.raw_result_paths.append(raw_path)
        if create_visualizations and original is not None:
            visual_dir.mkdir(parents=True, exist_ok=True)
            overlay = render_overlay(original, page)
            cv2.imwrite(str(visual_dir / f"{page.page_id}.png"), overlay)
            cv2.imwrite(str(visual_dir / f"{page.page_id}-comparison.png"), render_comparison(original, page))
    associate_question_regions(result, getattr(config, "association", None))
    write_comparison_report(result, root / "diagram-comparison.json", diagram_detector is not None)
    result.processing_metadata = {
        "wall_time_seconds": perf_counter() - started,
        "page_count": len(result.pages),
        "configuration": asdict(config) if config is not None else {},
        "diagram_detector_enabled": diagram_detector is not None,
    }
    return result
