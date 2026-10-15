"""Stable public entry point for integration with another Python application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analysis import analyze_pdf
from .config import LayoutConfig
from .detectors import DiagramDetector, PageAnalysis, PageAnalysisDetector
from .errors import ModelUnavailableError, ProcessingError
from .models import DocumentResult
from .paddle_adapter import PaddlePPStructureV3Adapter
from .serialization import validate_document_result, write_document_result
from .yolo_diagram import YoloDiagramDetector


@dataclass(frozen=True)
class ProcessOptions:
    """I/O and optional-detector policy for :func:`process_pdf`."""

    output_dir: Path
    use_paddle_layout: bool = True
    export_crops: bool = True
    export_visualizations: bool = True
    strict_detector_failures: bool = True
    write_result_json: bool = True


class _NoOpDetector:
    name = "no_layout_detector"

    def analyze(self, image, page) -> PageAnalysis:
        return PageAnalysis()


def process_pdf(
    pdf_path: str | Path,
    options: ProcessOptions,
    config: LayoutConfig | None = None,
    *,
    layout_detector: PageAnalysisDetector | None = None,
    diagram_detector: DiagramDetector | None = None,
) -> DocumentResult:
    """Process one PDF and return a validated, versioned :class:`DocumentResult`.

    Supplying adapters makes OCR, layout, and diagram implementations replaceable.
    With the default options, PP-StructureV3 is used. Set ``use_paddle_layout``
    false to run rendering/preprocessing/association without an OCR/layout model.
    """
    config = config or LayoutConfig()
    detector = layout_detector
    if detector is None:
        detector = PaddlePPStructureV3Adapter(config.paddle, config.question_anchors) if options.use_paddle_layout else _NoOpDetector()
    diagrams = diagram_detector
    if diagrams is None and config.yolo_diagrams.enabled:
        diagrams = YoloDiagramDetector(config.yolo_diagrams)
    result = analyze_pdf(
        pdf_path, options.output_dir, detector, config,
        create_visualizations=options.export_visualizations,
        diagram_detector=diagrams, export_crops=options.export_crops,
    )
    _raise_for_detector_failures(result, options)
    validate_document_result(result)
    if options.write_result_json:
        write_document_result(result, Path(options.output_dir) / "result.json")
    return result


def _raise_for_detector_failures(result: DocumentResult, options: ProcessOptions) -> None:
    if not options.strict_detector_failures:
        return
    warnings = [warning for page in result.pages for warning in page.detector_warnings]
    unavailable = [warning for warning in warnings if "unavailable" in warning.lower() or "not found" in warning.lower()]
    if unavailable:
        raise ModelUnavailableError("; ".join(unavailable))
    failed = [warning for warning in warnings if "_failed:" in warning]
    if failed:
        raise ProcessingError("; ".join(failed))
