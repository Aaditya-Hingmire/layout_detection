"""Optional PP-StructureV3 adapter with defensive result normalization."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .config import PaddleConfig, QuestionAnchorConfig
from .detectors import PageAnalysis
from .models import BoundingBox, OcrLine, PageResult, QuestionAnchorCandidate, Region, RegionClass

logger = logging.getLogger(__name__)


class PaddlePPStructureV3Adapter:
    """Adapter for PaddleOCR >=3's documented ``PPStructureV3.predict`` API.

    The pipeline is created only on first use so importing this package does not
    require PaddleOCR. Tests and alternate deployments may inject a compatible
    pipeline object exposing ``predict(input=...)``.
    """

    name = "paddle_pp_structure_v3"

    def __init__(
        self,
        config: PaddleConfig | None = None,
        question_config: QuestionAnchorConfig | None = None,
        pipeline: Any | None = None,
    ) -> None:
        self.config = config or PaddleConfig()
        self.question_config = question_config or QuestionAnchorConfig()
        self._pipeline = pipeline

    def analyze(self, image: NDArray[np.uint8], page: PageResult) -> PageAnalysis:
        try:
            result_object = next(iter(self._get_pipeline().predict(input=image)))
            raw = _result_to_json(result_object)
            regions = self._extract_regions(raw, page)
            lines = self._extract_ocr_lines(raw, page)
            anchors = _find_question_anchors(lines, self.question_config)
            warnings: list[str] = []
            if not regions and not lines:
                warnings.append("paddle_output_contains_no_supported_layout_or_ocr_evidence")
            return PageAnalysis(regions, lines, anchors, raw, warnings)
        except Exception as error:  # external model/model-output failures must not abort a PDF run
            logger.warning("Paddle analysis failed for %s: %s", page.page_id, error)
            return PageAnalysis(raw_result=None, warnings=[f"{self.name}_failed: {type(error).__name__}: {error}"])

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from paddleocr import PPStructureV3
        except ImportError as error:
            raise RuntimeError("PaddleOCR is unavailable; install answer-sheet-layout[paddle]") from error
        kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": self.config.use_doc_orientation_classify,
            "use_doc_unwarping": self.config.use_doc_unwarping,
            "use_textline_orientation": self.config.use_textline_orientation,
        }
        if self.config.device is not None:
            kwargs["device"] = self.config.device
        self._pipeline = PPStructureV3(**kwargs)
        return self._pipeline

    def _extract_regions(self, raw: dict[str, Any], page: PageResult) -> list[Region]:
        result = _unwrap(raw)
        layout = _first_mapping(result, "layout_det_res", "layout_result", "layout_res") or result
        boxes = layout.get("boxes") or layout.get("layout_boxes") or []
        labels = layout.get("labels") or layout.get("layout_labels") or []
        scores = layout.get("scores") or layout.get("layout_scores") or []
        regions: list[Region] = []
        for index, box in enumerate(boxes):
            normalized = _box_from_value(box)
            if normalized is None:
                continue
            original_box = page.transform_from_original.map_box_to_original(normalized)
            label = str(_field_from_box(box, "label", str(labels[index]) if index < len(labels) else "unknown"))
            confidence = _confidence(_field_from_box(box, "score", scores[index] if index < len(scores) else None))
            reading_order = _field_from_box(box, "reading_order", index)
            regions.append(Region(
                id=f"{page.page_id}-region-{index + 1}", page_id=page.page_id,
                region_class=_region_class(label), bounding_box=original_box,
                source_detector=self.name, confidence=confidence,
                attributes={"raw_label": label, "reading_order": reading_order},
            ))
        return regions

    def _extract_ocr_lines(self, raw: dict[str, Any], page: PageResult) -> list[OcrLine]:
        result = _unwrap(raw)
        containers = list(_mappings_with_keys(result, {"rec_texts", "rec_polys"}))
        if not containers and isinstance(result, dict):
            containers = [result]
        lines: list[OcrLine] = []
        seen: set[tuple[str, tuple[float, float, float, float]]] = set()
        for container in containers:
            texts = container.get("rec_texts") or []
            polygons = container.get("rec_polys") or container.get("dt_polys") or []
            scores = container.get("rec_scores") or container.get("dt_scores") or []
            for index, text in enumerate(texts):
                if index >= len(polygons) or not str(text).strip():
                    continue
                polygon = _polygon_from_value(polygons[index])
                if not polygon:
                    continue
                processed_box = _box_from_polygon(polygon)
                box = page.transform_from_original.map_box_to_original(processed_box)
                key = (str(text), box.as_xyxy())
                if key in seen:
                    continue
                seen.add(key)
                original_polygon = tuple(page.transform_from_original.map_point_to_original(x, y) for x, y in polygon)
                source_order = container.get("reading_order") or container.get("reading_orders") or []
                if not isinstance(source_order, (list, tuple)):
                    source_order = []
                reading_order = source_order[index] if index < len(source_order) else len(lines)
                lines.append(OcrLine(
                    id=f"{page.page_id}-ocr-{len(lines) + 1}", page_id=page.page_id, text=str(text),
                    bounding_box=box, polygon=original_polygon, source_detector=self.name,
                    confidence=_confidence(scores[index]) if index < len(scores) else None,
                    reading_order=int(reading_order) if isinstance(reading_order, (int, float)) else len(lines),
                ))
        return lines


def _result_to_json(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if not isinstance(value, dict):
        raise TypeError("Paddle result did not expose a JSON mapping")
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _unwrap(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("res", raw)
    return value if isinstance(value, dict) else raw


def _first_mapping(value: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, dict):
            return candidate
    return None


def _mappings_with_keys(value: Any, keys: set[str]) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if keys.issubset(value.keys()):
            yield value
        for child in value.values():
            yield from _mappings_with_keys(child, keys)
    elif isinstance(value, list):
        for child in value:
            yield from _mappings_with_keys(child, keys)


def _polygon_from_value(value: Any) -> tuple[tuple[float, float], ...]:
    try:
        points = tuple((float(point[0]), float(point[1])) for point in value)
    except (TypeError, ValueError, IndexError):
        return ()
    return points if len(points) >= 3 else ()


def _box_from_polygon(polygon: tuple[tuple[float, float], ...]) -> BoundingBox:
    xs, ys = zip(*polygon)
    return BoundingBox.from_xyxy(min(xs), min(ys), max(xs), max(ys))


def _box_from_value(value: Any) -> BoundingBox | None:
    if isinstance(value, dict):
        value = value.get("coordinate") or value.get("bbox") or value.get("box")
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4 and not isinstance(value[0], (list, tuple)):
        left, top, right, bottom = map(float, value)
        return BoundingBox.from_xyxy(left, top, right, bottom)
    polygon = _polygon_from_value(value)
    return _box_from_polygon(polygon) if polygon else None


def _field_from_box(box: Any, key: str, fallback: Any) -> Any:
    return box.get(key, fallback) if isinstance(box, dict) else fallback


def _confidence(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 1.0 else None


def _region_class(label: str) -> RegionClass:
    token = label.lower().replace(" ", "_")
    mapping = {"text": RegionClass.PRINTED_TEXT, "table": RegionClass.TABLE, "formula": RegionClass.EQUATION,
               "image": RegionClass.IMAGE, "figure": RegionClass.IMAGE, "header": RegionClass.HEADER,
               "footer": RegionClass.FOOTER}
    return mapping.get(token, RegionClass.UNKNOWN)


def _find_question_anchors(lines: list[OcrLine], config: QuestionAnchorConfig) -> list[QuestionAnchorCandidate]:
    patterns = config.compiled_patterns()
    anchors: list[QuestionAnchorCandidate] = []
    for line in lines:
        if line.confidence is not None and line.confidence < config.min_confidence:
            continue
        for pattern in patterns:
            if pattern.fullmatch(line.text):
                warnings = () if line.confidence is not None else ("ocr_confidence_unavailable",)
                anchors.append(QuestionAnchorCandidate(
                    id=f"{line.page_id}-anchor-{len(anchors) + 1}", page_id=line.page_id,
                    ocr_line_id=line.id, text=line.text, bounding_box=line.bounding_box,
                    rule=pattern.pattern, confidence=line.confidence, warnings=warnings,
                ))
                break
    return anchors
