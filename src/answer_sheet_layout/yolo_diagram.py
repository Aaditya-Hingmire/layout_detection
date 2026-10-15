"""Optional Ultralytics YOLO adapter for a task-trained ``diagram_region`` model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import YoloDiagramConfig
from .detectors import DiagramDetection
from .models import BoundingBox, PageResult, Region, RegionClass

logger = logging.getLogger(__name__)


class YoloDiagramDetector:
    """YOLO adapter that intentionally has no generic pretrained default.

    ``trained_for_diagrams`` is an auditable caller declaration, not a claim made
    by this library. The checkpoint/version remain attached to every region.
    """

    name = "ultralytics_yolo_diagram"

    def __init__(self, config: YoloDiagramConfig, model: Any | None = None) -> None:
        self.config = config
        self._model = model

    def detect_diagrams(self, image: NDArray[np.uint8], page: PageResult) -> DiagramDetection:
        if not self.config.enabled:
            return DiagramDetection(warnings=["yolo_diagram_detector_disabled"])
        try:
            results = self._get_model().predict(
                source=image, conf=self.config.confidence_threshold, iou=self.config.iou_threshold, verbose=False,
            )
            if not results:
                return DiagramDetection(warnings=["yolo_diagram_detector_returned_no_results"])
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                return DiagramDetection(warnings=["yolo_diagram_detector_returned_no_boxes"])
            names = getattr(result, "names", {})
            regions: list[Region] = []
            for index, (xyxy, cls, score) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
                class_id = int(cls.item())
                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id))
                elif isinstance(names, (list, tuple)) and class_id < len(names):
                    class_name = str(names[class_id])
                else:
                    class_name = str(class_id)
                if class_name != "diagram_region":
                    continue
                processed = BoundingBox.from_xyxy(*(float(value) for value in xyxy.tolist()))
                original = page.transform_from_original.map_box_to_original(processed)
                regions.append(Region(
                    id=f"{page.page_id}-yolo-diagram-{index + 1}", page_id=page.page_id,
                    region_class=RegionClass.DIAGRAM, bounding_box=original, source_detector=self.name,
                    confidence=float(score.item()), attributes={
                        "model_version": self.config.model_version,
                        "model_path": str(self.config.model_path),
                        "model_trained_for_diagrams": True,
                        "class_name": class_name,
                    },
                ))
            return DiagramDetection(regions)
        except Exception as error:
            logger.warning("YOLO diagram detection failed for %s: %s", page.page_id, error)
            return DiagramDetection(warnings=[f"{self.name}_failed: {type(error).__name__}: {error}"])

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        assert self.config.model_path is not None
        if not Path(self.config.model_path).is_file():
            raise FileNotFoundError(f"diagram checkpoint not found: {self.config.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Ultralytics is unavailable; install answer-sheet-layout[yolo]") from error
        self._model = YOLO(str(self.config.model_path))
        return self._model
