from pathlib import Path

import numpy as np

from answer_sheet_layout.config import YoloDiagramConfig
from answer_sheet_layout.diagrams import merge_diagram_regions, save_diagram_crops
from answer_sheet_layout.models import AffineTransform, BoundingBox, PageQuality, PageResult, Region, RegionClass
from answer_sheet_layout.yolo_diagram import YoloDiagramDetector


def _page(tmp_path: Path) -> PageResult:
    return PageResult("page-1", 0, 100, 100, tmp_path / "original.png", None, AffineTransform.identity(), PageQuality(1, 1, None))


def test_disabled_yolo_is_safe(tmp_path: Path) -> None:
    result = YoloDiagramDetector(YoloDiagramConfig()).detect_diagrams(np.zeros((100, 100, 3), dtype=np.uint8), _page(tmp_path))
    assert result.regions == []
    assert result.warnings == ["yolo_diagram_detector_disabled"]


def test_yolo_overlap_replaces_baseline_in_canonical_diagram_list(tmp_path: Path) -> None:
    page = _page(tmp_path)
    baseline = Region("base", "page-1", RegionClass.IMAGE, BoundingBox(10, 10, 40, 40), "paddle")
    yolo = Region("yolo", "page-1", RegionClass.DIAGRAM, BoundingBox(12, 12, 40, 40), "ultralytics_yolo_diagram", 0.9)
    page.regions.append(baseline)
    merged = merge_diagram_regions(page, [yolo], 0.5)
    assert [region.id for region in merged] == ["yolo"]
    assert yolo.attributes["overlaps_baseline_region_ids"] == ["base"]


def test_diagram_crop_reference_is_returned(tmp_path: Path) -> None:
    page = _page(tmp_path)
    page.diagram_regions = [Region("diagram", "page-1", RegionClass.DIAGRAM, BoundingBox(20, 20, 20, 20), "test")]
    save_diagram_crops(np.zeros((100, 100, 3), dtype=np.uint8), page, tmp_path / "crops")
    assert page.diagram_regions[0].crop_reference is not None
    assert Path(page.diagram_regions[0].crop_reference).is_file()
