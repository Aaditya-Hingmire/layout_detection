import numpy as np

from answer_sheet_layout.models import (AffineTransform, BoundingBox, OcrLine, PageQuality, PageResult,
                                         QuestionAnchorCandidate, Region, RegionClass)
from answer_sheet_layout.visualization import render_comparison, render_overlay


def test_overlay_uses_page_coordinate_boxes(tmp_path) -> None:
    page = PageResult("page-1", 0, 100, 80, tmp_path / "original.png", None, AffineTransform.identity(),
                      PageQuality(1, 1, None))
    page.regions.append(Region("r1", "page-1", RegionClass.TABLE, BoundingBox(10, 10, 30, 20), "test"))
    page.ocr_lines.append(OcrLine("o1", "page-1", "Q1", BoundingBox(10, 10, 20, 10), ((10, 10), (30, 10), (30, 20), (10, 20)), "test"))
    page.question_anchor_candidates.append(QuestionAnchorCandidate("a1", "page-1", "o1", "Q1", BoundingBox(10, 10, 20, 10), "rule", None))
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    overlay = render_overlay(image, page)
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay, image)
    assert render_comparison(image, page).shape == (80, 200, 3)
