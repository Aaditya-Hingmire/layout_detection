from pathlib import Path

import numpy as np

from answer_sheet_layout.config import QuestionAnchorConfig
from answer_sheet_layout.models import AffineTransform, PageQuality, PageResult
from answer_sheet_layout.paddle_adapter import PaddlePPStructureV3Adapter


class _FakeResult:
    json = {
        "res": {
            "layout_det_res": {
                "boxes": [[10, 20, 80, 90]],
                "labels": ["table"],
                "scores": [0.85],
            },
            "overall_ocr_res": {
                "rec_texts": ["Q.1", "Ordinary answer"],
                "rec_scores": [0.95, 0.80],
                "rec_polys": [
                    [[10, 20], [50, 20], [50, 35], [10, 35]],
                    [[10, 45], [140, 45], [140, 70], [10, 70]],
                ],
            },
        }
    }


class _FakePipeline:
    def predict(self, *, input):
        assert input.shape == (100, 200, 3)
        return [_FakeResult()]


def _page() -> PageResult:
    return PageResult(
        "page-1", 0, 200, 100, Path("original.png"), None, AffineTransform.identity(),
        PageQuality(1, 1, None),
    )


def test_normalizes_layout_ocr_and_candidate_anchor() -> None:
    adapter = PaddlePPStructureV3Adapter(question_config=QuestionAnchorConfig(), pipeline=_FakePipeline())
    analysis = adapter.analyze(np.zeros((100, 200, 3), dtype=np.uint8), _page())
    assert analysis.regions[0].bounding_box.as_xyxy() == (10, 20, 80, 90)
    assert analysis.regions[0].region_class.value == "table"
    assert analysis.ocr_lines[0].text == "Q.1"
    assert analysis.ocr_lines[0].reading_order == 0
    assert analysis.question_anchor_candidates[0].status == "candidate"
    assert analysis.question_anchor_candidates[0].ocr_line_id == "page-1-ocr-1"


def test_detector_failure_is_returned_as_warning() -> None:
    class BrokenPipeline:
        def predict(self, *, input):
            raise RuntimeError("model unavailable")

    analysis = PaddlePPStructureV3Adapter(pipeline=BrokenPipeline()).analyze(
        np.zeros((10, 10, 3), dtype=np.uint8), _page(),
    )
    assert analysis.regions == []
    assert "paddle_pp_structure_v3_failed: RuntimeError" in analysis.warnings[0]
