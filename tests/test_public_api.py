import json
from pathlib import Path

import fitz
import numpy as np
import pytest

from answer_sheet_layout import (BoundingBox, ProcessOptions, Region, RegionClass, ResultValidationError,
                                  process_pdf, serialize_document_result, validate_document_result)
from answer_sheet_layout.detectors import PageAnalysis


class _FixtureDetector:
    name = "fixture_layout"

    def analyze(self, image, page):
        assert isinstance(image, np.ndarray)
        return PageAnalysis(regions=[Region("fixture-region", page.page_id, RegionClass.HANDWRITTEN_TEXT,
                                             BoundingBox(10, 10, 30, 20), self.name, 0.9)])


def _make_pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page(width=200, height=300).insert_text((20, 30), "Fixture")
    document.save(path)
    document.close()


def test_process_pdf_is_importable_serializable_and_writes_artifacts(tmp_path: Path) -> None:
    pdf = tmp_path / "fixture.pdf"; _make_pdf(pdf)
    result = process_pdf(pdf, ProcessOptions(tmp_path / "output", use_paddle_layout=False), layout_detector=_FixtureDetector())
    payload = json.loads(serialize_document_result(result))
    assert payload["schema_version"] == "1.0"
    assert payload["pages"][0]["regions"][0]["id"] == "fixture-region"
    assert (tmp_path / "output" / "result.json").is_file()
    assert result.processing_metadata["page_count"] == 1


def test_public_validation_rejects_mismatched_region_page(tmp_path: Path) -> None:
    pdf = tmp_path / "fixture.pdf"; _make_pdf(pdf)
    result = process_pdf(pdf, ProcessOptions(tmp_path / "output", use_paddle_layout=False), layout_detector=_FixtureDetector())
    result.pages[0].regions[0] = Region("bad", "other-page", RegionClass.UNKNOWN, BoundingBox(0, 0, 1, 1), "test")
    with pytest.raises(ResultValidationError, match="mismatched"):
        validate_document_result(result)
