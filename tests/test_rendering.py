from pathlib import Path

import fitz
import pytest

from answer_sheet_layout.config import LayoutConfig, PreprocessingConfig, RenderingConfig
from answer_sheet_layout.errors import InputValidationError
from answer_sheet_layout.rendering import render_pdf


def _make_pdf(path: Path, pages: int = 2) -> None:
    document = fitz.open()
    for number in range(pages):
        page = document.new_page(width=200, height=300)
        page.insert_text((30, 40), f"Page {number + 1}")
    document.save(path)
    document.close()


def test_renders_pages_in_document_order_and_preserves_originals(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    _make_pdf(pdf)
    result = render_pdf(pdf, tmp_path / "out", LayoutConfig(RenderingConfig(dpi=100)))
    assert [page.page_id for page in result.pages] == ["page-1", "page-2"]
    assert [page.page_index for page in result.pages] == [0, 1]
    assert all(page.original_image_path.is_file() for page in result.pages)
    assert all(page.processed_image_path is None for page in result.pages)


def test_rejects_non_pdf_input(tmp_path: Path) -> None:
    source = tmp_path / "invalid.pdf"
    source.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(InputValidationError, match="PDF header"):
        render_pdf(source, tmp_path / "out")


def test_rejects_empty_pdf(tmp_path: Path) -> None:
    source = tmp_path / "empty.pdf"
    source.write_bytes(b"")
    with pytest.raises(InputValidationError, match="empty"):
        render_pdf(source, tmp_path / "out")


def test_processed_artifacts_are_separate_from_original_artifacts(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    _make_pdf(pdf, pages=1)
    result = render_pdf(
        pdf,
        tmp_path / "out",
        LayoutConfig(RenderingConfig(dpi=100), PreprocessingConfig(enhance_contrast=True)),
    )
    page = result.pages[0]
    assert page.original_image_path.is_file()
    assert page.processed_image_path is not None
    assert page.processed_image_path.is_file()
    assert page.original_image_path != page.processed_image_path
    assert page.transform_from_original.operations[0].name == "contrast_enhancement"
