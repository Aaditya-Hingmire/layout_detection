"""Validated, deterministic PDF rendering using PyMuPDF."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import fitz
import numpy as np

from .config import LayoutConfig
from .errors import InputValidationError, RenderingError
from .models import DocumentResult, PageResult
from .preprocessing import assess_page_quality, preprocess_page

logger = logging.getLogger(__name__)


def render_pdf(pdf_path: str | Path, output_dir: str | Path, config: LayoutConfig | None = None) -> DocumentResult:
    """Validate, render pages in PDF order, and write original/processed image artifacts."""
    config = config or LayoutConfig()
    source = Path(pdf_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    _validate_source(source)
    try:
        document = fitz.open(source)
    except (fitz.FileDataError, RuntimeError) as error:
        raise InputValidationError("input is not a readable PDF") from error
    try:
        if document.is_encrypted:
            raise InputValidationError("encrypted PDFs are not supported")
        if document.page_count == 0:
            raise InputValidationError("PDF contains no pages")
        if document.page_count > config.rendering.max_pages:
            raise InputValidationError(f"PDF exceeds max_pages={config.rendering.max_pages}")
        original_dir = destination / "original"
        processed_dir = destination / "processed"
        original_dir.mkdir(parents=True, exist_ok=True)
        pages: list[PageResult] = []
        for index in range(document.page_count):
            page = document.load_page(index)
            pixmap = page.get_pixmap(dpi=config.rendering.dpi, colorspace=fitz.csRGB, alpha=False)
            image = _pixmap_to_bgr(pixmap)
            _enforce_pixel_limit(image, config.rendering.max_pixels_per_page)
            page_id = f"page-{index + 1}"
            original_path = original_dir / f"{page_id}.png"
            _write_image(original_path, image)
            quality = assess_page_quality(image, config.preprocessing)
            processed, transform = preprocess_page(image, config.preprocessing, quality)
            processed_path = None
            if config.preprocessing.deskew or config.preprocessing.denoise or config.preprocessing.enhance_contrast:
                processed_dir.mkdir(parents=True, exist_ok=True)
                processed_path = processed_dir / f"{page_id}.png"
                _write_image(processed_path, processed)
            pages.append(PageResult(page_id, index, image.shape[1], image.shape[0], original_path,
                                    processed_path, transform, quality))
            logger.info("Rendered %s at %s DPI", page_id, config.rendering.dpi)
        return DocumentResult(source, pages)
    except InputValidationError:
        raise
    except Exception as error:
        raise RenderingError(f"failed to render PDF: {source.name}") from error
    finally:
        document.close()


def _validate_source(source: Path) -> None:
    if not source.is_file():
        raise InputValidationError("input path must be an existing file")
    if source.suffix.lower() != ".pdf":
        raise InputValidationError("input file must use the .pdf extension")
    if source.stat().st_size == 0:
        raise InputValidationError("input PDF is empty")
    with source.open("rb") as file:
        if file.read(5) != b"%PDF-":
            raise InputValidationError("input does not have a PDF header")


def _pixmap_to_bgr(pixmap: fitz.Pixmap) -> np.ndarray:
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n != 3:
        raise RenderingError(f"unsupported rendered channel count: {pixmap.n}")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _enforce_pixel_limit(image: np.ndarray, max_pixels: int) -> None:
    if image.shape[0] * image.shape[1] > max_pixels:
        raise InputValidationError(f"rendered page exceeds max_pixels_per_page={max_pixels}")


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RenderingError(f"could not write image artifact: {path}")
