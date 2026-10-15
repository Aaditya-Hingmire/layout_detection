"""Stable, versioned result serialization and public-contract validation."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import ResultValidationError
from .models import DocumentResult, SCHEMA_VERSION


def validate_document_result(result: DocumentResult) -> None:
    """Validate invariants needed by external callers before serialization."""
    if result.schema_version != SCHEMA_VERSION:
        raise ResultValidationError(f"unsupported result schema version: {result.schema_version}")
    page_ids = [page.page_id for page in result.pages]
    if len(page_ids) != len(set(page_ids)):
        raise ResultValidationError("page IDs must be unique")
    for page in result.pages:
        if page.width_px <= 0 or page.height_px <= 0:
            raise ResultValidationError(f"invalid dimensions for {page.page_id}")
        for region in page.regions + page.diagram_regions:
            if region.page_id != page.page_id:
                raise ResultValidationError(f"region {region.id} has a mismatched page ID")
        for line in page.ocr_lines:
            if line.page_id != page.page_id:
                raise ResultValidationError(f"OCR line {line.id} has a mismatched page ID")
        for anchor in page.question_anchor_candidates:
            if anchor.page_id != page.page_id:
                raise ResultValidationError(f"anchor {anchor.id} has a mismatched page ID")
    if result.question_association and result.question_association.unaccounted_content_ids:
        raise ResultValidationError("question association has unaccounted content items")


def serialize_document_result(result: DocumentResult, *, indent: int | None = 2) -> str:
    validate_document_result(result)
    return json.dumps(result.to_dict(), indent=indent, ensure_ascii=False)


def write_document_result(result: DocumentResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_document_result(result), encoding="utf-8")
    return path
