"""Deterministic, reviewable question-wise spatial association."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AssociationConfig
from .models import (AssociatedContent, DocumentResult, OcrLine, QuestionAssociationResult,
                     QuestionWiseLayout, Region, RegionClass, UnresolvedContent)


@dataclass(frozen=True)
class _Anchor:
    question_id: str
    page_index: int
    page_id: str
    y: float
    candidate: object


def associate_question_regions(document: DocumentResult, config: AssociationConfig | None = None) -> QuestionAssociationResult:
    """Associate all detected content without inferring academic meaning.

    Pages are divided into vertical candidate-anchor segments. Diagrams are only
    assigned when wholly contained in one segment; otherwise they remain explicit
    ambiguous/unassigned output.
    """
    config = config or AssociationConfig()
    anchors = _ordered_anchors(document)
    layouts = {anchor.question_id: QuestionWiseLayout(anchor.question_id, anchor.candidate) for anchor in anchors}
    unresolved: list[UnresolvedContent] = []
    superseded = _superseded_baseline_region_ids(document)
    for page in sorted(document.pages, key=lambda item: item.page_index):
        page_anchors = [anchor for anchor in anchors if anchor.page_id == page.page_id]
        for item, is_diagram in _page_content(page, superseded):
            segment = _find_segment(item.bounding_box.y, item.bounding_box.y + item.bounding_box.height, page_anchors)
            if segment is not None:
                anchor, crosses_boundary = segment
                if crosses_boundary:
                    unresolved.append(UnresolvedContent(item, "ambiguous", "content crosses a question-anchor boundary", tuple(anchor.question_id for anchor in page_anchors)))
                    continue
                confidence = config.diagram_same_segment_confidence if is_diagram else config.same_page_confidence
                _associate(layouts[anchor.question_id], item, confidence, ("same_page_anchor_segment",), False)
                continue
            previous = _previous_anchor(anchors, page.page_index)
            if previous is not None and (not page_anchors or item.bounding_box.y < page_anchors[0].y):
                if is_diagram and page_anchors:
                    unresolved.append(UnresolvedContent(item, "ambiguous", "diagram is before a new anchor; prior-page continuation is plausible but not unique", (previous.question_id, page_anchors[0].question_id)))
                else:
                    _associate(layouts[previous.question_id], item, config.continuation_confidence, ("continuation_before_next_anchor",), True)
                continue
            unresolved.append(UnresolvedContent(item, "unassigned", "no candidate question anchor provides spatial or sequential evidence"))
    for page in document.pages:
        for region in page.regions:
            if region.id in superseded:
                unresolved.append(UnresolvedContent(_content_from_region(region), "deduplicated", "superseded by overlapping canonical diagram detection"))
    expected = _all_content_keys(document)
    accounted = {
        f"{content.content_type}:{content.content_id}"
        for layout in layouts.values() for content in layout.associated_content
    } | {f"{item.content.content_type}:{item.content.content_id}" for item in unresolved}
    result = QuestionAssociationResult(list(layouts.values()), unresolved, len(expected), len(expected & accounted), sorted(expected - accounted))
    document.question_association = result
    return result


def _ordered_anchors(document: DocumentResult) -> list[_Anchor]:
    anchors: list[_Anchor] = []
    for page in sorted(document.pages, key=lambda item: item.page_index):
        for candidate in sorted(page.question_anchor_candidates, key=lambda item: (item.bounding_box.y, item.bounding_box.x, item.id)):
            anchors.append(_Anchor(f"candidate:{candidate.id}", page.page_index, page.page_id, candidate.bounding_box.y, candidate))
    return anchors


def _page_content(page, superseded: set[str]) -> list[tuple[AssociatedContent, bool]]:
    items: list[tuple[AssociatedContent, bool]] = []
    anchor_line_ids = {candidate.ocr_line_id for candidate in page.question_anchor_candidates}
    for line in page.ocr_lines:
        if line.id not in anchor_line_ids:
            items.append((_content_from_ocr(line), False))
    canonical_ids = {region.id for region in page.diagram_regions}
    for region in page.regions:
        if region.id not in superseded and region.id not in canonical_ids:
            items.append((_content_from_region(region), region.region_class == RegionClass.DIAGRAM))
    existing = {item[0].content_id for item in items}
    for region in page.diagram_regions:
        if region.id not in existing:
            items.append((_content_from_region(region), True))
    return sorted(items, key=lambda item: (item[0].bounding_box.y, item[0].bounding_box.x, item[0].content_id))


def _find_segment(top: float, bottom: float, anchors: list[_Anchor]) -> tuple[_Anchor, bool] | None:
    if not anchors or top < anchors[0].y:
        return None
    selected = [anchor for anchor in anchors if anchor.y <= top][-1]
    next_anchors = [anchor for anchor in anchors if anchor.y > selected.y]
    return selected, bool(next_anchors and bottom > next_anchors[0].y)


def _previous_anchor(anchors: list[_Anchor], page_index: int) -> _Anchor | None:
    preceding = [anchor for anchor in anchors if anchor.page_index < page_index]
    return preceding[-1] if preceding else None


def _associate(layout: QuestionWiseLayout, content: AssociatedContent, confidence: float, evidence: tuple[str, ...], continuation: bool) -> None:
    layout.associated_content.append(AssociatedContent(content.content_id, content.content_type, content.page_id, content.bounding_box, content.source_detector, content.crop_reference, content.detector_confidence, confidence, evidence))
    if content.page_id not in layout.page_ids:
        layout.page_ids.append(content.page_id)
    if continuation:
        layout.continuation_from_previous_page = True
        layout.confidence_indicator = "tentative_continuation"


def _content_from_region(region: Region) -> AssociatedContent:
    return AssociatedContent(region.id, "region", region.page_id, region.bounding_box, region.source_detector, region.crop_reference, region.confidence, 0.0, ())


def _content_from_ocr(line: OcrLine) -> AssociatedContent:
    return AssociatedContent(line.id, "ocr_line", line.page_id, line.bounding_box, line.source_detector, None, line.confidence, 0.0, ())


def _superseded_baseline_region_ids(document: DocumentResult) -> set[str]:
    return {region_id for page in document.pages for diagram in page.diagram_regions for region_id in diagram.attributes.get("overlaps_baseline_region_ids", [])}


def _all_content_keys(document: DocumentResult) -> set[str]:
    keys: set[str] = set()
    for page in document.pages:
        anchor_line_ids = {candidate.ocr_line_id for candidate in page.question_anchor_candidates}
        keys.update(f"ocr_line:{line.id}" for line in page.ocr_lines if line.id not in anchor_line_ids)
        keys.update(f"region:{region.id}" for region in page.regions)
        keys.update(f"region:{region.id}" for region in page.diagram_regions)
    return keys
