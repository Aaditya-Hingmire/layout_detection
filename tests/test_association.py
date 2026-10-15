from pathlib import Path

from answer_sheet_layout.association import associate_question_regions
from answer_sheet_layout.models import (AffineTransform, BoundingBox, DocumentResult, OcrLine, PageQuality, PageResult,
                                         QuestionAnchorCandidate, Region, RegionClass)


def _page(index: int) -> PageResult:
    return PageResult(f"page-{index + 1}", index, 500, 700, Path(f"page-{index + 1}.png"), None,
                      AffineTransform.identity(), PageQuality(1, 1, None))


def _line(page: PageResult, ident: str, text: str, y: float) -> OcrLine:
    return OcrLine(ident, page.page_id, text, BoundingBox(30, y, 100, 20), ((30, y), (130, y), (130, y + 20), (30, y + 20)), "test", 0.9)


def _anchor(page: PageResult, line: OcrLine, ident: str) -> QuestionAnchorCandidate:
    return QuestionAnchorCandidate(ident, page.page_id, line.id, line.text, line.bounding_box, "test-rule", 0.9)


def test_associates_text_diagram_subquestion_and_continuation() -> None:
    first, second = _page(0), _page(1)
    q1 = _line(first, "o-q1", "Q.1", 20)
    q1a = _line(first, "o-q1a", "1(a)", 300)
    body = _line(first, "o-body", "answer text", 70)
    first.ocr_lines = [q1, body, q1a]
    first.question_anchor_candidates = [_anchor(first, q1, "a-q1"), _anchor(first, q1a, "a-q1a")]
    first.diagram_regions = [Region("diagram-1", first.page_id, RegionClass.DIAGRAM, BoundingBox(40, 130, 180, 100), "yolo", 0.8, "crop.png")]
    continued = _line(second, "o-cont", "continued answer", 40)
    second.ocr_lines = [continued]
    result = associate_question_regions(DocumentResult(Path("source.pdf"), [first, second]))
    q1_layout = next(layout for layout in result.question_layouts if layout.anchor_evidence.id == "a-q1")
    q1a_layout = next(layout for layout in result.question_layouts if layout.anchor_evidence.id == "a-q1a")
    assert {item.content_id for item in q1_layout.associated_content} == {"o-body", "diagram-1"}
    assert {item.content_id for item in q1a_layout.associated_content} == {"o-cont"}
    assert q1a_layout.continuation_from_previous_page
    assert q1a_layout.confidence_indicator == "tentative_continuation"
    assert result.total_content_items == result.accounted_content_items
    assert result.unaccounted_content_ids == []


def test_diagram_crossing_two_anchor_segments_is_ambiguous_and_accounted_for() -> None:
    page = _page(0)
    q1 = _line(page, "o-q1", "Q1", 20)
    q2 = _line(page, "o-q2", "Question 2", 200)
    page.ocr_lines = [q1, q2]
    page.question_anchor_candidates = [_anchor(page, q1, "a-q1"), _anchor(page, q2, "a-q2")]
    page.diagram_regions = [Region("ambiguous-diagram", page.page_id, RegionClass.DIAGRAM, BoundingBox(50, 150, 150, 100), "yolo", 0.8)]
    result = associate_question_regions(DocumentResult(Path("source.pdf"), [page]))
    unresolved = next(item for item in result.unresolved_content if item.content.content_id == "ambiguous-diagram")
    assert unresolved.status == "ambiguous"
    assert "crosses" in unresolved.reason


def test_content_before_first_unreadable_anchor_is_retained_as_unassigned() -> None:
    page = _page(0)
    page.regions = [Region("orphan", page.page_id, RegionClass.HANDWRITTEN_TEXT, BoundingBox(20, 20, 100, 40), "layout")]
    result = associate_question_regions(DocumentResult(Path("source.pdf"), [page]))
    assert result.unresolved_content[0].content.content_id == "orphan"
    assert result.unresolved_content[0].status == "unassigned"
