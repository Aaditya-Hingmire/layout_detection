import json

from answer_sheet_layout.benchmark import evaluate_benchmark


def _box(x, y, width, height):
    return {"x": x, "y": y, "width": width, "height": height}


def _prediction(include_diagram: bool):
    diagram = [{"id": "pred-d1", "bounding_box": _box(50, 40, 30, 30), "source_detector": "yolo", "confidence": .9}] if include_diagram else []
    members = [{"content_id": "pred-t1"}] + ([{"content_id": "pred-d1"}] if include_diagram else [])
    return {"pages": [{"page_id": "page-1", "width_px": 100, "height_px": 100, "regions": [],
                         "ocr_lines": [{"id": "pred-t1", "text": "answer", "bounding_box": _box(10, 30, 25, 15), "confidence": .9}],
                         "diagram_regions": diagram,
                         "question_anchor_candidates": [{"id": "pred-a1", "text": "Q1", "bounding_box": _box(10, 5, 20, 10), "confidence": .9}],
                         "detector_warnings": []}],
            "question_association": {"question_layouts": [{"anchor_evidence": {"id": "pred-a1"}, "associated_content": members}]},
            "processing_metadata": {"wall_time_seconds": 1.0}}


def test_benchmark_compares_variants_and_marks_synthetic_data(tmp_path):
    dataset = tmp_path / "dataset"; annotations = dataset / "annotations"; annotations.mkdir(parents=True)
    annotation = {"document_id": "doc-1", "student_id": "student-1", "source_type": "synthetic", "pages": [{
        "page_id": "page-1", "width_px": 100, "height_px": 100, "tags": ["small_diagram"],
        "regions": [{"id": "t1", "class": "handwritten_text", "bbox": _box(10, 30, 25, 15)}, {"id": "d1", "class": "diagram_region", "bbox": _box(50, 40, 30, 30)}],
        "anchors": [{"id": "a1", "text": "Q1", "bbox": _box(10, 5, 20, 10)}],
        "associations": [{"anchor_id": "a1", "member_ids": ["t1", "d1"]}],
    }]}
    (annotations / "doc-1.json").write_text(json.dumps(annotation), encoding="utf-8")
    (dataset / "split-manifest.json").write_text(json.dumps({"unit": "student_id", "splits": {"test": ["student-1"]}}), encoding="utf-8")
    baseline = tmp_path / "baseline"; yolo = tmp_path / "yolo"; baseline.mkdir(); yolo.mkdir()
    (baseline / "doc-1.json").write_text(json.dumps(_prediction(False)), encoding="utf-8")
    (yolo / "doc-1.json").write_text(json.dumps(_prediction(True)), encoding="utf-8")
    report = evaluate_benchmark(dataset, {"baseline": baseline, "baseline_plus_yolo": yolo}, tmp_path / "out")
    assert report["dataset"]["source_counts"] == {"synthetic": 1}
    assert report["variants"]["baseline"]["by_source_type"]["synthetic"]["text_region_localization"]["recall"] == 1.0
    assert report["variants"]["baseline"]["diagram_region_localization"]["recall"] == 0.0
    assert report["variants"]["baseline_plus_yolo"]["diagram_region_localization"]["recall"] == 1.0
    assert report["variants"]["baseline_plus_yolo"]["question_association"]["accuracy"] == 1.0
    assert (tmp_path / "out" / "metrics.json").is_file()
