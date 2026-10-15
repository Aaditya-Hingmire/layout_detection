"""Reproducible, annotation-driven benchmark for structural layout detection."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from .diagrams import intersection_over_union
from .models import BoundingBox


IOU_THRESHOLD = 0.5


def evaluate_benchmark(dataset_dir: str | Path, variants: dict[str, str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Evaluate result.json artifacts against manual document-level annotations."""
    root, output = Path(dataset_dir), Path(output_dir)
    annotations = _load_annotations(root)
    split_warnings = _validate_splits(root, annotations)
    report: dict[str, Any] = {
        "benchmark_schema_version": "1.0",
        "dataset": {"path": str(root), "documents": len(annotations), "source_counts": dict(Counter(a.get("source_type", "unknown") for a in annotations))},
        "split_warnings": split_warnings,
        "variants": {},
    }
    for name, directory in variants.items():
        predictions, missing = _load_predictions(Path(directory), annotations)
        metrics, errors = _evaluate_variant(annotations, predictions)
        metrics["missing_prediction_documents"] = missing
        metrics["variant_path"] = str(directory)
        metrics["configuration_and_model_versions"] = _metadata(predictions)
        metrics["by_source_type"] = {
            source: _evaluate_variant([item for item in annotations if item.get("source_type", "unknown") == source], predictions)[0]
            for source in sorted({item.get("source_type", "unknown") for item in annotations})
        }
        visual_paths = _write_error_visuals(root, output / name / "visual-errors", annotations, predictions, errors)
        metrics["visual_error_reports"] = visual_paths
        metrics["failure_analysis"] = _failure_analysis(errors, annotations)
        report["variants"][name] = metrics
        _write_failure_markdown(output / name / "failure-analysis.md", name, metrics)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_annotations(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "annotations").glob("*.json"))
    if not files:
        raise ValueError("no annotation JSON files found under annotations/")
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def _validate_splits(root: Path, annotations: list[dict[str, Any]]) -> list[str]:
    path = root / "split-manifest.json"
    if not path.is_file():
        return ["Missing split-manifest.json; document/student leakage was not checked."]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = {name: set(values) for name, values in manifest.get("splits", {}).items()}
    for first in splits:
        for second in splits:
            if first < second and splits[first] & splits[second]:
                raise ValueError(f"split leakage: {first} and {second} share IDs")
    ids = {annotation[manifest.get("unit", "document_id")] for annotation in annotations}
    listed = set().union(*splits.values()) if splits else set()
    return [f"{len(ids - listed)} annotated unit(s) are absent from the split manifest."] if ids - listed else []


def _load_predictions(directory: Path, annotations: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for annotation in annotations:
        document_id = annotation["document_id"]
        candidates = [directory / f"{document_id}.json", directory / document_id / "result.json"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            missing.append(document_id)
        else:
            loaded[document_id] = json.loads(path.read_text(encoding="utf-8"))
    return loaded, missing


def _evaluate_variant(annotations: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_text, pred_text = _collect_boxes(annotations, predictions, "text")
    gt_diagram, pred_diagram = _collect_boxes(annotations, predictions, "diagram")
    gt_anchor, pred_anchor = _collect_boxes(annotations, predictions, "anchor")
    text_metric, text_errors = _detection_metrics(gt_text, pred_text)
    diagram_metric, diagram_errors = _detection_metrics(gt_diagram, pred_diagram)
    anchor_metric, anchor_errors = _anchor_metrics(gt_anchor, pred_anchor)
    association = _association_metric(annotations, predictions)
    coordinates = _coordinate_metric(predictions)
    performance = _performance_metric(predictions)
    return ({"text_region_localization": text_metric, "diagram_region_localization": diagram_metric,
             "question_anchor_detection": anchor_metric, "question_association": association,
             "coordinate_correctness": coordinates, "processing": performance}, text_errors + diagram_errors + anchor_errors)


def _collect_boxes(annotations: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], kind: str) -> tuple[list[dict], list[dict]]:
    gt: list[dict] = []
    pred: list[dict] = []
    for annotation in annotations:
        doc = annotation["document_id"]
        result = predictions.get(doc, {})
        by_page = {page["page_id"]: page for page in result.get("pages", [])}
        for page in annotation.get("pages", []):
            key = (doc, page["page_id"])
            if kind == "text":
                gt.extend(_tag(item, key, "text") for item in page.get("regions", []) if item.get("class") in {"handwritten_text", "printed_text", "text"})
                pred.extend(_tag(item, key, "text") for item in by_page.get(page["page_id"], {}).get("ocr_lines", []))
            elif kind == "diagram":
                gt.extend(_tag(item, key, "diagram") for item in page.get("regions", []) if item.get("class") == "diagram_region")
                predicted_page = by_page.get(page["page_id"], {})
                pred.extend(_tag(item, key, "diagram") for item in predicted_page.get("diagram_regions", []))
            else:
                gt.extend(_tag(item, key, "anchor") for item in page.get("anchors", []))
                pred.extend(_tag(item, key, "anchor") for item in by_page.get(page["page_id"], {}).get("question_anchor_candidates", []))
    return gt, pred


def _tag(item: dict, key: tuple[str, str], kind: str) -> dict:
    tagged = dict(item)
    tagged["key"] = key
    tagged["kind"] = kind
    tagged["score"] = item.get("confidence", 0.5) if item.get("confidence") is not None else 0.5
    return tagged


def _box(item: dict) -> BoundingBox:
    value = item.get("bounding_box") or item.get("bbox")
    return BoundingBox(float(value["x"]), float(value["y"]), float(value["width"]), float(value["height"]))


def _detection_metrics(ground_truth: list[dict], predictions: list[dict], iou_threshold: float = IOU_THRESHOLD, collect_errors: bool = True, include_map95: bool = True) -> tuple[dict, list[dict]]:
    matched_gt: set[int] = set()
    tp = fp = 0
    errors: list[dict] = []
    ranked = sorted(enumerate(predictions), key=lambda pair: pair[1]["score"], reverse=True)
    precision_curve: list[float] = []
    recall_curve: list[float] = []
    for _, prediction in ranked:
        candidates = [(index, intersection_over_union(_box(prediction), _box(truth))) for index, truth in enumerate(ground_truth)
                      if index not in matched_gt and truth["key"] == prediction["key"]]
        index, iou = max(candidates, key=lambda pair: pair[1], default=(-1, 0.0))
        if iou >= iou_threshold:
            matched_gt.add(index); tp += 1
        else:
            fp += 1
            if collect_errors: errors.append({"type": "false_positive", "kind": prediction["kind"], "key": prediction["key"], "id": prediction.get("id")})
        precision_curve.append(tp / (tp + fp))
        recall_curve.append(tp / len(ground_truth) if ground_truth else 0.0)
    for index, truth in enumerate(ground_truth):
        if collect_errors and index not in matched_gt:
            errors.append({"type": "false_negative", "kind": truth["kind"], "key": truth["key"], "id": truth.get("id")})
    metric = _metric(tp, fp, len(ground_truth) - tp, _average_precision(precision_curve, recall_curve) if ground_truth else None)
    metric["iou_threshold"] = iou_threshold
    if iou_threshold == IOU_THRESHOLD and include_map95:
        ap_values = [_detection_metrics(ground_truth, predictions, threshold, False, False)[0]["map50"] for threshold in [0.5 + .05 * index for index in range(10)]]
        metric["map50_95"] = sum(value for value in ap_values if value is not None) / len(ap_values) if any(value is not None for value in ap_values) else None
    return metric, errors


def _anchor_metrics(ground_truth: list[dict], predictions: list[dict]) -> tuple[dict, list[dict]]:
    compatible = []
    for item in predictions:
        item = dict(item); item["anchor_text"] = _normalise(item.get("text", "")); compatible.append(item)
    # Text agreement is required in addition to geometry for anchor matching.
    original = _detection_metrics
    matched_gt: set[int] = set(); tp = fp = 0; errors: list[dict] = []
    for prediction in sorted(compatible, key=lambda item: item["score"], reverse=True):
        candidates = [(index, intersection_over_union(_box(prediction), _box(truth))) for index, truth in enumerate(ground_truth)
                      if index not in matched_gt and truth["key"] == prediction["key"] and prediction["anchor_text"] == _normalise(truth.get("text", ""))]
        index, iou = max(candidates, key=lambda pair: pair[1], default=(-1, 0.0))
        if iou >= IOU_THRESHOLD:
            matched_gt.add(index); tp += 1
        else:
            fp += 1; errors.append({"type": "false_positive", "kind": "anchor", "key": prediction["key"], "id": prediction.get("id")})
    for index, truth in enumerate(ground_truth):
        if index not in matched_gt:
            errors.append({"type": "false_negative", "kind": "anchor", "key": truth["key"], "id": truth.get("id")})
    return _metric(tp, fp, len(ground_truth) - tp, None), errors


def _metric(tp: int, fp: int, fn: int, map50: float | None) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"iou_threshold": IOU_THRESHOLD, "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None,
            "map50": map50}


def _average_precision(precision: list[float], recall: list[float]) -> float | None:
    if not precision:
        return 0.0
    total = 0.0
    for threshold in [index / 100 for index in range(101)]:
        values = [value for value, rec in zip(precision, recall) if rec >= threshold]
        total += max(values, default=0.0)
    return total / 101


def _normalise(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()


def _association_metric(annotations: list[dict], predictions: dict[str, dict]) -> dict:
    total = correct = 0
    for annotation in annotations:
        result = predictions.get(annotation["document_id"], {})
        predicted = result.get("question_association", {}).get("question_layouts", [])
        associated = {content["content_id"]: layout.get("anchor_evidence", {}).get("id") for layout in predicted for content in layout.get("associated_content", [])}
        predicted_pages = {page["page_id"]: page for page in result.get("pages", [])}
        for page in annotation.get("pages", []):
            predicted_page = predicted_pages.get(page["page_id"], {})
            predicted_anchors = predicted_page.get("question_anchor_candidates", [])
            anchor_mapping = {}
            for truth in page.get("anchors", []):
                candidates = [candidate for candidate in predicted_anchors if _normalise(candidate.get("text", "")) == _normalise(truth.get("text", ""))]
                winner = max(candidates, key=lambda candidate: intersection_over_union(_box(truth), _box(candidate)), default=None)
                if winner is not None and intersection_over_union(_box(truth), _box(winner)) >= IOU_THRESHOLD:
                    anchor_mapping[truth["id"]] = winner["id"]
            for group in page.get("associations", []):
                for member_id in group.get("member_ids", []):
                    total += 1
                    truth_item = next((item for item in page.get("regions", []) if item["id"] == member_id), None)
                    predicted_items = (predicted_page.get("regions", []) + predicted_page.get("ocr_lines", []) + predicted_page.get("diagram_regions", []))
                    winner = max(predicted_items, key=lambda candidate: intersection_over_union(_box(truth_item), _box(candidate)), default=None) if truth_item else None
                    if winner is not None and intersection_over_union(_box(truth_item), _box(winner)) >= IOU_THRESHOLD and associated.get(winner.get("id")) == anchor_mapping.get(group["anchor_id"]):
                        correct += 1
    return {"definition": "correctly assigned annotated members / annotated association members", "correct": correct, "total": total,
            "accuracy": correct / total if total else None,
            "warning": None if total else "No directly comparable association IDs were supplied."}


def _coordinate_metric(predictions: dict[str, dict]) -> dict:
    total = valid = 0; violations: list[dict] = []
    for document_id, result in predictions.items():
        for page in result.get("pages", []):
            width, height = page.get("width_px", 0), page.get("height_px", 0)
            for collection in ("regions", "ocr_lines", "diagram_regions", "question_anchor_candidates"):
                for item in page.get(collection, []):
                    total += 1
                    try:
                        box = _box(item); okay = box.x >= 0 and box.y >= 0 and box.x + box.width <= width and box.y + box.height <= height
                    except (KeyError, TypeError, ValueError):
                        okay = False
                    if okay: valid += 1
                    else: violations.append({"document_id": document_id, "page_id": page.get("page_id"), "item_id": item.get("id"), "collection": collection})
    return {"valid": valid, "total": total, "rate": valid / total if total else None, "violations": violations}


def _performance_metric(predictions: dict[str, dict]) -> dict:
    times = [result.get("processing_metadata", {}).get("wall_time_seconds") for result in predictions.values()]
    times = [time for time in times if isinstance(time, (int, float))]
    pages = [page for result in predictions.values() for page in result.get("pages", [])]
    failures = [page for page in pages if any("failed" in warning for warning in page.get("detector_warnings", []))]
    return {"documents_with_timing": len(times), "mean_wall_time_seconds": sum(times) / len(times) if times else None,
            "page_count": len(pages), "failure_pages": len(failures), "failure_rate": len(failures) / len(pages) if pages else None}


def _metadata(predictions: dict[str, dict]) -> list[dict]:
    entries = []
    for key, value in predictions.items():
        versions = sorted({str(region.get("attributes", {}).get("model_version")) for page in value.get("pages", [])
                           for region in page.get("diagram_regions", []) if region.get("attributes", {}).get("model_version")})
        entries.append({"document_id": key, "processing_metadata": value.get("processing_metadata", {}), "diagram_model_versions": versions})
    return entries


def _write_error_visuals(root: Path, output: Path, annotations: list[dict], predictions: dict[str, dict], errors: list[dict]) -> list[str]:
    output.mkdir(parents=True, exist_ok=True); paths: list[str] = []
    by_doc = {item["document_id"]: item for item in annotations}
    for error in errors[:30]:
        doc, page_id = error["key"]; annotation = by_doc[doc]
        page = next(item for item in annotation["pages"] if item["page_id"] == page_id)
        image_path = root / page.get("image", "")
        if not image_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = predictions.get(doc, {}); predicted_page = next((item for item in result.get("pages", []) if item["page_id"] == page_id), {})
        for item in page.get("regions", []): _draw(image, item, (0, 180, 0), "GT")
        for item in page.get("anchors", []): _draw(image, item, (0, 180, 0), "GT-A")
        for item in (predicted_page.get("regions", []) + predicted_page.get("ocr_lines", []) + predicted_page.get("diagram_regions", []) + predicted_page.get("question_anchor_candidates", [])):
            _draw(image, item, (0, 0, 255), "PRED")
        path = output / f"{doc}-{page_id}-{error['kind']}-{error['type']}.png"; cv2.imwrite(str(path), image); paths.append(str(path))
    return paths


def _draw(image, item: dict, color: tuple[int, int, int], label: str) -> None:
    try:
        box = _box(item); x1, y1, x2, y2 = map(round, box.as_xyxy()); cv2.rectangle(image, (x1, y1), (x2, y2), color, 2); cv2.putText(image, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    except (KeyError, TypeError, ValueError):
        return


def _failure_analysis(errors: list[dict], annotations: list[dict]) -> list[dict]:
    tags = {(doc["document_id"], page["page_id"]): page.get("tags", []) for doc in annotations for page in doc.get("pages", [])}
    counter: Counter[str] = Counter()
    for error in errors:
        for tag in tags.get(tuple(error["key"]), ["untagged"]): counter[f"{error['kind']}:{tag}"] += 1
    return [{"category": category, "error_count": count} for category, count in counter.most_common()]


def _write_failure_markdown(path: Path, variant: str, metrics: dict) -> None:
    failures = metrics["failure_analysis"]
    source = "No errors were available." if not failures else "\n".join(f"- {item['category']}: {item['error_count']}" for item in failures)
    recommendation = "Collect real handwritten annotations before deciding on custom training." if not failures else "Prioritize annotation review and model/error inspection for the listed categories; consider custom training only after repeated real-scan errors are measured."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Failure analysis: {variant}\n\n## Prioritized observed errors\n\n{source}\n\n## Recommendation\n\n{recommendation}\n", encoding="utf-8")
