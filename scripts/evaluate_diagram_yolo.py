"""Run measured validation for a task-trained one-class diagram detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _validate_split_manifest(path: Path | None) -> list[str]:
    if path is None:
        return ["No split manifest supplied; document/student leakage was not checked."]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = {name: set(values) for name, values in manifest["splits"].items()}
    warnings: list[str] = []
    for first in splits:
        for second in splits:
            if first < second and splits[first] & splits[second]:
                raise ValueError(f"split leakage: {first} and {second} share IDs")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    warnings = _validate_split_manifest(args.split_manifest)
    from ultralytics import YOLO

    metrics = YOLO(args.model).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True, save_json=True)
    box = metrics.box
    image_metrics = getattr(box, "image_metrics", {})
    false_negatives = [name for name, value in image_metrics.items() if value.get("fn", 0)] if isinstance(image_metrics, dict) else []
    report = {
        "model": args.model,
        "data": args.data,
        "metrics": {"precision": box.mp, "recall": box.mr, "map50": box.map50, "map50_95": box.map,
                    "map75": box.map75},
        "false_negative_examples": false_negatives,
        "per_image_metrics": image_metrics,
        "warnings": warnings,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
