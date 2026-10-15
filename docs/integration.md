# Python integration contract

## Stable public API

```python
from pathlib import Path
from answer_sheet_layout import LayoutConfig, ProcessOptions, process_pdf

result = process_pdf(
    Path("answer-sheet.pdf"),
    ProcessOptions(output_dir=Path("artifacts")),
    LayoutConfig(),
)
```

`process_pdf(pdf_path, options, config=None, *, layout_detector=None,
diagram_detector=None)` is the stable entry point. It returns a validated
`DocumentResult` and writes `result.json` unless
`ProcessOptions(write_result_json=False)` is selected.

`ProcessOptions` controls the output directory, Paddle use, crop export,
visualization export, strict detector failures, and JSON writing. No absolute
paths, machine settings, or secrets are stored in package configuration.

## Input and output

Input is a readable, unencrypted PDF path plus an explicit writable output
directory. `LayoutConfig` controls rendering/preprocessing and optional model
configuration. Inject `layout_detector` or `diagram_detector` to use a different
implementation behind the corresponding protocol.

Output uses `schema_version: "1.0"` and contains the source document reference, ordered pages,
original-coordinate regions, OCR lines, candidate anchors, diagram regions,
question association, warnings, artifact paths, and processing metadata.
Use `serialize_document_result(result)` for versioned JSON and
`validate_document_result(result)` before crossing an application boundary.

## Errors

- `InputValidationError`: missing, malformed, encrypted, oversized, or empty PDF.
- `RenderingError`: a valid PDF cannot be rendered or an image artifact cannot be written.
- `ModelUnavailableError`: requested Paddle/YOLO runtime or checkpoint is unavailable.
- `ProcessingError`: a detector failed in strict mode.
- `ResultValidationError`: a result violates page/region/accounting invariants.

Set `strict_detector_failures=False` to retain detector failures as page warnings
for diagnostic runs. This does not hide them from the returned result.

## Dependency and model setup

```powershell
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,paddle]" # PP-StructureV3 / PaddleOCR
python -m pip install -e ".[dev,yolo]"   # optional task-trained diagram detector
```

Paddle requires a compatible PaddlePaddle CPU/GPU installation. YOLO remains
disabled until a task-trained `diagram_region` checkpoint and model version are
configured. See `docs/diagram-training.md` for training/evaluation requirements.

## Minimal result example

```json
{
  "schema_version": "1.0",
  "pages": [{
    "page_id": "page-1",
    "regions": [{"id": "page-1-region-1", "page_id": "page-1", "region_class": "printed_text"}],
    "question_anchor_candidates": []
  }],
  "question_association": null
}
```

The actual result includes bounding boxes, source provenance, confidence where
available, crop references, warnings, and transformation metadata.
