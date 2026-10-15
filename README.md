# Answer Sheet Layout Detection

Answer Sheet Layout Detection is a Python component that turns an answer-sheet
PDF into page images and structured layout data. It can identify text regions,
OCR lines, question-label candidates, and (when an explicitly trained model is
provided) diagram regions.

This project is a layout-analysis foundation. It does **not**:

- grade answers or calculate marks;
- identify students or manage student records;
- decide whether an answer is correct;
- provide a web or desktop user interface; or
- guarantee that every OCR question label is a real question.

The main input is one readable, unencrypted PDF. The main machine-readable
output is `result.json`. Human verification is done with the generated page
images and visualization overlays.

## How the project works

The normal analysis pipeline is:

1. Validate the PDF.
2. Render every page to a PNG at the requested DPI.
3. Measure page quality (blur, contrast, and possible skew).
4. Optionally deskew, enhance contrast, or denoise the page.
5. Run the configured layout/OCR detector.
6. Detect question-label candidates using conservative OCR patterns.
7. Optionally detect diagrams with a task-trained YOLO checkpoint.
8. Associate content with question candidates using deterministic page-space
   rules.
9. Save JSON, raw detector responses, crops, and visualizations.

Coordinates in the result are pixel coordinates in the original rendered page
space. The top-left corner is `(0, 0)`.

## Requirements

- Windows PowerShell, macOS/Linux shell, or an equivalent terminal.
- Python 3.10 or newer.
- A readable `.pdf` file.
- A writable output directory.

The default rendering limits are:

| Setting | Default |
| --- | ---: |
| Rendering resolution | 200 DPI |
| Maximum pages | 500 |
| Maximum rendered pixels per page | 40,000,000 |

Encrypted PDFs, empty PDFs, invalid files, and PDFs that exceed these limits
are rejected.

## Installation

Run these commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If the `py` launcher is unavailable, replace `py -3.11` with the path to a
Python 3.10+ executable.

The base installation supports PDF rendering, preprocessing, result
serialization, and tests. Full OCR/layout analysis requires the optional
PaddleOCR installation:

```powershell
python -m pip install -e ".[dev,paddle]"
```

PaddlePaddle may require a compatible CPU or GPU wheel for your operating
system and CUDA version. Install that compatible runtime according to the
PaddlePaddle documentation before running the analysis command.

Diagram detection is a separate optional feature:

```powershell
python -m pip install -e ".[dev,yolo]"
```

Installing the YOLO package alone is not enough. You also need a checkpoint
trained for this project's `diagram_region` class. A generic pretrained object
detection checkpoint must not be treated as a validated answer-sheet diagram
detector.

## Quick start

Assume the input file is:

```text
D:\Layout-Detection\input.pdf
```

### 1. Render and inspect the PDF

This is the simplest smoke test and does not require PaddleOCR:

```powershell
answer-sheet-layout render `
    .\input.pdf `
    .\artifacts-render `
    --dpi 200 `
    --enhance-contrast
```

If the installed console command is not found, use:

```powershell
python -m answer_sheet_layout.cli render `
    .\input.pdf `
    .\artifacts-render `
    --dpi 200 `
    --enhance-contrast
```

### 2. Run baseline layout and OCR analysis

Install the Paddle extras first, then run:

```powershell
answer-sheet-layout analyze `
    .\input.pdf `
    .\artifacts-analysis `
    --dpi 200 `
    --enhance-contrast `
    --verbose
```

Useful preprocessing options are:

```text
--deskew             Correct estimated page skew.
--enhance-contrast   Apply conservative CLAHE contrast enhancement.
--denoise            Apply denoising.
--no-crops           Do not export diagram crops.
--no-visualizations  Do not export overlay images.
--verbose            Enable detailed logging.
```

### 3. Run diagram detection

Use this only with a task-trained diagram checkpoint:

```powershell
answer-sheet-layout analyze `
    .\input.pdf `
    .\artifacts-yolo `
    --dpi 200 `
    --enhance-contrast `
    --enable-yolo `
    --yolo-model .\models\diagram-region-v1.pt `
    --yolo-model-version diagram-region-v1 `
    --verbose
```

The model version is recorded in the processing metadata so that output can be
traced to a particular training run.

## Where to find the output

For an analysis output directory such as `artifacts-analysis`, the usual
structure is:

```text
artifacts-analysis/
|-- result.json
|-- diagram-comparison.json
|-- original/
|   |-- page-1.png
|   `-- page-2.png
|-- processed/
|   |-- page-1.png
|   `-- page-2.png
|-- raw/
|   `-- paddle_pp_structure_v3/
|       |-- page-1.json
|       `-- page-2.json
|-- visualizations/
|   |-- page-1.png
|   |-- page-1-comparison.png
|   `-- ...
`-- crops/
    `-- diagrams/
        |-- page-1/
        `-- page-2/
```

Some directories are intentionally absent when their feature is disabled or
when no corresponding artifact was produced.

Open the generated images in VS Code or File Explorer:

```powershell
code .\artifacts-analysis\visualizations
ii .\artifacts-analysis\visualizations
ii .\artifacts-analysis\crops\diagrams
```

The visualization colors are:

| Color | Meaning |
| --- | --- |
| Green | Baseline layout regions |
| Blue | OCR lines |
| Orange | Question-anchor candidates |
| Purple | Canonical diagram regions |

`page-N-comparison.png` shows the original page beside its overlay. Use
`original/page-N.png` to verify the source rendering and
`processed/page-N.png` to verify the image supplied to the detector.

## Understanding `result.json`

The result follows schema version `1.0` and contains:

```json
{
  "schema_version": "1.0",
  "source_pdf": "...",
  "pages": [],
  "question_association": null,
  "processing_metadata": {}
}
```

Each item in `pages` includes:

| Field | What it tells you |
| --- | --- |
| `page_id` | Stable identifier such as `page-1` |
| `width_px`, `height_px` | Rendered image dimensions |
| `original_image_path` | Immutable source rendering |
| `processed_image_path` | Detector input after preprocessing, if any |
| `quality` | Blur, contrast, skew estimate, and quality warnings |
| `regions` | Baseline layout regions |
| `ocr_lines` | Recognized text lines and their boxes |
| `question_anchor_candidates` | OCR lines matching question-label patterns |
| `diagram_regions` | Merged diagram detections |
| `raw_result_paths` | Saved raw detector responses |
| `detector_warnings` | Detector failures and unsupported-result warnings |

A diagram entry contains a bounding box, detector source, and optionally a
confidence and crop reference. `diagram_regions` being empty is not enough to
conclude that a page has no diagram: first check `detector_warnings`.

### Inspect the result with PowerShell

```powershell
$result = Get-Content .\artifacts-analysis\result.json -Raw |
    ConvertFrom-Json

# Page count and schema version
$result.pages.Count
$result.schema_version

# Counts by page
$result.pages |
    Select-Object page_id,
        @{Name="Regions"; Expression={$_.regions.Count}},
        @{Name="OCRLines"; Expression={$_.ocr_lines.Count}},
        @{Name="Anchors"; Expression={$_.question_anchor_candidates.Count}},
        @{Name="Diagrams"; Expression={$_.diagram_regions.Count}}

# Diagram boxes and confidence
$result.pages |
    ForEach-Object {
        $_.diagram_regions |
            Select-Object id, page_id, bounding_box, confidence, source_detector
    }

# Detector warnings
$result.pages |
    ForEach-Object {
        [PSCustomObject]@{
            Page = $_.page_id
            Warnings = ($_.detector_warnings -join "; ")
        }
    }
```

### Inspect the result with Python

```powershell
python -c "import json; from pathlib import Path; p=json.loads(Path(r'.\artifacts-analysis\result.json').read_text()); print('schema=', p['schema_version'], 'pages=', len(p['pages'])); [print(x['page_id'], 'regions=', len(x['regions']), 'ocr=', len(x['ocr_lines']), 'diagrams=', len(x['diagram_regions']), 'warnings=', len(x['detector_warnings'])) for x in p['pages']]"
```

Print all OCR text:

```powershell
python -c "import json; from pathlib import Path; p=json.loads(Path(r'.\artifacts-analysis\result.json').read_text()); [print(line['text']) for page in p['pages'] for line in page['ocr_lines']]"
```

## How to decide whether detection worked

### Rendering worked when

- `result.json` exists.
- `pages` contains the expected number of pages.
- `original/page-N.png` exists for every page.
- Each page has positive `width_px` and `height_px`.
- There are no input or rendering errors.

### Baseline OCR/layout worked when

- Paddle dependencies are installed and compatible.
- `detector_warnings` does not contain
  `paddle_pp_structure_v3_failed`.
- Raw files exist under `raw/paddle_pp_structure_v3/`.
- `regions` and/or `ocr_lines` contain detections where expected.
- The visualization boxes align with the page content.

An empty `regions` array can mean that no regions were detected, but it can
also mean the detector failed. Always inspect `detector_warnings`.

### Diagram detection worked when

- YOLO was explicitly enabled.
- The supplied checkpoint is trained for `diagram_region`.
- Expected boxes appear in `diagram_regions`.
- Purple boxes appear in the visualization.
- Corresponding crops appear under `crops/diagrams/`.
- The model has been evaluated on held-out documents.

`diagram-comparison.json` reports observed counts. Its precision, recall, and
mAP values remain unavailable until ground-truth annotations and a held-out
evaluation split are provided.

## Question association

The analyzer writes `question_association` into `result.json`. It groups OCR
lines, layout regions, and canonical diagrams beneath **candidate** question
anchors using deterministic vertical page segments.

The association is intentionally conservative:

- an anchor is an OCR pattern match, not a confirmed question;
- content crossing an anchor boundary may be unresolved;
- content before a page's first anchor may be treated as a continuation;
- ambiguous content is retained in `unresolved_content`;
- no semantic answer understanding or grading is performed.

The rules are documented in
[docs/question-association.md](docs/question-association.md).

## Python API

Use `process_pdf` when integrating the component into another Python program:

```python
from pathlib import Path

from answer_sheet_layout import LayoutConfig, ProcessOptions, process_pdf

result = process_pdf(
    Path("answer-sheet.pdf"),
    ProcessOptions(output_dir=Path("artifacts")),
    LayoutConfig(),
)

for page in result.pages:
    print(
        page.page_id,
        len(page.regions),
        len(page.ocr_lines),
        len(page.diagram_regions),
    )
```

`process_pdf` returns a validated `DocumentResult` and writes
`result.json` by default. Detectors can be replaced by supplying
`layout_detector` or `diagram_detector`. Set
`use_paddle_layout=False` for rendering/preprocessing and association tests
without the Paddle model.

Typed errors include:

- `InputValidationError`: invalid, empty, encrypted, or oversized PDF.
- `RenderingError`: failure while rendering or writing an image.
- `ModelUnavailableError`: requested model/runtime is unavailable.
- `ProcessingError`: detector failure in strict mode.
- `ResultValidationError`: result violates the public schema invariants.

The full integration contract is in
[docs/integration.md](docs/integration.md).

## Training and evaluating diagram detection

For custom YOLO training:

1. Label only the visual extent of diagrams as `diagram_region`.
2. Keep every page from one source document, preferably one student, in one
   split only.
3. Use train, validation, and test splits with no document/student leakage.
4. Evaluate on held-out pages before using the model for analysis.

The expected dataset layout, label format, training command, and evaluation
command are in [docs/diagram-training.md](docs/diagram-training.md).

## Benchmarking

The annotation-driven benchmark compares baseline and baseline-plus-YOLO
artifacts without grading answers:

```powershell
python .\scripts\benchmark_layout.py `
    --dataset .\benchmark-data `
    --output .\benchmark-output `
    --variant baseline=.\runs\baseline `
    --variant baseline_plus_yolo=.\runs\baseline-plus-yolo
```

It produces machine-readable metrics, failure analysis, and visual error
reports. The annotation schema and split requirements are documented in
[docs/benchmark.md](docs/benchmark.md).

## Tests

Run the test suite from the activated virtual environment:

```powershell
python -m pytest
```

## Troubleshooting

### `answer-sheet-layout` is not recognized

Activate the virtual environment and reinstall the package:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Alternatively use `python -m answer_sheet_layout.cli ...`.

### Paddle initialization fails

Install the optional extras and verify the PaddlePaddle version is compatible
with your Python version and hardware:

```powershell
python -m pip install -e ".[dev,paddle]"
```

The run may still produce rendered images and a `result.json`, but OCR/layout
fields should not be trusted while `detector_warnings` reports a Paddle
failure.

### No diagrams are detected

Check that:

1. `--enable-yolo` was supplied.
2. `--yolo-model` points to an existing checkpoint.
3. `--yolo-model-version` is supplied.
4. The checkpoint was trained for `diagram_region`.
5. `diagram_regions`, `detector_warnings`, and `diagram-comparison.json` were
   inspected together.

## Repository guide

- [src/answer_sheet_layout/models.py](src/answer_sheet_layout/models.py):
  versioned page, region, bounding-box, and document result models.
- [src/answer_sheet_layout/config.py](src/answer_sheet_layout/config.py):
  rendering, preprocessing, detector, and association configuration.
- [src/answer_sheet_layout/rendering.py](src/answer_sheet_layout/rendering.py):
  PDF validation, rendering, and artifact storage.
- [src/answer_sheet_layout/preprocessing.py](src/answer_sheet_layout/preprocessing.py):
  image-quality checks and reversible preprocessing.
- [src/answer_sheet_layout/paddle_adapter.py](src/answer_sheet_layout/paddle_adapter.py):
  optional PP-StructureV3 adapter and result normalization.
- [src/answer_sheet_layout/analysis.py](src/answer_sheet_layout/analysis.py):
  end-to-end analysis orchestration and debug artifact generation.
- [src/answer_sheet_layout/visualization.py](src/answer_sheet_layout/visualization.py):
  original-coordinate overlay generation.
- [src/answer_sheet_layout/yolo_diagram.py](src/answer_sheet_layout/yolo_diagram.py):
  optional task-trained diagram detector.
- [src/answer_sheet_layout/diagrams.py](src/answer_sheet_layout/diagrams.py):
  diagram merging, crop storage, and comparison reports.
- [src/answer_sheet_layout/cli.py](src/answer_sheet_layout/cli.py):
  `render` and `analyze` command-line entry points.
