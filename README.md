# Answer Sheet Layout Detection

This repository is a standalone, importable PDF layout-detection component. It supports rendering, preprocessing, replaceable OCR/layout adapters, optional diagram detection, and structural candidate-question association. It does not perform grading, marks processing, student records, databases, or UI work.

## Install

Use Python 3.10 or later in a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional Paddle baseline after selecting a compatible PaddlePaddle CPU
or GPU wheel for your OS/CUDA setup:

```powershell
python -m pip install -e ".[dev,paddle]"
```

YOLO is separately optional and disabled by default:

```powershell
python -m pip install -e ".[dev,yolo]"
```

If the `py` launcher is not installed, use the path to a Python 3.10+ executable instead.

## Render a PDF

```powershell
answer-sheet-layout render .\input.pdf .\artifacts --dpi 200 --deskew --enhance-contrast
```

The command writes immutable source renderings to `artifacts/original/` and separately writes any processed images to `artifacts/processed/`. It also writes `artifacts/result.json`, containing quality findings and the transformation chain needed to map processed-image coordinates back to the original render.

## Baseline layout and OCR analysis

```powershell
answer-sheet-layout analyze .\input.pdf .\artifacts --dpi 200
```

Use `--no-crops` or `--no-visualizations` when those artifacts are not wanted.

This writes the original page image, any processed image, `raw/paddle_pp_structure_v3/page-N.json`, and a
`visualizations/page-N.png` overlay plus `page-N-comparison.png` (original left, overlay right). Green boxes are PP-Structure layout regions,
blue boxes are OCR lines, and orange boxes are **candidate** question anchors. A
candidate is merely an OCR pattern match (`Q.1`, `Question 2`, `1(a)`); it is not a
question-to-answer association. Detector errors and unsupported raw result shapes are retained as warnings in `result.json`.

## Optional diagram-region detector

The module never runs a generic pretrained YOLO checkpoint as a diagram detector.
It requires an explicit model path/version and the `--enable-yolo` opt-in, which
declares that the supplied checkpoint was trained for the one supported class,
`diagram_region`.

```powershell
answer-sheet-layout analyze .\input.pdf .\artifacts --enable-yolo --yolo-model .\models\diagram-region-v1.pt --yolo-model-version diagram-region-v1
```

The analysis output includes `crops/diagrams/`, purple canonical diagram boxes in
the visualization, and `diagram-comparison.json`. Counts in that comparison are
observations, not accuracy metrics. See [the training guide](docs/diagram-training.md)
for document/student-level split requirements and measured evaluation commands.

## Question-wise structural association

`analyze` also writes `result.json.question_association`. It groups page-space
OCR lines and regions beneath **candidate** anchors using deterministic vertical
segments, marks cross-page assignments as tentative, and retains every ambiguous,
unassigned, or diagram-de-duplicated input in `unresolved_content`. See
[the association rules](docs/question-association.md).

## Benchmark

The annotation-driven benchmark evaluates baseline and baseline-plus-YOLO result
artifacts separately, without grading. See [the benchmark guide](docs/benchmark.md)
for the schema, document/student-level split rules, commands, machine-readable
metrics, visual error reports, and failure analysis.

## Python integration

Use the stable `process_pdf(...)` entry point and its validated `DocumentResult`.
See [the integration contract](docs/integration.md) for the public API, typed
errors, versioned JSON, model setup, and an integration example.

## Test

```powershell
python -m pytest
```

## Package layout

- `models.py`: stable page, region, bounding-box, and document result schemas.
- `config.py`: explicit rendering, limits, and preprocessing configuration.
- `rendering.py`: validated, ordered PyMuPDF rendering and artifact storage.
- `preprocessing.py`: conservative OpenCV quality checks and reversible transforms.
- `detectors.py`: replaceable detector interfaces and normalized analysis result.
- `paddle_adapter.py`: optional PP-StructureV3 adapter and defensive result normalization.
- `analysis.py`: analysis orchestration, raw artifact storage, and visual-debug output.
- `visualization.py`: original-page-coordinate overlays.
- `yolo_diagram.py`: optional task-trained YOLO adapter; disabled without explicit configuration.
- `diagrams.py`: detector de-duplication, crop storage, and comparison reports.
- `cli.py`: rendering endpoint for manual verification and integration smoke tests.
