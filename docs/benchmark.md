# Reproducible layout benchmark

## Prepare annotations

Copy `datasets/benchmark/annotation.example.json` into a benchmark dataset's
`annotations/` directory once per complete answer sheet. Do not split page crops
from the same answer sheet or student across train, validation, and test.

Annotate text boxes, `diagram_region` boxes, question anchors, and the intended
anchor-to-region structural association. Tag difficult pages: faint handwriting,
ruled backgrounds, small diagrams, skew, subquestions, and continuations.

## Generate prediction artifacts

Run both variants over the same held-out documents. Keep their `result.json`
outputs in separate directories, named by document ID.

```powershell
answer-sheet-layout analyze .\heldout\sheet-001.pdf .\runs\baseline\sheet-001
answer-sheet-layout analyze .\heldout\sheet-001.pdf .\runs\baseline-plus-yolo\sheet-001 --enable-yolo --yolo-model .\models\diagram-region-v1.pt --yolo-model-version diagram-region-v1
```

## Evaluate

```powershell
python .\scripts\benchmark_layout.py --dataset .\benchmark-data --output .\benchmark-output --variant baseline=.\runs\baseline --variant baseline_plus_yolo=.\runs\baseline-plus-yolo
```

The evaluator writes `metrics.json`, per-variant `failure-analysis.md`, and up
to 30 page overlays in `visual-errors/` when annotation image paths are present.
It measures IoU-0.5 precision, recall, F1, and AP/mAP50 for localized text and
diagrams; exact-text-plus-IoU anchor precision/recall; association-member
accuracy; coordinate in-bounds rate; wall time; and detector failure rate.

Metrics are reported per variant and source counts identify synthetic versus real
handwritten evidence. Do not make deployment or custom-training decisions from
synthetic results alone. The prioritized failure report is derived from observed
false positives and false negatives by page tag.
