# Diagram-region training and evaluation

## Before training

There is no generic pretrained YOLO checkpoint validated for handwritten academic
answer sheets. A general checkpoint may be used to initialize training, but it
must not be enabled for production diagram detection until it has been trained
and evaluated on this task.

Label exactly one class, `diagram_region`, as the inclusive visual extent of a
student diagram, graph, geometric construction, circuit, or labeled figure.
Do not label ordinary handwriting, question numbers, prose, equations that have
no visual figure, or tables unless the project later defines separate classes.
Record ambiguous cases for adjudication rather than silently guessing.

Use one YOLO text file per image: `class_id x_center y_center width height`,
where values are normalized to `[0, 1]`. Keep every page from the same source
document (preferably the same student) in exactly one split. This prevents
handwriting/layout leakage across train, validation, and test.

Expected layout:

```text
diagram_dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
  split-manifest.json
```

Copy `datasets/diagram_regions.yaml.example` and replace its `path` value.

## Train

Install the optional detector dependency, then initialize from a general
checkpoint only as transfer learning:

```powershell
python -m pip install -e ".[dev,yolo]"
yolo detect train data=.\datasets\diagram_regions.yaml model=yolo26n.pt epochs=100 imgsz=1536 project=.\runs name=diagram-region-v1
```

The resulting `best.pt` is a candidate task-trained model, not evidence of
quality by itself. Store its run ID, source data revision, class definition, and
validation report as its model version.

## Evaluate the held-out split

```powershell
python .\scripts\evaluate_diagram_yolo.py --model .\runs\diagram-region-v1\weights\best.pt --data .\datasets\diagram_regions.yaml --split-manifest .\datasets\split-manifest.json --output .\runs\diagram-region-v1\evaluation.json
```

The generated report contains precision, recall, mAP50, mAP50-95, per-image
metrics, and examples with false negatives when the framework exposes them.
Manually review saved validation plots and false-positive examples; mAP alone is
not a sufficient answer-sheet suitability decision.
