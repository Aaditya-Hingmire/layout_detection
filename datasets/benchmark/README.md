# Layout benchmark dataset

Store one JSON annotation per complete answer sheet in `annotations/`, using
`annotation.example.json` as the schema. `source_type` must be either
`real_handwritten` or `synthetic`; reports always count them separately.

Use page tags to record difficult conditions, including `faint_handwriting`,
`ruled_background`, `small_diagram`, `skewed_page`, `subquestion`, and
`continuation_page`. Add images under `images/` when visual error reports are
needed.

Create `split-manifest.json` with `unit` set to `student_id` or `document_id`.
Every document/page for the same value must belong to only one split.

For each evaluated variant, store results as either
`<variant-results>/<document_id>.json` or
`<variant-results>/<document_id>/result.json`. These are ordinary module
`result.json` outputs from `answer-sheet-layout analyze`.
