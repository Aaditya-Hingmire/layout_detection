# Question-wise spatial association

Question anchors remain OCR-pattern **candidates**. This stage does not confirm
question semantics and does not grade content.

The deterministic initial rules are:

1. Sort candidate anchors by original page order and original page coordinates.
2. Divide each page into vertical segments from one anchor to the next.
3. Associate OCR lines, layout regions, tables, equations, and wholly-contained
   canonical diagrams with that segment.
4. Treat content before a new page's first anchor as a tentative continuation of
   the last prior-page candidate, when one exists.
5. Leave diagrams crossing anchor boundaries, diagrams before a new anchor with a
   plausible prior-page continuation, and all content without evidence unresolved.
6. Preserve de-duplicated baseline visual regions as accounted-for evidence rather
   than emitting duplicate diagram associations.

Every associated item retains its original page ID, bounding box, detector source,
detector confidence, and crop reference. The JSON result is stored at
`result.json.question_association` after `analyze` completes.
`total_content_items`, `accounted_content_items`, and
`unaccounted_content_ids` make any accounting gap explicit.

No ML or LLM association is included. Add one only after reviewing measured
unresolved and misassociation cases on a document/student-disjoint corpus.
