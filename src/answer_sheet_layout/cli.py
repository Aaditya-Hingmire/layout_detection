"""Command-line rendering endpoint."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import LayoutConfig, PreprocessingConfig, RenderingConfig, YoloDiagramConfig
from .errors import LayoutDetectionError
from .api import ProcessOptions, process_pdf
from .rendering import render_pdf
from .serialization import write_document_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="answer-sheet-layout")
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="render a PDF and save page artifacts")
    render.add_argument("pdf")
    render.add_argument("output_dir")
    render.add_argument("--dpi", type=int, default=200)
    render.add_argument("--deskew", action="store_true")
    render.add_argument("--enhance-contrast", action="store_true")
    render.add_argument("--denoise", action="store_true")
    render.add_argument("--verbose", action="store_true")
    analyze = commands.add_parser("analyze", help="render, run PP-StructureV3, and save debug artifacts")
    analyze.add_argument("pdf")
    analyze.add_argument("output_dir")
    analyze.add_argument("--dpi", type=int, default=200)
    analyze.add_argument("--deskew", action="store_true")
    analyze.add_argument("--enhance-contrast", action="store_true")
    analyze.add_argument("--denoise", action="store_true")
    analyze.add_argument("--no-visualizations", action="store_true")
    analyze.add_argument("--no-crops", action="store_true")
    analyze.add_argument("--enable-yolo", action="store_true", help="enable an explicitly task-trained diagram checkpoint")
    analyze.add_argument("--yolo-model", help="path to a custom diagram_region checkpoint")
    analyze.add_argument("--yolo-model-version", help="training/run version recorded with detections")
    analyze.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if args.command in {"render", "analyze"}:
        try:
            yolo = YoloDiagramConfig(
                enabled=getattr(args, "enable_yolo", False), model_path=getattr(args, "yolo_model", None),
                model_version=getattr(args, "yolo_model_version", None), trained_for_diagrams=getattr(args, "enable_yolo", False),
            )
            config = LayoutConfig(RenderingConfig(dpi=args.dpi), PreprocessingConfig(
                deskew=args.deskew, enhance_contrast=args.enhance_contrast, denoise=args.denoise,
            ), yolo_diagrams=yolo)
            if args.command == "render":
                result = render_pdf(args.pdf, args.output_dir, config)
            else:
                result = process_pdf(
                    args.pdf,
                    ProcessOptions(output_dir=args.output_dir, export_visualizations=not args.no_visualizations,
                                   export_crops=not args.no_crops),
                    config,
                )
        except (LayoutDetectionError, ValueError) as error:
            logging.error("%s", error)
            return 2
        output = write_document_result(result, result.pages[0].original_image_path.parent.parent / "result.json")
        logging.info("Processed %d page(s); metadata: %s", len(result.pages), output)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
