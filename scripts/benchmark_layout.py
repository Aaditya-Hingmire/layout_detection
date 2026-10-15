"""CLI entry point for annotation-driven layout benchmark evaluation."""

from __future__ import annotations

import argparse

from answer_sheet_layout.benchmark import evaluate_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=RESULTS_DIR")
    args = parser.parse_args()
    variants = {}
    for value in args.variant:
        name, separator, directory = value.partition("=")
        if not separator or not name or not directory:
            parser.error("--variant must be NAME=RESULTS_DIR")
        variants[name] = directory
    evaluate_benchmark(args.dataset, variants, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
