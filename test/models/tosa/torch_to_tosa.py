from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_mlir import fx as torch_mlir_fx

from reference_models import MODEL_BUILDERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a reference model to TOSA MLIR.")
    parser.add_argument(
        "model",
        nargs="?",
        default="mobilenet_v2",
        choices=sorted(MODEL_BUILDERS),
        help="Reference model to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the generated MLIR. Defaults to <model>_tosa.mlir.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available reference models and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        for name in sorted(MODEL_BUILDERS):
            print(name)
        return

    model, example_inputs = MODEL_BUILDERS[args.model]()
    output_path = args.output or Path(f"{args.model}_tosa.mlir")

    tosa_mlir = torch_mlir_fx.export_and_import(
        model,
        *example_inputs,
        output_type=torch_mlir_fx.OutputType.TOSA,
    )

    output_path.write_text(str(tosa_mlir))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()