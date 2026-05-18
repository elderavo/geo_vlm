"""Convert dense road predictions into vector road features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_spec = {
        "stage": "vectorize_roads",
        "data_root": str(args.data_root),
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "status": "scaffold_only",
        "expected_output": "road_segments.geojson",
        "next_step": "Implement thresholding, skeletonization, snapping, and graph creation.",
    }

    (args.output_dir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2), encoding="utf-8"
    )

    print("Prepared vectorization run specification.")
    print("TODO: implement raster-to-vector road conversion.")


if __name__ == "__main__":
    main()
