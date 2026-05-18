"""Supervised road-decoder training entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_spec = {
        "stage": "train_road_decoder",
        "data_root": str(args.data_root),
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "config": config,
        "status": "scaffold_only",
        "next_step": "Implement dense decoder training for road masks or centerlines.",
    }

    (args.output_dir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2), encoding="utf-8"
    )

    print("Prepared road-decoder run specification.")
    print("TODO: implement supervised decoder training.")


if __name__ == "__main__":
    main()
