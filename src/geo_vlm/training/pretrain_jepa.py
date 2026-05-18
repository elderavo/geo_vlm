"""Self-supervised JEPA pretraining entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_spec = {
        "stage": "pretrain_jepa",
        "data_root": str(args.data_root),
        "config": config,
        "status": "scaffold_only",
        "next_step": "Implement JEPA encoder/target encoder training loop.",
    }

    (args.output_dir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2), encoding="utf-8"
    )

    print("Prepared JEPA run specification.")
    print("TODO: implement the actual JEPA training loop.")


if __name__ == "__main__":
    main()
