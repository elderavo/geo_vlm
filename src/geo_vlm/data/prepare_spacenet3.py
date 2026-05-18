"""Prepare official SpaceNet 3 road data for downstream experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchgeo.datasets import SpaceNet3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_manifest(root: Path) -> dict[str, object]:
    """Inspect the canonical TorchGeo dataset and summarize the split.

    TorchGeo is used here as the dataset adapter instead of hand-parsing the
    directory layout. That follows the Adapter pattern: the rest of the project
    depends on a small stable interface while TorchGeo owns dataset-specific IO.
    """

    dataset = SpaceNet3(root=str(root), split="train", download=False)
    return {
        "dataset": "SpaceNet3",
        "split": "train",
        "count": len(dataset),
        "root": str(root),
        "notes": [
            "Official labels are the supervised baseline.",
            "Use unlabeled imagery separately for JEPA pretraining.",
        ],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.root)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
