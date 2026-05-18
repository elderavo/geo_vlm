"""Submit a geospatial JEPA training run to Slurm with submitit."""

from __future__ import annotations

import argparse
from pathlib import Path

import submitit
import yaml

from geo_vlm.training.pretrain_jepa import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--tasks-per-node", type=int, default=1)
    parser.add_argument("--time", type=int, default=30)
    return parser.parse_args()


class Trainer:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def __call__(self) -> None:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        train(config, Path(config["logging"]["folder"]))


def main() -> None:
    args = parse_args()
    executor = submitit.AutoExecutor(folder=str(args.folder / "job_%j"))
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_mem_per_gpu="55G",
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=args.tasks_per_node,
    )
    job = executor.submit(Trainer(args.config))
    print(job.job_id)


if __name__ == "__main__":
    main()
