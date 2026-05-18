"""Submit a geospatial JEPA training run to Slurm with submitit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import submitit
import yaml

from geo_vlm.training.pretrain_jepa import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--tasks-per-node", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--time", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


class Trainer:
    def __init__(self, config_path: Path, output_dir: Path, epochs: int | None) -> None:
        self.config_path = config_path
        self.output_dir = output_dir
        self.epochs = epochs

    def __call__(self) -> None:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        output_dir = Path(str(self.output_dir.expanduser()).replace("%j", job_id).replace("%JOBID", job_id))
        if self.epochs is not None:
            config["optimization"]["epochs"] = self.epochs
        config["logging"]["folder"] = str(output_dir)
        train(config, output_dir)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    log_root = Path(config["logging"]["folder"]).expanduser()
    output_dir = (args.output_dir or (log_root / "runs" / "%j")).expanduser()
    folder = (args.folder or (log_root / "jobs")).expanduser()
    launcher_config = config.get("launcher", {})
    partition = args.partition or launcher_config.get("partition") or "genai"
    nodes = args.nodes or launcher_config.get("nodes") or 1
    tasks_per_node = args.tasks_per_node or launcher_config.get("tasks_per_node") or 1
    num_gpus = args.num_gpus or launcher_config.get("num_gpus") or tasks_per_node
    time_minutes = args.time or launcher_config.get("time_minutes") or 30

    executor = submitit.AutoExecutor(folder=str(folder / "job_%j"))
    executor.update_parameters(
        slurm_partition=partition,
        timeout_min=time_minutes,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=num_gpus,
    )
    job = executor.submit(Trainer(args.config, output_dir, args.epochs))
    print(job.job_id)


if __name__ == "__main__":
    main()
