"""Submit JEPA mask visualization to Slurm with submitit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import submitit
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--tasks-per-node", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--time", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    return parser.parse_args()


class MaskVisualizer:
    def __init__(
        self,
        *,
        config_path: Path,
        output_dir: Path,
        num_samples: int,
        seed: int,
        indices: list[int] | None,
    ) -> None:
        self.config_path = config_path
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.seed = seed
        self.indices = indices

    def __call__(self) -> None:
        from geo_vlm.analysis.visualize_jepa_masks import visualize

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        output_dir = Path(str(self.output_dir.expanduser()).replace("%j", job_id).replace("%JOBID", job_id))
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        visualize(
            config=config,
            output_dir=output_dir,
            num_samples=self.num_samples,
            seed=self.seed,
            indices=self.indices,
        )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    log_root = Path(config["logging"]["folder"]).expanduser()
    run_id = args.run_id
    if run_id is None and args.run_dir is not None:
        run_id = args.run_dir.expanduser().name
    output_dir = (args.output_dir or (log_root / "mask_visuals" / (run_id or "%j"))).expanduser()
    folder = (args.folder or (log_root / "mask_visual_jobs")).expanduser()
    launcher_config = config.get("launcher", {})
    partition = args.partition or launcher_config.get("partition") or "genai"
    nodes = args.nodes or 1
    tasks_per_node = args.tasks_per_node or 1
    num_gpus = args.num_gpus or launcher_config.get("num_gpus") or tasks_per_node

    executor = submitit.AutoExecutor(folder=str(folder / "job_%j"))
    executor.update_parameters(
        slurm_partition=partition,
        timeout_min=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=4,
        gpus_per_node=num_gpus,
    )
    job = executor.submit(
        MaskVisualizer(
            config_path=args.config,
            output_dir=output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            indices=args.indices,
        )
    )
    print(job.job_id)


if __name__ == "__main__":
    main()
