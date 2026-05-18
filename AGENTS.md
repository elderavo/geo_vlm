# Agent Notes

## Cluster Environment

- The interactive shell is on the `submit` node by default.
- The `submit` node has no GPUs and limited compute.
- Do not run GPU training, meaningful smoke training, or heavy validation directly on the submit node.
- Use Slurm via `submitit` for GPU tasks.
- Default GPU partition for this project: `genai`.
- The `beards` partition is also GPU-capable, but use `genai` unless the user says otherwise.
- Do not specify `mem_per_gpu` for submitit jobs.
- JEPA logs and training artifacts should go under `~/ijepa_logs`.
- Scratch or temporary files should go under `~/tmp`, not `/tmp`.
- Khartoum RGB GeoTIFFs are located at `~/AOI_5_Khartoum/PS-RGB`.

## Execution Guidance

- Ask before running nontrivial commands, including training, smoke training, Slurm submission, dependency sync/install, or long tests.
- Cheap inspections, syntax checks, and config reads are fine.
- Keep `configs/jepa.yaml` aligned with the cluster defaults above unless the user explicitly changes the target environment.
