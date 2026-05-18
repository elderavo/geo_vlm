# geo-vlm

`geo-vlm` is an MVP research pipeline for converting satellite imagery into standards-based geospatial outputs.

The first target is road extraction from the SpaceNet 3 Khartoum dataset:

```text
SpaceNet imagery
  -> JEPA encoder
  -> shared image embeddings
  -> road-specific decoder head
  -> road mask / centerline prediction
  -> deterministic vectorization
  -> GeoSPARQL-compatible WKT
```

## Core idea

The JEPA encoder is the reusable backbone. It learns latent image structure from unlabeled imagery through self-supervised training.

Task-specific decoder heads sit on top of that shared representation:

- a **road head** learns to predict road masks or centerlines,
- a future **building head** could learn to predict building masks or polygons,
- later heads can reuse the same encoder for additional geospatial objects.

This keeps the system extensible without forcing every downstream task to relearn image features from scratch.

## Why the pipeline is split this way

The model should predict geospatial structure, not text serialization.

For roads:

1. the JEPA encoder produces reusable image features,
2. the road decoder head converts those features into road-specific outputs,
3. deterministic post-processing converts road predictions into vector geometry,
4. deterministic export writes those vectors as GeoSPARQL WKT.

That separation keeps the system easier to debug:

- if perception fails, inspect the decoder output,
- if geometry fails, inspect vectorization,
- if RDF fails, inspect serialization.

## MVP workflow

1. Prepare the official SpaceNet 3 road labels with TorchGeo.
2. Pretrain a JEPA encoder on unlabeled imagery.
3. Train a supervised road decoder using the official labels.
4. Vectorize predicted road masks into road geometries.
5. Export those geometries as GeoSPARQL-compatible RDF/WKT.

The staged shell entrypoints live in [`scripts/`](scripts/).

## Repository layout

```text
configs/
  jepa.yaml
  road_decoder.yaml

scripts/
  00_setup_env.sh
  01_prepare_spacenet3.sh
  02_pretrain_jepa.sh
  03_train_road_decoder.sh
  04_vectorize_roads.sh
  05_export_geosparql.sh
  06_run_pipeline.sh

src/geo_vlm/
  data/
  training/
  postprocess/
  export/
```

## Current implementation status

Implemented:

- project and stage scaffolding,
- TorchGeo-based SpaceNet preparation entrypoint,
- direct `PS-RGB` GeoTIFF vanilla I-JEPA pretraining path,
- upstream-style I-JEPA ViT encoder, predictor, masking, optimizer, and schedules adapted from the local `~/ijepa` checkout,
- configurable JEPA input channels for later non-RGB experiments,
- submitit launcher for Slurm JEPA runs,
- deterministic GeoSPARQL exporter,
- staged shell workflow.

Still to implement:

- supervised road decoder training loop,
- raster-to-vector road post-processing,
- evaluation and experiment tracking.

## Geospatial JEPA baseline

The first geospatial pretraining milestone is intentionally conservative:

- imagery source: raw SpaceNet 3 Khartoum `PS-RGB` GeoTIFFs,
- model input: three channels,
- goal: prove that the vanilla JEPA recipe trains on satellite imagery before adding extra spectral products.

This is a **geospatial transfer baseline**, not yet a multiband experiment.

The current `configs/jepa.yaml` expects TIFFs under:

```text
~/AOI_5_Khartoum/PS-RGB
```

and discovers files matching:

```text
*_PS-RGB_*.tif
```

Before training, the pipeline writes a `dataset_manifest.json` with representative TIFF metadata such as channel count, dtype, dimensions, CRS, and value ranges.

Run locally:

```bash
uv run python -m geo_vlm.training.pretrain_jepa \
  --config configs/jepa.yaml
```

This writes `latest.pt`, `best.ckpt`, `config_resolved.yaml`,
`dataset_manifest.json`, and a JSONL training log. For the current validation
milestone, treat those artifacts as the baseline evidence that vanilla I-JEPA
can run stably on Khartoum satellite crops before adding road-specific decoder
training or geospatial vectorization.

Submit through Slurm with `submitit`:

```bash
uv run python -m geo_vlm.training.launch_jepa_submitit \
  --config configs/jepa.yaml
```

By default, the launcher targets the `genai` Slurm partition, requests one GPU
task, writes submitit logs under `~/ijepa_logs/jobs`, and writes training
artifacts under `~/ijepa_logs/runs/<job_id>`. Do not run GPU training directly
on the submit node.

Run a longer Khartoum validation without editing the config:

```bash
uv run python -m geo_vlm.training.launch_jepa_submitit \
  --config configs/jepa.yaml \
  --epochs 50 \
  --time 120 \
  --num-gpus 4
```

`--num-gpus` controls the Slurm GPU allocation. The current training loop still
runs one Python training process, so multi-GPU allocation is useful for
experimentation and upcoming distributed work but does not by itself enable
distributed data parallel training.

Generate first-pass mask visualizations:

```bash
uv run python -m geo_vlm.analysis.launch_mask_visuals_submitit \
  --config configs/jepa.yaml
```

Attach mask visualizations to a specific training run:

```bash
uv run python -m geo_vlm.analysis.launch_mask_visuals_submitit \
  --config configs/jepa.yaml \
  --run-id 25369098
```

These images overlay sampled I-JEPA context and target masks on stretched
Khartoum RGB crops. They are **not learned attention maps**; they show whether
the vanilla I-JEPA masking task is hiding and revealing geospatially relevant
content before we inspect learned embeddings. The launcher imports the image
visualization code inside the Slurm job, so OpenCV/image-library issues on the
submit node do not block submission. Outputs default to
`~/ijepa_logs/mask_visuals/<run_id>` when `--run-id` or `--run-dir` is provided,
otherwise `~/ijepa_logs/mask_visuals/<visualizer_job_id>`. You can also pass an
explicit `--output-dir` containing `%j` or `%JOBID`, which will be replaced by
the Slurm job id inside the compute job.

Roadmap for additional imagery products:

1. `PS-RGB` control run,
2. explicit support for PAN / multispectral input channels,
3. controlled comparisons so additional bands earn their complexity.

## Environment setup

This project uses `uv`.

```bash
uv sync
```

Run each stage independently while developing:

```bash
./scripts/01_prepare_spacenet3.sh
./scripts/02_pretrain_jepa.sh
./scripts/03_train_road_decoder.sh
./scripts/04_vectorize_roads.sh
./scripts/05_export_geosparql.sh
```

Once each stage works on its own, run the full workflow:

```bash
./scripts/06_run_pipeline.sh
```

## Design principle

The intended architecture is:

> shared representation learning first, task-specific decoding second, deterministic geospatial serialization last.

That is the main project idea.
