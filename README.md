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
- deterministic GeoSPARQL exporter,
- staged shell workflow.

Still to implement:

- JEPA model and training loop,
- supervised road decoder training loop,
- raster-to-vector road post-processing,
- evaluation and experiment tracking.

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
