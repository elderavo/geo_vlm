#!/usr/bin/env bash

# Stage 3: supervised road decoding.
#
# Goal:
# - Reuse the pretrained JEPA encoder and train a supervised decoder head on
#   official SpaceNet road labels.
#
# Why a decoder instead of "GeoSPARQL latents":
# - GeoSPARQL is a symbolic geospatial standard, not a neural target space.
# - For roads, the useful supervised target is geometry-bearing output such as
#   a road mask or centerline mask, which can later be vectorized cleanly.
#
# Recommended first experiment:
# - Freeze the JEPA encoder.
# - Train a lightweight dense decoder.
# - Measure whether JEPA features improve label efficiency versus a baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

: "${JEPA_CHECKPOINT:="${JEPA_RUN_DIR}/best.ckpt"}"

log "Training supervised road decoder"
uv run python -m geo_vlm.training.train_road_decoder \
  --data-root "${DATA_ROOT}/processed/spacenet3" \
  --encoder-checkpoint "${JEPA_CHECKPOINT}" \
  --config "${CONFIG_ROOT}/road_decoder.yaml" \
  --output-dir "${DECODER_RUN_DIR}"

log "Road decoder training finished"
