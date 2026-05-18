#!/usr/bin/env bash

# Stage 4: convert model predictions into vector road geometry.
#
# Goal:
# - Turn dense road predictions into geometry-bearing objects such as
#   LineStrings and graph edges.
#
# Architectural role:
# - This is the bridge between neural perception and symbolic geospatial data.
# - Keep it deterministic when possible: thresholding, skeletonization,
#   snapping, and graph construction should be separately testable.
#
# Design pattern:
# - This is an adapter layer. It converts the decoder's raster-like output into
#   the domain model expected by graph and RDF exporters.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

: "${DECODER_CHECKPOINT:="${DECODER_RUN_DIR}/best.ckpt"}"

log "Vectorizing road predictions"
uv run python -m geo_vlm.postprocess.vectorize_roads \
  --data-root "${DATA_ROOT}/processed/spacenet3" \
  --decoder-checkpoint "${DECODER_CHECKPOINT}" \
  --output-dir "${VECTOR_RUN_DIR}"

log "Vector road outputs are ready"
