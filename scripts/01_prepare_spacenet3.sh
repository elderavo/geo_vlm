#!/usr/bin/env bash

# Stage 1: prepare the official supervised road dataset.
#
# Goal:
# - Use the official SpaceNet 3 Khartoum training labels as the clean baseline.
# - Build train/validation metadata from the canonical labels before bringing
#   in noisier external sources such as OpenStreetMap.
#
# Architectural role:
# - This stage owns "data truth" for supervised learning.
# - It should produce deterministic manifests that later stages consume.
# - TorchGeo should be used in the Python implementation so raster imagery and
#   vector labels stay geospatially aligned through a tested dataset wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

log "Preparing SpaceNet 3 Khartoum training data"
uv run python -m geo_vlm.data.prepare_spacenet3 \
  --root "${SPACENET3_ROOT}" \
  --output-dir "${DATA_ROOT}/processed/spacenet3"

log "SpaceNet 3 manifests are ready"
