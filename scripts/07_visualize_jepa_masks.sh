#!/usr/bin/env bash

# Stage 7: render first-pass JEPA mask visualizations on a compute node.
#
# This stage submits a lightweight visualization job through Slurm. It should
# not run directly on the submit node because image/plotting dependencies and
# filesystem reads can be brittle or slow there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

log "Submitting JEPA mask visualization job"
uv run python -m geo_vlm.analysis.launch_mask_visuals_submitit \
  --config "${CONFIG_ROOT}/jepa.yaml" \
  "$@"
