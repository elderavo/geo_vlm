#!/usr/bin/env bash

# Stage 2: self-supervised JEPA pretraining.
#
# Goal:
# - Train an encoder pair on unlabeled imagery so the model learns reusable
#   geospatial image structure before seeing road labels.
#
# Architectural role:
# - This is representation learning, not the road detector itself.
# - The encoder should learn useful latent structure; the downstream decoder
#   will later learn the supervised task contract.
#
# Best-practice note:
# - Save checkpoints and configs together so the exact encoder used by later
#   supervised experiments is auditable and reproducible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

log "Starting JEPA pretraining"
uv run python -m geo_vlm.training.pretrain_jepa \
  --data-root "${SPACENET3_ROOT}" \
  --config "${CONFIG_ROOT}/jepa.yaml" \
  --output-dir "${JEPA_RUN_DIR}"

log "JEPA pretraining finished"
