#!/usr/bin/env bash

# Shared configuration for the geospatial ML pipeline.
#
# Why this file exists:
# - Every stage needs the same dataset paths and output roots.
# - Keeping those values in one place prevents silent drift between scripts.
# - This follows the "single source of truth" principle used in most build
#   systems and keeps later automation simpler.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${DATA_ROOT:="${PROJECT_ROOT}/data"}"
: "${SPACENET3_ROOT:="${DATA_ROOT}/spacenet3"}"
: "${ARTIFACT_ROOT:="${HOME}/ijepa_logs"}"
: "${CONFIG_ROOT:="${PROJECT_ROOT}/configs"}"

: "${JEPA_RUN_DIR:="${ARTIFACT_ROOT}"}"
: "${DECODER_RUN_DIR:="${ARTIFACT_ROOT}/road_decoder"}"
: "${VECTOR_RUN_DIR:="${ARTIFACT_ROOT}/vectors"}"
: "${RDF_RUN_DIR:="${ARTIFACT_ROOT}/rdf"}"

mkdir -p \
  "${DATA_ROOT}" \
  "${ARTIFACT_ROOT}" \
  "${JEPA_RUN_DIR}" \
  "${DECODER_RUN_DIR}" \
  "${VECTOR_RUN_DIR}" \
  "${RDF_RUN_DIR}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}
