#!/usr/bin/env bash

# Stage 5: export vector roads as GeoSPARQL-compatible RDF.
#
# Goal:
# - Serialize predicted road features into a symbolic representation that other
#   geospatial systems can query.
#
# Architectural role:
# - This stage should not relearn geometry.
# - It should deterministically map domain objects into RDF triples such as:
#   road segment -> geometry resource -> WKT literal.
#
# Design note:
# - Keeping RDF generation deterministic follows the "functional core,
#   imperative shell" pattern: learned perception happens earlier; knowledge
#   serialization is transparent, testable, and debuggable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

log "Exporting GeoSPARQL RDF"
uv run python -m geo_vlm.export.export_geosparql \
  --input-dir "${VECTOR_RUN_DIR}" \
  --output-dir "${RDF_RUN_DIR}"

log "GeoSPARQL export finished"
