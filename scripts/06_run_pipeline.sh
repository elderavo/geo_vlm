#!/usr/bin/env bash

# Convenience entrypoint: run the complete workflow in order.
#
# Workflow summary:
# 1. Prepare the uv-managed environment.
# 2. Build official SpaceNet 3 training manifests.
# 3. Pretrain a JEPA encoder on unlabeled imagery.
# 4. Train a supervised road decoder on official labels.
# 5. Vectorize road predictions.
# 6. Export the resulting geometry as GeoSPARQL RDF.
#
# Use this once each individual stage works on its own. During development,
# prefer running the stage scripts independently so failures are easier to
# localize.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/00_setup_env.sh"
"${SCRIPT_DIR}/01_prepare_spacenet3.sh"
"${SCRIPT_DIR}/02_pretrain_jepa.sh"
"${SCRIPT_DIR}/03_train_road_decoder.sh"
"${SCRIPT_DIR}/04_vectorize_roads.sh"
"${SCRIPT_DIR}/05_export_geosparql.sh"
