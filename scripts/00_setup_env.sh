#!/usr/bin/env bash

# Stage 0: prepare the Python environment.
#
# Goal:
# - Use uv as the project Python manager.
# - Install the locked project dependencies before running any ML stage.
#
# Design note:
# - Environment setup is isolated from data prep and training on purpose.
#   Separating setup from execution is a standard reproducibility practice:
#   when a run fails later, you know whether the problem is "the environment"
#   or "the experiment."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

log "Syncing Python environment with uv"
uv sync

log "Environment is ready"
