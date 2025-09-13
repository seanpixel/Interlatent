#!/usr/bin/env bash
set -euo pipefail

# Determine location of environment.yaml relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../environment.yaml"

if [ ! -f "$ENV_FILE" ]; then
  echo "environment.yaml not found at $ENV_FILE" >&2
  exit 1
fi

# Default environment name from file or first argument
DEFAULT_NAME=$(grep '^name:' "$ENV_FILE" | awk '{print $2}')
ENV_NAME=${1:-$DEFAULT_NAME}

# Prefer mamba if available for faster installs
if command -v mamba >/dev/null 2>&1; then
  CONDA_CMD="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
else
  echo "Neither conda nor mamba is installed. Please install one first." >&2
  exit 1
fi

if $CONDA_CMD env list | grep -q "^$ENV_NAME\s"; then
  echo "Updating existing environment '$ENV_NAME' using $CONDA_CMD"
  $CONDA_CMD env update -f "$ENV_FILE" -n "$ENV_NAME"
else
  echo "Creating environment '$ENV_NAME' using $CONDA_CMD"
  $CONDA_CMD env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

echo "Environment ready. Activate with: conda activate $ENV_NAME"

