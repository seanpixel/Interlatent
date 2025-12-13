#!/usr/bin/env bash
# Convenience wrapper around run_transcoder.py
# Usage example:
#   ./scripts/run_transcoder.sh --env-id CartPole-v1 --algo PPO \
#       --model-path pretrained/ppo-CartPole-v1.zip --metric pole_angle:obs[2]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/run_transcoder.py" "$@"

